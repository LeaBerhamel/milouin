#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script principal : traitement des rafales de photos.

Usage: python scripts/02_process_rafales.py --photos data/photos --tableur data/observations.csv
"""

import argparse
import glob
import json
import os
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from utils.annotate import annotate_all
from utils.burst_detector import BurstDetector
from utils.console import print_header
from utils.journal import export_journal
from utils.crop_export import save_marque_crops, parse_nom_crop_correction
from utils.pipeline import CanardIdentificationPipeline
from utils.qualite import export_incoherences_csv
from utils.report import build_report
from utils.synthese import (
    build_synthese,
    export_graphiques_synthese,
    export_synthese_csv,
    print_synthese,
)

AMBIGUOUS_STATUSES = ("AMBIGU", "MULTIPLE_CANARDS", "PARTIELLEMENT_IDENTIFIE")


class AucunePhotoError(Exception):
    """Levée quand le dossier photos ne contient aucune image exploitable."""


class AucuneRafaleError(Exception):
    """Levée quand aucune rafale n'a pu être formée (pas de timestamp EXIF)."""


def _charger_canards_ajoutes_manuellement(output_dir):
    """Relit dictionnaire/detections_manuelles.json (écrit par
    module_corrections/fenetre_rafale.py quand un chercheur signale à la
    main un canard raté par l'IA -- voir ecrire_corrections.enregistrer_detection_manquee)
    et regroupe les entrées par photo source.

    SANS cette réinjection, un nouveau passage de run_pipeline() (nouvelles
    photos ajoutées au lot, ou simple ré-analyse) réécrit resultats_par_rafale.json
    depuis zéro à partir des seules détections YOLO du moment : tout canard
    ajouté à la main lors d'un passage précédent disparaît silencieusement
    de son rafale (même si son crop, sa bague et ses corrections espèce/
    sexe/couleur restent, eux, bien enregistrés sur disque dans
    crops_corrections/, corrections.csv et dictionnaire_identifications.json
    -- d'où l'illusion que "l'image ne s'enregistre pas vraiment").

    Retourne {photo_source: [(canard_index, bbox), ...]}, trié par
    canard_index. Tolérant : fichier absent ou illisible -> dict vide,
    n'importe jamais le run.
    """
    chemin = Path(output_dir) / "detections_manuelles.json"
    if not chemin.exists():
        return {}
    try:
        with open(chemin, encoding="utf-8") as f:
            detections = json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}

    par_photo = {}
    for crop_filename, infos in detections.items():
        photo_source = infos.get("photo_source")
        bbox = infos.get("bbox")
        if not photo_source or not bbox:
            continue
        parsed = parse_nom_crop_correction(crop_filename)
        # canard_index vient du nom de fichier (voir parse_nom_crop_correction) --
        # c'est lui qui a servi à nommer le crop et à indexer corrections.csv /
        # dictionnaire_identifications.json pour ce canard, il faut le
        # reproduire à l'identique pour que ces corrections restent liées.
        canard_index = parsed[1] if parsed else None
        if canard_index is None:
            continue
        par_photo.setdefault(photo_source, []).append((canard_index, bbox))

    for photo_source in par_photo:
        par_photo[photo_source].sort(key=lambda t: t[0])

    return par_photo


def _reinjecter_canards_manuels(result, canards_manuels_par_photo):
    """Réinjecte dans result["canards"] les ajouts manuels connus pour la
    photo de cette rafale (voir _charger_canards_ajoutes_manuellement),
    s'ils n'y sont pas déjà (un même run ne les ajoute qu'une fois).

    Ne touche à rien d'autre : espèce/sexe/couleur/bague de ces canards
    restent gérées comme avant, via corrections.csv / dictionnaire_identifications.json
    et affichées par utils/report.py au moment de la génération du rapport --
    ici on ne fait que garantir que le canard lui-même (et donc sa ligne
    dans le rapport, et le bouton pour le corriger à nouveau) ne disparaisse
    plus après un nouveau passage du pipeline.
    """
    photo_source = result.get("photo_analysee")
    entrees = canards_manuels_par_photo.get(photo_source)
    if not entrees:
        return

    canards = result.setdefault("canards", [])
    indices_existants = {c.get("canard_index") for c in canards}
    for canard_index, bbox in entrees:
        if canard_index in indices_existants:
            continue
        canards.append({
            "canard_index": canard_index,
            "bbox": list(bbox),
            "status": "AJOUT_MANUEL",
            "ajoute_manuellement": True,
        })
        indices_existants.add(canard_index)


def run_pipeline(photos_dir, tableur_path, output_dir, model_detection="models/detection/best.pt",
                  model_couleur=None, model_espece=None, model_sexe=None, burst_gap=5,
                  progress_callback=None, port_corrections=None):
    """
    Exécute le pipeline complet (détection -> classification -> matching -> rapport)
    et retourne un dict avec les chemins des fichiers produits.

    C'est la fonction réutilisée à la fois par la CLI (main() ci-dessous) et par
    l'interface graphique (scripts/interface_graphique.py) : toute la logique
    métier vit ici, les deux façades ne font que collecter les paramètres et
    afficher/retourner le résultat.

    progress_callback(etape: str, current: int, total: int) est appelé à chaque
    rafale traitée si fourni (utilisé par la barre de progression de l'interface
    graphique ; ignoré en CLI, qui utilise tqdm directement).

    port_corrections: Port local du serveur de corrections démarré par
    interface_graphique.py (voir module_corrections/serveur_corrections.py),
    s'il tourne. Transmis tel quel à build_report pour générer le lien
    "✏️ Corriger cette rafale" sur chaque ligne du rapport. None en usage
    CLI pur (pas de serveur, pas de fenêtre pour réagir au clic).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff")
    photos = []
    for ext in extensions:
        # glob sur Linux/macOS est sensible à la casse : on couvre aussi la
        # version majuscule de l'extension (ex. *.JPG) pour ne rien manquer.
        # "**" + recursive=True : scanne aussi les sous-dossiers (ex.
        # data/photos/job1/, data/photos/job2/...), pas seulement le premier
        # niveau -- pratique si les photos restent organisées par lot/job
        # plutôt que d'être toutes aplaties dans un seul dossier. Le chemin
        # complet (dossier inclus) est conservé pour chaque photo, donc deux
        # fichiers de même nom dans deux sous-dossiers différents (ex.
        # job1/RCNX0001.jpg et job2/RCNX0001.jpg, fréquent avec les pièges
        # photographiques qui recommencent leur numérotation) restent bien
        # deux entrées distinctes -- rien n'est écrasé.
        photos.extend(glob.glob(os.path.join(photos_dir, "**", ext), recursive=True))
        photos.extend(glob.glob(os.path.join(photos_dir, "**", ext.upper()), recursive=True))
    photos = sorted(set(photos))

    print_header("🦆 TRAITEMENT DES RAFALES DE PHOTOS")
    print(f"Photos trouvées : {len(photos)}")
    print(f"Tableur : {tableur_path}")
    print(f"Modèle détection : {model_detection}")

    if not photos:
        raise AucunePhotoError(
            f"Aucune photo trouvée dans '{photos_dir}'. Vérifie que le dossier existe et "
            "contient bien des images (.jpg, .jpeg, .png, .bmp, .tif/.tiff, quelle que soit la casse)."
        )

    pipeline = CanardIdentificationPipeline(
        tableur_path=tableur_path,
        model_detection_path=model_detection,
        model_couleur_path=model_couleur,
        model_espece_path=model_espece,
        model_sexe_path=model_sexe,
    )

    print_header("📸 GROUPEMENT EN RAFALES")
    burst_detector = BurstDetector()
    rafales = burst_detector.group_photos_into_bursts(photos, max_gap_seconds=burst_gap)

    print(f"\n✓ {len(rafales)} rafales détectées")
    if not rafales:
        raise AucuneRafaleError(
            "Aucune rafale n'a pu être formée (aucun timestamp EXIF exploitable). Vérifie que "
            "les photos ont bien leurs métadonnées EXIF d'origine (une photo recompressée ou "
            "capturée d'écran perd souvent l'EXIF)."
        )
    print(f"  • Moyenne : {len(photos) / len(rafales):.1f} photos/rafale")
    print(f"  • Min : {min(r['nb_photos'] for r in rafales)} photos")
    print(f"  • Max : {max(r['nb_photos'] for r in rafales)} photos")

    print_header("🔍 TRAITEMENT DES RAFALES")
    crops_dir = output_dir / "crops_marques_nasales"
    canards_manuels_par_photo = _charger_canards_ajoutes_manuellement(output_dir)
    nb_canards_reinjectes = 0
    resultats = []
    iterator = rafales if progress_callback else tqdm(rafales, desc="Progression")
    for i, burst in enumerate(iterator):
        result = pipeline.process_burst(burst)
        result["rafale_id"] = i
        avant = len(result.get("canards") or [])
        _reinjecter_canards_manuels(result, canards_manuels_par_photo)
        nb_canards_reinjectes += len(result.get("canards") or []) - avant
        # Sauvegarde en haute résolution de chaque marque nasale détectée sur
        # la photo analysée (crop indépendant de la classification couleur,
        # qui elle reste calculée en mémoire) : ajoute "crop_filename" /
        # "crop_path" à chaque entrée de result["marques"].
        save_marque_crops(result, crops_dir)
        resultats.append(result)
        if progress_callback:
            progress_callback("traitement", i + 1, len(rafales))
    if any(result.get("marques") for result in resultats):
        print(f"✓ Crops de marques nasales -> {crops_dir}/")
    if nb_canards_reinjectes:
        print(f"✓ {nb_canards_reinjectes} canard(s) ajouté(s) manuellement lors d'une correction "
              f"précédente réintégré(s) (dictionnaire/detections_manuelles.json)")

    print_header("📊 RÉSULTATS")
    stats = calculate_stats(resultats)
    print_stats(stats)

    print_header("💾 EXPORT DES RÉSULTATS")

    json_path = output_dir / "resultats_par_rafale.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(resultats, f, indent=2, default=str, ensure_ascii=False)
    print(f"✓ {json_path}")

    # Tableau récapitulatif (vue d'ensemble par rafale)
    recap_path = output_dir / "tableau_recapitulatif.csv"
    export_tableau_recapitulatif(resultats, recap_path)
    print(f"✓ {recap_path}")
    
    # Tableau détaillé (canard par canard)
    canards_path = output_dir / "tableau_canards.csv"
    export_tableau_canards(resultats, canards_path)
    print(f"✓ {canards_path}")

    cas_ambigus = [r for r in resultats if r.get("status") in AMBIGUOUS_STATUSES]
    ambigus_path = None
    if cas_ambigus:
        ambigus_path = output_dir / "cas_ambigus.json"
        with open(ambigus_path, "w", encoding="utf-8") as f:
            json.dump(cas_ambigus, f, indent=2, default=str, ensure_ascii=False)
        print(f"✓ {ambigus_path} ({len(cas_ambigus)} rafales à vérifier)")

    print_header("🖼️  PHOTOS ANNOTÉES")
    photos_dir_out = output_dir / "photos_annotees"
    chemins_annotes = annotate_all(resultats, photos_dir_out)
    print(f"✓ {len(chemins_annotes)} photo(s) annotée(s) -> {photos_dir_out}/")
    print("   (triées dans a_verifier/ | identifies/ | non_concluant/)")

    print_header("📈 SYNTHÈSE DU LOT")
    synthese = build_synthese(resultats)
    print_synthese(synthese)
    synthese_path = output_dir / "synthese_lot.csv"
    export_synthese_csv(synthese, synthese_path)
    print(f"\n✓ {synthese_path}")

    graphique_path = export_graphiques_synthese(synthese, output_dir / "graphiques_synthese.png")
    if graphique_path:
        print(f"✓ {graphique_path}")

    nb_incoherences = export_incoherences_csv(resultats, output_dir / "incoherences.csv")
    incoherences_path = output_dir / "incoherences.csv"
    if nb_incoherences:
        print(f"✓ {incoherences_path} ({nb_incoherences} incohérence(s) tableur/IA)")
    else:
        print("✓ Aucune incohérence tableur/IA détectée")

    journal_path = export_journal(resultats, output_dir / "journal_cas_limites.txt")
    print(f"✓ {journal_path}")

    rapport_path = build_report(resultats, synthese, chemins_annotes, output_dir, graphique_path, port_corrections)
    print(f"✓ {rapport_path}  ← ouvrir ce fichier dans un navigateur")

    print_header("✅ TRAITEMENT TERMINÉ")

    return {
        "resultats": resultats,
        "stats": stats,
        "synthese": synthese,
        "rapport_path": str(rapport_path),
        "json_path": str(json_path),
        "recap_path": str(recap_path),
        "canards_path": str(canards_path),
        "ambigus_path": str(ambigus_path) if ambigus_path else None,
        "synthese_path": str(synthese_path),
        "graphique_path": str(graphique_path) if graphique_path else None,
        "incoherences_path": str(incoherences_path) if nb_incoherences else None,
        "journal_path": str(journal_path),
        # Chemins des photos annotées par rafale_id -- nécessaire à
        # interface_graphique.py pour retrouver quelle photo afficher dans
        # la fenêtre "photo + liste" ouverte depuis un clic "Corriger" du
        # rapport (voir _traiter_queue / cas "correction"). Jusqu'ici
        # calculé en interne (chemins_annotes ci-dessus) mais jamais
        # renvoyé -- sans ça l'interface n'a aucun moyen de le retrouver.
        "chemins_annotes": chemins_annotes,
    }


def main():
    parser = argparse.ArgumentParser(description="Traitement des rafales de photos")
    parser.add_argument("--photos", required=True, help="Dossier contenant les photos")
    parser.add_argument("--tableur", required=True, help="Fichier CSV des observations")
    parser.add_argument("--model-detection", default="models/detection/best.pt")
    parser.add_argument("--model-couleur", default=None)
    parser.add_argument("--model-espece", default=None)
    parser.add_argument("--model-sexe", default=None)
    parser.add_argument("--output", default="output", help="Dossier de sortie")
    parser.add_argument("--burst-gap", type=int, default=5, help="Écart max entre photos d'une rafale (secondes)")
    parser.add_argument("--skip-ia-couleur", action="store_true", help="Ne pas charger IA couleur")
    parser.add_argument("--skip-ia-espece", action="store_true", help="Ne pas charger IA espèce")
    parser.add_argument("--skip-ia-sexe", action="store_true", help="Ne pas charger IA sexe")
    args = parser.parse_args()

    try:
        run_pipeline(
            photos_dir=args.photos,
            tableur_path=args.tableur,
            output_dir=args.output,
            model_detection=args.model_detection,
            model_couleur=args.model_couleur if not args.skip_ia_couleur else None,
            model_espece=args.model_espece if not args.skip_ia_espece else None,
            model_sexe=args.model_sexe if not args.skip_ia_sexe else None,
            burst_gap=args.burst_gap,
        )
    except (AucunePhotoError, AucuneRafaleError) as e:
        print(f"\n❌ {e}")


def calculate_stats(resultats):
    """Calcule les statistiques agrégées sur les résultats du traitement."""
    stats = {
        "total_rafales": len(resultats),
        "identifies": sum(1 for r in resultats if r["status"] == "IDENTIFIE"),
        "identifies_multiple": sum(1 for r in resultats if r["status"] == "IDENTIFIE_MULTIPLE"),
        "partiellement_identifies": sum(1 for r in resultats if r["status"] == "PARTIELLEMENT_IDENTIFIE"),
        "sans_canard": sum(1 for r in resultats if r["status"] == "AUCUN_CANARD"),
        "ambigus": sum(1 for r in resultats if r["status"] == "AMBIGU"),
        "non_trouves": sum(1 for r in resultats if r["status"] == "NON_TROUVE_TABLEUR"),
        "multiple_canards": sum(1 for r in resultats if r["status"] == "MULTIPLE_CANARDS"),
        "non_marques": sum(1 for r in resultats if r["status"] == "NON_MARQUE"),
    }

    methodes = {}
    for r in resultats:
        if "methode" in r:
            methodes[r["methode"]] = methodes.get(r["methode"], 0) + 1
    stats["methodes"] = methodes

    ia_counts = {"detection": 0, "couleur": 0, "espece": 0, "sexe": 0}
    for r in resultats:
        for ia in r.get("ia_utilisees", []):
            if ia in ia_counts:
                ia_counts[ia] += 1
    stats["ia_utilisees"] = ia_counts

    stats["canards_sans_marque"] = sum(
        1 for r in resultats for c in r.get("canards", [])
        if c.get("status") == "IDENTIFIE" and c.get("marque_absente")
    )

    return stats


def print_stats(stats):
    """Affiche les statistiques calculées par calculate_stats."""
    total = stats["total_rafales"]

    def pct(cle):
        return stats[cle] / total * 100

    print(f"""
Total rafales : {total}

✅ Identifiés : {stats['identifies']} ({pct('identifies'):.1f}%)
✅ Identifiés (plusieurs canards) : {stats['identifies_multiple']} ({pct('identifies_multiple'):.1f}%)
🟡 Partiellement identifiés : {stats['partiellement_identifies']} ({pct('partiellement_identifies'):.1f}%)
❌ Sans canard : {stats['sans_canard']} ({pct('sans_canard'):.1f}%)
⚠️  Ambigus : {stats['ambigus']} ({pct('ambigus'):.1f}%)
❓ Non trouvés : {stats['non_trouves']} ({pct('non_trouves'):.1f}%)
🦆 Plusieurs canards non résolus : {stats['multiple_canards']} ({pct('multiple_canards'):.1f}%)
🏷️ Canards non marqués classifiés : {stats['non_marques']} ({pct('non_marques'):.1f}%)

Méthodes d'identification :
""")
    for methode, count in stats["methodes"].items():
        print(f"  • {methode}: {count}")

    print("\nUtilisation des IAs :")
    for ia, count in stats["ia_utilisees"].items():
        print(f"  • {ia.upper()}: {count} rafales")

    if stats["canards_sans_marque"]:
        print(f"\n⚠️  Canards identifiés SANS marque nasale détectée : {stats['canards_sans_marque']} "
              "(espèce/sexe/heure seuls — voir journal_cas_limites.txt)")


def export_tableau_recapitulatif(resultats, output_path):
    """Exporte le tableau récapitulatif par rafale (vue d'ensemble)."""
    rows = []
    
    for result in resultats:
        rafale_id = result["rafale_id"]
        rafale_info = result["rafale"]
        
        # Horodatage médian de la rafale
        median_time = rafale_info["median_time"]
        if hasattr(median_time, 'year'):
            annee = median_time.year
            date = median_time.strftime("%d/%m/%Y")
            heure = median_time.strftime("%H:%M:%S")
        else:
            # Si c'est une string, on la parse
            from datetime import datetime
            try:
                dt = datetime.fromisoformat(str(median_time))
                annee = dt.year
                date = dt.strftime("%d/%m/%Y")
                heure = dt.strftime("%H:%M:%S")
            except (ValueError, TypeError):
                annee = ""
                date = ""
                heure = ""
        
        # Comptage des canards
        canards = result.get("canards", [])
        nb_canards_total = len(canards)
        nb_canards_identifies = sum(1 for c in canards if c.get("status") == "IDENTIFIE")
        nb_canards_non_marques = sum(1 for c in canards if c.get("status") == "NON_MARQUE")
        
        # Bagues trouvées
        bagues = [c.get("bague") for c in canards if c.get("bague")]
        bagues_str = ", ".join(bagues) if bagues else ""
        
        # Marques détectées
        nb_marques = len(result.get("marques", []))
        
        # Confiance minimale (pour les canards identifiés)
        confiances = [c.get("confiance_globale") for c in canards 
                     if c.get("status") == "IDENTIFIE" and c.get("confiance_globale") is not None]
        if confiances:
            confiance_min = f"{min(confiances) * 100:.0f}%"
        else:
            confiance_min = ""
        
        # À vérifier ?
        statut = result.get("status", "")
        a_verifier = "OUI" if statut in ("AMBIGU", "MULTIPLE_CANARDS", "PARTIELLEMENT_IDENTIFIE") else "NON"
        
        # Photo analysée (juste le nom, pas le chemin complet)
        photo_analysee = Path(result.get("photo_analysee", "")).name
        
        rows.append({
            "rafale_id": rafale_id,
            "annee": annee,
            "date": date,
            "heure": heure,
            "nb_photos": rafale_info["nb_photos"],
            "statut": statut,
            "nb_canards_total": nb_canards_total,
            "nb_canards_identifies": nb_canards_identifies,
            "nb_canards_non_marques": nb_canards_non_marques,
            "bagues": bagues_str,
            "marques_detectees": nb_marques,
            "confiance_min": confiance_min,
            "a_verifier": a_verifier,
            "photo_analysee": photo_analysee,
        })
    
    pd.DataFrame(rows).to_csv(output_path, index=False, encoding="utf-8")


def export_tableau_canards(resultats, output_path):
    """Exporte le tableau détaillé par canard (une ligne par canard × photo)."""
    rows = []

    for result in resultats:
        rafale_id = result["rafale_id"]
        
        for photo_path in result["rafale"]["photos"]:
            # Extraction de l'horodatage de cette photo spécifique
            photo_name = Path(photo_path).name
            
            # Trouve le timestamp de cette photo dans la rafale
            photo_idx = result["rafale"]["photos"].index(photo_path)
            if photo_idx < len(result["rafale"]["timestamps"]):
                photo_dt = result["rafale"]["timestamps"][photo_idx]
                annee = photo_dt.year
                date = photo_dt.strftime("%d/%m/%Y")
                heure = photo_dt.strftime("%H:%M:%S")
            else:
                # Fallback si timestamp manquant
                annee = ""
                date = ""
                heure = ""
            
            # Cas : plusieurs canards sur la photo
            if "canards" in result and len(result.get("canards", [])) > 0:
                for canard in result["canards"]:
                    canard_num = canard.get("canard_index", 0) + 1  # Numérotation humaine (1, 2, 3...)
                    statut = canard.get("status", "")
                    
                    # Espèce : priorité au tableur, sinon détectée
                    espece = canard.get("espece") or canard.get("espece_detectee") or ""
                    
                    # Sexe : priorité au tableur, sinon détecté
                    sexe = canard.get("sexe") or canard.get("sexe_detectee") or ""
                    
                    # Couleur : priorité au tableur, sinon détectée
                    couleur = canard.get("couleur_marque") or canard.get("couleur_detectee") or ""
                    
                    # Marque présente ?
                    marque_absente = canard.get("marque_absente", False)
                    marque_presente = "NON" if marque_absente else "OUI"
                    
                    rows.append({
                        "rafale_id": rafale_id,
                        "annee": annee,
                        "date": date,
                        "heure": heure,
                        "canard_num": canard_num,
                        "statut_canard": statut,
                        "bague": canard.get("bague", ""),
                        "espece": espece,
                        "sexe": sexe,
                        "couleur_marque": couleur,
                        "marque_presente": marque_presente,
                        "photo": photo_name,
                    })
            
            # Cas : aucun canard détecté (rafale vide)
            else:
                rows.append({
                    "rafale_id": rafale_id,
                    "annee": annee,
                    "date": date,
                    "heure": heure,
                    "canard_num": "",
                    "statut_canard": result.get("status", ""),
                    "bague": "",
                    "espece": "",
                    "sexe": "",
                    "couleur_marque": "",
                    "marque_presente": "",
                    "photo": photo_name,
                })

    pd.DataFrame(rows).to_csv(output_path, index=False, encoding="utf-8")


if __name__ == "__main__":
    main()