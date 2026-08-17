# -*- coding: utf-8 -*-
"""
Génère un rapport HTML unique (output/rapport.html) consultable dans un
navigateur : vignettes des photos annotées, chiffres clés, tableau
filtrable par statut et triable par colonne — pour éviter d'avoir à
jongler entre le CSV, le JSON et le dossier d'images.
"""

import csv
import html
import json
import os
from pathlib import Path

from utils.crop_export import nom_crop_correction

CATEGORIE_PAR_STATUT = {
    "IDENTIFIE": "identifies",
    "IDENTIFIE_MULTIPLE": "identifies",
    "AMBIGU": "a_verifier",
    "MULTIPLE_CANARDS": "a_verifier",
    "PARTIELLEMENT_IDENTIFIE": "a_verifier",
    "AUCUN_CANARD": "non_concluant",
    "NON_TROUVE_TABLEUR": "non_concluant",
    "NON_MARQUE": "non_concluant",
}

# Catégorie à part, prioritaire sur CATEGORIE_PAR_STATUT : une rafale dont
# au moins un canard a été corrigé à la main (voir _etat_correction_rafale)
# bascule dans "image_corrigee" plutôt que dans sa catégorie de statut
# habituelle, pour pouvoir la retrouver d'un coup de filtre quel que soit
# son statut d'origine (IDENTIFIE, NON_MARQUE...).
CATEGORIE_IMAGE_CORRIGEE = "image_corrigee"

CSS = """
body { font-family: -apple-system, Segoe UI, Arial, sans-serif; margin: 24px; background: #f7f7f8; color: #222; }
h1 { margin-bottom: 4px; }
.sous-titre { color: #666; margin-top: 0; }
.chiffres { display: flex; gap: 16px; flex-wrap: wrap; margin: 20px 0; }
.chiffre-card { background: white; border-radius: 8px; padding: 12px 18px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); min-width: 140px; }
.chiffre-card .valeur { font-size: 26px; font-weight: 700; }
.chiffre-card .label { color: #666; font-size: 13px; }
.graphique { max-width: 100%; margin: 20px 0; background: white; padding: 12px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
.filtres { margin: 16px 0; }
.filtres button { padding: 8px 14px; margin-right: 8px; border: 1px solid #ccc; background: white; border-radius: 6px; cursor: pointer; font-size: 14px; }
.filtres button.actif { background: #333; color: white; border-color: #333; }
table { border-collapse: collapse; width: 100%; background: white; border-radius: 8px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
th, td { padding: 8px 10px; text-align: left; border-bottom: 1px solid #eee; font-size: 13px; vertical-align: middle; }
th { background: #fafafa; cursor: pointer; user-select: none; position: sticky; top: 0; }
th:hover { background: #f0f0f0; }
tr:hover { background: #fafafa; }
img.vignette { max-width: 140px; max-height: 100px; border-radius: 4px; display: block; }
.badge { padding: 2px 8px; border-radius: 10px; font-size: 12px; color: white; display: inline-block; }
.badge-identifies { background: #2e9e46; }
.badge-a_verifier { background: #e07b00; }
.badge-non_concluant { background: #888; }
.badge-image_corrigee { background: #2e6ba8; }
.confiance-haute { color: #2e9e46; font-weight: 600; }
.confiance-moyenne { color: #e07b00; font-weight: 600; }
.confiance-faible { color: #d13c3c; font-weight: 600; }
.badge-corrige { padding: 2px 8px; border-radius: 10px; font-size: 12px; color: white; display: inline-block; background: #2e6ba8; cursor: help; }
.corrige-vide { color: #ccc; }
.info-icone { cursor: pointer; color: #888; font-size: 12px; margin-left: 4px; text-decoration: none; border: 1px solid #ccc; border-radius: 50%; width: 15px; height: 15px; display: inline-flex; align-items: center; justify-content: center; vertical-align: middle; }
.info-icone:hover { background: #eee; }
.entete-actions { display: flex; justify-content: space-between; align-items: flex-start; gap: 16px; flex-wrap: wrap; }
.btn-archiver { align-self: flex-start; padding: 8px 14px; border: 1px solid #c0392b; background: white; color: #c0392b; border-radius: 6px; cursor: pointer; font-size: 14px; text-decoration: none; display: inline-block; white-space: nowrap; }
.btn-archiver:hover { background: #c0392b; color: white; }
.btn-verifier-serie { padding: 8px 14px; margin-left: 4px; margin-right: 16px; border: 1px solid #1d6fa5; background: #eaf4fb; color: #1d6fa5; border-radius: 6px; cursor: pointer; font-size: 14px; text-decoration: none; display: inline-block; }
.btn-verifier-serie:hover { background: #1d6fa5; color: white; }
.barre-contextuelle { margin: 12px 0; padding: 10px 14px; background: #eaf4fb; border: 1px solid #cfe4f3; border-radius: 8px; display: flex; align-items: center; }
.barre-contextuelle .btn-verifier-serie { margin: 0; background: white; }
.modale-fond { display: none; position: fixed; inset: 0; background: rgba(0,0,0,0.4); align-items: center; justify-content: center; z-index: 100; }
.modale-fond.visible { display: flex; }
.modale { background: white; border-radius: 8px; padding: 24px 28px; max-width: 560px; max-height: 80vh; overflow-y: auto; box-shadow: 0 4px 20px rgba(0,0,0,0.25); }
.modale h2 { margin-top: 0; font-size: 18px; }
.modale table { box-shadow: none; margin: 12px 0; }
.modale table th, .modale table td { font-size: 13px; }
.modale .fermer { float: right; cursor: pointer; font-size: 20px; color: #888; line-height: 1; background: none; border: none; }
.modale .fermer:hover { color: #222; }
.modale code { background: #f0f0f0; padding: 1px 5px; border-radius: 3px; }
"""

JS = """
var etatMarque = false;

function filtrer(cat) {
  document.querySelectorAll('.filtres button[data-cat]').forEach(function(b) {
    b.classList.toggle('actif', b.dataset.cat === cat);
  });
  // Barre contextuelle "🔎 Vérifier les rafales à vérifier, une par une" --
  // n'a de sens que quand ce filtre précis est actif (voir CSS
  // .barre-contextuelle) : si elle existe (absente quand aucune rafale
  // n'est "à vérifier", voir report.py::bouton_verifier_serie), elle
  // apparaît/disparaît avec le filtre plutôt que de rester affichée en
  // permanence au-dessus du tableau.
  var barreVerifierSerie = document.getElementById('barre-verifier-serie');
  if (barreVerifierSerie) {
    barreVerifierSerie.style.display = (cat === 'a_verifier') ? '' : 'none';
  }
  appliquerFiltres(cat);
}

function toggleMarque() {
  etatMarque = !etatMarque;
  document.getElementById('btn-marque').classList.toggle('actif', etatMarque);
  var boutonCategorieActif = document.querySelector('.filtres button[data-cat].actif');
  appliquerFiltres(boutonCategorieActif ? boutonCategorieActif.dataset.cat : 'toutes');
}

function appliquerFiltres(cat) {
  document.querySelectorAll('tbody tr').forEach(function(tr) {
    var okCategorie = (cat === 'toutes' || tr.dataset.categorie === cat);
    var okMarque = (!etatMarque || tr.dataset.marque === '1');
    tr.style.display = (okCategorie && okMarque) ? '' : 'none';
  });
}

function trier(colIndex, numerique) {
  var tbody = document.querySelector('tbody');
  var lignes = Array.from(tbody.querySelectorAll('tr'));
  var asc = tbody.dataset.triCol == colIndex ? tbody.dataset.triAsc !== '1' : true;
  lignes.sort(function(a, b) {
    var va = a.children[colIndex].dataset.tri || a.children[colIndex].innerText;
    var vb = b.children[colIndex].dataset.tri || b.children[colIndex].innerText;
    if (numerique) { va = parseFloat(va) || -Infinity; vb = parseFloat(vb) || -Infinity; return asc ? va - vb : vb - va; }
    return asc ? va.localeCompare(vb) : vb.localeCompare(va);
  });
  lignes.forEach(function(tr) { tbody.appendChild(tr); });
  tbody.dataset.triCol = colIndex;
  tbody.dataset.triAsc = asc ? '1' : '0';
}

function ouvrirModale(id) {
  document.getElementById(id).classList.add('visible');
}

function fermerModale(id) {
  document.getElementById(id).classList.remove('visible');
}

function confirmerArchivage(lien) {
  if (confirm("Archiver le contenu actuel de output/ et repartir d'un dossier vide ?\\n\\n" +
              "Une confirmation supplémentaire s'ouvrira dans l'application Suivi Canards -- " +
              "rien n'est déplacé tant que tu ne l'as pas validée là-bas.")) {
    window.location.href = lien;
  }
}
"""


SYMBOLE_PAR_NIVEAU_CONFIANCE = {
    "confiance-haute": "●",
    "confiance-moyenne": "▲",
    "confiance-faible": "■",
}


def _niveau_css(confiance_globale):
    if confiance_globale is None:
        return ""
    if confiance_globale >= 0.90:
        return "confiance-haute"
    if confiance_globale >= 0.70:
        return "confiance-moyenne"
    return "confiance-faible"


def _resume_canards(result):
    """Texte résumant les canards d'une rafale : bagues, espèces, sexes."""
    canards = result.get("canards", [])
    if not canards:
        return "-", "-", "-", None
    bagues = [str(c.get("bague")) for c in canards if c.get("bague")]
    especes = [str(c.get("espece") or c.get("espece_detectee")) for c in canards
               if c.get("espece") or c.get("espece_detectee")]
    sexes = [str(c.get("sexe") or c.get("sexe_detectee")) for c in canards
             if c.get("sexe") or c.get("sexe_detectee")]
    confiances = [c.get("confiance_globale") for c in canards if c.get("confiance_globale") is not None]
    confiance_min = min(confiances) if confiances else None
    return (", ".join(bagues) or "-", ", ".join(sorted(set(especes))) or "-",
            ", ".join(sorted(set(sexes))) or "-", confiance_min)


def _charger_etat_corrections(dossier_dictionnaire):
    """Lit les 4 fichiers écrits par module_corrections/ecrire_corrections.py
    (corrections.csv, dictionnaire_identifications.json, crops_exclus.json,
    detections_manuelles.json -- tous à la racine de dossier_dictionnaire,
    qui est output_dir : voir interface_graphique.py, self.dictionnaire_dir_courant)
    et retourne {crop_filename: date_iso_str_ou_None} pour chaque crop qui
    porte au moins une trace d'une correction manuelle.

    Volontairement tolérant : un fichier absent (aucune correction faite
    depuis ce lot, ou rapport généré en usage CLI pur sans module_corrections)
    ne fait pas planter la génération du rapport, juste aucun badge "Corrigé".
    N'est appelée qu'une seule fois par génération de rapport (pas par
    ligne), les 4 fichiers pouvant grossir au fil des corrections.

    Les 4 fichiers stockent en réalité le CHEMIN COMPLET du crop (voir
    module_corrections/ecrire_corrections.py, valeurs["crop"] vient de
    save_canard_crop() qui renvoie str(dest_dir / crop_filename)) -- alors
    que _etat_correction_rafale ci-dessous recalcule seulement le NOM DE
    FICHIER via nom_crop_correction (qui ignore volontairement le dossier,
    justement pour ne pas dépendre d'un chemin absolu). Sans normalisation,
    les deux ne matchent jamais et le badge "Corrigé" n'apparaît quasiment
    jamais. On indexe donc systématiquement par Path(...).name, jamais par
    la valeur brute lue dans ces fichiers.
    """
    dossier_dictionnaire = Path(dossier_dictionnaire)
    etat = {}

    def _nom(crop):
        return Path(crop).name if crop else None

    corrections_csv = dossier_dictionnaire / "corrections.csv"
    if corrections_csv.exists():
        with open(corrections_csv, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                crop = _nom(row.get("crop_path"))
                if crop:
                    etat[crop] = row.get("date") or etat.get(crop)

    chemin_dico = dossier_dictionnaire / "dictionnaire_identifications.json"
    if chemin_dico.exists():
        try:
            with open(chemin_dico, encoding="utf-8") as f:
                dictionnaire = json.load(f)
            for crop, infos in dictionnaire.items():
                # Ne compte que les identifications réellement issues d'une
                # correction manuelle (voir ecrire_corrections.enregistrer_bague,
                # source="correction_ponctuelle_rapport") -- un crop identifié
                # directement par le matching temporel du pipeline lui-même
                # n'y figure jamais (ce dictionnaire est écrit uniquement par
                # la fenêtre "photo + liste", pas par 02_process_rafales.py).
                crop = _nom(crop)
                if crop:
                    etat[crop] = infos.get("corrige_le") or etat.get(crop)
        except (json.JSONDecodeError, OSError):
            pass

    chemin_exclusions = dossier_dictionnaire / "crops_exclus.json"
    if chemin_exclusions.exists():
        try:
            with open(chemin_exclusions, encoding="utf-8") as f:
                exclusions = json.load(f)
            for e in exclusions:
                crop = _nom(e.get("crop_filename"))
                if crop:
                    etat[crop] = e.get("date_exclusion") or etat.get(crop)
        except (json.JSONDecodeError, OSError):
            pass

    chemin_detections = dossier_dictionnaire / "detections_manuelles.json"
    if chemin_detections.exists():
        try:
            with open(chemin_detections, encoding="utf-8") as f:
                detections = json.load(f)
            for crop, infos in detections.items():
                crop = _nom(crop)
                if crop:
                    etat[crop] = infos.get("ajoute_le") or etat.get(crop)
        except (json.JSONDecodeError, OSError):
            pass

    return etat


def _etat_correction_rafale(result, etat_corrections):
    """Pour une rafale : (corrige, date_plus_recente) où corrige est True
    si AU MOINS UN de ses canards porte une correction manuelle (bague,
    espèce/sexe/couleur, exclusion, ou ajout manuel de canard/marque).

    Recalcule le nom de crop de correction attendu pour chaque canard avec
    exactement la même formule que save_canard_crop() (voir
    utils.crop_export.nom_crop_correction) -- pas besoin que le fichier
    crop existe réellement sur disque, seul le NOM compte pour retrouver
    l'entrée dans etat_corrections."""
    photo_source = result.get("photo_analysee")
    if not photo_source:
        return False, None

    rafale_id = result.get("rafale_id")
    dates = []
    for canard in result.get("canards", []):
        idx = canard.get("canard_index", 0)
        crop_filename = nom_crop_correction(photo_source, rafale_id, idx)
        date = etat_corrections.get(crop_filename)
        if crop_filename in etat_corrections:
            dates.append(date)

    if not dates:
        return False, None
    dates_connues = sorted(d for d in dates if d)
    return True, (dates_connues[-1] if dates_connues else None)


def _categorie_rafale(result, etat_corrections, corrige=None):
    """Catégorie de filtrage HTML (identifies/a_verifier/non_concluant/
    image_corrigee) d'une rafale -- même règle de priorité que
    _ligne_html : une correction manuelle prime sur le statut d'origine.
    `corrige` peut être passé si déjà connu (évite de rappeler
    _etat_correction_rafale une seconde fois pour la même rafale)."""
    if corrige is None:
        corrige, _ = _etat_correction_rafale(result, etat_corrections)
    if corrige:
        return CATEGORIE_IMAGE_CORRIGEE
    return CATEGORIE_PAR_STATUT.get(result.get("status", "?"), "non_concluant")


def _ligne_html(result, chemins_annotes, output_dir, port_corrections=None, etat_corrections=None):
    rid = result.get("rafale_id")
    status = result.get("status", "?")
    corrige, date_correction = _etat_correction_rafale(result, etat_corrections or {})
    categorie = _categorie_rafale(result, etat_corrections or {}, corrige=corrige)
    bagues, especes, sexes, confiance_min = _resume_canards(result)
    avec_marque = "1" if result.get("marques") else "0"

    chemin_img = chemins_annotes.get(rid)
    if chemin_img:
        rel = os.path.relpath(chemin_img, output_dir)
        alt = f"Rafale {rid}, statut {status}" + (f", bague(s) {bagues}" if bagues != "-" else "")
        vignette = (f'<a href="{html.escape(rel)}" target="_blank">'
                    f'<img class="vignette" src="{html.escape(rel)}" loading="lazy" '
                    f'alt="{html.escape(alt)}"></a>')
    else:
        vignette = "-"

    conf_txt = f"{confiance_min * 100:.0f}%" if confiance_min is not None else "-"
    conf_css = _niveau_css(confiance_min)
    conf_symbole = SYMBOLE_PAR_NIVEAU_CONFIANCE.get(conf_css, "")
    conf_affichage = f"{conf_symbole} {conf_txt}".strip() if conf_symbole else conf_txt
    message = result.get("message", "")

    # Notes supplémentaires (photos de nuit uniquement - on retire l'alerte marque absente)
    notes = []
    if any(m.get("nuit_probable") for m in result.get("marques", [])):
        notes.append("🌙 couleur de marque peu fiable (photo de nuit probable)")

    # Combine le message principal avec les notes
    if notes:
        message = message + (" · " if message else "") + " · ".join(notes)

    # Lien "Corriger cette rafale" -- un par RAFALE, pas par canard : la
    # fenêtre "photo + liste" ouverte côté appli affiche tous les canards
    # de la photo avec leurs bbox déjà dessinées (annotate_all), le
    # chercheur choisit lequel corriger une fois la fenêtre ouverte. Ça
    # évite d'empiler un lien par canard (et le problème du crop
    # inexistant pour les non-bagués) : le serveur local retrouve tout
    # depuis les résultats déjà en mémoire côté appli, via rafale_id.
    # N'apparaît que si l'appli (interface_graphique.py) a démarré le
    # serveur de corrections -- absent en usage CLI pur (02_process_rafales.py
    # seul), où aucune fenêtre ne peut de toute façon réagir à ce lien.
    cellule_correction = ""
    if port_corrections:
        lien = f"http://127.0.0.1:{port_corrections}/corriger?rafale={rid}"
        cellule_correction = f'\n  <td><a href="{html.escape(lien)}">✏️ Corriger cette rafale</a></td>'

    # Colonne "Corrigé" : signale qu'une (ou plusieurs) correction manuelle
    # a été enregistrée pour cette rafale depuis la fenêtre "photo + liste"
    # -- voir _etat_correction_rafale. Absente de tri_corrige (data-tri)
    # quand rien n'a été corrigé, pour que le tri alphabétique regroupe
    # d'abord les rafales corrigées.
    if corrige:
        titre = f"Corrigée à la main le {html.escape(date_correction)}" if date_correction else "Corrigée à la main"
        cellule_corrige = f'<td data-tri="1"><span class="badge-corrige" title="{titre}">✏️ corrigé</span></td>'
    else:
        cellule_corrige = '<td data-tri="0" class="corrige-vide">—</td>'

    return f"""<tr data-categorie="{categorie}" data-marque="{avec_marque}">
  <td>{vignette}</td>
  <td data-tri="{rid}">{rid}</td>
  <td><span class="badge badge-{categorie}">{html.escape(status)}</span></td>
  <td>{html.escape(bagues)}</td>
  <td>{html.escape(especes)}</td>
  <td>{html.escape(sexes)}</td>
  <td class="{conf_css}" data-tri="{confiance_min or 0}" title="{'Confiance ' + conf_css.replace('confiance-', '') if conf_css else ''}">{conf_affichage}</td>
  {cellule_corrige}
  <td>{html.escape(message)}</td>{cellule_correction}
</tr>"""


def _chiffre_card(valeur, label):
    return f'<div class="chiffre-card"><div class="valeur">{valeur}</div><div class="label">{html.escape(label)}</div></div>'


def build_report(resultats, synthese, chemins_annotes, output_dir, graphique_path=None, port_corrections=None):
    """
    Construit le rapport HTML complet et l'écrit dans output_dir/rapport.html.

    Args:
        resultats: Liste des résultats de rafales (process_burst).
        synthese: Dictionnaire retourné par utils.synthese.build_synthese.
        chemins_annotes: Dict {rafale_id: chemin image annotée}, retourné
            par utils.annotate.annotate_all.
        output_dir: Dossier de sortie (où vit déjà photos_annotees/).
        graphique_path: Chemin du PNG de synthèse (optionnel), pour
            l'intégrer en haut du rapport.
        port_corrections: Port local (int) du serveur de corrections
            démarré par interface_graphique.py, s'il tourne. Si fourni,
            chaque ligne reçoit un lien "✏️ Corriger cette rafale" qui
            ouvre la fenêtre "photo + liste" côté appli, ET un bouton
            "🗄️ Archiver et repartir à zéro" apparaît en haut du rapport
            (même serveur, route /archiver -- voir serveur_corrections.py
            et InterfaceCanard._archiver_resultats_courants). Absent en
            usage CLI pur (pas de fenêtre pour réagir à ces clics).

    Returns:
        Chemin (str) du fichier HTML écrit.
    """
    output_dir = Path(output_dir)

    cartes = [
        _chiffre_card(synthese["nb_rafales"], "Rafales traitées"),
        _chiffre_card(synthese["nb_canards_detectes"], "Canards détectés"),
        _chiffre_card(synthese["nb_canards_identifies"], "Canards identifiés"),
        _chiffre_card(synthese["nb_individus_uniques"], "Individus uniques (bagues)"),
    ]

    graphique_html = ""
    if graphique_path:
        rel = os.path.relpath(graphique_path, output_dir)
        graphique_html = f'<img class="graphique" src="{html.escape(rel)}" alt="Graphiques de synthèse">'

    etat_corrections = _charger_etat_corrections(output_dir)
    lignes = [_ligne_html(r, chemins_annotes, output_dir, port_corrections, etat_corrections) for r in resultats]
    nb_image_corrigee = sum(
        1 for r in resultats if _etat_correction_rafale(r, etat_corrections)[0]
    )

    # Rafales actuellement dans la catégorie "à vérifier" (donc ni déjà
    # corrigées à la main, ni identifiées/non concluantes) -- dans l'ORDRE
    # de resultats, pour que le bouton "🔎 Vérifier toutes" ci-dessous
    # enchaîne les fenêtres dans le même ordre que le tableau du rapport.
    # Sert aussi à afficher un compteur sur le bouton de filtre existant.
    ids_a_verifier = [
        r.get("rafale_id") for r in resultats
        if _categorie_rafale(r, etat_corrections) == "a_verifier"
    ]

    th_correction = '\n  <th scope="col">Corriger</th>' if port_corrections else ""

    bouton_archiver = ""
    if port_corrections:
        lien_archiver = f"http://127.0.0.1:{port_corrections}/archiver"
        bouton_archiver = (
            f'<a class="btn-archiver" href="javascript:void(0)" '
            f'onclick="confirmerArchivage(\'{html.escape(lien_archiver)}\')">'
            f'🗄️ Archiver et repartir à zéro</a>'
        )

    # Bouton "🔎 Vérifier toutes" : ouvre côté appli, une par une, TOUTES
    # les fenêtres de correction des rafales "à vérifier", en enchaînant
    # automatiquement sur la suivante à chaque validation -- plutôt que de
    # devoir revenir cliquer "✏️ Corriger cette rafale" dans le rapport
    # après chacune (voir module_corrections/serveur_corrections.py, route
    # /corriger_serie, et InterfaceCanard._ouvrir_correction_serie).
    # Un lien direct plutôt qu'un import de serveur_corrections.construire_
    # lien_correction_serie : même choix que le lien "✏️ Corriger cette
    # rafale" ci-dessous (_ligne_html), pour que ce module reste autonome
    # de module_corrections/ (utilisable en CLI pur sans lui).
    bouton_verifier_serie = ""
    if port_corrections and ids_a_verifier:
        ids_txt = ",".join(str(rid) for rid in ids_a_verifier)
        lien_serie = f"http://127.0.0.1:{port_corrections}/corriger_serie?rafales={ids_txt}"
        bouton_verifier_serie = (
            f'<a class="btn-verifier-serie" href="{html.escape(lien_serie)}">'
            f'🔎 Vérifier les {len(ids_a_verifier)} rafale(s) "à vérifier", une par une</a>'
        )

    modale_confiance = """
<div id="modale-confiance" class="modale-fond">
  <div class="modale">
    <button class="fermer" onclick="fermerModale('modale-confiance')">&times;</button>
    <h2>Comment est calculée la confiance ?</h2>
    <p>C'est une moyenne pondérée de 3 scores, chacun entre 0 et 100% :</p>
    <table>
      <tr><th>Composante</th><th>Poids</th><th>D'où ça vient</th></tr>
      <tr><td>Temporelle</td><td>50%</td><td>À quel point l'heure de la photo colle à une
          observation du tableur : 95% si le match est à moins de 30s et sans ambiguïté,
          85% si à moins de 180s, ou 90%/75% pour les mêmes cas quand il a fallu filtrer
          par espèce/sexe pour désambiguïser.</td></tr>
      <tr><td>Espèce</td><td>25%</td><td>Probabilité renvoyée par l'IA Espèce pour la
          classe qu'elle a choisie (ex. 99% sûre que c'est un milouin).</td></tr>
      <tr><td>Sexe</td><td>25%</td><td>Probabilité renvoyée par l'IA Sexe pour la classe
          qu'elle a choisie.</td></tr>
    </table>
    <p>Si une IA n'a pas été utilisée pour ce canard (modèle non fourni, ou classification
       ratée), son poids est retiré et les poids restants sont ramenés à un total de 100%
       -- la confiance temporelle seule redevient alors tout le score, comme avant l'ajout
       des IA de classification.</p>
    <p><em>Exemple :</em> match temporel strict (95%) + IA Espèce très sûre (98%) mais IA
       Sexe non utilisée → poids disponibles = 50% + 25% = 75% →
       <code>(0.50×0.95 + 0.25×0.98) / 0.75 ≈ 96%</code>.</p>
    <p>Le tableau affiche le POURCENTAGE LE PLUS BAS parmi tous les canards de la rafale
       (le maillon le plus faible), avec un symbole selon le niveau :</p>
    <ul>
      <li><span class="confiance-haute">● haute</span> : ≥ 90%</li>
      <li><span class="confiance-moyenne">▲ moyenne</span> : entre 70% et 90%</li>
      <li><span class="confiance-faible">■ faible</span> : &lt; 70%</li>
    </ul>
    <p style="color:#888; font-size:12px;">Un score élevé reflète la cohérence des
       données (heure + classification), pas une certitude absolue -- il reste possible
       qu'une classification confiante soit fausse sur un cas atypique.</p>
  </div>
</div>"""

    html_doc = f"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="utf-8">
<title>Rapport d'identification des canards</title>
<style>{CSS}</style>
</head>
<body>
<div class="entete-actions">
  <div>
    <h1>🦆 Rapport d'identification des canards</h1>
    <p class="sous-titre">{synthese['nb_photos']} photos traitées en {synthese['nb_rafales']} rafales</p>
  </div>
  {bouton_archiver}
</div>

<div class="chiffres">{''.join(cartes)}</div>

{graphique_html}

<div class="filtres">
  <button class="actif" data-cat="toutes" onclick="filtrer('toutes')">Toutes ({len(resultats)})</button>
  <button data-cat="identifies" onclick="filtrer('identifies')">✅ Identifiées</button>
  <button data-cat="a_verifier" onclick="filtrer('a_verifier')">⚠️ À vérifier ({len(ids_a_verifier)})</button>
  <button data-cat="non_concluant" onclick="filtrer('non_concluant')">❌ Non concluantes</button>
  <button data-cat="image_corrigee" onclick="filtrer('image_corrigee')">✏️ Image corrigée ({nb_image_corrigee})</button>
  <button id="btn-marque" onclick="toggleMarque()">🏷️ Avec marque nasale détectée</button>
</div>

{f'<div id="barre-verifier-serie" class="barre-contextuelle" style="display:none;">{bouton_verifier_serie}</div>' if bouton_verifier_serie else ''}

<table>
<thead>
<tr>
  <th scope="col">Photo</th>
  <th scope="col" onclick="trier(1, true)">Rafale #</th>
  <th scope="col" onclick="trier(2, false)">Statut</th>
  <th scope="col" onclick="trier(3, false)">Bague(s)</th>
  <th scope="col" onclick="trier(4, false)">Espèce</th>
  <th scope="col" onclick="trier(5, false)">Sexe</th>
  <th scope="col" onclick="trier(6, true)">Confiance
    <a class="info-icone" href="javascript:void(0)" onclick="event.stopPropagation(); ouvrirModale('modale-confiance')" title="Comment est calculée la confiance ?">?</a>
  </th>
  <th scope="col" onclick="trier(7, true)">Corrigé</th>
  <th scope="col">Détail</th>{th_correction}
</tr>
</thead>
<tbody>
{''.join(lignes)}
</tbody>
</table>

{modale_confiance}

<script>{JS}</script>
</body>
</html>"""

    out_path = output_dir / "rapport.html"
    out_path.write_text(html_doc, encoding="utf-8")
    return str(out_path)