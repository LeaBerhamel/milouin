#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fusionne le corrections.csv (+ les crops référencés dans crops_corrections/)
d'UNE SOURCE (output/ courant, ou une archive datée sous output_archives/)
dans un corrections.csv MAÎTRE cumulatif -- même problème et même remède
que 07_fusionner_dictionnaire.py : 01_archiver_resultats.py déplace tout
output/ (corrections.csv compris) vers output_archives/<date>/ à chaque
nouveau lot, donc sans cette fusion les corrections d'un lot archivé
deviennent invisibles pour l'écran "Réentraîner depuis les corrections"
(interface_preparation_ia.py, EcranCorrections) dès que le lot suivant est
traité.

Contrairement à dictionnaire_identifications.json (un dict, une entrée par
crop), corrections.csv est un journal d'ajouts (plusieurs lignes possibles
pour un même crop au fil du temps, une ligne ne portant que les champs
réellement changés à ce moment-là -- voir ecrire_corrections.py). La
fusion est donc une CONCATÉNATION des lignes de la source dans le maître
(dédupliquée si une ligne identique existe déjà, pour rester idempotent si
on relance la fusion deux fois sur la même source), et non un remplacement
par clé comme pour le dictionnaire.

Problème réglé au passage : corrections.csv stocke le CHEMIN COMPLET du
crop au moment de la correction (voir ecrire_corrections.py). Une fois la
source archivée (shutil.move), ce chemin ne pointe plus vers rien -- le
fichier a physiquement suivi le dossier archivé. Cette fusion retrouve
donc chaque crop par son NOM DE FICHIER dans <source>/crops_corrections/
(jamais par le chemin stocké, qui peut être périmé), le copie dans
<maître>/crops_corrections/, puis réécrit crop_path pour pointer vers sa
nouvelle place dans le maître -- ainsi 10_fusionner_corrections_dataset.py
(et l'aperçu de EcranCorrections) continuent à trouver l'image, quel que
soit le nombre d'archivages passés entretemps. Un crop introuvable dans la
source (image supprimée à la main, ou repli "rafaleX_canardYY" sans image
réelle -- voir fenetre_rafale._ouvrir_correction) n'est pas perdu : la
ligne est quand même copiée dans le maître (au cas où l'image
réapparaîtrait), simplement ignorée en aval comme aujourd'hui.

Usage:
    python scripts/15_fusionner_corrections.py --source output_archives/2026-08-01_14h32
    python scripts/15_fusionner_corrections.py --source output
"""

import argparse
import csv
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

COLONNES_CORRECTIONS = ["date", "crop_path", "espece_corrigee", "sexe_corrige", "couleur_corrigee"]


def charger_corrections(csv_path: Path):
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return []
    with open(csv_path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fusionner_corrections_csv(
    maitre_csv: Path,
    maitre_crops_dir: Path,
    source_csv: Path,
    source_crops_dir: Path,
):
    """Fusionne en place (retourne des compteurs pour le récap affiché).

    Args:
        maitre_csv: corrections.csv cumulatif à mettre à jour (créé s'il
            n'existe pas encore).
        maitre_crops_dir: dossier crops_corrections/ à côté de maitre_csv,
            où les crops de la source sont copiés.
        source_csv: corrections.csv de la source à intégrer (output/
            courant, ou une archive datée).
        source_crops_dir: crops_corrections/ de cette même source (pour
            retrouver les images par nom de fichier).

    Returns:
        dict de compteurs : nouvelles_lignes, deja_presentes,
        crops_copies, crops_introuvables.
    """
    compteurs = {
        "nouvelles_lignes": 0,
        "deja_presentes": 0,
        "crops_copies": 0,
        "crops_introuvables": 0,
    }

    maitre_lignes = charger_corrections(maitre_csv)
    source_lignes = charger_corrections(source_csv)
    if not source_lignes:
        return compteurs

    # Clé de déduplication : le contenu complet de la ligne APRÈS
    # réécriture du crop_path (deux corrections différentes sur le même
    # crop, à des dates différentes, doivent rester deux lignes distinctes
    # -- seule une ligne strictement identique à une déjà présente est un
    # doublon, ex. fusion relancée deux fois sur la même archive).
    deja_vues = {tuple(l.get(c, "") for c in COLONNES_CORRECTIONS) for l in maitre_lignes}

    source_crops_dir = Path(source_crops_dir)
    maitre_crops_dir = Path(maitre_crops_dir)

    for ligne in source_lignes:
        ligne = dict(ligne)  # copie -- ne pas modifier l'original en cas de ré-essai
        nom_crop = Path(ligne.get("crop_path", "")).name

        crop_source = source_crops_dir / nom_crop if nom_crop else None
        if crop_source and crop_source.exists():
            maitre_crops_dir.mkdir(parents=True, exist_ok=True)
            crop_dest = maitre_crops_dir / nom_crop
            if not crop_dest.exists():
                shutil.copy2(crop_source, crop_dest)
            ligne["crop_path"] = str(crop_dest)
            compteurs["crops_copies"] += 1
        else:
            # Image introuvable dans la source (déjà le cas avant fusion,
            # ou repli sans image réelle) -- on garde la ligne telle
            # quelle, simplement ignorée en aval comme aujourd'hui.
            compteurs["crops_introuvables"] += 1

        cle = tuple(ligne.get(c, "") for c in COLONNES_CORRECTIONS)
        if cle in deja_vues:
            compteurs["deja_presentes"] += 1
            continue
        deja_vues.add(cle)
        maitre_lignes.append(ligne)
        compteurs["nouvelles_lignes"] += 1

    maitre_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(maitre_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLONNES_CORRECTIONS)
        writer.writeheader()
        for ligne in maitre_lignes:
            writer.writerow({c: ligne.get(c, "") for c in COLONNES_CORRECTIONS})

    return compteurs


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", required=True,
                         help="Dossier source (output/ courant, ou une archive datée sous "
                              "output_archives/) contenant corrections.csv + crops_corrections/")
    parser.add_argument("--maitre", default="output_archives/dictionnaire_maitre",
                         help="Dossier du corrections.csv maître cumulatif (par défaut : "
                              "output_archives/dictionnaire_maitre, le même dossier que le "
                              "dictionnaire d'identification maître)")
    args = parser.parse_args()

    source_dir = Path(args.source)
    source_csv = source_dir / "corrections.csv"
    if not source_csv.exists():
        print(f"ℹ️  Aucun {source_csv} -- rien à fusionner.")
        return

    maitre_dir = Path(args.maitre)
    compteurs = fusionner_corrections_csv(
        maitre_csv=maitre_dir / "corrections.csv",
        maitre_crops_dir=maitre_dir / "crops_corrections",
        source_csv=source_csv,
        source_crops_dir=source_dir / "crops_corrections",
    )

    print(f"✓ Fusion terminée -> {maitre_dir / 'corrections.csv'}")
    print(f"  + {compteurs['nouvelles_lignes']} nouvelle(s) ligne(s) de correction")
    if compteurs["deja_presentes"]:
        print(f"  ↻ {compteurs['deja_presentes']} ligne(s) déjà présente(s), ignorée(s) "
              "(fusion déjà faite ?)")
    print(f"  🖼 {compteurs['crops_copies']} crop(s) copié(s) vers le maître")
    if compteurs["crops_introuvables"]:
        print(f"  ⚠️ {compteurs['crops_introuvables']} ligne(s) sans image retrouvée "
              "(ignorée(s) lors du réentraînement, comme avant)")


if __name__ == "__main__":
    main()