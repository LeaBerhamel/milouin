#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prépare un dossier autonome, prêt à transmettre à un relecteur humain, ne
contenant QUE les crops de marque nasale non identifiés automatiquement
(output/crops_marques_nasales/ contient tous les crops, identifiés compris --
ce script isole le sous-ensemble utile à la relecture).

Les crops déjà exclus définitivement lors d'une relecture précédente
(output/crops_exclus.json, voir scripts/08_importer_identifications_manuelles.py)
sont automatiquement écartés : pas la peine de les faire relire une seconde
fois.

Convention pour la colonne "identifiant_manuel" que le relecteur remplit :

    LT6         -> une vraie bague : le crop est identifié.
    ILLISIBLE   -> marque visible mais impossible à lire (flou, angle,
                   reflet, marque abîmée...).
    PAS_MARQUE  -> ce crop n'est pas une marque nasale (fausse détection).
    (vide)      -> pas encore relu, laissé tel quel.

Une fois le CSV rempli, réintègre-le avec :
    python scripts/08_importer_identifications_manuelles.py --csv <ce_csv>

Prérequis : avoir déjà lancé 02_process_rafales.py (crops sauvegardés) puis
03_export_json_dictionnaire.py (crops_non_identifies.csv généré).

Usage:
    python scripts/04_preparer_dossier_relecture.py

Sortie :
    output/a_identifier_manuellement/
        crops_non_identifies.csv   (copie + colonne identifiant_manuel, pour l'avoir avec les images)
        <crop_filename>.jpg        (un fichier par ligne du CSV)
    output/a_identifier_manuellement.zip   (même contenu, zippé pour l'envoi)
"""

import argparse
import csv
import json
import shutil
import zipfile
from pathlib import Path


def charger_exclusions(path: Path) -> set:
    if not path.exists():
        return set()
    with open(path, encoding="utf-8") as f:
        return {e["crop_filename"] for e in json.load(f)}


def main():
    parser = argparse.ArgumentParser(
        description="Prépare le dossier (+ zip) à transmettre au relecteur humain")
    parser.add_argument("--csv", default="output/crops_non_identifies.csv",
                         help="CSV produit par 03_export_json_dictionnaire.py")
    parser.add_argument("--crops-dir", default="output/crops_marques_nasales",
                         help="Dossier contenant TOUS les crops (identifiés + non identifiés)")
    parser.add_argument("--exclusions", default="output/crops_exclus.json",
                         help="Crops déjà relus et exclus définitivement (illisible / pas une marque)")
    parser.add_argument("--output", default="output/a_identifier_manuellement",
                         help="Dossier de sortie (créé s'il n'existe pas)")
    parser.add_argument("--no-zip", action="store_true",
                         help="Ne pas créer le zip, dossier seul")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    crops_dir = Path(args.crops_dir)
    output_dir = Path(args.output)

    if not csv_path.exists():
        raise SystemExit(
            f"❌ {csv_path} introuvable. Lance d'abord "
            "scripts/03_export_json_dictionnaire.py."
        )
    if not crops_dir.exists():
        raise SystemExit(f"❌ {crops_dir} introuvable.")

    output_dir.mkdir(parents=True, exist_ok=True)

    with open(csv_path, encoding="utf-8") as f:
        toutes_les_lignes = list(csv.DictReader(f))

    exclus = charger_exclusions(Path(args.exclusions))
    rows = [r for r in toutes_les_lignes if r["crop_filename"] not in exclus]
    n_deja_exclus = len(toutes_les_lignes) - len(rows)

    # On ajoute une colonne vide "identifiant_manuel" : c'est là que le
    # relecteur écrira la bague qu'il identifie à l'œil sur chaque crop (ou
    # ILLISIBLE / PAS_MARQUE -- voir convention dans l'entête du script). Le
    # CSV original (sans cette colonne) n'est pas modifié, on en écrit une
    # copie enrichie dans le dossier de relecture.
    fieldnames = list(toutes_les_lignes[0].keys()) + ["identifiant_manuel"] if toutes_les_lignes else \
        ["crop_filename", "rafale_id", "photo", "date", "heure",
         "statut_canard", "espece_detectee", "sexe_detectee", "diagnostic", "identifiant_manuel"]

    copied, manquants = 0, []
    for row in rows:
        row["identifiant_manuel"] = ""
        src = crops_dir / row["crop_filename"]
        if src.exists():
            shutil.copy2(src, output_dir / row["crop_filename"])
            copied += 1
        else:
            manquants.append(row["crop_filename"])

    csv_out = output_dir / "crops_non_identifies.csv"
    with open(csv_out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"✓ {copied} crop(s) copié(s) -> {output_dir}/")
    if n_deja_exclus:
        print(f"ℹ️  {n_deja_exclus} crop(s) déjà exclu(s) lors d'une relecture précédente "
              "(illisible / pas une marque) -- écarté(s), pas re-proposé(s).")
    if manquants:
        print(f"⚠️  {len(manquants)} fichier(s) référencés dans le CSV mais introuvables "
              f"dans {crops_dir} (ex. {manquants[:3]})")

    if not rows:
        print("ℹ️  Rien à relire pour l'instant (tout est déjà identifié ou exclu).")
        return

    if not args.no_zip:
        zip_path = output_dir.with_suffix(".zip")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for path in output_dir.iterdir():
                zf.write(path, arcname=path.name)
        print(f"✓ {zip_path} (prêt à transmettre)")

    print(f"\nUne fois relu, réintègre avec :")
    print(f"  python scripts/08_importer_identifications_manuelles.py --csv {csv_out}")


if __name__ == "__main__":
    main()