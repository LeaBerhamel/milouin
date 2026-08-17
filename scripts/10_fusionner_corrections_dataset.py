#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fusionne les corrections manuelles issues de l'écran d'identification (corrections.csv)
dans les répertoires de datasets d'entraînement enrichis.

Usage:
    python scripts/10_fusionner_corrections_dataset.py --corrections output/corrections.csv
"""

import argparse
import csv
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def fusionner_corrections(
    corrections_csv: Path,
    dataset_espece: Path,
    dataset_sexe: Path,
    dataset_couleur: Path,
):
    if not corrections_csv.exists():
        print(f"❌ Fichier de corrections introuvable : {corrections_csv}")
        return

    compteurs = {"espece": 0, "sexe": 0, "couleur": 0}

    with open(corrections_csv, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            crop_path = REPO_ROOT / row.get("crop_path", "")
            if not crop_path.exists():
                continue

            # 1. Espèce
            espece = row.get("espece_corrigee") or row.get("espece")
            if espece and espece.lower() not in ("inconnu", ""):
                dest = dataset_espece / espece.lower() / crop_path.name
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(crop_path, dest)
                compteurs["espece"] += 1

            # 2. Sexe
            sexe = row.get("sexe_corrige") or row.get("sexe")
            if sexe and sexe.lower() not in ("inconnu", "indetermine_juvenile", ""):
                dest = dataset_sexe / sexe.lower() / crop_path.name
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(crop_path, dest)
                compteurs["sexe"] += 1

            # 3. Couleur
            couleur = row.get("couleur_corrigee") or row.get("couleur")
            if couleur and couleur.lower() not in ("indeterminable", ""):
                dest = dataset_couleur / couleur.lower() / crop_path.name
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(crop_path, dest)
                compteurs["couleur"] += 1

    print("✅ Fusion des corrections terminée avec succès !")
    print(f"   • Espèce  : {compteurs['espece']} images ajoutées/mises à jour")
    print(f"   • Sexe    : {compteurs['sexe']} images ajoutées/mises à jour")
    print(f"   • Couleur : {compteurs['couleur']} images ajoutées/mises à jour")


def main():
    parser = argparse.ArgumentParser(description="Fusionne corrections.csv dans les datasets d'entraînement.")
    parser.add_argument("--corrections", default="output/corrections.csv", help="Chemin vers corrections.csv")
    parser.add_argument("--dataset-espece", default="dataset_espece", help="Dossier dataset espèce")
    parser.add_argument("--dataset-sexe", default="dataset_sexe", help="Dossier dataset sexe")
    parser.add_argument("--dataset-couleur", default="dataset_couleur", help="Dossier dataset couleur")
    args = parser.parse_args()

    fusionner_corrections(
        corrections_csv=REPO_ROOT / args.corrections,
        dataset_espece=REPO_ROOT / args.dataset_espece,
        dataset_sexe=REPO_ROOT / args.dataset_sexe,
        dataset_couleur=REPO_ROOT / args.dataset_couleur,
    )


if __name__ == "__main__":
    main()