#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Réintègre dans le dictionnaire MAÎTRE les identifications faites à la main
par un relecteur humain sur le CSV produit par
scripts/04_preparer_dossier_relecture.py (colonne "identifiant_manuel"
remplie).

Convention pour la colonne "identifiant_manuel" (voir aussi
04_preparer_dossier_relecture.py) :

    LT6         -> une vraie bague : le crop est identifié, ajouté au
                   dictionnaire maître comme les identifications automatiques.
    ILLISIBLE   -> marque nasale visible mais impossible à lire (flou,
                   angle, reflet, marque abîmée...). Exclu DÉFINITIVEMENT :
                   ne réapparaîtra plus jamais dans un dossier de relecture.
    PAS_MARQUE  -> ce crop n'est pas une marque nasale (fausse détection du
                   modèle de détection). Exclu définitivement, et journalisé
                   séparément -- utile pour un futur ré-entraînement du
                   détecteur (liste de faux positifs connus).
    (vide)      -> pas encore relu, laissé tel quel : réapparaîtra au
                   prochain scripts/04_preparer_dossier_relecture.py.

Les crops exclus (ILLISIBLE / PAS_MARQUE) sont accumulés dans
output/crops_exclus.json (jamais écrasé, seulement enrichi) et retirés de
la liste des non-identifiés du maître -- ils ne consommeront donc plus
jamais de temps de relecture.

Usage :
    python scripts/08_importer_identifications_manuelles.py \\
        --csv output/a_identifier_manuellement/crops_non_identifies.csv
"""

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils.dictionnaire import est_code_exclusion, enregistrer_exclusion  # noqa: E402


def charger_json(path: Path, defaut):
    if not path.exists():
        return defaut
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def charger_non_identifies(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return {row["crop_filename"]: row for row in csv.DictReader(f)}


def sauvegarder_json(data, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def sauvegarder_non_identifies(non_id: dict, path: Path):
    fieldnames = ["crop_filename", "date", "heure", "espece_detectee", "sexe_detectee"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in non_id.values():
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", required=True,
                         help="CSV relu par un humain (colonne identifiant_manuel remplie)")
    parser.add_argument("--maitre-dictionnaire", default="output/dictionnaire_identifications.json",
                         help="Dictionnaire maître cumulatif")
    parser.add_argument("--maitre-non-identifies", default="output/crops_non_identifies.csv",
                         help="CSV maître des non-identifiés")
    parser.add_argument("--exclusions", default="output/crops_exclus.json",
                         help="Fichier cumulatif des crops exclus (illisibles / faux positifs)")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise SystemExit(f"❌ {csv_path} introuvable.")

    with open(csv_path, encoding="utf-8") as f:
        # Le CSV peut avoir été édité dans Excel/LibreOffice -> tolère ; ou ,
        # comme séparateur en détectant automatiquement.
        echantillon = f.read(4096)
        f.seek(0)
        try:
            dialecte = csv.Sniffer().sniff(echantillon, delimiters=";,")
        except csv.Error:
            dialecte = csv.excel
        rows = list(csv.DictReader(f, dialect=dialecte))

    if not rows:
        print(f"ℹ️  {csv_path} est vide, rien à importer.")
        return
    if "identifiant_manuel" not in rows[0]:
        raise SystemExit(
            f"❌ Colonne 'identifiant_manuel' absente de {csv_path} -- "
            "ce n'est pas un CSV produit par 04_preparer_dossier_relecture.py ?"
        )

    dict_path = Path(args.maitre_dictionnaire)
    non_id_path = Path(args.maitre_non_identifies)
    exclusions_path = Path(args.exclusions)

    maitre_dict = charger_json(dict_path, {})
    maitre_non_id = charger_non_identifies(non_id_path)
    exclusions = charger_json(exclusions_path, [])

    n_identifies, n_exclus, n_ignores_vides, n_deja_traites = 0, 0, 0, 0

    for row in rows:
        crop = row["crop_filename"]
        valeur = (row.get("identifiant_manuel") or "").strip()

        if valeur == "":
            n_ignores_vides += 1
            continue

        if est_code_exclusion(valeur):
            maitre_non_id.pop(crop, None)
            if enregistrer_exclusion(exclusions, crop, valeur):
                n_exclus += 1
            else:
                n_deja_traites += 1
            continue

        # Sinon : une vraie bague.
        if crop in maitre_dict:
            n_deja_traites += 1
        else:
            n_identifies += 1
        maitre_dict[crop] = {
            "identifiant": valeur,
            "date": row.get("date"),
            "heure": row.get("heure"),
            "sexe": row.get("sexe_detectee") or None,
            "espece": row.get("espece_detectee") or None,
            "source": "manuel",
        }
        maitre_non_id.pop(crop, None)

    sauvegarder_json(maitre_dict, dict_path)
    sauvegarder_non_identifies(maitre_non_id, non_id_path)
    sauvegarder_json(exclusions, exclusions_path)

    print(f"✓ {dict_path} mis à jour ({len(maitre_dict)} identification(s) au total)")
    print(f"  + {n_identifies} nouvelle(s) identification(s) manuelle(s)")
    print(f"✓ {exclusions_path} mis à jour ({len(exclusions)} crop(s) exclu(s) au total)")
    print(f"  + {n_exclus} nouvelle(s) exclusion(s) (illisible / pas une marque)")
    if n_deja_traites:
        print(f"  ↻ {n_deja_traites} ligne(s) déjà traitée(s) précédemment (ignorée(s))")
    if n_ignores_vides:
        print(f"  … {n_ignores_vides} ligne(s) encore vide(s) (pas encore relues, laissées de côté)")
    print(f"✓ {non_id_path} ({len(maitre_non_id)} crop(s) toujours en attente de relecture)")

    n_illisible = sum(1 for e in exclusions if e["raison"] == "illisible")
    n_pas_marque = sum(1 for e in exclusions if e["raison"] == "pas_marque")
    if n_pas_marque:
        print(f"\nℹ️  {n_pas_marque} crop(s) marqué(s) 'pas une marque' au total dans "
              f"{exclusions_path} -- utile à consulter avant un futur ré-entraînement "
              "du modèle de détection (faux positifs connus).")
    if n_illisible:
        print(f"ℹ️  {n_illisible} crop(s) marqué(s) 'illisible' au total.")


if __name__ == "__main__":
    main()