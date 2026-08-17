#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fusionne le dictionnaire_identifications.json (et crops_non_identifies.csv)
d'UN NOUVEAU LOT dans le dictionnaire MAÎTRE cumulatif -- nécessaire car
02_process_rafales.py régénère resultats_par_rafale.json à chaque lancement
(pas d'accumulation automatique d'un lot à l'autre) : sans cette étape, le
dictionnaire d'un lot précédent serait perdu en traitant le suivant.

Workflow recommandé pour ajouter un nouveau lot de photos :

    1. Traiter le nouveau lot dans SON PROPRE dossier de sortie (pas celui
       du dictionnaire maître, pour ne rien écraser) :

         python scripts/02_process_rafales.py --photos data/photos_lot2 \\
             --tableur data/observations.xlsx --output output/lots/lot2

         python scripts/03_export_json_dictionnaire.py \\
             --resultats output/lots/lot2/resultats_par_rafale.json \\
             --tableur data/observations.xlsx --output output/lots/lot2

    2. Fusionner ce lot dans le dictionnaire maître (output/ par défaut) :

         python scripts/07_fusionner_dictionnaire.py \\
             --nouveau output/lots/lot2/dictionnaire_identifications.json

    3. (Re)générer les crops confirmés / la banque de référence à partir du
       maître mis à jour, comme d'habitude :

         python scripts/04_copier_crops_valides.py
         python scripts/05_construire_banque_reference.py

Règle de fusion :
    - Une identification du nouveau lot est ajoutée au maître (ou remplace
      l'entrée existante si le même crop y était déjà -- ne devrait arriver
      que si vous retraitez un lot déjà fusionné).
    - Si un crop était marqué "non identifié" dans le maître mais apparaît
      maintenant identifié dans le nouveau lot (ex. le tableur a été
      complété entre-temps et vous avez retraité les mêmes photos), il est
      RETIRÉ de la liste des non-identifiés du maître -- il ne doit plus y
      traîner en double.
    - Un crop non identifié du nouveau lot n'est ajouté au maître que s'il
      n'y est pas déjà identifié.

Comme les noms de crop sont désormais uniques par photo source (voir
utils/crop_export.py), fusionner plusieurs lots ne peut pas mélanger deux
crops différents sous le même nom.

Usage:
    python scripts/07_fusionner_dictionnaire.py --nouveau output/lots/lot2/dictionnaire_identifications.json
"""

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils.dictionnaire import charger_exclusions, crops_deja_exclus  # noqa: E402


def charger_dictionnaire(path):
    path = Path(path)
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def charger_non_identifies(path):
    path = Path(path)
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return {row["crop_filename"]: row for row in csv.DictReader(f)}


def fusionner(maitre_dict, maitre_non_id, nouveau_dict, nouveau_non_id, exclus=frozenset()):
    """
    Fusionne en place (retourne aussi des compteurs pour le récap affiché).

    Args:
        exclus: Ensemble de crop_filename déjà exclus définitivement
            (ILLISIBLE / PAS_MARQUE, voir output/crops_exclus.json et
            utils/dictionnaire.py). Un crop exclu ne doit jamais être
            réinjecté dans la liste des non-identifiés du maître -- sinon
            il réapparaîtrait dans un futur dossier de relecture
            (04_preparer_dossier_relecture.py) alors qu'un chercheur l'a
            déjà explicitement écarté.

    Returns:
        dict de compteurs : nouveaux_identifies, ecrases, recuperes,
        nouveaux_non_identifies, ignores_deja_identifies, ignores_exclus.
    """
    compteurs = {
        "nouveaux_identifies": 0,
        "ecrases": 0,
        "recuperes": 0,
        "nouveaux_non_identifies": 0,
        "ignores_deja_identifies": 0,
        "ignores_exclus": 0,
    }

    for crop, infos in nouveau_dict.items():
        if crop in maitre_dict:
            compteurs["ecrases"] += 1
        else:
            compteurs["nouveaux_identifies"] += 1
        maitre_dict[crop] = infos
        if maitre_non_id.pop(crop, None) is not None:
            compteurs["recuperes"] += 1

    # Un crop déjà exclu (ILLISIBLE / PAS_MARQUE) ne doit jamais traîner
    # dans le maître des non-identifiés, même s'il y était déjà avant ce
    # correctif -- on le purge à chaque fusion, pas seulement à l'ajout.
    for crop in list(maitre_non_id):
        if crop in exclus:
            maitre_non_id.pop(crop, None)

    for crop, row in nouveau_non_id.items():
        if crop in maitre_dict:
            compteurs["ignores_deja_identifies"] += 1
            continue
        if crop in exclus:
            compteurs["ignores_exclus"] += 1
            continue
        if crop not in maitre_non_id:
            compteurs["nouveaux_non_identifies"] += 1
        maitre_non_id[crop] = row

    return compteurs


def sauvegarder(maitre_dict, maitre_non_id, dict_path, non_id_path):
    with open(dict_path, "w", encoding="utf-8") as f:
        json.dump(maitre_dict, f, indent=2, ensure_ascii=False)

    fieldnames = ["crop_filename", "date", "heure", "espece_detectee", "sexe_detectee"]
    with open(non_id_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in maitre_non_id.values():
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--nouveau", required=True,
                         help="dictionnaire_identifications.json du nouveau lot à fusionner")
    parser.add_argument("--nouveau-non-identifies", default=None,
                         help="crops_non_identifies.csv du nouveau lot "
                              "(par défaut : cherché à côté de --nouveau)")
    parser.add_argument("--maitre-dictionnaire", default="output/dictionnaire_identifications.json",
                         help="Dictionnaire maître cumulatif (créé s'il n'existe pas encore)")
    parser.add_argument("--maitre-non-identifies", default="output/crops_non_identifies.csv",
                         help="CSV maître des non-identifiés (créé s'il n'existe pas encore)")
    parser.add_argument("--exclusions", default="output/crops_exclus.json",
                         help="Fichier cumulatif des crops exclus (illisibles / faux positifs), "
                              "voir 08_importer_identifications_manuelles.py")
    args = parser.parse_args()

    nouveau_path = Path(args.nouveau)
    if not nouveau_path.exists():
        raise SystemExit(f"❌ {nouveau_path} introuvable.")

    nouveau_non_id_path = Path(args.nouveau_non_identifies) if args.nouveau_non_identifies \
        else nouveau_path.parent / "crops_non_identifies.csv"

    maitre_dict_path = Path(args.maitre_dictionnaire)
    maitre_non_id_path = Path(args.maitre_non_identifies)
    maitre_dict_path.parent.mkdir(parents=True, exist_ok=True)

    maitre_dict = charger_dictionnaire(maitre_dict_path)
    maitre_non_id = charger_non_identifies(maitre_non_id_path)
    nouveau_dict = charger_dictionnaire(nouveau_path)
    nouveau_non_id = charger_non_identifies(nouveau_non_id_path)
    exclus = crops_deja_exclus(charger_exclusions(Path(args.exclusions)))

    compteurs = fusionner(maitre_dict, maitre_non_id, nouveau_dict, nouveau_non_id, exclus)
    sauvegarder(maitre_dict, maitre_non_id, maitre_dict_path, maitre_non_id_path)

    print(f"✓ Fusion terminée -> {maitre_dict_path} ({len(maitre_dict)} identification(s) au total)")
    print(f"  + {compteurs['nouveaux_identifies']} nouvelle(s) identification(s)")
    if compteurs["ecrases"]:
        print(f"  ↻ {compteurs['ecrases']} identification(s) déjà présente(s), mise(s) à jour "
              "(lot déjà fusionné ?)")
    if compteurs["recuperes"]:
        print(f"  ⤴ {compteurs['recuperes']} crop(s) récupéré(s) (étaient non identifiés, "
              "le sont maintenant)")
    print(f"✓ {maitre_non_id_path} ({len(maitre_non_id)} crop(s) toujours non identifié(s) au total)")
    if compteurs["ignores_deja_identifies"]:
        print(f"  ({compteurs['ignores_deja_identifies']} crop(s) du nouveau lot ignoré(s) côté "
              "non-identifiés : déjà identifiés par ailleurs)")
    if compteurs["ignores_exclus"]:
        print(f"  ({compteurs['ignores_exclus']} crop(s) du nouveau lot ignoré(s) côté "
              "non-identifiés : déjà exclus -- ILLISIBLE/PAS_MARQUE, voir "
              f"{args.exclusions})")


if __name__ == "__main__":
    main()