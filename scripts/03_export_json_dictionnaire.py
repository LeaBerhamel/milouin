#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Génère le dictionnaire JSON final {crop_marque_nasale.jpg: identifiant_canard}
à partir des résultats du pipeline (output/resultats_par_rafale.json) et des
crops de marque nasale sauvegardés par save_marque_crops.

IMPORTANT (v2) : le pipeline (utils/pipeline.py, méthode match_exact_time)
stocke le résultat d'un match avec des clés en MINUSCULES ("bague", "espece",
"sexe", "couleur_marque") et ne conserve PAS la DATE/HEURE de la ligne du
tableur qui a matché (seulement l'identité de l'observation, pas son
horodatage officiel). La v1 de ce script cherchait "BAGUE"/"DATE"/"HEURE" en
majuscules -> toujours None. Corrigé ici : on lit les bonnes clés, et on
retrouve la DATE/HEURE officielle en refaisant une recherche légère dans le
tableur lui-même (bague + horodatage de la photo), sans aucune IA -> pas
besoin de relancer tout le pipeline (torch/ultralytics), juste pandas.

Prérequis : avoir lancé scripts/02_process_rafales.py au préalable (avec la
version mise à jour qui sauvegarde les crops dans
output/crops_marques_nasales/).

Usage:
    python scripts/03_export_json_dictionnaire.py

Sorties (dans --output, par défaut output/) :
    - dictionnaire_identifications.json : {crop_filename: {identifiant, date, heure, sexe, espece}}
    - crops_non_identifies.csv : crops dont le canard associé n'a pas pu être
      rattaché à une bague, avec la date/heure de la photo (aucune ligne du
      tableur n'a été retenue pour ceux-là).
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils.tableur import load_and_clean_tableur  # noqa: E402


def _format_date(value):
    """Normalise une date/datetime (objet pandas ou string) en 'YYYY-MM-DD'."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    return str(value).split(" ")[0]


def _format_heure(value):
    """Normalise une heure/datetime (objet pandas ou string) en 'HH:MM:SS'."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    s = str(value)
    if " " in s:
        s = s.split(" ")[1]
    return s


def find_observation_datetime(tableur, bague, photo_datetime, tolerance_seconds=180):
    """
    Retrouve, DANS LE TABLEUR, la date/heure officielle de l'observation qui a
    servi à identifier ce canard -- le pipeline ne la conserve pas dans ses
    résultats, seulement la bague/espèce/sexe/couleur. On reproduit ici la
    même logique de recherche que match_exact_time (bague + fenêtre
    temporelle autour de l'horodatage de la photo), sans aucune IA.

    Args:
        tableur: DataFrame nettoyé (colonnes DATE, HEURE, DATETIME, BAGUE...).
        bague: Identifiant (valeur de la clé "bague" du résultat du pipeline).
        photo_datetime: Horodatage de la photo (ex. rafale["median_time"]).
        tolerance_seconds: Fenêtre de recherche (180s = large, comme le
            pipeline en dernier recours -- un vrai match a déjà été validé
            par le pipeline, ici on ne fait que retrouver LA ligne exacte).

    Returns:
        (date, heure) tuple de strings, ou (None, None) si rien retrouvé
        (ne devrait arriver que si le tableur a changé entre-temps).
    """
    if bague is None or photo_datetime is None:
        return None, None

    photo_dt = pd.to_datetime(photo_datetime)
    tol = pd.Timedelta(seconds=tolerance_seconds)

    candidats = tableur[
        (tableur["BAGUE"].astype(str) == str(bague)) &
        ((tableur["DATETIME"] - photo_dt).abs() <= tol)
    ]
    if len(candidats) == 0:
        return None, None

    candidats = candidats.copy()
    candidats["ecart"] = (candidats["DATETIME"] - photo_dt).abs()
    row = candidats.sort_values("ecart").iloc[0]
    return _format_date(row.get("DATE")), _format_heure(row.get("HEURE"))


def build_dictionnaire(resultats, tableur):
    """
    Parcourt les résultats de rafales et construit :
      - `identifications` : dict {crop_filename: {...}} pour les marques
        rattachées à un canard IDENTIFIE.
      - `non_identifies` : liste de dicts (une ligne par crop non identifié).
    """
    identifications = {}
    non_identifies = []

    for rafale in resultats:
        rafale_time = (rafale.get("rafale") or {}).get("median_time")
        canards = rafale.get("canards", [])
        marques = rafale.get("marques", [])

        for marque in marques:
            crop_filename = marque.get("crop_filename")
            if not crop_filename:
                continue

            canard_idx = marque.get("canard_assigne")
            canard = canards[canard_idx] if canard_idx is not None and canard_idx < len(canards) else None

            if canard is not None and canard.get("status") == "IDENTIFIE":
                identifiant = canard.get("bague")
                date, heure = find_observation_datetime(tableur, identifiant, rafale_time)
                if date is None:
                    # Repli : la date/heure officielle du tableur n'a pas pu
                    # être retrouvée (cas limite) -> on utilise celle de la
                    # photo elle-même, déjà très proche (c'est justement le
                    # critère qui a permis le match), plutôt que de perdre
                    # l'info.
                    date, heure = _format_date(rafale_time), _format_heure(rafale_time)
                identifications[crop_filename] = {
                    "identifiant": identifiant,
                    "date": date,
                    "heure": heure,
                    "sexe": canard.get("sexe") or canard.get("sexe_detectee"),
                    "espece": canard.get("espece") or canard.get("espece_detectee"),
                }
            else:
                non_identifies.append({
                    "crop_filename": crop_filename,
                    "date": _format_date(rafale_time),
                    "heure": _format_heure(rafale_time),
                    "espece_detectee": canard.get("espece_detectee") if canard else None,
                    "sexe_detectee": canard.get("sexe_detectee") if canard else None,
                })

    return identifications, non_identifies


def export_csv(non_identifies, csv_path):
    # Colonnes volontairement réduites à ce qui est utile pour quelqu'un qui
    # doit juste regarder une image et deviner une bague : rafale_id (interne),
    # le chemin complet de la photo, le statut technique et le diagnostic sont
    # retirés -- le nom du crop suffit à retrouver l'image.
    fieldnames = ["crop_filename", "date", "heure", "espece_detectee", "sexe_detectee"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(non_identifies)


def main():
    parser = argparse.ArgumentParser(
        description="Export du dictionnaire JSON crop -> identifiant à partir des résultats du pipeline")
    parser.add_argument("--resultats", default="output/resultats_par_rafale.json",
                         help="Chemin vers resultats_par_rafale.json")
    parser.add_argument("--tableur", default="data/observations.xlsx",
                         help="Tableur d'observations (pour retrouver la DATE/HEURE officielle)")
    parser.add_argument("--output", default="output", help="Dossier de sortie")
    args = parser.parse_args()

    resultats_path = Path(args.resultats)
    if not resultats_path.exists():
        raise SystemExit(
            f"❌ {resultats_path} introuvable. Lance d'abord scripts/02_process_rafales.py."
        )
    tableur_path = Path(args.tableur)
    if not tableur_path.exists():
        raise SystemExit(f"❌ {tableur_path} introuvable.")

    with open(resultats_path, encoding="utf-8") as f:
        resultats = json.load(f)

    tableur = load_and_clean_tableur(str(tableur_path), verbose=False)

    identifications, non_identifies = build_dictionnaire(resultats, tableur)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "dictionnaire_identifications.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(identifications, f, indent=2, ensure_ascii=False)

    csv_path = output_dir / "crops_non_identifies.csv"
    export_csv(non_identifies, csv_path)

    print(f"✓ {json_path} ({len(identifications)} crop(s) identifié(s))")
    sans_date = sum(1 for v in identifications.values() if v["date"] is None)
    if sans_date:
        print(f"  ⚠️  {sans_date} identification(s) sans date/heure retrouvée (repli photo utilisé ou tableur modifié)")
    print(f"✓ {csv_path} ({len(non_identifies)} crop(s) non identifié(s), référencés avec date/heure)")


if __name__ == "__main__":
    main()