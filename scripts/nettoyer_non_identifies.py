import csv
import json
from pathlib import Path

archive = Path("output_archives/2026-08-04_12h30")
csv_path = archive / "crops_non_identifies.csv"
exclus_path = archive / "crops_exclus.json"

exclus = {e["crop_filename"] for e in json.load(open(exclus_path, encoding="utf-8"))}

with open(csv_path, encoding="utf-8") as f:
    rows = list(csv.DictReader(f))
    fieldnames = rows[0].keys() if rows else ["crop_filename", "date", "heure", "espece_detectee", "sexe_detectee"]

restants = [r for r in rows if r["crop_filename"] not in exclus]

with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(restants)

print(f"{len(rows) - len(restants)} ligne(s) retirée(s) (déjà exclues)")
print(f"{len(restants)} ligne(s) encore réellement en attente de relecture")