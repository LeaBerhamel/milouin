# -*- coding: utf-8 -*-
"""
Copie les crops identifiés (depuis dictionnaire_identifications.json)
dans crops_valides/<identifiant>/, organisés par individu.

Usage :
    python scripts/04_copier_crops_valides.py
    python scripts/04_copier_crops_valides.py --dictionnaire output/lots/lot2/dictionnaire_identifications.json

Expose aussi copier_crops() comme fonction réutilisable -- appelée
directement par interface_graphique.py (chargement dynamique du module,
voir _generer_dictionnaire_et_suggestions) pour enchaîner 03->04->05->06
sans repasser par la ligne de commande à chaque traitement.
"""

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "output"
# NOTE : le dossier réellement produit par 02_process_rafales.py est
# "crops_marques_nasales" (pas "crops") -- avec l'ancien défaut ("output/crops",
# qui n'existe jamais), le script échouait silencieusement en CLI avec
# "Dossier crops introuvable" dès qu'on ne précisait pas --crops-dir à la main.
CROPS_DIR = OUTPUT_DIR / "crops_marques_nasales"
CROPS_VALIDES_DIR = OUTPUT_DIR / "crops_valides"


def load_dict(path: Path) -> dict[str, Any]:
    """Charge le dictionnaire JSON depuis un fichier."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
        return dict(data)  # type: ignore[arg-type]


def copier_crops(dict_path: Path, crops_src: Path, crops_dest: Path, log=print) -> dict:
    """Copie chaque crop identifié dans dictionnaire_identifications.json
    (dict_path) depuis crops_src vers crops_dest/<identifiant>/.

    Additif et idempotent : crops_dest n'est jamais vidé, et recopier un
    crop déjà présent l'écrase juste avec le même contenu -- donc appeler
    cette fonction à répétition (ex. à chaque traitement, voir
    interface_graphique.py) fait bien grandir crops_dest au fil du temps
    plutôt que de le réinitialiser.

    Args:
        dict_path: chemin vers un dictionnaire_identifications.json
            (format {crop_filename: {"identifiant": ..., ...}}).
        crops_src: dossier où se trouvent physiquement les crops listés
            dans dict_path (typiquement <output_dir>/crops_marques_nasales).
        crops_dest: dossier de destination organisé par individu
            (typiquement <output_dir>/crops_valides, ou un dossier
            maître cumulatif partagé entre plusieurs lots/archives).
        log: fonction d'affichage (print par défaut, injectable pour
            rediriger vers un journal Tkinter).

    Returns:
        dict de compteurs : {"n_total", "n_copies", "n_manquants",
        "individus": {identifiant: nb_crops}}.
    """
    dict_path = Path(dict_path)
    crops_src = Path(crops_src)
    crops_dest = Path(crops_dest)

    if not dict_path.exists():
        log(f"❌ Dictionnaire introuvable : {dict_path}")
        return {"n_total": 0, "n_copies": 0, "n_manquants": 0, "individus": {}}

    if not crops_src.exists():
        log(f"❌ Dossier crops introuvable : {crops_src}")
        return {"n_total": 0, "n_copies": 0, "n_manquants": 0, "individus": {}}

    dictionnaire = load_dict(dict_path)
    n_total = len(dictionnaire)
    n_copies = 0
    n_manquants = 0
    individus: dict[str, int] = {}

    crops_dest.mkdir(parents=True, exist_ok=True)

    for crop_filename, metadata in dictionnaire.items():
        identifiant = metadata.get("identifiant")
        if not identifiant:
            log(f"  ⚠️  Pas d'identifiant pour {crop_filename}, ignoré.")
            continue

        src = crops_src / crop_filename
        if not src.exists():
            log(f"  ⚠️  Crop manquant : {crop_filename}")
            n_manquants += 1
            continue

        dest_dir = crops_dest / identifiant
        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest_dir / crop_filename)
        n_copies += 1
        individus[identifiant] = individus.get(identifiant, 0) + 1

    log(f"✅ {n_copies}/{n_total} crop(s) copié(s) dans {crops_dest}")
    if n_manquants:
        log(f"⚠️  {n_manquants} crop(s) absent(s) du dossier source (ignorés).")
    if individus:
        log(f"📊 {len(individus)} individu(s) concerné(s) par cette copie :")
        for ident, count in sorted(individus.items(), key=lambda x: x[1], reverse=True):
            log(f"   {ident} : {count} crop(s)")

    return {"n_total": n_total, "n_copies": n_copies, "n_manquants": n_manquants, "individus": individus}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dictionnaire",
        default=str(OUTPUT_DIR / "dictionnaire_identifications.json"),
        help="Chemin vers le dictionnaire d'identifications (défaut : output/dictionnaire_identifications.json)",
    )
    parser.add_argument(
        "--crops-dir",
        default=str(CROPS_DIR),
        help="Dossier source des crops (défaut : output/crops_marques_nasales)",
    )
    parser.add_argument(
        "--output",
        default=str(CROPS_VALIDES_DIR),
        help="Dossier de destination organisé par individu (défaut : output/crops_valides)",
    )
    args = parser.parse_args()

    copier_crops(Path(args.dictionnaire), Path(args.crops_dir), Path(args.output))


if __name__ == "__main__":
    main()