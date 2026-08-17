#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Construit la banque de référence visuelle : une empreinte (embedding) par
crop confirmé de output/crops_valides/<identifiant>/ (généré par
scripts/04_copier_crops_valides.py), utilisée ensuite pour proposer des
correspondances visuelles aux crops non identifiés par le tableur
(scripts/06_proposer_correspondances_visuelles.py).

Ne prend QUE les crops confirmés (verifie_tableur=True) comme référence --
c'est volontaire : on ne veut pas qu'une identification déjà incertaine
(repli photo) serve de base à en proposer d'autres, ce qui propagerait
l'incertitude.

Comme le dossier crops_valides/ s'enrichit à chaque lancement de
04_copier_crops_valides.py (il n'est jamais vidé), relancer ce script après
chaque nouveau lot traité enrichit automatiquement la banque avec les
nouvelles confirmations.

Usage:
    python scripts/05_construire_banque_reference.py
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils.embeddings import extract_embedding  # noqa: E402


def construire_banque(crops_valides_dir):
    """
    Parcourt output/crops_valides/<identifiant>/*.jpg et calcule une
    empreinte par crop.

    Returns:
        dict {identifiant: [(crop_filename, embedding), ...]}
    """
    crops_valides_dir = Path(crops_valides_dir)
    banque = {}
    if not crops_valides_dir.exists():
        return banque

    identifiant_dirs = sorted(p for p in crops_valides_dir.iterdir() if p.is_dir())
    for identifiant_dir in identifiant_dirs:
        crops = sorted(list(identifiant_dir.glob("*.jpg")) + list(identifiant_dir.glob("*.jpeg")))
        if not crops:
            continue
        embeddings = []
        for crop_path in crops:
            try:
                emb = extract_embedding(crop_path)
            except Exception as e:
                print(f"  ⚠️  {crop_path.name} ignoré ({e})")
                continue
            embeddings.append((crop_path.name, emb))
        if embeddings:
            banque[identifiant_dir.name] = embeddings
            print(f"  {identifiant_dir.name} : {len(embeddings)} crop(s)")
    return banque


def sauvegarder_banque(banque, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    index = []
    vecteurs = []
    for identifiant, embeddings in banque.items():
        for crop_name, emb in embeddings:
            index.append({"identifiant": identifiant, "crop_filename": crop_name})
            vecteurs.append(emb)

    if vecteurs:
        np.savez(output_dir / "embeddings.npz", vecteurs=np.array(vecteurs, dtype=np.float32))
    with open(output_dir / "index.json", "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--crops-valides", default="output/crops_valides",
                         help="Dossier des crops confirmés (sortie de scripts/04_copier_crops_valides.py)")
    parser.add_argument("--output", default="output/banque_reference",
                         help="Dossier de sortie de la banque")
    args = parser.parse_args()

    crops_valides_dir = Path(args.crops_valides)
    if not crops_valides_dir.exists():
        raise SystemExit(
            f"❌ {crops_valides_dir} introuvable. Lance d'abord "
            "scripts/03_export_json_dictionnaire.py puis scripts/04_copier_crops_valides.py."
        )

    print("Chargement du modèle d'empreintes visuelles (ResNet18 pré-entraîné)...")
    print("Calcul des empreintes par individu :")
    banque = construire_banque(crops_valides_dir)

    sauvegarder_banque(banque, args.output)

    n_individus = len(banque)
    n_crops = sum(len(v) for v in banque.values())
    print(f"\n✓ Banque de référence construite : {n_individus} individu(s), {n_crops} crop(s) au total.")
    print(f"  -> {args.output}/embeddings.npz + index.json")
    if n_crops == 0:
        print("  ⚠️  Aucun crop trouvé dans crops_valides/. "
              "Lance d'abord scripts/04_copier_crops_valides.py.")
    mono_crop = sum(1 for v in banque.values() if len(v) == 1)
    if mono_crop:
        print(f"  ℹ️  {mono_crop} individu(s) n'ont qu'un seul crop de référence : "
              "les propositions les concernant seront moins fiables (une seule vue).")


if __name__ == "__main__":
    main()