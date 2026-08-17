#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Propose, pour les crops NON identifiés par le tableur (ceux listés dans
output/crops_non_identifies.csv, généré par
scripts/03_export_json_dictionnaire.py), les individus déjà connus dont la
marque nasale ressemble le plus visuellement -- via la banque de référence
construite par scripts/05_construire_banque_reference.py.

⚠️ CECI NE FAIT AUCUNE IDENTIFICATION AUTOMATIQUE. La similarité visuelle
par un CNN pré-entraîné généraliste (ImageNet, jamais ré-entraîné sur vos
marques nasales) est un INDICE, pas une preuve : deux marques de couleur ou
forme proches peuvent se ressembler sans être le même individu, et une même
marque peut varier visuellement selon l'éclairage/l'angle/l'usure. Chaque
proposition est à vérifier PHOTO À L'APPUI (crop candidat vs crop à
vérifier, mis côte à côte) avant toute correction du dictionnaire officiel
-- ne jamais écrire directement dans dictionnaire_identifications.json à
partir de ce script.

Sortie : output/propositions_visuelles.csv, une ligne par crop analysé avec
son (ou ses) meilleur(s) candidat(s) et un score de similarité (0 à 1,
1 = identique). Trois statuts, à ajuster avec l'usage :
    CORRESPONDANCE_PROBABLE          score >= 0.90 : à vérifier en priorité
    RESSEMBLANCE_A_REVOIR            0.80 <= score < 0.90
    INDIVIDU_POTENTIELLEMENT_NOUVEAU score < 0.80

Usage:
    python scripts/06_proposer_correspondances_visuelles.py
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils.embeddings import extract_embedding  # noqa: E402

SEUIL_PROBABLE = 0.90
SEUIL_A_REVOIR = 0.80


def charger_banque(banque_dir):
    banque_dir = Path(banque_dir)
    npz_path = banque_dir / "embeddings.npz"
    index_path = banque_dir / "index.json"
    if not npz_path.exists() or not index_path.exists():
        raise SystemExit(
            f"❌ Banque introuvable dans {banque_dir}. "
            "Lance d'abord scripts/05_construire_banque_reference.py."
        )
    vecteurs = np.load(npz_path)["vecteurs"]
    with open(index_path, encoding="utf-8") as f:
        index = json.load(f)
    return vecteurs, index


def meilleurs_candidats(embedding, vecteurs, index, top_k=3):
    """
    Retourne jusqu'à top_k (identifiant, crop_filename_reference, score),
    triés par score décroissant -- un seul résultat par identifiant (son
    meilleur crop de référence), pour ne pas monopoliser le classement avec
    plusieurs crops du même individu.
    """
    if len(vecteurs) == 0:
        return []
    scores = vecteurs @ embedding  # produit scalaire = cosinus (vecteurs normalisés)
    meilleur_par_individu = {}
    for score, entry in zip(scores, index):
        ident = entry["identifiant"]
        if ident not in meilleur_par_individu or score > meilleur_par_individu[ident][0]:
            meilleur_par_individu[ident] = (float(score), entry["crop_filename"])
    classement = sorted(
        ((ident, crop, score) for ident, (score, crop) in meilleur_par_individu.items()),
        key=lambda x: x[2], reverse=True,
    )
    return classement[:top_k]


def statut_pour_score(score):
    if score is None:
        return "AUCUNE_REFERENCE"
    if score >= SEUIL_PROBABLE:
        return "CORRESPONDANCE_PROBABLE"
    if score >= SEUIL_A_REVOIR:
        return "RESSEMBLANCE_A_REVOIR"
    return "INDIVIDU_POTENTIELLEMENT_NOUVEAU"


def proposer_correspondances(crops_a_verifier_dir, crops_non_identifies_csv, banque_dir,
                              output_csv, top_k=3, log=print):
    """
    Compare chaque crop non identifié (listé dans crops_non_identifies_csv) à
    la banque de référence et écrit les meilleures correspondances dans
    output_csv.

    Fonction réutilisable (extraite de main()) : appelée par la CLI mais
    aussi par interface_graphique.py pour enchaîner cette étape
    automatiquement après un traitement.

    Returns:
        dict avec n_lignes, n_probable, n_a_revoir, n_nouveau, n_manquants
        ou None si la banque ou le CSV des non-identifiés sont introuvables
        (message déjà loggé, ne lève pas d'exception -- cas normal au
        premier lancement, tant qu'aucun individu n'est encore confirmé).
    """
    banque_dir = Path(banque_dir)
    npz_path = banque_dir / "embeddings.npz"
    index_path = banque_dir / "index.json"
    if not npz_path.exists() or not index_path.exists():
        log(f"ℹ️  Banque de référence introuvable ou vide dans {banque_dir} -- "
            "aucun individu confirmé pour l'instant, pas de suggestion possible.")
        return None
    vecteurs = np.load(npz_path)["vecteurs"]
    with open(index_path, encoding="utf-8") as f:
        index = json.load(f)

    csv_path = Path(crops_non_identifies_csv)
    if not csv_path.exists():
        log(f"❌ {csv_path} introuvable. Lance d'abord scripts/03_export_json_dictionnaire.py.")
        return None
    with open(csv_path, encoding="utf-8") as f:
        crops_a_traiter = [row["crop_filename"] for row in csv.DictReader(f)]

    if not crops_a_traiter:
        log("ℹ️  Aucun crop non identifié -- rien à comparer.")
        return {"n_lignes": 0, "n_probable": 0, "n_a_revoir": 0, "n_nouveau": 0, "n_manquants": 0}

    n_individus_banque = len(set(e["identifiant"] for e in index))
    log("Chargement du modèle d'empreintes visuelles (ResNet18 pré-entraîné)...")
    log(f"{len(crops_a_traiter)} crop(s) à comparer à une banque de "
        f"{len(index)} crop(s) de référence ({n_individus_banque} individu(s) connus).")

    crops_dir = Path(crops_a_verifier_dir)
    lignes = []
    manquants = []

    for crop_filename in crops_a_traiter:
        crop_path = crops_dir / crop_filename
        if not crop_path.exists():
            manquants.append(crop_filename)
            continue

        emb = extract_embedding(crop_path)
        candidats = meilleurs_candidats(emb, vecteurs, index, top_k=top_k)

        ligne = {"crop_filename": crop_filename}
        meilleur_score = candidats[0][2] if candidats else None
        ligne["statut"] = statut_pour_score(meilleur_score)
        for i in range(top_k):
            if i < len(candidats):
                ident, crop_ref, score = candidats[i]
                ligne[f"candidat_{i + 1}"] = ident
                ligne[f"score_{i + 1}"] = round(score, 4)
                ligne[f"crop_reference_{i + 1}"] = crop_ref
            else:
                ligne[f"candidat_{i + 1}"] = ""
                ligne[f"score_{i + 1}"] = ""
                ligne[f"crop_reference_{i + 1}"] = ""
        lignes.append(ligne)

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["crop_filename", "statut"]
    for i in range(top_k):
        fieldnames += [f"candidat_{i + 1}", f"score_{i + 1}", f"crop_reference_{i + 1}"]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(lignes)

    n_probable = sum(1 for l in lignes if l["statut"] == "CORRESPONDANCE_PROBABLE")
    n_a_revoir = sum(1 for l in lignes if l["statut"] == "RESSEMBLANCE_A_REVOIR")
    n_nouveau = sum(1 for l in lignes if l["statut"] == "INDIVIDU_POTENTIELLEMENT_NOUVEAU")

    log(f"\n✓ {output_path} ({len(lignes)} crop(s) analysé(s))")
    log(f"  {n_probable} correspondance(s) probable(s) (score >= {SEUIL_PROBABLE}) -- à vérifier en priorité")
    log(f"  {n_a_revoir} ressemblance(s) à revoir (score >= {SEUIL_A_REVOIR})")
    log(f"  {n_nouveau} individu(s) potentiellement nouveau(x) (score < {SEUIL_A_REVOIR})")
    if manquants:
        log(f"  ⚠️  {len(manquants)} crop(s) introuvable(s) sur disque dans {crops_dir} : "
            f"{manquants[:5]}" + (", ..." if len(manquants) > 5 else ""))
    log("\n⚠️  Rappel : ce sont des PROPOSITIONS à vérifier photo à l'appui, pas des identifications.")

    return {"n_lignes": len(lignes), "n_probable": n_probable, "n_a_revoir": n_a_revoir,
            "n_nouveau": n_nouveau, "n_manquants": len(manquants)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--crops-a-verifier", default="output/crops_marques_nasales",
                         help="Dossier contenant les crops (haute résolution) à comparer")
    parser.add_argument("--crops-non-identifies", default="output/crops_non_identifies.csv",
                         help="CSV listant les crops non identifiés par le tableur")
    parser.add_argument("--banque", default="output/banque_reference",
                         help="Dossier de la banque de référence")
    parser.add_argument("--output", default="output/propositions_visuelles.csv")
    parser.add_argument("--top-k", type=int, default=3, help="Nombre de candidats proposés par crop")
    args = parser.parse_args()

    resultat = proposer_correspondances(
        args.crops_a_verifier, args.crops_non_identifies, args.banque, args.output, top_k=args.top_k,
    )
    if resultat is None:
        raise SystemExit(1)


if __name__ == "__main__":
    main()