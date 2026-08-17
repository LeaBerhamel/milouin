# -*- coding: utf-8 -*-
"""
Lecture des correspondances visuelles suggérées par
scripts/06_proposer_correspondances_visuelles.py (fichier
output/propositions_visuelles.csv), et libellé lisible pour un score de
similarité.

Extrait de scripts/09_ecran_identification.py pour être réutilisé aussi
par la fenêtre de correction déclenchée depuis le rapport HTML -- une
seule source de vérité pour lire ce CSV plutôt que deux copies qui
pourraient diverger.

Fichier optionnel : produit par la chaîne de consolidation (03->07->04->
05->06), pas par 02_process_rafales.py. charger_propositions() renvoie un
dict vide si le fichier n'existe pas encore -- l'appelant doit alors se
comporter comme s'il n'y avait aucune suggestion (pas planter).
"""

import csv
from pathlib import Path

VERT = "#3fae5b"
ORANGE = "#d4a017"
GRIS = "#999999"


def charger_propositions(path: Path, top_k: int = 3) -> dict:
    """Retourne {crop_filename: (candidats, statut)} où candidats est une
    liste de tuples (identifiant, crop_reference, score), dans l'ordre
    décroissant de ressemblance (candidat_1 en premier).

    Fichier absent -> dict vide (comportement attendu, pas une erreur).
    """
    path = Path(path)
    if not path.exists():
        return {}
    props = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            candidats = []
            for i in range(1, top_k + 1):
                ident = row.get(f"candidat_{i}", "")
                if not ident:
                    continue
                candidats.append((
                    ident,
                    row.get(f"crop_reference_{i}", ""),
                    row.get(f"score_{i}", ""),
                ))
            props[row["crop_filename"]] = (candidats, row.get("statut", "AUCUNE_REFERENCE"))
    return props


def libelle_pour_score(score_str):
    """(texte, couleur) lisibles pour un score de similarité (chaîne ou
    nombre, peut être vide/invalide -- renvoie alors un libellé neutre)."""
    try:
        score = float(score_str)
    except (TypeError, ValueError):
        return ("ressemblance inconnue", GRIS)
    if score >= 0.90:
        return ("ressemblance forte", VERT)
    if score >= 0.80:
        return ("ressemblance moyenne", ORANGE)
    return ("ressemblance faible", GRIS)


def score_pourcentage(score_str):
    """Score formaté en pourcentage entier ('94%'), ou '' si invalide."""
    try:
        return f"{float(score_str) * 100:.0f}%"
    except (TypeError, ValueError):
        return ""
