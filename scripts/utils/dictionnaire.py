# -*- coding: utf-8 -*-
"""
Logique partagée autour de la valeur "bague" saisie par un humain (que ce
soit via 09_ecran_identification.py, la fenêtre de correction déclenchée
depuis le rapport, ou tout futur écran) : distinguer une vraie bague d'un
code d'exclusion, et gérer le fichier cumulatif des exclusions.

Extrait de scripts/08_importer_identifications_manuelles.py pour être une
source de vérité unique -- 07_fusionner_dictionnaire.py et le module de
correction déclenché depuis le rapport en ont aussi besoin.

Convention pour une valeur de bague saisie à la main :

    LT6         -> une vraie bague.
    ILLISIBLE   -> marque nasale visible mais impossible à lire (flou,
                   angle, reflet, marque abîmée...). Exclu DÉFINITIVEMENT.
    PAS_MARQUE  -> ce n'est pas une marque nasale (fausse détection).
                   Exclu définitivement, journalisé séparément.
    (vide)      -> pas de valeur, rien à faire.
"""

import json
from datetime import date
from pathlib import Path

CODES_EXCLUSION = {
    "ILLISIBLE": "illisible",
    "PAS_MARQUE": "pas_marque",
}


def est_code_exclusion(valeur: str) -> bool:
    """True si `valeur` (une fois normalisée) est un code d'exclusion
    (ILLISIBLE / PAS_MARQUE) plutôt qu'une vraie bague."""
    return (valeur or "").strip().upper() in CODES_EXCLUSION


def raison_exclusion(valeur: str) -> str:
    """Raison courte ('illisible' / 'pas_marque') pour un code d'exclusion.
    Lève KeyError si `valeur` n'est pas un code d'exclusion -- vérifier
    avec est_code_exclusion() avant si besoin."""
    return CODES_EXCLUSION[(valeur or "").strip().upper()]


def charger_exclusions(path: Path) -> list:
    path = Path(path)
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def sauvegarder_exclusions(exclusions: list, path: Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(exclusions, f, ensure_ascii=False, indent=2)


def crops_deja_exclus(exclusions: list) -> set:
    """Ensemble des crop_filename déjà présents dans une liste d'exclusions
    (telle que retournée par charger_exclusions)."""
    return {e["crop_filename"] for e in exclusions}


def enregistrer_exclusion(exclusions: list, crop_filename: str, valeur: str) -> bool:
    """Ajoute `crop_filename` à la liste `exclusions` (mutée en place) avec
    la raison déduite de `valeur` (ILLISIBLE/PAS_MARQUE), sauf s'il y est
    déjà. Retourne True si une entrée a réellement été ajoutée.

    Ne sauvegarde rien sur disque -- appeler sauvegarder_exclusions()
    ensuite (regroupé côté appelant pour éviter d'écrire à chaque crop).
    """
    if crop_filename in crops_deja_exclus(exclusions):
        return False
    exclusions.append({
        "crop_filename": crop_filename,
        "raison": raison_exclusion(valeur),
        "date_exclusion": date.today().isoformat(),
    })
    return True
