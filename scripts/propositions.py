# -*- coding: utf-8 -*-
"""
Détection des incohérences entre ce que déclare le tableur d'observations
et ce que les IA de classification ont vu sur la photo.

Utile pour deux choses différentes :
- Repérer une IA de classification peu fiable (si les IA se trompent
  souvent sur des cas par ailleurs bien identifiés).
- Repérer des erreurs de SAISIE dans le tableur lui-même (ex. sexe inversé
  à la capture) — c'est en général la cause la plus fréquente, car le
  matching temporel qui a permis l'identification est indépendant de la
  classification espèce/sexe.
"""

import csv
from pathlib import Path


def find_incoherences(resultats):
    """
    Parcourt tous les canards identifiés et retourne ceux pour lesquels au
    moins un champ ("espece", "sexe", "couleur_marque") diffère entre la
    valeur déclarée dans le tableur et la valeur vue par l'IA.

    Returns:
        Liste de dictionnaires, un par incohérence détectée (un canard peut
        apparaître plusieurs fois s'il a plusieurs champs incohérents).
    """
    lignes = []
    for r in resultats:
        for canard in r.get("canards", []):
            cohérent = canard.get("cohérent")
            if not cohérent:
                continue
            for champ, ok in cohérent.items():
                if ok:
                    continue
                valeur_ia = {
                    "espece": canard.get("espece_detectee"),
                    "sexe": canard.get("sexe_detectee"),
                    "couleur_marque": canard.get("couleur_detectee"),
                }.get(champ)
                lignes.append({
                    "rafale_id": r.get("rafale_id"),
                    "bague": canard.get("bague"),
                    "champ": champ,
                    "valeur_tableur": canard.get(champ),
                    "valeur_ia": valeur_ia,
                    "photo": r.get("photo_analysee"),
                })
    return lignes


def export_incoherences_csv(resultats, output_path):
    """Exporte les incohérences tableur/IA en CSV. Retourne le nombre de
    lignes exportées (0 si aucune incohérence, dans ce cas aucun fichier
    n'est écrit)."""
    lignes = find_incoherences(resultats)
    if not lignes:
        return 0

    output_path = Path(output_path)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["rafale_id", "bague", "champ", "valeur_tableur", "valeur_ia", "photo"])
        writer.writeheader()
        writer.writerows(lignes)
    return len(lignes)
