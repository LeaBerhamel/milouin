# -*- coding: utf-8 -*-
"""Chargement et nettoyage du tableur d'observations (Excel ou CSV)."""

import pandas as pd

from .config import logger, normaliser_espece, normaliser_sexe

COL_DATE = "DATE"
COL_HEURE = "HEURE"
COL_BAGUE = "BAGUE"
COL_CODE_MARQUE = "CODE MARQUE"
COL_COULEUR_MARQUE_SRC = "COULEUR MARQUE"
COL_FLAG_CORRECTE = "INDENTIFICATION CORRECTE"
COL_MODE = "MODE"
COL_ESPECE = "ESPECE"
COL_SEXE = "SEXE"


def load_tableur(tableur_path, verbose=True):
    """Charge le tableur d'observations (Excel ou CSV, séparateur tab ou virgule)."""
    if verbose:
        logger.info("Chargement du tableur...")

    if tableur_path.endswith((".xlsx", ".xls")):
        tableur = pd.read_excel(tableur_path, engine="openpyxl")
        if verbose:
            logger.info("Excel chargé : %d lignes", len(tableur))
    elif tableur_path.endswith(".csv"):
        try:
            tableur = pd.read_csv(tableur_path, sep="\t", encoding="utf-8")
        except Exception:
            logger.debug("Séparateur tabulation non détecté, repli sur la virgule", exc_info=True)
            tableur = pd.read_csv(tableur_path, sep=",", encoding="utf-8")
        if verbose:
            logger.info("CSV chargé : %d lignes", len(tableur))
    else:
        raise ValueError(f"Format non supporté : {tableur_path}")

    return tableur


def clean_tableur(tableur, verbose=True):
    """Filtre et normalise le tableur (identifications correctes, mode photo, dates, identifiants)."""
    def log(msg):
        if verbose:
            logger.info(msg.strip())

    if COL_FLAG_CORRECTE in tableur.columns:
        avant = len(tableur)
        tableur[COL_FLAG_CORRECTE] = tableur[COL_FLAG_CORRECTE].apply(
            lambda v: v if isinstance(v, bool) else str(v).strip().upper() == "VRAI"
        )
        tableur = tableur[tableur[COL_FLAG_CORRECTE] == True]
        log(f"   ✓ Filtre VRAI : {len(tableur)}/{avant} lignes conservées")

    if COL_MODE in tableur.columns:
        avant = len(tableur)
        tableur = tableur[tableur[COL_MODE].str.upper() == "PHOTO"]
        log(f"   ✓ Filtre PHOTO : {len(tableur)}/{avant} lignes conservées")

    if COL_DATE in tableur.columns:
        # Excel peut avoir déjà converti la colonne en datetime
        if not pd.api.types.is_datetime64_any_dtype(tableur[COL_DATE]):
            tableur[COL_DATE] = pd.to_datetime(tableur[COL_DATE], format="%d/%m/%Y", errors="coerce")

    if COL_HEURE in tableur.columns:
        tableur[COL_HEURE] = tableur[COL_HEURE].astype(str)
        tableur["DATETIME"] = pd.to_datetime(
            tableur[COL_DATE].astype(str).str.split(" ").str[0] + " " + tableur[COL_HEURE]
        )

    # BAGUE peut être vide : on retombe sur CODE MARQUE comme identifiant
    if COL_BAGUE in tableur.columns:
        if tableur[COL_BAGUE].isna().any() and COL_CODE_MARQUE in tableur.columns:
            tableur[COL_BAGUE] = (
                tableur[COL_BAGUE].fillna("").astype(str) + "_" +
                tableur[COL_CODE_MARQUE].fillna("").astype(str)
            )
            tableur[COL_BAGUE] = tableur[COL_BAGUE].str.strip("_")
            log("   ✓ BAGUE complétée avec CODE MARQUE")
    elif COL_CODE_MARQUE in tableur.columns:
        tableur[COL_BAGUE] = tableur[COL_CODE_MARQUE]
        log("   ✓ Utilisation de CODE MARQUE comme identifiant")

    if COL_COULEUR_MARQUE_SRC in tableur.columns:
        tableur["COULEUR_MARQUE"] = tableur[COL_COULEUR_MARQUE_SRC]

    # Unifie ESPECE et SEXE vers un seul vocabulaire (codes "AYTFER"/
    # "AYTFUL", "M"/"F"/"J") dès le chargement -- avant, ces colonnes
    # gardaient tel quel le contenu du tableur (parfois un nom français,
    # parfois un code, selon comment la ligne avait été saisie), ce qui
    # faisait apparaître le même individu sous deux libellés différents
    # plus loin dans le pipeline (matching temporel, synthèses). Normaliser
    # ici, une seule fois à la source, plutôt que dans chaque consommateur.
    if COL_ESPECE in tableur.columns:
        tableur[COL_ESPECE] = tableur[COL_ESPECE].apply(normaliser_espece)
    if COL_SEXE in tableur.columns:
        tableur[COL_SEXE] = tableur[COL_SEXE].apply(normaliser_sexe)

    avant = len(tableur)
    tableur = tableur.dropna(subset=["DATETIME"])
    log(f"   ✓ Lignes avec date/heure valide : {len(tableur)}/{avant}")

    if COL_BAGUE in tableur.columns:
        avant = len(tableur)
        # `!= ""` seul laisse passer les BAGUE manquantes : en pandas,
        # NaN != "" vaut True (NaN n'est jamais égal à rien, pas même via
        # !=), donc une ligne sans BAGUE ET sans CODE MARQUE de repli
        # (cf. bloc ci-dessus) restait considérée comme "avec identifiant"
        # -- elle finissait avec une BAGUE NaN dans le pipeline (matching
        # temporel, synthèse) au lieu d'être exclue ici comme prévu.
        bague_valide = tableur[COL_BAGUE].notna() & (tableur[COL_BAGUE].astype(str).str.strip() != "")
        tableur = tableur[bague_valide]
        log(f"   ✓ Lignes avec identifiant : {len(tableur)}/{avant}")

    log(f"\n   ✅ {len(tableur)} observations valides chargées\n")
    return tableur


def load_and_clean_tableur(tableur_path, verbose=True):
    """Charge puis nettoie le tableur d'observations en un seul appel."""
    return clean_tableur(load_tableur(tableur_path, verbose=verbose), verbose=verbose)