# -*- coding: utf-8 -*-
"""Tests unitaires du module tableur."""

import pandas as pd
import pytest
from utils.tableur import clean_tableur, load_and_clean_tableur, load_tableur


def test_load_csv(sample_csv):
    """Vérifie que le CSV se charge correctement."""
    df = load_tableur(sample_csv, verbose=False)
    assert len(df) > 0
    assert "DATE" in df.columns
    assert "HEURE" in df.columns


def test_load_excel(sample_excel):
    """Vérifie que le Excel se charge correctement."""
    df = load_tableur(sample_excel, verbose=False)
    assert len(df) == 2
    assert "BAGUE" in df.columns


def test_clean_filters_vrai(sample_csv):
    """Vérifie que seules les lignes VRAI sont gardées."""
    df = load_tableur(sample_csv, verbose=False)
    df_clean = clean_tableur(df, verbose=False)
    
    # Sur 5 lignes, 3 ont VRAI (mais 1 n'a pas de date valide, 1 n'est pas PHOTO)
    assert len(df_clean) == 2


def test_clean_filters_photo_mode(sample_csv):
    """Vérifie que seul le mode PHOTO est gardé."""
    df = load_and_clean_tableur(sample_csv, verbose=False)
    assert all(df["MODE"].str.upper() == "PHOTO")


def test_clean_creates_datetime(sample_csv):
    """Vérifie que la colonne DATETIME est créée."""
    df = load_and_clean_tableur(sample_csv, verbose=False)
    assert "DATETIME" in df.columns
    assert pd.api.types.is_datetime64_any_dtype(df["DATETIME"])


def test_clean_removes_invalid_datetime(sample_csv):
    """Vérifie que les lignes sans date/heure valide sont supprimées."""
    df = load_and_clean_tableur(sample_csv, verbose=False)
    assert not df["DATETIME"].isna().any()


def test_clean_removes_empty_bague():
    """Vérifie que les lignes sans identifiant sont supprimées."""
    # Créer un DataFrame avec BAGUE vide
    df = pd.DataFrame({
        "DATE": ["15/03/2024"],
        "HEURE": ["14:23:45"],
        "BAGUE": [""],
        "MODE": ["PHOTO"],
        "INDENTIFICATION CORRECTE": ["VRAI"]
    })
    df["DATE"] = pd.to_datetime(df["DATE"], format="%d/%m/%Y")
    df_clean = clean_tableur(df, verbose=False)
    assert len(df_clean) == 0


def test_unsupported_format_raises_error(tmp_path):
    """Vérifie qu'un format non supporté lève une erreur."""
    invalid_file = tmp_path / "test.txt"
    invalid_file.write_text("test")
    
    with pytest.raises(ValueError, match="Format non supporté"):
        load_tableur(str(invalid_file), verbose=False)