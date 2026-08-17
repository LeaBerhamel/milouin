# -*- coding: utf-8 -*-
"""Configuration et fixtures partagées pour pytest."""

from datetime import datetime

import pandas as pd
import pytest
from PIL import Image


@pytest.fixture
def sample_csv(tmp_path):
    """Crée un fichier CSV de test avec des observations valides."""
    csv_path = tmp_path / "observations.csv"
    data = """DATE\tHEURE\tBAGUE\tESPECE\tCOULEUR MARQUE\tMODE\tINDENTIFICATION CORRECTE
15/03/2024\t14:23:45\tA12345\tColvert\tJaune\tPHOTO\tVRAI
16/03/2024\t09:15:30\tB67890\tSouchet\tVert\tPHOTO\tVRAI
17/03/2024\t16:45:12\t\tChipeau\tRouge\tVISUEL\tVRAI
18/03/2024\t11:30:00\tC11111\tPilet\tBleu\tPHOTO\tFAUX
19/03/2024\t\tD22222\tSarcelle\tOrange\tPHOTO\tVRAI"""
    csv_path.write_text(data, encoding="utf-8")
    return str(csv_path)


@pytest.fixture
def sample_excel(tmp_path):
    """Crée un fichier Excel de test."""
    excel_path = tmp_path / "observations.xlsx"
    df = pd.DataFrame({
        "DATE": ["15/03/2024", "16/03/2024"],
        "HEURE": ["14:23:45", "09:15:30"],
        "BAGUE": ["A12345", "B67890"],
        "ESPECE": ["Colvert", "Souchet"],
        "COULEUR MARQUE": ["Jaune", "Vert"],
        "MODE": ["PHOTO", "PHOTO"],
        "INDENTIFICATION CORRECTE": ["VRAI", "VRAI"]
    })
    df.to_excel(excel_path, index=False, engine="openpyxl")
    return str(excel_path)


@pytest.fixture
def sample_photo_with_exif(tmp_path):
    """Crée une photo de test avec métadonnées EXIF."""
    photo_path = tmp_path / "test_photo.jpg"
    img = Image.new("RGB", (640, 480), color="blue")
    
    # Ajouter des métadonnées EXIF basiques
    exif_data = img.getexif()
    exif_data[36867] = "2024:03:15 14:23:45"  # DateTimeOriginal
    
    img.save(photo_path, exif=exif_data)
    return str(photo_path)


@pytest.fixture
def sample_photos_burst(tmp_path):
    """Crée une série de photos simulant une rafale."""
    photos = []
    base_time = datetime(2024, 3, 15, 14, 23, 0)
    
    for i in range(5):
        photo_path = tmp_path / f"burst_{i:03d}.jpg"
        img = Image.new("RGB", (640, 480), color="green")
        exif_data = img.getexif()
        
        # Photos espacées de 1 seconde
        timestamp = base_time.replace(second=base_time.second + i)
        exif_data[36867] = timestamp.strftime("%Y:%m:%d %H:%M:%S")
        
        img.save(photo_path, exif=exif_data)
        photos.append(str(photo_path))
    
    return photos