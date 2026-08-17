# -*- coding: utf-8 -*-
"""Tests unitaires du détecteur de rafales."""

from datetime import datetime

from utils.burst_detector import BurstDetector


def test_extract_datetime_from_valid_photo(sample_photo_with_exif):
    """Vérifie l'extraction d'EXIF sur une photo valide."""
    detector = BurstDetector()
    dt = detector.extract_datetime(sample_photo_with_exif)
    
    assert dt is not None
    assert isinstance(dt, datetime)
    assert dt.year == 2024
    assert dt.month == 3
    assert dt.day == 15


def test_extract_datetime_returns_none_for_invalid_photo(tmp_path):
    """Vérifie que None est retourné si pas d'EXIF."""
    invalid_photo = tmp_path / "no_exif.jpg"
    invalid_photo.write_bytes(b"fake image data")
    
    detector = BurstDetector()
    dt = detector.extract_datetime(str(invalid_photo))
    assert dt is None


def test_group_photos_into_bursts(sample_photos_burst):
    """Vérifie que les photos sont bien groupées en rafales."""
    detector = BurstDetector()
    rafales = detector.group_photos_into_bursts(sample_photos_burst, max_gap_seconds=2)
    
    assert len(rafales) == 1  # Toutes les photos dans 1 rafale
    assert rafales[0]["nb_photos"] == 5
    assert rafales[0]["duration"] == 4.0  # 5 photos espacées de 1s = 4s


def test_group_photos_splits_on_gap(sample_photos_burst, tmp_path):
    """Vérifie que les rafales sont séparées si l'écart dépasse le seuil."""
    from PIL import Image
    
    # Ajouter une photo 10 secondes après
    late_photo = tmp_path / "burst_late.jpg"
    img = Image.new("RGB", (640, 480), color="red")
    exif_data = img.getexif()
    exif_data[36867] = "2024:03:15 14:23:10"
    img.save(late_photo, exif=exif_data)
    
    all_photos = sample_photos_burst + [str(late_photo)]
    
    detector = BurstDetector()
    rafales = detector.group_photos_into_bursts(all_photos, max_gap_seconds=2)
    
    assert len(rafales) == 2  # Devrait créer 2 rafales distinctes