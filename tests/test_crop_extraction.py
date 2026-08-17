# -*- coding: utf-8 -*-
"""Tests unitaires de l'extraction de crops."""

from pathlib import Path
from unittest.mock import Mock, patch

from utils.crop_extraction import extract_crops, list_photos


def test_list_photos_finds_jpg(tmp_path):
    """Vérifie que list_photos trouve bien les fichiers jpg/JPG."""
    (tmp_path / "photo1.jpg").touch()
    (tmp_path / "photo2.JPG").touch()
    (tmp_path / "readme.txt").touch()
    
    photos = list_photos(tmp_path)
    
    assert len(photos) == 2
    assert all(p.suffix.lower() == ".jpg" for p in photos)


def test_extract_crops_creates_output_dir(tmp_path, sample_photos_burst):
    """Vérifie que les dossiers de sortie sont créés."""
    output_dir = tmp_path / "output"
    model_path = tmp_path / "model.pt"
    model_path.touch()
    
    with patch("utils.crop_extraction.YOLO") as mock_yolo:
        mock_yolo.return_value.predict = Mock(return_value=[])
        
        extract_crops(
            photos_dir=Path(sample_photos_burst[0]).parent,
            model_detection_path=str(model_path),
            output_dir=str(output_dir),
            target_class=0,
            label_field="test"
        )
        
        assert (output_dir / "crops").exists()
        assert (output_dir / "metadata.json").exists()