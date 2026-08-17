# -*- coding: utf-8 -*-
"""Tests unitaires du pipeline principal."""

from unittest.mock import MagicMock, Mock, patch

import pytest
from utils.pipeline import CanardIdentificationPipeline


@pytest.fixture
def mock_pipeline(sample_csv, tmp_path):
    """Crée un pipeline mocké pour les tests."""
    model_path = tmp_path / "mock_model.pt"
    model_path.touch()
    
    with patch("utils.pipeline.YOLO") as mock_yolo:
        mock_yolo.return_value = MagicMock()
        pipeline = CanardIdentificationPipeline(
            tableur_path=sample_csv,
            model_detection_path=str(model_path)
        )
        return pipeline


def test_pipeline_initializes(mock_pipeline):
    """Vérifie que le pipeline s'initialise correctement."""
    assert mock_pipeline.tableur is not None
    assert len(mock_pipeline.tableur) >= 0


def test_detect_canards_returns_structure(mock_pipeline, sample_photo_with_exif):
    """Vérifie que detect_canards retourne la bonne structure."""
    # Mocker la réponse YOLO
    mock_pipeline.ia_detection.predict = Mock(return_value=[])
    
    result = mock_pipeline.detect_canards(sample_photo_with_exif)
    
    assert "canards" in result
    assert "marques" in result
    assert isinstance(result["canards"], list)
    assert isinstance(result["marques"], list)


def test_select_best_photo_chooses_sharpest(mock_pipeline, sample_photos_burst):
    """Vérifie que la photo la plus nette est sélectionnée."""
    burst = {
        "photos": sample_photos_burst,
        "nb_photos": len(sample_photos_burst)
    }
    
    result = mock_pipeline.select_best_photo(burst)
    
    assert "best_photo" in result
    assert result["best_photo"] in sample_photos_burst
    assert "all_scores" in result


def test_match_exact_time_no_match(mock_pipeline):
    """Vérifie le comportement quand aucune observation ne correspond."""
    from datetime import datetime
    test_time = datetime(2099, 1, 1, 12, 0, 0)
    
    result = mock_pipeline.match_exact_time(test_time, tolerance_seconds=30)
    
    assert result["nb_matches"] == 0


def test_process_burst_no_duck_detected(mock_pipeline, sample_photos_burst):
    """Vérifie le comportement quand aucun canard n'est détecté."""
    from datetime import datetime
    
    burst = {
        "photos": sample_photos_burst,
        "nb_photos": len(sample_photos_burst),
        "timestamps": [datetime(2024, 3, 15, 14, 23, i) for i in range(len(sample_photos_burst))],
        "median_time": datetime(2024, 3, 15, 14, 23, 2),
        "start_time": datetime(2024, 3, 15, 14, 23, 0),
        "end_time": datetime(2024, 3, 15, 14, 23, 4),
        "duration": 4.0
    }
    
    # Mocker detect_canards pour retourner 0 canards
    mock_pipeline.detect_canards = Mock(return_value={"canards": [], "marques": []})
    
    result = mock_pipeline.process_burst(burst)
    
    assert result["status"] == "AUCUN_CANARD"
    # Non-régression : _build_rafale_info() est appelé même sur ce chemin
    # (AUCUN_CANARD) et exige burst["timestamps"] -- voir
    # test_build_rafale_info_includes_timestamps ci-dessous pour le detail
    # du bug (KeyError: 'timestamps' observé en production le 2026-08-03).
    assert result["rafale"]["timestamps"] == burst["timestamps"]


def test_build_rafale_info_includes_timestamps(mock_pipeline, sample_photos_burst):
    """Non-régression : KeyError: 'timestamps' observé en production le
    2026-08-03 dans 02_process_rafales.py::export_tableau_canards, qui
    indexe result["rafale"]["timestamps"] pour chaque photo de la rafale.
    _build_rafale_info() doit toujours reporter la clé "timestamps" du
    burst d'origine (voir utils/burst_detector.py, qui la fournit
    systématiquement), sans quoi export_tableau_canards plante dès qu'un
    lot contient plusieurs photos par rafale.
    """
    timestamps = [datetime(2024, 3, 15, 14, 23, i) for i in range(len(sample_photos_burst))]
    burst = {
        "nb_photos": len(sample_photos_burst),
        "photos": sample_photos_burst,
        "timestamps": timestamps,
        "median_time": timestamps[len(timestamps) // 2],
        "start_time": timestamps[0],
        "end_time": timestamps[-1],
        "duration": (timestamps[-1] - timestamps[0]).total_seconds(),
    }

    rafale_info = mock_pipeline._build_rafale_info(burst)

    assert "timestamps" in rafale_info
    assert rafale_info["timestamps"] == timestamps
    # Reproduit exactement l'indexation qui plantait dans
    # export_tableau_canards : un timestamp par photo de la rafale.
    for photo_idx, _ in enumerate(rafale_info["photos"]):
        assert photo_idx < len(rafale_info["timestamps"])