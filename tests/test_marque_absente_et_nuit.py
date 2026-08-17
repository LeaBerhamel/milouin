# -*- coding: utf-8 -*-
"""
Tests unitaires des deux correctifs de production discutés hors scope de
l'interface de préparation de dataset :

1. "couleur_fiable" / "nuit_probable" : une couleur de marque détectée sur
   une photo de nuit (flash IR) est signalée comme peu fiable au lieu
   d'être traitée avec la même confiance qu'en plein jour, et ne doit pas
   générer de fausse incohérence avec le tableur.
2. "marque_absente" : un canard identifié sans AUCUNE marque nasale
   détectée sur la photo (donc uniquement sur la base espèce/sexe/heure)
   est signalé explicitement, au lieu de rester une identification
   silencieusement indiscernable d'un vrai match par marque.
"""

from datetime import datetime
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
from utils.nuit import est_photo_de_nuit_bgr
from utils.pipeline import CanardIdentificationPipeline


@pytest.fixture
def mock_pipeline(sample_csv, tmp_path):
    """Crée un pipeline mocké pour les tests (identique à test_pipeline.py)."""
    model_path = tmp_path / "mock_model.pt"
    model_path.touch()

    with patch("utils.pipeline.YOLO") as mock_yolo:
        mock_yolo.return_value = MagicMock()
        pipeline = CanardIdentificationPipeline(
            tableur_path=sample_csv,
            model_detection_path=str(model_path)
        )
        return pipeline


def _burst(sample_photos_burst, median_time):
    return {
        "photos": sample_photos_burst,
        "nb_photos": len(sample_photos_burst),
        "median_time": median_time,
        "start_time": median_time,
        "end_time": median_time,
        "duration": 0.0,
    }


# ---------------------------------------------------------------------------
# est_photo_de_nuit_bgr (heuristique de saturation partagée avec
# interface_preparation_ia.py, voir utils/nuit.py)
# ---------------------------------------------------------------------------

def test_est_photo_de_nuit_bgr_detecte_image_grise():
    """Une image quasi grise (flash IR) doit être détectée comme nuit probable."""
    crop_gris = np.full((40, 40, 3), 110, dtype=np.uint8)
    assert est_photo_de_nuit_bgr(crop_gris) is True


def test_est_photo_de_nuit_bgr_ignore_image_saturee():
    """Une image bien saturée (couleur de jour) ne doit pas être signalée nuit."""
    crop_colore = np.zeros((40, 40, 3), dtype=np.uint8)
    crop_colore[:, :, 2] = 200  # rouge saturé (BGR)
    assert est_photo_de_nuit_bgr(crop_colore) is False


def test_est_photo_de_nuit_bgr_jamais_d_exception_sur_crop_invalide():
    """Un crop vide ou None ne doit jamais lever d'exception (False par défaut)."""
    assert est_photo_de_nuit_bgr(None) is False
    assert est_photo_de_nuit_bgr(np.zeros((0, 0, 3), dtype=np.uint8)) is False


# ---------------------------------------------------------------------------
# Intégration dans process_burst
# ---------------------------------------------------------------------------

def test_identification_sans_marque_est_signalee(mock_pipeline, sample_photos_burst):
    """
    Trou de sécurité corrigé : un canard identifié uniquement sur
    espèce/sexe/horodatage (aucune marque nasale détectée sur la photo)
    doit porter marque_absente=True, à la fois au niveau rafale et au
    niveau du canard, pour que ce soit visible dans le rapport/CSV/journal.
    """
    mock_pipeline.detect_canards = Mock(return_value={
        "canards": [{
            "bbox": [0, 0, 10, 10], "confiance": 0.9,
            "espece_detectee": "Colvert", "confiance_espece": 0.9,
            "sexe_detectee": None, "confiance_sexe": None,
        }],
        "marques": [],  # <- aucune marque détectée
    })

    burst = _burst(sample_photos_burst, datetime(2024, 3, 15, 14, 23, 45))
    result = mock_pipeline.process_burst(burst)

    assert result["status"] == "IDENTIFIE"
    assert result["marque_absente"] is True
    assert result["canards"][0]["marque_absente"] is True


def test_identification_avec_marque_nasale_non_signalee(mock_pipeline, sample_photos_burst):
    """À l'inverse, si une marque nasale est détectée, marque_absente doit être False."""
    mock_pipeline.detect_canards = Mock(return_value={
        "canards": [{
            "bbox": [0, 0, 10, 10], "confiance": 0.9,
            "espece_detectee": "Colvert", "confiance_espece": 0.9,
            "sexe_detectee": None, "confiance_sexe": None,
        }],
        "marques": [{
            "bbox": [1, 1, 5, 5], "confiance": 0.8,
            "couleur_detectee": "Jaune", "confiance_couleur": 0.8,
            "nuit_probable": False, "couleur_fiable": True,
        }],
    })

    burst = _burst(sample_photos_burst, datetime(2024, 3, 15, 14, 23, 45))
    result = mock_pipeline.process_burst(burst)

    assert result["status"] == "IDENTIFIE"
    assert result["marque_absente"] is False
    assert result["cohérent"]["couleur_marque"] is True  # Jaune == Jaune


def test_couleur_de_nuit_n_est_pas_comparee_au_tableur(mock_pipeline, sample_photos_burst):
    """
    Une couleur détectée sur une photo de nuit probable est peu fiable :
    elle ne doit pas être comparée au tableur (pas de fausse incohérence),
    même si elle diffère de la couleur déclarée.
    """
    mock_pipeline.detect_canards = Mock(return_value={
        "canards": [{
            "bbox": [0, 0, 10, 10], "confiance": 0.9,
            "espece_detectee": "Colvert", "confiance_espece": 0.9,
            "sexe_detectee": None, "confiance_sexe": None,
        }],
        "marques": [{
            "bbox": [1, 1, 5, 5], "confiance": 0.8,
            # couleur différente de "Jaune" (déclarée dans le tableur),
            # mais détectée de nuit -> ne doit pas déclencher d'incohérence.
            "couleur_detectee": "marron", "confiance_couleur": 0.4,
            "nuit_probable": True, "couleur_fiable": False,
        }],
    })

    burst = _burst(sample_photos_burst, datetime(2024, 3, 15, 14, 23, 45))
    result = mock_pipeline.process_burst(burst)

    assert result["couleur_fiable"] is False
    assert "couleur_marque" not in (result.get("cohérent") or {})


def test_plusieurs_canards_sans_marque_tous_signales(mock_pipeline, sample_photos_burst):
    """Dans le cas multi-canards, la marque (ou son absence) concerne la
    photo entière : chaque canard résolu doit hériter du même flag."""
    mock_pipeline.detect_canards = Mock(return_value={
        "canards": [
            {"bbox": [0, 0, 10, 10], "confiance": 0.9,
             "espece_detectee": "Colvert", "confiance_espece": 0.9,
             "sexe_detectee": None, "confiance_sexe": None},
            {"bbox": [20, 0, 10, 10], "confiance": 0.9,
             "espece_detectee": "Souchet", "confiance_espece": 0.9,
             "sexe_detectee": None, "confiance_sexe": None},
        ],
        "marques": [],
    })

    burst = _burst(sample_photos_burst, datetime(2024, 3, 15, 15, 0, 0))
    result = mock_pipeline.process_burst(burst)

    assert result["status"] in {"IDENTIFIE_MULTIPLE", "PARTIELLEMENT_IDENTIFIE", "MULTIPLE_CANARDS"}
    for canard in result["canards"]:
        assert canard["marque_absente"] is True


# ---------------------------------------------------------------------------
# Association géométrique marque <-> canard (le "trou" identifié : avant,
# la liste de marques restait globale à la rafale, sans lien avec un
# canard en particulier)
# ---------------------------------------------------------------------------

def test_assignation_geometrique_bas_niveau():
    """_assigner_marques_aux_canards doit assigner chaque marque au canard
    dont la bbox contient son centre."""
    canards = [
        {"bbox": [0, 0, 50, 50]},    # canard A, coin haut-gauche
        {"bbox": [200, 200, 50, 50]},  # canard B, loin de A
    ]
    marques = [
        {"bbox": [10, 10, 5, 5]},      # centre ~(12,12) -> dans A
        {"bbox": [210, 210, 5, 5]},    # centre ~(212,212) -> dans B
    ]
    assignations = CanardIdentificationPipeline._assigner_marques_aux_canards(canards, marques)

    assert [a["marque_index"] for a in assignations[0]] == [0]
    assert assignations[0][0]["certaine"] is True
    assert [a["marque_index"] for a in assignations[1]] == [1]
    assert assignations[1][0]["certaine"] is True


def test_assignation_geometrique_fallback_canard_le_plus_proche():
    """Une marque dont le centre n'est contenu dans AUCUNE bbox canard doit
    être assignée au canard le plus proche, avec certaine=False."""
    canards = [
        {"bbox": [0, 0, 20, 20]},      # canard proche
        {"bbox": [500, 500, 20, 20]},  # canard très loin
    ]
    marques = [{"bbox": [58, 58, 4, 4]}]  # loin des deux bbox+marge (35 max pour le canard proche)

    assignations = CanardIdentificationPipeline._assigner_marques_aux_canards(canards, marques)

    assert [a["marque_index"] for a in assignations[0]] == [0]
    assert assignations[0][0]["certaine"] is False
    assert assignations[1] == []


def test_deux_canards_deux_couleurs_distinctes_correctement_associees(mock_pipeline, sample_photos_burst):
    """
    LE FIX PRINCIPAL : avec deux canards portant chacun une marque de
    couleur différente sur la même photo, chaque canard doit récupérer
    SA PROPRE couleur (et non la couleur de l'autre, ni aucune couleur du
    tout comme avant l'association géométrique).
    """
    mock_pipeline.detect_canards = Mock(return_value={
        "canards": [
            {"bbox": [0, 0, 50, 50], "confiance": 0.9,
             "espece_detectee": "Colvert", "confiance_espece": 0.9,
             "sexe_detectee": None, "confiance_sexe": None},
            {"bbox": [300, 0, 50, 50], "confiance": 0.9,
             "espece_detectee": "Souchet", "confiance_espece": 0.9,
             "sexe_detectee": None, "confiance_sexe": None},
        ],
        "marques": [
            {"bbox": [10, 10, 6, 6], "confiance": 0.8,
             "couleur_detectee": "Jaune", "confiance_couleur": 0.9,
             "nuit_probable": False, "couleur_fiable": True},
            {"bbox": [310, 10, 6, 6], "confiance": 0.8,
             "couleur_detectee": "Vert", "confiance_couleur": 0.9,
             "nuit_probable": False, "couleur_fiable": True},
        ],
    })

    burst = _burst(sample_photos_burst, datetime(2024, 3, 16, 9, 15, 30))
    result = mock_pipeline.process_burst(burst)

    canard_colvert = next(c for c in result["canards"] if c["canard_index"] == 0)
    canard_souchet = next(c for c in result["canards"] if c["canard_index"] == 1)

    assert canard_colvert["couleur_detectee"] == "Jaune"
    assert canard_colvert["marque_absente"] is False
    assert canard_souchet["couleur_detectee"] == "Vert"
    assert canard_souchet["marque_absente"] is False


def test_marques_info_porte_le_canard_assigne(mock_pipeline, sample_photos_burst):
    """Le résultat exposé côté rafale (clé "marques") doit lui aussi
    indiquer à quel canard chaque marque a été assignée, pour la
    traçabilité dans le rapport/JSON."""
    mock_pipeline.detect_canards = Mock(return_value={
        "canards": [
            {"bbox": [0, 0, 50, 50], "confiance": 0.9,
             "espece_detectee": "Colvert", "confiance_espece": 0.9,
             "sexe_detectee": None, "confiance_sexe": None},
            {"bbox": [300, 0, 50, 50], "confiance": 0.9,
             "espece_detectee": "Souchet", "confiance_espece": 0.9,
             "sexe_detectee": None, "confiance_sexe": None},
        ],
        "marques": [
            {"bbox": [310, 10, 6, 6], "confiance": 0.8,
             "couleur_detectee": "Vert", "confiance_couleur": 0.9,
             "nuit_probable": False, "couleur_fiable": True},
        ],
    })

    burst = _burst(sample_photos_burst, datetime(2024, 3, 16, 9, 15, 30))
    result = mock_pipeline.process_burst(burst)

    assert result["marques"][0]["canard_assigne"] == 1
    assert result["marques"][0]["assignation_certaine"] is True
