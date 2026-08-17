# -*- coding: utf-8 -*-
"""Détection et groupement des rafales de photos selon leur horodatage EXIF."""

from datetime import datetime

from PIL import Image
from tqdm import tqdm

from .config import logger

# Tags EXIF pouvant contenir la date de prise de vue, par ordre de préférence
EXIF_DATETIME_TAGS = (36867, 36868, 306)  # DateTimeOriginal, DateTimeDigitized, DateTime
EXIF_DATETIME_FORMAT = "%Y:%m:%d %H:%M:%S"


class BurstDetector:
    """Groupe les photos en rafales selon leur timestamp EXIF."""

    def group_photos_into_bursts(self, photo_paths, max_gap_seconds=5):
        """
        Groupe les photos en rafales selon leur timestamp EXIF.

        Args:
            photo_paths: Liste de chemins de photos.
            max_gap_seconds: Écart maximum entre deux photos d'une même rafale (secondes).

        Returns:
            Liste de dictionnaires décrivant chaque rafale.
        """
        photos_data = []

        logger.info("Extraction des timestamps EXIF...")
        for photo_path in tqdm(photo_paths, desc="Lecture EXIF"):
            dt = self.extract_datetime(photo_path)
            if dt:
                photos_data.append({"path": photo_path, "datetime": dt})
            else:
                logger.warning("Pas de timestamp EXIF : %s", photo_path)

        if not photos_data:
            logger.error("Aucune photo avec timestamp EXIF trouvée")
            return []

        photos_data.sort(key=lambda x: x["datetime"])

        rafales = []
        current_burst = []

        for photo in photos_data:
            if not current_burst:
                current_burst.append(photo)
                continue

            gap = (photo["datetime"] - current_burst[-1]["datetime"]).total_seconds()
            if gap <= max_gap_seconds:
                current_burst.append(photo)
            else:
                rafales.append(self._build_burst_info(current_burst))
                current_burst = [photo]

        if current_burst:
            rafales.append(self._build_burst_info(current_burst))

        return rafales

    def _build_burst_info(self, photos):
        """Construit les infos d'une rafale à partir de ses photos."""
        timestamps = [p["datetime"] for p in photos]

        return {
            "photos": [p["path"] for p in photos],
            "timestamps": timestamps,
            "median_time": timestamps[len(timestamps) // 2],
            "start_time": min(timestamps),
            "end_time": max(timestamps),
            "duration": (max(timestamps) - min(timestamps)).total_seconds(),
            "nb_photos": len(photos),
        }

    def extract_datetime(self, photo_path):
        """
        Extrait la date/heure EXIF d'une photo.

        Returns:
            Un objet datetime, ou None si absent ou illisible.
        """
        try:
            img = Image.open(photo_path)
            exif_data = img._getexif()
            if not exif_data:
                return None

            date_str = next(
                (exif_data[tag] for tag in EXIF_DATETIME_TAGS if tag in exif_data),
                None,
            )
            if date_str:
                return datetime.strptime(date_str, EXIF_DATETIME_FORMAT)

            return None
        except Exception:
            logger.debug("Impossible de lire l'EXIF de la photo", exc_info=True)
            return None
