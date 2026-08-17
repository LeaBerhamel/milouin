# -*- coding: utf-8 -*-
"""Extraction de crops (marques nasales ou canards) à partir des détections YOLO.

NOUVEAU : gère nativement des sous-dossiers par job CVAT sous photos_dir
(ex. job1/, job2/, job3/), comme les autres scripts du projet :
  - les photos de chaque job sont trouvées même si photos_dir/ contient des
    sous-dossiers (l'ancien comportement ne regardait qu'au premier niveau) ;
  - chaque crop est nommé avec un préfixe de job (ex. job1__RCNX0045_marque_00.jpg)
    pour éviter qu'une même photo de deux jobs différents (même nom, contenu
    différent) ne génère des crops qui s'écrasent silencieusement ;
  - metadata.json est FUSIONNÉ avec le contenu déjà présent (par crop_name) au
    lieu d'être écrasé à chaque appel, pour ne pas perdre le travail de tri
    déjà fait sur un lot précédent si on relance l'extraction plus tard sur
    de nouvelles photos avec le même --output.
S'il n'y a pas de sous-dossiers, l'ancien comportement "à plat" (un seul lot,
sans préfixe) est utilisé automatiquement.
"""

import json
from pathlib import Path

import cv2
from tqdm import tqdm
from ultralytics import YOLO

from .config import logger
from .fichiers import find_image, list_job_dirs


def list_photos(photos_dir):
    """Liste les photos .jpg/.jpeg d'un dossier (non récursif), sans doublon.

    Utilise la casse du système de fichiers plutôt que deux motifs glob
    séparés ("*.jpg" et "*.JPG") : sur un système insensible à la casse
    (Windows, Mac par défaut), les deux motifs matchent les MÊMES fichiers,
    ce qui les faisait compter et traiter deux fois."""
    photos_dir = Path(photos_dir)
    return sorted(
        p for p in photos_dir.iterdir()
        if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg")
    )


def list_labels(labels_dir):
    """Liste les fichiers de labels YOLO (.txt) d'un dossier (non récursif)."""
    labels_dir = Path(labels_dir)
    return sorted(p for p in labels_dir.iterdir() if p.is_file() and p.suffix.lower() == ".txt")


def load_existing_metadata(metadata_path: Path):
    """Charge metadata.json existant (liste), ou [] s'il n'existe pas encore."""
    if not metadata_path.exists():
        return []
    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        logger.warning("metadata.json existant illisible, il sera reconstruit : %s", metadata_path)
        return []


def extract_crops(photos_dir, model_detection_path, output_dir, target_class,
                   label_field, conf=0.5, margin=10, name_prefix="crop"):
    """
    Extrait les crops d'une classe YOLO donnée sur un ensemble de photos.

    Args:
        photos_dir: Dossier contenant les photos brutes. Peut contenir des
            sous-dossiers par job (job1/, job2/, ...) ; sinon comportement
            "à plat" classique.
        model_detection_path: Chemin du modèle YOLO de détection.
        output_dir: Dossier de sortie (un sous-dossier "crops/" y sera créé).
        target_class: ID de classe YOLO à extraire, OU liste d'IDs (ex. [1, 2, 3]
            pour regrouper plusieurs anciens libellés de classe sous un même crop).
        label_field: Nom du champ de métadonnées à remplir manuellement
            (ex. "couleur" ou "espece").
        conf: Seuil de confiance minimum pour les détections.
        margin: Marge en pixels ajoutée autour de chaque bounding box.
        name_prefix: Préfixe utilisé pour nommer les fichiers de crop.

    Returns:
        Tuple (nombre de crops extraits (ce run), chemin du fichier de métadonnées).
    """
    target_classes = {target_class} if isinstance(target_class, int) else set(target_class)

    logger.info("Chargement du modèle YOLO...")
    model = YOLO(model_detection_path)
    logger.info("Modèle chargé")

    output_dir = Path(output_dir)
    crops_dir = output_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    photos_dir = Path(photos_dir)
    job_names = list_job_dirs(photos_dir)
    multi_job = job_names != [""]
    if multi_job:
        logger.info("📁 %d job(s) détecté(s) sous %s : %s", len(job_names), photos_dir, ", ".join(job_names))

    metadata_path = output_dir / "metadata.json"
    metadata = load_existing_metadata(metadata_path)
    existing_names = {m["crop_name"] for m in metadata}

    extracted_this_run = 0

    for job in job_names:
        job_photos_dir = photos_dir / job if job else photos_dir
        photos = list_photos(job_photos_dir)
        prefix = f"{job}__" if job else ""

        label = f" (job '{job}')" if job else ""
        logger.info("Photos à traiter%s : %d", label, len(photos))

        for photo_path in tqdm(photos, desc=f"Extraction{label}"):
            results = model.predict(str(photo_path), conf=conf, verbose=False)

            img = cv2.imread(str(photo_path))
            if img is None:
                continue

            for result in results:
                boxes = result.boxes
                if boxes is None:
                    continue

                for j, box in enumerate(boxes):
                    if int(box.cls[0]) not in target_classes:
                        continue

                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    conf_score = float(box.conf[0])

                    x1 = max(0, x1 - margin)
                    y1 = max(0, y1 - margin)
                    x2 = min(img.shape[1], x2 + margin)
                    y2 = min(img.shape[0], y2 + margin)

                    crop = img[y1:y2, x1:x2]
                    crop_name = f"{prefix}{photo_path.stem}_{name_prefix}_{j:02d}.jpg"

                    if crop_name in existing_names:
                        # déjà extrait lors d'un run précédent (même photo, même job) : on
                        # ré-écrit quand même le fichier (au cas où le modèle a changé),
                        # mais on ne duplique pas son entrée metadata.
                        cv2.imwrite(str(crops_dir / crop_name), crop)
                        continue

                    cv2.imwrite(str(crops_dir / crop_name), crop)

                    metadata.append({
                        "crop_name": crop_name,
                        "source_photo": str(photo_path),
                        "bbox": [x1, y1, x2, y2],
                        "confiance": conf_score,
                        label_field: None,  # à remplir manuellement lors du triage
                    })
                    existing_names.add(crop_name)
                    extracted_this_run += 1

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    return extracted_this_run, metadata_path


def extract_crops_from_labels(photos_dir, labels_dir, output_dir, target_class,
                               label_field, margin=10, name_prefix="crop"):
    """
    Extrait les crops d'une classe donnée à partir des annotations YOLO déjà
    existantes (ground truth), plutôt que des prédictions d'un modèle.

    À utiliser pour amorcer un dataset de classification (couleur, espèce...)
    avec 100% des boîtes annotées à la main, sans dépendre du rappel du
    modèle de détection actuel (contrairement à extract_crops, qui ne peut
    récupérer que ce que le modèle détecte avec conf >= seuil : forcément
    moins que le nombre réel de boîtes annotées tant que le modèle n'est pas
    parfait).

    Args:
        photos_dir: Dossier des photos brutes (mêmes sous-dossiers par job
            que labels_dir si applicable, ex. job1/, job2/, ...).
        labels_dir: Dossier des labels YOLO ground-truth (mêmes sous-dossiers
            par job), ex. raw_data/labels_migrated/.
        output_dir, target_class, label_field, margin, name_prefix : voir
            extract_crops (même signification).

    Returns:
        Tuple (nombre de crops extraits ce run, chemin du fichier de métadonnées).
    """
    target_classes = {target_class} if isinstance(target_class, int) else set(target_class)

    output_dir = Path(output_dir)
    crops_dir = output_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    photos_dir = Path(photos_dir)
    labels_dir = Path(labels_dir)
    job_names = list_job_dirs(labels_dir)
    multi_job = job_names != [""]
    if multi_job:
        logger.info("📁 %d job(s) détecté(s) sous %s : %s", len(job_names), labels_dir, ", ".join(job_names))

    metadata_path = output_dir / "metadata.json"
    metadata = load_existing_metadata(metadata_path)
    existing_names = {m["crop_name"] for m in metadata}

    extracted_this_run = 0
    missing_images = []

    for job in job_names:
        job_labels_dir = labels_dir / job if job else labels_dir
        job_photos_dir = photos_dir / job if job else photos_dir
        labels = list_labels(job_labels_dir)
        prefix = f"{job}__" if job else ""

        label = f" (job '{job}')" if job else ""
        logger.info("Annotations à traiter%s : %d", label, len(labels))

        for label_path in tqdm(labels, desc=f"Extraction (ground truth){label}"):
            image_path = find_image(job_photos_dir, label_path.stem)
            if image_path is None:
                missing_images.append(str(label_path))
                continue

            img = cv2.imread(str(image_path))
            if img is None:
                continue
            img_h, img_w = img.shape[:2]

            lines = [line for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]

            for j, line in enumerate(lines):
                parts = line.split()
                cls_id = int(parts[0])
                if cls_id not in target_classes:
                    continue
                xc, yc, w, h = map(float, parts[1:5])

                x1 = int((xc - w / 2) * img_w) - margin
                y1 = int((yc - h / 2) * img_h) - margin
                x2 = int((xc + w / 2) * img_w) + margin
                y2 = int((yc + h / 2) * img_h) + margin
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(img_w, x2), min(img_h, y2)

                crop = img[y1:y2, x1:x2]
                crop_name = f"{prefix}{label_path.stem}_{name_prefix}_{j:02d}.jpg"

                if crop_name in existing_names:
                    # déjà extrait lors d'un run précédent : on ré-écrit quand même
                    # le fichier (photo/annotation potentiellement mise à jour),
                    # mais on ne duplique pas son entrée metadata.
                    cv2.imwrite(str(crops_dir / crop_name), crop)
                    continue

                cv2.imwrite(str(crops_dir / crop_name), crop)
                metadata.append({
                    "crop_name": crop_name,
                    "source_photo": str(image_path),
                    "bbox": [x1, y1, x2, y2],
                    "confiance": None,  # ground truth : pas de score de confiance
                    label_field: None,  # à remplir manuellement lors du triage
                })
                existing_names.add(crop_name)
                extracted_this_run += 1

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    if missing_images:
        logger.warning("%d label(s) sans image correspondante (crops non générés) : %s",
                        len(missing_images), ", ".join(missing_images[:5]) +
                        (", ..." if len(missing_images) > 5 else ""))

    return extracted_this_run, metadata_path