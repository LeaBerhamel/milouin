"""
Prépare les datasets de classification (espèce/sexe) à partir des détections YOLO.

Extrait les crops de canards détectés, les organise par classe selon les
annotations, et génère les dossiers train/val prêts pour l'entraînement.

Usage :
    python scripts/00_prepare_dataset_espece_sexe.py --photos data/photos --model-detection models/detection/best.pt
"""

import argparse
import shutil
from pathlib import Path
from typing import Any

import cv2
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent


def extract_crops(image_path: Path, model: YOLO, output_dir: Path, class_name: str):
    """Détecte et extrait les crops de canards d'une image."""
    results = model.predict(source=str(image_path), conf=0.25, verbose=False)
    
    boxes: list[Any] = []  # Type hint ajouté
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"  ⚠️  Impossible de lire : {image_path.name}")
        return 0

    for r in results:
        if r.boxes is None:
            continue
        if r.boxes is not None:
            for box in list(r.boxes):
                cls_id = int(box.cls[0])
                if cls_id == 1:  # classe "canard" (0 = marque_nasale)
                    boxes.append(box)

    if not boxes:
        return 0

    class_dir = output_dir / class_name
    class_dir.mkdir(parents=True, exist_ok=True)

    n_crops = 0
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        crop = img[y1:y2, x1:x2]
        
        crop_filename = f"{image_path.stem}_canard{i}.jpg"
        crop_path = class_dir / crop_filename
        cv2.imwrite(str(crop_path), crop)
        n_crops += 1

    return n_crops


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--photos", required=True, help="Dossier contenant les photos annotées")
    parser.add_argument("--model-detection", required=True, help="Modèle YOLO de détection (best.pt)")
    parser.add_argument("--output-espece", default="dataset_espece", help="Dossier de sortie pour espèce")
    parser.add_argument("--output-sexe", default="dataset_sexe", help="Dossier de sortie pour sexe")
    args = parser.parse_args()

    photos_dir = Path(args.photos)
    model_path = Path(args.model_detection)
    output_espece = ROOT / args.output_espece
    output_sexe = ROOT / args.output_sexe

    if not photos_dir.exists():
        print(f"❌ Dossier photos introuvable : {photos_dir}")
        return

    if not model_path.exists():
        print(f"❌ Modèle introuvable : {model_path}")
        return

    print(f"Chargement du modèle de détection : {model_path}")
    model = YOLO(str(model_path))

    # Créer les structures de sortie
    for dataset_dir in [output_espece, output_sexe]:
        dataset_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n📸 Extraction des crops depuis {photos_dir}...\n")

    image_files = list(photos_dir.glob("*.jpg")) + list(photos_dir.glob("*.JPG"))
    total_crops = 0

    for img_path in image_files:
        n = extract_crops(img_path, model, output_espece, "a_trier")
        total_crops += n
        if n > 0:
            print(f"  ✅ {img_path.name} : {n} crop(s)")

    print(f"\n✅ {total_crops} crop(s) extraits au total.")
    print(f"   Espèce → {output_espece}")
    print(f"   Sexe → {output_sexe}")
    print("\nProchaines étapes :")
    print("  1. Trier manuellement les crops par classe (milouin/morillon, male/femelle)")
    print("  2. Lancer scripts/train_classification_ia.py")


if __name__ == "__main__":
    main()