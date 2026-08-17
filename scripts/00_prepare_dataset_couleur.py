#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extraction 100% ground-truth des crops marque_nasale, à partir des
labels YOLO déjà migrés/corrigés (raw_data/labels/<job>/*.txt, classe 0).
Remplace le dataset_couleur existant (pollué par des crops issus du
modèle) par une base propre : 1 crop par boîte annotée à la main,
zéro faux positif, zéro doublon.
"""
import json
from pathlib import Path

from PIL import Image
from utils.fichiers import find_image

# scripts/00_prepare_dataset_couleur.py -> REPO_ROOT = <repo>
REPO_ROOT = Path(__file__).resolve().parent.parent
RAW = REPO_ROOT / "models" / "detection" / "raw_data"
IMAGES = RAW / "images"
LABELS = RAW / "labels"
OUT = REPO_ROOT / "dataset_couleur"
OUT_CROPS = OUT / "crops"
OUT_CROPS.mkdir(parents=True, exist_ok=True)

MARGIN = 0.15  # marge autour de la boîte (15% de sa taille), utile pour le tri couleur


metadata = []
n_boxes = 0
n_images_with_mark = 0
skipped = []

for job in sorted(d.name for d in IMAGES.iterdir() if d.is_dir()):
    img_job_dir = IMAGES / job
    lbl_job_dir = LABELS / job
    if not lbl_job_dir.exists():
        continue
    for txt in sorted(lbl_job_dir.glob("*.txt")):
        lines = [
            ligne_txt.strip()
            for ligne_txt in txt.read_text(encoding="utf-8", errors="ignore").splitlines()
            if ligne_txt.strip()
        ]
        marque_lines = []
        for line in lines:
            parts = line.split()
            if parts and parts[0] == "0":
                marque_lines.append(parts)
        if not marque_lines:
            continue
        img_path = find_image(img_job_dir, txt.stem)
        if img_path is None:
            skipped.append(f"{job}/{txt.stem}")
            continue
        n_images_with_mark += 1
        with Image.open(img_path) as im:
            W, H = im.size
            for idx, parts in enumerate(marque_lines):
                cx, cy, w, h = (float(x) for x in parts[1:5])
                # bbox pixels
                x1 = (cx - w / 2) * W
                y1 = (cy - h / 2) * H
                x2 = (cx + w / 2) * W
                y2 = (cy + h / 2) * H
                # marge
                mw = (x2 - x1) * MARGIN
                mh = (y2 - y1) * MARGIN
                x1 = max(0, int(x1 - mw))
                y1 = max(0, int(y1 - mh))
                x2 = min(W, int(x2 + mw))
                y2 = min(H, int(y2 + mh))
                crop = im.crop((x1, y1, x2, y2))
                crop_name = f"{job}__{txt.stem}_marque_{idx:02d}.jpg"
                crop.save(OUT_CROPS / crop_name, quality=95)
                n_boxes += 1
                metadata.append({
                    "crop_name": crop_name,
                    "source_photo": f"raw_data/images/{job}/{img_path.name}",
                    "bbox": [x1, y1, x2, y2],
                    "confiance": None,
                    "couleur": None,
                })

with open(OUT / "metadata.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, ensure_ascii=False, indent=2)

print(f"Images avec au moins 1 marque : {n_images_with_mark}")
print(f"Crops marque_nasale extraits (ground truth) : {n_boxes}")
if skipped:
    print(f"⚠️ Labels sans image correspondante ({len(skipped)}): {skipped[:10]}")