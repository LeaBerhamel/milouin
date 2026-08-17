"""
Vérifie visuellement que les annotations YOLO sont correctes en dessinant
les boîtes par-dessus les images d'origine.

Ne modifie rien : sauvegarde des copies annotées dans un dossier à part
pour que tu puisses les regarder.

Usage :
    python scripts/00b_check_annotations.py

Par défaut, ça vérifie raw_data/ (avant split). Pour vérifier après le
split :
    python scripts/00b_check_annotations.py --images dataset/images/train --labels dataset/labels/train

NOUVEAU : supporte des sous-dossiers par job CVAT sous --images / --labels
(ex. raw_data/images/job1/, job2/, job3/), pour distinguer plusieurs lots
qui peuvent contenir des images de même nom (pièges photo qui réutilisent
la même numérotation d'un job à l'autre). S'il n'y a pas de sous-dossiers,
l'ancien comportement "à plat" est utilisé automatiquement.
"""

import argparse
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def draw_boxes(img_path: Path, label_path: Path, out_path: Path):
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"  ⚠️  Impossible de lire l'image : {img_path.name}")
        return False

    h, w = img.shape[:2]

    if not label_path.exists():
        print(f"  ⚠️  Pas de label pour : {img_path.name}")
        return False

    lines = [
        ligne_txt.strip()
        for ligne_txt in label_path.read_text().splitlines()
        if ligne_txt.strip()
    ]
    if not lines:
        print(f"  ⚠️  Fichier label vide : {label_path.name}")
        return False

    for line in lines:
        parts = line.split()
        if len(parts) != 5:
            print(f"  ❌ Ligne mal formée dans {label_path.name} : '{line}'")
            continue
        cls, xc, yc, bw, bh = parts
        xc, yc, bw, bh = float(xc), float(yc), float(bw), float(bh)

        # Vérification des valeurs (doivent être normalisées entre 0 et 1)
        if not (0 <= xc <= 1 and 0 <= yc <= 1 and 0 < bw <= 1 and 0 < bh <= 1):
            print(f"  ❌ Coordonnées hors norme dans {label_path.name} : '{line}' "
                  f"(attendu : valeurs entre 0 et 1)")
            continue

        # conversion en pixels
        x1 = int((xc - bw / 2) * w)
        y1 = int((yc - bh / 2) * h)
        x2 = int((xc + bw / 2) * w)
        y2 = int((yc + bh / 2) * h)

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), max(2, w // 400))
        cv2.putText(img, f"classe {cls}", (x1, max(0, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, w / 1200, (0, 255, 0), 2)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)
    return True


def list_job_dirs(base: Path) -> list[str]:
    """Retourne les noms de sous-dossiers "job" trouvés sous base/, ou
    [""] (job unique = comportement à plat) s'il n'y en a aucun."""
    subdirs = sorted(d.name for d in base.iterdir() if d.is_dir())
    return subdirs if subdirs else [""]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", default="raw_data/images")
    parser.add_argument("--labels", default="raw_data/labels")
    parser.add_argument("--out", default="check_annotations_output")
    parser.add_argument("--limit", type=int, default=0,
                         help="Limiter le nombre d'images traitées (0 = toutes, "
                              "par job si plusieurs jobs)")
    args = parser.parse_args()

    images_base = ROOT / args.images
    labels_base = ROOT / args.labels
    out_dir = ROOT / args.out

    if not images_base.exists():
        print(f"❌ Dossier introuvable : {images_base}")
        return

    job_names = list_job_dirs(images_base)
    multi_job = job_names != [""]
    if multi_job:
        print(f"📁 {len(job_names)} job(s) détecté(s) : {', '.join(job_names)}\n")

    total_ok = 0
    total_count = 0

    for job in job_names:
        images_dir = images_base / job if job else images_base
        labels_dir = labels_base / job if job else labels_base

        if not images_dir.exists():
            print(f"⚠️  Dossier images manquant pour le job '{job}' : {images_dir}")
            continue

        img_paths = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        if args.limit:
            img_paths = img_paths[: args.limit]

        if not img_paths:
            print(f"⚠️  Aucune image trouvée dans {images_dir}")
            continue

        label = f" (job '{job}')" if job else ""
        print(f"Vérification de {len(img_paths)} image(s){label}...\n")

        ok_count = 0
        for img_path in img_paths:
            label_path = labels_dir / f"{img_path.stem}.txt"
            # Sortie namée par job pour éviter d'écraser les aperçus entre jobs
            job_out_dir = out_dir / job if job else out_dir
            out_path = job_out_dir / img_path.name
            if draw_boxes(img_path, label_path, out_path):
                ok_count += 1

        print(f"\n✅ [{job or 'lot unique'}] {ok_count}/{len(img_paths)} "
              f"images annotées avec succès.\n")
        total_ok += ok_count
        total_count += len(img_paths)

    print(f"\n{'=' * 50}")
    print(f"TOTAL : {total_ok}/{total_count} images annotées avec succès.")
    print(f"Regarde le résultat dans : {out_dir}")
    print("-> vérifie que chaque boîte verte entoure bien la marque nasale.")


if __name__ == "__main__":
    main()