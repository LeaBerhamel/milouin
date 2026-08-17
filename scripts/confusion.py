"""
Petits utilitaires partagés (recherche de fichiers, géométrie de bbox) utilisés
par plusieurs scripts de préparation de dataset (00_prepare_dataset_couleur.py,
00_prepare_dataset_espece_sexe.py, 00_migrate_annotations.py) et par
utils/crop_extraction.py.

Ces fonctions étaient auparavant recopiées (avec de légères divergences, ex.
gestion du .png inconsistante) dans chacun de ces fichiers. Centralisées ici
pour éviter que les copies ne divergent silencieusement.

Note : models/detection/scripts/ a ses propres copies de list_job_dirs/iou et
n'importe volontairement pas ce module — ce dossier est conçu pour rester
autonome (aucune dépendance vers scripts/utils/), au cas où il serait déplacé
ou réutilisé isolément pour ré-entraîner YOLO ailleurs.
"""
from pathlib import Path

# Extensions d'image reconnues, dans cet ordre de préférence.
IMAGE_EXTS = [".jpg", ".JPG", ".jpeg", ".JPEG", ".png", ".PNG"]


def list_job_dirs(base: Path) -> list:
    """Retourne les noms de sous-dossiers "job" trouvés sous base/, ou
    [""] (job unique = comportement à plat) s'il n'y en a aucun."""
    subdirs = sorted(d.name for d in base.iterdir() if d.is_dir())
    return subdirs if subdirs else [""]


def find_image(images_dir: Path, stem: str):
    """Trouve le fichier image correspondant à un nom de base (label sans
    extension), quelle que soit la casse/l'extension utilisée (.jpg/.JPG/
    .jpeg/.JPEG/.png/.PNG). Retourne None si aucune ne correspond."""
    images_dir = Path(images_dir)
    for ext in IMAGE_EXTS:
        candidate = images_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def iou(box_a, box_b) -> float:
    """Intersection-over-union de deux boîtes pixel (x1, y1, x2, y2)."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def union_box(box_a, box_b):
    """Union pixel de deux boîtes (x1, y1, x2, y2)."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    return min(ax1, bx1), min(ay1, by1), max(ax2, bx2), max(ay2, by2)
