"""
Répartit tes images annotées (+ fichiers .txt YOLO associés) entre
train/ et val/, dans la structure attendue par Ultralytics.

Avant de lancer ce script, organise raw_data/ PAR JOB CVAT (recommandé,
évite les collisions de noms quand plusieurs pièges photo réutilisent
les mêmes numéros de fichier, ex. RCNX0045.JPG dans 2 jobs différents) :

    raw_data/
      images/
        job1/   RCNX0045.JPG, RCNX0046.JPG, ...
        job2/   RCNX0045.JPG, ...   <- même nom, autre dossier = pas de conflit
        job3/   ...
      labels/
        job1/   RCNX0045.txt, ...
        job2/   ...
        job3/   ...

  - même nom de base entre image et label DANS UN MÊME JOB : ex.
    raw_data/images/job1/duck_001.jpg + raw_data/labels/job1/duck_001.txt
  - si tu pars d'annotations à l'ancien schéma (5 classes), lance d'abord
    00_migrate_annotations.py et copie son résultat (raw_data/labels_migrated/)
    dans raw_data/labels/<job>/ avant de lancer ce script.

Rétrocompatibilité : si raw_data/images/ ne contient AUCUN sous-dossier
(que des fichiers images directement dedans), l'ancien comportement "à plat"
est utilisé automatiquement, sans rien à changer.

Usage :
    python scripts/01_split_dataset.py --val-ratio 0.2 --seed 42

CORRECTIF (voir conversation) :
  - DATASET pointe désormais vers <repo>/dataset (le dossier réellement
    utilisé par le notebook d'entraînement, /content/project/dataset/data.yaml
    sur Colab), et non plus vers models/detection/dataset comme avant.
  - Ajout d'une détection de doublons de contenu binaire (même image présente
    deux fois sous des noms différents, ex. "IMG (2).JPG"), pour éviter
    qu'une même photo se retrouve à la fois en train et en val (fuite de
    données qui fausse les métriques de validation).
  - NOUVEAU : support des sous-dossiers "job1/", "job2/", "job3/" sous
    raw_data/images/ et raw_data/labels/, pour distinguer plusieurs lots
    CVAT sans collision de noms de fichiers. Chaque paire est désormais
    identifiée par (job, nom_fichier) et non plus seulement par nom_fichier ;
    les fichiers copiés dans dataset/ sont préfixés par le nom du job
    (ex. job1__RCNX0045.jpg) pour rester uniques une fois tous les jobs
    fusionnés dans le même dossier train/ ou val/.
"""

import argparse
import hashlib
import random
import shutil
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

# models/detection/scripts/01_split_dataset.py -> ROOT = models/detection
ROOT = Path(__file__).resolve().parent.parent
RAW_IMAGES = ROOT / "raw_data" / "images"
RAW_LABELS = ROOT / "raw_data" / "labels"

# <repo>/models/detection -> REPO_ROOT = <repo>  (= "canard-detection")
# C'est ce dossier/dataset que le notebook utilise réellement
# (/content/project/dataset/data.yaml sur Colab).
REPO_ROOT = ROOT.parent.parent
DATASET = REPO_ROOT / "dataset"


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def list_job_dirs(base: Path) -> list[str]:
    """Retourne les noms de sous-dossiers "job" trouvés sous base/, ou
    [""] (job unique = comportement à plat) s'il n'y en a aucun."""
    subdirs = sorted(d.name for d in base.iterdir() if d.is_dir())
    return subdirs if subdirs else [""]


def find_pairs():
    """Trouve les paires (job, image, label) valides sous raw_data/,
    en dédoublonnant par contenu binaire IDENTIQUE À L'INTÉRIEUR D'UN
    MÊME JOB (deux jobs différents peuvent légitimement contenir des
    photos différentes qui partagent le même nom de fichier)."""
    job_names = list_job_dirs(RAW_IMAGES)
    multi_job = job_names != [""]

    if multi_job:
        print(f"📁 {len(job_names)} job(s) détecté(s) sous raw_data/images/ : "
              f"{', '.join(job_names)}")
    else:
        print("📁 Pas de sous-dossiers détectés : mode à plat (1 seul lot).")

    all_pairs = []
    for job in job_names:
        images_dir = RAW_IMAGES / job if job else RAW_IMAGES
        labels_dir = RAW_LABELS / job if job else RAW_LABELS

        if not images_dir.exists():
            print(f"  ⚠️  Dossier images manquant pour le job '{job}' : {images_dir}")
            continue
        if not labels_dir.exists():
            print(f"  ⚠️  Dossier labels manquant pour le job '{job}' : {labels_dir}")
            continue

        candidates = sorted(
            p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS
        )

        by_hash = {}
        for p in candidates:
            by_hash.setdefault(md5(p), []).append(p)

        def is_messy_name(p: Path) -> bool:
            name = p.stem
            return "(2)" in name or "copie" in name.lower()

        canonical = []
        n_dupes = 0
        for _h, paths in by_hash.items():
            if len(paths) > 1:
                n_dupes += len(paths) - 1
                clean = [p for p in paths if not is_messy_name(p)]
                canonical.append(clean[0] if clean else paths[0])
            else:
                canonical.append(paths[0])

        if n_dupes:
            print(f"  ⚠️  [{job or 'lot unique'}] {n_dupes} image(s) dupliquée(s) "
                  f"(même contenu binaire, nom différent) ignorée(s).")

        job_pairs = []
        missing_labels = []
        for img_path in sorted(canonical):
            label_path = labels_dir / f"{img_path.stem}.txt"
            if label_path.exists():
                job_pairs.append((job, img_path, label_path))
            else:
                missing_labels.append(img_path.name)

        if missing_labels:
            print(f"  ⚠️  [{job or 'lot unique'}] {len(missing_labels)} image(s) "
                  f"sans fichier .txt associé (ignorées) :")
            for name in missing_labels[:10]:
                print(f"       - {name}")
            if len(missing_labels) > 10:
                print(f"       ... et {len(missing_labels) - 10} autres")

        print(f"  ✅ [{job or 'lot unique'}] {len(job_pairs)} paire(s) valide(s).")
        all_pairs.extend(job_pairs)

    # Dédoublonnage global inter-jobs : si la MÊME photo (contenu identique)
    # a été annotée deux fois dans deux jobs différents, on ne la garde
    # qu'une fois pour éviter une fuite train/val.
    by_hash_global = {}
    for job, img_path, label_path in all_pairs:
        by_hash_global.setdefault(md5(img_path), []).append((job, img_path, label_path))

    deduped = []
    n_cross_job_dupes = 0
    for _h, entries in by_hash_global.items():
        if len(entries) > 1:
            n_cross_job_dupes += len(entries) - 1
        deduped.append(entries[0])

    if n_cross_job_dupes:
        print(f"\n⚠️  {n_cross_job_dupes} image(s) identique(s) trouvée(s) dans "
              f"plusieurs jobs à la fois (même photo annotée deux fois) : "
              f"une seule copie conservée.")

    return deduped


def clear_dir(d: Path):
    for f in d.glob("*"):
        if f.is_file():
            f.unlink()


def output_name(job: str, path: Path) -> str:
    """Nom de fichier final dans dataset/, préfixé par le job pour éviter
    toute collision une fois tous les jobs fusionnés dans train/ ou val/."""
    return f"{job}__{path.name}" if job else path.name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-ratio", type=float, default=0.2,
                         help="Proportion des données pour la validation (défaut : 0.2)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--move", action="store_true",
                         help="Déplacer les fichiers au lieu de les copier (défaut : copie, "
                              "ne touche pas à raw_data/)")
    args = parser.parse_args()

    if not RAW_IMAGES.exists() or not RAW_LABELS.exists():
        print(f"❌ Dossiers introuvables. Attendu :\n  {RAW_IMAGES}\n  {RAW_LABELS}")
        return

    pairs = find_pairs()
    n = len(pairs)
    if n == 0:
        print("❌ Aucune paire image/label trouvée dans raw_data/. "
              "Vérifie que les .txt et images ont bien le même nom de base "
              "(et sont dans le même sous-dossier de job, si tu en utilises).")
        return

    print(f"\n{n} paire(s) image/label trouvée(s) au total.")
    if n < 150:
        print(f"⚠️  Rappel : 150-200 images minimum recommandées "
              f"(500-800 pour une bonne précision). Tu en as {n}.")

    random.seed(args.seed)
    random.shuffle(pairs)

    n_val = max(1, round(n * args.val_ratio))
    val_pairs = pairs[:n_val]
    train_pairs = pairs[n_val:]

    # Nettoyage des dossiers de destination pour éviter les mélanges
    for sub in ["images/train", "images/val", "labels/train", "labels/val"]:
        d = DATASET / sub
        d.mkdir(parents=True, exist_ok=True)
        clear_dir(d)

    op = shutil.move if args.move else shutil.copy2
    op_name = "déplacées" if args.move else "copiées"

    for job, img_path, label_path in train_pairs:
        op(str(img_path), str(DATASET / "images" / "train" / output_name(job, img_path)))
        op(str(label_path), str(DATASET / "labels" / "train" / output_name(job, label_path)))

    for job, img_path, label_path in val_pairs:
        op(str(img_path), str(DATASET / "images" / "val" / output_name(job, img_path)))
        op(str(label_path), str(DATASET / "labels" / "val" / output_name(job, label_path)))

    print(f"\n✅ Terminé : {len(train_pairs)} en train, {len(val_pairs)} en val "
          f"({op_name}).")
    print(f"   -> {DATASET / 'images' / 'train'}")
    print(f"   -> {DATASET / 'images' / 'val'}")


if __name__ == "__main__":
    main()