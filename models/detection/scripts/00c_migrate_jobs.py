#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Migre les jobs CVAT sous raw_data/images/<job>/ + raw_data/labels/<job>/
qui utilisent encore l'ancien schéma à 5 classes, en appelant
00_migrate_annotations.py une fois par job concerné (schéma auto-détecté).

Un job est considéré "ancien schéma" si un de ses .txt contient un indice
de classe > 1 (le nouveau schéma n'utilise que 0: marque_nasale, 1: canard).
Les jobs déjà au nouveau schéma sont ignorés (rien à migrer).

À placer dans le dossier scripts/ du dépôt (à côté de
00_migrate_annotations.py), puis lancer depuis la racine du dépôt :

    python scripts/00c_migrate_jobs.py

Options utiles :
    --dry-run            n'affiche que le diagnostic par job, ne migre rien
    --classes-file FILE   même fichier de classes pour tous les jobs à migrer
                          (sinon l'ordre par défaut du projet est utilisé)

Après exécution, les labels migrés de chaque job se trouvent dans :
    models/detection/raw_data/labels_migrated/<job>/

Étape suivante (MANUELLE, volontairement pas automatique) : vérifier ces
labels (ex. avec 00b_check_annotations.py --images raw_data/images/<job>
--labels raw_data/labels_migrated/<job>) puis les copier toi-même dans
raw_data/labels/<job>/, en écrasant l'ancien schéma.
"""

import argparse
import subprocess
import sys
from pathlib import Path

# scripts/00c_migrate_jobs.py -> ROOT = models/detection
ROOT = Path(__file__).resolve().parent.parent
RAW_IMAGES = ROOT / "raw_data" / "images"
RAW_LABELS = ROOT / "raw_data" / "labels"
MIGRATE_SCRIPT = ROOT.parent.parent / "scripts" / "00_migrate_annotations.py"


def list_job_dirs(base: Path) -> list[str]:
    if not base.exists():
        return []
    subdirs = sorted(d.name for d in base.iterdir() if d.is_dir())
    return subdirs


def detect_schema(labels_dir: Path) -> tuple[str, int]:
    """Retourne ('ancien'|'nouveau'|'vide', indice_max_trouve)."""
    max_id = -1
    n_files = 0
    for txt in labels_dir.glob("*.txt"):
        n_files += 1
        for line in txt.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            try:
                cls_id = int(parts[0])
            except (ValueError, IndexError):
                continue
            max_id = max(max_id, cls_id)
    if n_files == 0:
        return "vide", max_id
    return ("ancien" if max_id > 1 else "nouveau"), max_id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                         help="N'affiche que le diagnostic, ne migre rien.")
    parser.add_argument("--classes-file", default=None,
                         help="Fichier de classes (ordre ancien schéma) à utiliser "
                              "pour TOUS les jobs migrés. Sinon ordre par défaut du projet.")
    args = parser.parse_args()

    if not MIGRATE_SCRIPT.exists():
        print(f"❌ Introuvable : {MIGRATE_SCRIPT}\n"
              f"   Attendu à la racine du dépôt sous scripts/00_migrate_annotations.py.")
        sys.exit(1)

    jobs = list_job_dirs(RAW_IMAGES)
    if not jobs:
        print(f"❌ Aucun sous-dossier trouvé sous {RAW_IMAGES}. "
              f"Ce script suppose une organisation par job (job1/, job2/, ...).")
        sys.exit(1)

    print(f"📁 {len(jobs)} job(s) trouvé(s) : {', '.join(jobs)}\n")

    to_migrate = []
    for job in jobs:
        labels_dir = RAW_LABELS / job
        if not labels_dir.exists():
            print(f"  ⚠️  [{job}] pas de dossier labels correspondant ({labels_dir}) — ignoré.")
            continue
        schema, max_id = detect_schema(labels_dir)
        if schema == "vide":
            print(f"  ⚠️  [{job}] aucun .txt trouvé — ignoré.")
        elif schema == "nouveau":
            print(f"  ✅ [{job}] déjà au nouveau schéma (indice max = {max_id}) — rien à faire.")
        else:
            print(f"  🔄 [{job}] ancien schéma détecté (indice max = {max_id}) — à migrer.")
            to_migrate.append(job)

    if not to_migrate:
        print("\nAucun job à migrer.")
        return

    if args.dry_run:
        print(f"\n(dry-run) Jobs qui seraient migrés : {', '.join(to_migrate)}")
        return

    print(f"\n{len(to_migrate)} job(s) à migrer : {', '.join(to_migrate)}\n")

    for job in to_migrate:
        out_labels = RAW_LABELS.parent / "labels_migrated" / job
        # IMPORTANT : dossiers PARTAGÉS (pas de sous-dossier par job), pour que
        # les crops de tous les jobs (migrés maintenant ou plus tard) vivent au
        # même endroit que ceux d'une migration combinée. --job-name force le
        # préfixe de nom de fichier (ex. job1__RCNX0045_g00_3.jpg) pour éviter
        # toute collision entre jobs, sans dupliquer les crops dans deux
        # structures différentes comme précédemment.
        out_espece = ROOT.parent.parent / "dataset_espece_migrated"
        out_sexe = ROOT.parent.parent / "dataset_sexe_migrated"

        cmd = [
            sys.executable, str(MIGRATE_SCRIPT),
            "--raw-images", str(RAW_IMAGES / job),
            "--raw-labels", str(RAW_LABELS / job),
            "--output-labels", str(out_labels),
            "--output-espece", str(out_espece),
            "--output-sexe", str(out_sexe),
            "--job-name", job,
        ]
        if args.classes_file:
            cmd += ["--classes-file", args.classes_file]

        print(f"{'=' * 60}\n▶️  Migration du job '{job}'\n{'=' * 60}")
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"❌ Échec de la migration pour '{job}' (code {result.returncode}).")
            continue
        print(f"✅ [{job}] labels migrés -> {out_labels}")

    print("\n" + "=" * 60)
    print("PROCHAINES ÉTAPES (manuelles, volontairement pas automatiques) :")
    print("=" * 60)
    for job in to_migrate:
        print(f"\n  Job '{job}' :")
        print("    1. Vérifier visuellement :")
        print("       python scripts/00b_check_annotations.py \\")
        print(f"           --images raw_data/images/{job} \\")
        print(f"           --labels raw_data/labels_migrated/{job} \\")
        print(f"           --out check_annotations_output/{job}")
        print("    2. Si ça a l'air bon, écraser l'ancien schéma :")
        print("       (depuis models/detection/) ")
        print(f"       cp raw_data/labels_migrated/{job}/*.txt raw_data/labels/{job}/")
    print("\nUne fois tous les jobs vérifiés et copiés, lance :")
    print("  python scripts/01_split_dataset.py --val-ratio 0.2 --seed 42")


if __name__ == "__main__":
    main()