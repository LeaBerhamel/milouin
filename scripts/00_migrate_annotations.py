#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Migre les annotations CVAT/YOLO de l'ancien schéma (5 classes) vers le
nouveau schéma à 2 classes de détection + attributs classifiés en aval.

Ancien schéma (déprécié, voir ANNOTATION_GUIDE.md) :
    0: marque_nasale
    1: canard_male
    2: canard_femelle
    3: milouin
    4: morillon
(ordre par défaut ; utiliser --classes-file pour un export CVAT avec un
ordre différent — un fichier texte listant un nom de classe par ligne,
dans l'ordre des indices YOLO.)

Ce script produit, à partir des mêmes fichiers .txt YOLO d'origine :

1. Un jeu de labels de DÉTECTION à 2 classes (0: marque_nasale, 1: canard),
   dans --output-labels — prêt à remplacer raw_data/labels/ avant de lancer
   01_split_dataset.py.
2. Un dataset de classification ESPÈCE (crops + dossiers milouin/morillon),
   dans --output-espece — uniquement pour les boîtes annotées à l'époque
   avec une classe d'espèce explicite (milouin/morillon).
3. Un dataset de classification SEXE (crops + dossiers male/femelle),
   dans --output-sexe — uniquement pour les boîtes annotées à l'époque avec
   une classe de sexe explicite (canard_male/canard_femelle).

Important : l'ancien schéma ne permettait pas d'annoter espèce ET sexe sur
la même boîte. Certains lots contiennent donc, pour un même canard, DEUX
boîtes qui se chevauchent (une boîte canard_male/canard_femelle pour le
sexe + une boîte milouin/morillon pour l'espèce). Ce script les détecte
(par recouvrement IoU, voir --merge-iou) et les fusionne en une seule boîte
de détection "canard" (l'union des deux rectangles), à partir de laquelle
un seul crop est extrait et utilisé à la fois pour le dataset espèce et le
dataset sexe. Le triage manuel reste nécessaire pour compléter les classes
sous-représentées, et toute nouvelle annotation doit suivre le nouveau
schéma (ANNOTATION_GUIDE.md) qui permet de renseigner les deux directement
sur une seule boîte.

Nouvelles espèces (--classes-file) : les 5 noms ci-dessus sont reconnus
explicitement, mais un export CVAT peut lister d'autres noms (ex. une
sixième classe "foulque_macroule" ajoutée depuis). Tout nom qui n'est ni
marque_nasale ni une variante mâle/femelle est traité automatiquement comme
une nouvelle espèce (crop -> --output-espece/<nom>/), au lieu d'être
ignoré : la boîte de détection "canard" est conservée dans tous les cas, et
les nouvelles espèces rencontrées sont listées en fin d'exécution pour
vérification. Les noms mal orthographiés ne sont donc plus rejetés
silencieusement mais créent chacun leur propre dossier — d'où l'intérêt de
vérifier la liste affichée en fin de migration.

NOUVEAU : supporte des sous-dossiers par job CVAT sous raw_data/images/ et
raw_data/labels/ (ex. job1/, job2/, job3/), comme 00b_check_annotations.py
et 01_split_dataset.py. Chaque job est migré indépendamment :
  - les labels migrés gardent la même structure par job
    (--output-labels/job1/, job2/, job3/), prête à être copiée directement
    dans raw_data/labels/ pour remplacer l'ancien schéma ;
  - les crops espèce/sexe sont écrits dans les mêmes dossiers partagés
    --output-espece / --output-sexe (les classes milouin/morillon/male/
    femelle sont les mêmes quel que soit le job), mais chaque nom de fichier
    est préfixé par le nom du job pour éviter toute collision entre deux
    jobs qui contiendraient une image de même nom (ex. job1__RCNX0045_g00_1.jpg).
S'il n'y a pas de sous-dossiers, l'ancien comportement "à plat" (un seul lot)
est utilisé automatiquement.

Usage :
    python scripts/00_migrate_annotations.py
    python scripts/00_migrate_annotations.py --raw-images models/detection/raw_data/images \
        --raw-labels models/detection/raw_data/labels --classes-file mon_export_cvat/classes.txt
"""

import argparse
from pathlib import Path

import cv2
from utils.config import logger
from utils.console import print_header
from utils.fichiers import find_image, iou, list_job_dirs, union_box

DEFAULT_OLD_CLASSES = [
    "marque_nasale",
    "canard_male",
    "canard_femelle",
    "milouin",
    "morillon",
]

# Mapping nom de classe (ancien schéma) -> (nouvelle_classe_detection, espece, sexe)
# Noms explicitement connus. Tout autre nom rencontré dans --classes-file est
# résolu dynamiquement par resolve_class_mapping() ci-dessous (nouvelle
# espèce) plutôt que d'être ignoré — voir docstring du module.
OLD_CLASS_MAPPING = {
    "marque_nasale": {"detection": "marque_nasale", "espece": None, "sexe": None},
    "canard_male": {"detection": "canard", "espece": None, "sexe": "male"},
    "canard_femelle": {"detection": "canard", "espece": None, "sexe": "femelle"},
    "milouin": {"detection": "canard", "espece": "milouin", "sexe": None},
    "morillon": {"detection": "canard", "espece": "morillon", "sexe": None},
}

# Variantes courantes de nom pour le sexe (au cas où un export CVAT utilise
# des noms de classe plus courts que canard_male/canard_femelle).
SEXE_NAME_VARIANTS = {"male": "male", "femelle": "femelle", "m": "male", "f": "femelle"}

NEW_DETECTION_CLASSES = {"marque_nasale": 0, "canard": 1}


def resolve_class_mapping(old_name, new_species_seen):
    """Résout le mapping d'un nom de classe non listé dans OLD_CLASS_MAPPING.

    Tout nom qui n'est ni marque_nasale ni une variante mâle/femelle est
    traité comme une NOUVELLE ESPÈCE (plutôt qu'ignoré) : la boîte reste une
    boîte de détection "canard", et un crop est généré dans
    --output-espece/<old_name>/. old_name est ajouté à new_species_seen pour
    être listé (et donc vérifiable) en fin d'exécution.
    """
    if old_name in SEXE_NAME_VARIANTS:
        return {"detection": "canard", "espece": None, "sexe": SEXE_NAME_VARIANTS[old_name]}
    new_species_seen.add(old_name)
    return {"detection": "canard", "espece": old_name, "sexe": None}


def load_classes(classes_file):
    """Charge la liste ordonnée des noms de classe de l'ancien schéma."""
    if classes_file is None:
        return DEFAULT_OLD_CLASSES
    with open(classes_file, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def yolo_to_pixel_bbox(x_center, y_center, w, h, img_w, img_h, margin=10):
    """Convertit une bbox YOLO normalisée en coordonnées pixel, avec marge."""
    x1 = (x_center - w / 2) * img_w
    y1 = (y_center - h / 2) * img_h
    x2 = (x_center + w / 2) * img_w
    y2 = (y_center + h / 2) * img_h

    x1 = max(0, int(x1 - margin))
    y1 = max(0, int(y1 - margin))
    x2 = min(img_w, int(x2 + margin))
    y2 = min(img_h, int(y2 + margin))
    return x1, y1, x2, y2


class UnionFind:
    """Petite structure union-find pour regrouper les boîtes d'un même canard."""

    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, i):
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, i, j):
        ri, rj = self.find(i), self.find(j)
        if ri != rj:
            self.parent[ri] = rj

    def groups(self):
        buckets = {}
        for i in range(len(self.parent)):
            buckets.setdefault(self.find(i), []).append(i)
        return list(buckets.values())


def group_canard_boxes(entries, iou_threshold):
    """
    Regroupe les entrées "canard" (mapping["detection"] == "canard") d'une même
    image dont les boîtes pixel se chevauchent (IoU >= iou_threshold), pour
    fusionner les boîtes sexe/espèce d'un même canard annoté deux fois dans
    l'ancien schéma.
    """
    n = len(entries)
    uf = UnionFind(n)
    for i in range(n):
        for j in range(i + 1, n):
            if iou(entries[i]["pixel_bbox"], entries[j]["pixel_bbox"]) >= iou_threshold:
                uf.union(i, j)
    return uf.groups()


def migrate_job(job, raw_images, raw_labels, old_classes, output_labels,
                 output_espece, output_sexe, margin, merge_iou, stats,
                 unresolved_class_ids, missing_images, new_species_seen):
    """Migre un seul job (ou le lot unique si job == '')."""
    label_files = sorted(raw_labels.glob("*.txt"))
    if not label_files:
        print(f"  ⚠️  Aucun fichier .txt trouvé dans {raw_labels}")
        return

    label = f" (job '{job}')" if job else ""
    print(f"\n{len(label_files)} fichier(s) d'annotation trouvé(s){label}.")

    output_labels.mkdir(parents=True, exist_ok=True)
    prefix = f"{job}__" if job else ""

    for label_path in label_files:
        stem = label_path.stem
        img_path = find_image(raw_images, stem)

        with open(label_path, "r", encoding="utf-8") as f:
            raw_lines = [line.strip() for line in f if line.strip()]

        marque_entries = []
        canard_entries = []

        img = None
        img_w = img_h = None

        def ensure_image_loaded(img_path=img_path, stem=stem):
            nonlocal img, img_w, img_h
            if img is not None:
                return img is not False
            if img_path is None:
                key = f"{job}/{stem}" if job else stem
                if key not in missing_images:
                    missing_images.append(key)
                img = False
                return False
            loaded = cv2.imread(str(img_path))
            if loaded is None:
                key = f"{job}/{stem}" if job else stem
                if key not in missing_images:
                    missing_images.append(key)
                img = False
                return False
            img = loaded
            img_h, img_w = img.shape[:2]
            return True

        for line in raw_lines:
            parts = line.split()
            if len(parts) != 5:
                continue
            old_cls_id = int(parts[0])
            x, y, w, h = map(float, parts[1:])

            if old_cls_id >= len(old_classes):
                unresolved_class_ids.add(old_cls_id)
                stats["ignorees"] += 1
                continue

            old_name = old_classes[old_cls_id]
            mapping = OLD_CLASS_MAPPING.get(old_name)
            if mapping is None:
                # Nom absent du mapping figé : traité comme nouvelle espèce
                # (ou variante mâle/femelle) au lieu d'être jeté en silence.
                mapping = resolve_class_mapping(old_name, new_species_seen)

            entry = {"old_cls_id": old_cls_id, "x": x, "y": y, "w": w, "h": h, "mapping": mapping}

            if mapping["detection"] == "marque_nasale":
                marque_entries.append(entry)
                continue

            needs_crop = mapping["espece"] is not None or mapping["sexe"] is not None
            if needs_crop and ensure_image_loaded():
                entry["pixel_bbox"] = yolo_to_pixel_bbox(x, y, w, h, img_w, img_h, margin=0)
            else:
                entry["pixel_bbox"] = None
            canard_entries.append(entry)

        new_lines = []
        for entry in marque_entries:
            new_cls_id = NEW_DETECTION_CLASSES["marque_nasale"]
            new_lines.append(f"{new_cls_id} {entry['x']} {entry['y']} {entry['w']} {entry['h']}")
            stats["marque_nasale"] += 1

        groupable = [e for e in canard_entries if e["pixel_bbox"] is not None]
        ungroupable = [e for e in canard_entries if e["pixel_bbox"] is None]

        groups = group_canard_boxes(groupable, merge_iou) if groupable else []

        for group_idx, idx_group in enumerate(groups):
            group_entries = [groupable[i] for i in idx_group]
            if len(group_entries) > 1:
                stats["canard_fusionnes"] += 1

            merged_bbox = group_entries[0]["pixel_bbox"]
            for e in group_entries[1:]:
                merged_bbox = union_box(merged_bbox, e["pixel_bbox"])
            x1, y1, x2, y2 = merged_bbox

            xc = ((x1 + x2) / 2) / img_w
            yc = ((y1 + y2) / 2) / img_h
            bw = (x2 - x1) / img_w
            bh = (y2 - y1) / img_h
            new_cls_id = NEW_DETECTION_CLASSES["canard"]
            new_lines.append(f"{new_cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
            stats["canard"] += 1

            espece_val = next((e["mapping"]["espece"] for e in group_entries
                                if e["mapping"]["espece"] is not None), None)
            sexe_val = next((e["mapping"]["sexe"] for e in group_entries
                              if e["mapping"]["sexe"] is not None), None)

            if espece_val is not None or sexe_val is not None:
                cx1 = max(0, x1 - margin)
                cy1 = max(0, y1 - margin)
                cx2 = min(img_w, x2 + margin)
                cy2 = min(img_h, y2 + margin)
                crop = img[cy1:cy2, cx1:cx2]
                ids_suffix = "-".join(str(e["old_cls_id"]) for e in group_entries)
                # préfixe job + group_idx : rend le nom unique par boîte ET par job,
                # sans quoi deux canards de la même classe sur la même photo (ou la
                # même photo dans deux jobs) s'écrasaient silencieusement.
                crop_name = f"{prefix}{stem}_g{group_idx:02d}_{ids_suffix}.jpg"

                if espece_val is not None:
                    dest_dir = output_espece / espece_val
                    dest_dir.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(dest_dir / crop_name), crop)
                    stats["espece_crops"] += 1

                if sexe_val is not None:
                    dest_dir = output_sexe / sexe_val
                    dest_dir.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(dest_dir / crop_name), crop)
                    stats["sexe_crops"] += 1
                elif espece_val is not None:
                    dest_dir = output_sexe / "indetermine_juvenile"
                    dest_dir.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(dest_dir / crop_name), crop)
                    stats["sexe_crops_juvenile"] += 1

        for entry in ungroupable:
            new_cls_id = NEW_DETECTION_CLASSES["canard"]
            new_lines.append(f"{new_cls_id} {entry['x']} {entry['y']} {entry['w']} {entry['h']}")
            stats["canard"] += 1

        with open(output_labels / label_path.name, "w", encoding="utf-8") as f:
            f.write("\n".join(new_lines) + ("\n" if new_lines else ""))

    print(f"  ✅ Terminé{label}.")


def main():
    parser = argparse.ArgumentParser(description="Migration des annotations vers le nouveau schéma")
    parser.add_argument("--raw-images", default="models/detection/raw_data/images")
    parser.add_argument("--raw-labels", default="models/detection/raw_data/labels")
    parser.add_argument("--classes-file", default=None,
                         help="Fichier listant les anciens noms de classe, un par ligne, "
                              "dans l'ordre des indices YOLO. Défaut : ordre standard du projet "
                              "(marque_nasale, canard_male, canard_femelle, milouin, morillon).")
    parser.add_argument("--output-labels", default="models/detection/raw_data/labels_migrated",
                         help="Dossier de sortie pour les labels de détection à 2 classes "
                              "(un sous-dossier par job y sera créé si plusieurs jobs).")
    parser.add_argument("--output-espece", default="dataset_espece_migrated")
    parser.add_argument("--output-sexe", default="dataset_sexe_migrated")
    parser.add_argument("--margin", type=int, default=15, help="Marge en pixels autour des crops")
    parser.add_argument("--merge-iou", type=float, default=0.2,
                         help="Seuil de recouvrement (IoU) à partir duquel deux boîtes "
                              "'canard' de la même image sont considérées comme le même "
                              "individu et fusionnées (une boîte sexe + une boîte espèce "
                              "qui se chevauchent, typiquement). Mettre à 1.1 pour désactiver "
                              "toute fusion.")
    parser.add_argument("--job-name", default=None,
                         help="Force le préfixe utilisé pour les noms de crops (espèce/sexe), "
                              "même si --raw-images pointe déjà directement sur le dossier d'un "
                              "seul job (donc sans sous-dossiers détectables). À utiliser quand "
                              "ce script est appelé une fois par job par un script englobant "
                              "(ex. 00c_migrate_jobs.py), pour éviter que deux jobs différents "
                              "ne génèrent des crops de même nom qui s'écrasent silencieusement.")
    args = parser.parse_args()

    print_header("🔄 MIGRATION DES ANNOTATIONS VERS LE NOUVEAU SCHÉMA")

    old_classes = load_classes(args.classes_file)
    print("Ancien schéma de classes (index -> nom) :")
    for i, name in enumerate(old_classes):
        if name in OLD_CLASS_MAPPING:
            flag = ""
        elif name in SEXE_NAME_VARIANTS:
            flag = f"  ↳ traité comme sexe '{SEXE_NAME_VARIANTS[name]}'"
        else:
            flag = "  🆕 non reconnu -> traité comme nouvelle espèce"
        print(f"  {i}: {name}{flag}")

    raw_images_base = Path(args.raw_images)
    raw_labels_base = Path(args.raw_labels)
    if not raw_images_base.exists() or not raw_labels_base.exists():
        logger.error("Dossiers introuvables : %s / %s", raw_images_base, raw_labels_base)
        return

    output_labels_base = Path(args.output_labels)
    output_espece = Path(args.output_espece)
    output_sexe = Path(args.output_sexe)

    if args.job_name:
        # Appelé pour un seul job déjà isolé par le script englobant (ex.
        # 00c_migrate_jobs.py) : raw_images_base/raw_labels_base pointent
        # directement sur le dossier de CE job, pas de sous-dossiers à détecter.
        job_names = [args.job_name]
        multi_job = True  # force le préfixe de nom de fichier (voir migrate_job)
        print(f"\n📁 Job forcé via --job-name : {args.job_name}")
    else:
        job_names = list_job_dirs(raw_images_base)
        multi_job = job_names != [""]
        if multi_job:
            print(f"\n📁 {len(job_names)} job(s) détecté(s) : {', '.join(job_names)}")

    stats = {
        "marque_nasale": 0, "canard": 0, "canard_fusionnes": 0,
        "espece_crops": 0, "sexe_crops": 0, "sexe_crops_juvenile": 0, "ignorees": 0,
    }
    unresolved_class_ids = set()
    missing_images = []
    new_species_seen = set()

    for job in job_names:
        if args.job_name:
            # raw_images_base/raw_labels_base/output_labels_base pointent déjà
            # directement sur le dossier de ce job : ne pas ré-ajouter job au chemin.
            raw_images = raw_images_base
            raw_labels = raw_labels_base
            output_labels = output_labels_base
        else:
            raw_images = raw_images_base / job if job else raw_images_base
            raw_labels = raw_labels_base / job if job else raw_labels_base
            output_labels = output_labels_base / job if job else output_labels_base

        if not raw_images.exists():
            print(f"  ⚠️  Dossier images manquant pour le job '{job}' : {raw_images}")
            continue
        if not raw_labels.exists():
            print(f"  ⚠️  Dossier labels manquant pour le job '{job}' : {raw_labels}")
            continue

        migrate_job(job, raw_images, raw_labels, old_classes, output_labels,
                    output_espece, output_sexe, args.margin, args.merge_iou,
                    stats, unresolved_class_ids, missing_images, new_species_seen)

    print_header("✅ MIGRATION TERMINÉE")
    print(f"Boîtes marque_nasale (détection)     : {stats['marque_nasale']}")
    print(f"Boîtes canard (détection)            : {stats['canard']}")
    print(f"  dont fusionnées (sexe+espèce)      : {stats['canard_fusionnes']}")
    print(f"Crops espèce générés                 : {stats['espece_crops']} -> {output_espece}/")
    print(f"Crops sexe générés                   : {stats['sexe_crops']} -> {output_sexe}/")
    print(f"  dont indetermine_juvenile          : {stats['sexe_crops_juvenile']} "
          f"-> {output_sexe}/indetermine_juvenile/")
    if new_species_seen:
        print(f"\n🆕 Nouvelle(s) espèce(s) détectée(s) dans --classes-file et migrée(s) "
              f"automatiquement (à vérifier) : {', '.join(sorted(new_species_seen))}")
    if stats["ignorees"]:
        logger.warning("Boîtes ignorées (index de classe hors de --classes-file) : %d", stats["ignorees"])
    if unresolved_class_ids:
        logger.warning("Index de classe non résolus (absents de --classes-file) : %s", sorted(unresolved_class_ids))
    if missing_images:
        logger.warning("%d label(s) sans image correspondante (crops non générés)", len(missing_images))

    print(f"\nLabels de détection migrés écrits dans : {output_labels_base}/"
          + ("<job>/" if multi_job else ""))
    if multi_job:
        print("Prochaine étape : pour chaque job, copier "
              f"{output_labels_base}/<job>/*.txt dans raw_data/labels/<job>/ "
              "(en écrasant l'ancien schéma), puis relancer "
              "models/detection/scripts/01_split_dataset.py.")
    else:
        print("Prochaine étape : vérifier ces labels puis les copier dans "
              "models/detection/raw_data/labels/ (en écrasant l'ancien schéma), "
              "puis lancer models/detection/scripts/01_split_dataset.py.")


if __name__ == "__main__":
    main()