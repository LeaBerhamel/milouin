#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Construit un dataset au format YOLO (images/ + labels/ + data.yaml) pour
réentraîner le modèle de DÉTECTION (models/detection/best.pt), en combinant :

  1. Les bbox déjà trouvées par la détection actuelle (canards + marques
     nasales), lues dans resultats_par_rafale.json.
  2. Les bbox ajoutées à la main depuis la fenêtre "photo + liste"
     (module_corrections/fenetre_rafale.py -- canvas interactif, canards
     ratés par l'IA), lues dans dictionnaire/detections_manuelles.json.

Pourquoi combiner les deux plutôt qu'exporter SEULEMENT les ajouts manuels :
un fichier d'annotation YOLO décrit TOUT le contenu d'une photo. Si on
n'y met que le canard raté, l'entraînement comprend "il n'y a qu'UN objet
sur cette photo" et désapprend les autres canards qui étaient pourtant
bien détectés dessus -- l'inverse de l'effet recherché.

Usage (un seul lot) :

    python scripts/13_exporter_annotations_detection.py \\
        --resultats output/resultats_par_rafale.json \\
        --dictionnaire output/dictionnaire \\
        --dataset-dir dataset_detection

Usage (plusieurs lots, mêmes options que 07_fusionner_dictionnaire.py) :

    python scripts/13_exporter_annotations_detection.py \\
        --resultats output/resultats_par_rafale.json output/lots/lot2/resultats_par_rafale.json \\
        --dictionnaire output/dictionnaire output/lots/lot2/dictionnaire \\
        --dataset-dir dataset_detection

⚠️ IMPORTANT -- ordre des classes : ce script écrit les classes dans
l'ordre marque_nasale=0, canard=1 (voir CLASSES ci-dessous) et génère un
data.yaml avec ce même ordre. Si best.pt existant utilise un ordre
différent (vérifiable dans son propre data.yaml d'origine, ou via
`YOLO("models/detection/best.pt").names` en Python), il faut soit
harmoniser CLASSES ci-dessous avant d'exporter, soit repartir d'un
entraînement from scratch (pas d'un fine-tuning) -- mélanger les deux
ordres sans y prendre garde entraînerait le modèle à confondre canard et
marque nasale.

Ne modifie ni resultats_par_rafale.json, ni dictionnaire/ (lecture seule) :
recopie les photos sources dans dataset_dir/images/, n'y touche jamais.
"""

import argparse
import json
import random
from pathlib import Path
import shutil

try:
    from PIL import Image
except ImportError:
    Image = None

# Ordre des classes pour CE dataset exporté -- voir avertissement ci-dessus
# avant de fine-tuner un best.pt existant plutôt que de repartir de zéro.
CLASSES = ["marque_nasale", "canard"]


def charger_json(path: Path):
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def collecter_annotations(resultats_paths, dictionnaire_paths):
    """Retourne {photo_path: [(classe_idx, x, y, w, h), ...]} en combinant
    les détections IA (resultats_par_rafale.json, un ou plusieurs lots) et
    les ajouts manuels (detections_manuelles.json, un ou plusieurs
    dictionnaires). photo_path reste tel quel dans les JSON (chemin absolu
    de la machine où le lot a été traité) -- résolu plus tard, à l'export,
    contre le disque courant.
    """
    par_photo = {}
    stats = {"depuis_ia": 0, "depuis_ajout_manuel": 0, "photos_introuvables": 0}

    idx_marque = CLASSES.index("marque_nasale")
    idx_canard = CLASSES.index("canard")

    for resultats_path in resultats_paths:
        resultats = charger_json(resultats_path)
        if resultats is None:
            print(f"⚠️  {resultats_path} introuvable, ignoré.")
            continue

        for rafale in resultats:
            photo = rafale.get("photo_analysee")
            if not photo:
                continue
            boites = par_photo.setdefault(photo, [])

            for canard in rafale.get("canards", []):
                bbox = canard.get("bbox")
                if bbox:
                    boites.append((idx_canard, tuple(bbox)))
                    stats["depuis_ia"] += 1

            for marque in rafale.get("marques", []):
                bbox = marque.get("bbox")
                if bbox:
                    boites.append((idx_marque, tuple(bbox)))
                    stats["depuis_ia"] += 1

    for dossier_dictionnaire in dictionnaire_paths:
        detections = charger_json(Path(dossier_dictionnaire) / "detections_manuelles.json")
        if not detections:
            continue

        for entree in detections.values():
            photo = entree.get("photo_source")
            bbox = entree.get("bbox")
            if not photo or not bbox:
                continue
            classe = entree.get("classe", "canard")
            idx_classe = idx_canard if classe == "canard" else idx_marque
            boites = par_photo.setdefault(photo, [])
            # Évite le doublon si la même bbox existe déjà (ex. script relancé
            # deux fois sur le même dictionnaire) -- comparaison exacte,
            # suffisant puisque les coordonnées viennent du même clic source.
            paire = (idx_classe, tuple(bbox))
            if paire not in boites:
                boites.append(paire)
                stats["depuis_ajout_manuel"] += 1

    return par_photo, stats


def bbox_vers_ligne_yolo(classe_idx, bbox, largeur_img, hauteur_img):
    x, y, w, h = bbox
    cx = (x + w / 2) / largeur_img
    cy = (y + h / 2) / hauteur_img
    nw = w / largeur_img
    nh = h / hauteur_img
    # Coordonnées hors image (bbox mal reconvertie, photo redimensionnée
    # entre-temps...) : on les rogne plutôt que d'écrire un fichier YOLO
    # invalide qui ferait planter l'entraînement sur cette seule image.
    cx, cy = min(max(cx, 0.0), 1.0), min(max(cy, 0.0), 1.0)
    nw, nh = min(max(nw, 0.0), 1.0), min(max(nh, 0.0), 1.0)
    return f"{classe_idx} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}"


def exporter(par_photo, dataset_dir: Path, val_ratio: float, seed: int):
    if Image is None:
        print("❌ Pillow n'est pas installé (pip install Pillow) -- impossible de lire la "
              "résolution des photos pour normaliser les bbox.")
        return

    photos_valides = []
    for photo_path, boites in par_photo.items():
        if not boites:
            continue
        if not Path(photo_path).exists():
            continue
        photos_valides.append(photo_path)

    photos_introuvables = len(par_photo) - len(photos_valides)
    if not photos_valides:
        print("❌ Aucune photo source accessible sur ce disque -- vérifie que les chemins de "
              "resultats_par_rafale.json / detections_manuelles.json pointent vers des photos "
              "présentes ici (ils sont enregistrés tels quels, machine d'origine incluse).")
        return

    random.Random(seed).shuffle(photos_valides)
    n_val = max(1, round(len(photos_valides) * val_ratio)) if len(photos_valides) > 1 else 0
    ensembles = {"val": set(photos_valides[:n_val]), "train": set(photos_valides[n_val:])}

    for sous_dossier in ("images/train", "images/val", "labels/train", "labels/val"):
        (dataset_dir / sous_dossier).mkdir(parents=True, exist_ok=True)

    compte_objets = {c: 0 for c in CLASSES}
    n_photos_ecrites = 0

    for split, ensemble in ensembles.items():
        for photo_path in ensemble:
            boites = par_photo[photo_path]
            try:
                with Image.open(photo_path) as img:
                    largeur_img, hauteur_img = img.size
            except Exception as e:  # noqa: BLE001
                print(f"⚠️  Impossible d'ouvrir {photo_path} ({e}), ignorée.")
                continue

            nom = Path(photo_path).stem
            # Suffixe court sur le nom de fichier pour éviter les collisions
            # entre deux photos de même nom venant de lots différents
            # (ex. RCNX0012.JPG dans job1 ET dans job2).
            suffixe = str(abs(hash(photo_path)))[:6]
            nom_unique = f"{nom}_{suffixe}"

            dest_image = dataset_dir / f"images/{split}/{nom_unique}{Path(photo_path).suffix}"
            dest_label = dataset_dir / f"labels/{split}/{nom_unique}.txt"

            shutil.copyfile(photo_path, dest_image)
            lignes = []
            for classe_idx, bbox in boites:
                lignes.append(bbox_vers_ligne_yolo(classe_idx, bbox, largeur_img, hauteur_img))
                compte_objets[CLASSES[classe_idx]] += 1
            dest_label.write_text("\n".join(lignes) + "\n", encoding="utf-8")
            n_photos_ecrites += 1

    data_yaml = dataset_dir / "data.yaml"
    data_yaml.write_text(
        "path: " + str(dataset_dir.resolve()) + "\n"
        "train: images/train\n"
        "val: images/val\n"
        f"nc: {len(CLASSES)}\n"
        f"names: {CLASSES}\n",
        encoding="utf-8",
    )

    print(f"✅ Dataset écrit dans {dataset_dir.resolve()}")
    print(f"   {n_photos_ecrites} photos ({len(ensembles['train'])} train / {len(ensembles['val'])} val)")
    for classe, n in compte_objets.items():
        print(f"   - {classe} : {n} bbox")
    if photos_introuvables:
        print(f"⚠️  {photos_introuvables} photo(s) référencée(s) mais introuvable(s) sur ce disque, ignorée(s).")
    print(f"\nPour réentraîner (nécessite ultralytics -- pip install ultralytics) :")
    print(f"   python scripts/16_reentrainer_detection.py --data {data_yaml} --epochs 50")
    print("   (fine-tuning à partir du modèle actuel par défaut, puis comparaison automatique "
          "au modèle en production sur ce même dataset -- voir l'avertissement sur l'ordre des "
          "classes en haut de ce script avant de lancer).")


def main():
    parser = argparse.ArgumentParser(
        description="Exporte un dataset YOLO (détection) combinant les bbox du pipeline et les "
                    "ajouts manuels de canards ratés.")
    parser.add_argument("--resultats", nargs="+", required=True,
                         help="Un ou plusieurs resultats_par_rafale.json (un par lot).")
    parser.add_argument("--dictionnaire", nargs="+", required=True,
                         help="Un ou plusieurs dossiers dictionnaire/ (un par lot) contenant "
                              "detections_manuelles.json.")
    parser.add_argument("--dataset-dir", default="dataset_detection",
                         help="Dossier de sortie du dataset YOLO (défaut : dataset_detection).")
    parser.add_argument("--val-ratio", type=float, default=0.15,
                         help="Proportion de photos pour la validation (défaut : 0.15).")
    parser.add_argument("--seed", type=int, default=42, help="Graine du split train/val.")
    args = parser.parse_args()

    resultats_paths = [Path(p) for p in args.resultats]
    dictionnaire_paths = [Path(p) for p in args.dictionnaire]

    par_photo, stats = collecter_annotations(resultats_paths, dictionnaire_paths)
    print(f"📊 {stats['depuis_ia']} bbox depuis le pipeline, "
          f"{stats['depuis_ajout_manuel']} bbox ajoutée(s) manuellement, "
          f"{len(par_photo)} photo(s) au total avant filtrage.")

    exporter(par_photo, Path(args.dataset_dir), args.val_ratio, args.seed)


if __name__ == "__main__":
    main()