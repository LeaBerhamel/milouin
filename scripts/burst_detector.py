# -*- coding: utf-8 -*-
"""
Sauvegarde sur disque, en haute résolution, des crops de marque nasale
détectés par le pipeline, et enrichissement des résultats avec le nom de
fichier du crop correspondant.

Ce module est volontairement séparé de pipeline.py : il ne touche à aucune
logique de détection/classification/matching existante, il se contente de
lire la photo déjà analysée et les bbox déjà calculées (result["marques"])
pour en extraire des fichiers image.

Utilisation typique (voir scripts/02_process_rafales.py) :

    from utils.crop_export import save_marque_crops

    for i, burst in enumerate(rafales):
        result = pipeline.process_burst(burst)
        result["rafale_id"] = i
        save_marque_crops(result, output_dir / "crops_marques_nasales")
        resultats.append(result)
"""

import hashlib
import re
from pathlib import Path

import cv2


def save_marque_crops(result, crops_dir, margin=15, jpeg_quality=100):
    """
    Extrait et sauvegarde en haute résolution chaque marque nasale détectée
    sur la photo analysée d'une rafale, directement depuis l'image source
    (pas depuis une version redimensionnée) pour préserver la résolution
    native.

    Mute `result` en place : ajoute "crop_filename" (str) et "crop_path"
    (str) à chaque dict de result["marques"].

    Args:
        result: Dictionnaire résultat d'une rafale (retour de
            pipeline.process_burst). "rafale_id" n'est plus utilisé pour le
            nommage (voir plus bas) et n'est donc plus requis.
        crops_dir: Dossier de sortie pour les crops (créé si besoin).
        margin: Marge en pixels ajoutée autour de la bbox détectée, pour ne
            pas couper le bord de la marque (les bbox YOLO sont rarement
            pixel-parfaites).
        jpeg_quality: Qualité JPEG de sauvegarde (100 = pas de perte visible,
            fichiers plus lourds -- cohérent avec la contrainte "haute
            résolution" du cahier des charges).

    Returns:
        Liste des noms de fichiers de crop créés (dans l'ordre de
        result["marques"]).
    """
    marques = result.get("marques", [])
    if not marques:
        return []

    photo_path = result.get("photo_analysee")
    if not photo_path:
        return []

    crops_dir = Path(crops_dir)
    crops_dir.mkdir(parents=True, exist_ok=True)

    img = cv2.imread(str(photo_path))
    if img is None:
        # Photo illisible (chemin invalide, fichier corrompu...) : on ne
        # plante pas le run entier pour un crop manquant, on journalise
        # simplement l'absence via l'appelant (crop_filename reste absent).
        return []

    h_img, w_img = img.shape[:2]
    stem = Path(photo_path).stem
    # Empreinte courte du chemin ABSOLU de la photo source (pas juste son nom)
    # -> deux runs différents (dossiers --photos différents, rafale_id qui
    # repart à 0 à chaque lancement) ne peuvent jamais générer le même nom de
    # crop pour deux photos distinctes, même en cas de doublon de nom de
    # fichier caméra-piège (RCNX0001.JPG très courant d'un job à l'autre).
    # Bonus : relancer sur LA MÊME photo régénère exactement le même nom
    # (ré-écriture propre, pas de doublon qui s'accumule).
    path_hash = hashlib.md5(str(Path(photo_path).resolve()).encode("utf-8")).hexdigest()[:8]
    created = []

    for m in marques:
        x, y, w, h = m["bbox"]
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(w_img, x + w + margin)
        y2 = min(h_img, y + h + margin)

        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        crop_filename = f"{stem}_{path_hash}_marque{m['marque_index']:02d}_crop.jpg"
        crop_path = crops_dir / crop_filename

        cv2.imwrite(str(crop_path), crop, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])

        m["crop_filename"] = crop_filename
        m["crop_path"] = str(crop_path)
        created.append(crop_filename)

    return created


def nom_crop_correction(photo_path, rafale_id, canard_index):
    """Nom de fichier déterministe qu'utilise (ou utiliserait) le crop de
    correction d'un canard précis, SANS lire ni écrire d'image -- extrait
    de save_canard_crop() pour être réutilisable ailleurs (utils/report.py
    s'en sert pour retrouver, au moment de générer le rapport HTML, si ce
    canard a déjà été corrigé, sans avoir besoin que le crop existe
    réellement sur disque).

    Même formule que le nom réellement écrit par save_canard_crop -- ne
    jamais faire diverger les deux, sous peine de casser la détection de
    correction côté rapport.
    """
    stem = Path(photo_path).stem
    path_hash = hashlib.md5(str(Path(photo_path).resolve()).encode("utf-8")).hexdigest()[:8]
    return f"{stem}_{path_hash}_rafale{rafale_id}_canard{canard_index:02d}_crop.jpg"


_RE_CROP_CORRECTION = re.compile(r"_rafale(\d+)_canard(\d+)_crop\.jpg$")


def parse_nom_crop_correction(crop_filename):
    """Extrait (rafale_id, canard_index) d'un nom de fichier produit par
    nom_crop_correction/save_canard_crop -- inverse de la formule ci-dessus.

    Utilisé par 02_process_rafales.py pour retrouver, à partir d'une entrée
    de dictionnaire/detections_manuelles.json (indexée par ce nom de
    fichier), à quel canard_index un ajout manuel doit être réinjecté après
    un nouveau passage du pipeline -- SANS avoir à faire porter ce champ à
    detections_manuelles.json elle-même (le nom de fichier reste la seule
    source de vérité pour cette convention, comme partout ailleurs dans ce
    module).

    Retourne None si crop_filename ne suit pas la convention attendue
    (ex. crop de marque nasale, ou fichier renommé à la main).
    """
    m = _RE_CROP_CORRECTION.search(Path(crop_filename).name)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def save_canard_crop(photo_path, bbox, dest_dir, rafale_id, canard_index, margin=15, jpeg_quality=100):
    """
    Extrait et sauvegarde le crop d'UN canard précis, à la volée, au
    moment où un chercheur valide une correction pour lui depuis la
    fenêtre "photo + liste" (voir module_corrections/fenetre_rafale.py).

    Contrairement à save_marque_crops (appelée pour TOUTES les marques de
    TOUTES les rafales pendant 02_process_rafales.py), cette fonction n'est
    appelée que pour le canard réellement corrigé -- inutile d'extraire un
    crop pour chaque canard de chaque photo si personne ne le corrige
    jamais. Même pattern (marge, format JPEG) que save_marque_crops, pour
    rester cohérent visuellement avec les crops de marques.

    Args:
        photo_path: Chemin de la photo source ANALYSÉE (result["photo_analysee"],
            pas la version annotée/redimensionnée -- on veut la résolution
            native, comme pour save_marque_crops).
        bbox: (x, y, w, h) du canard, en pixels sur la photo source
            (result["canards"][i]["bbox"]).
        dest_dir: Dossier de sortie (créé si besoin). Convention utilisée
            par l'appelant : dictionnaire/crops_corrections/.
        rafale_id: Identifiant de la rafale (pour un nom de fichier lisible
            et traçable, ex. dans corrections.csv).
        canard_index: Index du canard dans result["canards"] (numérotation
            humaine faite par l'appelant si besoin, pas ici).

    Returns:
        Le chemin (str) du fichier créé, ou None si la photo est illisible
        ou si la bbox ne donne aucun pixel exploitable.
    """
    if not photo_path:
        return None

    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    img = cv2.imread(str(photo_path))
    if img is None:
        return None

    h_img, w_img = img.shape[:2]
    x, y, w, h = bbox
    x1 = max(0, x - margin)
    y1 = max(0, y - margin)
    x2 = min(w_img, x + w + margin)
    y2 = min(h_img, y + h + margin)

    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    # Même logique de nommage que save_marque_crops : empreinte du chemin
    # ABSOLU de la photo source, pour ne jamais collisionner entre deux
    # runs différents, tout en régénérant le même nom si on ré-extrait le
    # même canard (ré-écriture propre, pas d'accumulation de doublons).
    crop_filename = nom_crop_correction(photo_path, rafale_id, canard_index)
    crop_path = dest_dir / crop_filename

    cv2.imwrite(str(crop_path), crop, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])

    return str(crop_path)