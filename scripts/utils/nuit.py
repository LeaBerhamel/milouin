# -*- coding: utf-8 -*-
"""
Détection "photo de nuit" (flash IR), heuristique partagée.

La nuit, le flash infrarouge des caméras de piège produit des images quasi
grises (R≈G≈B) sur lesquelles la couleur d'une marque nasale n'est pas
fiable. On calcule la saturation HSV moyenne du crop (0-255) ; en dessous
du seuil, le crop est considéré comme une photo de nuit probable. Seuil
choisi empiriquement large (une vraie photo de jour dépasse largement 20)
pour éviter les faux positifs sur des marques ternes en plein jour.

Ce module est partagé par :
- scripts/interface_preparation_ia.py : décide quels crops couleur sont mis
  de côté par défaut lors de la préparation du dataset d'entraînement.
- scripts/utils/pipeline.py : signale, en production, qu'une couleur
  détectée sur une photo de nuit est peu fiable (voir "couleur_fiable"
  dans les résultats), sans jamais bloquer le traitement.
"""

SATURATION_SEUIL_NUIT = 18


def est_photo_de_nuit(crop_image) -> bool:
    """Heuristique : True si le crop (image PIL déjà ouverte, mode RGB ou
    convertible) est probablement une photo de nuit au flash IR (quasi
    grise). Ne lève jamais d'exception -> False par défaut si illisible."""
    from PIL import ImageStat

    try:
        hsv = crop_image.convert("HSV")
        saturation_moyenne = ImageStat.Stat(hsv).mean[1]
        return saturation_moyenne < SATURATION_SEUIL_NUIT
    except Exception:
        return False


def est_photo_de_nuit_bgr(crop_bgr) -> bool:
    """Même heuristique que est_photo_de_nuit(), mais pour un crop OpenCV
    (numpy array BGR), comme utilisé par utils/pipeline.py. Retourne False
    par défaut si le crop est vide/illisible (jamais d'exception)."""
    if crop_bgr is None or getattr(crop_bgr, "size", 0) == 0:
        return False
    try:
        import cv2
        from PIL import Image

        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        return est_photo_de_nuit(Image.fromarray(rgb))
    except Exception:
        return False
