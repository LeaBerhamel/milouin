# -*- coding: utf-8 -*-
"""
Extraction d'empreintes visuelles (embeddings) pour la ré-identification de
canards par leur marque nasale.

Approche : un CNN pré-entraîné sur ImageNet (ResNet18, déjà utilisé ailleurs
dans ce projet -- voir scripts/train_classification_ia.py), utilisé UNIQUEMENT
comme extracteur de caractéristiques (sa dernière couche de classification
est retirée) -- aucun entraînement supplémentaire nécessaire, aucune donnée
à préparer. Chaque crop devient un vecteur de 512 nombres ; deux crops
visuellement proches (couleur, forme, texture de la marque) ont des vecteurs
proches (similarité cosinus proche de 1).

⚠️ Ce n'est PAS un modèle entraîné spécifiquement pour reconnaître VOS
marques nasales -- c'est un extracteur de caractéristiques génériques
"prêt à l'emploi", utilisé ici comme heuristique d'aide à la décision (voir
scripts/06_proposer_correspondances_visuelles.py). Si les propositions
générées se révèlent peu fiables à l'usage, la suite logique serait d'entraîner
un vrai modèle de ré-identification sur vos données (comme cela a été fait
pour couleur/espèce/sexe) plutôt que de se reposer sur ce module.
"""

from pathlib import Path

import numpy as np

_model = None
_transform = None
_device = None


def _get_model():
    """Charge le modèle une seule fois (réutilisé pour tous les appels)."""
    global _model, _transform, _device
    if _model is not None:
        return _model, _transform, _device

    import torch
    from torchvision import models, transforms

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = torch.nn.Identity()  # retire la classification -> vecteur 512-d brut
    model.eval()
    model.to(device)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    _model, _transform, _device = model, transform, device
    return _model, _transform, _device


def extract_embedding(image_path):
    """
    Calcule l'empreinte visuelle d'un crop.

    Args:
        image_path: Chemin vers le fichier image du crop.

    Returns:
        np.ndarray de forme (512,), normalisé (norme euclidienne = 1) pour
        que le produit scalaire entre deux embeddings soit directement leur
        similarité cosinus.
    """
    import torch
    from PIL import Image

    model, transform, device = _get_model()
    img = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        vec = model(tensor).cpu().numpy()[0]

    norme = np.linalg.norm(vec)
    if norme > 0:
        vec = vec / norme
    return vec.astype(np.float32)


def cosine_similarity(a, b):
    """Similarité cosinus entre deux embeddings déjà normalisés (produit scalaire)."""
    return float(np.dot(a, b))