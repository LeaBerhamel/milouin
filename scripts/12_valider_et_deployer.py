#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Valide un modèle réentraîné sur un jeu de test fixe et le déploie s'il est plus performant.

Usage:
    python scripts/12_valider_et_deployer.py \
        --nouveau models/espece/v2/modele_espece.pt \
        --actuel models/espece/modele_espece.pt \
        --test-dir dataset_espece_test
"""

import argparse
import json
import shutil
from pathlib import Path

import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.models import resnet18

REPO_ROOT = Path(__file__).resolve().parent.parent


def evaluer_modele(model_path: Path, test_dir: Path) -> float:
    # Le .pt contient le state_dict brut (pas de dict englobant) et les classes
    # vivent dans un fichier séparé <model>.classes.json à côté — même convention
    # que utils/pipeline.py::_load_classifier() et
    # interface_preparation_ia.py::entrainer_classification(). Avant, ce module
    # attendait {"model_state_dict":..., "classes":...}, ce qui faisait échouer
    # (KeyError: 'classes') toute évaluation d'un modèle produit par ces deux-là.
    classes_path = model_path.with_suffix(".classes.json")
    if not classes_path.exists():
        raise FileNotFoundError(
            f"Fichier de classes introuvable pour {model_path} (attendu : {classes_path})."
        )
    with open(classes_path, "r", encoding="utf-8") as f:
        classes = json.load(f)

    state_dict = torch.load(model_path, map_location="cpu")

    model = resnet18()
    model.fc = torch.nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(state_dict)
    model.eval()

    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    test_dataset = ImageFolder(root=str(test_dir), transform=transform)

    # Garde-fou : ImageFolder attribue les indices de classe par ordre
    # alphabétique des sous-dossiers *présents dans test_dir*. Si ce jeu de
    # classes ne correspond pas exactement (même ordre, même contenu) à celui
    # utilisé à l'entraînement, les indices ne veulent plus dire la même chose
    # côté modèle et côté vérité terrain, et l'accuracy calculée est fausse
    # sans qu'aucune erreur ne soit levée. Voir utils/pipeline.py pour le même
    # piège côté production.
    if test_dataset.classes != classes:
        raise ValueError(
            "Le jeu de classes du dossier de test ne correspond pas à celui du "
            f"modèle : test={test_dataset.classes} vs modèle={classes}. "
            "L'évaluation serait faussée (indices de classe décalés) — "
            "vérifie que test_dir contient exactement les mêmes classes, dans "
            "les mêmes noms, que le dataset d'entraînement."
        )

    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total if total > 0 else 0.0


def valider_et_deployer(nouveau_modele: Path, ancien_modele: Path, test_dir: Path, seuil_min: float = 0.85):
    if not test_dir.exists():
        print(f"❌ Dossier de test fixe introuvable : {test_dir}. Validation annulée.")
        return

    print("📊 Évaluation du nouveau modèle sur le jeu de test fixe...")
    acc_nouveau = evaluer_modele(nouveau_modele, test_dir)
    print(f"   • Exactitude (Accuracy) nouveau modèle : {acc_nouveau:.2%}")

    acc_actuel = 0.0
    if ancien_modele.exists():
        acc_actuel = evaluer_modele(ancien_modele, test_dir)
        print(f"   • Exactitude (Accuracy) modèle actuel   : {acc_actuel:.2%}")

    if acc_nouveau >= acc_actuel and acc_nouveau >= seuil_min:
        print("✅ Le nouveau modèle est validé ! Remplacement en production...")
        ancien_modele.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(nouveau_modele, ancien_modele)
        # Le .classes.json doit voyager avec le .pt, sinon le modèle déployé
        # n'a plus son fichier de classes et utils/pipeline.py retombe sur le
        # tri alphabétique de secours (fragile) au lieu de l'ordre réel
        # d'entraînement.
        nouvelles_classes = nouveau_modele.with_suffix(".classes.json")
        if nouvelles_classes.exists():
            shutil.copy2(nouvelles_classes, ancien_modele.with_suffix(".classes.json"))
        print(f"🚀 Modèle mis à jour : {ancien_modele}")
    else:
        print("⚠️ Le nouveau modèle n'a pas dépassé les performances actuelles. Non déployé.")


def main():
    parser = argparse.ArgumentParser(description="Valide et déploie un nouveau modèle.")
    parser.add_argument("--nouveau", required=True, help="Chemin vers le nouveau modèle v2/v3")
    parser.add_argument("--actuel", required=True, help="Chemin vers le modèle en production")
    parser.add_argument("--test-dir", required=True, help="Dossier du jeu de test (hold-out)")
    parser.add_argument("--seuil-min", type=float, default=0.85, help="Seuil d'exactitude minimal")
    args = parser.parse_args()

    valider_et_deployer(
        nouveau_modele=REPO_ROOT / args.nouveau,
        ancien_modele=REPO_ROOT / args.actuel,
        test_dir=REPO_ROOT / args.test_dir,
        seuil_min=args.seuil_min,
    )


if __name__ == "__main__":
    main()