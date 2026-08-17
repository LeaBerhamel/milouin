#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Réentraîne un modèle de classification (espece, sexe, couleur) et enregistre la version.

Usage:
    python scripts/11_reentrainer_modele.py --type espece --version v2 --epochs 10
"""

import argparse
import json
from pathlib import Path

import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.models import ResNet18_Weights, resnet18

REPO_ROOT = Path(__file__).resolve().parent.parent


def reentrainer_classification(dataset_dir: Path, output_model_path: Path, epochs: int = 10):
    if not dataset_dir.exists():
        print(f"❌ Le dossier de dataset {dataset_dir} n'existe pas.")
        return

    transform = T.Compose([
        T.Resize((224, 224)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    dataset = ImageFolder(root=str(dataset_dir), transform=transform)
    if len(dataset) == 0:
        print(f"❌ Aucune image trouvée dans {dataset_dir}.")
        return

    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    # Transfer Learning ResNet18
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, len(dataset.classes))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    print(f"🚀 Début du réentraînement sur {dataset_dir.name} ({len(dataset)} images, {len(dataset.classes)} classes)...")

    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(dataset)
        print(f"  Époque {epoch + 1}/{epochs} - Loss: {epoch_loss:.4f}")

    output_model_path.parent.mkdir(parents=True, exist_ok=True)
    # IMPORTANT : on sauvegarde le state_dict brut (pas un dict englobant), et les
    # classes dans un fichier .classes.json séparé, à côté du .pt. C'est le format
    # attendu par utils/pipeline.py::_load_classifier() et par
    # interface_preparation_ia.py::entrainer_classification() — un ancien format
    # {"model_state_dict":..., "classes":...} rendait les modèles produits ici
    # illisibles par le pipeline de production, et cassait aussi la validation
    # dans 12_valider_et_deployer.py::evaluer_modele() lorsqu'elle était appelée
    # sur un modèle venant d'ailleurs (elle attendait déjà ce format brut).
    torch.save(model.state_dict(), output_model_path)
    classes_path = output_model_path.with_suffix(".classes.json")
    with open(classes_path, "w", encoding="utf-8") as f:
        json.dump(dataset.classes, f, ensure_ascii=False, indent=2)
    print(f"💾 Nouveau modèle sauvegardé dans : {output_model_path}")
    print(f"💾 Classes sauvegardées dans : {classes_path}\n")


def reentrainer_detection(version: str, epochs: int = 50):
    """Exporte le dataset de détection puis lance le réentraînement YOLO."""
    import subprocess

    print("\n📦 1. Exportation/fusion des annotations pour la détection...")
    script_export = REPO_ROOT / "scripts/13_exporter_annotations_detection.py"
    cmd_export = [
        sys.executable,
        str(script_export),
        "--resultats", str(REPO_ROOT / "output/resultats_par_rafale.json"),
        "--dictionnaire", str(REPO_ROOT / "output/dictionnaire"),
        "--dataset-dir", str(REPO_ROOT / "dataset_detection_corrections"),
        "--val-ratio", "0.15"
    ]
    subprocess.run(cmd_export, check=True)

    print(f"\n🚀 2. Lancement du réentraînement YOLO ({epochs} époques)...")
    output_dir = REPO_ROOT / f"models/detection/{version}"
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd_yolo = [
        "yolo", "task=detect", "mode=train",
        f"model={REPO_ROOT / 'models/detection/best.pt'}",
        f"data={REPO_ROOT / 'dataset_detection_corrections/data.yaml'}",
        f"epochs={epochs}",
        "imgsz=640",
        f"project={REPO_ROOT / 'models/detection'}",
        f"name={version}"
    ]
    subprocess.run(cmd_yolo, check=True)
    print(f"\n✅ Réentraînement Détection terminé. Modèle enregistré sous : {output_dir}\n")


def main():
    parser = argparse.ArgumentParser(description="Réentraîne un modèle de manière incrémentale.")
    # 'detection' est ajouté aux choix possibles
    parser.add_argument("--type", choices=["espece", "sexe", "couleur", "detection"], required=True, help="Type de modèle")
    parser.add_argument("--version", default="v2", help="Nom de version du modèle (ex: v2, v3)")
    parser.add_argument("--epochs", type=int, default=10, help="Nombre d'époques d'entraînement")
    args = parser.parse_args()

    if args.type == "detection":
        reentrainer_detection(args.version, epochs=args.epochs)
    else:
        dataset_map = {
            "espece": (REPO_ROOT / "dataset_espece", REPO_ROOT / f"models/espece/{args.version}/modele_espece.pt"),
            "sexe": (REPO_ROOT / "dataset_sexe", REPO_ROOT / f"models/sexe/{args.version}/modele_sexe.pt"),
            "couleur": (REPO_ROOT / "dataset_couleur", REPO_ROOT / f"models/couleur/{args.version}/modele_couleur.pt"),
        }
        dataset_dir, output_model_path = dataset_map[args.type]
        reentrainer_classification(dataset_dir, output_model_path, epochs=args.epochs)


if __name__ == "__main__":
    main()