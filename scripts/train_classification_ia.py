"""
Entraîne un modèle de classification (espèce ou sexe) avec transfer learning ResNet18.

Usage :
    python scripts/train_classification_ia.py --dataset dataset_espece --model-name espece --epochs 50
    python scripts/train_classification_ia.py --dataset dataset_sexe --model-name sexe --epochs 50
"""

import argparse
import json
from datetime import date
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

from utils.config import logger

ROOT = Path(__file__).resolve().parent.parent

# scripts/gestion_modeles.py se trouve dans le même dossier que ce script
# (les deux vivent sous scripts/) : import direct, pas besoin de manipuler sys.path.
import gestion_modeles as gm


def train_model(dataset_path: Path, model_name: str, epochs: int = 50, batch_size: int = 32):
    """Entraîne un modèle de classification depuis un dataset organisé par classe."""
    
    # Transformations (augmentation pour train, normalisation pour val)
    data_transforms = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
        "val": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
    }

    # Chargement des datasets
    image_datasets = {
        split: datasets.ImageFolder(dataset_path / split, data_transforms[split])
        for split in ["train", "val"]
    }

    dataloaders = {
        split: DataLoader(image_datasets[split], batch_size=batch_size, shuffle=(split == "train"), num_workers=4)
        for split in ["train", "val"]
    }

    dataset_sizes = {split: len(image_datasets[split]) for split in ["train", "val"]}  # type: ignore[arg-type]
    class_names = image_datasets["train"].classes

    logger.info(f"Classes détectées : {class_names}")
    logger.info(f"Taille du dataset : train={dataset_sizes['train']}, val={dataset_sizes['val']}")

    # Modèle : ResNet18 pré-entraîné
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(class_names))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    best_acc = 0.0
    best_model_wts = model.state_dict()

    for epoch in range(epochs):
        logger.info(f"\nÉpoque {epoch + 1}/{epochs}")
        logger.info("-" * 40)

        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += int(torch.sum(preds == labels.data).item())

            if phase == "train":
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = float(running_corrects) / dataset_sizes[phase]

            logger.info(f"{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()

    logger.info(f"\n✅ Meilleure accuracy validation : {best_acc:.4f}")

    # Sauvegarde du meilleur modèle
    model.load_state_dict(best_model_wts)

    if model_name not in gm.IA_CONFIG:
        # IA inconnue de gestion_modeles.py (config.py utilise un autre nom que
        # detection/couleur/espece/sexe) : on retombe sur l'ancien comportement,
        # avec un avertissement explicite plutôt qu'un écrasement silencieux.
        output_path = ROOT / "models" / model_name / f"modele_{model_name}.pt"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), output_path)
        logger.warning(
            f"IA '{model_name}' inconnue de gestion_modeles.py (attendu : "
            f"{list(gm.IA_CONFIG)}) — sauvegarde legacy sans versioning ni "
            f"classes.json : {output_path}"
        )
        return

    # Sauvegarde versionnée (comme interface_preparation_ia.py) : le modèle
    # est écrit dans models/<ia>/versions/vN_<date>/, PAS directement dans
    # models/<ia>/modele_<ia>.pt. La production active n'est donc jamais
    # écrasée par ce script — il faut explicitement lancer
    # `gestion_modeles.py promote` pour l'activer, une fois les métriques
    # jugées satisfaisantes.
    cfg = gm.IA_CONFIG[model_name]
    version_name = gm.next_version_name(model_name)
    vdir = gm.versions_dir(model_name) / version_name
    vdir.mkdir(parents=True, exist_ok=True)

    model_path = vdir / cfg["model_file"]
    torch.save(model.state_dict(), model_path)

    classes_payload = json.dumps(class_names, ensure_ascii=False, indent=2)
    if cfg["classes_file"]:
        (vdir / cfg["classes_file"]).write_text(classes_payload, encoding="utf-8")
    # Convention "<model>.classes.json" attendue en priorité par
    # utils/pipeline.py::_load_classifier() (voir son commentaire, qui citait
    # justement ce script) : on l'écrit systématiquement à côté du .pt.
    classes_json_path = model_path.with_suffix(".classes.json")
    classes_json_path.write_text(classes_payload, encoding="utf-8")

    metrics = {
        "val_acc": best_acc,
        "date": date.today().isoformat(),
        "notes": "Entraîné via train_classification_ia.py (script CLI local, hors interface_preparation_ia.py)",
        "dataset": str(dataset_path),
        "source_model_file": model_path.name,
    }
    (vdir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    logger.info(f"✅ Version enregistrée : models/{model_name}/versions/{version_name}/")
    logger.info(f"   classes ({len(class_names)}) : {class_names}")
    logger.info("⚠️  Cette version N'EST PAS active en production (aucun fichier de models/"
                f"{model_name}/ n'a été modifié).")
    logger.info(f"   Historique / comparaison : python scripts/gestion_modeles.py list --ia {model_name}")
    logger.info(f"   Pour l'activer :            python scripts/gestion_modeles.py promote --ia {model_name} --version {version_name}")

    # Validation finale
    model.eval()
    all_preds: list[int] = []
    all_labels: list[int] = []

    with torch.no_grad():
        for inputs, labels in dataloaders["val"]:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    from sklearn.metrics import classification_report

    logger.info("\nRapport de classification (validation) :")
    logger.info("\n" + classification_report(all_labels, all_preds, target_names=class_names))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Dossier racine du dataset (contient train/ et val/)")
    parser.add_argument("--model-name", required=True, help="Nom du modèle (espece, sexe, couleur...)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not (dataset_path / "train").exists() or not (dataset_path / "val").exists():
        print(f"❌ Le dataset doit contenir des dossiers train/ et val/")
        return

    train_model(dataset_path, args.model_name, args.epochs, args.batch_size)


if __name__ == "__main__":
    main()