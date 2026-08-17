"""
Entraîne un modèle YOLO26 (transfer learning) pour détecter la marque
nasale directement sur la photo du canard.

Usage basique :
    python scripts/02_train.py

Avec options :
    python scripts/02_train.py --model yolo26n.pt --epochs 150 --imgsz 960 --batch 8
"""

import argparse
from pathlib import Path

import yaml
from ultralytics import YOLO

# models/detection/scripts/02_train.py -> ROOT = models/detection
# (utilisé pour les artefacts d'entraînement : runs/, weights, etc.)
ROOT = Path(__file__).resolve().parent.parent

# <repo>/models/detection -> REPO_ROOT = <repo> (= "canard-detection")
# CORRECTIF : le dataset réel (celui alimenté par 01_split_dataset.py, et
# celui utilisé par le notebook Colab : /content/project/dataset/data.yaml)
# vit à la racine du repo, PAS dans models/detection/dataset. L'ancien code
# pointait par erreur vers ROOT / "dataset" (models/detection/dataset), un
# dossier différent — désormais obsolète et supprimé — que 01_split_dataset.py
# ne remplit plus depuis son propre correctif. Les deux scripts pointent
# maintenant vers le même dossier.
REPO_ROOT = ROOT.parent.parent
DATA_YAML = REPO_ROOT / "dataset" / "data.yaml"


def build_resolved_data_yaml() -> Path:
    """
    Certaines versions d'Ultralytics résolvent le champ 'path: .' de data.yaml
    par rapport au dossier courant (cwd) plutôt que par rapport au dossier du
    fichier data.yaml lui-même, ce qui casse la détection de dataset selon
    l'endroit d'où le script est lancé.

    On génère donc un yaml "local" avec un chemin ABSOLU vers dataset/, pour
    éliminer toute ambiguïté. Ce fichier est régénéré à chaque lancement,
    inutile de le committer (voir .gitignore).
    """
    with open(DATA_YAML, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    dataset_dir = (REPO_ROOT / "dataset").resolve()
    cfg["path"] = str(dataset_dir)

    local_yaml = REPO_ROOT / "dataset" / "data.local.yaml"
    with open(local_yaml, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)

    return local_yaml


def get_checkpoint_fitness(weights_path: Path) -> float:
    """
    Lit le score de performance ('fitness', une combinaison de mAP) stocké
    dans le checkpoint par Ultralytics. Plus c'est haut, meilleur est le modèle.
    Retourne -inf si le score est introuvable (pour ne pas planter la comparaison).
    """
    import torch
    try:
        ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"  ⚠️  Impossible de lire {weights_path} ({e})")
        return float("-inf")

    fitness = ckpt.get("best_fitness")
    if fitness is None:
        fitness = ckpt.get("fitness")
    if fitness is None:
        # Certains formats stockent la fitness dans un tenseur/array à 1 élément
        return float("-inf")
    try:
        return float(fitness)
    except (TypeError, ValueError):
        return float("-inf")


def find_best_overall() -> Path | None:
    """
    Compare TOUS les runs précédents (pas juste le plus récent) et retourne
    le best.pt ayant le meilleur score de performance réel.
    """
    candidates = list((ROOT / "runs" / "detect").glob("*/weights/best.pt"))
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]

    print(f"Comparaison de {len(candidates)} run(s) précédent(s) pour trouver le meilleur...")
    scored = []
    for c in candidates:
        fitness = get_checkpoint_fitness(c)
        run_name = c.parent.parent.name
        print(f"  {run_name} : fitness = {fitness:.4f}" if fitness != float("-inf")
              else f"  {run_name} : score illisible, ignoré")
        scored.append((fitness, c))

    scored.sort(key=lambda x: x[0], reverse=True)
    best_fitness, best_path = scored[0]

    if best_fitness == float("-inf"):
        # Aucun score lisible dans aucun run : on retombe sur le plus récent
        print("  Aucun score exploitable trouvé, utilisation du run le plus récent à la place.")
        return max(candidates, key=lambda p: p.stat().st_mtime)

    return best_path


def next_run_name(base_name: str) -> str:
    """Évite d'écraser un run existant : marque_nasale_v1, _v2, _v3, ..."""
    runs_dir = ROOT / "runs" / "detect"
    if not (runs_dir / base_name).exists():
        return base_name
    i = 2
    while (runs_dir / f"{base_name}_run{i}").exists():
        i += 1
    return f"{base_name}_run{i}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None,
                         help="Poids de départ explicites (ex. yolo26n.pt, ou un chemin vers "
                              "un best.pt précis). Si non précisé : reprend automatiquement "
                              "depuis le meilleur modèle du run le plus récent s'il existe, "
                              "sinon part de yolo26n.pt (pré-entraîné COCO).")
    parser.add_argument("--fresh", action="store_true",
                         help="Ignore les runs précédents et repart de zéro (yolo26n.pt), "
                              "même si un best.pt existe déjà.")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--imgsz", type=int, default=960,
                         help="Taille d'image. La marque nasale est un petit objet : "
                              "960 (voire 1280) donne souvent de meilleurs résultats que 640.")
    parser.add_argument("--batch", type=int, default=16,
                         help="Réduis si tu manques de VRAM/RAM (ex. 4 ou 8).")
    parser.add_argument("--device", default=None,
                         help="'0' pour GPU 0, 'cpu' pour forcer le CPU. "
                              "Laisse vide pour auto-détection.")
    parser.add_argument("--patience", type=int, default=30,
                         help="Arrêt anticipé si pas d'amélioration après N époques.")
    parser.add_argument("--name", default="marque_nasale_v1",
                         help="Nom du run (dossier créé sous runs/detect/). Si ce nom existe "
                              "déjà, un suffixe _run2, _run3... est ajouté automatiquement "
                              "pour ne rien écraser.")
    args = parser.parse_args()

    if not DATA_YAML.exists():
        print(f"❌ Fichier introuvable : {DATA_YAML}")
        print("   Vérifie que dataset/data.yaml existe et que le split "
              "(scripts/01_split_dataset.py) a bien été fait.")
        return

    resolved_yaml = build_resolved_data_yaml()

    # --- Choix des poids de départ ---
    if args.model is not None:
        start_weights = args.model
        print(f"Poids de départ (explicite) : {start_weights}")
    elif args.fresh:
        start_weights = "yolo26n.pt"
        print("Option --fresh : on repart de zéro depuis yolo26n.pt (pré-entraîné COCO).")
    else:
        latest_best = find_best_overall()
        if latest_best is not None:
            start_weights = str(latest_best)
            print(f"Reprise automatique depuis le MEILLEUR modèle trouvé : {start_weights}")
            print("   (utilise --fresh pour repartir de zéro, ou --model pour forcer un autre "
                  "point de départ)")
        else:
            start_weights = "yolo26n.pt"
            print("Aucun entraînement précédent trouvé : on part de yolo26n.pt (pré-entraîné COCO).")

    run_name = next_run_name(args.name)
    if run_name != args.name:
        print(f"Le nom '{args.name}' existe déjà, ce run sera nommé '{run_name}'.")

    print(f"\nChargement du modèle : {start_weights}")
    model = YOLO(start_weights)  # télécharge automatiquement les poids COCO si nécessaire

    train_kwargs = dict(
        data=str(resolved_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        name=run_name,
        project=str(ROOT / "runs" / "detect"),
        # Augmentations utiles vu que le dataset sera petit :
        degrees=10.0,      # légère rotation (les canards ne sont pas toujours droits)
        translate=0.1,
        scale=0.3,
        fliplr=0.5,        # flip horizontal ok pour une marque nasale
        flipud=0.0,        # pas de flip vertical (peu réaliste)
        mosaic=1.0,
        close_mosaic=15,   # désactive le mosaic sur les dernières époques
    )
    if args.device is not None:
        train_kwargs["device"] = args.device

    model.train(**train_kwargs)

    print("\n✅ Entraînement terminé.")
    print(f"   Meilleurs poids : runs/detect/{run_name}/weights/best.pt")
    print(f"   Résultats/courbes : runs/detect/{run_name}/")


if __name__ == "__main__":
    main()