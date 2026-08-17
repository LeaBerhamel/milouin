"""
Promeut le meilleur modèle de détection entraîné vers models/detection/best.pt,
le chemin par défaut utilisé par scripts/02_process_rafales.py.

Compare TOUS les runs sous models/detection/runs/detect/*/weights/best.pt
selon leur score de performance ('fitness') et copie le meilleur vers
models/detection/best.pt.

Usage :
    python models/detection/scripts/04_promote_best.py
    python models/detection/scripts/04_promote_best.py --run marque_nasale_v1_run2
"""

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEST = ROOT / "best.pt"


def get_checkpoint_fitness(weights_path: Path) -> float:
    """Lit le score de performance ('fitness') stocké dans le checkpoint."""
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
        return float("-inf")
    try:
        return float(fitness)
    except (TypeError, ValueError):
        return float("-inf")


def find_best_overall() -> Path | None:
    """Compare tous les runs et retourne le best.pt le plus performant."""
    candidates = list((ROOT / "runs" / "detect").glob("*/weights/best.pt"))
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]

    print(f"Comparaison de {len(candidates)} run(s) pour trouver le meilleur...")
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
        print("  Aucun score exploitable trouvé, utilisation du run le plus récent à la place.")
        return max(candidates, key=lambda p: p.stat().st_mtime)

    return best_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default=None,
                         help="Nom d'un run précis à promouvoir (ex. marque_nasale_v1_run2). "
                              "Si non précisé : compare tous les runs et prend le meilleur.")
    args = parser.parse_args()

    if args.run:
        source = ROOT / "runs" / "detect" / args.run / "weights" / "best.pt"
        if not source.exists():
            print(f"❌ Run introuvable : {source}")
            raise SystemExit(1)
    else:
        source = find_best_overall()
        if source is None:
            print("❌ Aucun run d'entraînement trouvé sous models/detection/runs/detect/")
            raise SystemExit(1)

    shutil.copy2(source, DEST)
    print(f"✅ Modèle promu : {source.relative_to(ROOT)} → {DEST.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
