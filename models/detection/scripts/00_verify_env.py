"""
Vérifie que l'environnement est prêt : Python, PyTorch, CUDA, Ultralytics.
Lancer en premier, avant tout le reste :
    python scripts/00_verify_env.py
"""

import platform
import sys


def main():
    print("=" * 60)
    print("VÉRIFICATION DE L'ENVIRONNEMENT")
    print("=" * 60)

    print(f"\nPython version : {sys.version}")
    print(f"Plateforme     : {platform.platform()}")

    # --- PyTorch ---
    try:
        import torch
        print(f"\nPyTorch version : {torch.__version__}")
        cuda_ok = torch.cuda.is_available()
        print(f"CUDA disponible : {cuda_ok}")
        if cuda_ok:
            print(f"  GPU détecté    : {torch.cuda.get_device_name(0)}")
        else:
            print("  -> L'entraînement se fera sur CPU (plus lent, mais "
                  "fonctionnel pour un petit dataset / test).")
    except ImportError:
        print("\n❌ PyTorch n'est pas installé.")
        print("   Installe-le avec (CPU uniquement, simple) :")
        print("   pip install torch")
        print("   Ou avec CUDA (GPU NVIDIA), va sur https://pytorch.org/get-started/locally/")
        sys.exit(1)

    # --- Ultralytics ---
    try:
        import ultralytics
        print(f"\nUltralytics version : {ultralytics.__version__}")
    except ImportError:
        print("\n❌ Le package 'ultralytics' n'est pas installé.")
        print("   Installe-le avec : pip install ultralytics")
        sys.exit(1)

    # --- OpenCV ---
    try:
        import cv2
        print(f"OpenCV version      : {cv2.__version__}")
    except ImportError:
        print("\n⚠️  OpenCV non installé (pip install opencv-python) — "
              "pas bloquant pour l'entraînement, utile pour le prétraitement.")

    print("\n" + "=" * 60)
    print("✅ Environnement prêt.")
    print("=" * 60)


if __name__ == "__main__":
    main()
