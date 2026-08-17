"""
Utilise le modèle entraîné pour détecter la marque nasale sur une image
(ou un dossier d'images) et affiche/sauvegarde les boîtes englobantes.

Usage :
    python scripts/03_predict.py --source chemin/vers/photo.jpg
    python scripts/03_predict.py --source chemin/vers/dossier/ --conf 0.4
"""

import argparse
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True,
                         help="Image, dossier d'images, ou vidéo à analyser.")
    parser.add_argument("--weights", default=None,
                         help="Chemin vers best.pt. Par défaut : dernier run d'entraînement trouvé.")
    parser.add_argument("--conf", type=float, default=0.25,
                         help="Seuil de confiance minimum.")
    parser.add_argument("--imgsz", type=int, default=960)
    args = parser.parse_args()

    weights = args.weights
    if weights is None:
        # cherche le run le plus récent
        candidates = sorted((ROOT / "runs" / "detect").glob("*/weights/best.pt"),
                             key=lambda p: p.stat().st_mtime, reverse=True)
        if not candidates:
            print("❌ Aucun modèle entraîné trouvé sous runs/detect/*/weights/best.pt")
            print("   Lance d'abord scripts/02_train.py, ou précise --weights.")
            return
        weights = str(candidates[0])
        print(f"Utilisation du modèle le plus récent : {weights}")

    model = YOLO(weights)

    results = model.predict(
        source=args.source,
        conf=args.conf,
        imgsz=args.imgsz,
        save=True,          # sauvegarde les images annotées
        save_txt=True,      # sauvegarde aussi les coordonnées en .txt (format YOLO)
        save_conf=True,
    )

    for r in results:
        print(f"\n{r.path}")
        if r.boxes is None or len(r.boxes) == 0:
            print("  Aucune marque nasale détectée.")
            continue
        for box in r.boxes:
            xyxy = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            print(f"  Boîte (x1,y1,x2,y2) = "
                  f"({xyxy[0]:.1f}, {xyxy[1]:.1f}, {xyxy[2]:.1f}, {xyxy[3]:.1f})  "
                  f"conf={conf:.2f}")

    print("\n✅ Résultats annotés sauvegardés sous : runs/detect/predict*/")


if __name__ == "__main__":
    main()
