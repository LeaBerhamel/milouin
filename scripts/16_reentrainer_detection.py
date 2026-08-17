#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Réentraîne le modèle de DÉTECTION (YOLO, canards + marques nasales) à
partir d'un dataset exporté par scripts/13_exporter_annotations_detection.py,
et compare le résultat au modèle actuellement en production
(models/detection/best.pt) sur le même jeu de validation -- pour savoir
objectivement si le nouveau modèle est réellement meilleur avant de le
promouvoir avec scripts/gestion_modeles.py.

Contrairement aux 3 IA de classification (espèce/sexe/couleur, voir
11_reentrainer_modele.py + 12_valider_et_deployer.py), il n'existait
jusqu'ici aucun script du projet pour lancer cet entraînement : seule la
commande `yolo detect train ...` brute était suggérée en fin de
13_exporter_annotations_detection.py, sans comparaison ni intégration au
système de versions (scripts/gestion_modeles.py, qui gère pourtant déjà
"detection" comme un type d'IA à part entière).

Usage :
    # Entraîner à partir du dataset exporté par le script 13
    python scripts/16_reentrainer_detection.py \\
        --data dataset_detection/data.yaml --epochs 50

    # Comparer seulement deux modèles déjà entraînés (sans réentraîner),
    # ex. pour revalider une version après-coup
    python scripts/16_reentrainer_detection.py --comparer-seulement \\
        --nouveau models/detection/versions/v3_2026-08-14/best.pt \\
        --data dataset_detection/data.yaml
"""

import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_DETECTION_ACTUEL = REPO_ROOT / "models" / "detection" / "best.pt"

# Métriques retenues pour juger un modèle de détection -- les équivalents
# YOLO de l'accuracy utilisée pour la classification (12_valider_et_deployer.py).
# map50 (mAP à IoU=0.50) est la métrique la plus lisible/habituelle pour
# juger un détecteur ; les autres sont données à titre indicatif.
METRIQUES_DETECTION = ("map50", "map", "precision", "recall")


def entrainer_detection(data_yaml: Path, epochs: int, imgsz: int, batch: int,
                         device: str, model_depart: Path, project_dir: Path, log=print):
    """Fine-tune model_depart (par défaut le best.pt actuel) sur data_yaml.

    Retourne (chemin_vers_best_pt, métriques) où métriques est un dict
    avec les clés de METRIQUES_DETECTION, extraites des résultats de
    validation qu'ultralytics calcule automatiquement en fin
    d'entraînement (sur le split "val" de data_yaml).
    """
    from ultralytics import YOLO

    log(f"🚀 Entraînement détection à partir de {model_depart.name} "
        f"({epochs} époques, imgsz={imgsz}, batch={batch}, device={device})...")
    model = YOLO(str(model_depart))
    resultats = model.train(
        data=str(data_yaml), epochs=epochs, imgsz=imgsz, batch=batch,
        device=device, project=str(project_dir), name="entrainement", exist_ok=True,
    )
    # results_dict expose des clés du type "metrics/mAP50(B)" -- on les
    # ramène aux noms courts de METRIQUES_DETECTION pour rester cohérent
    # avec val_acc/val_loss côté classification (gestion_modeles.py les
    # affiche indifféremment via fmt_metrics()).
    rd = getattr(resultats, "results_dict", {}) or {}
    metriques = {
        "map50": rd.get("metrics/mAP50(B)", 0.0),
        "map": rd.get("metrics/mAP50-95(B)", 0.0),
        "precision": rd.get("metrics/precision(B)", 0.0),
        "recall": rd.get("metrics/recall(B)", 0.0),
    }
    best_pt = Path(model.trainer.best)
    log(f"✅ Entraînement terminé -- mAP50 = {metriques['map50']:.1%}, "
        f"meilleur modèle : {best_pt}")
    return best_pt, metriques


def evaluer_detection(model_path: Path, data_yaml: Path, device: str = "cpu", log=print) -> dict:
    """Évalue un modèle de détection (n'importe lequel, pas seulement un
    tout juste entraîné) sur le split "val" de data_yaml. Équivalent
    détection de evaluer_modele() dans 12_valider_et_deployer.py.

    Ne modifie ni n'entraîne rien -- utilisé pour comparer deux modèles
    existants sur un pied d'égalité (même jeu de validation)."""
    from ultralytics import YOLO

    if not model_path.exists():
        log(f"⚠️ Modèle introuvable : {model_path} -- évalué comme 0 sur toutes les métriques.")
        return {k: 0.0 for k in METRIQUES_DETECTION}

    model = YOLO(str(model_path))
    resultats = model.val(data=str(data_yaml), device=device, verbose=False)
    return {
        "map50": float(resultats.box.map50),
        "map": float(resultats.box.map),
        "precision": float(resultats.box.mp),
        "recall": float(resultats.box.mr),
    }


def comparer(metriques_nouveau: dict, metriques_actuel: dict) -> str:
    """Tableau texte comparant deux jeux de métriques détection, même
    présentation que gestion_modeles.py::cmd_compare()."""
    lignes = [f"{'métrique':12s} {'nouveau':>10s} {'actuel':>10s}", "-" * 36]
    for k in METRIQUES_DETECTION:
        lignes.append(f"{k:12s} {metriques_nouveau.get(k, 0):>10.1%} {metriques_actuel.get(k, 0):>10.1%}")
    return "\n".join(lignes)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True, help="data.yaml du dataset exporté (script 13).")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="cpu", help='"cpu", "0" (1ère carte graphique), etc.')
    parser.add_argument("--model-depart", default=str(MODEL_DETECTION_ACTUEL),
                         help="Modèle à partir duquel fine-tuner (défaut : le modèle actuel en production).")
    parser.add_argument("--comparer-seulement", action="store_true",
                         help="Ne pas entraîner : juste comparer --nouveau au modèle actuel sur --data.")
    parser.add_argument("--nouveau", default=None,
                         help="Avec --comparer-seulement : chemin du modèle à comparer au modèle actuel.")
    args = parser.parse_args()

    data_yaml = Path(args.data)

    if args.comparer_seulement:
        if not args.nouveau:
            parser.error("--comparer-seulement nécessite --nouveau <chemin_du_modele>")
        m_nouveau = evaluer_detection(Path(args.nouveau), data_yaml, device=args.device)
        m_actuel = evaluer_detection(MODEL_DETECTION_ACTUEL, data_yaml, device=args.device)
        print(comparer(m_nouveau, m_actuel))
        return

    best_pt, metriques_nouveau = entrainer_detection(
        data_yaml=data_yaml, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        device=args.device, model_depart=Path(args.model_depart),
        project_dir=REPO_ROOT / "runs_detection",
    )
    metriques_actuel = evaluer_detection(MODEL_DETECTION_ACTUEL, data_yaml, device=args.device)
    print()
    print(comparer(metriques_nouveau, metriques_actuel))
    print(f"\nPour enregistrer cette version :")
    print(f"   python scripts/gestion_modeles.py register --ia detection --model {best_pt} "
          f"--map50 {metriques_nouveau['map50']:.4f} --notes \"...\"")


if __name__ == "__main__":
    main()
