#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gestion des versions des 4 IA du projet (détection, couleur, espèce, sexe).

Chaque entraînement (fait sur ton PC ou sur Colab) est enregistré comme une
version numérotée et datée, avec ses métriques et des notes -> plus jamais
de "c'est quoi ce best.pt déjà, il vient d'où ?" un mois plus tard.

Structure créée sous <repo>/models/<ia>/ :

    models/<ia>/
        modele.ext              <- version ACTIVE (celle utilisée par le pipeline)
        classes.json            <- (couleur/espece/sexe uniquement)
        active_version.json     <- pointe vers la version actuellement promue
        versions/
            v1_2026-07-29/
                modele.ext
                classes.json     (si applicable)
                metrics.json     (métriques + notes + dataset utilisé)
            v2_2026-08-03/
                ...

Workflow typique après un entraînement Colab :

    1. Télécharge le(s) fichier(s) du notebook (best.pt, ou modele_xxx.pt +
       classes_xxx.json).
    2. Enregistre-les comme nouvelle version :

         python scripts/gestion_modeles.py register --ia couleur \\
             --model ~/Downloads/modele_couleur.pt \\
             --classes ~/Downloads/classes_couleur.json \\
             --val-acc 0.91 --val-loss 0.23 \\
             --notes "340 crops GT, triage bleu/marron/indeterminable"

    3. Si les métriques sont bonnes, promeus cette version comme officielle :

         python scripts/gestion_modeles.py promote --ia couleur --latest

    4. Regarde l'historique à tout moment :

         python scripts/gestion_modeles.py list
         python scripts/gestion_modeles.py list --ia couleur
         python scripts/gestion_modeles.py compare --ia couleur --v1 v1_2026-07-29 --v2 v2_2026-08-03

Les IA "classification" (couleur/espece/sexe) ont un fichier classes.json
(ordre des classes, indispensable pour que l'inférence retombe juste).
L'IA "detection" n'en a pas (YOLO gère ses classes via data.yaml).
"""
import argparse
import json
import re
import shutil
from datetime import date
from pathlib import Path

# scripts/gestion_modeles.py -> REPO_ROOT = <repo>
REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_ROOT = REPO_ROOT / "models"

IA_CONFIG = {
    "detection": {"model_file": "best.pt", "classes_file": None},
    "couleur": {"model_file": "modele_couleur.pt", "classes_file": "classes_couleur.json"},
    "espece": {"model_file": "modele_espece.pt", "classes_file": "classes_espece.json"},
    "sexe": {"model_file": "modele_sexe.pt", "classes_file": "classes_sexe.json"},
}

METRIC_ARGS = ["val_acc", "val_loss", "precision", "recall", "map50", "map"]


def ia_dir(ia: str) -> Path:
    return MODELS_ROOT / ia


def versions_dir(ia: str) -> Path:
    return ia_dir(ia) / "versions"


def list_versions(ia: str):
    vdir = versions_dir(ia)
    if not vdir.exists():
        return []
    return sorted(
        (p for p in vdir.iterdir() if p.is_dir() and re.match(r"v\d+_", p.name)),
        key=lambda p: (lambda m: int(m.group(1)) if m else 0)(re.search(r"v(\d+)_", p.name)),
    )


def next_version_name(ia: str) -> str:
    existing = list_versions(ia)
    if not existing:
        n = 1
    else:
        match = re.search(r"v(\d+)_", existing[-1].name)
        if match:
            last_n = int(match.group(1))
            n = last_n + 1
        else:
            n = 1
    return f"v{n}_{date.today().isoformat()}"


def active_version_info(ia: str):
    p = ia_dir(ia) / "active_version.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def load_metrics(vpath: Path):
    p = vpath / "metrics.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def fmt_metrics(metrics: dict) -> str:
    keys = [k for k in METRIC_ARGS if metrics.get(k) is not None]
    if not keys:
        return "(pas de métriques enregistrées)"
    return " / ".join(f"{k}={metrics[k]:.3f}" if isinstance(metrics[k], float) else f"{k}={metrics[k]}"
                       for k in keys)


# --------------------------------------------------------------------------
# register
# --------------------------------------------------------------------------

def cmd_register(args):
    if args.ia not in IA_CONFIG:
        print(f"❌ IA inconnue : {args.ia}. Choix possibles : {list(IA_CONFIG)}")
        raise SystemExit(1)

    cfg = IA_CONFIG[args.ia]
    model_src = Path(args.model)
    if not model_src.exists():
        print(f"❌ Fichier modèle introuvable : {model_src}")
        raise SystemExit(1)

    if cfg["classes_file"] and not args.classes:
        print(f"❌ --classes requis pour l'IA '{args.ia}' (classifieur : il faut l'ordre des classes).")
        raise SystemExit(1)
    classes_src = Path(args.classes) if args.classes else None
    if classes_src and not classes_src.exists():
        print(f"❌ Fichier classes introuvable : {classes_src}")
        raise SystemExit(1)

    version_name = next_version_name(args.ia)
    vdir = versions_dir(args.ia) / version_name
    vdir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(model_src, vdir / cfg["model_file"])
    if classes_src:
        shutil.copy2(classes_src, vdir / cfg["classes_file"])
        # scripts/utils/pipeline.py cherche en priorité "<model>.classes.json"
        # (convention de scripts/train_classification_ia.py). On duplique donc
        # le fichier de classes sous ce nom pour que les modèles enregistrés
        # via Colab + register/promote soient chargés sans avertissement de
        # fallback, quelle que soit l'IA utilisée.
        classes_json_path = (vdir / cfg["model_file"]).with_suffix(".classes.json")
        if classes_json_path != vdir / cfg["classes_file"]:
            shutil.copy2(classes_src, classes_json_path)

    metrics = {}
    if args.metrics_file:
        metrics.update(json.loads(Path(args.metrics_file).read_text(encoding="utf-8")))
    for key in METRIC_ARGS:
        val = getattr(args, key.replace("-", "_"))
        if val is not None:
            metrics[key] = val

    metrics["date"] = date.today().isoformat()
    metrics["notes"] = args.notes or ""
    metrics["dataset"] = args.dataset_note or ""
    metrics["source_model_file"] = model_src.name

    (vdir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"✅ Version enregistrée : models/{args.ia}/versions/{version_name}/")
    print(f"   {fmt_metrics(metrics)}")
    print(f"\nPour l'activer :  python scripts/gestion_modeles.py promote --ia {args.ia} --version {version_name}")


# --------------------------------------------------------------------------
# promote
# --------------------------------------------------------------------------

def cmd_promote(args):
    if args.ia not in IA_CONFIG:
        print(f"❌ IA inconnue : {args.ia}. Choix possibles : {list(IA_CONFIG)}")
        raise SystemExit(1)
    cfg = IA_CONFIG[args.ia]

    if args.latest:
        versions = list_versions(args.ia)
        if not versions:
            print(f"❌ Aucune version enregistrée pour '{args.ia}'. Utilise 'register' d'abord.")
            raise SystemExit(1)
        vdir = versions[-1]
    else:
        if not args.version:
            print("❌ Précise --version <nom> ou --latest.")
            raise SystemExit(1)
        vdir = versions_dir(args.ia) / args.version
        if not vdir.exists():
            print(f"❌ Version introuvable : {vdir}")
            raise SystemExit(1)

    dest_dir = ia_dir(args.ia)
    dest_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(vdir / cfg["model_file"], dest_dir / cfg["model_file"])
    if cfg["classes_file"]:
        shutil.copy2(vdir / cfg["classes_file"], dest_dir / cfg["classes_file"])
        classes_json_src = (vdir / cfg["model_file"]).with_suffix(".classes.json")
        classes_json_dest = (dest_dir / cfg["model_file"]).with_suffix(".classes.json")
        if classes_json_src.exists():
            shutil.copy2(classes_json_src, classes_json_dest)

    (dest_dir / "active_version.json").write_text(
        json.dumps({"version": vdir.name, "promoted_on": date.today().isoformat()},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    metrics = load_metrics(vdir)
    print(f"✅ Version promue comme officielle : {vdir.name} -> models/{args.ia}/{cfg['model_file']}")
    print(f"   {fmt_metrics(metrics)}")


# --------------------------------------------------------------------------
# list
# --------------------------------------------------------------------------

def cmd_list(args):
    ias = [args.ia] if args.ia else list(IA_CONFIG)
    for ia in ias:
        print(f"\n=== {ia} ===")
        active = active_version_info(ia)
        active_name = active["version"] if active else None
        versions = list_versions(ia)
        if not versions:
            print("  (aucune version enregistrée)")
            continue
        for vdir in versions:
            metrics = load_metrics(vdir)
            marker = "★ ACTIVE" if vdir.name == active_name else "        "
            notes = f" — {metrics['notes']}" if metrics.get("notes") else ""
            print(f"  {marker}  {vdir.name:22s} {fmt_metrics(metrics)}{notes}")


# --------------------------------------------------------------------------
# compare
# --------------------------------------------------------------------------

def cmd_compare(args):
    v1_dir = versions_dir(args.ia) / args.v1
    v2_dir = versions_dir(args.ia) / args.v2
    for v in (v1_dir, v2_dir):
        if not v.exists():
            print(f"❌ Version introuvable : {v}")
            raise SystemExit(1)

    m1, m2 = load_metrics(v1_dir), load_metrics(v2_dir)
    keys = sorted(set(m1) | set(m2) - {"notes", "dataset", "source_model_file"})
    print(f"\n{'métrique':15s} {args.v1:>15s} {args.v2:>15s}")
    print("-" * 48)
    for k in keys:
        if k in ("notes", "dataset", "source_model_file"):
            continue
        v1val = m1.get(k, "-")
        v2val = m2.get(k, "-")
        print(f"{k:15s} {str(v1val):>15s} {str(v2val):>15s}")


# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_reg = sub.add_parser("register", help="Enregistre un nouveau modèle entraîné comme nouvelle version.")
    p_reg.add_argument("--ia", required=True, choices=list(IA_CONFIG))
    p_reg.add_argument(
        "--model", required=True,
        help="Chemin vers le fichier modèle téléchargé (best.pt / modele_xxx.pt)",
    )
    p_reg.add_argument(
        "--classes", default=None,
        help="Chemin vers classes_xxx.json (requis sauf pour detection)",
    )
    p_reg.add_argument(
        "--notes", default=None,
        help="Note libre (ex. contexte, changements par rapport à la version précédente)",
    )
    p_reg.add_argument(
        "--dataset-note", default=None,
        help="Note sur le dataset utilisé (ex. 'raw_data corrigé 340/1815, seed42')",
    )
    p_reg.add_argument("--metrics-file", default=None, help="Fichier JSON de métriques additionnelles à fusionner")
    for key in METRIC_ARGS:
        p_reg.add_argument(f"--{key.replace('_', '-')}", type=float, default=None)
    p_reg.set_defaults(func=cmd_register)

    p_prom = sub.add_parser("promote", help="Active une version comme modèle officiel du pipeline.")
    p_prom.add_argument("--ia", required=True, choices=list(IA_CONFIG))
    p_prom.add_argument("--version", default=None, help="Nom de version (ex. v2_2026-08-03)")
    p_prom.add_argument("--latest", action="store_true", help="Prend automatiquement la version la plus récente")
    p_prom.set_defaults(func=cmd_promote)

    p_list = sub.add_parser("list", help="Liste les versions enregistrées (toutes IA, ou une seule avec --ia).")
    p_list.add_argument("--ia", default=None, choices=list(IA_CONFIG))
    p_list.set_defaults(func=cmd_list)

    p_cmp = sub.add_parser("compare", help="Compare les métriques de deux versions d'une même IA.")
    p_cmp.add_argument("--ia", required=True, choices=list(IA_CONFIG))
    p_cmp.add_argument("--v1", required=True)
    p_cmp.add_argument("--v2", required=True)
    p_cmp.set_defaults(func=cmd_compare)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()