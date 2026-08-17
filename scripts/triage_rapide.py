#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Triage rapide des crops marque_nasale par couleur (2 classes : bleu / marron).

Affiche chaque crop du dataset_couleur (celui sans "couleur" encore renseignée),
attend une touche, et met à jour metadata.json au fur et à mesure -> reprenable
à tout moment (relance le script, il continue où tu t'es arrêté).

Touches :
    b -> bleu
    m -> marron
    n -> non identifiable (ex : photo de nuit, floue, marque non visible) -> réponse définitive, ne réapparaît plus
    s -> incertain / à revoir plus tard (laisse couleur telle quelle, passe au
         suivant, réapparaîtra au prochain lancement)
    r -> retour arrière (revoir/corriger le(s) précédent(s), autant de fois que voulu)
    q -> quitter (sauvegarde et s'arrête)

Pour corriger UN SEUL crop après avoir quitté le script (sans tout réinitialiser) :
    mets manuellement sa "couleur" à null dans metadata.json, il réapparaîtra
    automatiquement au prochain lancement.

Réinitialiser TOUT le triage (efface toutes les couleurs déjà assignées) :
    python scripts/triage_rapide.py --dataset dataset_couleur --reset

Usage :
    python scripts/triage_rapide.py --dataset dataset_couleur
"""
import argparse
import json
from pathlib import Path

COLOR_KEYS = {"b": "bleu", "m": "marron", "n": "indeterminable"}
VALID_CHOICES = set(COLOR_KEYS) | {"s", "r", "q"}


def load_metadata(meta_path: Path):
    return json.loads(meta_path.read_text(encoding="utf-8"))


def save_metadata(meta_path: Path, metadata):
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


def do_reset(meta_path: Path, metadata):
    n_colored = sum(1 for e in metadata if e.get("couleur"))
    print(f"⚠️  Ça va effacer les {n_colored} couleur(s) déjà assignée(s) sur {len(metadata)} crops.")
    confirm = input("   Tape OUI (en majuscules) pour confirmer, autre chose pour annuler : ").strip()
    if confirm != "OUI":
        print("Annulé, rien n'a été modifié.")
        return
    for e in metadata:
        e["couleur"] = None
    save_metadata(meta_path, metadata)
    print(f"✅ Réinitialisé : les {len(metadata)} crops sont à nouveau à trier.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="dataset_couleur",
                         help="Dossier contenant crops/ et metadata.json")
    parser.add_argument("--reset", action="store_true",
                         help="Efface toutes les couleurs déjà triées et quitte (pas de triage dans ce mode).")
    args = parser.parse_args()

    base = Path(args.dataset)
    meta_path = base / "metadata.json"
    crops_dir = base / "crops"

    if not meta_path.exists():
        print(f"❌ Introuvable : {meta_path}")
        return

    metadata = load_metadata(meta_path)

    if args.reset:
        do_reset(meta_path, metadata)
        return

    todo = [e for e in metadata if not e.get("couleur")]
    print(f"{len(metadata)} crops au total, {len(todo)} restant(s) à trier.\n")

    if not todo:
        print("Rien à trier. (Utilise --reset si tu veux tout recommencer.)")
        return

    try:
        from PIL import Image
    except ImportError:
        print("❌ Pillow requis : pip install pillow")
        return

    n_done_this_session = 0
    i = 0
    while i < len(todo):
        entry = todo[i]
        crop_path = crops_dir / entry["crop_name"]
        if not crop_path.exists():
            print(f"⚠️  Fichier manquant, ignoré : {crop_path}")
            i += 1
            continue

        img = Image.open(crop_path)
        img.show()  # ouvre dans la visionneuse par défaut du système

        statut = f"  (actuellement : {entry['couleur']})" if entry.get("couleur") else ""
        while True:
            choice = input(
                f"[{i + 1}/{len(todo)}] [{entry['crop_name']}]{statut}  "
                f"(b)leu / (m)arron / (n)on identifiable / (s)kip / (r)etour / (q)uitter : "
            ).strip().lower()
            if choice in VALID_CHOICES:
                break
            print("  -> touche non reconnue, réessaie.")

        if choice == "q":
            break
        elif choice in COLOR_KEYS:
            entry["couleur"] = COLOR_KEYS[choice]
            n_done_this_session += 1
            save_metadata(meta_path, metadata)
            i += 1
        elif choice == "s":
            i += 1  # ne touche pas à couleur, passe au suivant
        elif choice == "r":
            i = max(0, i - 1)  # revoir le(s) précédent(s) ; b/m écrase, s repasse devant sans changer

    n_bleu = sum(1 for e in metadata if e.get("couleur") == "bleu")
    n_marron = sum(1 for e in metadata if e.get("couleur") == "marron")
    n_indeterminable = sum(1 for e in metadata if e.get("couleur") == "indeterminable")
    n_restant = sum(1 for e in metadata if not e.get("couleur"))
    print(f"\n✅ Session terminée : {n_done_this_session} crop(s) trié(s)/corrigé(s) cette fois-ci.")
    print(
        f"   Total : {n_bleu} bleu / {n_marron} marron / "
        f"{n_indeterminable} non identifiable / {n_restant} restant(s) à trier."
    )


if __name__ == "__main__":
    main()