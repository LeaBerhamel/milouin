#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Archive le contenu actuel de output/ (avec un horodatage) puis recrée un
dossier output/ vide, prêt pour un nouveau traitement.

À lancer AVANT de traiter un nouveau lot de photos (Lancer.command /
02_process_rafales.py) : plus besoin de choisir un dossier de sortie
différent à chaque fois dans l'interface graphique, ni de gérer soi-même
des noms de "lots" -- output/ reste toujours le dossier par défaut, et rien
n'est jamais perdu (tout ancien contenu est déplacé, daté, dans
output_archives/).

⚠️ Le workflow ci-dessous est pour un usage EN LIGNE DE COMMANDE pur
(Option B du README). Si tu passes par l'interface graphique
(Lancer.bat/Lancer.command), tout ceci est fait AUTOMATIQUEMENT à chaque
nouveau lot traité -- dictionnaire d'identification ET corrections.csv --
dans un dossier maître séparé (output_archives/dictionnaire_maitre/, jamais
archivé lui-même) : voir interface_graphique.py,
_fusionner_source_dans_maitre() / _rattraper_archives_non_fusionnees().
Rien à lancer à la main dans ce cas, y compris pour retrouver plus tard les
corrections d'un lot déjà archivé dans "🔁 Réentraîner depuis les
corrections…" (interface_preparation_ia.py) -- son champ "Fichier de
corrections" pointe déjà sur ce maître par défaut.

Workflow recommandé pour ajouter un nouveau lot de photos (ligne de commande) :

    1. python scripts/01_archiver_resultats.py
       -> archive le output/ existant dans output_archives/<date>/
       -> recrée un output/ vide

    2. Traiter les nouvelles photos comme d'habitude (Lancer.command ou
       02_process_rafales.py), TOUJOURS avec le dossier de sortie par
       défaut "output/" -- rien à changer dans l'interface graphique.

    3. python scripts/03_export_json_dictionnaire.py
       -> dictionnaire du NOUVEAU lot seul, écrit dans output/ (comme
          d'habitude, puisque output/ est vide au départ)

    4. python scripts/07_fusionner_dictionnaire.py \\
           --nouveau output_archives/<date>/dictionnaire_identifications.json
       -> fusionne l'ANCIEN dictionnaire (celui qui vient d'être archivé)
          dans le NOUVEAU (output/dictionnaire_identifications.json, qui
          devient ainsi le dictionnaire maître à jour, mai + juin + ...)

    5. python scripts/15_fusionner_corrections.py \\
           --source output_archives/<date> --maitre output
       -> même chose pour les corrections manuelles (espèce/sexe/couleur) :
          fusionne output_archives/<date>/corrections.csv (et les crops
          qu'il référence) dans output/corrections.csv, pour que
          09_ecran_identification.py / "Réentraîner depuis les
          corrections…" les retrouvent aussi, sans perdre celles du lot
          qui vient d'être archivé. Sans effet si le lot archivé n'a aucune
          correction (rien à fusionner).

    6. python scripts/04_copier_crops_valides.py
    7. python scripts/05_construire_banque_reference.py

Le chemin exact de l'archive à utiliser aux étapes 4 et 5 est affiché par
ce script au moment de l'archivage -- pas besoin de le retenir par cœur,
juste copier-coller la ligne affichée.

Usage:
    python scripts/01_archiver_resultats.py
    python scripts/01_archiver_resultats.py --output un_autre_dossier
"""

import argparse
import logging
import shutil
from datetime import datetime
from pathlib import Path


def _liberer_handlers_de_fichier(dossier: Path) -> list[dict]:
    """Ferme et détache tout logging.FileHandler dont le fichier vit sous
    `dossier` (ex. output/pipeline_production.log, voir utils/config.py qui
    l'ouvre et le garde ouvert pendant toute la session de l'appli).

    Nécessaire avant un shutil.move()/rmtree() : sur Windows (contrairement
    à Mac/Linux), un fichier encore ouvert par le processus ne peut être ni
    déplacé ni supprimé -- shutil.move copie tout output/ avec succès, puis
    plante précisément sur ce fichier-là en tentant de supprimer l'original
    (PermissionError: [WinError 32]), laissant un output/ résiduel non
    nettoyé derrière lui.

    Retourne la liste des handlers retirés (avec leur logger d'origine et
    leur formatter), pour pouvoir les rouvrir ailleurs après le déplacement
    -- voir _rouvrir_handlers_de_fichier ci-dessous.
    """
    dossier = dossier.resolve()
    retires = []
    # On regarde tous les loggers connus, pas seulement le root logger : peu
    # coûteux (quelques dizaines d'entrées max) et robuste si un jour un
    # logger nommé a son propre FileHandler plutôt que d'hériter du root.
    loggers = [logging.getLogger()] + [
        logging.getLogger(nom) for nom in logging.root.manager.loggerDict
    ]
    for lg in loggers:
        for handler in list(lg.handlers):
            if not isinstance(handler, logging.FileHandler):
                continue
            try:
                chemin_handler = Path(handler.baseFilename).resolve()
            except (AttributeError, OSError):
                continue
            if dossier in chemin_handler.parents or chemin_handler == dossier:
                lg.removeHandler(handler)
                retires.append({
                    "logger": lg, "formatter": handler.formatter,
                    "level": handler.level, "chemin_relatif": chemin_handler.relative_to(dossier),
                })
                handler.close()
    return retires


def _rouvrir_handlers_de_fichier(retires: list[dict], nouveau_dossier: Path) -> None:
    """Recrée, au même endroit relatif sous nouveau_dossier (le output/
    fraîchement vidé), les FileHandler retirés par
    _liberer_handlers_de_fichier -- pour que le logging continue à
    fonctionner sans avoir à relancer l'appli après un archivage."""
    for info in retires:
        chemin = nouveau_dossier / info["chemin_relatif"]
        chemin.parent.mkdir(parents=True, exist_ok=True)
        nouveau_handler = logging.FileHandler(chemin, encoding="utf-8")
        nouveau_handler.setFormatter(info["formatter"])
        nouveau_handler.setLevel(info["level"])
        info["logger"].addHandler(nouveau_handler)


def archiver(output_dir: Path, archives_dir: Path = None):
    """Déplace le contenu de output_dir vers archives_dir/<horodatage>/ et
    recrée output_dir vide. Retourne le chemin (Path) de la destination,
    ou None si output_dir n'existait pas ou était déjà vide (rien à
    archiver -- output_dir est quand même (re)créé dans ce cas).

    Extrait de main() pour être appelable depuis l'interface graphique
    (bouton "🗄️ Archiver" du rapport HTML, voir interface_graphique.py et
    module_corrections/serveur_corrections.py) sans passer par une
    sous-commande CLI.

    Si archives_dir n'est PAS précisé, il est calculé à côté de output_dir
    (output_dir.resolve().parent / "output_archives"), jamais comme un
    chemin relatif au dossier de travail courant (cwd). C'est important en
    mode packagé (SuiviCanards.exe, voir packaging/SuiviCanards.spec) :
    Windows démarre un .exe lancé par double-clic avec le dossier de
    l'exécutable comme dossier de travail -- PAS la racine du projet. Un
    ancien défaut Path("output_archives") (relatif) atterrissait donc dans
    SuiviCanards/output_archives/ au lieu de la racine du projet, à côté de
    data/models/output/ -- invisible pour tout le reste de l'application,
    qui cherche les archives à côté de output_dir (voir
    interface_graphique.py, _archiver_resultats_courants /
    _fusionner_source_dans_maitre, qui utilisent déjà ce même calcul).
    """
    output_dir = Path(output_dir)
    if archives_dir is None:
        archives_dir = output_dir.resolve().parent / "output_archives"
    archives_dir = Path(archives_dir)

    if not output_dir.exists() or not any(output_dir.iterdir()):
        output_dir.mkdir(parents=True, exist_ok=True)
        return None

    horodatage = datetime.now().strftime("%Y-%m-%d_%Hh%M")
    destination = archives_dir / horodatage

    # Si on relance deux fois la même minute (rare), on ajoute un suffixe
    # plutôt que d'écraser une archive existante.
    suffixe = 1
    while destination.exists():
        suffixe += 1
        destination = archives_dir / f"{horodatage}_{suffixe}"

    archives_dir.mkdir(parents=True, exist_ok=True)
    handlers_a_rouvrir = _liberer_handlers_de_fichier(output_dir)
    try:
        shutil.move(str(output_dir), str(destination))
    finally:
        # Qu'on ait réussi ou non : output_dir doit être recréé si besoin
        # (soit vide après un déplacement réussi, soit parce qu'un échec
        # partiel de rmtree peut l'avoir laissé en piteux état) et le
        # logging fichier doit reprendre dans tous les cas -- jamais de
        # session où le logger reste silencieusement sans FileHandler.
        output_dir.mkdir(parents=True, exist_ok=True)
        _rouvrir_handlers_de_fichier(handlers_a_rouvrir, output_dir)

    return destination


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", default="output",
                         help="Dossier de résultats à archiver (défaut : output)")
    parser.add_argument("--archives", default=None,
                         help="Dossier où ranger les archives datées "
                              "(défaut : output_archives/ à côté de --output)")
    args = parser.parse_args()

    output_dir = Path(args.output)
    archives_dir = Path(args.archives) if args.archives else None

    destination = archiver(output_dir, archives_dir)

    if destination is None:
        print(f"ℹ️  {output_dir}/ n'existe pas ou est déjà vide -- rien à archiver.")
        return

    print(f"✓ Ancien contenu de '{args.output}/' archivé -> {destination}/")
    print(f"✓ Nouveau dossier '{args.output}/' vide, prêt pour le prochain traitement.")
    print()

    dict_archive = destination / "dictionnaire_identifications.json"
    if dict_archive.exists():
        print("📌 Pour ne pas perdre les identifications précédentes : une fois le")
        print("   nouveau lot traité (Lancer.command) et exporté (script 03), fusionne")
        print("   l'ancien dictionnaire archivé dans le nouveau avec :")
        print()
        print(f"   python scripts/07_fusionner_dictionnaire.py --nouveau {dict_archive}")
    else:
        print("ℹ️  Aucun dictionnaire_identifications.json trouvé dans l'archive "
              "(aucune identification n'avait encore été exportée) -- rien à fusionner.")

    print()

    corrections_archive = destination / "corrections.csv"
    if corrections_archive.exists():
        print("📌 Idem pour les corrections manuelles (espèce/sexe/couleur) faites sur ce lot :")
        print()
        print(f"   python scripts/15_fusionner_corrections.py --source {destination} --maitre {output_dir}")
    else:
        print("ℹ️  Aucun corrections.csv trouvé dans l'archive "
              "(aucune correction manuelle n'avait encore été faite sur ce lot) -- rien à fusionner.")


if __name__ == "__main__":
    main()