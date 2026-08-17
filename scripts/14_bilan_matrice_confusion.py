# -*- coding: utf-8 -*-
"""
Bilan de production : matrice de confusion espèce/sexe/couleur, construite
en comparant ce que les 3 IA de classification avaient détecté
(resultats_par_rafale.json) à ce qu'un chercheur a effectivement corrigé
depuis le rapport HTML (dictionnaire/corrections.csv -- voir
module_corrections/ecrire_corrections.py). Toute la logique vit dans
utils/confusion.py.

⚠️ Ne couvre QUE les cas relus ET corrigés -- ce n'est pas un taux
d'exactitude global de l'IA en production, voir la docstring de
utils/confusion.py pour le détail de cette limite. Utile pour repérer les
confusions récurrentes (ex. "milouin souvent corrigé en morillon") avant
de lancer un réentraînement ciblé via interface_preparation_ia.py.

Usage :
    python 14_bilan_matrice_confusion.py --output <output_dir>

<output_dir> est le dossier passé en --output à 02_process_rafales.py --
celui qui contient resultats_par_rafale.json. S'il contient aussi
directement corrections.csv (organisation standard, voir
interface_graphique.py::dictionnaire_dir_courant == output_dir_courant),
--dictionnaire n'est pas nécessaire.

Écrit dans <output_dir> :
    matrice_confusion_espece.csv
    matrice_confusion_sexe.csv
    matrice_confusion_couleur.csv
    matrice_confusion.png   (un panneau par classe qui a des données)
(seules les classes avec au moins une confusion enregistrée produisent
un fichier -- aucun fichier n'est écrit si rien n'est encore exploitable).
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils.confusion import (
    construire_matrices_confusion,
    construire_bilan_exactitude,
    construire_matrice_marque_nasale,
    construire_bilan_detection_canards,
    analyser_seuil_confiance,
    ecrire_matrice_csv,
    exporter_graphique_confusion,
)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", required=True,
                         help="Dossier output_dir (contient resultats_par_rafale.json)")
    parser.add_argument("--dictionnaire", default=None,
                         help="Dossier contenant corrections.csv, si différent de --output "
                              "(défaut : --output)")
    args = parser.parse_args()

    output_dir = Path(args.output)
    dossier_dictionnaire = Path(args.dictionnaire) if args.dictionnaire else output_dir

    resultats_path = output_dir / "resultats_par_rafale.json"
    if not resultats_path.exists():
        print(f"Introuvable : {resultats_path}")
        sys.exit(1)

    with open(resultats_path, encoding="utf-8") as f:
        resultats = json.load(f)

    matrices, stats = construire_matrices_confusion(resultats, dossier_dictionnaire)

    print(f"Corrections lues dans corrections.csv : {stats['nb_corrections_lues']}")
    print(f"  dont exploitées (crop retrouvé dans resultats_par_rafale.json) : {stats['nb_exploitees']}")
    print(f"  dont ignorées (pas de correspondance, ou aucune valeur corrigée) : "
          f"{stats['nb_ignorees_pas_de_correspondance']}")
    print()

    if not any(matrices.values()):
        print("Aucune confusion espèce/sexe/couleur exploitable pour l'instant -- pas encore de "
              "correction enregistrée, ou aucune ne correspond à une rafale connue de ce "
              "resultats_par_rafale.json. (Les bilans de détection marque nasale/canard "
              "ci-dessous sont indépendants et peuvent quand même contenir des données.)")
    else:
        for classe, matrice in matrices.items():
            if not matrice:
                continue
            print(f"--- {classe.upper()} ---")
            for predit in sorted(matrice):
                for corrige, count in sorted(matrice[predit].items(), key=lambda kv: -kv[1]):
                    print(f"  IA a dit « {predit} »  ->  corrigé en « {corrige} »  : {count} fois")
            chemin_csv = ecrire_matrice_csv(matrice, output_dir / f"matrice_confusion_{classe}.csv")
            print(f"  -> {chemin_csv}")
            print()

        chemin_png = exporter_graphique_confusion(matrices, output_dir / "matrice_confusion.png")
        if chemin_png:
            print(f"Graphique : {chemin_png}")

        # Bilan d'exactitude par classe : s'appuie sur
        # dictionnaire/revues_classification.json (écrit à CHAQUE validation du
        # formulaire, changement ou non -- voir ecrire_corrections.py), qui
        # donne le dénominateur qui manquait à la matrice ci-dessus. Absent
        # (taux_exactitude=None partout) pour un lot traité avant l'ajout de
        # cette fonctionnalité.
        bilan = construire_bilan_exactitude(resultats, dossier_dictionnaire, matrices=matrices)
        print("\n--- TAUX D'EXACTITUDE PAR CLASSE (parmi les cas relus) ---")
        for classe, stats in bilan.items():
            if stats["nb_revus"] == 0:
                print(f"  {classe} : aucune revue enregistrée pour l'instant "
                      f"(dictionnaire/revues_classification.json vide ou absent)")
                continue
            pct = stats["taux_exactitude"] * 100
            print(f"  {classe} : {pct:.1f}% correct sur {stats['nb_revus']} cas relus "
                  f"({stats['nb_corriges']} corrigés)")

        # Croisement confiance <-> corrections : utile pour fixer un seuil de
        # relecture systématique (ex. "toujours relire sous 70%").
        analyse_confiance = analyser_seuil_confiance(resultats, dossier_dictionnaire)
        if analyse_confiance["buckets"]:
            print("\n--- EXACTITUDE PAR TRANCHE DE CONFIANCE (parmi les cas relus) ---")
            for b in analyse_confiance["buckets"]:
                pct = b["taux_exactitude"] * 100
                print(f"  [{b['borne_basse']*100:.0f}-{b['borne_haute']*100:.0f}%) : "
                      f"{pct:.1f}% correct sur {b['nb_revus']} cas ({b['nb_corriges']} corrigés)")
            if analyse_confiance["nb_sans_confiance"]:
                print(f"  ({analyse_confiance['nb_sans_confiance']} cas relus sans confiance "
                      f"connue, exclus de ce découpage)")

    # Détection de marque nasale (présente/absente) -- distinct de la
    # classification couleur/espèce/sexe d'une marque déjà détectée.
    # Couvre TOUS les cas relus (changés ou non), voir la docstring de
    # construire_matrice_marque_nasale.
    marque_nasale = construire_matrice_marque_nasale(dossier_dictionnaire)
    print("\n--- DÉTECTION DE MARQUE NASALE (vrai/faux positif/négatif) ---")
    if marque_nasale["nb_revus"] == 0:
        print("  Aucune revue enregistrée pour l'instant "
              "(dictionnaire/revues_marque_nasale.json vide ou absent).")
    else:
        print(f"  Vrais positifs (marque détectée, confirmée)      : {marque_nasale['vrai_positif']}")
        print(f"  Faux positifs  (marque détectée à tort)          : {marque_nasale['faux_positif']}")
        print(f"  Faux négatifs  (marque manquée par l'IA)         : {marque_nasale['faux_negatif']}")
        print(f"  Vrais négatifs (absence confirmée)               : {marque_nasale['vrai_negatif']}")
        print(f"  Précision : {marque_nasale['precision']*100:.1f}%  |  "
              f"Rappel : {marque_nasale['rappel']*100:.1f}%  |  "
              f"Exactitude : {marque_nasale['exactitude']*100:.1f}%  "
              f"(sur {marque_nasale['nb_revus']} cas relus)")

    # Détection de canard elle-même (le cadre YOLO) -- précision/rappel
    # seulement, pas de vrai négatif défini pour une tâche de détection.
    detection_canards = construire_bilan_detection_canards(dossier_dictionnaire)
    print("\n--- DÉTECTION DE CANARD (vrai positif/faux positif/faux négatif) ---")
    if detection_canards["precision"] is None and detection_canards["rappel"] is None:
        print("  Aucune donnée pour l'instant (aucun canard confirmé, supprimé ou ajouté "
              "à la main depuis la fenêtre de correction).")
    else:
        print(f"  Vrais positifs (canard confirmé)       : {detection_canards['vrai_positif']}")
        print(f"  Faux positifs  (fausse détection supprimée) : {detection_canards['faux_positif']}")
        print(f"  Faux négatifs  (canard raté par l'IA)  : {detection_canards['faux_negatif']}")
        if detection_canards["precision"] is not None:
            print(f"  Précision : {detection_canards['precision']*100:.1f}%")
        if detection_canards["rappel"] is not None:
            print(f"  Rappel : {detection_canards['rappel']*100:.1f}%")

    print(
        "\n⚠️ Rappel : ces bilans ne couvrent que les cas relus (par choix du chercheur, "
        "souvent les cas déjà suspects) -- pas un échantillon aléatoire de la production. "
        "Voir les docstrings de utils/confusion.py pour le détail de cette limite."
    )


if __name__ == "__main__":
    main()