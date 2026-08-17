# -*- coding: utf-8 -*-
"""
Bilan "matrice de confusion" espèce/sexe/couleur, construit à partir des
corrections manuelles déjà enregistrées depuis le rapport (voir
module_corrections/ecrire_corrections.py::enregistrer_correction_espece_sexe_couleur,
qui écrit dictionnaire/corrections.csv) comparées à ce que les IA de
classification avaient détecté pour ce même canard (resultats_par_rafale.json,
champs canard["espece_detectee"] / ["sexe_detectee"] / ["couleur_detectee"]
-- jamais réécrits par une correction, voir ecrire_corrections.py : les 4
fonctions d'écriture ne modifient QUE les 4 fichiers de dictionnaire/,
jamais resultats_par_rafale.json lui-même -- sauf depuis la fenêtre
"photo + liste" qui, elle, réécrit ce fichier pour les cadres/canards
ajoutés ou supprimés à la main, mais PAS pour une correction espèce/
sexe/couleur, qui ne passe que par corrections.csv).

⚠️ LIMITE METHODOLOGIQUE IMPORTANTE : ce n'est PAS un bilan de précision
globale de l'IA en production. corrections.csv ne contient QUE les cas où
le chercheur a effectivement CHANGÉ une valeur -- enregistrer_correction_
espece_sexe_couleur() n'écrit rien quand le formulaire est validé sans
rien changer (voir son docstring : "uniquement les champs réellement
changés"). Un canard jamais corrigé peut donc être :
  - correctement classifié par l'IA et confirmé tel quel par un chercheur
    (mais cette confirmation n'est enregistrée nulle part pour espèce/
    sexe/couleur -- seule la bague l'est, dans dictionnaire_identifications.json),
  - ou tout simplement jamais relu.
Cette matrice montre donc, PARMI LES SEULS CAS RELUS ET CORRIGÉS, quelles
confusions reviennent le plus souvent (ex. "milouin souvent corrigé en
morillon") -- utile pour prioriser un réentraînement ciblé ou repérer un
biais systématique, mais elle ne permet pas de calculer un taux
d'exactitude ("l'IA se trompe X% du temps") : le dénominateur (nombre de
cas réellement relus, corrects ou non) n'est pas connu.
"""

import csv
from collections import defaultdict
from pathlib import Path

from .crop_export import nom_crop_correction

# classe -> nom de la colonne correspondante dans corrections.csv (voir
# COLONNES_CORRECTIONS dans module_corrections/ecrire_corrections.py --
# "sexe_corrige" est bien sans "e" final, contrairement aux deux autres :
# ne jamais "corriger" cette apparente coquille sans mettre ce fichier ET
# ecrire_corrections.py à jour ensemble).
CLASSES_COLONNE_CSV = {
    "espece": "espece_corrigee",
    "sexe": "sexe_corrige",
    "couleur": "couleur_corrigee",
}


def _index_predictions_par_crop(resultats):
    """{crop_filename: {"espece": ..., "sexe": ..., "couleur": ...}} --
    une entrée par canard de chaque rafale de `resultats` (le contenu de
    resultats_par_rafale.json), avec le NOM DE CROP DÉTERMINISTE que ce
    canard aurait si on le corrigeait (nom_crop_correction -- même
    formule que save_canard_crop, voir utils/report.py::_etat_correction_rafale
    qui fait exactement la même chose pour le badge "Corrigé" du rapport).
    Ne dépend pas de l'existence réelle du fichier crop sur disque."""
    index = {}
    for result in resultats:
        photo_source = result.get("photo_analysee")
        rafale_id = result.get("rafale_id")
        if not photo_source:
            continue
        for canard in result.get("canards", []):
            idx = canard.get("canard_index", 0)
            crop_filename = nom_crop_correction(photo_source, rafale_id, idx)
            index[crop_filename] = {
                "espece": canard.get("espece_detectee"),
                "sexe": canard.get("sexe_detectee"),
                "couleur": canard.get("couleur_detectee"),
            }
    return index


def _lire_corrections_csv(dossier_dictionnaire):
    chemin = Path(dossier_dictionnaire) / "corrections.csv"
    if not chemin.exists():
        return []
    with open(chemin, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _lire_revues_classification(dossier_dictionnaire):
    """Lit dictionnaire/revues_classification.json (voir
    module_corrections/ecrire_corrections.py::enregistrer_revue_classification)
    -- {crop_filename: {"classes_revues": [...], "revu_le": ...}}, écrit à
    CHAQUE validation du formulaire, contrairement à corrections.csv qui ne
    contient que les cas changés. C'est le dénominateur qui manquait pour
    calculer un vrai taux d'exactitude (voir construire_bilan_exactitude).

    Tolérant à l'absence du fichier (lots traités avant l'ajout de cette
    fonctionnalité, ou aucune correction encore faite dessus) : retourne
    {} plutôt que de faire planter le bilan."""
    chemin = Path(dossier_dictionnaire) / "revues_classification.json"
    if not chemin.exists():
        return {}
    import json
    try:
        with open(chemin, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def construire_matrices_confusion(resultats, dossier_dictionnaire):
    """
    Args:
        resultats: contenu de resultats_par_rafale.json (liste de dicts,
            un par rafale -- même format que ce que lit fenetre_rafale.py).
        dossier_dictionnaire: dossier contenant corrections.csv (typiquement
            output_dir, voir interface_graphique.py::dictionnaire_dir_courant).

    Returns:
        (matrices, stats)
        matrices: {"espece": {predit: {corrige: count}}, "sexe": ..., "couleur": ...}
            -- predit vaut "(aucune)" si l'IA n'avait rien détecté du tout
            pour ce canard (canard non marqué, ou marque non assignée).
        stats: {"nb_corrections_lues": int, "nb_exploitees": int,
                "nb_ignorees_pas_de_correspondance": int}
            -- une ligne de corrections.csv est "ignorée" si son crop_path
            ne correspond à aucun canard connu dans `resultats` (rafale
            supprimée/renommée depuis, resultats_par_rafale.json d'un
            autre lot que celui de corrections.csv...) ou si elle ne
            contient en fait aucune valeur corrigée exploitable. Ne fait
            jamais planter le bilan, juste compté à part pour transparence.
    """
    predictions = _index_predictions_par_crop(resultats)
    lignes = _lire_corrections_csv(dossier_dictionnaire)

    matrices_brutes = {classe: defaultdict(lambda: defaultdict(int)) for classe in CLASSES_COLONNE_CSV}
    nb_exploitees = 0
    nb_ignorees = 0

    for ligne in lignes:
        crop_filename = Path(ligne.get("crop_path", "")).name if ligne.get("crop_path") else None
        pred = predictions.get(crop_filename) if crop_filename else None
        if pred is None:
            nb_ignorees += 1
            continue

        ligne_exploitee = False
        for classe, colonne_csv in CLASSES_COLONNE_CSV.items():
            valeur_corrigee = (ligne.get(colonne_csv) or "").strip()
            if not valeur_corrigee:
                continue
            valeur_predite = pred.get(classe) or "(aucune)"
            matrices_brutes[classe][valeur_predite][valeur_corrigee] += 1
            ligne_exploitee = True

        if ligne_exploitee:
            nb_exploitees += 1
        else:
            nb_ignorees += 1

    matrices = {classe: {p: dict(c) for p, c in m.items()} for classe, m in matrices_brutes.items()}
    stats = {
        "nb_corrections_lues": len(lignes),
        "nb_exploitees": nb_exploitees,
        "nb_ignorees_pas_de_correspondance": nb_ignorees,
    }
    return matrices, stats


def construire_bilan_exactitude(resultats, dossier_dictionnaire, matrices=None):
    """
    Calcule, PAR CLASSE (espece/sexe/couleur), un vrai taux d'exactitude
    parmi les cas RÉELLEMENT RELUS -- répond à la limite méthodologique
    documentée en tête de ce module (construire_matrices_confusion seule
    ne voit que les cas CHANGÉS, avec un dénominateur inconnu).

    S'appuie sur dictionnaire/revues_classification.json, écrit à CHAQUE
    validation du formulaire (changement ou non) par
    module_corrections/ecrire_corrections.py::enregistrer_revue_classification
    -- séparé de corrections.csv pour ne jamais changer le sens de
    "corrigé" dans le rapport HTML.

    Args:
        resultats: contenu de resultats_par_rafale.json.
        dossier_dictionnaire: dossier contenant corrections.csv ET
            revues_classification.json (typiquement le même que --output).
        matrices: résultat déjà calculé de construire_matrices_confusion,
            pour éviter de relire corrections.csv une deuxième fois si
            l'appelant l'a déjà fait (14_bilan_matrice_confusion.py).
            Recalculé ici si omis.

    Returns:
        {classe: {"nb_revus": int, "nb_corriges": int,
                   "taux_exactitude": float ou None}}
        taux_exactitude est None si nb_revus == 0 (rien à calculer pour
        cette classe -- ex. lot traité avant l'ajout de cette
        fonctionnalité, aucune donnée pour en tirer un taux).

    ⚠️ MÊME LIMITE D'ÉCHANTILLON que la matrice de confusion : ce sont les
    cas que le chercheur a CHOISI de relire (souvent les plus suspects),
    pas un échantillon aléatoire de la production. Le taux obtenu est
    donc correct SUR CE QUI A ÉTÉ RELU, pas nécessairement représentatif
    de l'exactitude réelle sur l'ensemble des rafales jamais rouvertes.
    """
    if matrices is None:
        matrices, _ = construire_matrices_confusion(resultats, dossier_dictionnaire)

    revues = _lire_revues_classification(dossier_dictionnaire)

    nb_revus = {classe: 0 for classe in CLASSES_COLONNE_CSV}
    for infos in revues.values():
        for classe in infos.get("classes_revues", []):
            if classe in nb_revus:
                nb_revus[classe] += 1

    bilan = {}
    for classe in CLASSES_COLONNE_CSV:
        nb_corriges = sum(
            count
            for corrections in matrices.get(classe, {}).values()
            for count in corrections.values()
        )
        revus = nb_revus[classe]
        bilan[classe] = {
            "nb_revus": revus,
            "nb_corriges": nb_corriges,
            "taux_exactitude": (1 - nb_corriges / revus) if revus else None,
        }
    return bilan


def _lire_json_optionnel(dossier_dictionnaire, nom_fichier):
    """Lit un fichier JSON de dictionnaire/ en tolérant son absence (lot
    traité avant l'ajout de la fonctionnalité correspondante) ou sa
    corruption -- retourne {} plutôt que de faire planter le bilan, même
    principe que _lire_revues_classification()."""
    chemin = Path(dossier_dictionnaire) / nom_fichier
    if not chemin.exists():
        return {}
    import json
    try:
        with open(chemin, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def construire_matrice_marque_nasale(dossier_dictionnaire):
    """Vraie matrice de confusion (Vrai/Faux Positif/Négatif) pour la
    DÉTECTION de marque nasale -- distincte de matrices["couleur"] etc.
    (qui comparent la CLASSIFICATION d'une marque déjà détectée, pas le
    fait qu'elle ait été détectée du tout).

    S'appuie sur dictionnaire/revues_marque_nasale.json, écrit à CHAQUE
    validation du formulaire (changement ou non -- voir
    module_corrections/ecrire_corrections.py::enregistrer_revue_marque_nasale),
    donc contrairement à la matrice espèce/sexe/couleur ci-dessus, celle-ci
    COUVRE BIEN les cas validés sans aucune modification : un canard sans
    marque, relu et confirmé sans marque, compte comme un Vrai Négatif
    même si rien n'a été "corrigé".

    Définition des 4 cases (voir docstring de enregistrer_revue_marque_nasale
    pour le détail) :
        IA a dit "marque présente"  + chercheur confirme présente -> VP
        IA a dit "marque présente"  + chercheur dit absente       -> FP
        IA a dit "marque absente"   + chercheur en signale une    -> FN
        IA a dit "marque absente"   + chercheur confirme absente  -> VN

    Returns:
        {"vrai_positif": int, "faux_positif": int, "faux_negatif": int,
         "vrai_negatif": int, "nb_revus": int,
         "precision": float ou None, "rappel": float ou None,
         "exactitude": float ou None}
        precision = VP / (VP + FP) -- parmi les marques que l'IA a
            signalées, combien existaient vraiment.
        rappel = VP / (VP + FN) -- parmi les marques réellement présentes,
            combien l'IA a détectées.
        exactitude = (VP + VN) / nb_revus -- taux de bonnes réponses tous
            cas confondus (présence ET absence).
        Toutes à None si nb_revus == 0 (rien à calculer).

    ⚠️ MÊME LIMITE D'ÉCHANTILLON que construire_bilan_exactitude : ne
    couvre que les canards dont le formulaire a été ouvert et validé au
    moins une fois -- pas un tirage aléatoire de la production.
    """
    revues = _lire_json_optionnel(dossier_dictionnaire, "revues_marque_nasale.json")

    vp = fp = fn = vn = 0
    for infos in revues.values():
        ia_dit_presente = not infos.get("marque_absente_detectee", True)
        chercheur_dit_presente = bool(infos.get("marque_presente_chercheur"))
        if ia_dit_presente and chercheur_dit_presente:
            vp += 1
        elif ia_dit_presente and not chercheur_dit_presente:
            fp += 1
        elif not ia_dit_presente and chercheur_dit_presente:
            fn += 1
        else:
            vn += 1

    nb_revus = vp + fp + fn + vn
    return {
        "vrai_positif": vp,
        "faux_positif": fp,
        "faux_negatif": fn,
        "vrai_negatif": vn,
        "nb_revus": nb_revus,
        "precision": (vp / (vp + fp)) if (vp + fp) else None,
        "rappel": (vp / (vp + fn)) if (vp + fn) else None,
        "exactitude": ((vp + vn) / nb_revus) if nb_revus else None,
    }


def construire_bilan_detection_canards(dossier_dictionnaire):
    """Précision/rappel pour la DÉTECTION de canard elle-même (le cadre
    YOLO), séparée de tout ce qui concerne la classification ou la
    marque nasale d'un canard déjà détecté.

    Combine 3 journaux distincts, chacun tenu à jour par
    module_corrections/ (voir ecrire_corrections.py) :
        - confirmations_canard.json : canards détectés par l'IA, relus et
          validés SANS être supprimés -> Vrais Positifs.
        - faux_positifs_canard.json : canards détectés par l'IA puis
          supprimés (🗑️) par le chercheur -> Faux Positifs.
        - detections_manuelles.json : canards ajoutés à la main parce que
          l'IA n'avait rien détecté à cet endroit -> Faux Négatifs (déjà
          collecté pour un autre usage -- réentraînement du détecteur,
          voir 13_exporter_annotations_detection.py -- réutilisé ici tel
          quel plutôt que dupliqué).

    Pas de Vrai Négatif ici : contrairement à la marque nasale (une
    propriété binaire par canard déjà identifié), il n'y a pas de liste
    finie de "non-canards" à énumérer sur une photo -- seule une
    précision/rappel a un sens pour une tâche de détection.

    Returns:
        {"vrai_positif": int, "faux_positif": int, "faux_negatif": int,
         "precision": float ou None, "rappel": float ou None}
        precision = VP / (VP + FP), rappel = VP / (VP + FN) ; None si le
        dénominateur correspondant est nul.

    ⚠️ MÊME LIMITE D'ÉCHANTILLON que les autres bilans de ce module : ne
    couvre que les rafales effectivement rouvertes dans la fenêtre
    "photo + liste", pas un tirage aléatoire de la production.
    """
    confirmations = _lire_json_optionnel(dossier_dictionnaire, "confirmations_canard.json")
    faux_positifs = _lire_json_optionnel(dossier_dictionnaire, "faux_positifs_canard.json")
    detections_manuelles = _lire_json_optionnel(dossier_dictionnaire, "detections_manuelles.json")

    vp = len(confirmations)
    fp = len(faux_positifs)
    fn = len(detections_manuelles)

    return {
        "vrai_positif": vp,
        "faux_positif": fp,
        "faux_negatif": fn,
        "precision": (vp / (vp + fp)) if (vp + fp) else None,
        "rappel": (vp / (vp + fn)) if (vp + fn) else None,
    }


def analyser_seuil_confiance(resultats, dossier_dictionnaire, largeur_bucket=0.1):
    """
    Croise la confiance de l'IA (canard["confiance"], déjà utilisée pour le
    tri/l'affichage du rapport HTML -- voir utils/report.py::_resume_canards)
    avec le résultat des revues de classification
    (dictionnaire/revues_classification.json) pour repérer, PARMI LES CAS
    RÉELLEMENT RELUS, si les corrections se concentrent sur les tranches de
    confiance basses -- utile pour fixer un seuil "en dessous de X%,
    toujours relire à la main" plutôt que de compter sur une relecture
    opportuniste.

    Un canard compte comme "corrigé" s'il apparaît dans corrections.csv
    pour AU MOINS UNE des 3 classes (espece/sexe/couleur) -- cette analyse
    ne distingue pas la classe fautive, contrairement à
    construire_matrices_confusion/construire_bilan_exactitude, parce que
    canard["confiance"] est déjà une confiance GLOBALE par canard (pas une
    confiance séparée par classe -- voir _resume_canards), donc croiser
    par classe n'aurait pas de sens ici.

    ⚠️ MÊME LIMITE D'ÉCHANTILLON que construire_bilan_exactitude (cas
    choisis par le chercheur, pas un tirage aléatoire) -- voir sa
    docstring. S'ajoute ici une deuxième limite : un canard sans
    "confiance" connue (ex. NON_MARQUE, ou ajouté à la main) est exclu du
    calcul par bucket et compté à part.

    Args:
        resultats: contenu de resultats_par_rafale.json.
        dossier_dictionnaire: dossier contenant corrections.csv ET
            revues_classification.json.
        largeur_bucket: taille de chaque tranche de confiance (0.1 par
            défaut : [0-10%), [10-20%), ... [90-100%]).

    Returns:
        {"buckets": [{"borne_basse": float, "borne_haute": float,
                       "nb_revus": int, "nb_corriges": int,
                       "taux_exactitude": float ou None}, ...],
         "nb_sans_confiance": int}
        buckets triés par confiance croissante, uniquement ceux avec
        nb_revus > 0 (pas de ligne vide à afficher).
    """
    revues = _lire_revues_classification(dossier_dictionnaire)
    lignes_corrections = _lire_corrections_csv(dossier_dictionnaire)
    crops_corriges = {
        Path(ligne["crop_path"]).name
        for ligne in lignes_corrections
        if ligne.get("crop_path") and any(
            (ligne.get(col) or "").strip() for col in CLASSES_COLONNE_CSV.values()
        )
    }

    nb_buckets = int(round(1.0 / largeur_bucket))
    compteurs = [{"nb_revus": 0, "nb_corriges": 0} for _ in range(nb_buckets)]
    nb_sans_confiance = 0

    for result in resultats:
        photo_source = result.get("photo_analysee")
        rafale_id = result.get("rafale_id")
        if not photo_source:
            continue
        for canard in result.get("canards", []):
            idx = canard.get("canard_index", 0)
            crop_filename = nom_crop_correction(photo_source, rafale_id, idx)
            if crop_filename not in revues:
                continue  # jamais relu -> hors échantillon, voir docstring

            confiance = canard.get("confiance")
            if confiance is None:
                nb_sans_confiance += 1
                continue

            bucket_idx = min(int(confiance / largeur_bucket), nb_buckets - 1)
            compteurs[bucket_idx]["nb_revus"] += 1
            if crop_filename in crops_corriges:
                compteurs[bucket_idx]["nb_corriges"] += 1

    buckets = []
    for i, compte in enumerate(compteurs):
        if compte["nb_revus"] == 0:
            continue
        buckets.append({
            "borne_basse": round(i * largeur_bucket, 2),
            "borne_haute": round((i + 1) * largeur_bucket, 2),
            "nb_revus": compte["nb_revus"],
            "nb_corriges": compte["nb_corriges"],
            "taux_exactitude": 1 - compte["nb_corriges"] / compte["nb_revus"],
        })

    return {"buckets": buckets, "nb_sans_confiance": nb_sans_confiance}


def matrice_vers_lignes(matrice):
    """Aplatit une matrice {predit: {corrige: count}} en tuples
    (predit, corrige, count) triés par count décroissant."""
    lignes = [(predit, corrige, count)
              for predit, corrections in matrice.items()
              for corrige, count in corrections.items()]
    lignes.sort(key=lambda l: -l[2])
    return lignes


def ecrire_matrice_csv(matrice, chemin):
    """Écrit une matrice à plat (predit_par_ia, corrige_par_chercheur,
    nombre_de_cas) -- un CSV par classe, plus facile à ouvrir dans un
    tableur qu'une vraie grille pour un nombre de classes variable."""
    chemin = Path(chemin)
    chemin.parent.mkdir(parents=True, exist_ok=True)
    with open(chemin, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["predit_par_ia", "corrige_par_chercheur", "nombre_de_cas"])
        writer.writerows(matrice_vers_lignes(matrice))
    return str(chemin)


def exporter_graphique_confusion(matrices, output_path):
    """PNG avec un panneau (heatmap) par classe qui a au moins une
    confusion enregistrée -- lignes = valeur prédite par l'IA, colonnes =
    valeur corrigée par le chercheur. Même convention Agg que
    utils/synthese.py::export_graphiques_synthese (pas d'affichage
    interactif nécessaire, juste écrire le PNG).

    Retourne le chemin écrit, ou None si aucune classe n'a de données
    (aucune correction exploitable pour l'instant) -- n'écrit alors rien,
    même chose que export_graphiques_synthese pour une synthèse vide."""
    panneaux = [(classe, m) for classe, m in matrices.items() if m]
    if not panneaux:
        return None

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(1, len(panneaux), figsize=(6.5 * len(panneaux), 5.5))
    if len(panneaux) == 1:
        axes = [axes]

    for ax, (classe, matrice) in zip(axes, panneaux, strict=True):
        predits = sorted(matrice)
        corriges = sorted({c for corrections in matrice.values() for c in corrections})
        grille = np.zeros((len(predits), len(corriges)), dtype=int)
        for i, p in enumerate(predits):
            for j, c in enumerate(corriges):
                grille[i, j] = matrice[p].get(c, 0)

        im = ax.imshow(grille, cmap="Reds", aspect="auto")
        ax.set_xticks(range(len(corriges)))
        ax.set_xticklabels(corriges, rotation=40, ha="right", fontsize=8)
        ax.set_yticks(range(len(predits)))
        ax.set_yticklabels(predits, fontsize=8)
        ax.set_xlabel("Corrigé par le chercheur")
        ax.set_ylabel("Prédit par l'IA")
        ax.set_title(classe.capitalize())

        seuil = grille.max() / 2 if grille.size else 0
        for i in range(len(predits)):
            for j in range(len(corriges)):
                valeur = grille[i, j]
                if valeur:
                    ax.text(j, i, str(valeur), ha="center", va="center", fontsize=9,
                            color="white" if valeur > seuil else "black")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        "Confusions IA -> corrections manuelles\n"
        "(uniquement les cas relus ET corrigés -- pas un taux d'exactitude global, voir utils/confusion.py)",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=110)
    plt.close(fig)
    return str(output_path)