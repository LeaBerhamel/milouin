# -*- coding: utf-8 -*-
"""
Écrit le résultat du formulaire de correction dans dictionnaire/ :
  - corrections.csv       : uniquement les champs espèce/sexe/couleur
                             réellement changés par rapport à la détection
                             (pas de bruit si l'IA avait déjà raison).
  - dictionnaire_identifications.json : la bague, si renseignée/changée.
  - crops_exclus.json     : si la bague est marquée ILLISIBLE ou PAS_MARQUE
                             (🚫/❌ dans le formulaire) -- même convention
                             que 08_importer_identifications_manuelles.py
                             (voir scripts/utils/dictionnaire.py), pour que
                             ce crop n'apparaisse plus jamais dans un futur
                             dossier de relecture.
  - detections_manuelles.json : les canards ajoutés à la main depuis la
                             fenêtre "photo + liste" (canvas interactif,
                             voir module_corrections/fenetre_rafale.py) --
                             photo source + bbox (x, y, largeur, hauteur)
                             en pixels natifs, PAS les champs espèce/sexe/
                             couleur (déjà couverts par corrections.csv).
                             Sert uniquement à un futur réentraînement du
                             modèle de DÉTECTION (YOLO) -- voir
                             scripts/13_exporter_annotations_detection.py.
  - revues_classification.json : QUE la classification (espèce/sexe/
                             couleur) ait été changée ou non, écrit à
                             CHAQUE validation du formulaire -- sert de
                             dénominateur à scripts/utils/confusion.py
                             (nombre de cas réellement relus, pas
                             seulement corrigés) sans changer le sens de
                             "corrigé" dans corrections.csv/le rapport.

Le format de corrections.csv (colonnes crop_path / espece_corrigee /
sexe_corrige / couleur_corrigee) suit exactement ce que
scripts/10_fusionner_corrections_dataset.py sait déjà lire -- voir sa
fonction fusionner_corrections(). Ne pas renommer ces colonnes sans mettre
ce script à jour aussi.

Autonome, ne modifie rien dans scripts/ (mais importe scripts/utils/dictionnaire.py
en lecture seule pour la logique ILLISIBLE/PAS_MARQUE -- une seule source
de vérité avec 07/08, voir README.md pour le branchement).
"""

import csv
import json
import sys
from datetime import datetime
from pathlib import Path

try:
    # Import direct depuis le dépôt réel une fois ce module copié à côté
    # de scripts/ (voir README.md) -- même pattern que formulaire_correction.py.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    from utils.dictionnaire import est_code_exclusion, enregistrer_exclusion, charger_exclusions, sauvegarder_exclusions
except ImportError:
    # Filet de sécurité pour la démo autonome (demo_standalone.py), testée
    # sans le reste du dépôt à côté -- copie minimale de la même logique.
    CODES_EXCLUSION = {"ILLISIBLE": "illisible", "PAS_MARQUE": "pas_marque"}

    def est_code_exclusion(valeur):
        return (valeur or "").strip().upper() in CODES_EXCLUSION

    def charger_exclusions(path):
        path = Path(path)
        if not path.exists():
            return []
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def sauvegarder_exclusions(exclusions, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(exclusions, f, ensure_ascii=False, indent=2)

    def enregistrer_exclusion(exclusions, crop_filename, valeur):
        deja = {e["crop_filename"] for e in exclusions}
        if crop_filename in deja:
            return False
        from datetime import date
        exclusions.append({
            "crop_filename": crop_filename,
            "raison": CODES_EXCLUSION[(valeur or "").strip().upper()],
            "date_exclusion": date.today().isoformat(),
        })
        return True

COLONNES_CORRECTIONS = ["date", "crop_path", "espece_corrigee", "sexe_corrige", "couleur_corrigee"]


def enregistrer_correction_espece_sexe_couleur(dossier_dictionnaire: Path, valeurs: dict) -> bool:
    """Ajoute une ligne à dictionnaire/corrections.csv SEULEMENT pour les
    champs qui diffèrent de la valeur détectée par l'IA.

    Retourne True si une ligne a été écrite, False si rien n'avait changé
    (évite d'ouvrir/fermer le fichier pour rien côté appelant si besoin).
    """
    ligne = {
        "date": datetime.now().isoformat(timespec="seconds"),
        "crop_path": valeurs.get("crop", ""),
        "espece_corrigee": "",
        "sexe_corrige": "",
        "couleur_corrigee": "",
    }

    a_ecrire = False

    if valeurs.get("espece") and valeurs["espece"] != valeurs.get("espece_detectee"):
        ligne["espece_corrigee"] = valeurs["espece"]
        a_ecrire = True

    if valeurs.get("sexe") and valeurs["sexe"] != valeurs.get("sexe_detecte"):
        ligne["sexe_corrige"] = valeurs["sexe"]
        a_ecrire = True

    if valeurs.get("couleur") and valeurs["couleur"] != valeurs.get("couleur_detectee"):
        ligne["couleur_corrigee"] = valeurs["couleur"]
        a_ecrire = True

    if not a_ecrire:
        return False

    dossier_dictionnaire.mkdir(parents=True, exist_ok=True)
    corrections_csv = dossier_dictionnaire / "corrections.csv"
    fichier_existe_deja = corrections_csv.exists()

    with open(corrections_csv, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLONNES_CORRECTIONS)
        if not fichier_existe_deja:
            writer.writeheader()
        writer.writerow(ligne)

    return True


def enregistrer_bague(dossier_dictionnaire: Path, valeurs: dict) -> bool:
    """Met à jour dictionnaire/dictionnaire_identifications.json avec la
    bague saisie/corrigée pour ce crop -- SAUF si la valeur est un code
    d'exclusion (ILLISIBLE / PAS_MARQUE, boutons 🚫/❌ du formulaire), qui
    part dans enregistrer_exclusion_bague() à la place (voir plus bas) :
    ce ne sont pas des bagues, elles ne doivent jamais atterrir dans le
    dictionnaire d'identifications.

    NOTE D'INTÉGRATION : ceci est une version simplifiée pour la démo
    autonome (un mapping crop -> bague). La vraie fusion (gestion des
    doublons, des suggestions visuelles déjà proposées par
    06_proposer_correspondances_visuelles.py) vit déjà dans
    07_fusionner_dictionnaire.py -- à l'intégration réelle, il faudra
    probablement appeler la logique de ce script plutôt que de la
    dupliquer ici. Signalé dans le README.
    """
    if not valeurs.get("bague") or est_code_exclusion(valeurs["bague"]):
        return False

    dossier_dictionnaire.mkdir(parents=True, exist_ok=True)
    chemin_json = dossier_dictionnaire / "dictionnaire_identifications.json"

    if chemin_json.exists():
        with open(chemin_json, encoding="utf-8") as f:
            dictionnaire = json.load(f)
    else:
        dictionnaire = {}

    # Clé = nom de fichier nu (pas le chemin complet renvoyé par
    # save_canard_crop) : c'est ce que 04_copier_crops_valides.py et les
    # autres lecteurs attendent (voir docstring du module en haut du
    # fichier). Même correctif que celui déjà appliqué côté lecture dans
    # scripts/utils/report.py pour corrections.csv.
    dictionnaire[Path(valeurs["crop"]).name] = {
        "bague": valeurs["bague"],
        "corrige_le": datetime.now().isoformat(timespec="seconds"),
        "source": "correction_ponctuelle_rapport",
    }

    with open(chemin_json, mode="w", encoding="utf-8") as f:
        json.dump(dictionnaire, f, ensure_ascii=False, indent=2)

    return True


def enregistrer_exclusion_bague(dossier_dictionnaire: Path, valeurs: dict) -> bool:
    """Si la bague a été marquée ILLISIBLE ou PAS_MARQUE (boutons dédiés du
    formulaire, pas le champ texte libre), ajoute ce crop à
    dictionnaire/crops_exclus.json -- même convention que
    08_importer_identifications_manuelles.py (voir utils/dictionnaire.py),
    pour qu'il ne réapparaisse plus jamais dans un futur dossier de
    relecture. Retourne True si une entrée a réellement été ajoutée
    (False si absent, si ce n'est pas un code d'exclusion, ou si déjà
    exclu précédemment).
    """
    if not valeurs.get("bague") or not est_code_exclusion(valeurs["bague"]):
        return False

    dossier_dictionnaire.mkdir(parents=True, exist_ok=True)
    chemin_exclusions = dossier_dictionnaire / "crops_exclus.json"
    exclusions = charger_exclusions(chemin_exclusions)

    # Nom de fichier nu, pas le chemin complet -- voir même correctif dans
    # enregistrer_bague() ci-dessus.
    ajoute = enregistrer_exclusion(exclusions, Path(valeurs["crop"]).name, valeurs["bague"])
    if ajoute:
        sauvegarder_exclusions(exclusions, chemin_exclusions)
    return ajoute


def enregistrer_detection_manquee(dossier_dictionnaire: Path, valeurs: dict) -> bool:
    """Ajoute une entrée à dictionnaire/detections_manuelles.json quand le
    canard corrigé a été ajouté à la main (canvas interactif de
    fenetre_rafale.py), donc raté par la détection YOLO.

    Ne fait rien pour un canard normalement détecté par l'IA (valeurs ne
    contient alors ni "ajoute_manuellement", ni "photo_source", ni "bbox"
    -- la fenêtre "photo + liste" ne les fournit que pour ce cas précis,
    voir _ouvrir_correction()). Une entrée par crop_path : revalider le
    même ajout manuel (ex. après avoir corrigé l'espèce) écrase la
    précédente au lieu de dupliquer la ligne côté export YOLO.

    Retourne True si une entrée a été écrite.
    """
    if not valeurs.get("ajoute_manuellement") or not valeurs.get("bbox") or not valeurs.get("photo_source"):
        return False

    dossier_dictionnaire.mkdir(parents=True, exist_ok=True)
    chemin_json = dossier_dictionnaire / "detections_manuelles.json"

    if chemin_json.exists():
        with open(chemin_json, encoding="utf-8") as f:
            detections = json.load(f)
    else:
        detections = {}

    # Nom de fichier nu, pas le chemin complet -- voir même correctif dans
    # enregistrer_bague() ci-dessus.
    detections[Path(valeurs["crop"]).name] = {
        "photo_source": valeurs["photo_source"],
        "bbox": list(valeurs["bbox"]),  # (x, y, largeur, hauteur), pixels natifs
        "classe": "canard",
        "ajoute_le": datetime.now().isoformat(timespec="seconds"),
    }

    with open(chemin_json, mode="w", encoding="utf-8") as f:
        json.dump(detections, f, ensure_ascii=False, indent=2)

    return True


def enregistrer_revue_classification(dossier_dictionnaire: Path, valeurs: dict) -> bool:
    """Enregistre qu'une revue de classification a eu lieu pour ce crop --
    QUE LE CHERCHEUR AIT CHANGÉ UNE VALEUR OU NON, contrairement à
    enregistrer_correction_espece_sexe_couleur() ci-dessus qui n'écrit rien
    quand rien n'a changé (pour ne pas polluer corrections.csv/le badge
    "✏️ corrigé" du rapport, qui doivent rester synonymes de "changé").

    Sert de DÉNOMINATEUR à scripts/utils/confusion.py::construire_bilan_exactitude
    -- corrections.csv seul dit "combien de fois l'IA s'est trompée PARMI
    LES CAS CORRIGÉS", ce fichier permet de savoir "sur combien de cas
    RELUS au total", donc de calculer un vrai taux d'exactitude par classe
    plutôt qu'une matrice de confusion brute sans dénominateur connu.

    Appelée à CHAQUE clic sur "Valider" du formulaire (voir _finaliser()
    dans formulaire_correction.py / traiter_validation_formulaire()
    ci-dessous), indépendamment des 3 autres fonctions d'écriture.

    Une classe (espece/sexe/couleur) ne compte comme "revue" que si l'IA
    avait produit une prédiction à comparer pour ce crop -- clé
    "<classe>_detectee"/"_detecte" non vide dans `valeurs` (transmise par
    formulaire_correction.py, voir valeurs_initiales) -- sinon il n'y avait
    rien à confirmer/infirmer pour cette classe précise sur ce crop (ex.
    modèle couleur non exécuté sur ce lot, ou canard ajouté à la main sans
    classification IA du tout).

    Une revalidation du même crop (ex. après avoir corrigé l'espèce puis
    rouvert pour corriger le sexe) écrase l'entrée précédente au lieu de
    dupliquer -- même convention que enregistrer_detection_manquee().

    Retourne True si une entrée a été écrite (False si aucune des 3
    classes n'avait de prédiction IA à comparer, ou si crop absent).
    """
    if not valeurs.get("crop"):
        return False

    classes_revues = []
    if (valeurs.get("espece_detectee") or "").strip():
        classes_revues.append("espece")
    if (valeurs.get("sexe_detecte") or "").strip():
        classes_revues.append("sexe")
    if (valeurs.get("couleur_detectee") or "").strip():
        classes_revues.append("couleur")

    if not classes_revues:
        return False

    dossier_dictionnaire.mkdir(parents=True, exist_ok=True)
    chemin_json = dossier_dictionnaire / "revues_classification.json"

    if chemin_json.exists():
        with open(chemin_json, encoding="utf-8") as f:
            revues = json.load(f)
    else:
        revues = {}

    # Nom de fichier nu, pas le chemin complet -- même correctif que
    # enregistrer_bague() ci-dessus.
    revues[Path(valeurs["crop"]).name] = {
        "classes_revues": classes_revues,
        "revu_le": datetime.now().isoformat(timespec="seconds"),
    }

    with open(chemin_json, mode="w", encoding="utf-8") as f:
        json.dump(revues, f, ensure_ascii=False, indent=2)

    return True


def enregistrer_revue_marque_nasale(dossier_dictionnaire: Path, valeurs: dict) -> bool:
    """Enregistre, à CHAQUE validation du formulaire (changement ou non),
    la comparaison entre ce que l'IA de détection avait dit pour la
    présence d'une marque nasale sur ce canard (`marque_absente`, un vrai
    booléen toujours renseigné par le pipeline -- voir utils/pipeline.py)
    et ce que le chercheur a effectivement confirmé via la case "🏷️ Marque
    présente" du formulaire.

    Même principe que enregistrer_revue_classification() ci-dessus (sert
    de dénominateur complet, pas seulement les cas changés), mais pour un
    booléen de détection plutôt qu'une classe à choix multiple -- ça
    donne directement les 4 cases d'une vraie matrice de confusion
    vrai/faux positif/négatif (voir utils/confusion.py::construire_matrice_marque_nasale) :
        IA a dit "marque présente" (marque_absente=False) :
            - chercheur confirme présente  -> Vrai Positif
            - chercheur dit absente        -> Faux Positif (l'IA a cru
              voir une marque qui n'existe pas)
        IA a dit "marque absente" (marque_absente=True) :
            - chercheur confirme absente   -> Vrai Négatif
            - chercheur en signale une     -> Faux Négatif (l'IA a raté
              une marque réelle)

    "marque_absente_detectee" dans `valeurs` (transmis par
    fenetre_rafale.py::_ouvrir_correction, via valeurs_initiales, puis
    formulaire_correction.py::_finaliser) vaut None quand l'IA de
    détection n'a jamais évalué ce canard du tout (canard ajouté à la
    main) -- dans ce cas, rien à comparer, on n'écrit rien (comme
    enregistrer_revue_classification saute une classe sans prédiction).

    La "vérité terrain" pour la présence de marque se déduit du champ
    "bague" final du formulaire plutôt que de relire l'état de la case à
    cocher directement : "PAS_MARQUE" => case décochée => marque absente
    confirmée ; toute autre valeur (code réel ou "ILLISIBLE") => case
    cochée => marque présente confirmée. Voir formulaire_correction.py::_valider,
    qui construit "bague" à partir des 2 cases selon exactement cette
    règle.

    Retourne True si une entrée a été écrite.
    """
    crop = valeurs.get("crop")
    if not crop:
        return False

    marque_absente_detectee = valeurs.get("marque_absente_detectee")
    if marque_absente_detectee is None:
        return False  # IA jamais passée sur ce canard (ajout manuel) -- rien à comparer

    bague = (valeurs.get("bague") or "").strip().upper()
    if not bague:
        # Ne devrait pas arriver via _valider() (qui écrit toujours
        # PAS_MARQUE/ILLISIBLE/un code) -- filet de sécurité pour ne
        # jamais écrire une revue avec une vérité terrain inconnue.
        return False
    marque_presente_chercheur = bague != "PAS_MARQUE"

    dossier_dictionnaire.mkdir(parents=True, exist_ok=True)
    chemin_json = dossier_dictionnaire / "revues_marque_nasale.json"

    if chemin_json.exists():
        with open(chemin_json, encoding="utf-8") as f:
            revues = json.load(f)
    else:
        revues = {}

    revues[Path(crop).name] = {
        "marque_absente_detectee": bool(marque_absente_detectee),
        "marque_presente_chercheur": marque_presente_chercheur,
        "revu_le": datetime.now().isoformat(timespec="seconds"),
    }

    with open(chemin_json, mode="w", encoding="utf-8") as f:
        json.dump(revues, f, ensure_ascii=False, indent=2)

    return True


def enregistrer_confirmation_canard(dossier_dictionnaire: Path, valeurs: dict) -> bool:
    """Enregistre qu'un canard détecté par l'IA (PAS un ajout manuel) a
    été relu et validé sans être supprimé -- le "vrai positif" du côté
    détection de canard (par opposition à detections_manuelles.json, qui
    couvre déjà le faux négatif : un canard ajouté à la main parce que
    l'IA ne l'avait pas détecté du tout -- voir enregistrer_detection_manquee).

    Ne fait rien pour un canard ajouté manuellement (rien à "confirmer"
    côté détection IA, ce cas est déjà couvert par
    detections_manuelles.json) ni si aucun crop n'est identifiable.

    Une revalidation du même crop écrase l'entrée précédente au lieu de
    dupliquer -- même convention que les autres fonctions d'écriture de
    ce module.

    Retourne True si une entrée a été écrite.
    """
    if valeurs.get("ajoute_manuellement") or not valeurs.get("crop"):
        return False

    dossier_dictionnaire.mkdir(parents=True, exist_ok=True)
    chemin_json = dossier_dictionnaire / "confirmations_canard.json"

    if chemin_json.exists():
        with open(chemin_json, encoding="utf-8") as f:
            confirmations = json.load(f)
    else:
        confirmations = {}

    confirmations[Path(valeurs["crop"]).name] = {
        "confirme_le": datetime.now().isoformat(timespec="seconds"),
    }

    with open(chemin_json, mode="w", encoding="utf-8") as f:
        json.dump(confirmations, f, ensure_ascii=False, indent=2)

    return True


def enregistrer_faux_positif_canard(dossier_dictionnaire: Path, crop_filename: str) -> bool:
    """Enregistre qu'un canard détecté par l'IA (PAS un ajout manuel) a
    été supprimé par le chercheur -- fausse détection (ombre, reflet,
    doublon de cadre...), voir fenetre_rafale.py::_supprimer_canard, qui
    appelle cette fonction AVANT de retirer le canard de
    resultats_par_rafale.json (une fois supprimé, plus aucune trace n'y
    subsisterait pour un futur bilan -- c'est tout l'intérêt de ce
    journal séparé).

    Appelée directement avec un nom de crop déterministe (voir
    utils/crop_export.py::nom_crop_correction) plutôt qu'un dict
    `valeurs` complet comme les autres fonctions de ce module : la
    suppression peut survenir sans jamais avoir ouvert le formulaire de
    correction (bouton 🗑️ disponible directement sur chaque ligne), donc
    sans qu'aucun crop n'ait forcément déjà été extrait sur disque.

    Une nouvelle suppression du même crop écrase l'entrée précédente
    (horodatage mis à jour) au lieu de dupliquer.

    Retourne True si une entrée a été écrite.
    """
    if not crop_filename:
        return False

    dossier_dictionnaire.mkdir(parents=True, exist_ok=True)
    chemin_json = dossier_dictionnaire / "faux_positifs_canard.json"

    if chemin_json.exists():
        with open(chemin_json, encoding="utf-8") as f:
            faux_positifs = json.load(f)
    else:
        faux_positifs = {}

    faux_positifs[Path(crop_filename).name] = {
        "supprime_le": datetime.now().isoformat(timespec="seconds"),
    }

    with open(chemin_json, mode="w", encoding="utf-8") as f:
        json.dump(faux_positifs, f, ensure_ascii=False, indent=2)

    return True


def annuler_correction(dossier_dictionnaire: Path, crop_path) -> dict:
    """Annule tout ce qui a été enregistré pour ce crop par les fonctions
    ci-dessus -- appelée par le bouton ↩️ "Annuler" de fenetre_rafale.py
    (_annuler_ligne / _supprimer_canard).

    Contrairement aux fonctions d'écriture, qui ne touchent chacune qu'un
    seul fichier, on ne sait pas ici lequel des quatre fichiers contient
    réellement une trace de ce crop (ça dépend de ce qui avait été
    corrigé : bague, espèce/sexe/couleur, exclusion, ajout manuel -- ou
    une combinaison). On nettoie donc les quatre, chacun étant un no-op
    silencieux si le crop n'y figure pas.

    Retourne un résumé pour log/debug (mêmes clés que
    traiter_validation_formulaire, préfixées "_annulee") :
        {"correction_annulee": bool, "bague_annulee": bool,
         "exclusion_annulee": bool, "detection_manquee_annulee": bool}
    """
    resultat = {
        "correction_annulee": False,
        "bague_annulee": False,
        "exclusion_annulee": False,
        "detection_manquee_annulee": False,
    }

    # dictionnaire_identifications.json / crops_exclus.json /
    # detections_manuelles.json sont maintenant indexés par nom de fichier
    # nu (voir enregistrer_bague/enregistrer_exclusion_bague/
    # enregistrer_detection_manquee ci-dessus) -- crop_path reçu ici peut
    # être le chemin complet (c'est ce que passe fenetre_rafale.py), donc
    # on normalise avant de chercher dans ces trois fichiers-là.
    # corrections.csv, lui, stocke bien le chemin complet (cf.
    # scripts/utils/report.py qui normalise déjà à la lecture) -- on ne
    # touche pas à sa comparaison plus bas.
    crop_filename = Path(crop_path).name

    # 1. corrections.csv -- retire TOUTES les lignes de ce crop (pas
    #    seulement la dernière) : 10_fusionner_corrections_dataset.py lit
    #    le fichier ligne par ligne sans dédoublonnage, donc une ancienne
    #    ligne oubliée pourrait être fusionnée par erreur plus tard.
    corrections_csv = dossier_dictionnaire / "corrections.csv"
    if corrections_csv.exists():
        with open(corrections_csv, encoding="utf-8") as f:
            lignes = list(csv.DictReader(f))
        lignes_conservees = [l for l in lignes if l.get("crop_path") != crop_path]
        if len(lignes_conservees) != len(lignes):
            with open(corrections_csv, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=COLONNES_CORRECTIONS)
                writer.writeheader()
                writer.writerows(lignes_conservees)
            resultat["correction_annulee"] = True

    # 2. dictionnaire_identifications.json -- retire la bague si présente.
    chemin_dico = dossier_dictionnaire / "dictionnaire_identifications.json"
    if chemin_dico.exists():
        with open(chemin_dico, encoding="utf-8") as f:
            dictionnaire = json.load(f)
        if crop_filename in dictionnaire:
            del dictionnaire[crop_filename]
            with open(chemin_dico, mode="w", encoding="utf-8") as f:
                json.dump(dictionnaire, f, ensure_ascii=False, indent=2)
            resultat["bague_annulee"] = True

    # 3. crops_exclus.json -- retire l'exclusion (ILLISIBLE/PAS_MARQUE) si
    #    présente, via les mêmes helpers que enregistrer_exclusion_bague()
    #    pour rester cohérent avec 08_importer_identifications_manuelles.py.
    chemin_exclusions = dossier_dictionnaire / "crops_exclus.json"
    if chemin_exclusions.exists():
        exclusions = charger_exclusions(chemin_exclusions)
        taille_avant = len(exclusions)
        exclusions = [e for e in exclusions if e.get("crop_filename") != crop_filename]
        if len(exclusions) != taille_avant:
            sauvegarder_exclusions(exclusions, chemin_exclusions)
            resultat["exclusion_annulee"] = True

    # 4. detections_manuelles.json -- retire l'ajout manuel (canvas
    #    interactif) si présent -- utilisé notamment par _supprimer_canard()
    #    quand on supprime carrément un canard ajouté à la main.
    chemin_detections = dossier_dictionnaire / "detections_manuelles.json"
    if chemin_detections.exists():
        with open(chemin_detections, encoding="utf-8") as f:
            detections = json.load(f)
        if crop_filename in detections:
            del detections[crop_filename]
            with open(chemin_detections, mode="w", encoding="utf-8") as f:
                json.dump(detections, f, ensure_ascii=False, indent=2)
            resultat["detection_manquee_annulee"] = True

    return resultat


def traiter_validation_formulaire(dossier_dictionnaire: Path, valeurs: dict) -> dict:
    """Point d'entrée unique appelé par on_valider() du formulaire :
    répartit vers les écritures ci-dessus. Retourne un résumé pour
    affichage/log :
        {"correction_ecrite": bool, "bague_ecrite": bool,
         "bague_exclue": bool, "bague_contredit_tableur": bool,
         "detection_manquee_ecrite": bool}

    bague_contredit_tableur est True quand le chercheur a changé (ou
    effacé/marqué illisible) une bague qui venait du tableur d'observations
    (rapprochement horaire, pas d'une détection IA) -- voir le
    messagebox.askyesno de formulaire_correction.py, qui bloque déjà
    l'écriture si le chercheur répond "Non" à cet avertissement : si on
    arrive ici avec ce cas, c'est qu'il a confirmé le changement. Ce
    booléen sert seulement à l'affichage d'un badge distinct côté fenêtre
    "photo + liste" (ex. "⚠️ modifiée (contredit le tableur)"), pas à
    bloquer quoi que ce soit -- le blocage a déjà eu lieu en amont.
    """
    bague_contredit_tableur = bool(
        valeurs.get("bague_confirmee_tableur")
        and valeurs.get("bague", "") != valeurs.get("bague_initiale_tableur", "")
    )

    return {
        "correction_ecrite": enregistrer_correction_espece_sexe_couleur(dossier_dictionnaire, valeurs),
        "bague_ecrite": enregistrer_bague(dossier_dictionnaire, valeurs),
        "bague_exclue": enregistrer_exclusion_bague(dossier_dictionnaire, valeurs),
        "bague_contredit_tableur": bague_contredit_tableur,
        "detection_manquee_ecrite": enregistrer_detection_manquee(dossier_dictionnaire, valeurs),
        # Écrites dans tous les cas (changement ou non) -- voir leurs
        # docstrings respectives. Contrairement aux 4 précédentes,
        # n'apparaissent pas dans un "bool de succès" côté appelant à
        # afficher : elles n'ont pas d'échec fonctionnel visible pour
        # l'utilisateur, juste un dénominateur pour un futur bilan
        # (14_bilan_matrice_confusion.py).
        "revue_classification_ecrite": enregistrer_revue_classification(dossier_dictionnaire, valeurs),
        "revue_marque_nasale_ecrite": enregistrer_revue_marque_nasale(dossier_dictionnaire, valeurs),
        "confirmation_canard_ecrite": enregistrer_confirmation_canard(dossier_dictionnaire, valeurs),
    }