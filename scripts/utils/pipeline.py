# -*- coding: utf-8 -*-
"""Pipeline complet d'identification de canards (détection + classification + matching)."""

from collections import Counter

import cv2
import pandas as pd
from ultralytics import YOLO

from .confiance import compute_confiance_globale
from .config import (
    CONFIDENCE_THRESHOLD_CANARD,
    CONFIDENCE_THRESHOLD_MARQUE,
    DETECTION_CLASSE_CANARD,
    DETECTION_CLASSE_MARQUE,
    DETECTION_IMGSZ,
    DETECTION_IMGSZ_ZOOM,
    ESPECE_CODE_TABLEUR,
    SEXE_CODE_TABLEUR,
    logger,
)
from .nuit import est_photo_de_nuit_bgr
from .tableur import load_and_clean_tableur


class CanardIdentificationPipeline:
    """
    Pipeline complet : détection YOLO (marque_nasale, canard), classification
    en cascade (espèce, sexe sur le crop canard ; couleur sur le crop marque),
    puis matching temporel avec le tableur d'observations, filtré par espèce
    puis par sexe en cas d'ambiguïté.

    Architecture (voir ANNOTATION_GUIDE.md) :
        Photo
          -> YOLO détection (marque_nasale, canard)
              -> crop marque_nasale -> IA Couleur
              -> crop canard        -> IA Espèce -> IA Sexe
          -> Matching temporel (filtré par espèce puis sexe si ambigu)
    """

    def __init__(self, tableur_path, model_detection_path,
                 model_couleur_path=None, model_espece_path=None, model_sexe_path=None):
        self.tableur = load_and_clean_tableur(tableur_path)

        logger.info("Chargement IA Détection...")
        self.ia_detection = YOLO(model_detection_path)
        logger.info("Modèle YOLO chargé")

        # IAs 2, 3, 4 : chargement différé (lazy), uniquement si un chemin est fourni.
        self._model_couleur_path = model_couleur_path
        self._model_espece_path = model_espece_path
        self._model_sexe_path = model_sexe_path
        self._ia_couleur = None
        self._ia_espece = None
        self._ia_sexe = None
        self._ia_couleur_classes = None
        self._ia_espece_classes = None
        self._ia_sexe_classes = None

        if model_couleur_path:
            logger.info("IA Couleur configurée (lazy loading)")
        if model_espece_path:
            logger.info("IA Espèce configurée (lazy loading)")
        if model_sexe_path:
            logger.info("IA Sexe configurée (lazy loading)")

    # ------------------------------------------------------------------
    # Chargement paresseux des classifieurs (torchvision resnet18)
    # ------------------------------------------------------------------

    def _load_classifier(self, model_path, class_names):
        """Charge un classifieur ResNet18 entraîné par scripts/train_classification_ia.py
        ou par un notebook Colab (IA_couleur.ipynb, IA_espece.ipynb, IA_sexe.ipynb)
        enregistré ensuite via scripts/gestion_modeles.py.

        Retourne (model, classes_reelles). `classes_reelles` est l'ordre RÉEL des
        classes tel qu'utilisé pendant l'entraînement, et non la liste passée en
        argument.

        Pourquoi : torchvision.datasets.ImageFolder trie les sous-dossiers par
        ordre ALPHABÉTIQUE pour attribuer les indices de classe (0, 1, 2...),
        indépendamment de l'ordre dans lequel on les liste ailleurs dans le code
        (ex. SEXES_CLASSES dans config.py). Utiliser le mauvais ordre revient à
        inverser les prédictions (ex. mâle/femelle échangés) sans aucune erreur
        visible. On lit donc l'ordre réel sauvegardé à côté du modèle plutôt que
        de le redeviner.

        Deux conventions de nommage coexistent pour ce fichier de classes selon
        l'origine du modèle, donc on essaie les deux avant d'abandonner :
          - "<model>.classes.json"   (train_classification_ia.py, script local)
          - "classes_<ia>.json"      (notebooks Colab + gestion_modeles.py,
                                       ex. modele_couleur.pt -> classes_couleur.json)
        """
        import json
        from pathlib import Path

        import torch
        import torch.nn as nn
        from torchvision import models

        model_path = Path(model_path)
        candidats = [model_path.with_suffix(".classes.json")]
        # modele_couleur.pt -> classes_couleur.json (convention gestion_modeles.py)
        if model_path.stem.startswith("modele_"):
            suffixe = model_path.stem[len("modele_"):]
            candidats.append(model_path.parent / f"classes_{suffixe}.json")
        # Filet de sécurité : n'importe quel classes_*.json à côté du modèle,
        # au cas où une future convention diffère encore des deux ci-dessus.
        candidats.extend(sorted(model_path.parent.glob("classes_*.json")))

        classes_path = next((c for c in candidats if c.exists()), None)

        if classes_path is not None:
            with open(classes_path, "r", encoding="utf-8") as f:
                classes_reelles = json.load(f)
        else:
            # Aucun fichier de classes trouvé (modèle entraîné avant l'ajout de
            # cette sauvegarde, ou fichier manquant) : on retombe sur le
            # comportement par défaut d'ImageFolder (tri alphabétique), qui est
            # presque certainement l'ordre réellement utilisé pendant
            # l'entraînement.
            classes_reelles = sorted(class_names)
            logger.warning(
                "Aucun fichier de classes trouvé pour %s (essayé : %s) : ordre "
                "des classes déduit par tri alphabétique de la config -> %s. "
                "Ré-entraîne ou ré-enregistre le modèle pour figer cet ordre de "
                "façon fiable.",
                model_path, [str(c) for c in candidats], classes_reelles,
            )

        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, len(classes_reelles))
        state_dict = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state_dict)
        model.eval()
        return model, classes_reelles

    def _classify_crop(self, model, crop_bgr, class_names):
        """Classifie un crop OpenCV (BGR) et retourne (classe_prédite, confiance)."""
        import torch
        from PIL import Image
        from torchvision import transforms

        if crop_bgr is None or crop_bgr.size == 0:
            return None, None

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        tensor = transform(pil_img).unsqueeze(0)

        with torch.no_grad():
            outputs = model(tensor)
            probs = torch.softmax(outputs, dim=1)[0]
            idx = int(torch.argmax(probs))

        return class_names[idx], float(probs[idx])

    @property
    def ia_couleur(self):
        if self._ia_couleur is None and self._model_couleur_path:
            from .config import COULEURS_CLASSES
            self._ia_couleur, self._ia_couleur_classes = self._load_classifier(
                self._model_couleur_path, COULEURS_CLASSES)
        return self._ia_couleur

    @property
    def ia_espece(self):
        if self._ia_espece is None and self._model_espece_path:
            from .config import ESPECES_CLASSES
            self._ia_espece, self._ia_espece_classes = self._load_classifier(
                self._model_espece_path, ESPECES_CLASSES)
        return self._ia_espece

    @property
    def ia_sexe(self):
        if self._ia_sexe is None and self._model_sexe_path:
            from .config import SEXES_CLASSES
            self._ia_sexe, self._ia_sexe_classes = self._load_classifier(
                self._model_sexe_path, SEXES_CLASSES)
        return self._ia_sexe

    # ------------------------------------------------------------------
    # Traitement d'une rafale
    # ------------------------------------------------------------------

    def process_burst(self, burst):
        """
        Traite une rafale complète : sélection de la meilleure photo, détection
        + classification en cascade, puis matching temporel (strict, puis
        élargi, filtré par espèce/sexe détectés) avec le tableur d'observations.

        Le résultat contient toujours les clés "canards" et "marques" (listes
        de bbox + classifications), quel que soit le statut final, pour
        permettre l'annotation visuelle des photos (voir utils/annotate.py).

        Args:
            burst: Dictionnaire décrivant la rafale (photos, horodatage, etc.)

        Returns:
            Dictionnaire décrivant le résultat de l'identification.
        """
        best_selection = self.select_best_photo(burst)
        main_photo = best_selection["best_photo"]

        detections = self.detect_canards(main_photo)

        # Association géométrique marque <-> canard (voir
        # _assigner_marques_aux_canards) : chaque marque est désormais liée
        # à un canard précis plutôt que de rester une liste à part au
        # niveau de la rafale entière.
        assignations_marques = self._assigner_marques_aux_canards(
            detections["canards"], detections["marques"])
        marque_vers_canard = {
            a["marque_index"]: {"canard_index": c_idx, "certaine": a["certaine"]}
            for c_idx, assignes in enumerate(assignations_marques)
            for a in assignes
        }

        marques_info = [
            {
                "marque_index": i,
                "bbox": m["bbox"],
                "couleur_detectee": m.get("couleur_detectee"),
                "confiance_couleur": m.get("confiance_couleur"),
                "nuit_probable": m.get("nuit_probable"),
                "couleur_fiable": m.get("couleur_fiable"),
                "canard_assigne": marque_vers_canard.get(i, {}).get("canard_index"),
                "assignation_certaine": marque_vers_canard.get(i, {}).get("certaine"),
            }
            for i, m in enumerate(detections["marques"])
        ]

        if len(detections["canards"]) == 0:
            return {
                "status": "AUCUN_CANARD",
                "rafale": self._build_rafale_info(burst),
                "photo_analysee": main_photo,
                "detections": {"nb_canards": 0, "nb_marques": len(marques_info)},
                "canards": [],
                "marques": marques_info,
                "ia_utilisees": ["detection"],
            }

        nb_canards = len(detections["canards"])
        nb_marques = len(detections["marques"])

        ia_utilisees = ["detection"]
        espece_detectee = sexe_detectee = couleur_detectee = None
        couleur_fiable = None

        if nb_canards == 1:
            canard = detections["canards"][0]
            espece_detectee = canard.get("espece_detectee")
            sexe_detectee = canard.get("sexe_detectee")
            if espece_detectee is not None:
                ia_utilisees.append("espece")
            if sexe_detectee is not None:
                ia_utilisees.append("sexe")

        # Marques assignées géométriquement à L'UNIQUE canard de la photo
        # (si nb_canards == 1, toute marque est nécessairement assignée à
        # ce canard : c'est le seul candidat -- voir
        # _assigner_marques_aux_canards). On ne devine pas de couleur si
        # plusieurs marques distinctes sont assignées au même canard
        # (ambiguïté rare mais réelle : mieux vaut ne rien affirmer).
        marques_du_canard = assignations_marques[0] if nb_canards == 1 else []
        if len(marques_du_canard) == 1:
            m_idx = marques_du_canard[0]["marque_index"]
            marque = detections["marques"][m_idx]
            couleur_detectee = marque.get("couleur_detectee")
            if couleur_detectee is not None:
                ia_utilisees.append("couleur")
                # Photo de nuit probable (flash IR) -> couleur peu fiable.
                # Voir utils/nuit.py. On garde la couleur détectée dans le
                # résultat (elle reste informative) mais ce booléen permet
                # au rapport de la signaler explicitement, et à
                # _build_coherence de ne pas lever une fausse incohérence.
                couleur_fiable = not marque.get("nuit_probable", False)

        # Aucune marque nasale assignée à ce canard : il sera identifié
        # (si identifié) uniquement sur la base espèce/sexe/horodatage,
        # sans aucune confirmation physique de la bague. Un canard non
        # marqué qui passe au bon moment, de la bonne espèce et du bon
        # sexe, peut sinon se voir attribuer silencieusement la bague d'un
        # autre individu : ce flag rend le cas visible partout (rapport,
        # CSV, annotation, journal) au lieu de le laisser passer sans
        # alerte.
        marque_absente = len(marques_du_canard) == 0 if nb_canards == 1 else (nb_marques == 0)

        detections_summary = {"nb_canards": nb_canards, "nb_marques": nb_marques}
        detections_ia = {
            "espece_detectee": espece_detectee,
            "sexe_detectee": sexe_detectee,
            "couleur_detectee": couleur_detectee,
            "couleur_fiable": couleur_fiable,
            "marque_absente": marque_absente,
        }

        if nb_canards > 1:
            return self._resolve_multiple_canards(
                burst, main_photo, detections, detections_summary, ia_utilisees, marques_info,
                assignations_marques,
            )

        # Un seul canard : on garde sa bbox pour l'annotation, quel que soit
        # le statut final (identifié, ambigu, ou non trouvé dans le tableur).
        canard_bbox = detections["canards"][0]["bbox"]

        def _canard_result(status, **extra):
            return {"canard_index": 0, "bbox": canard_bbox, "status": status,
                    **detections_ia, **extra}

        # ========== NOUVELLE LOGIQUE : si pas de marque, on n'identifie PAS ==========
        if marque_absente:
            # Canard sans marque nasale : on ne cherche PAS de correspondance dans le tableur
            # (un canard non marqué ne peut pas être "identifié" = associé à une bague).
            # MAIS on exige quand même que les IAs Espèce/Sexe le classifient.
            
            if espece_detectee is None and self.ia_espece is not None:
                logger.warning("Canard sans marque nasale : IA Espèce disponible mais n'a pas pu classifier")
            if sexe_detectee is None and self.ia_sexe is not None:
                logger.warning("Canard sans marque nasale : IA Sexe disponible mais n'a pas pu classifier")
            
            # Message adapté selon ce qui a été détecté -- même formulation
            # que le résumé multi-canards ("{n} non bagué : {détail}", voir
            # _resolve_multiple_canards) pour rester cohérent d'un cas à
            # l'autre : un chercheur qui lit "1 non bagué : 1 milouin male"
            # sur une rafale à 3 canards doit retrouver la même formulation
            # sur une rafale à 1 seul canard, pas un texte différent.
            if espece_detectee and sexe_detectee:
                message = f"1 canard non bagué : 1 {espece_detectee} {sexe_detectee}"
            elif espece_detectee or sexe_detectee:
                message = (
                    f"1 canard non bagué : 1 {espece_detectee or '?'} "
                    f"{sexe_detectee or '?'} (classification partielle)"
                )
            else:
                message = "1 canard non bagué (classification espèce/sexe indisponible)"
            
            return {
                "status": "NON_MARQUE",
                "rafale": self._build_rafale_info(burst),
                "photo_analysee": main_photo,
                "detections": detections_summary,
                **detections_ia,
                "message": message,
                "canards": [_canard_result("NON_MARQUE")],
                "marques": marques_info,
                "ia_utilisees": ia_utilisees,
            }

        # ========== Le reste du code (matching temporel) n'est atteint QUE si marque_absente=False ==========
        match_strict = self.match_exact_time(
            burst["median_time"], tolerance_seconds=30,
            espece_detectee=espece_detectee, sexe_detectee=sexe_detectee,
        )
        if match_strict["nb_matches"] == 1:
            cohérent = self._build_coherence(match_strict["result"], detections_ia)
            confiance_globale = compute_confiance_globale(
                0.95, canard.get("confiance_espece"), canard.get("confiance_sexe"),
            )
            return {
                "status": "IDENTIFIE",
                "methode": "MATCH_TEMPOREL_STRICT",
                "confiance": 0.95,
                "confiance_globale": confiance_globale,
                **match_strict["result"],
                **detections_ia,
                "cohérent": cohérent,
                "rafale": self._build_rafale_info(burst),
                "photo_analysee": main_photo,
                "detections": detections_summary,
                "canards": [_canard_result("IDENTIFIE", methode="MATCH_TEMPOREL_STRICT",
                                            confiance=0.95, confiance_globale=confiance_globale,
                                            cohérent=cohérent, **match_strict["result"])],
                "marques": marques_info,
                "ia_utilisees": ia_utilisees,
            }

        match_large = self.match_exact_time(
            burst["median_time"], tolerance_seconds=180,
            espece_detectee=espece_detectee, sexe_detectee=sexe_detectee,
        )
        if match_large["nb_matches"] == 1:
            cohérent = self._build_coherence(match_large["result"], detections_ia)
            confiance_globale = compute_confiance_globale(
                0.85, canard.get("confiance_espece"), canard.get("confiance_sexe"),
            )
            return {
                "status": "IDENTIFIE",
                "methode": "MATCH_TEMPOREL_LARGE",
                "confiance": 0.85,
                "confiance_globale": confiance_globale,
                **match_large["result"],
                **detections_ia,
                "cohérent": cohérent,
                "rafale": self._build_rafale_info(burst),
                "photo_analysee": main_photo,
                "detections": detections_summary,
                "canards": [_canard_result("IDENTIFIE", methode="MATCH_TEMPOREL_LARGE",
                                            confiance=0.85, confiance_globale=confiance_globale,
                                            cohérent=cohérent, **match_large["result"])],
                "marques": marques_info,
                "ia_utilisees": ia_utilisees,
            }

        if match_strict["nb_matches"] > 1 or match_large["nb_matches"] > 1:
            nb = match_strict["nb_matches"] if match_strict["nb_matches"] > 1 else match_large["nb_matches"]
            return {
                "status": "AMBIGU",
                "rafale": self._build_rafale_info(burst),
                "photo_analysee": main_photo,
                "detections": detections_summary,
                **detections_ia,
                "message": f"{nb} observations temporellement proches malgré le filtrage espèce/sexe",
                "candidats": match_large.get("candidates", match_strict.get("candidates", [])),
                "canards": [_canard_result("AMBIGU")],
                "marques": marques_info,
                "ia_utilisees": ia_utilisees,
            }

        return {
            "status": "NON_TROUVE_TABLEUR",
            "rafale": self._build_rafale_info(burst),
            "photo_analysee": main_photo,
            "detections": detections_summary,
            **detections_ia,
            "message": "Aucune observation correspondante dans le tableur",
            "canards": [_canard_result("NON_TROUVE_TABLEUR")],
            "marques": marques_info,
            "ia_utilisees": ia_utilisees,
        }

    def _resolve_multiple_canards(self, burst, main_photo, detections,
                                   detections_summary, ia_utilisees, marques_info,
                                   assignations_marques):
        """
        Tente une identification individuelle de chaque canard détecté quand
        plusieurs sont présents sur la photo, en filtrant le tableur avec la
        classification espèce/sexe propre à CE canard (au lieu d'abandonner
        globalement sur MULTIPLE_CANARDS sans utiliser les IAs).

        Args:
            burst: Dictionnaire décrivant la rafale.
            main_photo: Chemin de la photo analysée.
            detections: Résultat de detect_canards (contient déjà les
                classifications espèce/sexe par canard, calculées en amont).
            detections_summary: {"nb_canards": ..., "nb_marques": ...}.
            ia_utilisees: Liste des IA déjà utilisées (ex. ["detection"]).
            assignations_marques: Résultat de _assigner_marques_aux_canards
                (une liste de marques assignées par index de canard) :
                permet de savoir, PAR CANARD, quelle marque nasale (et donc
                quelle couleur) lui appartient réellement, au lieu de
                traiter les marques comme une simple liste globale à la
                rafale sans lien avec un individu précis.

        Returns:
            Dictionnaire de résultat pour la rafale, avec le détail par
            canard dans la clé "canards".
        """
        ia_utilisees = list(ia_utilisees)
        canard_results = []

        for idx, canard in enumerate(detections["canards"]):
            c_espece = canard.get("espece_detectee")
            c_sexe = canard.get("sexe_detectee")
            if c_espece is not None and "espece" not in ia_utilisees:
                ia_utilisees.append("espece")
            if c_sexe is not None and "sexe" not in ia_utilisees:
                ia_utilisees.append("sexe")

            # Marque(s) assignée(s) géométriquement à CE canard précis (voir
            # _assigner_marques_aux_canards) : contrairement à avant, la
            # couleur n'est plus ignorée dans ce chemin de code.
            marques_ce_canard = assignations_marques[idx] if idx < len(assignations_marques) else []
            c_couleur = c_couleur_fiable = None
            if len(marques_ce_canard) == 1:
                m_idx = marques_ce_canard[0]["marque_index"]
                marque = detections["marques"][m_idx]
                c_couleur = marque.get("couleur_detectee")
                if c_couleur is not None:
                    if "couleur" not in ia_utilisees:
                        ia_utilisees.append("couleur")
                    c_couleur_fiable = not marque.get("nuit_probable", False)
            marque_absente = len(marques_ce_canard) == 0

            canard_ia = {
                "espece_detectee": c_espece,
                "confiance_espece": canard.get("confiance_espece"),
                "sexe_detectee": c_sexe,
                "confiance_sexe": canard.get("confiance_sexe"),
                "couleur_detectee": c_couleur,
                "couleur_fiable": c_couleur_fiable,
                "marque_absente": marque_absente,
            }

            # ========== NOUVELLE LOGIQUE : si pas de marque, on skip le matching ==========
            if marque_absente:
                # Ce canard n'a pas de marque nasale détectée : on ne cherche PAS à l'identifier
                # (pas de match temporel = pas de bague attribuée). On garde juste sa classification.
                
                if c_espece and c_sexe:
                    diagnostic = f"Canard non marqué : {c_espece} {c_sexe} (aucune bague attribuable)"
                elif c_espece or c_sexe:
                    diagnostic = f"Canard non marqué : {c_espece or '?'} {c_sexe or '?'} (classification partielle)"
                else:
                    diagnostic = "Canard non marqué sans classification espèce/sexe"
                
                canard_results.append({
                    "canard_index": idx,
                    "bbox": canard["bbox"],
                    "status": "NON_MARQUE",
                    "methode": None,
                    "confiance": None,
                    "confiance_globale": None,
                    "diagnostic": diagnostic,
                    **canard_ia,
                    "bague": None,
                    "cohérent": None,
                })
                continue  # On passe au canard suivant SANS faire de match temporel

            # ========== Matching temporel (code existant, uniquement si marque présente) ==========
            chosen, methode, confiance = None, None, None
            diagnostic = None

            match_strict = self.match_exact_time(
                burst["median_time"], tolerance_seconds=30,
                espece_detectee=c_espece, sexe_detectee=c_sexe,
            )
            if match_strict["nb_matches"] == 1:
                chosen, methode, confiance = match_strict["result"], "MATCH_TEMPOREL_STRICT", 0.90
            else:
                match_large = self.match_exact_time(
                    burst["median_time"], tolerance_seconds=180,
                    espece_detectee=c_espece, sexe_detectee=c_sexe,
                )
                if match_large["nb_matches"] == 1:
                    chosen, methode, confiance = match_large["result"], "MATCH_TEMPOREL_LARGE", 0.75
                elif match_large["nb_matches"] == 0:
                    diagnostic = (
                        "Aucune observation du tableur ne correspond à ce créneau "
                        f"(espèce détectée : {c_espece or 'inconnue'}, sexe détecté : {c_sexe or 'inconnu'})."
                    )
                else:
                    candidats = match_large.get("candidates", [])
                    bagues = ", ".join(str(c.get("BAGUE", "?")) for c in candidats[:5])
                    diagnostic = (
                        f"{match_large['nb_matches']} observations du tableur restent compatibles avec ce "
                        f"créneau même après filtrage par espèce/sexe (bagues candidates : {bagues})."
                    )

            confiance_globale = (
                compute_confiance_globale(confiance, canard_ia["confiance_espece"], canard_ia["confiance_sexe"])
                if chosen else None
            )

            canard_results.append({
                "canard_index": idx,
                "bbox": canard["bbox"],
                "status": "IDENTIFIE" if chosen else "NON_RESOLU",
                "methode": methode,
                "confiance": confiance,
                "confiance_globale": confiance_globale,
                "diagnostic": diagnostic,
                **canard_ia,
                **(chosen or {}),
                "cohérent": self._build_coherence(chosen, canard_ia) if chosen else None,
            })

        # Une même bague ne peut pas correspondre à 2 canards sur la même
        # photo : si ça arrive, c'est que le filtrage espèce/sexe n'a pas
        # suffi à distinguer les individus (classification bruitée, ou
        # plusieurs observations très proches dans le tableur).
        bagues_vues = {}
        for c in canard_results:
            if c["status"] == "IDENTIFIE":
                bagues_vues.setdefault(c.get("bague"), []).append(c)
        for bague, groupe in bagues_vues.items():
            if len(groupe) <= 1:
                continue

            # Parmi les canards en conflit sur cette bague, ceux qui portent
            # réellement une marque nasale détectée (marque_absente == False)
            # sont les seuls candidats plausibles : une coïncidence
            # espèce/sexe/horodatage sans marque physique visible n'est pas
            # une preuve d'identité. Si un seul canard du groupe est dans ce
            # cas, c'est lui le vrai porteur -- les autres redeviennent
            # NON_RESOLU au lieu de rester marqués comme "identifiés" avec
            # sa bague.
            avec_marque = [c for c in groupe if not c.get("marque_absente")]
            if len(avec_marque) == 1:
                porteur = avec_marque[0]
                for c in groupe:
                    if c is porteur:
                        continue
                    c["status"] = "NON_RESOLU"
                    c["bague"] = None
                    c["diagnostic"] = (
                        f"Ce canard correspondait à la bague {bague} par espèce/sexe/horodatage, "
                        "mais aucune marque nasale n'a été détectée sur lui, contrairement à un "
                        "autre canard de la même photo : la bague lui a été attribuée à lui, pas "
                        "à ce canard."
                    )
                continue

            for c in groupe:
                c["status"] = "AMBIGU_DOUBLON"
                c["diagnostic"] = (
                    f"Ce canard a été apparié à la bague {bague}, mais un autre canard de la "
                    "même photo y a aussi été apparié : la classification espèce/sexe n'a pas "
                    "suffi à les distinguer, vérification manuelle nécessaire."
                )

        nb_identifies = sum(1 for c in canard_results if c["status"] == "IDENTIFIE")
        nb_non_marques = sum(1 for c in canard_results if c["status"] == "NON_MARQUE")
        nb_total = len(canard_results)

        if nb_identifies == nb_total:
            status = "IDENTIFIE_MULTIPLE"
        elif nb_identifies > 0:
            status = "PARTIELLEMENT_IDENTIFIE"
        else:
            status = "MULTIPLE_CANARDS"

        # Construction du message simplifié
        parts_message = []

        if nb_identifies > 0:
            suffixe = "s" if nb_identifies > 1 else ""
            parts_message.append(
                f"{nb_identifies} canard{suffixe} bagué{suffixe} identifié{suffixe}"
            )

        if nb_non_marques > 0:
            # Groupe les canards non marqués par espèce + sexe
            compte_non_marques = Counter()
            for c in canard_results:
                if c["status"] == "NON_MARQUE":
                    esp = c.get("espece_detectee") or "?"
                    sx = c.get("sexe_detectee") or "?"
                    compte_non_marques[(esp, sx)] += 1
            
            if compte_non_marques:
                detail = ", ".join(f"{n} {esp} {sx}" for (esp, sx), n in compte_non_marques.items())
                parts_message.append(f"{nb_non_marques} non bagué{'s' if nb_non_marques > 1 else ''} : {detail}")
            else:
                parts_message.append(f"{nb_non_marques} non bagué{'s' if nb_non_marques > 1 else ''}")

        # "non_resolus" = canards qui PORTENT une marque nasale (donc
        # bagués physiquement, marque_absente == False -- voir plus haut,
        # ce sont les seuls candidats au matching temporel) mais dont
        # l'identification a échoué (bague introuvable/ambiguë dans le
        # tableur, ou conflit résolu en défaveur de ce canard). "Bagué non
        # identifié" décrit donc plus précisément ce cas que "non résolu",
        # et se distingue clairement de "non bagué" (NON_MARQUE, pas de
        # marque du tout) juste au-dessus dans le message.
        non_resolus = [c for c in canard_results if c["status"] not in ("IDENTIFIE", "NON_MARQUE")]
        if non_resolus:
            resume = self._resume_espece_sexe(non_resolus)
            n = len(non_resolus)
            suffixe = "s" if n > 1 else ""
            label = f"{n} canard{suffixe} bagué{suffixe} non identifié{suffixe}"
            parts_message.append(f"{label} : {resume}" if resume else label)

        message = " et ".join(parts_message) if parts_message else "Aucun canard identifié"

        return {
            "status": status,
            "rafale": self._build_rafale_info(burst),
            "photo_analysee": main_photo,
            "detections": detections_summary,
            "canards": canard_results,
            "marques": marques_info,
            "message": message,
            "ia_utilisees": ia_utilisees,
        }

    @staticmethod
    def _resume_espece_sexe(canard_results):
        """
        Construit un résumé du type "2 milouin male, 1 milouin femelle" à
        partir des classifications espèce/sexe individuelles de chaque
        canard (canard_ia), qu'il ait pu être rattaché à une bague ou non.

        Returns:
            str ou None si aucun canard n'a de classification espèce+sexe
            exploitable.
        """
        compte = Counter()
        for c in canard_results:
            espece = c.get("espece_detectee")
            sexe = c.get("sexe_detectee")
            if espece and sexe:
                compte[(espece, sexe)] += 1
        if not compte:
            return None
        return ", ".join(f"{n} {espece} {sexe}" for (espece, sexe), n in compte.items())

    def detect_canards(self, photo_path):
        """
        Détecte les canards et les marques nasales sur une photo via YOLO, puis
        classifie chaque crop détecté avec les IAs disponibles (espèce/sexe pour
        les canards, couleur pour les marques).

        Le mapping des classes se fait par NOM (via result.names), pas par index
        numérique, pour rester correct quel que soit l'ordre déclaré dans data.yaml.

        Seuils de confiance et résolution d'inférence : voir
        CONFIDENCE_THRESHOLD_MARQUE, CONFIDENCE_THRESHOLD_CANARD et
        DETECTION_IMGSZ dans config.py. YOLO ne permet pas de passer un
        seuil de confiance différent par classe en un seul appel : on
        appelle donc predict() avec le PLUS BAS des deux seuils (pour ne
        rien perdre côté YOLO), puis on filtre nous-mêmes chaque détection
        avec le seuil propre à sa classe ci-dessous.

        Returns:
            Dictionnaire avec les listes de canards et de marques détectés,
            chacun enrichi des champs de classification disponibles.
        """
        seuil_plancher = min(CONFIDENCE_THRESHOLD_MARQUE, CONFIDENCE_THRESHOLD_CANARD)
        results = self.ia_detection.predict(
            photo_path, conf=seuil_plancher, imgsz=DETECTION_IMGSZ, verbose=False)

        img = None  # chargée paresseusement, seulement si une classification est nécessaire

        canards_brutes = []
        marques_brutes = []

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                class_name = result.names[cls_id]
                bbox = [int(x1), int(y1), int(x2 - x1), int(y2 - y1)]

                if class_name == DETECTION_CLASSE_MARQUE:
                    if conf < CONFIDENCE_THRESHOLD_MARQUE:
                        continue
                    marques_brutes.append({"bbox": bbox, "confiance": conf})
                elif class_name == DETECTION_CLASSE_CANARD:
                    if conf < CONFIDENCE_THRESHOLD_CANARD:
                        continue
                    canards_brutes.append({"bbox": bbox, "confiance": conf})
                # Toute autre classe (ancien schéma non migré, etc.) est ignorée.

        # --- Passe 2 : "detect-then-zoom" -----------------------------------
        # Même avec un seuil abaissé et DETECTION_IMGSZ augmenté, le détecteur
        # peut simplement ne rien voir sur certains canards : sur l'image
        # entière, une marque ne fait qu'une quinzaine de pixels, et ça
        # empire quand plusieurs canards se partagent le cadre. Une fois les
        # canards localisés (fiable -- ce sont de gros objets), on peut se
        # permettre de revenir dessus : pour chaque canard qui n'a encore
        # AUCUNE marque à proximité après la passe 1, on recadre précisément
        # autour de lui et on relance le détecteur sur ce crop zoomé, où la
        # marque occupe une bien plus grande fraction de l'image. Coût : un
        # appel YOLO de plus par canard sans marque -- proportionnel au
        # problème qu'on cherche à corriger, pas systématique à chaque canard.
        canards_sans_marque_proche = [
            c for c in canards_brutes
            if not self._a_marque_a_proximite(c["bbox"], marques_brutes)
        ]
        if canards_sans_marque_proche:
            if img is None:
                img = cv2.imread(photo_path)
            for canard in canards_sans_marque_proche:
                marques_brutes.extend(self._detecter_marques_zoom(img, canard["bbox"]))

        # Déduplication : YOLO applique déjà un NMS interne, mais celui-ci est
        # parfois insuffisant en pratique (deux boîtes quasi identiques sur le
        # même objet peuvent survivre avec des confiances différentes, ex.
        # 0.74 et 0.40 observés sur une même marque nasale). On applique donc
        # un filtre de sécurité supplémentaire par recouvrement (IoU), classe
        # par classe, en ne gardant que la détection la plus confiante de
        # chaque groupe de boîtes qui se chevauchent fortement.
        canards_brutes = self._deduplicate_detections(canards_brutes)
        marques_brutes = self._deduplicate_detections(marques_brutes)

        canards = []
        marques = []

        for m in marques_brutes:
            x1, y1, w, h = m["bbox"]
            x2, y2 = x1 + w, y1 + h
            marque = {"bbox": m["bbox"], "confiance": m["confiance"],
                      "couleur_detectee": None, "confiance_couleur": None,
                      "nuit_probable": None, "couleur_fiable": None}
            if self.ia_couleur is not None:
                if img is None:
                    img = cv2.imread(photo_path)
                crop = self._crop_bbox(img, x1, y1, x2, y2)
                couleur, conf_couleur = self._classify_crop(
                    self.ia_couleur, crop, self._ia_couleur_classes)
                marque["couleur_detectee"] = couleur
                marque["confiance_couleur"] = conf_couleur

                # La nuit, le flash IR rend l'image quasi grise : la couleur
                # détectée n'est alors pas fiable (même heuristique que
                # interface_preparation_ia.py, voir utils/nuit.py). On ne
                # bloque jamais rien ici : on continue de classifier et de
                # matcher normalement, mais ce champ permet au rapport de
                # signaler explicitement une couleur peu fiable au lieu de
                # la présenter avec la même confiance qu'en plein jour.
                nuit_probable = est_photo_de_nuit_bgr(crop)
                marque["nuit_probable"] = nuit_probable
                marque["couleur_fiable"] = couleur is not None and not nuit_probable
            marques.append(marque)

        for c in canards_brutes:
            x1, y1, w, h = c["bbox"]
            x2, y2 = x1 + w, y1 + h
            canard = {"bbox": c["bbox"], "confiance": c["confiance"],
                      "espece_detectee": None, "confiance_espece": None,
                      "sexe_detectee": None, "confiance_sexe": None}
            if self.ia_espece is not None or self.ia_sexe is not None:
                if img is None:
                    img = cv2.imread(photo_path)
                crop = self._crop_bbox(img, x1, y1, x2, y2)

                if self.ia_espece is not None:
                    espece, conf_espece = self._classify_crop(
                        self.ia_espece, crop, self._ia_espece_classes)
                    canard["espece_detectee"] = espece
                    canard["confiance_espece"] = conf_espece

                if self.ia_sexe is not None:
                    sexe, conf_sexe = self._classify_crop(
                        self.ia_sexe, crop, self._ia_sexe_classes)
                    canard["sexe_detectee"] = sexe
                    canard["confiance_sexe"] = conf_sexe
            canards.append(canard)

        return {"canards": canards, "marques": marques}

    @staticmethod
    def _iou(bbox_a, bbox_b):
        """Calcule l'IoU (intersection sur union) entre deux bbox au format [x, y, w, h]."""
        ax1, ay1, aw, ah = bbox_a
        bx1, by1, bw, bh = bbox_b
        ax2, ay2 = ax1 + aw, ay1 + ah
        bx2, by2 = bx1 + bw, by1 + bh

        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)

        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h

        if inter_area == 0:
            return 0.0

        area_a = aw * ah
        area_b = bw * bh
        union_area = area_a + area_b - inter_area
        if union_area <= 0:
            return 0.0

        return inter_area / union_area

    # Marge de tolérance (en pixels) pour l'association géométrique
    # marque_nasale <-> canard : une bbox "canard" du détecteur colle
    # rarement pixel-parfait à l'animal réel, donc on l'élargit un peu
    # avant de tester si elle contient le centre d'une marque.
    MARGE_ASSOCIATION_MARQUE = 15

    @staticmethod
    def _bbox_center(bbox):
        x, y, w, h = bbox
        return (x + w / 2.0, y + h / 2.0)

    @staticmethod
    def _distance(point_a, point_b):
        return ((point_a[0] - point_b[0]) ** 2 + (point_a[1] - point_b[1]) ** 2) ** 0.5

    @staticmethod
    def _bbox_contains_point(bbox, point, margin=0):
        x, y, w, h = bbox
        px, py = point
        return (x - margin) <= px <= (x + w + margin) and (y - margin) <= py <= (y + h + margin)

    @staticmethod
    def _expand_bbox(bbox, margin):
        x, y, w, h = bbox
        return [x - margin, y - margin, w + 2 * margin, h + 2 * margin]

    @staticmethod
    def _overlap_ratio(bbox_marque, bbox_canard):
        """
        Fraction de l'aire de `bbox_marque` recouverte par `bbox_canard`
        (intersection / aire de la marque -- pas une IoU classique : le
        canard est presque toujours beaucoup plus grand que la marque, donc
        diviser par l'union écraserait le ratio même quand la marque est
        entièrement contenue dans le canard).

        Retourne une valeur entre 0.0 (aucun recouvrement) et 1.0 (marque
        entièrement recouverte par la bbox canard).
        """
        mx, my, mw, mh = bbox_marque
        cx, cy, cw, ch = bbox_canard
        mx2, my2 = mx + mw, my + mh
        cx2, cy2 = cx + cw, cy + ch

        inter_x1, inter_y1 = max(mx, cx), max(my, cy)
        inter_x2, inter_y2 = min(mx2, cx2), min(my2, cy2)
        inter_w, inter_h = max(0, inter_x2 - inter_x1), max(0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h

        marque_area = mw * mh
        if marque_area <= 0:
            return 0.0
        return inter_area / marque_area

    # Si le deuxième meilleur canard recouvre au moins cette fraction du
    # recouvrement du meilleur, on considère l'assignation ambiguë (deux
    # canards rapprochés/chevauchants plausibles) plutôt que de trancher
    # silencieusement.
    RATIO_AMBIGUITE = 0.6

    @classmethod
    def _assigner_marques_aux_canards(cls, canards, marques):
        """
        Associe géométriquement chaque marque nasale détectée au canard
        auquel elle appartient, sur la photo courante.

        PROBLÈME RÉSOLU : auparavant, la liste "marques" restait globale à
        la rafale entière, sans aucun lien avec un canard en particulier
        (voir la discussion produit : "il n'y a aucune association
        géométrique entre une bbox marque_nasale détectée et le canard
        auquel elle appartient"). Résultat concret : quand plusieurs
        canards étaient présents sur une photo, la couleur de marque
        n'était même pas transmise à _resolve_multiple_canards, et pour
        un seul canard, une deuxième marque nasale détectée par erreur
        (faux positif du détecteur) suffisait à ce que la couleur ne soit
        plus utilisée du tout (nb_marques == 1 devenait faux).

        Méthode (recouvrement, pas point central) : pour chaque marque, on
        calcule la fraction de son aire recouverte par la bbox de chaque
        canard (élargie de MARGE_ASSOCIATION_MARQUE px, car les bbox de
        détection sont rarement pixel-parfaites), et on l'assigne au canard
        avec le plus grand recouvrement. Un simple test "le centre de la
        marque est-il dans la bbox du canard ?" est trop fragile : sur des
        canards groupés/chevauchants, un centre peut tomber dans deux bbox
        à la fois (ou juste à l'extérieur de la bonne bbox par imprécision
        du détecteur) sans que ça ressorte. Le recouvrement capture mieux
        "à quel canard appartient réellement cette marque", et permet de
        détecter l'ambiguïté quand deux canards se disputent sérieusement
        la même marque (voir RATIO_AMBIGUITE) au lieu de trancher en
        silence.

        Si aucun canard ne recouvre la marque, on retombe sur le canard le
        plus proche (distance centre à centre) plutôt que de la perdre à
        cause d'un léger décalage de cadrage -- mais cette assignation est
        alors marquée "certaine": False, pour rester traçable.

        Args:
            canards: Liste de dicts {"bbox": [x, y, w, h], ...}.
            marques: Liste de dicts {"bbox": [x, y, w, h], ...}.

        Returns:
            Liste (même longueur que `canards`) de listes de dicts
            {"marque_index": int, "certaine": bool} -- les marques
            assignées à chaque canard, dans l'ordre des indices de
            `marques`.
        """
        assignations = [[] for _ in canards]
        if not canards or not marques:
            return assignations

        for m_idx, marque in enumerate(marques):
            marque_bbox = marque["bbox"]

            recouvrements = [
                cls._overlap_ratio(marque_bbox, cls._expand_bbox(canard["bbox"], cls.MARGE_ASSOCIATION_MARQUE))
                for canard in canards
            ]
            meilleur = max(recouvrements)

            if meilleur > 0:
                choisi = max(range(len(canards)), key=lambda c_idx: recouvrements[c_idx])
                autres = sorted(
                    (r for i, r in enumerate(recouvrements) if i != choisi), reverse=True
                )
                deuxieme = autres[0] if autres else 0.0
                certaine = deuxieme < cls.RATIO_AMBIGUITE * meilleur
            else:
                # Aucun recouvrement : fallback au canard le plus proche
                # (distance centre à centre), non certain.
                centre = cls._bbox_center(marque_bbox)
                choisi = min(
                    range(len(canards)),
                    key=lambda c_idx: cls._distance(centre, cls._bbox_center(canards[c_idx]["bbox"])),
                )
                certaine = False

            assignations[choisi].append({"marque_index": m_idx, "certaine": certaine})

        return assignations

    @classmethod
    def _a_marque_a_proximite(cls, canard_bbox, marques):
        """
        True si au moins une marque de `marques` recouvre (même
        partiellement) la bbox du canard élargie de MARGE_ASSOCIATION_MARQUE.
        Utilisé pour décider, avant la passe "zoom", quels canards ont déjà
        une marque candidate et n'ont donc pas besoin d'un deuxième passage.
        """
        canard_elargi = cls._expand_bbox(canard_bbox, cls.MARGE_ASSOCIATION_MARQUE)
        return any(cls._overlap_ratio(m["bbox"], canard_elargi) > 0 for m in marques)

    # Marge ajoutée autour de la bbox d'un canard pour former le crop de la
    # passe "zoom" (fraction de sa plus grande dimension) : la tête/bec, où
    # se trouve la marque nasale, déborde souvent légèrement de la bbox
    # canard détectée par YOLO.
    MARGE_ZOOM_CANARD_FRACTION = 0.4

    # En dessous de cette taille (px, sur le plus petit côté du crop),
    # zoomer n'apporte plus rien d'utile -- le crop est déjà trop petit pour
    # que la marque y soit mieux résolue qu'à l'échelle de l'image entière.
    TAILLE_MIN_CROP_ZOOM = 20

    def _detecter_marques_zoom(self, img, canard_bbox):
        """
        Passe "detect-then-zoom" : relance la détection YOLO sur un crop
        recadré précisément autour d'UN canard, à une résolution où la
        marque nasale occupe une bien plus grande fraction de l'image que
        sur la photo entière (voir la note dans detect_canards). Ne garde
        que les détections de classe marque_nasale.

        Args:
            img: Image complète (déjà chargée par cv2.imread), ou None.
            canard_bbox: [x, y, w, h] du canard, dans le référentiel de
                l'image complète.

        Returns:
            Liste de dicts {"bbox": [x, y, w, h], "confiance": float},
            bbox reprojetées dans le référentiel de l'image complète (pas
            du crop) pour rester directement comparables aux détections de
            la passe 1.
        """
        if img is None:
            return []

        x, y, w, h = canard_bbox
        marge = int(max(w, h) * self.MARGE_ZOOM_CANARD_FRACTION)
        img_h, img_w = img.shape[:2]
        x1 = max(0, x - marge)
        y1 = max(0, y - marge)
        x2 = min(img_w, x + w + marge)
        y2 = min(img_h, y + h + marge)

        if (x2 - x1) < self.TAILLE_MIN_CROP_ZOOM or (y2 - y1) < self.TAILLE_MIN_CROP_ZOOM:
            return []

        crop = img[y1:y2, x1:x2]

        # `predict` accepte directement un tableau numpy (pas seulement un
        # chemin de fichier) : pas besoin d'écrire le crop sur disque.
        results = self.ia_detection.predict(
            crop, conf=CONFIDENCE_THRESHOLD_MARQUE, imgsz=DETECTION_IMGSZ_ZOOM, verbose=False)

        marques_trouvees = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                cls_id = int(box.cls[0])
                if result.names[cls_id] != DETECTION_CLASSE_MARQUE:
                    continue
                conf = float(box.conf[0])
                if conf < CONFIDENCE_THRESHOLD_MARQUE:
                    continue
                bx1, by1, bx2, by2 = box.xyxy[0].cpu().numpy()
                # Reprojection crop -> image complète : décalage de l'origine.
                bbox = [int(bx1) + x1, int(by1) + y1, int(bx2 - bx1), int(by2 - by1)]
                marques_trouvees.append({"bbox": bbox, "confiance": conf})

        return marques_trouvees

    @classmethod
    def _deduplicate_detections(cls, detections, iou_threshold=0.5):
        """
        Supprime les détections en double (même objet détecté deux fois) en ne
        gardant, pour chaque groupe de boîtes qui se chevauchent au-delà du
        seuil d'IoU, que celle avec la plus grande confiance.

        Args:
            detections: Liste de dicts {"bbox": [x, y, w, h], "confiance": float}.
            iou_threshold: Seuil d'IoU au-delà duquel deux boîtes sont
                considérées comme la même détection (défaut 0.5).

        Returns:
            Liste filtrée des détections gardées.
        """
        if len(detections) <= 1:
            return detections

        restantes = sorted(detections, key=lambda d: d["confiance"], reverse=True)
        gardees = []

        for det in restantes:
            if any(cls._iou(det["bbox"], gardee["bbox"]) > iou_threshold for gardee in gardees):
                continue
            gardees.append(det)

        return gardees

    @staticmethod
    def _crop_bbox(img, x1, y1, x2, y2, margin=10):
        """Découpe une bbox (avec marge) dans une image OpenCV, en restant dans les bornes."""
        if img is None:
            return None
        h, w = img.shape[:2]
        x1 = max(0, int(x1) - margin)
        y1 = max(0, int(y1) - margin)
        x2 = min(w, int(x2) + margin)
        y2 = min(h, int(y2) + margin)
        return img[y1:y2, x1:x2]

    def select_best_photo(self, burst):
        """
        Sélectionne la photo la plus nette d'une rafale.

        Returns:
            Dictionnaire avec la meilleure photo et les scores de toutes les photos.
        """
        scores = [
            {"path": photo_path, "score": self._calculate_photo_quality(photo_path)}
            for photo_path in burst["photos"]
        ]
        best = max(scores, key=lambda x: x["score"])

        return {"best_photo": best["path"], "all_scores": scores}

    def _calculate_photo_quality(self, photo_path):
        """
        Calcule un score de netteté via la variance du Laplacien.

        Returns:
            Score de netteté (float), 0 si la photo est illisible.
        """
        try:
            img_cv = cv2.imread(photo_path, cv2.IMREAD_GRAYSCALE)
            if img_cv is None:
                return 0
            return cv2.Laplacian(img_cv, cv2.CV_64F).var()
        except Exception:
            logger.debug("Impossible de calculer la netteté (image illisible)", exc_info=True)
            return 0

    def match_exact_time(self, photo_datetime, tolerance_seconds=30,
                          espece_detectee=None, sexe_detectee=None):
        """
        Recherche les observations du tableur proches d'un horodatage donné.

        Si plusieurs observations correspondent temporellement, on affine
        d'abord par espèce détectée (colonne ESPECE), puis par sexe détecté
        (colonne SEXE) si toujours ambigu — uniquement quand le filtrage ne
        vide pas complètement les candidats (sinon on le conserve tel quel,
        pour ne pas perdre une bonne correspondance sur une classification
        potentiellement bruitée).

        Args:
            photo_datetime: Horodatage de la photo.
            tolerance_seconds: Tolérance de matching en secondes.
            espece_detectee: Espèce détectée par l'IA Espèce (optionnel).
            sexe_detectee: Sexe détecté par l'IA Sexe (optionnel).

        Returns:
            Dictionnaire avec le nombre de correspondances et le résultat associé.
        """
        tolerance = pd.Timedelta(seconds=tolerance_seconds)
        matches = self.tableur[abs(self.tableur["DATETIME"] - photo_datetime) <= tolerance]

        # Conversion du vocabulaire IA ("milouin"/"male") vers les codes du
        # tableur ("AYTFER"/"M") -- voir ESPECE_CODE_TABLEUR / SEXE_CODE_TABLEUR
        # dans config.py. Comparer directement les deux vocabulaires (comme le
        # faisait ce code avant) ne correspond JAMAIS : ce filtrage était donc
        # un no-op silencieux, quel que soit le nombre de candidats.
        espece_code = ESPECE_CODE_TABLEUR.get((espece_detectee or "").lower())
        sexe_code = SEXE_CODE_TABLEUR.get((sexe_detectee or "").lower())

        if len(matches) > 1 and espece_code and "ESPECE" in matches.columns:
            filtered = matches[matches["ESPECE"].astype(str).str.upper() == espece_code]
            if len(filtered) > 0:
                matches = filtered

        if len(matches) > 1 and sexe_code and "SEXE" in matches.columns:
            filtered = matches[matches["SEXE"].astype(str).str.upper() == sexe_code]
            if len(filtered) > 0:
                matches = filtered

        # Un candidat temporel UNIQUE n'est pas automatiquement CE canard :
        # s'il ne correspond ni à l'espèce ni au sexe détectés pour lui, il
        # appartient probablement à un autre canard de la même rafale
        # (typiquement le seul individu bagué du groupe). Avant cette
        # vérification, ce candidat unique était accepté sans condition pour
        # CHAQUE canard interrogé : sur une photo à plusieurs canards, tous
        # héritaient silencieusement de la même bague dès qu'une seule
        # observation du tableur tombait dans la fenêtre temporelle (voir
        # _resolve_multiple_canards et la déduplication par bague qui
        # transforme ensuite ces doublons en AMBIGU_DOUBLON -> 0 identifiés).
        if len(matches) == 1:
            row = matches.iloc[0]
            if espece_code and "ESPECE" in matches.columns:
                if str(row.get("ESPECE", "")).strip().upper() != espece_code:
                    return {"nb_matches": 0}
            if sexe_code and "SEXE" in matches.columns:
                valeur_sexe = str(row.get("SEXE", "")).strip().upper()
                if valeur_sexe and valeur_sexe != sexe_code:
                    return {"nb_matches": 0}

        nb = len(matches)

        if nb == 0:
            return {"nb_matches": 0}

        if nb == 1:
            row = matches.iloc[0]
            return {
                "nb_matches": 1,
                "result": {
                    "bague": row.get("BAGUE", "") or row.get("CODE MARQUE", "INCONNU"),
                    "espece": row.get("ESPECE", "INCONNU"),
                    "couleur_marque": row.get("COULEUR_MARQUE") or row.get("COULEUR MARQUE", "INCONNU"),
                    "sexe": row.get("SEXE", "N/A"),
                    "code_marque": row.get("CODE MARQUE", ""),
                },
            }

        return {"nb_matches": nb, "candidates": matches.to_dict("records")}

    @staticmethod
    def _build_coherence(tableur_result, detections_ia):
        """
        Compare les valeurs issues du tableur (déclarées) aux valeurs détectées
        par les IAs, pour signaler les divergences.

        Returns:
            None si aucune IA de classification n'a produit de résultat à
            comparer, sinon un dict {champ: bool} — True si cohérent.
        """
        checks = {}

        espece_detectee = detections_ia.get("espece_detectee")
        if espece_detectee is not None:
            checks["espece"] = str(tableur_result.get("espece", "")).strip().lower() == espece_detectee.lower()

        sexe_detectee = detections_ia.get("sexe_detectee")
        if sexe_detectee is not None:
            checks["sexe"] = str(tableur_result.get("sexe", "")).strip().lower() == sexe_detectee.lower()

        couleur_detectee = detections_ia.get("couleur_detectee")
        couleur_fiable = detections_ia.get("couleur_fiable")
        if couleur_detectee is not None and couleur_fiable is not False:
            checks["couleur_marque"] = (
                str(tableur_result.get("couleur_marque", "")).strip().lower() == couleur_detectee.lower()
            )

        if not checks:
            return None
        return checks

    def _build_rafale_info(self, burst):
        """Construit les infos de rafale pour l'export JSON."""
        return {
            "nb_photos": burst["nb_photos"],
            "photos": burst["photos"],
            "timestamps": burst["timestamps"],  # requis par export_tableau_canards()
            "median_time": str(burst["median_time"]),
            "start_time": str(burst["start_time"]),
            "end_time": str(burst["end_time"]),
            "duration": burst["duration"],
        }