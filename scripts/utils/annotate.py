# -*- coding: utf-8 -*-
"""
Annotation visuelle des photos traitées par le pipeline.

Dessine, pour chaque rafale, les bbox détectées (canards + marques nasales)
sur la photo analysée, avec un label résumant ce que le pipeline a conclu
pour chaque canard (bague / espèce / sexe / confiance), afin de pouvoir
vérifier visuellement les résultats sans ouvrir le JSON.

Les photos sont triées dans 3 sous-dossiers selon ce qu'un chercheur doit
en faire :
    - a_verifier/    : nécessite une relecture humaine (ambigu, doublon,
                        plusieurs canards non résolus, ou identifié mais
                        avec une confiance faible)
    - identifies/     : identification jugée fiable, pas d'action requise
    - non_concluant/  : aucun canard détecté, ou aucune correspondance
                        dans le tableur
"""

from pathlib import Path

import cv2

from .confiance import niveau_confiance

# Couleurs BGR (OpenCV) par NIVEAU DE CONFIANCE (voir utils/confiance.py),
# utilisées pour les canards identifiés.
COULEUR_PAR_NIVEAU = {
    "haute": (0, 170, 0),      # vert : confiance_globale >= 0.90
    "moyenne": (0, 165, 255),  # orange : 0.70 <= confiance_globale < 0.90
    "faible": (0, 60, 255),    # rouge-orangé : confiance_globale < 0.70
    "inconnu": (128, 128, 128),
}
# Couleurs par statut, pour les canards non identifiés (pas de confiance à colorer).
COULEUR_PAR_STATUT = {
    "AMBIGU": (0, 165, 255),
    "AMBIGU_DOUBLON": (0, 0, 255),
    "NON_RESOLU": (0, 165, 255),
    "NON_TROUVE_TABLEUR": (128, 128, 128),
    "NON_MARQUE": (200, 200, 0),  # cyan-verdâtre : canard sans marque (classifié mais non identifiable)
}
COULEUR_MARQUE = (255, 200, 0)
COULEUR_TEXTE_FOND = (0, 0, 0)
COULEUR_TEXTE = (255, 255, 255)

FONT = cv2.FONT_HERSHEY_SIMPLEX

# Sous-dossier de destination selon le statut de la RAFALE.
STATUTS_A_VERIFIER = {"AMBIGU", "MULTIPLE_CANARDS", "PARTIELLEMENT_IDENTIFIE"}
STATUTS_IDENTIFIES = {"IDENTIFIE", "IDENTIFIE_MULTIPLE"}
STATUTS_NON_CONCLUANT = {"AUCUN_CANARD", "NON_TROUVE_TABLEUR", "NON_MARQUE"}

# Seuil sous lequel une rafale par ailleurs "IDENTIFIE(E)" est quand même
# redirigée vers a_verifier/ : une identification à faible confiance vaut
# mieux vérifiée qu'ignorée.
SEUIL_CONFIANCE_A_VERIFIER = 0.70


def _sous_dossier(result):
    """Détermine dans quel sous-dossier ranger la photo annotée d'une rafale."""
    status = result.get("status")

    if status in STATUTS_IDENTIFIES:
        confiances = [c.get("confiance_globale") for c in result.get("canards", [])
                      if c.get("confiance_globale") is not None]
        if confiances and min(confiances) < SEUIL_CONFIANCE_A_VERIFIER:
            return "a_verifier"
        return "identifies"

    if status in STATUTS_A_VERIFIER:
        return "a_verifier"

    return "non_concluant"


def _couleur_canard(canard, status_rafale):
    """Couleur du rectangle d'un canard : par niveau de confiance s'il est
    identifié, sinon par statut."""
    if canard.get("status") == "IDENTIFIE":
        niveau = niveau_confiance(canard.get("confiance_globale"))
        return COULEUR_PAR_NIVEAU[niveau]
    return COULEUR_PAR_STATUT.get(canard.get("status", status_rafale), COULEUR_PAR_NIVEAU["inconnu"])


def _label_canard(canard):
    """Construit le texte à afficher au-dessus d'un canard, incluant la
    confiance globale en % quand elle est disponible."""
    parts = []
    bague = canard.get("bague")
    if bague:
        parts.append(f"#{bague}")
    espece = canard.get("espece") or canard.get("espece_detectee")
    if espece:
        parts.append(str(espece))
    sexe = canard.get("sexe") or canard.get("sexe_detectee")
    if sexe:
        parts.append(str(sexe))
    couleur = canard.get("couleur_marque") or canard.get("couleur_detectee")
    if couleur:
        texte_couleur = str(couleur)
        if canard.get("couleur_fiable") is False:
            texte_couleur += " 🌙?"
        parts.append(texte_couleur)
    confiance_globale = canard.get("confiance_globale")
    if confiance_globale is not None:
        parts.append(f"{confiance_globale * 100:.0f}%")
    if canard.get("marque_absente") and canard.get("status") == "IDENTIFIE":
        parts.append("⚠️ sans marque")
    if canard.get("status") == "NON_MARQUE":
        parts.append("(non marqué)")
    if not parts:
        parts.append(canard.get("status", "?"))
    return " | ".join(parts)


def _draw_box_with_label(img, bbox, color, label):
    x, y, w, h = bbox
    x2, y2 = x + w, y + h
    cv2.rectangle(img, (x, y), (x2, y2), color, 2)

    (text_w, text_h), _ = cv2.getTextSize(label, FONT, 0.5, 1)
    label_y = max(0, y - 6)
    cv2.rectangle(img, (x, label_y - text_h - 4), (x + text_w + 6, label_y + 2), color, -1)
    cv2.putText(img, label, (x + 3, label_y - 2), FONT, 0.5, COULEUR_TEXTE, 1, cv2.LINE_AA)


def annotate_result(result, output_dir):
    """
    Dessine les bbox et labels sur la photo analysée d'une rafale, et
    sauvegarde l'image annotée dans le sous-dossier approprié de output_dir
    (a_verifier / identifies / non_concluant — voir _sous_dossier).

    Args:
        result: Un résultat de CanardIdentificationPipeline.process_burst.
        output_dir: Dossier racine (Path ou str) des photos annotées.

    Returns:
        Chemin (str) de l'image annotée écrite, ou None si la photo
        source n'a pas pu être chargée.
    """
    photo_path = result.get("photo_analysee")
    if not photo_path:
        return None

    img = cv2.imread(photo_path)
    if img is None:
        return None

    for marque in result.get("marques", []):
        couleur = marque.get("couleur_detectee") or "marque"
        label = str(couleur)
        if marque.get("nuit_probable"):
            label += " 🌙?"
        canard_assigne = marque.get("canard_assigne")
        if canard_assigne is not None:
            label += f" →#{canard_assigne}"
            if marque.get("assignation_certaine") is False:
                label += "?"
        _draw_box_with_label(img, marque["bbox"], COULEUR_MARQUE, label)

    for canard in result.get("canards", []):
        couleur = _couleur_canard(canard, result.get("status"))
        _draw_box_with_label(img, canard["bbox"], couleur, _label_canard(canard))

    header = f"Rafale {result.get('rafale_id', '?')} - {result.get('status', '')}"
    cv2.rectangle(img, (0, 0), (img.shape[1], 28), COULEUR_TEXTE_FOND, -1)
    cv2.putText(img, header, (6, 20), FONT, 0.6, COULEUR_TEXTE, 1, cv2.LINE_AA)

    dest_dir = Path(output_dir) / _sous_dossier(result)
    dest_dir.mkdir(parents=True, exist_ok=True)
    out_path = dest_dir / f"rafale_{result.get('rafale_id', 0):03d}_{result.get('status', 'INCONNU')}.jpg"
    cv2.imwrite(str(out_path), img)
    return str(out_path)


def annotate_all(resultats, output_dir):
    """Annote toutes les rafales d'une liste de résultats.

    Returns:
        Dictionnaire {rafale_id: chemin de l'image annotée} pour les
        rafales annotées avec succès (une rafale sans photo exploitable
        est absente du dict).
    """
    chemins = {}
    for result in resultats:
        chemin = annotate_result(result, output_dir)
        if chemin:
            chemins[result.get("rafale_id")] = chemin
    return chemins