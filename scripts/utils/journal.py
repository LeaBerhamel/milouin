# -*- coding: utf-8 -*-
"""
Journal des cas limites : pour chaque rafale nécessitant une vérification
manuelle, explique EN LANGAGE CLAIR pourquoi le pipeline n'a pas pu
conclure tout seul — au lieu de devoir déduire la raison en recoupant les
champs du JSON (status, message, candidats, cohérent...).

Un chercheur doit pouvoir lire ce fichier et savoir, pour chaque rafale à
vérifier, quoi regarder en priorité sur la photo.
"""

from pathlib import Path

STATUTS_A_EXPLIQUER = {"AMBIGU", "MULTIPLE_CANARDS", "PARTIELLEMENT_IDENTIFIE"}


def _explique_rafale(result):
    """Construit le texte explicatif d'une rafale à vérifier."""
    lignes = [f"Rafale {result.get('rafale_id')} — statut : {result.get('status')}"]
    lignes.append(f"  Photo analysée : {result.get('photo_analysee')}")

    status = result.get("status")

    if status == "AMBIGU":
        # Un seul canard détecté, mais plusieurs observations du tableur
        # correspondent encore au créneau horaire même après filtrage
        # espèce/sexe : impossible de choisir automatiquement laquelle.
        lignes.append(f"  Cause : {result.get('message', 'ambiguïté non précisée')}")
        candidats = result.get("candidats", [])
        if candidats:
            lignes.append("  Observations candidates dans le tableur :")
            for c in candidats[:8]:
                lignes.append(f"    - bague {c.get('BAGUE', '?')} | espèce {c.get('ESPECE', '?')} "
                               f"| sexe {c.get('SEXE', '?')} | heure {c.get('DATETIME', '?')}")
        lignes.append("  À vérifier : regarder la photo pour confirmer espèce/sexe/marque "
                       "et choisir manuellement la bonne ligne du tableur ci-dessus.")

    elif status in {"MULTIPLE_CANARDS", "PARTIELLEMENT_IDENTIFIE"}:
        nb_canards = result.get("detections", {}).get("nb_canards", len(result.get("canards", [])))
        lignes.append(f"  Cause : {nb_canards} canards détectés sur la photo ; "
                       f"{result.get('message', '')}")
        for canard in result.get("canards", []):
            if canard.get("status") == "IDENTIFIE":
                lignes.append(f"    - Canard #{canard['canard_index']} : identifié -> "
                               f"bague {canard.get('bague')} ({canard.get('espece')}, {canard.get('sexe')}) — OK")
            elif canard.get("status") == "AMBIGU_DOUBLON":
                lignes.append(f"    - Canard #{canard['canard_index']} : {canard.get('diagnostic')}")
            else:
                espece = canard.get("espece_detectee") or "espèce non déterminée"
                sexe = canard.get("sexe_detectee") or "sexe non déterminé"
                lignes.append(f"    - Canard #{canard['canard_index']} ({espece}, {sexe}) non résolu : "
                               f"{canard.get('diagnostic') or 'raison non précisée'}")
        lignes.append("  À vérifier : ouvrir la photo annotée (dossier a_verifier/) pour confirmer "
                       "quel canard correspond à quelle observation du tableur.")

    return "\n".join(lignes)


def _explique_sans_marque(result):
    """Construit le texte explicatif d'une rafale identifiée sans aucune
    marque nasale détectée (voir la note dans utils/pipeline.py) : le
    matching ne repose alors que sur espèce/sexe/horodatage, sans
    confirmation physique de la bague."""
    lignes = [f"Rafale {result.get('rafale_id')} — statut : {result.get('status')}"]
    lignes.append(f"  Photo analysée : {result.get('photo_analysee')}")
    for canard in result.get("canards", []):
        if canard.get("status") == "IDENTIFIE" and canard.get("marque_absente"):
            lignes.append(
                f"    - Canard #{canard.get('canard_index')} : identifié -> bague "
                f"{canard.get('bague')} ({canard.get('espece')}, {canard.get('sexe')}) "
                f"— AUCUNE marque nasale détectée sur la photo, uniquement espèce/sexe/heure."
            )
    lignes.append("  À vérifier : confirmer sur la photo qu'aucune marque n'est simplement "
                   "masquée/hors cadre, avant de valider cette identification.")
    return "\n".join(lignes)


def build_journal(resultats):
    """Construit le texte complet du journal (toutes les rafales à vérifier)."""
    a_expliquer = [r for r in resultats if r.get("status") in STATUTS_A_EXPLIQUER]
    sans_marque = [
        r for r in resultats
        if any(c.get("status") == "IDENTIFIE" and c.get("marque_absente") for c in r.get("canards", []))
    ]

    if not a_expliquer and not sans_marque:
        return "Aucune rafale ne nécessite de vérification manuelle sur ce lot. 🎉\n"

    blocs = []

    if a_expliquer:
        entetes = [
            f"Journal des cas limites — {len(a_expliquer)} rafale(s) à vérifier sur {len(resultats)}",
            "=" * 70,
            "",
        ]
        corps = [_explique_rafale(r) for r in a_expliquer]
        blocs.append("\n".join(entetes) + "\n\n".join(corps) + "\n")

    if sans_marque:
        entetes = [
            "",
            f"⚠️  Identifications SANS marque nasale détectée — {len(sans_marque)} rafale(s)",
            "=" * 70,
            "Ces canards ont été associés à une bague du tableur uniquement sur la base de "
            "l'espèce, du sexe et de l'horodatage, SANS confirmation par une marque nasale "
            "visible. Risque : un canard non marqué passant au bon moment peut être confondu "
            "avec un autre individu de même espèce/sexe. À vérifier en priorité.",
            "",
        ]
        corps = [_explique_sans_marque(r) for r in sans_marque]
        blocs.append("\n".join(entetes) + "\n\n".join(corps) + "\n")

    return "\n".join(blocs)


def export_journal(resultats, output_path):
    """Écrit le journal dans un fichier texte."""
    output_path = Path(output_path)
    output_path.write_text(build_journal(resultats), encoding="utf-8")
    return str(output_path)
