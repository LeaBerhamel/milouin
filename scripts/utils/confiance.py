# -*- coding: utf-8 -*-
"""
Calcul de la confiance globale d'une identification.

PROBLÈME RÉSOLU : auparavant, le champ "confiance" d'un résultat était une
constante fixe dépendant uniquement de la méthode de matching temporel
(0.95 / 0.85 / 0.90 / 0.75), et ignorait complètement les scores de
certitude retournés par les IA de classification (confiance_espece,
confiance_sexe — des probabilités softmax entre 0 et 1, déjà calculées
par _classify_crop mais jamais réutilisées après coup).

FORMULE : moyenne pondérée des composantes disponibles.

    confiance_globale = Σ (poids_i × valeur_i) / Σ (poids_i disponibles)

Poids de base :
    - temporelle : 0.50  (à quel point l'horodatage de la photo colle à
      une observation du tableur : 0.95 si match ≤30s et univoque, 0.85 si
      match ≤180s et univoque, 0.90 / 0.75 pour les mêmes cas quand la
      désambiguïsation a nécessité de filtrer par espèce/sexe)
    - espèce : 0.25  (probabilité softmax de la classe prédite par l'IA
      Espèce sur le crop du canard)
    - sexe : 0.25  (idem pour l'IA Sexe)

Si l'IA Espèce et/ou l'IA Sexe n'ont pas été utilisées (modèle non fourni,
ou plusieurs canards où la classification a échoué), leur poids est
retiré et les poids restants sont renormalisés pour totaliser 1 — la
confiance temporelle seule redevient alors 100% du score, comme avant.

Exemple : match temporel strict (0.95) + IA Espèce très sûre (0.98) mais
IA Sexe non utilisée :
    poids disponibles = temporelle (0.50) + espèce (0.25) = 0.75
    confiance_globale = (0.50×0.95 + 0.25×0.98) / 0.75 ≈ 0.96
"""

POIDS = {"temporelle": 0.50, "espece": 0.25, "sexe": 0.25}


def compute_confiance_globale(confiance_temporelle, confiance_espece=None, confiance_sexe=None):
    """
    Combine la confiance temporelle (matching) et les confiances de
    classification (espèce/sexe, si disponibles) en un seul score.

    Args:
        confiance_temporelle: Score fixe selon la méthode de match
            (voir constantes MATCH_TEMPOREL_STRICT / LARGE dans pipeline.py).
        confiance_espece: Probabilité softmax de l'IA Espèce (0-1), ou None
            si l'IA n'a pas été utilisée pour ce canard.
        confiance_sexe: Probabilité softmax de l'IA Sexe (0-1), ou None
            si l'IA n'a pas été utilisée pour ce canard.

    Returns:
        Score de confiance combiné, arrondi à 3 décimales.
    """
    composantes = {"temporelle": confiance_temporelle}
    if confiance_espece is not None:
        composantes["espece"] = confiance_espece
    if confiance_sexe is not None:
        composantes["sexe"] = confiance_sexe

    poids_total = sum(POIDS[k] for k in composantes)
    confiance = sum(POIDS[k] * v for k, v in composantes.items()) / poids_total
    return round(confiance, 3)


def niveau_confiance(confiance_globale):
    """Catégorise un score de confiance en 3 niveaux, pour l'affichage
    (couleur des annotations, tri dans le rapport HTML)."""
    if confiance_globale is None:
        return "inconnu"
    if confiance_globale >= 0.90:
        return "haute"
    if confiance_globale >= 0.70:
        return "moyenne"
    return "faible"
