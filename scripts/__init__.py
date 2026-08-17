# -*- coding: utf-8 -*-
"""
Synthèse chiffrée d'un lot traité : agrège, sur l'ensemble des rafales,
les comptes par statut, par espèce, par sexe et par bague identifiée.

Contrairement aux stats affichées en console par 02_process_rafales.py
(qui comptent des RAFALES), ce module compte des CANARDS individuels —
une rafale à 3 canards identifiés compte pour 3 dans "par espèce" / "par
sexe" / "par bague", pas pour 1.
"""

import csv
from collections import Counter
from pathlib import Path

from .config import normaliser_espece, normaliser_sexe


def _tous_les_canards(resultats):
    """Aplati la liste de résultats en une liste de canards individuels,
    chacun enrichi de son rafale_id et du statut global de la rafale."""
    canards = []
    for r in resultats:
        for c in r.get("canards", []):
            canards.append({**c, "rafale_id": r.get("rafale_id"), "status_rafale": r.get("status")})
    return canards


def build_synthese(resultats):
    """
    Calcule la synthèse chiffrée du lot.

    Returns:
        Dictionnaire avec :
        - nb_rafales, nb_photos
        - par_statut_rafale : Counter des statuts de rafale
        - nb_canards_detectes, nb_canards_identifies
        - par_espece : Counter des espèces (canards identifiés)
        - par_sexe : Counter des sexes (canards identifiés)
        - par_bague : Counter des bagues (nb de fois vue -> individus repérés
          plusieurs fois dans le lot)
        - nb_individus_uniques : nombre de bagues distinctes identifiées
    """
    canards = _tous_les_canards(resultats)
    identifies = [c for c in canards if c.get("status") == "IDENTIFIE"]
    sans_marque = [c for c in identifies if c.get("marque_absente")]
    # Canards confirmés NON bagués (statut NON_MARQUE côté pipeline, voir
    # utils/pipeline.py) -- distinct de "sans_marque" ci-dessus, qui lui
    # ne concerne que des canards IDENTIFIE (bague connue via le tableur)
    # mais sans marque nasale VISIBLE sur cette photo. Les statuts
    # intermédiaires (AMBIGU, NON_TROUVE_TABLEUR, NON_RESOLU...) ne sont
    # comptés ni ici ni dans identifies : ni confirmés bagués, ni
    # confirmés non bagués.
    non_bagues = [c for c in canards if c.get("status") == "NON_MARQUE"]

    # .strip().upper() -- une bague est un identifiant, pas du texte libre :
    # sans cette normalisation, "ab123" et "AB123" saisis à des moments
    # différents (correction manuelle vs tableur) compteraient comme deux
    # individus distincts au lieu d'un seul revu deux fois.
    par_bague = Counter(c["bague"].strip().upper() for c in identifies if c.get("bague"))

    return {
        "nb_rafales": len(resultats),
        "nb_photos": sum(r["rafale"]["nb_photos"] for r in resultats),
        "par_statut_rafale": Counter(r["status"] for r in resultats),
        "nb_canards_detectes": len(canards),
        "nb_canards_identifies": len(identifies),
        "nb_canards_identifies_sans_marque": len(sans_marque),
        "nb_canards_non_bagues": len(non_bagues),
        "par_espece": Counter(normaliser_espece(c.get("espece") or c.get("espece_detectee")) for c in identifies),
        "par_sexe": Counter(normaliser_sexe(c.get("sexe") or c.get("sexe_detectee")) for c in identifies),
        # Mêmes répartitions espèce/sexe, mais pour les canards CONFIRMÉS
        # non bagués -- permet de comparer bagués vs non bagués par
        # espèce/sexe plutôt que de ne montrer que les bagués (voir
        # export_graphiques_synthese).
        "par_espece_non_bague": Counter(
            normaliser_espece(c.get("espece") or c.get("espece_detectee")) for c in non_bagues),
        "par_sexe_non_bague": Counter(
            normaliser_sexe(c.get("sexe") or c.get("sexe_detectee")) for c in non_bagues),
        "par_bague": par_bague,
        "nb_individus_uniques": len(par_bague),
    }


def print_synthese(synthese):
    """Affiche la synthèse de façon lisible en console."""
    print(f"\n📸 Photos traitées : {synthese['nb_photos']}  |  Rafales : {synthese['nb_rafales']}")

    print("\n📋 Rafales par statut :")
    for status, count in synthese["par_statut_rafale"].most_common():
        print(f"   {status:<28} {count}")

    print(f"\n🦆 Canards détectés (toutes rafales) : {synthese['nb_canards_detectes']}")
    print(f"   dont identifiés (bague trouvée)   : {synthese['nb_canards_identifies']}")
    if synthese["nb_canards_identifies_sans_marque"]:
        print(f"   ⚠️  dont SANS marque nasale détectée (espèce/sexe/heure seuls) : "
              f"{synthese['nb_canards_identifies_sans_marque']} — à vérifier en priorité")

    print("\n🔖 Individus uniques identifiés (par bague) :", synthese["nb_individus_uniques"])

    if synthese["par_espece"] or synthese["par_espece_non_bague"]:
        print("\n🐦 Répartition par espèce :")
        if synthese["par_espece"]:
            print("   Bagués (identifiés) :")
            for espece, count in synthese["par_espece"].most_common():
                print(f"      {espece:<20} {count}")
        if synthese["par_espece_non_bague"]:
            print("   Non bagués :")
            for espece, count in synthese["par_espece_non_bague"].most_common():
                print(f"      {espece:<20} {count}")

    if synthese["par_sexe"] or synthese["par_sexe_non_bague"]:
        print("\n♂♀ Répartition par sexe :")
        if synthese["par_sexe"]:
            print("   Bagués (identifiés) :")
            for sexe, count in synthese["par_sexe"].most_common():
                print(f"      {sexe:<20} {count}")
        if synthese["par_sexe_non_bague"]:
            print("   Non bagués :")
            for sexe, count in synthese["par_sexe_non_bague"].most_common():
                print(f"      {sexe:<20} {count}")

    if synthese["par_bague"]:
        revues = {b: n for b, n in synthese["par_bague"].items() if n > 1}
        if revues:
            print(f"\n🔁 Individus revus plusieurs fois dans ce lot : {len(revues)}")
            for bague, n in sorted(revues.items(), key=lambda x: -x[1]):
                print(f"   {bague:<15} vu {n} fois")


def export_synthese_csv(synthese, output_path):
    """Exporte la synthèse (une ligne par catégorie/valeur) en CSV, pour
    ouverture facile dans Excel."""
    rows = [{"categorie": "GLOBAL", "valeur": "nb_photos", "count": synthese["nb_photos"]},
            {"categorie": "GLOBAL", "valeur": "nb_rafales", "count": synthese["nb_rafales"]},
            {"categorie": "GLOBAL", "valeur": "nb_canards_detectes", "count": synthese["nb_canards_detectes"]},
            {"categorie": "GLOBAL", "valeur": "nb_canards_identifies", "count": synthese["nb_canards_identifies"]},
            {"categorie": "GLOBAL", "valeur": "nb_canards_identifies_sans_marque",
             "count": synthese["nb_canards_identifies_sans_marque"]},
            {"categorie": "GLOBAL", "valeur": "nb_canards_non_bagues",
             "count": synthese["nb_canards_non_bagues"]},
            {"categorie": "GLOBAL", "valeur": "nb_individus_uniques", "count": synthese["nb_individus_uniques"]}]

    for status, count in synthese["par_statut_rafale"].items():
        rows.append({"categorie": "statut_rafale", "valeur": status, "count": count})
    for espece, count in synthese["par_espece"].items():
        rows.append({"categorie": "espece_bague", "valeur": espece, "count": count})
    for espece, count in synthese["par_espece_non_bague"].items():
        rows.append({"categorie": "espece_non_bague", "valeur": espece, "count": count})
    for sexe, count in synthese["par_sexe"].items():
        rows.append({"categorie": "sexe_bague", "valeur": sexe, "count": count})
    for sexe, count in synthese["par_sexe_non_bague"].items():
        rows.append({"categorie": "sexe_non_bague", "valeur": sexe, "count": count})
    for bague, count in synthese["par_bague"].items():
        rows.append({"categorie": "bague", "valeur": bague, "count": count})

    output_path = Path(output_path)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["categorie", "valeur", "count"])
        writer.writeheader()
        writer.writerows(rows)


def export_graphiques_synthese(synthese, output_path):
    """
    Génère un PNG à 3 panneaux : rafales par statut, canards par espèce,
    canards par sexe. Les 2 derniers panneaux affichent deux barres par
    catégorie -- bagués (identifiés) et non bagués -- plutôt que les seuls
    bagués, pour ne pas donner l'impression trompeuse que le lot ne
    contient presque aucun canard (le statut de rafale le plus fréquent
    est typiquement NON_MARQUE, voir le premier panneau).

    N'écrit rien si la synthèse ne contient aucune donnée à afficher.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    COULEUR_BAGUE = "#4C8BF5"
    COULEUR_NON_BAGUE = "#bbbbbb"

    # panneaux : (titre, counter_principal, counter_secondaire_ou_None,
    # libelle_principal, libelle_secondaire) -- le panneau "Rafales par
    # statut" n'a pas de second counter (une seule série de barres).
    panneaux = [
        ("Rafales par statut", synthese["par_statut_rafale"], None, None, None),
        ("Canards par espèce", synthese["par_espece"], synthese["par_espece_non_bague"],
         "Bagués (identifiés)", "Non bagués"),
        ("Canards par sexe", synthese["par_sexe"], synthese["par_sexe_non_bague"],
         "Bagués (identifiés)", "Non bagués"),
    ]
    panneaux = [p for p in panneaux if p[1] or p[2]]
    if not panneaux:
        return None

    fig, axes = plt.subplots(1, len(panneaux), figsize=(5 * len(panneaux), 4.5))
    if len(panneaux) == 1:
        axes = [axes]

    for ax, (titre, principal, secondaire, libelle_principal, libelle_secondaire) in zip(
            axes, panneaux, strict=True):
        if secondaire is None:
            items = principal.most_common()
            labels = [str(k) for k, _ in items]
            valeurs = [v for _, v in items]
            ax.bar(labels, valeurs, color=COULEUR_BAGUE)
            for i, v in enumerate(valeurs):
                ax.text(i, v, str(v), ha="center", va="bottom", fontsize=9)
        else:
            # Union des catégories des deux séries, triée par total
            # décroissant (bagués + non bagués), pour que les catégories
            # les plus représentées ressortent en premier comme avant.
            labels = sorted(
                set(principal) | set(secondaire),
                key=lambda k: -(principal.get(k, 0) + secondaire.get(k, 0)),
            )
            positions = range(len(labels))
            largeur = 0.38
            valeurs_principal = [principal.get(k, 0) for k in labels]
            valeurs_secondaire = [secondaire.get(k, 0) for k in labels]
            barres_principal = ax.bar(
                [p - largeur / 2 for p in positions], valeurs_principal, largeur,
                color=COULEUR_BAGUE, label=libelle_principal)
            barres_secondaire = ax.bar(
                [p + largeur / 2 for p in positions], valeurs_secondaire, largeur,
                color=COULEUR_NON_BAGUE, label=libelle_secondaire)
            ax.set_xticks(list(positions))
            ax.set_xticklabels([str(k) for k in labels])
            ax.legend(fontsize=8)
            for barres in (barres_principal, barres_secondaire):
                for barre in barres:
                    hauteur = barre.get_height()
                    if hauteur:
                        ax.text(barre.get_x() + barre.get_width() / 2, hauteur, str(int(hauteur)),
                                ha="center", va="bottom", fontsize=9)
        ax.set_title(titre)
        ax.tick_params(axis="x", rotation=30)

    fig.tight_layout()
    output_path = Path(output_path)
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    return str(output_path)