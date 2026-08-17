# -*- coding: utf-8 -*-
"""
Formulaire de correction à 4 champs (bague, couleur, espèce, sexe),
utilisé aussi bien pour :
  - le clic ponctuel depuis le rapport HTML (un seul crop, via le serveur
    local -- voir serveur_corrections.py) ;
  - le futur écran batch héritier de 09_ecran_identification.py (parcours
    du backlog des bagues non identifiées).

Autonome, ne modifie rien dans scripts/. Testable seul via
demo_standalone.py.

RÉSOLU (07/08) : le bug COULEURS_CLASSES manquant dans utils/config.py a
été corrigé directement dans scripts/utils/config.py (ESPECES_CLASSES /
SEXES_CLASSES / COULEURS_CLASSES y sont maintenant définies -- c'est la
même source de vérité que utils/pipeline.py utilise pour charger les
classifieurs, donc les listes ci-dessous restent synchronisées avec le
reste du projet par construction plutôt que par copie manuelle).
"""

import tkinter as tk
from tkinter import messagebox, ttk

try:
    # Import direct depuis le dépôt réel une fois ce module copié à côté
    # de scripts/ (voir README.md) -- une seule source de vérité avec
    # utils/pipeline.py, plutôt que deux listes à maintenir à la main.
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    from utils.config import COULEURS_CLASSES as COULEURS
    from utils.config import ESPECES_CLASSES as ESPECES
    from utils.config import SEXES_CLASSES as SEXES
except ImportError:
    # Filet de sécurité pour que la démo autonome (demo_standalone.py)
    # fonctionne même si ce dossier est testé isolément, sans le reste du
    # dépôt à côté.
    ESPECES = ["milouin", "morillon"]
    SEXES = ["male", "femelle", "indetermine_juvenile"]
    COULEURS = ["bleu", "marron", "indeterminable"]


def _texte_sur(valeur):
    """Ramène une valeur potentiellement non-chaîne (notamment un NaN
    float, que pandas laisse dans les cellules Excel vides et qui
    survit à l'aller-retour JSON) à une chaîne exploitable.

    Défense en profondeur : l'appelant (fenetre_rafale.py) est censé ne
    fournir que des chaînes, mais ce module est documenté comme
    autonome/réutilisable (demo_standalone.py, futurs appelants) --
    mieux vaut qu'il ne plante jamais sur .strip() si jamais on lui
    passe encore un NaN, plutôt que de dépendre uniquement de la
    discipline de chaque appelant."""
    if valeur is None:
        return ""
    if isinstance(valeur, float) and valeur != valeur:  # NaN != NaN
        return ""
    return str(valeur).strip()


def ouvrir_formulaire_correction(parent, valeurs_initiales, on_valider, identifiants_connus=None):
    """Ouvre un tk.Toplevel avec les 4 champs pré-remplis.

    valeurs_initiales : dict avec les clés "crop", "espece", "sexe",
        "couleur", "bague" (chaînes, peuvent être vides), et en plus
        "bague_confirmee_tableur" (bool, optionnel -- calculé côté
        appelant à partir de result["canards"][idx]["status"] : True si
        ce canard a le statut IDENTIFIE/IDENTIFIE_MULTIPLE, donc si sa
        bague vient du rapprochement temporel avec le tableur
        d'observations, pas d'une détection IA. Ce formulaire ne doit pas
        connaître la logique métier du statut, juste ce booléen déjà
        calculé).
    on_valider : callable(dict) appelé avec les valeurs finales au clic
        sur "✓ Valider" -- charge à l'appelant d'écrire dans
        corrections.csv / dictionnaire_identifications.json /
        crops_exclus.json (voir ecrire_corrections.py). Le dict transmis
        contient aussi "bague_confirmee_tableur" et "bague_initiale_tableur"
        (la bague d'origine si confirmée par le tableur, "" sinon), pour
        que l'appelant puisse détecter une contradiction avec le tableur
        sans redemander à l'utilisateur (déjà confirmé ci-dessous).
        N'est PAS appelé si le chercheur clique sur "Annuler" -- rien
        n'est alors écrit, le formulaire se referme simplement.
    identifiants_connus : liste optionnelle de codes de bagues déjà
        connus (ex. déjà vus dans le dictionnaire d'identifications), pour
        peupler le menu déroulant du champ Bague -- le chercheur peut
        toujours taper un code inédit, le champ reste éditable.

    Retourne le Toplevel créé (utile pour les tests).
    """
    # Copie défensive : normalise les 4 champs texte avant toute lecture,
    # une bonne fois pour toutes plutôt qu'à chaque .get() dans la suite
    # de la fonction -- voir _texte_sur() ci-dessus.
    valeurs_initiales = {
        **valeurs_initiales,
        "bague": _texte_sur(valeurs_initiales.get("bague")),
        "couleur": _texte_sur(valeurs_initiales.get("couleur")),
        "espece": _texte_sur(valeurs_initiales.get("espece")),
        "sexe": _texte_sur(valeurs_initiales.get("sexe")),
    }

    fenetre = tk.Toplevel(parent)
    fenetre.title(f"Corriger — {valeurs_initiales.get('crop', '')}")
    fenetre.resizable(False, False)

    conteneur = ttk.Frame(fenetre, padding=16)
    conteneur.pack(fill="both", expand=True)

    ttk.Label(conteneur, text=valeurs_initiales.get("crop", ""),
              font=("", 9, "italic")).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 12))

    var_bague = tk.StringVar(value=valeurs_initiales.get("bague", ""))
    var_couleur = tk.StringVar(value=valeurs_initiales.get("couleur", ""))
    var_espece = tk.StringVar(value=valeurs_initiales.get("espece", ""))
    var_sexe = tk.StringVar(value=valeurs_initiales.get("sexe", ""))

    # Deux cases distinctes plutôt qu'une seule "Bague présente" :
    #  - "🏷️ Marque présente" : ce canard porte-t-il une bague/marque
    #    nasale, visible ou non ? Décochée -> PAS_MARQUE, aucun champ à
    #    saisir (ni code, ni couleur).
    #  - "🔤 Code lisible" : la marque est présente mais son code est-il
    #    déchiffrable sur la photo ? Décochée (marque présente cochée) ->
    #    ILLISIBLE. La couleur de la marque, elle, reste saisissable même
    #    si le code n'est pas lisible (souvent la couleur se voit alors
    #    que le code ne se voit pas).
    #
    # Cases initiales calculées explicitement à partir des 3 formes que
    # peut prendre valeurs_initiales["bague"] -- un vrai code, ou un des
    # deux codes d'exclusion (ILLISIBLE/PAS_MARQUE, voir ecrire_corrections.py
    # est_code_exclusion()) si ce canard avait déjà été marqué comme tel
    # lors d'une correction précédente. Sans ce cas par cas, un canard
    # rouvert après avoir été marqué PAS_MARQUE se serait vu réafficher la
    # case "Marque présente" cochée (dès qu'une valeur non vide était
    # présente dans "bague", même un code d'exclusion).
    CODES_EXCLUSION = ("ILLISIBLE", "PAS_MARQUE")
    valeur_bague_initiale = (valeurs_initiales.get("bague", "") or "").strip().upper()
    couleur_connue = bool(valeurs_initiales.get("couleur", "").strip())
    bague_reelle_connue = bool(valeur_bague_initiale) and valeur_bague_initiale not in CODES_EXCLUSION
    # Vrai si l'appelant sait déjà (typiquement : un carré de marque nasale
    # vient d'être dessiné à la main sur la photo, voir fenetre_rafale.py /
    # _ajouter_marque_manuelle) qu'une marque existe pour ce canard, même
    # si son code/couleur n'a pas encore été saisi -- sans quoi la case
    # "Marque présente" restait décochée juste après avoir dessiné le
    # carré, tant que bague/couleur étaient vides.
    marque_deja_signalee = bool(valeurs_initiales.get("marque_deja_signalee"))

    if valeur_bague_initiale == "PAS_MARQUE" and not marque_deja_signalee:
        marque_par_defaut, lisible_par_defaut = False, False
    elif valeur_bague_initiale == "ILLISIBLE":
        marque_par_defaut, lisible_par_defaut = True, False
    else:
        marque_par_defaut = bague_reelle_connue or couleur_connue or marque_deja_signalee
        lisible_par_defaut = bague_reelle_connue

    var_marque_presente = tk.BooleanVar(value=marque_par_defaut)
    var_lisible = tk.BooleanVar(value=lisible_par_defaut)

    ttk.Label(conteneur, text="Bague :").grid(row=1, column=0, sticky="w", pady=4)
    champ_bague = ttk.Combobox(conteneur, textvariable=var_bague,
                                values=list(identifiants_connus or []), width=22)
    champ_bague.grid(row=1, column=1, sticky="ew", pady=4)

    ttk.Label(conteneur, text="Couleur bague :").grid(row=2, column=0, sticky="w", pady=4)
    combo_couleur = ttk.Combobox(conteneur, textvariable=var_couleur, values=COULEURS,
                                  state="readonly", width=22)
    combo_couleur.grid(row=2, column=1, sticky="ew", pady=4)

    ttk.Label(conteneur, text="Espèce :").grid(row=3, column=0, sticky="w", pady=4)
    ttk.Combobox(conteneur, textvariable=var_espece, values=ESPECES,
                 state="readonly", width=22).grid(row=3, column=1, sticky="ew", pady=4)

    ttk.Label(conteneur, text="Sexe :").grid(row=4, column=0, sticky="w", pady=4)
    ttk.Combobox(conteneur, textvariable=var_sexe, values=SEXES,
                 state="readonly", width=22).grid(row=4, column=1, sticky="ew", pady=4)

    cases = ttk.Frame(conteneur)
    cases.grid(row=5, column=0, columnspan=2, sticky="w", pady=(4, 8))
    chk_marque_presente = ttk.Checkbutton(cases, text="🏷️ Marque présente", variable=var_marque_presente)
    chk_marque_presente.pack(side="left")
    chk_lisible = ttk.Checkbutton(cases, text="🔤 Code lisible", variable=var_lisible)
    chk_lisible.pack(side="left", padx=(16, 0))

    def _rafraichir_champs(*_args):
        # Sans marque : ni code ni couleur à saisir, tout est grisé/vidé.
        # Marque présente mais illisible : la couleur reste saisissable
        # (souvent visible même sans le code), seul le code est grisé.
        # Marque présente et lisible : les deux champs sont ouverts.
        marque = var_marque_presente.get()
        chk_lisible.configure(state="normal" if marque else "disabled")
        if not marque:
            var_lisible.set(False)
            var_bague.set("")
            var_couleur.set("")
            champ_bague.configure(state="disabled")
            combo_couleur.configure(state="disabled")
        elif not var_lisible.get():
            var_bague.set("")
            champ_bague.configure(state="disabled")
            combo_couleur.configure(state="readonly")
        else:
            champ_bague.configure(state="normal")
            combo_couleur.configure(state="readonly")

    var_marque_presente.trace_add("write", _rafraichir_champs)
    var_lisible.trace_add("write", _rafraichir_champs)
    _rafraichir_champs()

    bague_confirmee_tableur = bool(valeurs_initiales.get("bague_confirmee_tableur"))
    bague_initiale_tableur = valeurs_initiales.get("bague", "") if bague_confirmee_tableur else ""

    if bague_confirmee_tableur:
        ttk.Label(
            conteneur,
            text="ℹ️ Cette bague vient du tableur d'observations (rapprochement\n"
                 "horaire), pas d'une détection IA.",
            font=("", 8, "italic"), foreground="#555555", justify="left",
        ).grid(row=7, column=0, columnspan=2, sticky="w", pady=(4, 0))

    def _confirmer_si_contredit_tableur(nouvelle_bague):
        """Retourne True si on peut continuer (pas de contradiction, ou
        contradiction confirmée par l'utilisateur), False si le
        chercheur a annulé (répondu "Non") -- dans ce cas l'appelant ne
        doit RIEN écrire, le formulaire reste ouvert."""
        if not bague_confirmee_tableur or nouvelle_bague == bague_initiale_tableur:
            return True
        return messagebox.askyesno(
            "Bague confirmée par le tableur",
            f"La bague « {bague_initiale_tableur} » vient du tableur d'observations "
            "(rapprochement horaire), pas d'une détection IA.\n\n"
            "La remplacer ou l'effacer signifie que tu as vérifié la photo et que "
            "le tableur est en fait erroné pour cette observation.\n\n"
            "Confirmer le changement ?",
            parent=fenetre,
        )

    def _finaliser(nouvelle_bague):
        if not _confirmer_si_contredit_tableur(nouvelle_bague):
            return  # "Non" : on n'écrit rien, le formulaire reste ouvert
        on_valider({
            "crop": valeurs_initiales.get("crop", ""),
            "bague": nouvelle_bague,
            "espece": var_espece.get().strip(),
            "sexe": var_sexe.get().strip(),
            "couleur": var_couleur.get().strip(),
            # Valeurs détectées par l'IA, transmises telles quelles pour
            # que l'appelant sache si le chercheur a réellement changé
            # quelque chose (pas d'écriture inutile dans corrections.csv).
            "espece_detectee": valeurs_initiales.get("espece", ""),
            "sexe_detecte": valeurs_initiales.get("sexe", ""),
            "couleur_detectee": valeurs_initiales.get("couleur", ""),
            "bague_confirmee_tableur": bague_confirmee_tableur,
            "bague_initiale_tableur": bague_initiale_tableur,
            # Transmis tel quel (True/False/None) pour que
            # enregistrer_revue_marque_nasale() puisse comparer à la
            # vérité terrain déduite de "bague" ci-dessus -- voir
            # fenetre_rafale.py::_ouvrir_correction pour l'origine de
            # cette valeur (canard.get("marque_absente")).
            "marque_absente_detectee": valeurs_initiales.get("marque_absente_detectee"),
        })
        fenetre.destroy()

    def _valider():
        # La valeur écrite pour "bague" dépend des deux cases plutôt que
        # de boutons dédiés :
        #   Marque absente             -> PAS_MARQUE
        #   Marque présente, illisible -> ILLISIBLE
        #   Marque présente, lisible   -> le code saisi dans le champ
        if not var_marque_presente.get():
            _finaliser("PAS_MARQUE")
        elif not var_lisible.get():
            _finaliser("ILLISIBLE")
        else:
            _finaliser(var_bague.get().strip())

    def _annuler():
        # Ferme simplement le formulaire sans rien écrire -- contrairement
        # aux anciens boutons "🚫 Illisible" / "❌ Pas de bague", qui
        # validaient tout de suite un statut. Ces cas sont maintenant
        # couverts par les deux cases à cocher ci-dessus, validées comme
        # le reste via "✓ Valider".
        fenetre.destroy()

    boutons = ttk.Frame(conteneur)
    boutons.grid(row=6, column=0, columnspan=2, pady=(16, 0), sticky="e")
    ttk.Button(boutons, text="Annuler", command=_annuler).pack(side="right", padx=(6, 0))
    ttk.Button(boutons, text="✓ Valider", command=_valider).pack(side="right")

    fenetre.bind("<Return>", lambda _e: _valider())
    fenetre.bind("<Escape>", lambda _e: _annuler())

    # Amène la fenêtre au premier plan -- point signalé comme délicat
    # (comportement variable selon l'OS) lors de la discussion de conception.
    fenetre.lift()
    fenetre.attributes("-topmost", True)
    fenetre.after(200, lambda: fenetre.attributes("-topmost", False))
    fenetre.focus_force()
    champ_bague.focus_set()

    return fenetre