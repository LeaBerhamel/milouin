# -*- coding: utf-8 -*-
"""
Fenêtre "photo + liste" ouverte quand un chercheur clique sur le lien
"✏️ Corriger cette rafale" du rapport HTML : affiche la photo annotée de
la rafale (bbox déjà dessinées par utils/annotate.py) et, à côté, la liste
de tous les canards détectés dessus, chacun avec un bouton ✏️ qui ouvre le
formulaire à 4 champs (formulaire_correction.py) pré-rempli.

Pour un canard qui a une marque nasale assignée mais pas de bague confirmée
(AMBIGU / NON_TROUVE_TABLEUR / PARTIELLEMENT_IDENTIFIE / IDENTIFIE_MULTIPLE /
NON_RESOLU / AMBIGU_DOUBLON -- voir STATUTS_AVEC_SUGGESTIONS ci-dessous),
affiche aussi jusqu'à 3 suggestions de bagues par ressemblance visuelle
(scripts/06_proposer_correspondances_visuelles.py, lu via
utils/propositions.py) -- cliquer une suggestion ouvre le même formulaire
avec le champ Bague pré-rempli de cet identifiant, sans jamais valider
tout seul.

La photo n'est plus un simple Label statique : c'est un Canvas interactif.
Un cliquer-glisser dessus ajoute une ligne "Canard ajouté manuellement" à
la liste (statut AJOUT_MANUEL) et ouvre directement le formulaire habituel,
vide, pour renseigner un canard que l'IA n'a pas détecté du tout -- y
compris sur une rafale où elle n'en a trouvé aucun. Les coordonnées du
cadre dessiné sont reconverties de la résolution d'affichage (miniature)
vers la résolution native de la photo source, via le facteur d'échelle
calculé au chargement de l'image.

Chaque canard/marque de la rafale (détecté par l'IA ou ajouté à la main)
a en plus un cadre pointillé éditable superposé à la photo : glisser une
poignée de coin le redimensionne, glisser son contour le déplace. La
bbox corrigée est aussitôt réenregistrée (et le crop déjà extrait, s'il
y en a un, ré-extrait à la nouvelle position) -- utile quand l'IA a
détecté le bon canard mais avec un cadre légèrement décalé/trop serré.

Chaque ligne de canard a aussi un bouton 🗑️ pour le supprimer purement
et simplement de la rafale -- qu'il ait été détecté par l'IA (fausse
détection, doublon de cadre sur le même canard...) ou ajouté à la main
par erreur ; toute correction déjà enregistrée pour lui depuis cette
fenêtre est alors annulée.

Chaque modification (cadre édité, canard/marque ajouté ou supprimé) fait
aussitôt régénérer la photo annotée sur disque (mêmes couleurs/labels que
la génération d'origine, voir utils/annotate.py) et met à jour le chemin
utilisé par le rapport HTML -- la vignette du rapport reflète donc l'état
corrigé, pas seulement la détection IA d'origine.

⚠️ Comme dans 09_ecran_identification.py : ressemblance visuelle, pas une
preuve -- le bandeau d'avertissement est repris tel quel.

La fenêtre reste ouverte après validation d'un canard (elle affiche juste
un badge ✓/⚠️ à côté de la ligne corrigée), pour enchaîner sur le canard
suivant de la même photo sans revenir cliquer dans le rapport.

Autonome : ne modifie rien dans scripts/, importe seulement en lecture
(utils.propositions, utils.crop_export) une fois copié à côté de scripts/
(voir README.md). Branché depuis interface_graphique.py -- voir son cas
"correction" dans _traiter_queue.
"""

import csv
import json
import queue
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

try:
    from PIL import Image, ImageTk
except ImportError:
    Image = ImageTk = None  # la fenêtre affichera un message clair plutôt que planter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ecrire_corrections import annuler_correction, traiter_validation_formulaire, enregistrer_faux_positif_canard
from formulaire_correction import ouvrir_formulaire_correction

try:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    from utils.propositions import charger_propositions, libelle_pour_score
    from utils.crop_export import save_canard_crop, nom_crop_correction
    from utils.annotate import annotate_result
except ImportError:
    # Filet de sécurité pour test isolé sans le reste du dépôt à côté --
    # la fenêtre reste utilisable, juste sans suggestions ni extraction de
    # crop réelle (l'appelant ne pourra alors pas vraiment valider).
    def charger_propositions(path, top_k=3):
        return {}

    def libelle_pour_score(score_str):
        return ("ressemblance inconnue", "#999999")

    def save_canard_crop(*args, **kwargs):
        return None

    def nom_crop_correction(*args, **kwargs):
        return None

    def annotate_result(*args, **kwargs):
        return None

    _recalcul_disponible = False
else:
    _recalcul_disponible = True

BG = "#222222"
FG = "white"
VERT = "#3fae5b"
GRIS = "#555555"
ORANGE = "#e0a030"

TAILLE_PHOTO_MAX = 700
TAILLE_SUGGESTION = 90
ZOOM_MIN = 1.0
ZOOM_MAX = 4.0
ZOOM_PAS = 1.25


def _charger_module_numerote(nom_fichier, nom_module):
    """Charge un script de scripts/ (ex. "06_proposer_correspondances_visuelles.py")
    comme un module Python normal -- nécessaire car son nom de fichier n'est
    pas un identifiant Python valide (commence par un chiffre). Utilisé
    uniquement par _recalculer_suggestions_rafale() pour réutiliser les
    fonctions de comparaison d'empreintes de 06 (charger_banque,
    meilleurs_candidats, statut_pour_score, extract_embedding) sans les
    dupliquer ici."""
    import importlib.util as ilu
    script_path = Path(__file__).resolve().parent.parent / "scripts" / nom_fichier
    spec = ilu.spec_from_file_location(nom_module, script_path)
    module = ilu.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Statuts pour lesquels la bague affichée vient du rapprochement temporel
# avec le tableur (donc PAS d'une détection IA) -- voir formulaire_correction.py,
# l'avertissement ne se déclenche que si on contredit une de ces valeurs.
STATUTS_BAGUE_CONFIRMEE_TABLEUR = ("IDENTIFIE", "IDENTIFIE_MULTIPLE")

# Statuts pour lesquels une bague existe déjà mais reste incertaine : on
# affiche quand même les suggestions visuelles, au cas où l'IA proposerait
# un meilleur candidat que le matching temporel. IDENTIFIE (bague sûre) en
# est volontairement exclu -- inutile de suggérer une alternative à une
# identification déjà confirmée sans ambiguïté.
#
# NON_RESOLU et AMBIGU_DOUBLON sont des statuts PAR CANARD INDIVIDUEL (pas
# des statuts de rafale entière) : un canard peut porter ce statut même
# sur une rafale par ailleurs bien identifiée, typiquement quand plusieurs
# canards de la même photo se disputent la même marque nasale. Sans eux
# ici, un canard concerné n'a jamais de suggestions affichées -- même s'il
# a une marque assignée exploitable -- sur les photos à plusieurs canards.
STATUTS_AVEC_SUGGESTIONS = (
    "IDENTIFIE_MULTIPLE", "AMBIGU", "NON_TROUVE_TABLEUR", "PARTIELLEMENT_IDENTIFIE",
    "NON_RESOLU", "AMBIGU_DOUBLON",
)


def _valeur_texte(*valeurs, defaut=""):
    """Renvoie la première valeur non vide de `valeurs`, ramenée à une
    chaîne exploitable.

    Sert de garde-fou contre les NaN (float) que pandas laisse dans les
    cellules Excel vides -- ces NaN survivent à l'aller-retour JSON
    (resultats_par_rafale.json) et ressortent ici comme de vrais objets
    float. Un simple `a or b or ""` ne suffit PAS : `float("nan")` est
    "truthy" en Python (bool(nan) vaut True), donc la chaîne de secours
    `or` renvoie le NaN tel quel au lieu de passer à la valeur suivante,
    et le premier `.strip()` rencontré en aval plante avec
    "AttributeError: 'float' object has no attribute 'strip'" -- c'est
    exactement l'erreur qui empêchait d'ouvrir la correction des canards
    dont la bague vient du rapprochement temporel (souvent sans couleur
    de marque détectée par l'IA, donc `couleur_marque` à NaN)."""
    for valeur in valeurs:
        if valeur is None:
            continue
        if isinstance(valeur, float) and valeur != valeur:  # NaN != NaN
            continue
        texte = str(valeur).strip()
        if texte:
            return texte
    return defaut


def _libelle_canard(canard):
    bague = _valeur_texte(canard.get("bague"))
    espece = _valeur_texte(canard.get("espece"), canard.get("espece_detectee"), defaut="?")
    sexe = _valeur_texte(canard.get("sexe"), canard.get("sexe_detectee"), defaut="?")
    statut = canard.get("status", "?")

    if canard.get("ajoute_manuellement"):
        marque_txt = "ajouté manuellement, à renseigner"
    elif canard.get("marque_absente"):
        marque_txt = "non bagué"
    elif bague:
        marque_txt = f"bague {bague}"
    elif statut in STATUTS_AVEC_SUGGESTIONS:
        marque_txt = "marque détectée, bague non lue"
    else:
        marque_txt = "marque détectée"

    return f"Canard {canard.get('canard_index', 0) + 1} · {marque_txt} · {espece} · {sexe}"


def _marque_assignee(result, canard_index):
    """La marque de result["marques"] assignée à ce canard, ou None s'il
    n'y en a pas exactement une (0 ou 2+ -- pas de crop unique exploitable,
    même limite que le reste du pipeline)."""
    marques = [m for m in result.get("marques", []) if m.get("canard_assigne") == canard_index]
    if len(marques) != 1:
        return None
    return marques[0]


def ouvrir_fenetre_rafale(parent, result, chemin_photo, dossier_dictionnaire,
                           propositions_path=None, crops_valides_dir=None,
                           on_rapport_a_mettre_a_jour=None,
                           on_photo_annotee_mise_a_jour=None,
                           serie_position=None, on_serie_suivante=None):
    """Ouvre la fenêtre "photo + liste" pour une rafale.

    Args:
        parent: Fenêtre Tkinter parente (root de l'appli).
        result: Le résultat de cette rafale (result["resultats_courants"][rafale_id]
            côté interface_graphique.py) -- dict avec "canards", "marques",
            "rafale_id", "status".
        chemin_photo: Chemin de la photo ANNOTÉE de cette rafale
            (self.chemins_annotes_courants[rafale_id]), bbox déjà dessinées.
        dossier_dictionnaire: Dossier où écrire les corrections (voir
            ecrire_corrections.py) -- typiquement <output>/dictionnaire/.
        propositions_path: Chemin de propositions_visuelles.csv (optionnel
            -- souvent absent juste après un Lancer.bat, la fenêtre
            fonctionne quand même sans lui, juste sans suggestions).
        crops_valides_dir: Dossier des crops de référence par identifiant
            (pour les vignettes des suggestions), typiquement
            <output>/crops_valides/.
        on_rapport_a_mettre_a_jour: Callable sans argument, fourni par
            interface_graphique.py (InterfaceCanard._regenerer_rapport),
            qui régénère output/rapport.html à partir des résultats
            actuellement en mémoire. Si fourni : appelé automatiquement
            après chaque validation/annulation/suppression de correction
            (colonne "Corrigé" du rapport tenue à jour sans action de
            l'utilisateur), et un bouton "🔄 Mettre à jour le rapport"
            apparaît en bas de fenêtre pour le déclencher manuellement
            (ex. après avoir corrigé plusieurs canards et vouloir
            rafraîchir l'onglet déjà ouvert dans le navigateur -- ce
            dernier n'est jamais rechargé tout seul, il faut du F5 côté
            navigateur après le clic). Absent (None) : aucune régénération,
            aucun bouton -- utile pour un test isolé de cette fenêtre.
        on_photo_annotee_mise_a_jour: Callable(rafale_id, nouveau_chemin),
            fourni par interface_graphique.py (InterfaceCanard), appelé
            juste après CHAQUE modification enregistrée (bbox déplacée/
            redimensionnée, canard ou marque ajouté/supprimé) -- voir
            _sauvegarder_resultat_rafale, qui régénère aussi la photo
            annotée sur disque (utils.annotate.annotate_result, mêmes
            couleurs/labels que le pipeline) pour qu'elle reflète l'état
            courant plutôt que la détection IA d'origine. Ce callback sert
            à mettre à jour self.chemins_annotes_courants[rafale_id] côté
            appli, pour que la vignette du rapport HTML (utils/report.py)
            pointe vers l'image fraîchement régénérée au prochain
            "🔄 Mettre à jour le rapport". Absent (None) : la photo est
            quand même régénérée sur disque (utile même en test isolé),
            juste sans notifier qui que ce soit du nouveau chemin.
        serie_position: Tuple (index_1_base, total) fourni quand cette
            fenêtre fait partie d'une revue en série (bouton "🔎 Vérifier
            toutes les rafales à vérifier" du rapport HTML -- voir
            interface_graphique.py::_ouvrir_correction_serie). Affiché dans
            le titre de la fenêtre ("(3/12)") et sur le bouton de bas de
            fenêtre. Absent (None) : fenêtre ouverte seule, comportement
            inchangé.
        on_serie_suivante: Callable sans argument, fourni uniquement quand
            serie_position l'est aussi -- appelée après fermeture de cette
            fenêtre pour enchaîner automatiquement sur la rafale suivante
            de la série (voir interface_graphique.py::_avancer_serie). Un
            bouton "▶️ Valider et passer à la suivante" remplace alors le
            bouton "Fermer" habituel ; sur la dernière rafale de la série,
            il se lit "✅ Terminer la série" mais déclenche le même
            enchaînement (qui s'arrête de lui-même, la série étant épuisée).

    Retourne le Toplevel créé.
    """
    dossier_dictionnaire = Path(dossier_dictionnaire)
    crops_corrections_dir = dossier_dictionnaire / "crops_corrections"
    crops_valides_dir = Path(crops_valides_dir) if crops_valides_dir else dossier_dictionnaire / "crops_valides"
    crops_marques_dir = dossier_dictionnaire / "crops_marques_nasales"
    propositions = charger_propositions(Path(propositions_path)) if propositions_path else {}

    def _identifiants_connus():
        """Codes de bagues déjà connus, pour le menu déroulant du champ
        Bague du formulaire -- un sous-dossier par identifiant sous
        crops_valides_dir (voir save_canard_crop / 05_construire_banque_reference.py,
        même convention que les vignettes de la section "↳ Ressemblances"
        ci-dessous). Recalculé à chaque ouverture du formulaire plutôt que
        mémorisé une fois pour toutes, pour inclure un identifiant tout
        juste créé par la correction précédente dans cette même fenêtre."""
        if not crops_valides_dir.exists():
            return []
        return sorted(p.name for p in crops_valides_dir.iterdir() if p.is_dir())

    rafale_id = result.get("rafale_id")
    canards = result.get("canards", [])

    titre_serie = f" ({serie_position[0]}/{serie_position[1]})" if serie_position else ""

    fenetre = tk.Toplevel(parent)
    fenetre.title(f"Corriger — Rafale #{rafale_id}{titre_serie}")
    fenetre.configure(bg=BG)
    fenetre.geometry("980x680")

    photos_refs = []  # référence forte, sinon Tkinter garbage-collecte les PhotoImage
    lignes_widgets = {}  # canard_index -> label de statut (pour poser ✓/⚠️ après coup)
    lignes_frames = {}  # canard_index -> Frame de la ligne (pour la détruire au 🗑️)
    boutons_annuler = {}  # canard_index -> bouton ↩️ (activé après une correction)
    crops_par_canard = {}  # canard_index -> crop_path utilisé par la dernière correction (pour ↩️/🗑️)
    # mode_dessin : ce que produit le prochain cliquer-glisser sur la photo.
    # "canard" (défaut) = nouveau canard ; "marque" = carré de marque nasale
    # pour mode_dessin["canard_index"] (armé par le bouton 🏷️ d'une ligne).
    mode_dessin = {"type": "canard", "canard_index": None}

    # -- bandeau d'avertissement (repris de 06/09) --
    tk.Label(
        fenetre, bg=BG, fg=ORANGE, font=("Helvetica", 9, "italic"), justify="left",
        text="⚠️ Les suggestions de bagues sont un indice de ressemblance visuelle, "
             "PAS une preuve d'identité -- à vérifier photo à l'appui.",
    ).pack(anchor="w", padx=14, pady=(10, 4))

    corps = tk.Frame(fenetre, bg=BG)
    corps.pack(fill="both", expand=True, padx=14, pady=6)

    # -- colonne gauche : photo annotée, interactive (cliquer-glisser = ajout manuel) --
    cadre_photo = tk.Frame(corps, bg=BG)
    cadre_photo.pack(side="left", fill="y", padx=(0, 16))

    # -- barre d'outils zoom --
    barre_zoom = tk.Frame(cadre_photo, bg=BG)
    barre_zoom.pack(fill="x", pady=(0, 4))
    lbl_zoom = tk.Label(barre_zoom, text="100%", bg=BG, fg="#bbbbbb", font=("Helvetica", 9), width=5)

    # -- canvas photo + barres de défilement (nécessaires dès qu'on zoome
    # au-delà de la taille affichée initiale) --
    cadre_canvas = tk.Frame(cadre_photo, bg=BG)
    cadre_canvas.pack()
    canvas_photo = tk.Canvas(cadre_canvas, bg="#111111", highlightthickness=0, cursor="tcross")
    vscroll_photo = ttk.Scrollbar(cadre_canvas, orient="vertical", command=canvas_photo.yview)
    hscroll_photo = ttk.Scrollbar(cadre_photo, orient="horizontal", command=canvas_photo.xview)
    canvas_photo.configure(yscrollcommand=vscroll_photo.set, xscrollcommand=hscroll_photo.set)
    canvas_photo.grid(row=0, column=0, in_=cadre_canvas)
    vscroll_photo.grid(row=0, column=1, sticky="ns", in_=cadre_canvas)
    hscroll_photo.pack(fill="x")

    # ratio taille_affichée (à l'écran, zoom courant compris) / taille_native
    # (celle utilisée par detect_canards() et par save_canard_crop) -- sert à
    # reconvertir les coordonnées d'un cadre dessiné à la souris vers des
    # bbox exploitables. Recalculé à chaque changement de zoom.
    etat_photo = {"echelle": 1.0, "disponible": False, "zoom": 1.0,
                  "img_native": None, "img_base": None, "photo_tk": None}

    def _echelle_photo():
        return etat_photo["echelle"]

    # -- édition des cadres existants : glisser une poignée de coin pour
    # redimensionner, glisser le contour pour déplacer. Couvre à la fois
    # les canards détectés par l'IA (bbox parfois imprécise à corriger) et
    # les canards/marques ajoutés à la main (_ajouter_canard_manuel /
    # _ajouter_marque_manuelle) -- ces derniers n'ont, contrairement aux
    # détections IA, AUCUN cadre déjà "cuit" dans les pixels de la photo
    # annotée (voir utils/annotate.py) : sans cette couche, ils
    # disparaissaient purement et simplement dès le clic terminé. Un même
    # mécanisme dessine donc, PAR-DESSUS la photo, un cadre pointillé pour
    # CHAQUE canard/marque de la rafale (qu'il soit d'origine IA ou
    # manuelle), redessiné au bon endroit à chaque zoom -- voir
    # _redessiner_toutes_boites, appelée en fin de _afficher_a_zoom.
    COULEUR_EDITION = {"canard": "#00e0ff", "marque": "#ffe066"}
    TAILLE_POIGNEE = 5

    boites_edition = {}  # (kind, idx) -> {"rect": id_canvas, "poignees": {coin: id_canvas}}
    edition_coin = {"kind": None, "idx": None, "coin": None, "coords": None}
    deplacement_boite = {"kind": None, "idx": None, "depart_x": 0, "depart_y": 0, "coords": None}

    def _positions_coins(x0, y0, x1, y1):
        return {"nw": (x0, y0), "ne": (x1, y0), "sw": (x0, y1), "se": (x1, y1)}

    def _bbox_vers_canvas(bbox):
        echelle = _echelle_photo()
        x, y, w, h = bbox
        return x * echelle, y * echelle, (x + w) * echelle, (y + h) * echelle

    def _canvas_vers_bbox(x0, y0, x1, y1):
        echelle = _echelle_photo()
        x0, x1 = sorted((x0, x1))
        y0, y1 = sorted((y0, y1))
        return [round(x0 / echelle), round(y0 / echelle),
                round((x1 - x0) / echelle), round((y1 - y0) / echelle)]

    def _clique_sur_boite_existante(x, y):
        """Vrai si (x, y) [coords canvas] tombe sur le contour ou une
        poignée d'un cadre déjà éditable -- sert à _debut_rectangle pour
        ignorer un clic qui doit être géré par l'item lui-même (déplacement
        ou redimensionnement) plutôt que déclencher un nouvel ajout par-dessus."""
        for item in canvas_photo.find_overlapping(x - 3, y - 3, x + 3, y + 3):
            tags = canvas_photo.gettags(item)
            if "boite_editable" in tags or "poignee_editable" in tags:
                return True
        return False

    def _supprimer_boite_edition(kind, idx):
        info = boites_edition.pop((kind, idx), None)
        if info is None:
            return
        canvas_photo.delete(info["rect"])
        for poignee_id in info["poignees"].values():
            canvas_photo.delete(poignee_id)

    def _redessiner_coords_boite(kind, idx, x0, y0, x1, y1):
        info = boites_edition.get((kind, idx))
        if info is None:
            return
        canvas_photo.coords(info["rect"], x0, y0, x1, y1)
        for coin, (cx, cy) in _positions_coins(x0, y0, x1, y1).items():
            canvas_photo.coords(
                info["poignees"][coin],
                cx - TAILLE_POIGNEE, cy - TAILLE_POIGNEE,
                cx + TAILLE_POIGNEE, cy + TAILLE_POIGNEE,
            )

    def _dessiner_boite_edition(kind, idx, bbox):
        _supprimer_boite_edition(kind, idx)
        x0, y0, x1, y1 = _bbox_vers_canvas(bbox)
        couleur = COULEUR_EDITION.get(kind, "#00e0ff")

        rect_id = canvas_photo.create_rectangle(
            x0, y0, x1, y1, outline=couleur, width=2, dash=(4, 2),
            tags=("boite_editable", f"boite_{kind}_{idx}"),
        )
        canvas_photo.tag_bind(rect_id, "<Button-1>",
                               lambda e, k=kind, i=idx: _debut_deplacement_boite(e, k, i))
        canvas_photo.tag_bind(rect_id, "<B1-Motion>", _glisser_deplacement_boite)
        canvas_photo.tag_bind(rect_id, "<ButtonRelease-1>", _fin_deplacement_boite)

        poignees = {}
        for coin, (cx, cy) in _positions_coins(x0, y0, x1, y1).items():
            poignee_id = canvas_photo.create_rectangle(
                cx - TAILLE_POIGNEE, cy - TAILLE_POIGNEE, cx + TAILLE_POIGNEE, cy + TAILLE_POIGNEE,
                fill=couleur, outline="#000000",
                tags=("poignee_editable", f"poignee_{kind}_{idx}_{coin}"),
            )
            canvas_photo.tag_bind(poignee_id, "<Button-1>",
                                   lambda e, k=kind, i=idx, c=coin: _debut_edition_coin(e, k, i, c))
            canvas_photo.tag_bind(poignee_id, "<B1-Motion>", _glisser_edition_coin)
            canvas_photo.tag_bind(poignee_id, "<ButtonRelease-1>", _fin_edition_coin)
            poignees[coin] = poignee_id

        boites_edition[(kind, idx)] = {"rect": rect_id, "poignees": poignees}

    def _redessiner_toutes_boites():
        if not photo_disponible:
            return
        for c in canards:
            bbox = c.get("bbox")
            if bbox:
                _dessiner_boite_edition("canard", c.get("canard_index", 0), bbox)
        for m in result.get("marques", []):
            bbox = m.get("bbox")
            if bbox:
                _dessiner_boite_edition("marque", m.get("marque_index", 0), bbox)

    def _debut_edition_coin(event, kind, idx, coin):
        info = boites_edition.get((kind, idx))
        if info is None:
            return
        x0, y0, x1, y1 = canvas_photo.coords(info["rect"])
        edition_coin.update(kind=kind, idx=idx, coin=coin, coords=[x0, y0, x1, y1])
        return "break"

    def _glisser_edition_coin(event):
        if edition_coin["idx"] is None:
            return
        x, y = canvas_photo.canvasx(event.x), canvas_photo.canvasy(event.y)
        x0, y0, x1, y1 = edition_coin["coords"]
        coin = edition_coin["coin"]
        if coin == "nw":
            x0, y0 = x, y
        elif coin == "ne":
            x1, y0 = x, y
        elif coin == "sw":
            x0, y1 = x, y
        else:  # "se"
            x1, y1 = x, y
        edition_coin["coords"] = [x0, y0, x1, y1]
        _redessiner_coords_boite(edition_coin["kind"], edition_coin["idx"], x0, y0, x1, y1)
        return "break"

    def _fin_edition_coin(event):
        if edition_coin["idx"] is None:
            return
        kind, idx = edition_coin["kind"], edition_coin["idx"]
        x0, y0, x1, y1 = edition_coin["coords"]
        edition_coin.update(kind=None, idx=None, coin=None, coords=None)
        _valider_nouvelle_bbox(kind, idx, x0, y0, x1, y1)
        return "break"

    def _debut_deplacement_boite(event, kind, idx):
        info = boites_edition.get((kind, idx))
        if info is None:
            return
        x0, y0, x1, y1 = canvas_photo.coords(info["rect"])
        x, y = canvas_photo.canvasx(event.x), canvas_photo.canvasy(event.y)
        deplacement_boite.update(kind=kind, idx=idx, depart_x=x, depart_y=y, coords=[x0, y0, x1, y1])
        return "break"

    def _glisser_deplacement_boite(event):
        if deplacement_boite["idx"] is None:
            return
        x, y = canvas_photo.canvasx(event.x), canvas_photo.canvasy(event.y)
        dx, dy = x - deplacement_boite["depart_x"], y - deplacement_boite["depart_y"]
        x0, y0, x1, y1 = deplacement_boite["coords"]
        _redessiner_coords_boite(deplacement_boite["kind"], deplacement_boite["idx"],
                                  x0 + dx, y0 + dy, x1 + dx, y1 + dy)
        return "break"

    def _fin_deplacement_boite(event):
        if deplacement_boite["idx"] is None:
            return
        kind, idx = deplacement_boite["kind"], deplacement_boite["idx"]
        x, y = canvas_photo.canvasx(event.x), canvas_photo.canvasy(event.y)
        dx, dy = x - deplacement_boite["depart_x"], y - deplacement_boite["depart_y"]
        x0, y0, x1, y1 = deplacement_boite["coords"]
        deplacement_boite.update(kind=None, idx=None, coords=None)
        _valider_nouvelle_bbox(kind, idx, x0 + dx, y0 + dy, x1 + dx, y1 + dy)
        return "break"

    def _regenerer_crop_si_necessaire(idx):
        """Après édition d'un cadre de canard : si une correction avait déjà
        extrait un crop pour ce canard (crops_par_canard), le ré-extrait
        avec la nouvelle bbox -- même nom de fichier (voir nom_crop_correction,
        basé sur rafale_id/canard_index, PAS sur la bbox) donc simple
        réécriture en place, aucun fichier orphelin créé."""
        if idx not in crops_par_canard:
            return
        canard = next((c for c in canards if c.get("canard_index") == idx), None)
        photo_source = result.get("photo_analysee")
        if canard is None or not photo_source or not canard.get("bbox"):
            return
        crop_path = save_canard_crop(photo_source, canard["bbox"], crops_corrections_dir, rafale_id, idx)
        if crop_path:
            crops_par_canard[idx] = crop_path

    def _regenerer_crop_marque(marque):
        """Même principe que _regenerer_crop_si_necessaire, pour une marque
        nasale déjà extraite (IA ou dessinée à la main -- voir
        _ajouter_marque_manuelle). Une marque jamais extraite (crop_filename
        absent) n'a rien à rafraîchir."""
        if not marque.get("crop_filename"):
            return
        photo_source = result.get("photo_analysee")
        if not photo_source or not marque.get("bbox"):
            return
        idx_marque = marque.get("marque_index", 0)
        crop_path = save_canard_crop(photo_source, marque["bbox"], crops_marques_dir, rafale_id, idx_marque)
        if crop_path:
            marque["crop_filename"] = Path(crop_path).name
            marque["crop_path"] = crop_path

    def _valider_nouvelle_bbox(kind, idx, x0, y0, x1, y1):
        bbox = _canvas_vers_bbox(x0, y0, x1, y1)
        if bbox[2] < 4 or bbox[3] < 4:
            # Cadre dégénéré (relâché hors photo, tremblement de souris sans
            # déplacement réel...) -- on ignore et on remet les cadres tels
            # qu'ils étaient avant le geste, plutôt que d'enregistrer une
            # bbox inexploitable.
            _redessiner_toutes_boites()
            return

        if kind == "canard":
            canard = next((c for c in canards if c.get("canard_index") == idx), None)
            if canard is None:
                return
            canard["bbox"] = bbox
            _regenerer_crop_si_necessaire(idx)
        else:
            marque = next((m for m in result.get("marques", []) if m.get("marque_index") == idx), None)
            if marque is None:
                return
            marque["bbox"] = bbox
            _regenerer_crop_marque(marque)

        _sauvegarder_resultat_rafale()
        _dessiner_boite_edition(kind, idx, bbox)
        _notifier_rapport_a_jour(silencieux=True)

    def _afficher_a_zoom(zoom):
        zoom = max(ZOOM_MIN, min(ZOOM_MAX, zoom))
        etat_photo["zoom"] = zoom
        img_base = etat_photo["img_base"]
        largeur = round(img_base.width * zoom)
        hauteur = round(img_base.height * zoom)
        img_zoom = img_base.resize((largeur, hauteur), Image.LANCZOS) if zoom != 1.0 else img_base
        photo_tk = ImageTk.PhotoImage(img_zoom)
        etat_photo["photo_tk"] = photo_tk  # remplace la référence précédente (pas besoin d'en garder l'historique)
        canvas_photo.delete("photo_img")
        canvas_photo.create_image(0, 0, anchor="nw", image=photo_tk, tags="photo_img")
        canvas_photo.lower("photo_img")
        canvas_photo.config(scrollregion=(0, 0, largeur, hauteur))
        etat_photo["echelle"] = largeur / etat_photo["img_native"].width
        lbl_zoom.config(text=f"{round(zoom * 100)}%")
        _redessiner_toutes_boites()

    def _zoom_avant():
        _afficher_a_zoom(etat_photo["zoom"] * ZOOM_PAS)

    def _zoom_arriere():
        _afficher_a_zoom(etat_photo["zoom"] / ZOOM_PAS)

    def _zoom_reinitialiser():
        _afficher_a_zoom(1.0)

    def _molette_zoom(event):
        # Ctrl + molette = zoom, comme la plupart des visionneuses d'images.
        # delta > 0 (Windows/Mac) ou event.num == 4 (Linux) -> zoom avant.
        if getattr(event, "num", None) == 4 or getattr(event, "delta", 0) > 0:
            _zoom_avant()
        else:
            _zoom_arriere()

    ttk.Button(barre_zoom, text="🔍−", width=3, command=_zoom_arriere).pack(side="left")
    lbl_zoom.pack(side="left", padx=4)
    ttk.Button(barre_zoom, text="🔍+", width=3, command=_zoom_avant).pack(side="left")
    ttk.Button(barre_zoom, text="↺ 100%", command=_zoom_reinitialiser).pack(side="left", padx=(6, 0))
    tk.Label(barre_zoom, text="  (Ctrl+molette pour zoomer)", bg=BG, fg="#777777",
             font=("Helvetica", 8, "italic")).pack(side="left")

    photo_disponible = False

    if Image is None:
        canvas_photo.config(width=320, height=200)
        canvas_photo.create_text(160, 100, text="⚠️ Pillow non installé : photo indisponible",
                                  fill=FG, width=280)
    elif chemin_photo and Path(chemin_photo).exists():
        img_native = Image.open(chemin_photo)
        img_affichee = img_native.copy()
        img_affichee.thumbnail((TAILLE_PHOTO_MAX, TAILLE_PHOTO_MAX))
        etat_photo["img_native"] = img_native
        etat_photo["img_base"] = img_affichee
        canvas_photo.config(width=img_affichee.width, height=img_affichee.height)
        photo_disponible = True  # avant l'appel : _afficher_a_zoom redessine aussi les cadres d'édition
        _afficher_a_zoom(1.0)
        canvas_photo.bind("<Control-MouseWheel>", _molette_zoom)   # Windows/Mac
        canvas_photo.bind("<Control-Button-4>", _molette_zoom)     # Linux molette haut
        canvas_photo.bind("<Control-Button-5>", _molette_zoom)     # Linux molette bas
    else:
        canvas_photo.config(width=320, height=200)
        canvas_photo.create_text(160, 100, text="⚠️ Photo annotée introuvable", fill=FG, width=280)

    lbl_mode_dessin = tk.Label(
        cadre_photo, bg=BG, fg="#888888", font=("Helvetica", 8, "italic"), justify="left",
        wraplength=TAILLE_PHOTO_MAX,
        text="✋ Cliquer-glisser sur la photo pour signaler un canard non détecté par l'IA.",
    )
    lbl_mode_dessin.pack(anchor="w", pady=(4, 0))

    # -- colonne droite : liste des canards, scrollable (rafale à N canards) --
    cadre_liste_ext = tk.Frame(corps, bg=BG)
    cadre_liste_ext.pack(side="left", fill="both", expand=True)

    canvas = tk.Canvas(cadre_liste_ext, bg=BG, highlightthickness=0)
    scrollbar = ttk.Scrollbar(cadre_liste_ext, orient="vertical", command=canvas.yview)
    cadre_liste = tk.Frame(canvas, bg=BG)
    cadre_liste.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
    canvas.create_window((0, 0), window=cadre_liste, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)
    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    msg_aucun_canard = None
    if not canards:
        msg_aucun_canard = tk.Label(cadre_liste, bg=BG, fg="#999999",
                                     text="Aucun canard détecté sur cette rafale.")
        msg_aucun_canard.pack(anchor="w", pady=10)

    def _ouvrir_correction(canard, bague_prereplie=None):
        idx = canard.get("canard_index", 0)
        bbox = canard.get("bbox")
        photo_source = result.get("photo_analysee")

        crop_path = None
        if bbox and photo_source:
            crop_path = save_canard_crop(photo_source, bbox, crops_corrections_dir, rafale_id, idx)
        if not crop_path:
            messagebox.showwarning(
                "Crop indisponible",
                "Impossible d'extraire l'image de ce canard (photo source introuvable). "
                "La correction espèce/sexe/couleur ne pourra pas être associée à une image, "
                "mais la bague peut quand même être enregistrée.",
                parent=fenetre,
            )
            crop_path = f"rafale{rafale_id}_canard{idx:02d}"  # repli : garde une clé stable
        crops_par_canard[idx] = crop_path  # mémorisé pour pouvoir annuler/supprimer plus tard

        bague_confirmee = canard.get("status") in STATUTS_BAGUE_CONFIRMEE_TABLEUR
        valeurs_initiales = {
            "crop": crop_path,
            # Ces 3 champs ne servent qu'à enregistrer_detection_manquee()
            # (voir ecrire_corrections.py) -- ignorés par le formulaire lui
            # -même et sans effet pour un canard normalement détecté par
            # l'IA (ajoute_manuellement est alors absent/False).
            "ajoute_manuellement": canard.get("ajoute_manuellement", False),
            "bbox": bbox,
            "photo_source": photo_source,
            "espece": _valeur_texte(canard.get("espece"), canard.get("espece_detectee")),
            "sexe": _valeur_texte(canard.get("sexe"), canard.get("sexe_detectee")),
            "couleur": _valeur_texte(canard.get("couleur_marque"), canard.get("couleur_detectee")),
            "bague": _valeur_texte(bague_prereplie) if bague_prereplie is not None
                     else _valeur_texte(canard.get("bague")),
            "bague_confirmee_tableur": bague_confirmee,
            # Défaut par défaut = "pas de marque" tant qu'on ne sait pas
            # (canard tout juste ajouté à la main, avant qu'un carré de
            # marque n'ait été dessiné) -- voir _ajouter_marque_manuelle,
            # qui met marque_absente à False dès qu'un carré est dessiné.
            # Sert à cocher automatiquement "🏷️ Marque présente" à
            # l'ouverture du formulaire dans ce cas, même si bague/couleur
            # sont encore vides (le chercheur vient de dessiner le carré
            # mais n'a pas encore rempli le code) -- voir formulaire_correction.py.
            "marque_deja_signalee": not canard.get("marque_absente", True),
            # Contrairement à la clé ci-dessus (qui sert seulement de
            # défaut d'affichage, donc suppose prudemment "pas de marque"
            # si on ne sait pas), celle-ci sert de PRÉDICTION IA brute
            # pour le bilan de détection -- voir ecrire_corrections.py::
            # enregistrer_revue_marque_nasale. None (pas de valeur par
            # défaut) signifie explicitement "l'IA n'a jamais évalué ce
            # canard" (ajout manuel), à ne pas confondre avec un booléen.
            "marque_absente_detectee": canard.get("marque_absente"),
        }

        def _retirer_marque_si_decochee(canard_index, bague):
            """Supprime le(s) carré(s) de marque nasale assigné(s) à ce
            canard quand la case "🏷️ Marque présente" du formulaire vient
            d'être décochée (bague finale = "PAS_MARQUE", voir
            formulaire_correction.py::_valider) -- sinon le carré resterait
            dans result["marques"] (donc dans resultats_par_rafale.json et
            le dataset de réentraînement détection, voir
            13_exporter_annotations_detection.py) alors que le chercheur
            vient justement de dire qu'il n'y a pas de marque ici.

            Ne fait rien si la case reste cochée, ou si ce canard n'a
            aucune marque assignée (cas normal : la plupart des cases
            décochées n'ont jamais eu de carré dessiné, il n'y a alors rien
            à retirer -- seul le bilan présence/absence, déjà géré par
            enregistrer_revue_marque_nasale(), est concerné)."""
            if bague != "PAS_MARQUE":
                return
            marques = result.get("marques", [])
            a_retirer = [m for m in marques if m.get("canard_assigne") == canard_index]
            if not a_retirer:
                return
            result["marques"] = [m for m in marques if m.get("canard_assigne") != canard_index]
            canard_obj = next((c for c in canards if c.get("canard_index") == canard_index), None)
            if canard_obj is not None:
                canard_obj["marque_absente"] = True
            _sauvegarder_resultat_rafale()
            for m in a_retirer:
                _supprimer_boite_edition("marque", m.get("marque_index", 0))

        def _on_valider(valeurs):
            # formulaire_correction.py reconstruit son propre dict (crop/
            # bague/espece/sexe/couleur...) sans reprendre les clés qu'on
            # lui a passées en plus dans valeurs_initiales (ajoute_manuellement/
            # bbox/photo_source) -- on les réinjecte ici depuis la closure,
            # plutôt que d'élargir le contrat de ce module partagé.
            valeurs["ajoute_manuellement"] = canard.get("ajoute_manuellement", False)
            valeurs["bbox"] = bbox
            valeurs["photo_source"] = photo_source
            resultat = traiter_validation_formulaire(dossier_dictionnaire, valeurs)
            _retirer_marque_si_decochee(idx, valeurs.get("bague"))
            _marquer_ligne_corrigee(idx, resultat)
            _notifier_rapport_a_jour(silencieux=True)

        ouvrir_formulaire_correction(fenetre, valeurs_initiales, _on_valider,
                                      identifiants_connus=_identifiants_connus())

    def _notifier_rapport_a_jour(silencieux=True):
        """Régénère output/rapport.html via le callback fourni par
        l'appelant, s'il y en a un. `silencieux=True` (cas des appels
        automatiques après validation/annulation/suppression) : n'affiche
        rien en cas de succès, un simple message discret en cas d'échec --
        on ne veut pas interrompre le chercheur avec une boîte de dialogue
        après CHAQUE correction. `silencieux=False` (bouton "🔄 Mettre à
        jour le rapport") : confirme le succès, pour que le clic ne
        paraisse pas n'avoir rien fait."""
        if on_rapport_a_mettre_a_jour is None:
            return
        try:
            on_rapport_a_mettre_a_jour()
        except Exception as e:  # noqa: BLE001 -- jamais bloquant pour la correction elle-même
            messagebox.showwarning(
                "Rapport non mis à jour",
                f"La correction a bien été enregistrée, mais la régénération du rapport a "
                f"échoué :\n{e}\n\nLe rapport reflétera cette correction au prochain "
                f"clic sur \"🔄 Mettre à jour le rapport\".",
                parent=fenetre,
            )
            return
        if not silencieux:
            messagebox.showinfo(
                "Rapport mis à jour",
                "✓ output/rapport.html régénéré avec les dernières corrections.\n\n"
                "Si tu l'as déjà ouvert dans le navigateur, pense à recharger la page (F5) "
                "pour voir les changements -- ce n'est pas automatique.",
                parent=fenetre,
            )

    def _marquer_ligne_corrigee(idx, resultat):
        lbl = lignes_widgets.get(idx)
        if lbl is None:
            return
        if resultat.get("bague_contredit_tableur"):
            lbl.config(text="⚠️ modifiée (contredit le tableur)", fg=ORANGE)
        else:
            lbl.config(text="✓ corrigé", fg=VERT)
        btn_annuler = boutons_annuler.get(idx)
        if btn_annuler is not None:
            btn_annuler.config(state="normal")

    def _annuler_ligne(idx):
        crop_path = crops_par_canard.get(idx)
        if not crop_path:
            messagebox.showinfo("Rien à annuler", "Aucune correction enregistrée pour ce canard.",
                                 parent=fenetre)
            return
        if not messagebox.askyesno(
            "Annuler la correction",
            "Annuler la dernière correction enregistrée pour ce canard "
            "(bague, espèce/sexe/couleur, exclusion) ?",
            parent=fenetre,
        ):
            return
        annuler_correction(dossier_dictionnaire, crop_path)
        lbl = lignes_widgets.get(idx)
        if lbl is not None:
            lbl.config(text="")
        btn_annuler = boutons_annuler.get(idx)
        if btn_annuler is not None:
            btn_annuler.config(state="disabled")
        _notifier_rapport_a_jour(silencieux=True)

    def _supprimer_canard(canard):
        idx = canard.get("canard_index", 0)
        if canard.get("ajoute_manuellement"):
            message = (
                "Supprimer ce canard ajouté manuellement ?\n\n"
                "Toute correction déjà enregistrée pour lui (bague, espèce/sexe/couleur) sera "
                "également annulée."
            )
        else:
            # Canard détecté par l'IA (pas un AJOUT_MANUEL) : cas typique --
            # fausse détection (ombre, reflet, deux cadres sur le même
            # canard...) qu'un chercheur veut retirer de la rafale. On est
            # plus explicite ici, car contrairement à un ajout manuel
            # récent, ce canard peut déjà porter une identification issue
            # du pipeline lui-même (matching temporel), pas seulement d'une
            # correction faite dans cette fenêtre.
            message = (
                "Supprimer ce canard détecté par l'IA ?\n\n"
                "À utiliser si ce n'est pas un vrai canard sur la photo (fausse détection) ou "
                "si un même canard a été détecté deux fois. Toute correction déjà enregistrée "
                "pour lui depuis cette fenêtre (bague, espèce/sexe/couleur) sera également "
                "annulée. La photo annotée du rapport sera régénérée sans son cadre."
            )
        if not messagebox.askyesno("Supprimer ce canard", message, parent=fenetre):
            return

        if not canard.get("ajoute_manuellement"):
            # Fausse détection IA à tracer AVANT de retirer le canard de
            # result["canards"] -- une fois supprimé, plus aucune trace
            # n'y subsisterait pour le bilan de détection (voir
            # utils/confusion.py::construire_bilan_detection_canards).
            # Nom de crop déterministe : pas besoin que le formulaire ait
            # déjà été ouvert pour ce canard (le 🗑️ est disponible
            # directement sur la ligne, sans passer par ✏️).
            photo_source = result.get("photo_analysee")
            if photo_source:
                crop_filename = nom_crop_correction(photo_source, rafale_id, idx)
                if crop_filename:
                    enregistrer_faux_positif_canard(dossier_dictionnaire, crop_filename)

        crop_path = crops_par_canard.get(idx)
        if crop_path:
            annuler_correction(dossier_dictionnaire, crop_path)
        if canard in canards:
            canards.remove(canard)
        result["canards"] = canards
        # Retire aussi la marque nasale manuelle éventuellement associée
        # (voir _ajouter_marque_manuelle) -- sinon elle resterait assignée
        # à un canard_index qui n'existe plus.
        marques = result.get("marques", [])
        marques_retirees = [m for m in marques if m.get("canard_assigne") == idx]
        result["marques"] = [m for m in marques if m.get("canard_assigne") != idx]
        _sauvegarder_resultat_rafale()

        _supprimer_boite_edition("canard", idx)
        for m in marques_retirees:
            _supprimer_boite_edition("marque", m.get("marque_index", 0))

        frame = lignes_frames.pop(idx, None)
        if frame is not None:
            frame.destroy()
        lignes_widgets.pop(idx, None)
        boutons_annuler.pop(idx, None)
        crops_par_canard.pop(idx, None)

        if not canards:
            nonlocal msg_aucun_canard
            msg_aucun_canard = tk.Label(cadre_liste, bg=BG, fg="#999999",
                                         text="Aucun canard détecté sur cette rafale.")
            msg_aucun_canard.pack(anchor="w", pady=10)

        _notifier_rapport_a_jour(silencieux=True)

    def _afficher_crop_marque(crop_filename):
        """Bouton "🔍 Marque" : ouvre un petit Toplevel avec le crop zoomé
        de la marque nasale détectée (ou dessinée à la main) pour ce
        canard -- utile pour vérifier le code/la couleur avant de remplir
        le formulaire, sans avoir à zoomer sur la photo entière où la
        marque est souvent minuscule."""
        if Image is None:
            messagebox.showwarning("Indisponible", "Pillow non installé : aperçu impossible.",
                                    parent=fenetre)
            return
        crop_path = crops_marques_dir / crop_filename
        if not crop_path.exists():
            messagebox.showwarning("Introuvable", f"Crop de marque introuvable :\n{crop_path}",
                                    parent=fenetre)
            return

        aperçu = tk.Toplevel(fenetre)
        aperçu.title(f"Marque nasale — {crop_filename}")
        aperçu.configure(bg=BG)

        img = Image.open(crop_path)
        # Agrandi si petit (souvent un tout petit crop) -- jamais réduit,
        # la marque doit rester lisible.
        facteur = max(1, min(6, TAILLE_PHOTO_MAX // max(img.width, img.height, 1)))
        img_zoom = img.resize((img.width * facteur, img.height * facteur), Image.LANCZOS)
        photo_tk = ImageTk.PhotoImage(img_zoom)
        aperçu.photo_ref = photo_tk  # référence forte, sinon garbage-collectée

        tk.Label(aperçu, image=photo_tk, bg=BG).pack(padx=10, pady=10)
        ttk.Button(aperçu, text="Fermer", command=aperçu.destroy).pack(pady=(0, 10))

        aperçu.lift()
        aperçu.attributes("-topmost", True)
        aperçu.after(200, lambda: aperçu.attributes("-topmost", False))
        aperçu.focus_force()

    def _armer_mode_marque(canard):
        idx = canard.get("canard_index", 0)
        mode_dessin["type"] = "marque"
        mode_dessin["canard_index"] = idx
        lbl_mode_dessin.config(
            fg=ORANGE,
            text=f"🏷️ Dessinez le carré de la marque nasale du canard {idx + 1}, "
                 "puis cliquer-glisser sur la photo.",
        )

    def _ajouter_ligne_canard(canard):
        idx = canard.get("canard_index", 0)

        ligne = tk.Frame(cadre_liste, bg="#2a2a2a")
        ligne.pack(fill="x", pady=3, padx=2)
        lignes_frames[idx] = ligne

        entete = tk.Frame(ligne, bg="#2a2a2a")
        entete.pack(fill="x", padx=8, pady=6)

        tk.Label(entete, text=_libelle_canard(canard), bg="#2a2a2a", fg=FG,
                 font=("Helvetica", 10), anchor="w", justify="left").pack(side="left")

        lbl_statut_ligne = tk.Label(entete, text="", bg="#2a2a2a", fg=VERT, font=("Helvetica", 9))
        lbl_statut_ligne.pack(side="left", padx=10)
        lignes_widgets[idx] = lbl_statut_ligne

        btn_annuler = ttk.Button(entete, text="↩️", width=3, state="disabled",
                                  command=lambda i=idx: _annuler_ligne(i))
        btn_annuler.pack(side="right", padx=(2, 0))
        boutons_annuler[idx] = btn_annuler

        ttk.Button(entete, text="✏️", width=3,
                   command=lambda c=canard: _ouvrir_correction(c)).pack(side="right")

        # -- 🗑️ (supprimer) : disponible pour tout canard, qu'il vienne
        # d'une détection IA (fausse détection à retirer) ou d'un ajout
        # manuel -- voir _supprimer_canard, qui adapte le message de
        # confirmation selon l'origine.
        ttk.Button(entete, text="🗑️", width=3,
                   command=lambda c=canard: _supprimer_canard(c)).pack(side="right", padx=(0, 2))

        # -- 🏷️ (dessiner le carré de la marque nasale) tant qu'aucune
        # marque nasale exploitable n'est déjà associée à ce canard --
        marque_du_canard = _marque_assignee(result, idx)
        if photo_disponible and marque_du_canard is None:
            ttk.Button(entete, text="🏷️", width=3,
                       command=lambda c=canard: _armer_mode_marque(c)).pack(side="right", padx=(0, 2))

        # -- 🔍 Marque (voir le crop zoomé de la marque nasale détectée),
        # dès qu'une marque exploitable est assignée à ce canard --
        if marque_du_canard is not None and marque_du_canard.get("crop_filename"):
            ttk.Button(entete, text="🔍 Marque", width=9,
                       command=lambda cf=marque_du_canard["crop_filename"]: _afficher_crop_marque(cf)
                       ).pack(side="right", padx=(0, 2))

        # -- suggestions visuelles, si applicable --
        if canard.get("status") in STATUTS_AVEC_SUGGESTIONS and not canard.get("marque_absente"):
            marque = _marque_assignee(result, idx)
            crop_filename = marque.get("crop_filename") if marque else None
            candidats, _statut = propositions.get(crop_filename, ([], "")) if crop_filename else ([], "")

            if candidats:
                cadre_suggestions = tk.Frame(ligne, bg="#2a2a2a")
                cadre_suggestions.pack(fill="x", padx=8, pady=(0, 8))
                tk.Label(cadre_suggestions, text="↳ Ressemblances :", bg="#2a2a2a",
                         fg="#bbbbbb", font=("Helvetica", 9)).pack(side="left", padx=(0, 6))

                for identifiant, crop_ref, score in candidats:
                    _libelle, couleur = libelle_pour_score(score)
                    pastille = tk.Frame(cadre_suggestions, bg="#3a3a3a", cursor="hand2")
                    pastille.pack(side="left", padx=3)

                    if Image is not None and crop_ref and crops_valides_dir:
                        ref_path = crops_valides_dir / identifiant / crop_ref
                        if ref_path.exists():
                            img = Image.open(ref_path)
                            img.thumbnail((TAILLE_SUGGESTION, TAILLE_SUGGESTION))
                            photo_ref = ImageTk.PhotoImage(img)
                            photos_refs.append(photo_ref)
                            tk.Label(pastille, image=photo_ref, bg="#3a3a3a").pack(padx=4, pady=(4, 0))

                    try:
                        score_txt = f"{float(score) * 100:.0f}%"
                    except (TypeError, ValueError):
                        score_txt = ""
                    tk.Label(pastille, text=f"{identifiant}  {score_txt}", bg="#3a3a3a", fg=couleur,
                             font=("Helvetica", 9, "bold")).pack(padx=6, pady=(0, 4))

                    pastille.bind(
                        "<Button-1>",
                        lambda e, c=canard, ident=identifiant: _ouvrir_correction(c, bague_prereplie=ident))
                    for w in pastille.winfo_children():
                        w.bind(
                            "<Button-1>",
                            lambda e, c=canard, ident=identifiant: _ouvrir_correction(c, bague_prereplie=ident))
            elif not propositions:
                pass  # propositions_visuelles.csv absent -- pas de section, comme dans 09
            else:
                tk.Label(ligne, text="↳ Aucune ressemblance trouvée dans la banque de référence.",
                         bg="#2a2a2a", fg="#888888", font=("Helvetica", 8, "italic")
                         ).pack(anchor="w", padx=8, pady=(0, 6))

    for canard in canards:
        _ajouter_ligne_canard(canard)

    def _sauvegarder_resultat_rafale():
        """Réécrit l'entrée de cette rafale dans resultats_par_rafale.json
        sur disque, pour que les canards/marques ajoutés ou supprimés à la
        main depuis cette fenêtre survivent à un futur relancement de
        l'appli (et soient pris en compte par un futur passage manuel de
        scripts/03_export_json_dictionnaire.py etc.). Ne fait rien si le
        fichier est introuvable -- ne doit jamais empêcher la correction
        elle-même de fonctionner en mémoire."""
        chemin = dossier_dictionnaire / "resultats_par_rafale.json"
        if not chemin.exists():
            return
        try:
            with open(chemin, encoding="utf-8") as f:
                tous_resultats = json.load(f)
            for i, r in enumerate(tous_resultats):
                if r.get("rafale_id") == rafale_id:
                    tous_resultats[i] = result
                    break
            with open(chemin, mode="w", encoding="utf-8") as f:
                json.dump(tous_resultats, f, indent=2, default=str, ensure_ascii=False)
        except Exception:  # noqa: BLE001 -- non bloquant, juste une persistance en bonus
            pass

        _regenerer_photo_annotee()

    def _regenerer_photo_annotee():
        """Redessine la photo annotée de CETTE rafale (utils.annotate.annotate_result,
        mêmes couleurs/labels/bandeau que la génération d'origine) à partir
        de l'état courant de `result` -- donc avec les cadres déplacés/
        redimensionnés, les canards/marques ajoutés ou supprimés à la main.
        Sans ça, la vignette du rapport HTML continuerait indéfiniment à
        montrer la détection IA d'origine (bbox non corrigées, canard
        supprimé encore visible, canard ajouté à la main jamais dessiné en
        dur dans les pixels) même après une correction bien enregistrée.

        Écrit au même chemin qu'à l'origine tant que le STATUT de la rafale
        n'a pas changé (annotate_result nomme le fichier d'après
        rafale_id + status, voir _sous_dossier) -- ce qui couvre le cas
        courant (on corrige une bbox ou on ajoute/supprime un canard sans
        que le statut d'ensemble de la rafale soit recalculé ici). Si le
        statut venait à changer par ailleurs, un nouveau fichier serait
        créé sans que l'ancien soit supprimé ; recalculer le statut n'est
        pas du ressort de cette fenêtre.

        Appelée uniquement depuis _sauvegarder_resultat_rafale, donc à
        chaque ajout/suppression de canard ou de marque et à chaque
        validation de cadre édité -- jamais en continu pendant un glisser.
        """
        photo_source = result.get("photo_analysee")
        if not photo_source:
            return
        try:
            nouveau_chemin = annotate_result(result, dossier_dictionnaire / "photos_annotees")
        except Exception:  # noqa: BLE001 -- non bloquant, la correction reste enregistrée
            return
        if not nouveau_chemin:
            return
        if on_photo_annotee_mise_a_jour is not None:
            try:
                on_photo_annotee_mise_a_jour(rafale_id, nouveau_chemin)
            except Exception:  # noqa: BLE001 -- jamais bloquant pour la correction elle-même
                pass

    # -- ajout manuel : cliquer-glisser sur le canvas photo. Deux modes,
    # voir mode_dessin : "canard" (défaut, cadre rouge) ou "marque" (cadre
    # jaune, armé via le bouton 🏷️ d'une ligne) --
    rect_en_cours = {"id": None, "x0": 0, "y0": 0}
    COULEUR_CADRE = {"canard": "#ff5050", "marque": "#e0c020"}

    def _debut_rectangle(event):
        if not photo_disponible:
            return
        x, y = canvas_photo.canvasx(event.x), canvas_photo.canvasy(event.y)
        if _clique_sur_boite_existante(x, y):
            # Le clic tombe sur le contour ou une poignée d'un cadre déjà
            # éditable -- c'est son propre tag_bind (_debut_deplacement_boite /
            # _debut_edition_coin) qui gère ce geste, pas un nouvel ajout.
            return
        rect_en_cours["x0"], rect_en_cours["y0"] = x, y
        if rect_en_cours["id"] is not None:
            canvas_photo.delete(rect_en_cours["id"])
        couleur = COULEUR_CADRE.get(mode_dessin["type"], "#ff5050")
        rect_en_cours["id"] = canvas_photo.create_rectangle(x, y, x, y, outline=couleur, width=2)

    def _glisser_rectangle(event):
        if rect_en_cours["id"] is None:
            return
        x, y = canvas_photo.canvasx(event.x), canvas_photo.canvasy(event.y)
        canvas_photo.coords(rect_en_cours["id"], rect_en_cours["x0"], rect_en_cours["y0"], x, y)

    def _fin_rectangle(event):
        if rect_en_cours["id"] is None:
            return
        x0, y0 = rect_en_cours["x0"], rect_en_cours["y0"]
        x1, y1 = canvas_photo.canvasx(event.x), canvas_photo.canvasy(event.y)
        canvas_photo.delete(rect_en_cours["id"])
        rect_en_cours["id"] = None

        x0, x1 = sorted((x0, x1))
        y0, y1 = sorted((y0, y1))
        # Un simple clic (cadre quasi nul) n'est pas une intention d'ajout --
        # sinon un clic accidentel sur la photo ouvrirait le formulaire.
        if (x1 - x0) < 8 or (y1 - y0) < 8:
            return

        echelle = _echelle_photo()
        bbox_natif = [
            round(x0 / echelle), round(y0 / echelle),
            round((x1 - x0) / echelle), round((y1 - y0) / echelle),
        ]  # (x, y, largeur, hauteur) en pixels natifs -- même convention que
           # partout ailleurs dans le projet (voir utils/crop_export.py,
           # result["canards"][i]["bbox"]), PAS deux coins (x0,y0,x1,y1).

        if mode_dessin["type"] == "marque":
            _ajouter_marque_manuelle(mode_dessin["canard_index"], bbox_natif)
            mode_dessin["type"] = "canard"
            mode_dessin["canard_index"] = None
            lbl_mode_dessin.config(
                fg="#888888",
                text="✋ Cliquer-glisser sur la photo pour signaler un canard non détecté par l'IA.",
            )
        else:
            _ajouter_canard_manuel(bbox_natif)

    def _ajouter_canard_manuel(bbox):
        nonlocal msg_aucun_canard
        nouvel_index = max([c.get("canard_index", -1) for c in canards], default=-1) + 1
        canard = {
            "canard_index": nouvel_index,
            "bbox": bbox,
            "status": "AJOUT_MANUEL",
            "ajoute_manuellement": True,
        }
        canards.append(canard)
        result.setdefault("canards", canards)  # canards était [] côté result si absent au départ
        _sauvegarder_resultat_rafale()
        _dessiner_boite_edition("canard", nouvel_index, bbox)  # sinon disparaît dès le clic terminé

        if msg_aucun_canard is not None:
            msg_aucun_canard.destroy()
            msg_aucun_canard = None

        _ajouter_ligne_canard(canard)
        _ouvrir_correction(canard)

    def _ajouter_marque_manuelle(canard_index, bbox):
        """Enregistre le carré de marque nasale dessiné à la main pour un
        canard donné (typiquement un AJOUT_MANUEL, qui n'a par définition
        aucune marque assignée par le pipeline). Extrait un crop haute
        résolution dans output/crops_marques_nasales/ -- même dossier et
        même convention que save_marque_crops (utils/crop_export.py) côté
        pipeline, pour que ce crop soit exploitable par un futur passage de
        03_export_json_dictionnaire.py / 06_proposer_correspondances_visuelles.py
        (suggestions de bague) tout comme une marque détectée par l'IA."""
        marques = result.setdefault("marques", [])
        nouvel_marque_index = max([m.get("marque_index", -1) for m in marques], default=-1) + 1
        photo_source = result.get("photo_analysee")

        crop_path = None
        if photo_source:
            crops_marques_dir = dossier_dictionnaire / "crops_marques_nasales"
            # save_canard_crop est un simple extracteur de bbox générique
            # (voir utils/crop_export.py) -- son paramètre "canard_index"
            # ne sert qu'à construire un nom de fichier unique, on lui
            # passe donc l'index de la marque plutôt qu'un canard ici.
            # Doit rester un entier (formaté en :02d dans le nom de fichier).
            crop_path = save_canard_crop(
                photo_source, bbox, crops_marques_dir, rafale_id, nouvel_marque_index)

        marques.append({
            "marque_index": nouvel_marque_index,
            "canard_assigne": canard_index,
            "bbox": bbox,
            "crop_filename": Path(crop_path).name if crop_path else None,
            "crop_path": crop_path,
            "ajoutee_manuellement": True,
        })

        # Le canard lui-même doit refléter qu'il porte désormais une marque
        # exploitable -- sinon _crops_a_recalculer() (bouton "🔄 Recalculer
        # les suggestions de cette photo") et la section "↳ Ressemblances"
        # continuent de l'ignorer : les deux filtrent sur
        # `canard["status"] in STATUTS_AVEC_SUGGESTIONS and not
        # canard["marque_absente"]`, qui reste faux pour un AJOUT_MANUEL
        # (status="AJOUT_MANUEL", absent de STATUTS_AVEC_SUGGESTIONS) ou un
        # canard NON_MARQUE (marque_absente=True) tant qu'on ne les met pas
        # à jour ici. Sans ça, dessiner la marque semblait "ne servir à
        # rien" : aucun recalcul possible avant la prochaine analyse
        # complète.
        canard = next((c for c in canards if c.get("canard_index") == canard_index), None)
        if canard is not None:
            canard["marque_absente"] = False
            if canard.get("status") not in STATUTS_AVEC_SUGGESTIONS:
                # NON_RESOLU : "a une marque assignée, bague pas encore
                # lue/confirmée" -- même statut que le pipeline utilise pour
                # un cas équivalent côté détection IA (voir sa définition
                # plus haut), donc déjà couvert par toute la logique
                # existante (suggestions, recalcul, libellé de ligne).
                canard["status"] = "NON_RESOLU"

        _sauvegarder_resultat_rafale()
        _dessiner_boite_edition("marque", nouvel_marque_index, bbox)  # sinon disparaît dès le clic terminé

        if not crop_path:
            messagebox.showwarning(
                "Crop indisponible",
                "Le carré a bien été enregistré, mais l'image n'a pas pu être extraite "
                "(photo source introuvable).",
                parent=fenetre,
            )
        else:
            messagebox.showinfo(
                "Marque nasale enregistrée",
                "✓ Carré de marque nasale enregistré.\n\n"
                "Clique sur \"🔄 Recalculer les suggestions de cette photo\" en bas de la fenêtre "
                "pour obtenir une suggestion de bague dès maintenant, sans attendre la prochaine "
                "analyse complète.",
                parent=fenetre,
            )

        # Le bouton 🏷️ de cette ligne n'a plus lieu d'être (une marque est
        # maintenant assignée) -- on rafraîchit juste la ligne concernée.
        _rafraichir_ligne(canard_index)

    def _rafraichir_ligne(idx, preserver_statut=False):
        """Détruit puis reconstruit la ligne d'un canard (pour refléter un
        changement -- nouvelle marque assignée, nouvelles suggestions...).
        Si preserver_statut, le badge ✓/⚠️ et l'état du bouton ↩️ sont
        recopiés sur la ligne reconstruite (sinon perdus, puisqu'une
        nouvelle ligne part toujours "vierge")."""
        texte_statut = fg_statut = etat_annuler = None
        if preserver_statut:
            lbl = lignes_widgets.get(idx)
            if lbl is not None:
                texte_statut, fg_statut = lbl.cget("text"), lbl.cget("fg")
            btn = boutons_annuler.get(idx)
            if btn is not None:
                etat_annuler = str(btn.cget("state"))

        frame = lignes_frames.get(idx)
        if frame is not None:
            frame.destroy()
            lignes_frames.pop(idx, None)

        canard_correspondant = next((c for c in canards if c.get("canard_index") == idx), None)
        if canard_correspondant is None:
            return
        _ajouter_ligne_canard(canard_correspondant)

        if preserver_statut and texte_statut is not None:
            lignes_widgets[idx].config(text=texte_statut, fg=fg_statut)
            if etat_annuler is not None:
                boutons_annuler[idx].config(state=etat_annuler)

    def _crops_a_recalculer():
        """{crop_filename: canard_index} pour les canards de CETTE rafale
        qui ont une marque assignée et pour lesquels une suggestion de
        bague a du sens -- mêmes critères que la section "↳ Ressemblances"
        (voir _ajouter_ligne_canard)."""
        crops = {}
        for c in canards:
            idx = c.get("canard_index", 0)
            if c.get("status") not in STATUTS_AVEC_SUGGESTIONS or c.get("marque_absente"):
                continue
            marque = _marque_assignee(result, idx)
            crop_filename = marque.get("crop_filename") if marque else None
            if crop_filename:
                crops[crop_filename] = idx
        return crops

    def _fusionner_propositions_csv(nouvelles_lignes):
        """Réécrit propositions_visuelles.csv en ne remplaçant QUE les
        lignes des crops recalculés -- toutes les suggestions des autres
        photos/rafales restent inchangées (c'est tout le but de ce
        recalcul ciblé plutôt que de relancer 06 sur l'ensemble)."""
        chemin = dossier_dictionnaire / "propositions_visuelles.csv"
        fieldnames = ["crop_filename", "statut"]
        for i in range(1, 4):
            fieldnames += [f"candidat_{i}", f"score_{i}", f"crop_reference_{i}"]

        lignes_existantes = {}
        if chemin.exists():
            with open(chemin, encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    lignes_existantes[row["crop_filename"]] = row
        lignes_existantes.update(nouvelles_lignes)

        chemin.parent.mkdir(parents=True, exist_ok=True)
        with open(chemin, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(lignes_existantes.values())
        return chemin

    def _recalculer_suggestions_rafale():
        """Bouton "🔄 Recalculer les suggestions de cette photo" : ne
        relance PAS toute la chaîne 03->06 (coûteux, et impacterait toutes
        les rafales) -- se contente de comparer les crops de CETTE rafale
        à la banque de référence DÉJÀ CALCULÉE (chargée telle quelle, pas
        reconstruite), puis met à jour uniquement leurs lignes dans
        propositions_visuelles.csv. Nécessaire notamment après avoir
        dessiné un carré de marque nasale à la main (_ajouter_marque_manuelle),
        qui ne crée sinon aucune suggestion avant la prochaine analyse
        complète."""
        if not _recalcul_disponible:
            messagebox.showwarning(
                "Indisponible",
                "scripts/utils n'est pas accessible depuis cette fenêtre isolée -- recalcul impossible.",
                parent=fenetre,
            )
            return

        crops_cibles = _crops_a_recalculer()
        if not crops_cibles:
            messagebox.showinfo(
                "Rien à recalculer",
                "Aucun canard de cette rafale n'a besoin d'une suggestion de bague pour l'instant "
                "(soit déjà identifié, soit sans marque assignée -- dessine son carré via 🏷️ d'abord).",
                parent=fenetre,
            )
            return

        banque_dir = dossier_dictionnaire / "banque_reference"
        if not (banque_dir / "embeddings.npz").exists():
            messagebox.showinfo(
                "Banque de référence vide",
                "Aucun individu confirmé n'est encore disponible comme référence. Lance une analyse "
                "complète au moins une fois (elle construit la banque automatiquement) avant de "
                "pouvoir recalculer une suggestion.",
                parent=fenetre,
            )
            return

        btn_recalculer.config(state="disabled", text="⏳ Calcul en cours…")
        resultat_file = queue.Queue()

        def _travail():
            try:
                propose06 = _charger_module_numerote(
                    "06_proposer_correspondances_visuelles.py", "proposer_correspondances_visuelles")
                vecteurs, index = propose06.charger_banque(banque_dir)
                crops_dir = dossier_dictionnaire / "crops_marques_nasales"
                nouvelles_lignes = {}
                for crop_filename in crops_cibles:
                    crop_path = crops_dir / crop_filename
                    if not crop_path.exists():
                        continue
                    emb = propose06.extract_embedding(crop_path)
                    candidats = propose06.meilleurs_candidats(emb, vecteurs, index, top_k=3)
                    meilleur_score = candidats[0][2] if candidats else None
                    ligne = {"crop_filename": crop_filename,
                             "statut": propose06.statut_pour_score(meilleur_score)}
                    for i in range(3):
                        if i < len(candidats):
                            ident, crop_ref, score = candidats[i]
                            ligne[f"candidat_{i + 1}"] = ident
                            ligne[f"score_{i + 1}"] = round(score, 4)
                            ligne[f"crop_reference_{i + 1}"] = crop_ref
                        else:
                            ligne[f"candidat_{i + 1}"] = ""
                            ligne[f"score_{i + 1}"] = ""
                            ligne[f"crop_reference_{i + 1}"] = ""
                    nouvelles_lignes[crop_filename] = ligne
                resultat_file.put(("ok", nouvelles_lignes))
            except Exception as e:  # noqa: BLE001 -- remonté proprement au thread Tkinter, pas planté
                resultat_file.put(("erreur", str(e)))

        threading.Thread(target=_travail, daemon=True).start()

        def _verifier():
            try:
                statut, payload = resultat_file.get_nowait()
            except queue.Empty:
                fenetre.after(150, _verifier)
                return

            btn_recalculer.config(state="normal", text="🔄 Recalculer les suggestions de cette photo")

            if statut == "erreur":
                messagebox.showerror("Erreur", f"Le recalcul a échoué :\n{payload}", parent=fenetre)
                return

            _fusionner_propositions_csv(payload)
            nonlocal propositions
            propositions = charger_propositions(dossier_dictionnaire / "propositions_visuelles.csv")
            for idx in set(crops_cibles.values()):
                _rafraichir_ligne(idx, preserver_statut=True)

            messagebox.showinfo(
                "Suggestions mises à jour",
                f"✓ {len(payload)} canard(s) de cette photo mis à jour "
                "(les suggestions des autres photos n'ont pas été touchées).",
                parent=fenetre,
            )

        fenetre.after(150, _verifier)

    if photo_disponible:
        canvas_photo.bind("<Button-1>", _debut_rectangle)
        canvas_photo.bind("<B1-Motion>", _glisser_rectangle)
        canvas_photo.bind("<ButtonRelease-1>", _fin_rectangle)

    # -- bas de fenêtre --
    bas = tk.Frame(fenetre, bg=BG)
    bas.pack(fill="x", padx=14, pady=10)
    btn_recalculer = ttk.Button(bas, text="🔄 Recalculer les suggestions de cette photo",
                                 command=_recalculer_suggestions_rafale)
    btn_recalculer.pack(side="left")
    if on_rapport_a_mettre_a_jour is not None:
        ttk.Button(bas, text="📄 Mettre à jour le rapport",
                   command=lambda: _notifier_rapport_a_jour(silencieux=False)).pack(side="left", padx=(8, 0))

    if serie_position is not None and on_serie_suivante is not None:
        # Revue en série (voir docstring) : ferme cette fenêtre puis
        # enchaîne aussitôt sur la rafale suivante -- on_serie_suivante
        # (interface_graphique.py::_avancer_serie) sait s'arrêter tout
        # seul si c'était la dernière, donc même libellé/action pour les
        # deux cas plutôt que deux boutons différents.
        index_1_base, total = serie_position
        derniere = index_1_base >= total
        texte_suivante = "✅ Terminer la série" if derniere else "▶️ Valider et passer à la suivante"

        def _passer_a_la_suivante():
            fenetre.destroy()
            on_serie_suivante()

        ttk.Button(bas, text=texte_suivante, command=_passer_a_la_suivante).pack(side="right")
        ttk.Button(bas, text="⏸️ Fermer (arrêter la série)", command=fenetre.destroy).pack(
            side="right", padx=(0, 8))
    else:
        ttk.Button(bas, text="Fermer", command=fenetre.destroy).pack(side="right")

    fenetre.lift()
    fenetre.attributes("-topmost", True)
    fenetre.after(200, lambda: fenetre.attributes("-topmost", False))
    fenetre.focus_force()

    # Ancre photos_refs à la fenêtre elle-même : sans ça, la liste locale
    # perd sa dernière référence dès que cette fonction retourne (aucune
    # closure interne ne la capture), Python la libère immédiatement, et
    # chaque PhotoImage qu'elle contenait est détruit côté Tcl -- le Label
    # se retrouve alors à pointer vers une image supprimée et s'affiche en
    # noir/vide, même si tout le reste (chargement PIL, thumbnail, config
    # du Label) s'est bien déroulé. `fenetre`, lui, reste vivant tant que
    # la fenêtre est ouverte (des callbacks de boutons le référencent),
    # donc lui accrocher photos_refs le garde vivant tout aussi longtemps.
    fenetre.photos_refs = photos_refs

    return fenetre