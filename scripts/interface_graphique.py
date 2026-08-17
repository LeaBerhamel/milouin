#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interface graphique du pipeline Canard Detection.

Pensée pour un chercheur qui n'a jamais ouvert un terminal : choisir un
dossier de photos et un tableur d'observations, cliquer sur "Lancer
l'analyse", puis ouvrir le rapport HTML généré — sans ligne de commande.

Lancement :
    - Windows : double-cliquer sur Lancer.bat (à la racine du projet)
    - Mac     : double-cliquer sur Lancer.command (à la racine du projet)
    - Ou directement : python scripts/interface_graphique.py
"""

import json
import logging
import queue
import sys
import threading
import tkinter as tk
import webbrowser
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

# --- Vérification des dépendances AVANT d'importer le pipeline -------------
# Un chercheur qui n'a pas fini son installation ne doit pas voir un
# traceback Python illisible : on affiche un message clair dans une fenêtre.
MISSING_DEPS_MSG = (
    "Une dépendance Python nécessaire n'est pas installée : {name}\n\n"
    "Pour corriger ça, ouvre un terminal dans le dossier du projet et lance :\n\n"
    "    pip install -r requirements.txt\n\n"
    "Puis relance l'interface."
)

try:
    import cv2  # noqa: F401
    import pandas as pd  # noqa: F401
    from ultralytics import YOLO  # noqa: F401
except ImportError as e:
    root = tk.Tk()
    root.withdraw()
    messagebox.showerror("Dépendance manquante", MISSING_DEPS_MSG.format(name=e.name or str(e)))
    sys.exit(1)

sys.path.insert(0, str(Path(__file__).resolve().parent))
# module_corrections/ est un dossier frère de scripts/ à la racine du projet
# (voir module_corrections/README.md) -- nécessaire pour le lien "✏️ Corriger
# cette rafale" du rapport HTML (serveur local + fenêtre "photo + liste").
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "module_corrections"))

from utils.config import BASE_DIR, DATA_DIR, MODELS_DIR, OUTPUT_DIR, REQUIRED_COLUMNS, logger  # noqa: E402
from utils.tableur import load_and_clean_tableur  # noqa: E402
from utils.report import build_report  # noqa: E402
from utils.confusion import (  # noqa: E402
    construire_matrices_confusion,
    construire_bilan_exactitude,
    construire_matrice_marque_nasale,
    construire_bilan_detection_canards,
    analyser_seuil_confiance,
    ecrire_matrice_csv,
    exporter_graphique_confusion,
)
from serveur_corrections import demarrer_serveur  # noqa: E402
from fenetre_rafale import ouvrir_fenetre_rafale  # noqa: E402

# Import tardif : ce module importe déjà lourdement (torch, ultralytics...),
# on ne veut le payer qu'une fois la fenêtre affichée.


class QueueLogHandler(logging.Handler):
    """Handler de logging qui pousse chaque message dans une queue thread-safe,
    pour affichage dans le widget Text de l'interface (Tkinter n'est pas
    thread-safe : le pipeline tourne dans un thread séparé)."""

    def __init__(self, log_queue):
        super().__init__()
        self.log_queue = log_queue

    def emit(self, record):
        self.log_queue.put(("log", self.format(record)))


class RedirectPrintToQueue:
    """Redirige les print() du pipeline (messages destinés au chercheur, ex.
    "✓ 42 rafales détectées") vers la même queue que les logs, pour qu'ils
    apparaissent dans la fenêtre au lieu de disparaître dans un terminal
    qui n'existe pas ici."""

    def __init__(self, log_queue):
        self.log_queue = log_queue
        self.buffer = ""

    def write(self, text):
        self.buffer += text
        while "\n" in self.buffer:
            line, self.buffer = self.buffer.split("\n", 1)
            self.log_queue.put(("log", line))

    def flush(self):
        pass


class InterfaceCanard(tk.Tk):
    """Fenêtre principale de l'interface."""

    def __init__(self):
        super().__init__()
        self.title("🦆 Suivi Canards — Traitement des photos")
        self.geometry("760x680")
        self.minsize(680, 560)

        self.log_queue = queue.Queue()
        self.worker_thread = None
        self.rapport_path = None
        self.output_dir_courant = None

        # -- Correction ponctuelle depuis le rapport HTML (lien "✏️ Corriger
        # cette rafale") -- voir module_corrections/README.md. Le serveur
        # tourne pendant toute la durée de vie de la fenêtre : on a besoin
        # du port avant même de lancer une analyse, pour le transmettre à
        # run_pipeline() (qui l'imprime dans le rapport HTML).
        self.serveur_corrections, self.port_corrections = demarrer_serveur(self.log_queue)
        # Rempli après chaque traitement réussi (voir _traitement_termine),
        # pour retrouver le contenu d'une rafale au clic sur son lien de
        # correction sans tout recalculer.
        self.resultats_courants = {}       # rafale_id -> result
        self.chemins_annotes_courants = {}  # rafale_id -> chemin photo annotée
        self.dictionnaire_dir_courant = None    # dossier où écrire les corrections (= output_dir)
        self.propositions_path_courant = None   # output_dir/propositions_visuelles.csv, si présent
        self.crops_valides_dir_courant = None   # output_dir/crops_valides, si présent
        # synthese/graphique du dernier traitement -- mémorisés uniquement
        # pour pouvoir régénérer rapport.html sans relancer tout le pipeline
        # (voir _regenerer_rapport, appelée depuis fenetre_rafale.py après
        # chaque correction) : build_report() en a besoin en plus des
        # résultats, mais ni l'un ni l'autre ne change entre deux
        # corrections successives sur le même lot.
        self.synthese_courante = None
        self.graphique_path_courant = None

        # -- Revue en série (bouton "🔎 Vérifier toutes les rafales à
        # vérifier" du rapport HTML) -- voir _ouvrir_correction_serie /
        # _avancer_serie. _serie_rafale_ids : la file des rafale_id encore
        # à ouvrir (déjà filtrée de celles absentes de resultats_courants) ;
        # _serie_index : position courante dans cette file.
        self._serie_rafale_ids = []
        self._serie_index = 0

        self.protocol("WM_DELETE_WINDOW", self._fermer_application)

        self._construire_interface()
        self._detecter_modeles_ia()
        self._preremplir_depuis_glisser_deposer()
        self.after(100, self._traiter_queue)

    def _fermer_application(self):
        """Arrêt propre du serveur de corrections avant de fermer la
        fenêtre (le thread est daemon donc ce n'est pas strictement
        indispensable, mais évite de laisser le port occupé un instant si
        l'utilisateur relance l'appli tout de suite après)."""
        try:
            self.serveur_corrections.shutdown()
        except Exception:  # noqa: BLE001 - on ferme la fenêtre quoi qu'il arrive
            pass
        self.destroy()

    # ------------------------------------------------------------------
    # Construction de l'interface
    # ------------------------------------------------------------------

    def _construire_interface(self):
        pad = {"padx": 10, "pady": 6}

        intro = tk.Label(
            self,
            text="Analyse les photos de pièges photographiques, identifie les canards bagués\n"
                 "et produit un rapport HTML à ouvrir dans un navigateur.",
            justify="left", fg="#333333",
        )
        intro.pack(anchor="w", **pad)

        # --- Entrées principales ---
        cadre_entrees = ttk.LabelFrame(self, text="1. Choisir les fichiers")
        cadre_entrees.pack(fill="x", **pad)

        self.var_photos = tk.StringVar(value=self._defaut_si_existe(DATA_DIR / "photos"))
        self.var_tableur = tk.StringVar(value=self._defaut_si_existe(
            DATA_DIR / "observations.xlsx", aussi=[DATA_DIR / "observations.csv"]))
        self.var_output = tk.StringVar(value=str(OUTPUT_DIR))

        self._ligne_chemin(cadre_entrees, "📁 Dossier des photos :", self.var_photos, dossier=True)
        self._ligne_chemin(cadre_entrees, "📋 Fichier tableur (.xlsx ou .csv) :", self.var_tableur, dossier=False,
                            filetypes=[("Tableur", "*.xlsx *.xls *.csv"), ("Tous les fichiers", "*.*")])
        self._ligne_chemin(cadre_entrees, "💾 Dossier de sortie (rapport, résultats) :", self.var_output, dossier=True)

        # --- Paramètres avancés ---
        cadre_avance = ttk.LabelFrame(self, text="2. Paramètres avancés (optionnel)")
        cadre_avance.pack(fill="x", **pad)
        tk.Label(cadre_avance, text="Écart max entre 2 photos d'une même rafale (secondes) :").grid(
            row=0, column=0, sticky="w", padx=10, pady=6)
        self.var_burst_gap = tk.IntVar(value=5)
        ttk.Spinbox(cadre_avance, from_=1, to=120, textvariable=self.var_burst_gap, width=6).grid(
            row=0, column=1, sticky="w", padx=10, pady=6)

        # --- Lancement ---
        cadre_action = tk.Frame(self)
        cadre_action.pack(fill="x", **pad)
        self.btn_lancer = tk.Button(
            cadre_action, text="🚀 Lancer l'analyse", command=self._lancer_analyse,
            bg="#4C8BF5", fg="white", font=("Helvetica", 12, "bold"), height=2,
        )
        self.btn_lancer.pack(side="left", fill="x", expand=True)

        self.progress = ttk.Progressbar(self, mode="determinate")
        self.progress.pack(fill="x", **pad)
        self.lbl_progress = tk.Label(self, text="")
        self.lbl_progress.pack(anchor="w", padx=10)

        # --- Journal ---
        cadre_journal = ttk.LabelFrame(self, text="Suivi du traitement")
        cadre_journal.pack(fill="both", expand=True, **pad)
        self.texte_journal = tk.Text(cadre_journal, height=14, state="disabled", bg="#111111", fg="#d0d0d0",
                                      font=("Courier", 9))
        scrollbar = ttk.Scrollbar(cadre_journal, command=self.texte_journal.yview)
        self.texte_journal.configure(yscrollcommand=scrollbar.set)
        self.texte_journal.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # --- Résultat ---
        cadre_resultat = tk.Frame(self)
        cadre_resultat.pack(fill="x", **pad)
        self.btn_ouvrir_rapport = tk.Button(
            cadre_resultat, text="📄 Ouvrir le rapport", command=self._ouvrir_rapport, state="disabled")
        self.btn_ouvrir_rapport.pack(side="left", padx=(0, 8))
        self.btn_ouvrir_dossier = tk.Button(
            cadre_resultat, text="📂 Ouvrir le dossier de sortie", command=self._ouvrir_dossier_sortie,
            state="disabled")
        self.btn_ouvrir_dossier.pack(side="left")
        self.btn_bilan_confusion = tk.Button(
            cadre_resultat, text="📊 Bilan matrice de confusion", command=self._lancer_bilan_confusion,
            state="disabled")
        self.btn_bilan_confusion.pack(side="left", padx=(8, 0))

    def _preremplir_depuis_glisser_deposer(self):
        """Si un ou plusieurs éléments ont été glissés sur le lanceur
        (Lancer.bat / Lancer.command), ils arrivent ici en argv[1:] : un
        dossier est pris comme dossier de photos, un fichier .xlsx/.xls/.csv
        comme tableur. Évite de refaire "Parcourir…" pour ce qu'on a déjà
        indiqué en déposant les fichiers sur l'icône."""
        elements_deposes = sys.argv[1:]
        if not elements_deposes:
            return

        dossier_trouve = None
        tableur_trouve = None
        for brut in elements_deposes:
            chemin = Path(brut)
            if chemin.is_dir() and dossier_trouve is None:
                dossier_trouve = chemin
            elif chemin.is_file() and chemin.suffix.lower() in (".xlsx", ".xls", ".csv") and tableur_trouve is None:
                tableur_trouve = chemin

        if dossier_trouve is not None:
            self.var_photos.set(str(dossier_trouve))
        if tableur_trouve is not None:
            self.var_tableur.set(str(tableur_trouve))

        if dossier_trouve or tableur_trouve:
            self.lbl_progress.config(
                text="📥 Dossier/fichier repris depuis le glisser-déposer — "
                     "vérifie puis clique sur \"Lancer l'analyse\"."
            )

    def _defaut_si_existe(self, chemin, aussi=None):
        candidats = [Path(chemin)] + [Path(c) for c in (aussi or [])]
        for c in candidats:
            if c.exists():
                return str(c)
        return str(candidats[0])

    def _ligne_chemin(self, parent, label, variable, dossier, filetypes=None):
        cadre = tk.Frame(parent)
        cadre.pack(fill="x", padx=10, pady=4)
        tk.Label(cadre, text=label, width=32, anchor="w").pack(side="left")
        entree = tk.Entry(cadre, textvariable=variable)
        entree.pack(side="left", fill="x", expand=True, padx=6)

        def parcourir():
            if dossier:
                chemin = filedialog.askdirectory(title=label, initialdir=variable.get() or str(BASE_DIR))
            else:
                chemin = filedialog.askopenfilename(
                    title=label, initialdir=str(Path(variable.get()).parent) if variable.get() else str(BASE_DIR),
                    filetypes=filetypes or [("Tous les fichiers", "*.*")])
            if chemin:
                variable.set(chemin)

        tk.Button(cadre, text="Parcourir…", command=parcourir).pack(side="left")

    def _detecter_modeles_ia(self):
        """Détecte automatiquement les IAs disponibles et les charge toutes (silencieusement)."""
        modeles = {
            "couleur": MODELS_DIR / "couleur" / "modele_couleur.pt",
            "espece": MODELS_DIR / "espece" / "modele_espece.pt",
            "sexe": MODELS_DIR / "sexe" / "modele_sexe.pt",
        }
        
        self.chemins_modeles = {}
        for nom, chemin in modeles.items():
            if chemin.exists():
                self.chemins_modeles[nom] = str(chemin)

    # ------------------------------------------------------------------
    # Lancement de l'analyse
    # ------------------------------------------------------------------

    def _lancer_analyse(self):
        photos_dir = self.var_photos.get().strip()
        tableur_path = self.var_tableur.get().strip()
        output_dir = self.var_output.get().strip()

        if not photos_dir or not Path(photos_dir).is_dir():
            messagebox.showerror("Dossier introuvable",
                                  f"Le dossier de photos indiqué est introuvable :\n{photos_dir}")
            return
        if not tableur_path or not Path(tableur_path).is_file():
            messagebox.showerror("Fichier introuvable",
                                  f"Le fichier tableur indiqué est introuvable :\n{tableur_path}")
            return

        erreur_colonnes = self._valider_tableur(tableur_path)
        if erreur_colonnes:
            messagebox.showerror("Tableur incomplet", erreur_colonnes)
            return

        self.btn_lancer.config(state="disabled", text="⏳ Traitement en cours…")
        self.btn_ouvrir_rapport.config(state="disabled")
        self.btn_ouvrir_dossier.config(state="disabled")
        self.btn_bilan_confusion.config(state="disabled")
        self.progress.config(value=0, maximum=100)
        self.lbl_progress.config(text="Chargement des modèles IA…")
        self.texte_journal.config(state="normal")
        self.texte_journal.delete("1.0", "end")
        self.texte_journal.config(state="disabled")

        params = {
            "photos_dir": photos_dir,
            "tableur_path": tableur_path,
            "output_dir": output_dir,
            "model_detection": str(MODELS_DIR / "detection" / "best.pt"),
            "model_couleur": self.chemins_modeles.get("couleur"),  # Charge automatiquement si disponible
            "model_espece": self.chemins_modeles.get("espece"),    # Charge automatiquement si disponible
            "model_sexe": self.chemins_modeles.get("sexe"),        # Charge automatiquement si disponible
            "burst_gap": self.var_burst_gap.get(),
            "port_corrections": self.port_corrections,
        }

        self.worker_thread = threading.Thread(target=self._executer_pipeline, args=(params,), daemon=True)
        self.worker_thread.start()

    def _valider_tableur(self, tableur_path):
        """Vérifie en amont que le tableur a les colonnes attendues, pour
        remplacer un plantage en cours de traitement par un message clair
        avant même de charger les modèles IA (qui prennent du temps)."""
        try:
            tableur = load_and_clean_tableur(tableur_path, verbose=False)
        except Exception as e:
            return (f"Impossible de lire le fichier tableur :\n{tableur_path}\n\n"
                    f"Vérifie qu'il s'agit bien d'un fichier Excel (.xlsx) ou CSV valide, "
                    f"non ouvert dans un autre programme.\n\nDétail technique : {e}")

        colonnes_manquantes = [c for c in REQUIRED_COLUMNS if c not in tableur.columns]
        if colonnes_manquantes:
            return (f"Le tableur '{Path(tableur_path).name}' ne contient pas toutes les colonnes "
                     f"attendues (après nettoyage).\n\nColonne(s) manquante(s) : "
                     f"{', '.join(colonnes_manquantes)}\n\n"
                     f"Colonnes attendues : {', '.join(REQUIRED_COLUMNS)}")
        if len(tableur) == 0:
            return (f"Le tableur '{Path(tableur_path).name}' ne contient aucune ligne exploitable "
                     f"une fois filtré (colonne 'INDENTIFICATION CORRECTE' à VRAI, mode PHOTO, "
                     f"date/heure valides). Vérifie le contenu du fichier.")
        return None

    @staticmethod
    def _charger_module_numerote(nom_fichier, nom_module):
        """Charge un script numéroté (ex. "03_export_json_dictionnaire.py")
        comme un module Python normal -- nécessaire car son nom de fichier
        n'est pas un identifiant Python valide (commence par un chiffre),
        donc pas importable avec un "import" classique. Même technique déjà
        utilisée ci-dessous pour 02_process_rafales.py."""
        import importlib.util as ilu
        script_path = Path(__file__).resolve().parent / nom_fichier
        spec = ilu.spec_from_file_location(nom_module, script_path)
        module = ilu.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    # ------------------------------------------------------------------
    # Dictionnaire MAÎTRE cumulatif (output_archives/dictionnaire_maitre/)
    # ------------------------------------------------------------------
    # Problème résolu ici : jusqu'à présent, 03->04->05->06 tournaient
    # uniquement sur le contenu de output_dir COURANT. Or après un clic sur
    # "🗄️ Archiver et repartir à zéro" (voir _archiver_resultats_courants),
    # output/ est vidé -- la banque de référence (05) et le dictionnaire
    # étaient donc reconstruits à partir de RIEN à chaque nouvelle archive,
    # perdant les individus confirmés lors des lots précédents. C'est
    # exactement ce que 07_fusionner_dictionnaire.py était censé corriger,
    # mais seulement à la main (voir son docstring et le README, section 4.6)
    # -- jamais appelé automatiquement par l'interface graphique.
    #
    # Le dossier output_archives/dictionnaire_maitre/ ci-dessous joue ce
    # rôle en continu, sans action manuelle : il contient la fusion de
    # TOUTES les archives (dictionnaire_identifications.json,
    # crops_non_identifies.csv, crops_exclus.json) ainsi que crops_valides/
    # et banque_reference/ reconstruits à partir de ce dictionnaire cumulé.
    # Il n'est jamais déplacé/vidé par 01_archiver_resultats.py puisqu'il
    # vit à côté des dossiers datés, pas dedans.

    MASTER_DIRNAME = "dictionnaire_maitre"

    @classmethod
    def _dossier_maitre(cls, archives_dir: Path) -> Path:
        d = archives_dir / cls.MASTER_DIRNAME
        d.mkdir(parents=True, exist_ok=True)
        return d

    @staticmethod
    def _fusionner_exclusions(maitre_path: Path, source_path: Path) -> list:
        """Union des crops_exclus.json du maître et d'une source (par
        crop_filename), sauvegardée dans le maître. Retourne la liste
        fusionnée (utilisée ensuite comme `exclus` pour fusionner07())."""
        def charger(p):
            if not p.exists():
                return []
            with open(p, encoding="utf-8") as f:
                return json.load(f)

        exclusions = {e["crop_filename"]: e for e in charger(maitre_path)}
        for e in charger(source_path):
            exclusions.setdefault(e["crop_filename"], e)

        fusion = list(exclusions.values())
        maitre_path.parent.mkdir(parents=True, exist_ok=True)
        with open(maitre_path, "w", encoding="utf-8") as f:
            json.dump(fusion, f, ensure_ascii=False, indent=2)
        return fusion

    def _fusionner_source_dans_maitre(self, source_dir: Path, maitre_dir: Path, log=print):
        """Intègre le dictionnaire_identifications.json (+ crops_non_identifies.csv
        + crops_exclus.json) d'UN dossier source (output/ courant, ou une
        archive datée) dans le dictionnaire maître, puis copie les crops
        confirmés de cette source dans maitre_dir/crops_valides/. Intègre
        aussi corrections.csv (+ crops_corrections/) de la même source dans
        le maître, voir 15_fusionner_corrections.py.

        Sans effet si source_dir ne contient pas de
        dictionnaire_identifications.json (rien à fusionner) -- en usage
        normal, corrections.csv ne peut exister que si ce fichier existe
        déjà aussi (une correction suppose un rapport déjà généré par
        l'étape 03), donc cette condition ne prive jamais la fusion des
        corrections en pratique."""
        dict_source = source_dir / "dictionnaire_identifications.json"
        if not dict_source.exists():
            return

        fusion07 = self._charger_module_numerote("07_fusionner_dictionnaire.py", "fusionner_dictionnaire")
        copier04 = self._charger_module_numerote("04_copier_crops_valides.py", "copier_crops_valides")

        maitre_dict_path = maitre_dir / "dictionnaire_identifications.json"
        maitre_non_id_path = maitre_dir / "crops_non_identifies.csv"
        maitre_exclus_path = maitre_dir / "crops_exclus.json"

        exclus_fusionnes = self._fusionner_exclusions(maitre_exclus_path, source_dir / "crops_exclus.json")
        exclus = {e["crop_filename"] for e in exclus_fusionnes}

        maitre_dict = fusion07.charger_dictionnaire(maitre_dict_path)
        maitre_non_id = fusion07.charger_non_identifies(maitre_non_id_path)
        source_dict = fusion07.charger_dictionnaire(dict_source)
        source_non_id = fusion07.charger_non_identifies(source_dir / "crops_non_identifies.csv")

        fusion07.fusionner(maitre_dict, maitre_non_id, source_dict, source_non_id, exclus=exclus)
        fusion07.sauvegarder(maitre_dict, maitre_non_id, maitre_dict_path, maitre_non_id_path)

        # Copie uniquement les crops confirmés PAR CETTE SOURCE (le maître
        # accumule déjà celles des sources précédentes) -- copier_crops()
        # est additif, jamais destructif (voir 04_copier_crops_valides.py).
        copier04.copier_crops(
            dict_source,
            source_dir / "crops_marques_nasales",
            maitre_dir / "crops_valides",
            log=log,
        )

        # Même problème, même remède, pour les corrections manuelles
        # (espèce/sexe/couleur) : corrections.csv suit output/ à chaque
        # archivage, donc EcranCorrections (interface_preparation_ia.py)
        # perdrait les corrections des lots précédents sans cette fusion.
        # Sans effet si source_dir ne contient pas de corrections.csv.
        corrections_source = source_dir / "corrections.csv"
        if corrections_source.exists():
            fusion15 = self._charger_module_numerote("15_fusionner_corrections.py", "fusionner_corrections_15")
            compteurs_corrections = fusion15.fusionner_corrections_csv(
                maitre_csv=maitre_dir / "corrections.csv",
                maitre_crops_dir=maitre_dir / "crops_corrections",
                source_csv=corrections_source,
                source_crops_dir=source_dir / "crops_corrections",
            )
            if compteurs_corrections["nouvelles_lignes"]:
                log(f"✓ {compteurs_corrections['nouvelles_lignes']} correction(s) manuelle(s) "
                    f"intégrée(s) au maître ({compteurs_corrections['crops_copies']} image(s) copiée(s))")

    def _rattraper_archives_non_fusionnees(self, archives_dir: Path, maitre_dir: Path, log=print):
        """Parcourt output_archives/<date>/ et fusionne dans le maître
        toute archive pas encore prise en compte -- utile pour les
        archives déjà créées AVANT ce correctif (rien n'était fusionné
        automatiquement jusqu'ici, voir explication ci-dessus), et pour
        celles créées entre deux lancements de l'interface.

        Un fichier etat_fusion.json dans maitre_dir retient les noms de
        dossiers déjà traités, pour ne jamais refaire ce travail
        (potentiellement coûteux si beaucoup d'archives) à chaque
        traitement."""
        etat_path = maitre_dir / "etat_fusion.json"
        deja_fusionnees = set(json.loads(etat_path.read_text(encoding="utf-8"))) if etat_path.exists() else set()

        if not archives_dir.exists():
            return

        a_fusionner = sorted(
            d for d in archives_dir.iterdir()
            if d.is_dir() and d.name != self.MASTER_DIRNAME and d.name not in deja_fusionnees
            and (d / "dictionnaire_identifications.json").exists()
        )
        if not a_fusionner:
            return

        log(f"↻ {len(a_fusionner)} archive(s) pas encore intégrée(s) au dictionnaire maître, rattrapage...")
        for archive in a_fusionner:
            self._fusionner_source_dans_maitre(archive, maitre_dir, log=log)
            deja_fusionnees.add(archive.name)

        etat_path.write_text(json.dumps(sorted(deja_fusionnees), ensure_ascii=False, indent=2), encoding="utf-8")
        log(f"✓ Dictionnaire maître à jour ({len(deja_fusionnees)} archive(s) intégrée(s) au total)")

    def _generer_dictionnaire_et_suggestions(self, resultat, params):
        """Enchaîne automatiquement, après chaque traitement réussi, les
        étapes qui ne demandaient jusqu'ici que la ligne de commande :
        03 (dictionnaire crop -> bague) -> 04 (crops_valides/<identifiant>)
        -> 05 (banque de référence visuelle) -> 06 (suggestions de bague
        pour les canards non identifiés par le tableur).

        Appelée depuis _executer_pipeline, dans le même thread de travail
        (donc les print() qu'elle déclenche remontent bien dans le journal
        via RedirectPrintToQueue). Volontairement encapsulée dans son propre
        try/except par l'appelant : un souci ici (ex. banque encore vide au
        tout premier lancement) ne doit jamais empêcher l'affichage du
        rapport principal.

        Returns:
            Path vers propositions_visuelles.csv si des suggestions ont été
            générées, sinon None (rien à afficher, cas normal si aucun
            individu n'est encore confirmé ou si tout a été identifié par
            le tableur).

        NOTE IMPORTANTE : la banque de référence (05) et crops_valides/
        utilisés ici sont désormais le dictionnaire MAÎTRE cumulatif
        (output_archives/dictionnaire_maitre/), pas ceux d'output_dir --
        sinon un clic sur "🗄️ Archiver et repartir à zéro" ferait perdre
        aux suggestions la mémoire des individus confirmés dans les
        archives précédentes. Voir les méthodes _fusionner_source_dans_maitre
        / _rattraper_archives_non_fusionnees juste au-dessus.
        """
        output_dir = Path(params["output_dir"])
        archives_dir = output_dir.resolve().parent / "output_archives"
        maitre_dir = self._dossier_maitre(archives_dir)

        print("\n--- 🔗 Mise à jour du dictionnaire et des suggestions de bague ---")

        # 03 : dictionnaire crop -> identifiant, à partir des résultats déjà
        # en mémoire (pas besoin de relire resultats_par_rafale.json sur
        # disque, resultat["resultats"] est la même donnée). Ce fichier
        # local à output_dir reste la source utilisée par la fenêtre de
        # correction pour ÉCRIRE les corrections de la session en cours
        # (voir ecrire_corrections.py / dossier_dictionnaire) ; c'est le
        # dictionnaire MAÎTRE (ci-dessous) qui sert de mémoire long terme.
        export03 = self._charger_module_numerote("03_export_json_dictionnaire.py", "export_dictionnaire")
        tableur = load_and_clean_tableur(str(params["tableur_path"]), verbose=False)
        identifications, non_identifies = export03.build_dictionnaire(resultat["resultats"], tableur)

        dict_path = output_dir / "dictionnaire_identifications.json"
        with open(dict_path, "w", encoding="utf-8") as f:
            json.dump(identifications, f, indent=2, ensure_ascii=False)
        csv_non_id_path = output_dir / "crops_non_identifies.csv"
        export03.export_csv(non_identifies, csv_non_id_path)
        print(f"✓ {dict_path.name} ({len(identifications)} crop(s) identifié(s))")
        print(f"✓ {csv_non_id_path.name} ({len(non_identifies)} crop(s) à comparer)")

        # Rattrape toute archive pas encore intégrée au maître (ex. archives
        # créées avant ce correctif, ou entre deux lancements de l'appli),
        # PUIS intègre le lot qu'on vient de traiter (output_dir courant).
        # Additif des deux côtés : jamais destructif pour les identifications
        # déjà connues.
        self._rattraper_archives_non_fusionnees(archives_dir, maitre_dir, log=print)
        self._fusionner_source_dans_maitre(output_dir, maitre_dir, log=print)

        # 05 : reconstruit la banque d'empreintes visuelles à partir de
        # maitre_dir/crops_valides (cumulatif sur TOUTES les archives, pas
        # seulement le lot courant -- rapide, ResNet18 déjà en cache après
        # le premier appel, voir utils/embeddings.py).
        banque05 = self._charger_module_numerote("05_construire_banque_reference.py", "construire_banque_reference")
        banque = banque05.construire_banque(maitre_dir / "crops_valides")
        banque05.sauvegarder_banque(banque, maitre_dir / "banque_reference")
        n_crops_banque = sum(len(v) for v in banque.values())
        print(f"✓ Banque de référence (cumulative) : {len(banque)} individu(s), {n_crops_banque} crop(s)")

        if not non_identifies or n_crops_banque == 0:
            print("ℹ️  Pas de suggestion à calculer (aucun canard non identifié, ou aucun individu "
                  "encore confirmé pour servir de référence).")
            return None

        # 06 : compare chaque crop non identifié DU LOT COURANT à la banque
        # maître.
        propose06 = self._charger_module_numerote(
            "06_proposer_correspondances_visuelles.py", "proposer_correspondances_visuelles")
        propositions_path = output_dir / "propositions_visuelles.csv"
        stats = propose06.proposer_correspondances(
            output_dir / "crops_marques_nasales",
            csv_non_id_path,
            maitre_dir / "banque_reference",
            propositions_path,
            log=print,
        )
        return propositions_path if stats and stats["n_lignes"] > 0 else None

    def _executer_pipeline(self, params):
        """Exécuté dans un thread séparé : redirige logs/print vers la queue,
        appelle run_pipeline(), et poste le résultat (succès ou erreur) dans
        la queue pour que le thread principal (Tkinter) l'affiche."""
        handler = QueueLogHandler(self.log_queue)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)

        ancien_stdout = sys.stdout
        sys.stdout = RedirectPrintToQueue(self.log_queue)

        try:
            # Import tardif du script numéroté (nom de fichier non-importable
            # tel quel avec "import 02_process_rafales" -> chargement manuel).
            # Coûteux (torch, ultralytics) : on ne le paie qu'au premier
            # lancement, une fois la fenêtre déjà affichée.
            process_rafales = self._charger_module_numerote("02_process_rafales.py", "process_rafales")

            def on_progress(etape, current, total):
                self.log_queue.put(("progress", (current, total)))

            resultat = process_rafales.run_pipeline(progress_callback=on_progress, **params)

            # Dictionnaire + suggestions de bague : automatique désormais,
            # mais ne doit jamais faire échouer l'affichage du rapport
            # principal si ça accroche (ex. tout premier lancement, banque
            # encore vide).
            try:
                propositions_path = self._generer_dictionnaire_et_suggestions(resultat, params)
                resultat["propositions_path"] = str(propositions_path) if propositions_path else None
            except Exception:  # noqa: BLE001
                logger.exception("Erreur pendant la génération des suggestions de bague (non bloquant)")
                resultat["propositions_path"] = None

            self.log_queue.put(("succes", resultat))
        except Exception as e:  # noqa: BLE001 - on veut attraper TOUTE erreur pour l'afficher proprement
            logger.exception("Erreur pendant le traitement")
            self.log_queue.put(("erreur", str(e)))
        finally:
            sys.stdout = ancien_stdout
            logger.removeHandler(handler)

    # ------------------------------------------------------------------
    # Communication thread de travail -> interface (queue, thread-safe)
    # ------------------------------------------------------------------

    def _traiter_queue(self):
        try:
            while True:
                type_msg, contenu = self.log_queue.get_nowait()
                if type_msg == "log":
                    self._ajouter_ligne_journal(contenu)
                elif type_msg == "progress":
                    current, total = contenu
                    self.progress.config(maximum=max(total, 1), value=current)
                    self.lbl_progress.config(text=f"Rafale {current}/{total}")
                elif type_msg == "succes":
                    self._traitement_termine(contenu)
                elif type_msg == "erreur":
                    self._traitement_echoue(contenu)
                elif type_msg == "correction":
                    self._ouvrir_correction(contenu)
                elif type_msg == "correction_serie":
                    self._ouvrir_correction_serie(contenu)
                elif type_msg == "bilan_confusion_termine":
                    self._bilan_confusion_termine(contenu)
                elif type_msg == "bilan_confusion_echoue":
                    self._bilan_confusion_echoue(contenu)
                elif type_msg == "archiver":
                    self._archiver_resultats_courants()
        except queue.Empty:
            pass
        self.after(100, self._traiter_queue)

    def _ajouter_ligne_journal(self, ligne):
        self.texte_journal.config(state="normal")
        self.texte_journal.insert("end", ligne + "\n")
        self.texte_journal.see("end")
        self.texte_journal.config(state="disabled")

    def _traitement_termine(self, resultat):
        self.rapport_path = resultat["rapport_path"]
        self.output_dir_courant = str(Path(resultat["rapport_path"]).parent)
        self.btn_lancer.config(state="normal", text="🚀 Lancer l'analyse")
        self.btn_ouvrir_rapport.config(state="normal")
        self.btn_ouvrir_dossier.config(state="normal")
        self.btn_bilan_confusion.config(state="normal")
        self.lbl_progress.config(text="✅ Traitement terminé")

        # -- Mémorisation pour les clics "✏️ Corriger cette rafale" du
        # rapport (voir _ouvrir_correction). corrections.csv /
        # dictionnaire_identifications.json / crops_exclus.json vivent
        # directement à la racine d'output_dir (même convention que
        # 03_export_json_dictionnaire.py / 08_importer_identifications_manuelles.py
        # / 10_fusionner_corrections_dataset.py), pas dans un sous-dossier.
        self.resultats_courants = {r.get("rafale_id"): r for r in resultat.get("resultats", [])}
        self.chemins_annotes_courants = resultat.get("chemins_annotes", {})
        self.dictionnaire_dir_courant = Path(self.output_dir_courant)
        self.synthese_courante = resultat.get("synthese")
        self.graphique_path_courant = resultat.get("graphique_path")

        # propositions_path est désormais calculé automatiquement à la fin de
        # _executer_pipeline (03->04->05->06 enchaînés) plutôt que simplement
        # vérifié a posteriori -- resultat["propositions_path"] fait foi ;
        # on retombe sur une vérification disque si absent (résultat produit
        # par une ancienne version / un run CLI externe).
        propositions_str = resultat.get("propositions_path")
        propositions = Path(propositions_str) if propositions_str else (
            Path(self.output_dir_courant) / "propositions_visuelles.csv")
        self.propositions_path_courant = propositions if propositions.exists() else None

        # crops_valides du dictionnaire MAÎTRE (pas celui d'output_dir seul)
        # : sinon les photos de référence affichées à côté d'une suggestion
        # ("↳ Ressemblances") ne montreraient que les individus confirmés
        # dans CE lot, alors que la banque comparée en 06 est déjà cumulative
        # sur toutes les archives (voir _generer_dictionnaire_et_suggestions).
        archives_dir = Path(self.output_dir_courant).resolve().parent / "output_archives"
        crops_valides = archives_dir / self.MASTER_DIRNAME / "crops_valides"
        self.crops_valides_dir_courant = crops_valides if crops_valides.exists() else None

        message = "Le traitement est terminé.\n\nClique sur \"Ouvrir le rapport\" pour voir les résultats dans ton navigateur."
        if self.propositions_path_courant is not None:
            message += "\n\nDes suggestions de bague (canards ressemblant à des individus déjà connus) ont été calculées automatiquement."
        messagebox.showinfo("Terminé", message)

    def _ouvrir_correction(self, contenu):
        """Appelé quand un chercheur clique sur le lien "✏️ Corriger cette
        rafale" du rapport HTML (requête reçue par
        module_corrections/serveur_corrections.py, poussée dans
        self.log_queue, consommée ici par _traiter_queue)."""
        rafale_id = contenu.get("rafale_id")
        result = self.resultats_courants.get(rafale_id)

        if result is None or self.dictionnaire_dir_courant is None:
            messagebox.showwarning(
                "Correction indisponible",
                "Impossible d'ouvrir cette correction : aucun résultat de traitement n'est "
                "en mémoire pour cette rafale (l'application a peut-être été relancée depuis "
                "ce clic, ou un nouveau traitement a été lancé entre-temps).\n\n"
                "Relance l'analyse et rouvre le rapport fraîchement généré.",
            )
            return

        chemin_photo = self.chemins_annotes_courants.get(rafale_id)
        ouvrir_fenetre_rafale(
            self, result, chemin_photo, self.dictionnaire_dir_courant,
            propositions_path=self.propositions_path_courant,
            crops_valides_dir=self.crops_valides_dir_courant,
            on_rapport_a_mettre_a_jour=self._regenerer_rapport,
            on_photo_annotee_mise_a_jour=self._photo_annotee_mise_a_jour,
        )

    def _ouvrir_correction_serie(self, contenu):
        """Appelé quand un chercheur clique sur "🔎 Vérifier toutes les
        rafales à vérifier" du rapport HTML (requête reçue par
        module_corrections/serveur_corrections.py, route /corriger_serie,
        poussée dans self.log_queue, consommée ici par _traiter_queue).

        Ne fait qu'amorcer la file (_serie_rafale_ids/_serie_index) : c'est
        _ouvrir_rafale_serie_courante qui ouvre effectivement chaque
        fenêtre, une à la fois, et _avancer_serie qui la fait progresser
        au fil des validations (voir fenetre_rafale.ouvrir_fenetre_rafale,
        paramètre on_serie_suivante)."""
        rafale_ids = contenu.get("rafale_ids", [])

        # Mêmes rafale_id que ceux du rapport HTML au moment du clic, mais
        # resultats_courants a pu changer entre-temps (nouveau traitement
        # lancé, ou appli relancée) -- on filtre plutôt que de planter au
        # milieu de la série sur une rafale introuvable.
        self._serie_rafale_ids = [rid for rid in rafale_ids if rid in self.resultats_courants]
        self._serie_index = 0
        nb_ignorees = len(rafale_ids) - len(self._serie_rafale_ids)

        if not self._serie_rafale_ids:
            messagebox.showwarning(
                "Vérification indisponible",
                "Impossible d'ouvrir cette série : aucun résultat de traitement n'est en mémoire "
                "pour ces rafales (l'application a peut-être été relancée depuis ce clic, ou un "
                "nouveau traitement a été lancé entre-temps).\n\n"
                "Relance l'analyse et rouvre le rapport fraîchement généré.",
            )
            return

        if nb_ignorees:
            messagebox.showinfo(
                "Certaines rafales sont introuvables",
                f"{nb_ignorees} rafale(s) sur {len(rafale_ids)} n'ont pas pu être retrouvées "
                f"(résultats plus en mémoire) et seront sautées. Les {len(self._serie_rafale_ids)} "
                "autres vont s'ouvrir l'une après l'autre.",
            )

        self._ouvrir_rafale_serie_courante()

    def _ouvrir_rafale_serie_courante(self):
        """Ouvre la fenêtre "photo + liste" de la rafale courante de
        _serie_rafale_ids (pointée par _serie_index), ou annonce la fin de
        la série si la file est épuisée. Voir _ouvrir_correction_serie /
        _avancer_serie."""
        if self._serie_index >= len(self._serie_rafale_ids):
            messagebox.showinfo(
                "Série terminée",
                f"Les {len(self._serie_rafale_ids)} rafale(s) de la série ont été passées en revue.",
            )
            return

        rafale_id = self._serie_rafale_ids[self._serie_index]
        result = self.resultats_courants.get(rafale_id)
        if result is None or self.dictionnaire_dir_courant is None:
            # Devenue introuvable entre l'amorçage de la série et son tour
            # (cas rare -- ex. supprimée depuis une autre fenêtre pendant
            # la série) : on saute plutôt que d'interrompre la revue.
            self._avancer_serie()
            return

        chemin_photo = self.chemins_annotes_courants.get(rafale_id)
        ouvrir_fenetre_rafale(
            self, result, chemin_photo, self.dictionnaire_dir_courant,
            propositions_path=self.propositions_path_courant,
            crops_valides_dir=self.crops_valides_dir_courant,
            on_rapport_a_mettre_a_jour=self._regenerer_rapport,
            on_photo_annotee_mise_a_jour=self._photo_annotee_mise_a_jour,
            serie_position=(self._serie_index + 1, len(self._serie_rafale_ids)),
            on_serie_suivante=self._avancer_serie,
        )

    def _avancer_serie(self):
        """Callback fourni à fenetre_rafale.ouvrir_fenetre_rafale
        (paramètre on_serie_suivante) : appelée juste après la fermeture de
        la fenêtre de la rafale courante (bouton "▶️ Valider et passer à la
        suivante" / "✅ Terminer la série"), enchaîne sur la rafale
        suivante de la file."""
        self._serie_index += 1
        self._ouvrir_rafale_serie_courante()

    def _photo_annotee_mise_a_jour(self, rafale_id, nouveau_chemin):
        """Callback fourni à fenetre_rafale.ouvrir_fenetre_rafale : appelé
        juste après chaque correction qui modifie les canards/marques d'une
        rafale (bbox éditée, canard/marque ajouté ou supprimé), une fois la
        photo annotée correspondante régénérée sur disque (voir
        fenetre_rafale._regenerer_photo_annotee). Met à jour
        self.chemins_annotes_courants pour que le prochain
        _regenerer_rapport() (déclenché juste après par le même appelant,
        via on_rapport_a_mettre_a_jour) fasse pointer la vignette du
        rapport HTML vers l'image fraîchement corrigée plutôt que vers la
        détection IA d'origine."""
        self.chemins_annotes_courants[rafale_id] = nouveau_chemin

    def _regenerer_rapport(self):
        """Reconstruit output/rapport.html à partir de ce qui est
        actuellement en mémoire (self.resultats_courants -- qui inclut les
        canards ajoutés/supprimés à la main depuis la fenêtre "photo +
        liste", voir fenetre_rafale._sauvegarder_resultat_rafale) et des
        fichiers de correction déjà écrits sur disque par
        ecrire_corrections.py (colonne "Corrigé", voir utils/report.py).

        Ne relance PAS le pipeline (détection/classification/matching/
        synthèse) : self.synthese_courante et self.graphique_path_courant
        restent ceux du dernier traitement complet, seule la colonne
        "Corrigé" (et Bague(s)/Espèce/Sexe si le canard corrigé a changé
        de valeur en mémoire) évolue d'un appel à l'autre.

        Appelée automatiquement après chaque validation/annulation/
        suppression de correction, et par le bouton "🔄 Mettre à jour le
        rapport" de la fenêtre "photo + liste" -- voir
        fenetre_rafale.ouvrir_fenetre_rafale, paramètre
        on_rapport_a_mettre_a_jour. Toute exception est laissée remonter :
        c'est fenetre_rafale._notifier_rapport_a_jour qui l'attrape et
        l'affiche proprement au chercheur, pas ici.
        """
        if self.output_dir_courant is None or self.synthese_courante is None:
            raise RuntimeError(
                "Aucun rapport en mémoire à régénérer (relance une analyse complète d'abord)."
            )
        resultats = list(self.resultats_courants.values())
        rapport_path = build_report(
            resultats, self.synthese_courante, self.chemins_annotes_courants,
            self.output_dir_courant, self.graphique_path_courant, self.port_corrections,
        )
        self.rapport_path = rapport_path
        return rapport_path

    def _archiver_resultats_courants(self):
        """Appelé quand un chercheur clique sur "🗄️ Archiver et repartir à
        zéro" dans le rapport HTML (requête reçue par
        module_corrections/serveur_corrections.py, route /archiver, poussée
        dans self.log_queue, consommée ici par _traiter_queue).

        Déplace tout output/ vers output_archives/<date>/ (via
        01_archiver_resultats.py, chargé dynamiquement -- même principe que
        _charger_script_numerique() dans interface_preparation_ia.py, son
        nom commençant par un chiffre) puis réinitialise tout ce qui
        pointait vers l'ancien output/, pour que _ouvrir_correction affiche
        proprement son message "Correction indisponible" si un ancien lien
        de ce rapport est encore cliqué après coup, plutôt que d'écrire
        dans un dossier qui n'existe plus.
        """
        if not self.output_dir_courant:
            messagebox.showinfo(
                "Rien à archiver",
                "Aucun résultat en mémoire à archiver (aucun traitement n'a encore été lancé "
                "depuis l'ouverture de l'application, ou un archivage a déjà eu lieu).",
            )
            return

        if not messagebox.askyesno(
            "Archiver les résultats",
            f"Déplacer le contenu actuel de « {self.output_dir_courant} » vers "
            f"output_archives/<date>/ et repartir d'un dossier de résultats vide ?\n\n"
            "Le rapport actuellement ouvert dans le navigateur ne reflétera plus le dossier "
            "après cette action (photos et liens de correction resteront affichés tels quels, "
            "mais ne pointeront plus vers un dossier à jour).\n\n"
            "Confirmer l'archivage ?",
        ):
            return

        try:
            import importlib.util as ilu
            script_path = Path(__file__).resolve().parent / "01_archiver_resultats.py"
            spec = ilu.spec_from_file_location("archivage_01", script_path)
            mod01 = ilu.module_from_spec(spec)
            spec.loader.exec_module(mod01)
            destination = mod01.archiver(Path(self.output_dir_courant))
        except Exception as e:  # noqa: BLE001
            logger.exception("Erreur pendant l'archivage")
            messagebox.showerror("L'archivage a échoué", str(e))
            return

        # Fusionne immédiatement ce lot dans le dictionnaire/corrections
        # maître (output_archives/dictionnaire_maitre/), plutôt que
        # d'attendre le traitement d'un prochain lot (voir
        # _fusionner_source_dans_maitre / _rattraper_archives_non_fusionnees
        # plus haut, qui ne s'exécutaient jusqu'ici qu'après un nouveau
        # 02_process_rafales). Sans ça, un chercheur qui archive puis va
        # directement dans "🔁 Réentraîner depuis les corrections…" ne
        # trouve rien : le champ par défaut pointe déjà vers le maître
        # (interface_preparation_ia.py, EcranCorrections), mais celui-ci
        # n'aurait pas encore ce lot tant qu'aucun nouveau traitement n'a
        # été lancé.
        message_fusion = ""
        if destination is not None:
            try:
                archives_dir = destination.parent
                maitre_dir = self._dossier_maitre(archives_dir)
                self._fusionner_source_dans_maitre(destination, maitre_dir, log=print)
            except Exception:  # noqa: BLE001
                logger.exception("Erreur pendant la fusion vers le dictionnaire maître après archivage")
                message_fusion = (
                    "\n\n⚠️ La fusion vers le dictionnaire/corrections maître a échoué "
                    "(voir pipeline_production.log). Les fichiers archivés restent intacts, "
                    "mais \"Réentraîner depuis les corrections…\" ne les verra pas tant que "
                    "ce n'est pas corrigé -- utilise \"Parcourir…\" pour pointer directement "
                    f"vers {destination / 'corrections.csv'} en attendant."
                )

        # Tout ce qui suit pointait vers l'ancien output/ -- on repart à
        # zéro comme au tout premier lancement de l'appli (voir __init__).
        self.rapport_path = None
        self.output_dir_courant = None
        self.resultats_courants = {}
        self.chemins_annotes_courants = {}
        self.dictionnaire_dir_courant = None
        self.propositions_path_courant = None
        self.crops_valides_dir_courant = None
        self.btn_ouvrir_rapport.config(state="disabled")
        self.btn_ouvrir_dossier.config(state="disabled")
        self.btn_bilan_confusion.config(state="disabled")

        if destination is None:
            messagebox.showinfo("Rien à archiver", "Le dossier de résultats était déjà vide.")
        else:
            messagebox.showinfo(
                "Archivage terminé",
                f"Résultats déplacés vers :\n{destination}\n\n"
                "Le dossier de résultats est prêt pour un nouveau traitement." + message_fusion,
            )

    def _traitement_echoue(self, message_erreur):
        self.btn_lancer.config(state="normal", text="🚀 Lancer l'analyse")
        self.lbl_progress.config(text="❌ Le traitement a échoué")
        messagebox.showerror(
            "Le traitement a échoué",
            f"{message_erreur}\n\n"
            "Le détail technique complet a été enregistré dans pipeline_production.log, "
            "à la racine du projet, si tu dois le transmettre pour de l'aide."
        )

    def _lancer_bilan_confusion(self):
        """Bouton "📊 Bilan matrice de confusion" : équivalent, depuis
        l'appli, de lancer `python 14_bilan_matrice_confusion.py --output
        <output_dir>` en ligne de commande -- même logique
        (utils/confusion.py), juste déclenchée par un clic plutôt qu'un
        terminal, et affichée dans une popup plutôt qu'imprimée sur
        stdout. Écrit les mêmes fichiers dans output_dir : un CSV par
        classe avec au moins une confusion enregistrée, et
        matrice_confusion.png (voir exporter_graphique_confusion).

        Utilise self.resultats_courants (l'état en mémoire, tenu à jour
        au fil des corrections -- voir _regenerer_rapport) plutôt que de
        relire resultats_par_rafale.json sur disque : les deux sont
        normalement synchronisés (fenetre_rafale._sauvegarder_resultat_rafale
        réécrit ce fichier à chaque ajout/suppression de canard/marque),
        mais s'appuyer sur l'état déjà en mémoire évite une lecture disque
        redondante et reste cohérent avec le reste de l'appli.

        Lancé dans un thread séparé (comme le pipeline principal) : la
        génération du graphique importe matplotlib et peut prendre une
        fraction de seconde à quelques secondes sur un gros dictionnaire
        de corrections -- pas de raison de geler la fenêtre pour ça."""
        if self.dictionnaire_dir_courant is None:
            messagebox.showwarning(
                "Indisponible",
                "Lance d'abord une analyse (ce bilan a besoin des résultats et de "
                "dictionnaire/corrections.csv pour comparer prédictions IA et corrections).",
            )
            return

        self.btn_bilan_confusion.config(state="disabled", text="⏳ Calcul en cours…")
        resultats = list(self.resultats_courants.values())
        dossier_dictionnaire = self.dictionnaire_dir_courant
        output_dir = Path(self.output_dir_courant)

        def _travail():
            try:
                matrices, stats = construire_matrices_confusion(resultats, dossier_dictionnaire)

                chemins_csv = []
                for classe, matrice in matrices.items():
                    if not matrice:
                        continue
                    chemin = ecrire_matrice_csv(matrice, output_dir / f"matrice_confusion_{classe}.csv")
                    chemins_csv.append(chemin)

                chemin_png = (
                    exporter_graphique_confusion(matrices, output_dir / "matrice_confusion.png")
                    if any(matrices.values()) else None
                )

                bilan = construire_bilan_exactitude(resultats, dossier_dictionnaire, matrices=matrices)
                analyse_confiance = analyser_seuil_confiance(resultats, dossier_dictionnaire)
                marque_nasale = construire_matrice_marque_nasale(dossier_dictionnaire)
                detection_canards = construire_bilan_detection_canards(dossier_dictionnaire)

                self.log_queue.put(("bilan_confusion_termine", {
                    "stats": stats,
                    "chemins_csv": chemins_csv,
                    "chemin_png": chemin_png,
                    "bilan": bilan,
                    "analyse_confiance": analyse_confiance,
                    "marque_nasale": marque_nasale,
                    "detection_canards": detection_canards,
                }))
            except Exception as e:  # noqa: BLE001 -- remonté proprement au thread Tkinter, pas planté
                logger.exception("Erreur pendant le calcul du bilan matrice de confusion")
                self.log_queue.put(("bilan_confusion_echoue", str(e)))

        threading.Thread(target=_travail, daemon=True).start()

    def _bilan_confusion_termine(self, contenu):
        """Consomme le résultat poussé par _lancer_bilan_confusion (voir
        _traiter_queue) : réactive le bouton et résume le bilan dans une
        popup, avec le même rappel méthodologique que le script CLI (voir
        utils/confusion.py -- ne couvre que les cas relus ET corrigés)."""
        self.btn_bilan_confusion.config(state="normal", text="📊 Bilan matrice de confusion")
        stats = contenu["stats"]
        marque_nasale = contenu["marque_nasale"]
        detection_canards = contenu["detection_canards"]

        rien_du_tout = (
            stats["nb_exploitees"] == 0
            and marque_nasale["nb_revus"] == 0
            and (detection_canards["precision"] is None and detection_canards["rappel"] is None)
        )
        if rien_du_tout:
            messagebox.showinfo(
                "Aucune donnée exploitable",
                "Aucune correction espèce/sexe/couleur, aucune revue de marque nasale et aucune "
                "donnée de détection de canard pour l'instant.\n\n"
                f"{stats['nb_corrections_lues']} ligne(s) lues dans corrections.csv, "
                f"{stats['nb_ignorees_pas_de_correspondance']} ignorée(s) (pas de correspondance "
                "avec une rafale connue, ou aucune valeur corrigée exploitable).",
            )
            return

        lignes = []

        if stats["nb_exploitees"] == 0:
            lignes.append("Aucune correction espèce/sexe/couleur exploitable pour l'instant "
                           f"({stats['nb_corrections_lues']} ligne(s) lues dans corrections.csv, "
                           f"{stats['nb_ignorees_pas_de_correspondance']} ignorée(s)).")
            lignes.append("")
        else:
            lignes.append(
                f"{stats['nb_exploitees']} correction(s) exploitée(s) sur {stats['nb_corrections_lues']} "
                f"lue(s) dans corrections.csv ({stats['nb_ignorees_pas_de_correspondance']} ignorée(s))."
            )
            lignes.append("")

            for classe, b in contenu["bilan"].items():
                if b["nb_revus"] == 0:
                    continue
                pct = b["taux_exactitude"] * 100
                lignes.append(f"{classe.capitalize()} : {pct:.0f}% correct sur {b['nb_revus']} cas relus "
                              f"({b['nb_corriges']} corrigé(s)).")

            buckets = contenu["analyse_confiance"]["buckets"]
            if buckets:
                lignes.append("")
                lignes.append("Par tranche de confiance :")
                for b in buckets:
                    pct = b["taux_exactitude"] * 100
                    lignes.append(f"  [{b['borne_basse']*100:.0f}-{b['borne_haute']*100:.0f}%) : "
                                  f"{pct:.0f}% correct sur {b['nb_revus']} cas")

        # Détection de marque nasale (présente/absente) -- couvre TOUS les
        # cas relus (changés ou non), indépendant des corrections espèce/
        # sexe/couleur ci-dessus.
        lignes.append("")
        lignes.append("--- Détection de marque nasale ---")
        if marque_nasale["nb_revus"] == 0:
            lignes.append("Aucune revue enregistrée pour l'instant.")
        else:
            lignes.append(
                f"VP {marque_nasale['vrai_positif']} / FP {marque_nasale['faux_positif']} / "
                f"FN {marque_nasale['faux_negatif']} / VN {marque_nasale['vrai_negatif']}"
            )
            lignes.append(
                f"Précision {marque_nasale['precision']*100:.0f}% · "
                f"Rappel {marque_nasale['rappel']*100:.0f}% · "
                f"Exactitude {marque_nasale['exactitude']*100:.0f}% "
                f"(sur {marque_nasale['nb_revus']} cas relus)"
            )

        # Détection de canard elle-même -- pas de vrai négatif défini pour
        # une tâche de détection, précision/rappel seulement.
        lignes.append("")
        lignes.append("--- Détection de canard ---")
        if detection_canards["precision"] is None and detection_canards["rappel"] is None:
            lignes.append("Aucune donnée pour l'instant.")
        else:
            lignes.append(
                f"VP {detection_canards['vrai_positif']} / FP {detection_canards['faux_positif']} / "
                f"FN {detection_canards['faux_negatif']}"
            )
            morceaux = []
            if detection_canards["precision"] is not None:
                morceaux.append(f"Précision {detection_canards['precision']*100:.0f}%")
            if detection_canards["rappel"] is not None:
                morceaux.append(f"Rappel {detection_canards['rappel']*100:.0f}%")
            if morceaux:
                lignes.append(" · ".join(morceaux))

        if contenu["chemins_csv"] or contenu["chemin_png"]:
            lignes.append("")
            noms = [Path(c).name for c in contenu["chemins_csv"]]
            if contenu["chemin_png"]:
                noms.append(Path(contenu["chemin_png"]).name)
            lignes.append("Fichiers écrits dans le dossier de sortie : " + ", ".join(noms))

        lignes.append("")
        lignes.append(
            "⚠️ Ne couvre que les cas relus par choix du chercheur (souvent les cas déjà "
            "suspects), pas un échantillon aléatoire de la production -- pas un taux "
            "d'exactitude global de l'IA."
        )
        messagebox.showinfo("Bilan matrice de confusion", "\n".join(lignes))

    def _bilan_confusion_echoue(self, message_erreur):
        self.btn_bilan_confusion.config(state="normal", text="📊 Bilan matrice de confusion")
        messagebox.showerror("Le calcul du bilan a échoué", message_erreur)

    def _ouvrir_rapport(self):
        if self.rapport_path:
            webbrowser.open(Path(self.rapport_path).resolve().as_uri())

    def _ouvrir_dossier_sortie(self):
        if not self.output_dir_courant:
            return
        chemin = Path(self.output_dir_courant).resolve()
        if sys.platform == "win32":
            import os
            os.startfile(chemin)  # noqa: S606
        elif sys.platform == "darwin":
            import subprocess
            subprocess.run(["open", str(chemin)], check=False)
        else:
            import subprocess
            subprocess.run(["xdg-open", str(chemin)], check=False)


def main():
    app = InterfaceCanard()
    app.mainloop()


if __name__ == "__main__":
    main()