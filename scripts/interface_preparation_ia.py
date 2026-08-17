#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interface graphique de préparation des datasets de classification (Couleur de
marque, Espèce) avant ré-entraînement.

Pensée pour un chercheur : répondre à "as-tu ajouté une espèce / une couleur ?",
extraire automatiquement les nouveaux crops depuis raw_data/, les trier à l'œil
(clic sur un bouton, avec possibilité d'ajouter une classe qui n'existait pas
encore), puis soit lancer l'entraînement directement sur cet ordinateur, soit
préparer un zip à envoyer sur le notebook Colab si l'ordinateur n'a pas de
carte graphique adaptée. Termine par l'enregistrement de la version dans
gestion_modeles.py (et, si voulu, sa promotion comme modèle actif).

Pré-requis : les annotations dans models/detection/raw_data/labels/ doivent
déjà être au NOUVEAU schéma à 2 classes (0=marque_nasale, 1=canard) — voir
scripts/00_migrate_annotations.py si ce n'est pas encore le cas. L'interface
détecte les jobs qui utilisent encore l'ancien schéma et prévient au lieu de
les ignorer silencieusement.

Lancement :
    python scripts/interface_preparation_ia.py
"""

import json
import queue
import shutil
import sys
import threading
import tkinter as tk
import zipfile
from datetime import date
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from types import SimpleNamespace

MISSING_DEPS_MSG = (
    "Une dépendance Python nécessaire n'est pas installée : {name}\n\n"
    "Pour corriger ça, ouvre un terminal dans le dossier du projet et lance :\n\n"
    "    pip install -r requirements.txt\n\n"
    "Puis relance l'interface."
)

try:
    from PIL import Image, ImageTk
except ImportError as e:
    root = tk.Tk()
    root.withdraw()
    messagebox.showerror("Dépendance manquante", MISSING_DEPS_MSG.format(name=e.name or str(e)))
    sys.exit(1)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils.burst_detector import BurstDetector  # noqa: E402
from utils.config import BASE_DIR, BURST_GAP_DEFAULT, MODELS_DIR, logger  # noqa: E402
from utils.fichiers import find_image, list_job_dirs  # noqa: E402
from utils.nuit import est_photo_de_nuit  # noqa: E402

REPO_ROOT = BASE_DIR
RAW_IMAGES = MODELS_DIR / "detection" / "raw_data" / "images"
RAW_LABELS = MODELS_DIR / "detection" / "raw_data" / "labels"
MARGIN = 0.15  # marge autour des boîtes, cohérent avec les scripts 00_prepare_dataset_*

# Une IA "classification" = une boîte de détection source (marque_nasale ou
# canard) à découper, un dossier racine ImageFolder, et le nom de fichier
# modèle attendu par gestion_modeles.py.
IA_DEFS = {
    "couleur": dict(
        label="Couleur de marque nasale",
        detection_class_id="0",
        staging_dir=REPO_ROOT / "dataset_couleur",
        crop_prefix="marque",
        model_key="couleur",
        model_filename="modele_couleur.pt",
    ),
    "espece": dict(
        label="Espèce",
        detection_class_id="1",
        staging_dir=REPO_ROOT / "dataset_espece",
        crop_prefix="canard",
        model_key="espece",
        model_filename="modele_espece.pt",
    ),
    # "sexe" n'est volontairement PAS ajouté au tri manuel (EcranConfig fixe
    # encore ias_a_traiter = ["couleur", "espece"] en dur) : le flux
    # extraction/tri par bbox n'a pas été adapté pour distinguer espèce et
    # sexe sur le même crop "canard". En revanche "sexe" est un type de
    # modèle à part entière pour le flux de réentraînement par corrections
    # (voir EcranCorrections ci-dessous), qui lui ne dépend pas de raw_data/
    # -- d'où cette entrée, utilisée seulement par ce second flux (label
    # affiché dans EcranVersioning, dossier de dataset dans CORRECTIONS_TYPES).
    "sexe": dict(
        label="Sexe",
        detection_class_id="1",
        staging_dir=REPO_ROOT / "dataset_sexe",
        crop_prefix="canard",
        model_key="sexe",
        model_filename="modele_sexe.pt",
    ),
    # "detection" n'a ni staging_dir ni crop_prefix pertinents (ce n'est pas
    # une classification de crops mais une détection de boîtes -- gérée par
    # EcranDetection et 16_reentrainer_detection.py, pas par EcranConfig /
    # EcranEntrainement / EcranCorrections ci-dessus). Ajoutée à IA_DEFS
    # uniquement pour que EcranVersioning (label affiché, comparaison à la
    # version active, dialogue "enregistrer un modèle entraîné ailleurs")
    # fonctionne pour la détection comme pour les 3 IA de classification --
    # voir gestion_modeles.IA_CONFIG["detection"], qui gère déjà ce type de
    # modèle sans classes.json.
    "detection": dict(
        label="Détection (canards + marques nasales)",
        detection_class_id=None,
        staging_dir=REPO_ROOT / "dataset_detection",
        crop_prefix=None,
        model_key="detection",
        model_filename="best.pt",
    ),
}


def _charger_script_numerique(nom_fichier: str, nom_module: str):
    """Charge un script scripts/<nom_fichier> comme module Python, même si
    son nom commence par un chiffre (ex. '10_fusionner_corrections_dataset.py'),
    ce qu'un `import` classique ne permet pas. Même principe que
    charger_gestion_modeles() plus bas, généralisé pour être réutilisable."""
    import importlib.util as ilu
    script_path = Path(__file__).resolve().parent / nom_fichier
    spec = ilu.spec_from_file_location(nom_module, script_path)
    if spec and spec.loader:
        mod = ilu.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    return None


# ============================================================================
# Aide-communication thread de travail -> interface (identique au principe
# utilisé par interface_graphique.py)
# ============================================================================

class RedirectPrintToQueue:
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


# ============================================================================
# Extraction des nouveaux crops depuis raw_data/ (fonctions pures, réutilisées
# à la fois par couleur et espèce — même logique, seule la classe ciblée change)
# ============================================================================

def yolo_box_to_pixels(cx, cy, w, h, W, H, margin=MARGIN):
    x1 = (cx - w / 2) * W
    y1 = (cy - h / 2) * H
    x2 = (cx + w / 2) * W
    y2 = (cy + h / 2) * H
    mw, mh = (x2 - x1) * margin, (y2 - y1) * margin
    x1 = max(0, int(x1 - mw))
    y1 = max(0, int(y1 - mh))
    x2 = min(W, int(x2 + mw))
    y2 = min(H, int(y2 + mh))
    return x1, y1, x2, y2


def load_metadata(staging_dir: Path):
    meta_path = staging_dir / "metadata.json"
    if meta_path.exists():
        return json.loads(meta_path.read_text(encoding="utf-8"))
    return []


def save_metadata(staging_dir: Path, metadata):
    (staging_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


def existing_crop_names(staging_dir: Path):
    """Tous les noms de fichiers crops déjà présents quelque part sous
    staging_dir (dans crops/ comme dans d'éventuels dossiers déjà triés
    milouin/, morillon/, bleu/... issus d'un tri précédent ou d'une
    migration antérieure) -> évite de ré-extraire deux fois la même boîte."""
    names = set()
    if staging_dir.exists():
        for p in staging_dir.rglob("*.jpg"):
            names.add(p.name)
    return names


def extraire_nouveaux_crops(ia_key: str, log):
    """Scanne raw_data/labels/<job>/*.txt, découpe les boîtes de la classe
    ciblée (marque_nasale ou canard) pas encore extraites, les ajoute à
    staging_dir/crops/ et à metadata.json (champ correspondant à None,
    en attente de tri). Retourne (n_nouveaux, n_ignores_ancien_schema)."""
    cfg = IA_DEFS[ia_key]
    staging_dir = Path(cfg["staging_dir"])  # type: ignore[arg-type]
    crops_dir = staging_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    metadata = load_metadata(staging_dir)
    deja_connus = existing_crop_names(staging_dir)
    class_id = cfg["detection_class_id"]

    n_nouveaux = 0
    n_ancien_schema = 0
    n_nuit = 0

    if not RAW_LABELS.exists():
        log(f"⚠️  Dossier introuvable : {RAW_LABELS}")
        return 0, 0

    for job in list_job_dirs(RAW_IMAGES):
        img_job_dir = RAW_IMAGES / job if job else RAW_IMAGES
        lbl_job_dir = RAW_LABELS / job if job else RAW_LABELS
        if not lbl_job_dir.exists():
            continue
        for txt in sorted(lbl_job_dir.glob("*.txt")):
            lines = [
                ligne_txt.strip()
                for ligne_txt in txt.read_text(encoding="utf-8", errors="ignore").splitlines()
                if ligne_txt.strip()
            ]
            cibles = []
            for line in lines:
                parts = line.split()
                if not parts:
                    continue
                if parts[0] not in ("0", "1"):
                    n_ancien_schema += 1
                    continue
                if parts[0] == class_id:
                    cibles.append(parts)
            if not cibles:
                continue

            img_path = find_image(img_job_dir, txt.stem)
            if img_path is None:
                continue

            with Image.open(img_path) as im:
                W, H = im.size
                for idx, parts in enumerate(cibles):
                    prefix = f"{job}__" if job else ""
                    crop_name = f"{prefix}{txt.stem}_{cfg['crop_prefix']}_{idx:02d}.jpg"
                    if crop_name in deja_connus:
                        continue
                    cx, cy, w, h = (float(x) for x in parts[1:5])
                    x1, y1, x2, y2 = yolo_box_to_pixels(cx, cy, w, h, W, H)
                    crop = im.crop((x1, y1, x2, y2))
                    crop.save(crops_dir / crop_name, quality=95)
                    entry = {
                        "crop_name": crop_name,
                        "source_photo": f"{job}/{img_path.name}" if job else img_path.name,
                        "bbox": [x1, y1, x2, y2],
                        cfg_field(ia_key): None,
                    }
                    if ia_key == "couleur":
                        # Photo de nuit (flash IR) -> couleur peu fiable, voir
                        # SATURATION_SEUIL_NUIT. Calculé une fois à l'extraction
                        # et conservé dans metadata.json (pas recalculé au tri).
                        entry["nuit_probable"] = est_photo_de_nuit(crop)
                        if entry["nuit_probable"]:
                            n_nuit += 1
                    metadata.append(entry)
                    deja_connus.add(crop_name)
                    n_nouveaux += 1

    save_metadata(staging_dir, metadata)
    if n_ancien_schema:
        log(f"⚠️  {n_ancien_schema} boîte(s) ignorée(s) : ancien schéma d'annotation détecté "
            f"(pas 0/1). Lance d'abord scripts/00_migrate_annotations.py sur ces jobs.")
    if n_nuit:
        log(f"🌙 {n_nuit} crop(s) repéré(s) comme photo(s) de nuit probable(s) (flash IR, "
            f"couleur peu fiable) — mis de côté par défaut à l'écran de tri.")
    return n_nouveaux, n_ancien_schema


def cfg_field(ia_key):
    return "couleur" if ia_key == "couleur" else "espece"


def classes_connues(ia_key: str):
    """Union des noms de classe déjà utilisés : sous-dossiers déjà triés
    (legacy ou reconstruits) + valeurs déjà présentes dans metadata.json."""
    cfg = IA_DEFS[ia_key]
    staging_dir = Path(cfg["staging_dir"])  # type: ignore[arg-type]
    noms = set()
    if staging_dir.exists():
        for d in staging_dir.iterdir():
            if d.is_dir() and d.name not in ("crops", "pret"):
                noms.add(d.name)
    for entry in load_metadata(staging_dir):
        val = entry.get(cfg_field(ia_key))
        if val:
            noms.add(val)
    return sorted(noms)


def grouper_par_rafale(todo_entries, gap_seconds):
    """Regroupe des entrées metadata (crops en attente de tri) selon la
    rafale de leur photo source (même logique que le pipeline principal :
    écart max en secondes entre 2 photos consécutives, cf. BurstDetector).
    Une rafale = un seul clic de classe pour tous ses crops. Les photos sans
    timestamp EXIF (que BurstDetector ne peut pas grouper) deviennent chacune
    leur propre rafale à 1 photo, pour ne perdre aucun crop.
    Retourne une liste de listes d'entrées (une liste par rafale)."""
    par_job = {}
    for entry in todo_entries:
        source = entry["source_photo"]
        if "/" in source:
            job, filename = source.split("/", 1)
        else:
            job, filename = "", source
        par_job.setdefault(job, []).append((filename, entry))

    bd = BurstDetector()
    groupes = []
    for job, items in par_job.items():
        img_dir = RAW_IMAGES / job if job else RAW_IMAGES
        filenames = sorted({fn for fn, _ in items})
        chemins = [img_dir / fn for fn in filenames]
        rafales = bd.group_photos_into_bursts(chemins, max_gap_seconds=gap_seconds)

        photo_vers_cle = {}
        for i, rafale in enumerate(rafales):
            for p in rafale["photos"]:
                photo_vers_cle[Path(p).name] = ("rafale", i)
        # photos sans EXIF (BurstDetector les ignore) -> une rafale à elles seules
        suivant = len(rafales)
        for fn in filenames:
            if fn not in photo_vers_cle:
                photo_vers_cle[fn] = ("seule", suivant)
                suivant += 1

        par_cle = {}
        for fn, entry in items:
            par_cle.setdefault(photo_vers_cle[fn], []).append(entry)
        groupes.extend(par_cle.values())

    return groupes


def reconstruire_dataset_pret(ia_key: str):
    """(Re)construit staging_dir/pret/<classe>/*.jpg à partir : des dossiers
    déjà triés (legacy) + des crops de staging_dir/crops/ déjà étiquetés dans
    metadata.json. Toujours régénéré en entier (idempotent), donc rien à
    invalider manuellement. Retourne un dict {classe: nb_images}."""
    cfg = IA_DEFS[ia_key]
    staging_dir = Path(cfg["staging_dir"])  # type: ignore[arg-type]
    pret_dir = staging_dir / "pret"
    if pret_dir.exists():
        shutil.rmtree(pret_dir)
    pret_dir.mkdir(parents=True)

    stats: dict[str, int] = {}

    def copier(src: Path, classe: str):
        dest_dir = pret_dir / classe
        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest_dir / src.name)
        stats[classe] = stats.get(classe, 0) + 1

    if staging_dir.exists():
        for d in staging_dir.iterdir():
            if d.is_dir() and d.name not in ("crops", "pret"):
                for img in d.glob("*.jpg"):
                    copier(img, d.name)

    crops_dir = staging_dir / "crops"
    for entry in load_metadata(staging_dir):
        classe = entry.get(cfg_field(ia_key))
        if not classe:
            continue
        src = crops_dir / entry["crop_name"]
        if src.exists():
            copier(src, classe)

    return stats


# ============================================================================
# Écran 1 : configuration (qu'est-ce qui a changé ?)
# ============================================================================

class EcranConfig(tk.Frame):
    def __init__(self, master, app):
        super().__init__(master)
        self.app = app
        pad = {"padx": 14, "pady": 8}

        tk.Label(self, text="🦆 Préparation des données pour ré-entraînement",
                 font=("Helvetica", 13, "bold")).pack(anchor="w", **pad)
        tk.Label(self, justify="left", fg="#333333", text=(
            "Ce que l'interface va faire :\n"
            "• Chercher les nouveaux crops (couleur de marque + espèce) dans raw_data/\n"
            "• Te les faire trier à l'œil, classe par classe\n"
            "• Préparer les dossiers prêts pour l'entraînement (couleur ET espèce)"
        )).pack(anchor="w", **pad)

        info = tk.Label(self, justify="left", fg="#666666", text=(
            "Pas besoin de préciser si c'est une classe nouvelle ou existante : trie simplement\n"
            "les nouveaux crops, et tape le nom d'une classe qui n'existe pas encore quand tu la\n"
            "vois — elle sera créée automatiquement, sans toucher à aucun fichier de code."
        ))
        info.pack(anchor="w", **pad)

        cadre_rafale = ttk.LabelFrame(self, text="Tri manuel")
        cadre_rafale.pack(fill="x", **pad)
        tk.Label(cadre_rafale, text="Écart max entre 2 photos d'une même rafale (secondes) :").grid(
            row=0, column=0, sticky="w", padx=10, pady=6)
        self.var_burst_gap = tk.IntVar(value=self.app.burst_gap)
        ttk.Spinbox(cadre_rafale, from_=1, to=120, textvariable=self.var_burst_gap, width=6).grid(
            row=0, column=1, sticky="w", padx=(0, 10))
        tk.Label(cadre_rafale, fg="#666666", justify="left", text=(
            "Les photos d'une même rafale (même canard, plusieurs déclenchements rapprochés)\n"
            "seront affichées ensemble au tri : un seul clic classe toute la rafale."
        )).grid(row=1, column=0, columnspan=2, sticky="w", padx=10, pady=(0, 6))

        cadre_nuit = ttk.LabelFrame(self, text="Photos de nuit (couleur)")
        cadre_nuit.pack(fill="x", **pad)
        self.var_exclure_nuit = tk.BooleanVar(value=self.app.exclure_nuit_couleur)
        ttk.Checkbutton(cadre_nuit, variable=self.var_exclure_nuit,
                         text="Mettre de côté les photos de nuit (flash IR) pour la couleur — recommandé"
                         ).pack(anchor="w", padx=10, pady=4)
        tk.Label(cadre_nuit, fg="#666666", justify="left", text=(
            "La nuit, le flash infrarouge rend les images quasi grises : la couleur de la\n"
            "marque n'y est pas fiable. Détectées automatiquement, ces photos ne seront pas\n"
            "montrées au tri (donc jamais utilisées pour l'entraînement couleur), sauf si tu\n"
            "décoches cette case, ou si tu choisis de les trier quand même depuis l'écran de tri."
        )).pack(anchor="w", padx=10, pady=(0, 6))

        tk.Button(self, text="Suivant : chercher les nouveaux crops →",
                  bg="#4C8BF5", fg="white", font=("Helvetica", 11, "bold"),
                  command=self._suivant).pack(anchor="w", **pad)

        cadre_corrections = ttk.LabelFrame(self, text="Ou : réentraîner depuis les corrections")
        cadre_corrections.pack(fill="x", **pad)
        tk.Label(cadre_corrections, fg="#666666", justify="left", text=(
            "Si tu as corrigé des identifications dans l'écran de relecture\n"
            "(09_ecran_identification.py), ces corrections (espèce, sexe, couleur)\n"
            "peuvent directement réentraîner les modèles concernés, sans passer\n"
            "par le tri manuel ci-dessus."
        )).pack(anchor="w", padx=10, pady=(6, 4))
        tk.Button(cadre_corrections, text="🔁 Réentraîner depuis les corrections…",
                  command=self.app.afficher_corrections).pack(anchor="w", padx=10, pady=(0, 8))

        cadre_detection = ttk.LabelFrame(self, text="Ou : réentraîner la détection (YOLO)")
        cadre_detection.pack(fill="x", **pad)
        tk.Label(cadre_detection, fg="#666666", justify="left", text=(
            "Pour améliorer la détection elle-même (canards ratés, marques nasales\n"
            "manquées) plutôt qu'une classification -- à partir des ajouts manuels\n"
            "faits dans la fenêtre de correction (module_corrections/fenetre_rafale.py)."
        )).pack(anchor="w", padx=10, pady=(6, 4))
        tk.Button(cadre_detection, text="🎯 Réentraîner la détection…",
                  command=self.app.afficher_detection).pack(anchor="w", padx=10, pady=(0, 8))

    def _suivant(self):
        self.app.ias_a_traiter = ["couleur", "espece"]
        self.app.burst_gap = self.var_burst_gap.get()
        self.app.exclure_nuit_couleur = self.var_exclure_nuit.get()
        self.app.afficher_extraction()


# ============================================================================
# Écran 2 : extraction des nouveaux crops (journal + barre de progression)
# ============================================================================

class EcranExtraction(tk.Frame):
    def __init__(self, master, app):
        super().__init__(master)
        self.app = app
        pad = {"padx": 14, "pady": 8}

        tk.Label(self, text="🔎 Extraction des nouveaux crops depuis raw_data/",
                 font=("Helvetica", 13, "bold")).pack(anchor="w", **pad)

        self.texte = tk.Text(self, height=14, bg="#111111", fg="#d0d0d0", font=("Courier", 9), state="disabled")
        self.texte.pack(fill="both", expand=True, padx=14, pady=6)

        self.progress = ttk.Progressbar(self, mode="indeterminate")
        self.progress.pack(fill="x", **pad)

        self.btn_suivant = tk.Button(self, text="Suivant : trier les crops →", state="disabled",
                                      bg="#4C8BF5", fg="white", font=("Helvetica", 11, "bold"),
                                      command=self.app.afficher_tri_suivant)
        self.btn_suivant.pack(anchor="w", **pad)

        self.log_queue = queue.Queue()
        self.after(100, self._traiter_queue)
        self.after(200, self._lancer)

    def _log(self, message):
        self.log_queue.put(message)

    def _traiter_queue(self):
        try:
            while True:
                msg = self.log_queue.get_nowait()
                self.texte.config(state="normal")
                self.texte.insert("end", msg + "\n")
                self.texte.see("end")
                self.texte.config(state="disabled")
        except queue.Empty:
            pass
        self.after(100, self._traiter_queue)

    def _lancer(self):
        self.progress.start(12)
        thread = threading.Thread(target=self._travailler, daemon=True)
        thread.start()

    def _travailler(self):
        resultats = {}
        for ia_key in self.app.ias_a_traiter:
            label = IA_DEFS[ia_key]["label"]
            self._log(f"— {label} —")
            n_nouveaux, n_ignores = extraire_nouveaux_crops(ia_key, self._log)
            staging_dir = Path(IA_DEFS[ia_key]["staging_dir"])  # type: ignore[arg-type]
            todo_entries = [e for e in load_metadata(staging_dir)
                             if not e.get(cfg_field(ia_key))]
            n_total_todo = len(todo_entries)  # utilisé pour décider d'ouvrir l'écran de tri
            if ia_key == "couleur" and self.app.exclure_nuit_couleur:
                n_mis_de_cote = sum(1 for e in todo_entries if e.get("nuit_probable"))
                n_a_trier = n_total_todo - n_mis_de_cote
                self._log(f"  {n_nouveaux} nouveau(x) crop(s) extrait(s). "
                           f"{n_a_trier} crop(s) en attente de tri "
                           f"({n_mis_de_cote} de nuit mis de côté).")
            else:
                n_a_trier = n_total_todo
                self._log(f"  {n_nouveaux} nouveau(x) crop(s) extrait(s). "
                           f"{n_a_trier} crop(s) au total en attente de tri.")
            resultats[ia_key] = n_total_todo
        self._log("\n✅ Extraction terminée.")
        self.after(0, lambda: self._extraction_terminee(resultats))

    def _extraction_terminee(self, resultats):
        self.progress.stop()
        self.progress.config(mode="determinate", value=100, maximum=100)
        self.app.file_tri = [k for k in self.app.ias_a_traiter if resultats.get(k, 0) > 0]
        if not self.app.file_tri:
            self._log("Rien à trier : toutes les classes existantes ont déjà tous leurs crops étiquetés.")
        self.btn_suivant.config(state="normal")


# ============================================================================
# Écran 3 : tri manuel (un par IA sélectionnée, à la suite)
# ============================================================================

class EcranTri(tk.Frame):
    """Affiche une rafale à la fois (toutes les photos prises à quelques
    secondes d'écart -> presque toujours le même canard), avec un bouton par
    classe déjà connue et un champ pour en taper une nouvelle. Un seul clic
    classe TOUTE la rafale ; un bouton permet de la découper si elle mélange
    en fait plusieurs canards différents. Écrit dans metadata.json à chaque
    décision -> reprenable si on ferme l'interface en cours de route."""

    MAX_VIGNETTES_AFFICHEES = 8

    def __init__(self, master, app, ia_key):
        super().__init__(master)
        self.app = app
        self.ia_key = ia_key
        self.cfg = IA_DEFS[ia_key]
        self.staging_dir = Path(self.cfg["staging_dir"])  # type: ignore[arg-type]
        self.field = cfg_field(ia_key)
        self.metadata = load_metadata(self.staging_dir)
        todo = [e for e in self.metadata if not e.get(self.field)]

        # Photos de nuit (couleur uniquement) : mises de côté par défaut, pas
        # montrées au tri tant qu'on ne clique pas sur "les trier quand même"
        # -> elles restent non étiquetées et donc absentes de dataset_couleur/pret/.
        self.mettre_de_cote_nuit = (ia_key == "couleur" and self.app.exclure_nuit_couleur)
        if self.mettre_de_cote_nuit:
            todo_nuit = [e for e in todo if e.get("nuit_probable")]
            todo = [e for e in todo if not e.get("nuit_probable")]
            self.groupes_nuit = grouper_par_rafale(todo_nuit, gap_seconds=self.app.burst_gap)
        else:
            self.groupes_nuit = []

        self.n_total_crops = len(todo)
        self.groupes = grouper_par_rafale(todo, gap_seconds=self.app.burst_gap)
        self.index = 0
        self.historique = []  # pile des index déjà traités, pour "retour arrière"
        self.photos_courantes = []  # références fortes, sinon Tk garbage-collect les images

        pad = {"padx": 14, "pady": 8}
        tk.Label(self, text=f"🖼️ Tri manuel — {self.cfg['label']}",
                 font=("Helvetica", 13, "bold")).pack(anchor="w", **pad)

        self.lbl_compteur = tk.Label(self, fg="#666666")
        self.lbl_compteur.pack(anchor="w", padx=14)

        self.cadre_nuit_info = tk.Frame(self)
        if self.groupes_nuit:
            n_nuit = sum(len(g) for g in self.groupes_nuit)
            tk.Label(self.cadre_nuit_info, fg="#a06000",
                     text=f"🌙 {n_nuit} crop(s) de nuit mis de côté ({len(self.groupes_nuit)} rafale(s)) "
                          f"— couleur peu fiable.").pack(side="left")
            tk.Button(self.cadre_nuit_info, text="Les trier quand même",
                      command=self._inclure_rafales_nuit).pack(side="left", padx=8)
        self.cadre_nuit_info.pack(anchor="w", padx=14, pady=(0, 4))

        self.cadre_vignettes = tk.Frame(self, bg="#222222")
        self.cadre_vignettes.pack(padx=14, pady=10, fill="x")

        self.btn_decouper = tk.Button(self, text="✂️ Cette rafale mélange plusieurs canards différents "
                                                   "— la traiter photo par photo",
                                       command=self._decouper_rafale_courante)
        self.btn_decouper.pack(anchor="w", padx=14)

        self.cadre_boutons = ttk.LabelFrame(self, text="Classe (s'applique à toute la rafale affichée)")
        self.cadre_boutons.pack(fill="x", **pad)

        cadre_nouvelle = tk.Frame(self)
        cadre_nouvelle.pack(fill="x", padx=14, pady=(0, 6))
        tk.Label(cadre_nouvelle, text="Nouvelle classe (si elle n'existe pas encore) :").pack(side="left")
        self.var_nouvelle_classe = tk.StringVar()
        entree = tk.Entry(cadre_nouvelle, textvariable=self.var_nouvelle_classe, width=25)
        entree.pack(side="left", padx=6)
        entree.bind("<Return>", lambda e: self._ajouter_classe())
        tk.Button(cadre_nouvelle, text="Ajouter", command=self._ajouter_classe).pack(side="left")

        cadre_nav = tk.Frame(self)
        cadre_nav.pack(fill="x", padx=14, pady=8)
        tk.Button(cadre_nav, text="⏭ Passer (revoir plus tard)", command=self._passer).pack(side="left")
        tk.Button(cadre_nav, text="⏮ Retour arrière", command=self._retour).pack(side="left", padx=8)
        self.lbl_avancement = tk.Label(cadre_nav, fg="#666666")
        self.lbl_avancement.pack(side="right")

        self.classes = list(classes_connues(ia_key))
        self._reconstruire_boutons()
        self._afficher_courant()

    # -- gestion des boutons de classe (dynamique : peut grandir en cours de tri) --

    def _reconstruire_boutons(self):
        for w in self.cadre_boutons.winfo_children():
            w.destroy()
        for i, classe in enumerate(self.classes):
            raccourci = f" ({i + 1})" if i < 9 else ""
            b = tk.Button(self.cadre_boutons, text=f"{classe}{raccourci}", width=16,
                          command=lambda c=classe: self._choisir(c))
            b.grid(row=i // 4, column=i % 4, padx=6, pady=6, sticky="w")
            if i < 9:
                self.bind_all(str(i + 1), lambda e, c=classe: self._choisir(c))

    def _ajouter_classe(self):
        nom = self.var_nouvelle_classe.get().strip().lower().replace(" ", "_")
        if not nom:
            return
        if nom not in self.classes:
            self.classes.append(nom)
            self.classes.sort()
            self._reconstruire_boutons()
        self.var_nouvelle_classe.set("")
        self._choisir(nom)

    # -- découpage d'une rafale mixte --

    def _etat_nuit_seule_restante(self):
        """Plus de rafale 'normale' à trier, mais des rafales de nuit sont
        encore mises de côté (bandeau du haut) : on ne saute pas directement
        à l'écran de fin (qui efface tout, y compris ce bandeau)."""
        n_nuit = sum(len(g) for g in self.groupes_nuit)
        self.lbl_compteur.config(text=f"Tout est trié à part {n_nuit} crop(s) de nuit mis de côté "
                                       f"(voir ci-dessus).")
        self.lbl_avancement.config(text="")
        self.btn_decouper.config(state="disabled")
        for w in self.cadre_vignettes.winfo_children():
            w.destroy()
        tk.Label(self.cadre_vignettes, bg="#222222", fg="white", justify="left", text=(
            "Clique \"Les trier quand même\" ci-dessus pour les revoir,\n"
            "ou continue directement sans les trier."
        )).pack(padx=20, pady=20)
        if not hasattr(self, "_btn_continuer_sans_nuit"):
            self._btn_continuer_sans_nuit = tk.Button(
                self, text="Continuer sans trier les photos de nuit →",
                bg="#4C8BF5", fg="white", font=("Helvetica", 11, "bold"), command=self._terminer)
            self._btn_continuer_sans_nuit.pack(anchor="w", padx=14, pady=8)

    def _inclure_rafales_nuit(self):
        """Réinjecte les rafales de nuit mises de côté dans la file à trier
        (à la suite de la position courante), et fait disparaître le bandeau."""
        self.groupes[self.index:self.index] = self.groupes_nuit
        self.groupes_nuit = []
        for w in self.cadre_nuit_info.winfo_children():
            w.destroy()
        if hasattr(self, "_btn_continuer_sans_nuit"):
            self._btn_continuer_sans_nuit.destroy()
            del self._btn_continuer_sans_nuit
        self._afficher_courant()

    def _decouper_rafale_courante(self):
        groupe = self.groupes[self.index]
        if len(groupe) <= 1:
            return
        self.groupes[self.index:self.index + 1] = [[e] for e in groupe]
        self._afficher_courant()

    # -- navigation --

    def _afficher_courant(self):
        if self.index >= len(self.groupes):
            if self.groupes_nuit:
                self._etat_nuit_seule_restante()
                return
            self._terminer()
            return
        groupe = self.groupes[self.index]
        n_rafales_restantes = len(self.groupes) - self.index
        self.lbl_compteur.config(
            text=f"Rafale {self.index + 1}/{len(self.groupes)}  —  {len(groupe)} photo(s) dans cette rafale  "
                 f"({n_rafales_restantes} rafale(s) restante(s))")
        self.lbl_avancement.config(text=f"{self.index}/{len(self.groupes)} rafale(s) déjà traitée(s) cette session")
        self.btn_decouper.config(state="normal" if len(groupe) > 1 else "disabled")

        for w in self.cadre_vignettes.winfo_children():
            w.destroy()
        self.photos_courantes = []

        a_afficher = groupe[:self.MAX_VIGNETTES_AFFICHEES]
        taille_vignette = 380 if len(groupe) == 1 else 160
        for entry in a_afficher:
            crop_path = self.staging_dir / "crops" / entry["crop_name"]
            cellule = tk.Frame(self.cadre_vignettes, bg="#222222")
            cellule.pack(side="left", padx=4, pady=4)
            if entry.get("nuit_probable"):
                tk.Label(cellule, text="🌙 nuit probable", bg="#222222", fg="#e0a030",
                         font=("Helvetica", 8)).pack(anchor="w")
            lbl = tk.Label(cellule, bg="#222222")
            lbl.pack()
            if not crop_path.exists():
                lbl.config(text="⚠️ manquant", fg="white")
                continue
            img = Image.open(crop_path)
            img.thumbnail((taille_vignette, taille_vignette))
            photo = ImageTk.PhotoImage(img)
            self.photos_courantes.append(photo)  # garde une référence forte
            lbl.config(image=photo)
        if len(groupe) > self.MAX_VIGNETTES_AFFICHEES:
            tk.Label(self.cadre_vignettes, text=f"+ {len(groupe) - self.MAX_VIGNETTES_AFFICHEES} autre(s)",
                     bg="#222222", fg="white").pack(side="left", padx=8)

    def _choisir(self, classe):
        groupe = self.groupes[self.index]
        for entry in groupe:
            entry[self.field] = classe
        save_metadata(self.staging_dir, self.metadata)
        self.historique.append(self.index)
        self.index += 1
        self._afficher_courant()

    def _passer(self):
        self.historique.append(self.index)
        self.index += 1
        self._afficher_courant()

    def _retour(self):
        if not self.historique:
            return
        self.index = self.historique.pop()
        self._afficher_courant()

    def _terminer(self):
        for w in self.winfo_children():
            w.destroy()
        pad = {"padx": 14, "pady": 8}
        tk.Label(self, text=f"✅ Tri terminé — {self.cfg['label']}",
                 font=("Helvetica", 13, "bold")).pack(anchor="w", **pad)
        restants = sum(1 for e in self.metadata if not e.get(self.field))
        if restants:
            texte_restants = (
                f"{restants} crop(s) laissé(s) de côté (passés) — "
                "tu pourras les retrier plus tard en relançant l'interface."
            )
            tk.Label(self, text=texte_restants).pack(anchor="w", padx=14)
        tk.Button(self, text="Continuer →", bg="#4C8BF5", fg="white", font=("Helvetica", 11, "bold"),
                  command=self.app.afficher_tri_suivant).pack(anchor="w", **pad)


# ============================================================================
# Écran 4 : entraînement (ou export pour Colab)
# ============================================================================

def torch_disponible():
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


def detecter_device():
    import torch
    if torch.cuda.is_available():
        return "cuda", torch.cuda.get_device_name(0)
    return "cpu", None


def estimer_duree_secondes(n_images, epochs, device):
    vitesse_img_par_s = 180 if device == "cuda" else 12  # estimation grossière
    return max(5, int(n_images * epochs / vitesse_img_par_s))


class EcranEntrainement(tk.Frame):
    def __init__(self, master, app):
        super().__init__(master)
        self.app = app
        pad = {"padx": 14, "pady": 8}

        tk.Label(self, text="🎓 Entraînement", font=("Helvetica", 13, "bold")).pack(anchor="w", **pad)

        self.stats_par_ia = {}
        for ia_key in self.app.ias_a_traiter:
            stats = reconstruire_dataset_pret(ia_key)
            self.stats_par_ia[ia_key] = stats
            cadre = ttk.LabelFrame(self, text=IA_DEFS[ia_key]["label"])
            cadre.pack(fill="x", **pad)
            if stats:
                texte = "  /  ".join(f"{c} : {n}" for c, n in sorted(stats.items()))
            else:
                texte = "Aucune image triée pour l'instant."
            tk.Label(cadre, text=texte, justify="left").pack(anchor="w", padx=10, pady=6)

        cadre_device = ttk.LabelFrame(self, text="Ordinateur utilisé")
        cadre_device.pack(fill="x", **pad)
        self.lbl_device = tk.Label(cadre_device, justify="left", fg="#333333")
        self.lbl_device.pack(anchor="w", padx=10, pady=6)
        self._detecter_et_afficher_device()

        cadre_params = ttk.LabelFrame(self, text="Paramètres d'entraînement")
        cadre_params.pack(fill="x", **pad)
        tk.Label(cadre_params, text="Époques :").grid(row=0, column=0, sticky="w", padx=10, pady=6)
        self.var_epochs = tk.IntVar(value=10)
        ttk.Spinbox(cadre_params, from_=1, to=100, textvariable=self.var_epochs, width=6).grid(
            row=0, column=1, sticky="w")
        tk.Label(cadre_params, text="Taille de batch :").grid(row=0, column=2, sticky="w", padx=10)
        self.var_batch = tk.IntVar(value=32)
        ttk.Spinbox(cadre_params, from_=4, to=128, textvariable=self.var_batch, width=6).grid(
            row=0, column=3, sticky="w")

        cadre_boutons = tk.Frame(self)
        cadre_boutons.pack(fill="x", **pad)
        self.btn_entrainer = tk.Button(
            cadre_boutons, text="🚀 Entraîner ici maintenant", bg="#4C8BF5", fg="white",
            font=("Helvetica", 11, "bold"), command=self._entrainer_ici)
        self.btn_entrainer.pack(side="left")
        tk.Button(cadre_boutons, text="📦 Préparer un zip pour Colab (pas d'entraînement ici)",
                  command=self._exporter_pour_colab).pack(side="left", padx=10)

        self.progress = ttk.Progressbar(self, mode="determinate")
        self.progress.pack(fill="x", **pad)
        self.texte = tk.Text(self, height=10, bg="#111111", fg="#d0d0d0", font=("Courier", 9), state="disabled")
        self.texte.pack(fill="both", expand=True, padx=14, pady=6)

        self.log_queue = queue.Queue()
        self.after(100, self._traiter_queue)

        if not any(self.stats_par_ia.values()):
            self.btn_entrainer.config(state="disabled")

    def _detecter_et_afficher_device(self):
        if not torch_disponible():
            self.lbl_device.config(
                text="PyTorch n'est pas installé sur cet ordinateur : impossible d'entraîner ici.\n"
                     "Utilise \"Préparer un zip pour Colab\" ci-dessous.")
            self.btn_entrainer.config(state="disabled")
            return
        device, nom_gpu = detecter_device()
        if device == "cuda":
            self.device = "cuda"
            self.lbl_device.config(text=f"✅ Carte graphique détectée : {nom_gpu}. L'entraînement sera rapide.")
        else:
            self.device = "cpu"
            self.lbl_device.config(
                text="⚠️ Pas de carte graphique compatible détectée : l'entraînement se fera sur le "
                     "processeur, ce qui peut être long (dizaines de minutes à quelques heures selon\n"
                     "le nombre d'images). Si c'est trop long, utilise plutôt \"Préparer un zip pour Colab\".")

    def _log(self, msg):
        self.log_queue.put(msg)

    def _traiter_queue(self):
        try:
            while True:
                msg = self.log_queue.get_nowait()
                self.texte.config(state="normal")
                self.texte.insert("end", msg + "\n")
                self.texte.see("end")
                self.texte.config(state="disabled")
        except queue.Empty:
            pass
        self.after(150, self._traiter_queue)

    # -- export pour Colab --

    def _exporter_pour_colab(self):
        dest = filedialog.asksaveasfilename(
            title="Enregistrer le zip pour Colab", defaultextension=".zip",
            initialfile="dataset_pret.zip", filetypes=[("Archive zip", "*.zip")])
        if not dest:
            return
        with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as zf:
            for ia_key in self.app.ias_a_traiter:
                pret_dir = Path(IA_DEFS[ia_key]["staging_dir"]) / "pret"  # type: ignore[arg-type]
                if not pret_dir.exists():
                    continue
                for f in pret_dir.rglob("*.jpg"):
                    zf.write(f, arcname=f"{ia_key}/{f.relative_to(pret_dir)}")
        messagebox.showinfo(
            "Zip prêt",
            f"Dataset(s) exporté(s) dans :\n{dest}\n\n"
            "Uploade ce zip à l'étape correspondante du notebook Colab. Une fois le modèle "
            "téléchargé (modele_xxx.pt + classes_xxx.json), relance cette interface : elle te "
            "proposera de l'enregistrer comme nouvelle version dès l'écran suivant, ou lance "
            "directement scripts/gestion_modeles.py register.")

    # -- entraînement local --

    def _entrainer_ici(self):
        n_total = sum(sum(s.values()) for s in self.stats_par_ia.values())
        est = estimer_duree_secondes(n_total, self.var_epochs.get(), getattr(self, "device", "cpu"))
        minutes = max(1, est // 60)
        if not messagebox.askyesno(
                "Confirmer",
                f"Estimation (très grossière) : environ {minutes} min pour {n_total} images "
                f"au total sur {'GPU' if getattr(self, 'device', 'cpu') == 'cuda' else 'CPU'}.\n\n"
                "Lancer l'entraînement maintenant ?"):
            return
        self.btn_entrainer.config(state="disabled")
        thread = threading.Thread(target=self._travailler_entrainement, daemon=True)
        thread.start()

    def _travailler_entrainement(self):
        resultats = {}
        for ia_key in self.app.ias_a_traiter:
            stats = self.stats_par_ia.get(ia_key) or {}
            if not stats or len(stats) < 2:
                self._log(f"— {IA_DEFS[ia_key]['label']} : pas assez de classes différentes, ignoré.")
                continue
            self._log(f"— Entraînement {IA_DEFS[ia_key]['label']} —")
            try:
                pret_dir = Path(IA_DEFS[ia_key]["staging_dir"]) / "pret"  # type: ignore[arg-type]
                model_path, classes_path, metriques = entrainer_classification(
                    pret_dir=pret_dir,
                    epochs=self.var_epochs.get(),
                    batch_size=self.var_batch.get(),
                    device_name=getattr(self, "device", "cpu"),
                    log=self._log,
                )
                resultats[ia_key] = {
                    "model_path": model_path, "classes_path": classes_path, "metriques": metriques,
                    "dataset_note": ", ".join(f"{c}:{n}" for c, n in sorted(stats.items())),
                }
                self._log(f"  ✅ terminé — précision validation : {metriques.get('val_acc', 0):.1%}")
            except Exception as e:  # noqa: BLE001
                logger.exception("Erreur pendant l'entraînement")
                self._log(f"  ❌ échec : {e}")
        self.after(0, lambda: self.app.afficher_versioning(resultats))


def entrainer_classification(pret_dir: Path, epochs: int, batch_size: int, device_name: str, log):
    """Entraîne un ResNet18 (transfer learning) sur pret_dir (ImageFolder),
    identique dans l'esprit à scripts/train_classification_ia.py, mais avec
    un callback de log par époque et des métriques de retour (pour
    l'enregistrement automatique dans gestion_modeles.py)."""
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from sklearn.metrics import accuracy_score
    from torch.utils.data import DataLoader
    from torchvision import datasets, models, transforms

    device = torch.device(device_name)

    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    dataset = datasets.ImageFolder(str(pret_dir), transform=data_transforms)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, len(dataset.classes))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    val_acc, val_loss = 0.0, 0.0
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        train_loss = running_loss / len(train_loader.dataset)  # type: ignore[arg-type]

        model.eval()
        val_running_loss = 0.0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item() * inputs.size(0)
                preds = outputs.argmax(1)
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())
        val_loss = val_running_loss / max(1, len(val_loader.dataset))  # type: ignore[arg-type]
        val_acc = accuracy_score(all_labels, all_preds) if all_labels else 0.0
        log(
            f"  Époque {epoch + 1}/{epochs} — "
            f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  val_acc={val_acc:.1%}"
        )

    tmp_dir = pret_dir.parent / "_tmp_entrainement"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    model_path = tmp_dir / "modele.pt"
    classes_path = tmp_dir / "classes.json"
    torch.save(model.state_dict(), model_path)
    classes_path.write_text(json.dumps(dataset.classes, ensure_ascii=False, indent=2), encoding="utf-8")

    return model_path, classes_path, {"val_acc": float(val_acc), "val_loss": float(val_loss)}


# ============================================================================
# Écran 5 : enregistrement de version (gestion_modeles.py) + promotion
# ============================================================================

def charger_gestion_modeles():
    import importlib.util as ilu
    script_path = Path(__file__).resolve().parent / "gestion_modeles.py"
    spec = ilu.spec_from_file_location("gestion_modeles", script_path)
    if spec and spec.loader:
        mod = ilu.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    return None


class EcranVersioning(tk.Frame):
    def __init__(self, master, app, resultats):
        super().__init__(master)
        self.app = app
        self.resultats = resultats
        self.gm = charger_gestion_modeles()
        pad = {"padx": 14, "pady": 8}

        tk.Label(self, text="💾 Enregistrement des versions", font=("Helvetica", 13, "bold")).pack(anchor="w", **pad)

        if not resultats:
            tk.Label(self, text="Aucun modèle entraîné ici. Utilise \"Enregistrer un modèle entraîné "
                                 "ailleurs (Colab)\" ci-dessous une fois le fichier téléchargé.",
                     justify="left").pack(anchor="w", padx=14)

        self.blocs = {}
        for ia_key, info in resultats.items():
            cadre = ttk.LabelFrame(self, text=IA_DEFS[ia_key]["label"])
            cadre.pack(fill="x", **pad)
            m = info["metriques"]
            tk.Label(cadre, text=f"val_acc={m.get('val_acc', 0):.1%}  val_loss={m.get('val_loss', 0):.3f}",
                     justify="left").pack(anchor="w", padx=10, pady=(6, 0))
            tk.Label(cadre, text="Notes (contexte, ex. \"ajout foulque_macroule, 40 crops\") :").pack(
                anchor="w", padx=10, pady=(6, 0))
            var_notes = tk.StringVar()
            tk.Entry(cadre, textvariable=var_notes, width=70).pack(anchor="w", padx=10, pady=(0, 6))

            cadre_boutons = tk.Frame(cadre)
            cadre_boutons.pack(anchor="w", padx=10, pady=(0, 8))
            btn_enr = tk.Button(cadre_boutons, text="Enregistrer comme nouvelle version",
                                 command=lambda k=ia_key, v=var_notes: self._enregistrer(k, v, promouvoir=False))
            btn_enr.pack(side="left")
            btn_enr_prom = tk.Button(cadre_boutons, text="Enregistrer + promouvoir (utiliser tout de suite)",
                                      bg="#4C8BF5", fg="white",
                                      command=lambda k=ia_key, v=var_notes: self._enregistrer(k, v, promouvoir=True))
            btn_enr_prom.pack(side="left", padx=8)
            self.blocs[ia_key] = {"cadre": cadre, "var_notes": var_notes}

        cadre_externe = ttk.LabelFrame(self, text="Enregistrer un modèle entraîné ailleurs (Colab)")
        cadre_externe.pack(fill="x", **pad)
        tk.Button(cadre_externe, text="Choisir un fichier .pt et l'enregistrer…",
                  command=self._enregistrer_modele_externe).pack(anchor="w", padx=10, pady=8)

        tk.Button(self, text="Terminer", command=self.app.destroy).pack(anchor="w", **pad)

    def _valider_avant_promotion(self, ia_key, model_path):
        """Tente une validation objective avant promotion. Retourne
        (peut_promouvoir: bool, message: str). Deux branches :
        - "detection" : comparaison mAP50 via 16_reentrainer_detection.py
          sur le dataset détection courant (voir _valider_detection).
        - couleur/espèce/sexe : accuracy via 12_valider_et_deployer.py sur
          dataset_<ia>_test/ (inchangé, voir _valider_classification).
        Ne bloque JAMAIS la promotion si aucun jeu de comparaison n'est
        disponible -- dans ce cas on prévient clairement l'utilisateur
        plutôt que de le laisser croire qu'une validation a eu lieu."""
        if ia_key == "detection":
            return self._valider_detection(model_path)
        return self._valider_classification(ia_key, model_path)

    def _valider_detection(self, model_path):
        data_yaml = REPO_ROOT / "dataset_detection" / "data.yaml"
        seuil_min = 0.5
        if not data_yaml.exists():
            ok = messagebox.askyesno(
                "Aucun dataset disponible",
                f"Pas de {data_yaml} : impossible de comparer objectivement ce modèle "
                "de détection au modèle actuel.\n\nPromouvoir quand même, sans comparaison ?")
            return ok, "aucun dataset détection disponible pour comparer"

        mod16 = _charger_script_numerique("16_reentrainer_detection.py", "reentrainement_detection_16")
        if mod16 is None:
            ok = messagebox.askyesno(
                "Comparaison impossible",
                "Impossible de charger 16_reentrainer_detection.py pour comparer "
                "ce modèle.\n\nPromouvoir quand même, sans comparaison ?")
            return ok, "script de comparaison non chargeable"

        try:
            m_nouveau = mod16.evaluer_detection(Path(model_path), data_yaml)
            m_actuel = mod16.evaluer_detection(mod16.MODEL_DETECTION_ACTUEL, data_yaml)
        except Exception as e:  # noqa: BLE001
            ok = messagebox.askyesno(
                "Échec de la comparaison",
                f"L'évaluation sur {data_yaml.parent.name}/ a échoué : {e}\n\n"
                "Promouvoir quand même, sans comparaison ?")
            return ok, "échec de l'évaluation"

        map_nouveau, map_actuel = m_nouveau["map50"], m_actuel["map50"]
        if map_nouveau >= map_actuel and map_nouveau >= seuil_min:
            return True, f"validé : mAP50={map_nouveau:.1%} (actuel : {map_actuel:.1%})"

        ok = messagebox.askyesno(
            "Comparaison non concluante",
            f"Nouveau modèle : mAP50={map_nouveau:.1%}\n"
            f"Modèle actuel : mAP50={map_actuel:.1%}\n"
            f"Seuil minimal : {seuil_min:.0%}\n\n"
            "Ce nouveau modèle ne dépasse pas le modèle actuel (ou n'atteint pas le "
            "seuil minimal).\n\nLe promouvoir quand même ?")
        return ok, f"comparaison non concluante (mAP50={map_nouveau:.1%} vs seuil {seuil_min:.0%}) — {'forcé' if ok else 'annulé'} par l'utilisateur"

    def _valider_classification(self, ia_key, model_path):
        test_dir = REPO_ROOT / f"dataset_{ia_key}_test"
        seuil_min = 0.85
        if not test_dir.exists():
            ok = messagebox.askyesno(
                "Aucun jeu de test disponible",
                f"Pas de dossier de test fixe pour '{IA_DEFS[ia_key]['label']}' "
                f"({test_dir.name}/) : impossible de valider objectivement ce "
                "modèle avant de le promouvoir.\n\n"
                "Promouvoir quand même, sans validation ?")
            return ok, "aucun jeu de test disponible pour ce type"

        mod12 = _charger_script_numerique("12_valider_et_deployer.py", "validation_12")
        if mod12 is None:
            ok = messagebox.askyesno(
                "Validation impossible",
                "Impossible de charger 12_valider_et_deployer.py pour valider "
                "ce modèle.\n\nPromouvoir quand même, sans validation ?")
            return ok, "script de validation non chargeable"

        try:
            acc_nouveau = mod12.evaluer_modele(Path(model_path), test_dir)
            modele_actuel = MODELS_DIR / ia_key / IA_DEFS[ia_key]["model_filename"]
            acc_actuel = mod12.evaluer_modele(modele_actuel, test_dir) if modele_actuel.exists() else 0.0
        except Exception as e:  # noqa: BLE001
            ok = messagebox.askyesno(
                "Échec de la validation",
                f"L'évaluation sur {test_dir.name}/ a échoué : {e}\n\n"
                "Promouvoir quand même, sans validation ?")
            return ok, "échec de l'évaluation"

        if acc_nouveau >= acc_actuel and acc_nouveau >= seuil_min:
            return True, f"validé : {acc_nouveau:.1%} sur {test_dir.name}/ (actuel : {acc_actuel:.1%})"

        ok = messagebox.askyesno(
            "Validation non concluante",
            f"Nouveau modèle : {acc_nouveau:.1%} sur {test_dir.name}/\n"
            f"Modèle actuel : {acc_actuel:.1%}\n"
            f"Seuil minimal : {seuil_min:.0%}\n\n"
            "Ce nouveau modèle ne dépasse pas le modèle actuel (ou n'atteint "
            "pas le seuil minimal). Le déploiement automatique de "
            "12_valider_et_deployer.py refuserait de le déployer.\n\n"
            "Le promouvoir quand même ?")
        return ok, f"validation non concluante ({acc_nouveau:.1%} vs seuil {seuil_min:.0%}) — {'forcé' if ok else 'annulé'} par l'utilisateur"


    def _enregistrer(self, ia_key, var_notes, promouvoir):
        info = self.resultats[ia_key]
        try:
            version_name = self._register(
                ia_key, info["model_path"], info["classes_path"],
                notes=var_notes.get(), dataset_note=info.get("dataset_note", ""),
                metriques=info["metriques"])
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("Erreur", f"Échec de l'enregistrement : {e}")
            return
        if promouvoir:
            peut_promouvoir, note_validation = self._valider_avant_promotion(ia_key, info["model_path"])
            if not peut_promouvoir:
                messagebox.showinfo(
                    "Enregistré, non promu",
                    f"Version {version_name} enregistrée pour '{ia_key}' mais PAS promue "
                    f"({note_validation}). Le modèle actuellement en production reste actif.")
                return
            if self.gm:
                self.gm.cmd_promote(SimpleNamespace(ia=ia_key, version=version_name, latest=False))
            messagebox.showinfo("Enregistré", f"Version {version_name} enregistrée pour '{ia_key}' "
                                               f"et promue comme modèle actif ({note_validation}).")
        else:
            messagebox.showinfo("Enregistré", f"Version {version_name} enregistrée pour '{ia_key}'.")

    def _register(self, ia_key, model_path, classes_path, notes, dataset_note, metriques):
        """Reproduit cmd_register de gestion_modeles.py mais retourne le nom
        de version créé (la fonction CLI ne fait qu'imprimer)."""
        if not self.gm:
            raise RuntimeError("Impossible de charger gestion_modeles.py")
        gm = self.gm
        cfg = gm.IA_CONFIG[ia_key]
        version_name = gm.next_version_name(ia_key)
        vdir = gm.versions_dir(ia_key) / version_name
        vdir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(model_path, vdir / cfg["model_file"])
        if cfg["classes_file"] and classes_path:
            shutil.copy2(classes_path, vdir / cfg["classes_file"])
            classes_json_path = (vdir / cfg["model_file"]).with_suffix(".classes.json")
            if classes_json_path != vdir / cfg["classes_file"]:
                shutil.copy2(classes_path, classes_json_path)
        metrics = dict(metriques)
        metrics["date"] = date.today().isoformat()
        metrics["notes"] = notes or ""
        metrics["dataset"] = dataset_note or ""
        metrics["source_model_file"] = Path(model_path).name
        (vdir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        return version_name

    def _enregistrer_modele_externe(self):
        ia_key = self._choisir_ia_dialogue()
        if ia_key is None:
            return
        model_path = filedialog.askopenfilename(title="Fichier modèle (.pt)", filetypes=[("PyTorch", "*.pt *.pth")])
        if not model_path:
            return
        classes_path = filedialog.askopenfilename(title="Fichier classes.json", filetypes=[("JSON", "*.json")])
        if not classes_path:
            return
        try:
            version_name = self._register(ia_key, model_path, classes_path,
                                           notes="Entraîné sur Colab", dataset_note="", metriques={})
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("Erreur", f"Échec de l'enregistrement : {e}")
            return
        if messagebox.askyesno("Promouvoir ?", f"Version {version_name} enregistrée pour '{ia_key}'.\n\n"
                                                f"La promouvoir tout de suite comme modèle actif ?"):
            peut_promouvoir, note_validation = self._valider_avant_promotion(ia_key, model_path)
            if peut_promouvoir and self.gm:
                self.gm.cmd_promote(SimpleNamespace(ia=ia_key, version=version_name, latest=False))
                messagebox.showinfo("Terminé", f"Version {version_name} enregistrée pour '{ia_key}' "
                                                f"et promue ({note_validation}).")
                return
            messagebox.showinfo("Terminé", f"Version {version_name} enregistrée pour '{ia_key}', "
                                            f"non promue ({note_validation}).")
            return
        messagebox.showinfo("Terminé", f"Version {version_name} enregistrée pour '{ia_key}'.")

    def _choisir_ia_dialogue(self):
        fen = tk.Toplevel(self)
        fen.title("Quelle IA ?")
        choix = {"valeur": None}
        tk.Label(fen, text="Pour quelle IA ce modèle a-t-il été entraîné ?").pack(padx=14, pady=10)
        for ia_key in IA_DEFS:
            tk.Button(fen, text=IA_DEFS[ia_key]["label"],
                      command=lambda k=ia_key: (choix.update(valeur=k), fen.destroy())).pack(pady=4)
        fen.grab_set()
        self.wait_window(fen)
        return choix["valeur"]


# ============================================================================
# Écran alternatif : réentraînement de la DÉTECTION (YOLO -- canards +
# marques nasales), à partir du dataset exporté par
# scripts/13_exporter_annotations_detection.py. Contrairement aux 3 IA de
# classification ci-dessus (couleur/espèce/sexe), il n'y avait jusqu'ici
# aucun moyen de lancer cet entraînement sans sortir de l'interface -- voir
# scripts/16_reentrainer_detection.py, qui fait le travail réel et que cet
# écran ne fait qu'appeler et journaliser.
# ============================================================================

class EcranDetection(tk.Frame):
    def __init__(self, master, app):
        super().__init__(master)
        self.app = app
        pad = {"padx": 14, "pady": 8}

        tk.Label(self, text="🎯 Réentraînement de la détection (YOLO)",
                 font=("Helvetica", 13, "bold")).pack(anchor="w", **pad)
        tk.Label(self, justify="left", fg="#333333", text=(
            "Combine les détections déjà trouvées par le pipeline et les ajouts\n"
            "manuels de canards ratés (fenêtre de correction), régénère le dataset\n"
            "YOLO, puis réentraîne -- ici ou via Colab -- à partir du modèle actuel.\n"
            "Une comparaison avec le modèle actuellement en production est proposée\n"
            "avant toute promotion, comme pour les modèles espèce/sexe/couleur."
        )).pack(anchor="w", **pad)

        cadre_dataset = ttk.LabelFrame(self, text="Dataset YOLO (images + labels + data.yaml)")
        cadre_dataset.pack(fill="x", **pad)
        self.var_data_yaml = tk.StringVar(
            value=str(REPO_ROOT / "dataset_detection" / "data.yaml"))
        ligne = tk.Frame(cadre_dataset)
        ligne.pack(fill="x", padx=10, pady=6)
        tk.Entry(ligne, textvariable=self.var_data_yaml, width=64).pack(side="left")
        tk.Button(ligne, text="Parcourir…", command=self._choisir_data_yaml).pack(side="left", padx=6)
        tk.Button(cadre_dataset, text="🔄 Régénérer le dataset depuis output/ courant",
                  command=self._regenerer_dataset).pack(anchor="w", padx=10, pady=(0, 8))
        self.lbl_dataset = tk.Label(cadre_dataset, fg="#666666", justify="left", text="")
        self.lbl_dataset.pack(anchor="w", padx=10, pady=(0, 6))
        self._rafraichir_etat_dataset()

        cadre_device = ttk.LabelFrame(self, text="Ordinateur utilisé")
        cadre_device.pack(fill="x", **pad)
        self.lbl_device = tk.Label(cadre_device, justify="left", fg="#333333")
        self.lbl_device.pack(anchor="w", padx=10, pady=6)
        self._detecter_et_afficher_device()

        cadre_params = ttk.LabelFrame(self, text="Paramètres d'entraînement")
        cadre_params.pack(fill="x", **pad)
        tk.Label(cadre_params, text="Époques :").grid(row=0, column=0, sticky="w", padx=10, pady=6)
        self.var_epochs = tk.IntVar(value=50)
        ttk.Spinbox(cadre_params, from_=1, to=500, textvariable=self.var_epochs, width=6).grid(
            row=0, column=1, sticky="w")
        tk.Label(cadre_params, text="Taille d'image :").grid(row=0, column=2, sticky="w", padx=(20, 10))
        self.var_imgsz = tk.IntVar(value=1280)
        ttk.Spinbox(cadre_params, from_=320, to=1920, increment=32, textvariable=self.var_imgsz, width=6).grid(
            row=0, column=3, sticky="w")
        tk.Label(cadre_params, text="Taille de batch :").grid(row=1, column=0, sticky="w", padx=10, pady=(0, 6))
        self.var_batch = tk.IntVar(value=8)
        ttk.Spinbox(cadre_params, from_=1, to=64, textvariable=self.var_batch, width=6).grid(
            row=1, column=1, sticky="w", pady=(0, 6))

        cadre_boutons = tk.Frame(self)
        cadre_boutons.pack(fill="x", **pad)
        tk.Button(cadre_boutons, text="← Retour", command=self.app.afficher_config).pack(side="left")
        self.btn_entrainer_ici = tk.Button(
            cadre_boutons, text="🚀 Entraîner ici", bg="#4C8BF5", fg="white",
            font=("Helvetica", 11, "bold"), command=self._lancer_ici)
        self.btn_entrainer_ici.pack(side="left", padx=10)
        self.btn_colab = tk.Button(
            cadre_boutons, text="📦 Préparer un zip pour Colab", command=self._exporter_pour_colab)
        self.btn_colab.pack(side="left")

        self.texte = tk.Text(self, height=10, bg="#111111", fg="#d0d0d0", font=("Courier", 9), state="disabled")
        self.texte.pack(fill="both", expand=True, padx=14, pady=6)

        self.log_queue = queue.Queue()
        self.after(100, self._traiter_queue)

        if not torch_disponible():
            tk.Label(self, fg="#B00020", justify="left", text=(
                "⚠️ PyTorch/ultralytics n'est pas installé sur cet ordinateur : entraîner ICI\n"
                "n'est pas possible (\"Préparer un zip pour Colab\" reste disponible)."
            )).pack(anchor="w", padx=14, pady=(0, 8))

    def _log(self, message):
        self.log_queue.put(message)

    def _traiter_queue(self):
        try:
            while True:
                msg = self.log_queue.get_nowait()
                self.texte.config(state="normal")
                self.texte.insert("end", msg + "\n")
                self.texte.see("end")
                self.texte.config(state="disabled")
        except queue.Empty:
            pass
        self.after(100, self._traiter_queue)

    def _rafraichir_etat_dataset(self):
        data_yaml = Path(self.var_data_yaml.get())
        if data_yaml.exists():
            self.lbl_dataset.config(
                fg="#1B7A1B",
                text=f"✅ Dataset trouvé : {data_yaml}")
        else:
            self.lbl_dataset.config(
                fg="#B00020",
                text=f"⚠️ Introuvable : {data_yaml} -- régénère-le ci-dessus, ou choisis-en un "
                     "autre déjà exporté.")

    def _choisir_data_yaml(self):
        chemin = filedialog.askopenfilename(title="data.yaml du dataset détection",
                                             filetypes=[("YOLO data.yaml", "data.yaml"), ("Tous", "*.*")])
        if chemin:
            self.var_data_yaml.set(chemin)
            self._rafraichir_etat_dataset()

    def _regenerer_dataset(self):
        mod13 = _charger_script_numerique("13_exporter_annotations_detection.py", "export_detection_13")
        if mod13 is None:
            messagebox.showerror("Erreur", "Impossible de charger 13_exporter_annotations_detection.py")
            return
        resultats_path = REPO_ROOT / "output" / "resultats_par_rafale.json"
        dictionnaire_dir = REPO_ROOT / "output"
        if not resultats_path.exists():
            messagebox.showwarning(
                "Introuvable", f"{resultats_path} n'existe pas -- traite d'abord un lot de photos "
                               "(bouton \"🚀 Lancer l'analyse\").")
            return
        dataset_dir = REPO_ROOT / "dataset_detection"
        par_photo, stats = mod13.collecter_annotations([resultats_path], [dictionnaire_dir])
        mod13.exporter(par_photo, dataset_dir, val_ratio=0.15, seed=42)
        self.var_data_yaml.set(str(dataset_dir / "data.yaml"))
        self._rafraichir_etat_dataset()
        messagebox.showinfo(
            "Dataset régénéré",
            f"{stats['depuis_ia']} bbox depuis le pipeline, {stats['depuis_ajout_manuel']} "
            f"ajoutée(s) manuellement, {len(par_photo)} photo(s) au total -> {dataset_dir}")

    def _detecter_et_afficher_device(self):
        if not torch_disponible():
            self.lbl_device.config(
                text="PyTorch/ultralytics n'est pas installé sur cet ordinateur : impossible "
                     "d'entraîner ici.\nUtilise \"Préparer un zip pour Colab\" ci-dessous.")
            self.device = "cpu"
            return
        device, nom_gpu = detecter_device()
        if device == "cuda":
            self.device = "0"  # convention ultralytics : index de carte graphique, pas "cuda"
            self.lbl_device.config(text=f"✅ Carte graphique détectée : {nom_gpu}. L'entraînement sera rapide.")
        else:
            self.device = "cpu"
            self.lbl_device.config(
                text="⚠️ Pas de carte graphique compatible détectée : l'entraînement se fera sur le\n"
                     "processeur, ce qui peut être très long pour un modèle de détection. Si c'est\n"
                     "trop long, utilise plutôt \"Préparer un zip pour Colab\".")

    def _lancer_ici(self):
        data_yaml = Path(self.var_data_yaml.get())
        if not data_yaml.exists():
            messagebox.showwarning("Dataset introuvable", f"{data_yaml} n'existe pas.")
            return
        if not messagebox.askyesno(
                "Confirmer",
                f"Entraîner la détection ici, {self.var_epochs.get()} époques ?\n\n"
                "Ça peut être long, surtout sans carte graphique."):
            return
        self.btn_entrainer_ici.config(state="disabled")
        self.btn_colab.config(state="disabled")
        thread = threading.Thread(target=self._travailler_entrainement, args=(data_yaml,), daemon=True)
        thread.start()

    def _travailler_entrainement(self, data_yaml):
        mod16 = _charger_script_numerique("16_reentrainer_detection.py", "reentrainement_detection_16")
        if mod16 is None:
            self._log("❌ Impossible de charger 16_reentrainer_detection.py")
            self.after(0, self._reactiver_boutons)
            return
        try:
            best_pt, metriques = mod16.entrainer_detection(
                data_yaml=data_yaml, epochs=self.var_epochs.get(), imgsz=self.var_imgsz.get(),
                batch=self.var_batch.get(), device=self.device,
                model_depart=mod16.MODEL_DETECTION_ACTUEL,
                project_dir=REPO_ROOT / "runs_detection", log=self._log,
            )
        except Exception as e:  # noqa: BLE001
            logger.exception("Erreur pendant le réentraînement de la détection")
            self._log(f"❌ échec : {e}")
            self.after(0, self._reactiver_boutons)
            return

        self._log("— Comparaison avec le modèle actuellement en production —")
        try:
            metriques_actuel = mod16.evaluer_detection(mod16.MODEL_DETECTION_ACTUEL, data_yaml,
                                                         device=self.device, log=self._log)
            self._log(mod16.comparer(metriques, metriques_actuel))
        except Exception as e:  # noqa: BLE001
            logger.exception("Erreur pendant la comparaison des modèles de détection")
            self._log(f"⚠️ Comparaison impossible : {e}")

        resultats = {"detection": {
            "model_path": best_pt, "classes_path": None, "metriques": metriques,
            "dataset_note": f"détection réentraînée depuis {data_yaml}",
        }}
        self.after(0, lambda: self.app.afficher_versioning(resultats))

    def _reactiver_boutons(self):
        self.btn_entrainer_ici.config(state=("normal" if torch_disponible() else "disabled"))
        self.btn_colab.config(state="normal")

    def _exporter_pour_colab(self):
        data_yaml = Path(self.var_data_yaml.get())
        if not data_yaml.exists():
            messagebox.showwarning("Dataset introuvable", f"{data_yaml} n'existe pas.")
            return
        dest = filedialog.asksaveasfilename(
            title="Enregistrer le zip pour Colab", defaultextension=".zip",
            initialfile="dataset_detection.zip", filetypes=[("Archive zip", "*.zip")])
        if not dest:
            return
        self.btn_entrainer_ici.config(state="disabled")
        self.btn_colab.config(state="disabled")
        thread = threading.Thread(target=self._travailler_colab, args=(data_yaml, dest), daemon=True)
        thread.start()

    def _travailler_colab(self, data_yaml, dest):
        self._log("— Préparation du zip pour Colab —")
        dataset_dir = data_yaml.parent
        with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in dataset_dir.rglob("*"):
                if f.is_file():
                    zf.write(f, arcname=str(f.relative_to(dataset_dir)))
        self._log(f"✅ Zip prêt -> {dest}")
        self.after(0, lambda: messagebox.showinfo(
            "Zip prêt",
            f"Dataset exporté dans :\n{dest}\n\n"
            "Uploade ce zip dans le notebook Colab d'entraînement détection. Une fois "
            "best.pt téléchargé, relance cette interface et utilise \"Enregistrer un "
            "modèle entraîné ailleurs (Colab)\" à l'écran suivant."))
        self.after(0, self._reactiver_boutons)


# ============================================================================
# Écran alternatif : réentraînement depuis les corrections manuelles
# (fusionne scripts/10_fusionner_corrections_dataset.py, 11_reentrainer_modele.py
# et 12_valider_et_deployer.py dans cette même interface, pour les 3 types
# espèce/sexe/couleur au lieu du seul type "espèce" que gérait
# Lancer_reentrainement.bat -- voir TODO.md, section 1, pour le contexte du bug
# corrigé ici : les corrections de couleur et de sexe étaient fusionnées dans
# les datasets mais ne servaient jamais à réentraîner les modèles correspondants.
# ============================================================================

CORRECTIONS_TYPES: dict[str, dict[str, tuple[str, ...]]] = {
    "espece": dict(colonnes=("espece_corrigee", "espece"), exclure=("inconnu", "")),
    "sexe": dict(colonnes=("sexe_corrige", "sexe"), exclure=("inconnu", "indetermine_juvenile", "")),
    "couleur": dict(colonnes=("couleur_corrigee", "couleur"), exclure=("indeterminable", "")),
}



def compter_corrections(corrections_csv: Path):
    """Aperçu (lecture seule) du nombre de corrections exploitables par type,
    avant de lancer la fusion réelle. Mêmes colonnes et mêmes valeurs
    exclues que scripts/10_fusionner_corrections_dataset.py, pour que
    l'aperçu corresponde exactement à ce que la fusion va faire — y compris
    l'exclusion des lignes dont le crop_path n'existe plus sur le disque
    (10_fusionner_corrections_dataset.py les ignore silencieusement)."""
    compteurs = {k: 0 for k in CORRECTIONS_TYPES}
    if not corrections_csv.exists():
        return compteurs
    import csv
    with open(corrections_csv, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            crop_path = REPO_ROOT / row.get("crop_path", "")
            if not crop_path.exists():
                continue
            for type_key, cfg in CORRECTIONS_TYPES.items():
                col_corrigee, col_brute = cfg["colonnes"]
                valeur = row.get(col_corrigee) or row.get(col_brute)
                if valeur and valeur.lower() not in cfg["exclure"]:
                    compteurs[type_key] += 1
    return compteurs


class EcranCorrections(tk.Frame):
    def __init__(self, master, app):
        super().__init__(master)
        self.app = app
        pad = {"padx": 14, "pady": 8}

        tk.Label(self, text="🔁 Réentraînement depuis les corrections",
                 font=("Helvetica", 13, "bold")).pack(anchor="w", **pad)
        tk.Label(self, justify="left", fg="#333333", text=(
            "Fusionne les corrections faites dans l'écran de relecture\n"
            "(09_ecran_identification.py) dans les datasets d'entraînement,\n"
            "puis réentraîne les modèles espèce / sexe / couleur concernés --\n"
            "directement sur cet ordinateur, ou via un zip à envoyer sur Colab."
        )).pack(anchor="w", **pad)

        cadre_fichier = ttk.LabelFrame(self, text="Fichier de corrections")
        cadre_fichier.pack(fill="x", **pad)
        # Le maître cumulatif (output_archives/dictionnaire_maitre/corrections.csv)
        # est alimenté automatiquement à chaque nouveau lot traité par
        # l'interface graphique -- voir interface_graphique.py,
        # _fusionner_source_dans_maitre() / 15_fusionner_corrections.py --
        # et contient donc aussi les corrections des lots déjà archivés,
        # contrairement à output/corrections.csv qui ne reflète que le lot
        # courant (vidé à chaque archivage). On retombe sur output/
        # seulement si le maître n'existe pas encore (ex. aucun lot
        # traité depuis l'ajout de cette fusion, ou usage isolé de ce
        # module sans être passé par l'interface graphique).
        chemin_maitre = REPO_ROOT / "output_archives" / "dictionnaire_maitre" / "corrections.csv"
        chemin_defaut = chemin_maitre if chemin_maitre.exists() else REPO_ROOT / "output" / "corrections.csv"
        self.var_chemin = tk.StringVar(value=str(chemin_defaut))
        tk.Entry(cadre_fichier, textvariable=self.var_chemin, width=60).pack(
            side="left", padx=10, pady=8)
        tk.Button(cadre_fichier, text="Parcourir…", command=self._parcourir).pack(
            side="left", padx=(0, 6))
        tk.Button(cadre_fichier, text="Analyser", command=self._analyser).pack(
            side="left", padx=(0, 10))

        self.cadre_types = ttk.LabelFrame(self, text="Modèles à réentraîner")
        self.cadre_types.pack(fill="x", **pad)
        self.var_types = {}
        self.lbl_types = {}
        for type_key in CORRECTIONS_TYPES:
            ligne = tk.Frame(self.cadre_types)
            ligne.pack(fill="x", padx=10, pady=3)
            var = tk.BooleanVar(value=False)
            self.var_types[type_key] = var
            cb = ttk.Checkbutton(ligne, variable=var, text=IA_DEFS[type_key]["label"], state="disabled")
            cb.pack(side="left")
            lbl = tk.Label(ligne, text="— corrections non analysées —", fg="#666666")
            lbl.pack(side="left", padx=8)
            self.lbl_types[type_key] = (cb, lbl)

        cadre_device = ttk.LabelFrame(self, text="Ordinateur utilisé")
        cadre_device.pack(fill="x", **pad)
        self.lbl_device = tk.Label(cadre_device, justify="left", fg="#333333")
        self.lbl_device.pack(anchor="w", padx=10, pady=6)
        self._detecter_et_afficher_device()

        cadre_params = ttk.LabelFrame(self, text="Paramètres d'entraînement")
        cadre_params.pack(fill="x", **pad)
        tk.Label(cadre_params, text="Époques :").grid(row=0, column=0, sticky="w", padx=10, pady=6)
        self.var_epochs = tk.IntVar(value=10)
        ttk.Spinbox(cadre_params, from_=1, to=100, textvariable=self.var_epochs, width=6).grid(
            row=0, column=1, sticky="w")
        tk.Label(cadre_params, text="Taille de batch :").grid(row=0, column=2, sticky="w", padx=10)
        self.var_batch = tk.IntVar(value=32)
        ttk.Spinbox(cadre_params, from_=4, to=128, textvariable=self.var_batch, width=6).grid(
            row=0, column=3, sticky="w")

        cadre_boutons = tk.Frame(self)
        cadre_boutons.pack(fill="x", **pad)
        tk.Button(cadre_boutons, text="← Retour", command=self.app.afficher_config).pack(
            side="left")
        self.btn_entrainer_ici = tk.Button(
            cadre_boutons, text="🚀 Fusionner + entraîner ici", bg="#4C8BF5", fg="white",
            font=("Helvetica", 11, "bold"), state="disabled", command=self._lancer_ici)
        self.btn_entrainer_ici.pack(side="left", padx=10)
        self.btn_colab = tk.Button(
            cadre_boutons, text="📦 Fusionner puis préparer un zip pour Colab",
            state="disabled", command=self._exporter_pour_colab)
        self.btn_colab.pack(side="left")

        self.texte = tk.Text(self, height=10, bg="#111111", fg="#d0d0d0", font=("Courier", 9), state="disabled")
        self.texte.pack(fill="both", expand=True, padx=14, pady=6)

        self.log_queue = queue.Queue()
        self.after(100, self._traiter_queue)

        if not torch_disponible():
            tk.Label(self, fg="#B00020", justify="left", text=(
                "⚠️ PyTorch n'est pas installé sur cet ordinateur : entraîner ICI n'est\n"
                "pas possible (\"Fusionner puis préparer un zip pour Colab\" reste\n"
                "disponible, lui, sans PyTorch). Installe les dépendances\n"
                "(requirements.txt) pour pouvoir entraîner directement ici."
            )).pack(anchor="w", padx=14, pady=(0, 8))

    def _detecter_et_afficher_device(self):
        """Même détection que EcranEntrainement -- purement informative ici :
        contrairement à EcranEntrainement, l'absence de GPU/PyTorch ne bloque
        jamais complètement cet écran, puisque l'export Colab reste possible
        (voir le bouton dédié et l'avertissement PyTorch plus bas)."""
        if not torch_disponible():
            self.lbl_device.config(
                text="PyTorch n'est pas installé sur cet ordinateur : impossible d'entraîner ici.\n"
                     "Utilise \"Fusionner puis préparer un zip pour Colab\" ci-dessous.")
            self.device = "cpu"
            return
        device, nom_gpu = detecter_device()
        if device == "cuda":
            self.device = "cuda"
            self.lbl_device.config(text=f"✅ Carte graphique détectée : {nom_gpu}. L'entraînement sera rapide.")
        else:
            self.device = "cpu"
            self.lbl_device.config(
                text="⚠️ Pas de carte graphique compatible détectée : l'entraînement se fera sur le\n"
                     "processeur, ce qui peut être long (dizaines de minutes à quelques heures selon\n"
                     "le nombre d'images). Si c'est trop long, utilise plutôt \"Fusionner puis préparer\n"
                     "un zip pour Colab\".")

    def _log(self, msg):
        self.log_queue.put(msg)

    def _traiter_queue(self):
        try:
            while True:
                msg = self.log_queue.get_nowait()
                self.texte.config(state="normal")
                self.texte.insert("end", msg + "\n")
                self.texte.see("end")
                self.texte.config(state="disabled")
        except queue.Empty:
            pass
        self.after(150, self._traiter_queue)

    def _parcourir(self):
        chemin = filedialog.askopenfilename(
            title="Fichier corrections.csv", filetypes=[("CSV", "*.csv")])
        if chemin:
            self.var_chemin.set(chemin)

    def _analyser(self):
        chemin = Path(self.var_chemin.get())
        compteurs = compter_corrections(chemin)
        if not chemin.exists():
            messagebox.showwarning("Introuvable", f"Fichier introuvable :\n{chemin}")
            return
        au_moins_un = False
        for type_key, n in compteurs.items():
            cb, lbl = self.lbl_types[type_key]
            if n > 0:
                cb.config(state="normal")
                self.var_types[type_key].set(True)
                lbl.config(text=f"{n} correction(s) exploitable(s)", fg="#1B7A1B")
                au_moins_un = True
            else:
                cb.config(state="disabled")
                self.var_types[type_key].set(False)
                lbl.config(text="aucune correction exploitable", fg="#666666")
        self.btn_entrainer_ici.config(state=("normal" if au_moins_un and torch_disponible() else "disabled"))
        self.btn_colab.config(state=("normal" if au_moins_un else "disabled"))
        if not au_moins_un:
            messagebox.showinfo("Aucune correction", "Aucune correction exploitable trouvée dans ce fichier.")

    def _lancer_ici(self):
        types_choisis = [k for k, v in self.var_types.items() if v.get()]
        if not types_choisis:
            messagebox.showwarning("Rien à faire", "Coche au moins un type de modèle à réentraîner.")
            return
        if not messagebox.askyesno(
                "Confirmer",
                "Fusionner les corrections puis entraîner ici : "
                + ", ".join(IA_DEFS[k]["label"] for k in types_choisis)
                + " ?\n\nÇa peut prendre du temps selon le nombre d'images et d'époques."):
            return
        self.btn_entrainer_ici.config(state="disabled")
        self.btn_colab.config(state="disabled")
        thread = threading.Thread(target=self._travailler_entrainement, args=(types_choisis,), daemon=True)
        thread.start()

    def _exporter_pour_colab(self):
        types_choisis = [k for k, v in self.var_types.items() if v.get()]
        if not types_choisis:
            messagebox.showwarning("Rien à faire", "Coche au moins un type de modèle à exporter.")
            return
        dest = filedialog.asksaveasfilename(
            title="Enregistrer le zip pour Colab", defaultextension=".zip",
            initialfile="corrections_pret.zip", filetypes=[("Archive zip", "*.zip")])
        if not dest:
            return
        self.btn_entrainer_ici.config(state="disabled")
        self.btn_colab.config(state="disabled")
        thread = threading.Thread(target=self._travailler_colab, args=(types_choisis, dest), daemon=True)
        thread.start()

    def _fusionner(self, types_choisis):
        """Fusionne les corrections cochées dans les datasets d'entraînement
        (10_fusionner_corrections_dataset.py). Commun aux deux chemins
        (entraînement ici / export Colab) -- la fusion doit TOUJOURS avoir
        lieu avant, y compris pour l'export Colab : sinon le zip envoyé sur
        Colab contiendrait un dataset pas à jour, pas les corrections
        qu'on vient de cocher. Journalise dans self.texte.

        Returns:
            True si la fusion a pu tourner (même si 0 image concrètement
            fusionnée), False si 10_fusionner_corrections_dataset.py n'a
            pas pu être chargé -- dans ce cas les deux chemins appelants
            s'arrêtent avant d'entraîner/zipper quoi que ce soit.
        """
        import io
        from contextlib import redirect_stdout

        mod10 = _charger_script_numerique("10_fusionner_corrections_dataset.py", "fusion_corrections_10")
        if mod10 is None:
            self._log("❌ Impossible de charger 10_fusionner_corrections_dataset.py")
            return False

        self._log("— Fusion des corrections dans les datasets —")
        buf = io.StringIO()
        with redirect_stdout(buf):
            mod10.fusionner_corrections(
                corrections_csv=Path(self.var_chemin.get()),
                dataset_espece=IA_DEFS["espece"]["staging_dir"],
                dataset_sexe=IA_DEFS["sexe"]["staging_dir"],
                dataset_couleur=IA_DEFS["couleur"]["staging_dir"],
            )
        for ligne in buf.getvalue().splitlines():
            self._log(ligne)
        return True

    def _reactiver_boutons(self):
        self.btn_entrainer_ici.config(state=("normal" if torch_disponible() else "disabled"))
        self.btn_colab.config(state="normal")

    def _travailler_entrainement(self, types_choisis):
        if not self._fusionner(types_choisis):
            self.after(0, self._reactiver_boutons)
            return

        resultats = {}
        for type_key in types_choisis:
            dataset_dir = Path(IA_DEFS[type_key]["staging_dir"])
            n_classes = sum(1 for d in dataset_dir.iterdir() if d.is_dir()) if dataset_dir.exists() else 0
            if n_classes < 2:
                self._log(f"— {IA_DEFS[type_key]['label']} : pas assez de classes différentes, ignoré.")
                continue
            self._log(f"— Entraînement {IA_DEFS[type_key]['label']} —")
            try:
                model_path, classes_path, metriques = entrainer_classification(
                    pret_dir=dataset_dir,
                    epochs=self.var_epochs.get(),
                    batch_size=self.var_batch.get(),
                    device_name=self.device,
                    log=self._log,
                )
                resultats[type_key] = {
                    "model_path": model_path, "classes_path": classes_path, "metriques": metriques,
                    "dataset_note": f"corrections fusionnées via {self.var_chemin.get()}",
                }
                self._log(f"  ✅ terminé — précision validation : {metriques.get('val_acc', 0):.1%}")
            except Exception as e:  # noqa: BLE001
                logger.exception("Erreur pendant le réentraînement par corrections")
                self._log(f"  ❌ échec : {e}")

        self.after(0, lambda: self.app.afficher_versioning(resultats))

    def _travailler_colab(self, types_choisis, dest):
        if not self._fusionner(types_choisis):
            self.after(0, self._reactiver_boutons)
            return

        self._log("— Préparation du zip pour Colab —")
        with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as zf:
            for type_key in types_choisis:
                dataset_dir = Path(IA_DEFS[type_key]["staging_dir"])
                if not dataset_dir.exists():
                    continue
                for f in dataset_dir.rglob("*.jpg"):
                    zf.write(f, arcname=f"{type_key}/{f.relative_to(dataset_dir)}")
        self._log(f"✅ Zip prêt -> {dest}")

        self.after(0, lambda: messagebox.showinfo(
            "Zip prêt",
            f"Dataset(s) exporté(s) dans :\n{dest}\n\n"
            "Uploade ce zip à l'étape correspondante du notebook Colab. Une fois le "
            "modèle téléchargé (modele_xxx.pt + classes_xxx.json), relance cette "
            "interface : elle te proposera de l'enregistrer comme nouvelle version dès "
            "l'écran suivant, ou lance directement scripts/gestion_modeles.py register."))
        self.after(0, self._reactiver_boutons)


# ============================================================================
# Application principale (enchaîne les écrans)
# ============================================================================

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("🦆 Préparation des données — Couleur / Espèce")
        self.geometry("820x720")
        self.minsize(700, 560)

        self.ias_a_traiter = []
        self.file_tri = []
        self.burst_gap = BURST_GAP_DEFAULT
        self.exclure_nuit_couleur = True
        self.frame_courante = None

        self.afficher_config()

    def _remplacer(self, frame):
        if self.frame_courante is not None:
            self.frame_courante.destroy()
        self.frame_courante = frame
        frame.pack(fill="both", expand=True)

    def afficher_config(self):
        self._remplacer(EcranConfig(self, self))

    def afficher_extraction(self):
        self._remplacer(EcranExtraction(self, self))

    def afficher_tri_suivant(self):
        if not self.file_tri:
            self._remplacer(EcranEntrainement(self, self))
            return
        ia_key = self.file_tri.pop(0)
        self._remplacer(EcranTri(self, self, ia_key))

    def afficher_versioning(self, resultats):
        self._remplacer(EcranVersioning(self, self, resultats))

    def afficher_corrections(self):
        self._remplacer(EcranCorrections(self, self))

    def afficher_detection(self):
        self._remplacer(EcranDetection(self, self))


def main():
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()