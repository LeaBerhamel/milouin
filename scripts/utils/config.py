"""
Configuration centrale du projet : chemins, logger, constantes.
"""

import logging
import sys
from pathlib import Path

# Chemins de base du projet.
#
# En exécution normale (python scripts/...), __file__ pointe vers ce
# fichier et BASE_DIR remonte à scripts/.
#
# Une fois packagé avec PyInstaller (voir packaging/SuiviCanards.spec,
# mode --onedir), sys.frozen vaut True et __file__ pointe vers un dossier
# d'extraction temporaire (_MEIPASS) qui n'existe plus après la fermeture
# du programme -- utiliser BASE_DIR = __file__ ferait planter tous les
# accès à data/, models/, output/. Il faut utiliser sys.executable, qui
# pointe vers le vrai emplacement de SuiviCanards.exe (dans le dossier
# dist/SuiviCanards/, copié à la racine du projet -- voir
# packaging/README.md).
if getattr(sys, "frozen", False):
    BASE_DIR = Path(sys.executable).resolve().parent
else:
    BASE_DIR = Path(__file__).resolve().parent.parent  # scripts/utils/config.py -> scripts/

# Dossiers de données
DATA_DIR = BASE_DIR.parent / "data"
PHOTOS_DIR = DATA_DIR / "photos"
TABLEUR_PATH = DATA_DIR / "observations.xlsx"

# Dossiers de sortie
OUTPUT_DIR = BASE_DIR.parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Dossiers de modèles
MODELS_DIR = BASE_DIR.parent / "models"
MODEL_DETECTION = MODELS_DIR / "detection" / "best.pt"
MODEL_COULEUR = MODELS_DIR / "couleur" / "modele_couleur.pt"
MODEL_ESPECE = MODELS_DIR / "espece" / "modele_espece.pt"
MODEL_SEXE = MODELS_DIR / "sexe" / "modele_sexe.pt"

# Configuration du logger (utilisé par train_classification_ia.py)
# Les autres scripts utilisent print() volontairement (scripts orientés
# utilisateur final, affichage terminal lisible). Cette coexistence est
# documentée et intentionnelle (voir README_LIVRAISON.md étape 3).
LOG_FILE = OUTPUT_DIR / "pipeline_production.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger(__name__)

# Constantes du pipeline
TOLERANCE_MATCHING_STRICT = 30  # secondes
TOLERANCE_MATCHING_LARGE = 180  # secondes
SEUIL_CONFIANCE_HAUTE = 0.85
SEUIL_CONFIANCE_MOYENNE = 0.70

# Constantes utilisées par le pipeline et les interfaces
REQUIRED_COLUMNS = ["DATE", "HEURE", "BAGUE", "ESPECE", "COULEUR_MARQUE"]
BURST_GAP_DEFAULT = 30  # secondes

# Paramètres de détection YOLO
CONFIDENCE_THRESHOLD_MARQUE = 0.25
CONFIDENCE_THRESHOLD_CANARD = 0.25
DETECTION_CLASSE_MARQUE = "marque_nasale"
DETECTION_CLASSE_CANARD = "canard"
DETECTION_IMGSZ = 1280
DETECTION_IMGSZ_ZOOM = 640

# Codes du tableur
ESPECE_CODE_TABLEUR = {"milouin": "AYTFER", "morillon": "AYTFUL"}
SEXE_CODE_TABLEUR = {"male": "M", "femelle": "F", "indetermine_juvenile": "J"}

# Table de correspondance INVERSE (code -> code, identité) en plus de
# ESPECE_CODE_TABLEUR/SEXE_CODE_TABLEUR (nom français -> code) : sert à
# normaliser_espece()/normaliser_sexe() ci-dessous, pour qu'une valeur DÉJÀ
# en code (venant du tableur) ou en nom français (venant de la
# classification IA) ressorte toujours sous la même forme -- clés en
# minuscules pour une comparaison insensible à la casse.
_ESPECE_VERS_CODE = {
    **ESPECE_CODE_TABLEUR,
    **{code.lower(): code for code in ESPECE_CODE_TABLEUR.values()},
}
_SEXE_VERS_CODE = {
    **SEXE_CODE_TABLEUR,
    **{code.lower(): code for code in SEXE_CODE_TABLEUR.values()},
}


def normaliser_espece(valeur):
    """Ramène une valeur d'espèce (nom français type "milouin" OU code déjà
    normalisé type "AYTFER", n'importe quelle casse) vers son code tableur
    -- seule fonction du projet à faire cette conversion, pour que le
    tableur d'observations (utils/tableur.py), le pipeline
    (utils/pipeline.py) et les synthèses (utils/synthese.py) unifient
    TOUS la même valeur pour la même espèce, plutôt que de mélanger les
    deux vocabulaires selon la source (IA vs tableur) comme c'était le cas
    auparavant. Une valeur absente ou inconnue est retournée telle quelle
    en majuscules (pas avalée silencieusement -- une espèce non
    répertoriée doit rester visible, pas disparaître)."""
    if not valeur:
        return "INCONNUE"
    return _ESPECE_VERS_CODE.get(str(valeur).strip().lower(), str(valeur).strip().upper())


def normaliser_sexe(valeur):
    """Même principe que normaliser_espece(), pour le sexe ("male"/"M",
    "femelle"/"F", "indetermine_juvenile"/"J")."""
    if not valeur:
        return "?"
    return _SEXE_VERS_CODE.get(str(valeur).strip().lower(), str(valeur).strip().upper())

# Classes des 3 classifieurs IA (couleur/espèce/sexe), utilisées par
# utils/pipeline.py (IAPipeline.ia_couleur / .ia_espece / .ia_sexe) comme
# FILET DE SÉCURITÉ uniquement -- voir le docstring de _load_classifier() :
# l'ordre réel utilisé pour l'inférence vient du fichier classes_*.json
# sauvegardé à côté de chaque modèle (torchvision.datasets.ImageFolder
# trie par ordre alphabétique, indépendamment de ces listes). Ces listes
# ne servent que si ce fichier de classes est absent (modèle ancien, ou
# fichier manquant) -- dans ce cas, un warning est loggé pour inciter à
# ré-entraîner/ré-enregistrer le modèle plutôt que de compter dessus.
#
# Ces trois noms étaient importés par utils/pipeline.py sans jamais avoir
# été définis ici -- ImportError garanti au premier accès à .ia_couleur,
# .ia_espece ou .ia_sexe. Valeurs déduites du reste du dépôt (à confirmer
# si de nouvelles classes sont ajoutées) :
#   - ESPECES_CLASSES : seules espèces gérées par le projet, voir
#     ESPECE_CODE_TABLEUR ci-dessus et 00_migrate_annotations.py.
#   - SEXES_CLASSES : mêmes clés que SEXE_CODE_TABLEUR ci-dessus.
#   - COULEURS_CLASSES : voir triage_rapide.py (COLOR_KEYS), seul autre
#     endroit du dépôt où les couleurs de bague possibles sont listées.
ESPECES_CLASSES = ["milouin", "morillon"]
SEXES_CLASSES = ["male", "femelle", "indetermine_juvenile"]
COULEURS_CLASSES = ["bleu", "marron", "indeterminable"]