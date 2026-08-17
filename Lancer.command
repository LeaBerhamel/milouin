#!/bin/bash
# Lancer.command — équivalent Mac de Lancer.bat (Windows).
#
# Double-cliquable depuis le Finder : installe les dépendances Python au
# tout premier lancement (venv local, isolé du reste du système), puis
# ouvre l'interface graphique. Les lancements suivants sont directs.
#
# Glisser-déposer : un dossier de photos et/ou un fichier .xlsx/.xls/.csv
# déposés sur cette icône sont transmis tels quels à interface_graphique.py
# (voir _preremplir_depuis_glisser_deposer côté Python) -- "$@" plus bas.

set -e

# Se placer à la racine du projet, quel que soit le dossier de travail
# depuis lequel macOS a lancé ce script (double-clic Finder, terminal, ou
# glisser-déposer) -- même logique que BASE_DIR dans scripts/utils/config.py.
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

echo "=== Suivi Canards ==="
echo ""

# --- Python : cherche python3, message clair sinon (pas de traceback illisible) ---
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 n'est pas installé (ou pas dans le PATH)."
    echo ""
    echo "Installe-le depuis https://www.python.org/downloads/ (version macOS),"
    echo "ou avec Homebrew : brew install python"
    echo ""
    read -p "Appuie sur Entrée pour fermer cette fenêtre..."
    exit 1
fi

VENV_DIR="$DIR/.venv"
PREMIERE_FOIS=0

if [ ! -d "$VENV_DIR" ]; then
    PREMIERE_FOIS=1
    echo "📦 Premier lancement : installation des dépendances..."
    echo "   (peut prendre 10 à 20 minutes -- ne ferme pas cette fenêtre)"
    echo ""
    python3 -m venv "$VENV_DIR"
fi

# Active l'environnement virtuel (isolé -- n'installe rien dans le Python
# système, pas besoin de sudo, pas de conflit avec d'autres projets Python).
source "$VENV_DIR/bin/activate"

if [ "$PREMIERE_FOIS" -eq 1 ]; then
    pip install --upgrade pip
    if [ -f "$DIR/requirements.txt" ]; then
        pip install -r "$DIR/requirements.txt"
    else
        echo "⚠️  requirements.txt introuvable à la racine du projet --"
        echo "   dépose-le à côté de Lancer.command puis relance."
        read -p "Appuie sur Entrée pour fermer cette fenêtre..."
        exit 1
    fi
    echo ""
    echo "✅ Installation terminée."
    echo ""
fi

# Lance l'interface graphique, en transmettant les éléments glissés-déposés
# sur l'icône (dossier de photos / fichier tableur) à interface_graphique.py.
python3 "$DIR/scripts/interface_graphique.py" "$@"
