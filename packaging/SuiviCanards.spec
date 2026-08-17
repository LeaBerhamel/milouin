# -*- mode: python ; coding: utf-8 -*-
"""
Spec PyInstaller pour "Suivi Canards".
"""

import sys
from pathlib import Path

block_cipher = None

PROJECT_ROOT = Path(SPECPATH).resolve().parent  # dossier racine du projet
ENTRY_POINT = str(PROJECT_ROOT / "scripts" / "interface_graphique.py")
SCRIPTS_DIR = str(PROJECT_ROOT / "scripts")

# Définition de TOUS les dossiers de code à inclure dans les chemins de recherche Python
search_paths = [
    SCRIPTS_DIR,
    str(PROJECT_ROOT / "scripts" / "utils"),
    str(PROJECT_ROOT / "scripts" / "module_corrections"),
    str(PROJECT_ROOT / "module_corrections"),
    str(PROJECT_ROOT / "utils"),
]

# Copie brute des fichiers .py pour qu'ils soient lisibles sur le disque dans _internal/
added_files = [
    (str(PROJECT_ROOT / "scripts" / "*.py"), "."),
]

if (PROJECT_ROOT / "scripts" / "utils").exists():
    added_files.append((str(PROJECT_ROOT / "scripts" / "utils" / "*.py"), "utils"))

if (PROJECT_ROOT / "scripts" / "module_corrections").exists():
    added_files.append((str(PROJECT_ROOT / "scripts" / "module_corrections" / "*.py"), "module_corrections"))
elif (PROJECT_ROOT / "module_corrections").exists():
    added_files.append((str(PROJECT_ROOT / "module_corrections" / "*.py"), "module_corrections"))

# Imports masqués pour s'assurer que tous les sous-modules de utils sont bien empaquetés
hidden_imports = [
    "utils",
    "utils.burst_detector",
    "utils.config",
    "serveur_corrections",
    "PIL._tkinter_finder",
    "sklearn.utils._typedefs",
    "sklearn.neighbors._partition_nodes",
    "torch",
    "torchvision",
    "torchvision.models",
    "torchvision.transforms",
    "ultralytics",
    "ultralytics.nn.modules",
    "cv2",
    "openpyxl",
    "openpyxl.cell._writer",
    "matplotlib",
    "matplotlib.backends.backend_agg",
]

icon_path = PROJECT_ROOT / "packaging" / "icon.ico"
exe_icon = str(icon_path) if icon_path.exists() else None

a = Analysis(
    [ENTRY_POINT],
    pathex=search_paths,
    binaries=[],
    datas=added_files,
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="SuiviCanards",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    target_arch=None,
    icon=exe_icon,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="SuiviCanards",
)