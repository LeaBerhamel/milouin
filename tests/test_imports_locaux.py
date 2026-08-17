# -*- coding: utf-8 -*-
"""
Test de non-régression : vérifie que chaque `from <module_local> import
a, b, c` présent dans scripts/, scripts/utils/ et module_corrections/
pointe bien vers des noms qui existent réellement dans le module visé.

Pourquoi ce test existe
------------------------
Bug réel rencontré : `ecrire_corrections.py` avait perdu la ligne
`def annuler_correction(...):` (corps de fonction resté orphelin après un
`return True` précédent), alors que `fenetre_rafale.py` continuait à faire
`from ecrire_corrections import annuler_correction, ...`. Résultat : un
ImportError qui ne se déclenche qu'à l'exécution, au moment où l'utilisateur
lance l'interface -- rien ne l'attrape avant (pas d'erreur de syntaxe : le
fichier reste un `.py` valide, juste avec une fonction en moins).

Ce test attrape cette catégorie de bug sans avoir besoin d'importer
réellement les modules (donc sans dépendre de tkinter/PIL/un affichage
disponible) : il analyse le code source avec `ast` et vérifie, pour
chaque import local, que le nom importé est bien défini au niveau module
dans le fichier cible (fonction, classe, variable, ou re-export via un
autre `from ... import ...`).

Limite assumée : ne résout que les imports "locaux" (modules présents
sous forme de .py à côté du fichier qui importe, ou dans scripts/utils/).
Les imports de bibliothèques externes (tkinter, PIL, etc.) sont ignorés :
ce n'est pas leur rôle d'être vérifiés ici.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

# Racine du dépôt = parent du dossier tests/ (voir structure du README :
# scripts/, module_corrections/ et tests/ sont tous à la racine).
RACINE = Path(__file__).resolve().parent.parent

DOSSIERS_A_VERIFIER = [
    RACINE / "module_corrections",
    RACINE / "scripts",
    RACINE / "scripts" / "utils",
]


def _fichiers_python(dossier: Path) -> list[Path]:
    if not dossier.is_dir():
        return []
    return sorted(p for p in dossier.glob("*.py") if p.name != "__init__.py")


def _symboles_definis(fichier: Path) -> set[str]:
    """Noms définis au niveau module dans `fichier` : fonctions, classes,
    affectations simples, et noms ré-exportés via un `from x import y`."""
    arbre = ast.parse(fichier.read_text(encoding="utf-8"), filename=str(fichier))
    noms: set[str] = set()

    def _visiter(corps: list[ast.stmt]) -> None:
        for noeud in corps:
            if isinstance(noeud, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                noms.add(noeud.name)
            elif isinstance(noeud, ast.Assign):
                for cible in noeud.targets:
                    if isinstance(cible, ast.Name):
                        noms.add(cible.id)
            elif isinstance(noeud, ast.AnnAssign) and isinstance(noeud.target, ast.Name):
                noms.add(noeud.target.id)
            elif isinstance(noeud, ast.ImportFrom):
                # re-export : `from autre_module import truc` rend `truc`
                # disponible aussi depuis CE module.
                for alias in noeud.names:
                    noms.add(alias.asname or alias.name)
            elif isinstance(noeud, ast.Import):
                for alias in noeud.names:
                    noms.add((alias.asname or alias.name).split(".")[0])
            elif isinstance(noeud, ast.If):
                # Pattern courant (ex: utils/config.py::BASE_DIR) : une
                # constante définie différemment selon une condition
                # (exécutable "frozen" vs script) -- toujours au niveau
                # module, juste indentée sous le `if`.
                _visiter(noeud.body)
                _visiter(noeud.orelse)
            elif isinstance(noeud, ast.Try):
                _visiter(noeud.body)
                for handler in noeud.handlers:
                    _visiter(handler.body)
                _visiter(noeud.orelse)
                _visiter(noeud.finalbody)

    _visiter(arbre.body)
    return noms


def _modules_locaux_disponibles() -> dict[str, Path]:
    """Map nom_de_module -> fichier .py, pour tous les fichiers sous les
    DOSSIERS_A_VERIFIER (permet de résoudre `from ecrire_corrections import
    ...` aussi bien que `from utils.propositions import ...`)."""
    modules: dict[str, Path] = {}
    for dossier in DOSSIERS_A_VERIFIER:
        for fichier in _fichiers_python(dossier):
            modules[fichier.stem] = fichier
            if dossier.name == "utils":
                modules[f"utils.{fichier.stem}"] = fichier
    return modules


def _cas_a_verifier():
    """Génère (fichier_importeur, nom_module_importe, {noms importés}) pour
    chaque `from <module_local> import ...` trouvé."""
    modules_locaux = _modules_locaux_disponibles()
    cas = []

    tous_fichiers = [f for d in DOSSIERS_A_VERIFIER for f in _fichiers_python(d)]
    for fichier in tous_fichiers:
        try:
            arbre = ast.parse(fichier.read_text(encoding="utf-8"), filename=str(fichier))
        except SyntaxError as exc:
            cas.append(pytest.param(fichier, None, None, id=f"{fichier.name}::erreur_syntaxe:{exc}"))
            continue

        for noeud in ast.walk(arbre):
            if not isinstance(noeud, ast.ImportFrom):
                continue
            if noeud.level and noeud.level > 0:
                continue  # import relatif (`from . import x`) -- hors périmètre
            module_nom = noeud.module or ""
            if module_nom not in modules_locaux:
                continue  # pas un module local qu'on peut vérifier (ex: tkinter, PIL)

            noms_importes = {alias.name for alias in noeud.names if alias.name != "*"}
            if noms_importes:
                cas.append(
                    pytest.param(
                        fichier, modules_locaux[module_nom], noms_importes,
                        id=f"{fichier.name}<-{module_nom}",
                    )
                )
    return cas


@pytest.mark.parametrize("fichier_importeur, fichier_cible, noms_importes", _cas_a_verifier())
def test_import_local_resout_vers_symboles_existants(fichier_importeur, fichier_cible, noms_importes):
    if fichier_cible is None:
        pytest.fail(f"{fichier_importeur} : erreur de syntaxe, impossible d'analyser les imports.")

    symboles_disponibles = _symboles_definis(fichier_cible)
    manquants = noms_importes - symboles_disponibles

    assert not manquants, (
        f"{fichier_importeur.relative_to(RACINE)} importe "
        f"{sorted(manquants)} depuis {fichier_cible.relative_to(RACINE)}, "
        f"mais {'ce nom n' + chr(39) + 'existe' if len(manquants) == 1 else 'ces noms n' + chr(39) + 'existent'} "
        f"pas au niveau module de ce fichier (def manquante, mal indentée, "
        f"ou corps de fonction orphelin -- voir le docstring de ce test)."
    )