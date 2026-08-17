# Configuration et vérification mypy

## État actuel

**mypy est configuré** dans `pyproject.toml` mais **pas encore activé en CI**,
car il n'a pas été vérifié localement avec toutes les dépendances installées
(notamment `torch`, `ultralytics`, `cv2`).

## Vérification locale (à faire avant d'activer en CI)

### 1. Installer les dépendances complètes

```bash
pip install -r requirements.txt -r requirements-dev.txt
```

### 2. Lancer mypy sur le code principal

```bash
# Vérifier scripts/
mypy scripts/ --ignore-missing-imports

# Si des erreurs apparaissent, les corriger ou ajouter des # type: ignore
# ciblés sur les lignes problématiques (en documentant pourquoi)
```

### 3. Vérifier les modules critiques

```bash
# Modules prioritaires (calculs de confiance, matching, pipeline)
mypy scripts/utils/confiance.py
mypy scripts/utils/pipeline.py
mypy scripts/02_process_rafales.py
```

### 4. Configuration actuelle (`pyproject.toml`)

```toml
[tool.mypy]
python_version = "3.10"
warn_return_any = true          # Alerte si une fonction retourne Any
warn_unused_configs = true      # Alerte si une config mypy n'est pas utilisée
ignore_missing_imports = true   # Ne plante pas sur les imports tiers sans stubs
```

**`ignore_missing_imports = true`** : nécessaire car certaines libs
(ultralytics, openpyxl) n'ont pas de type stubs complets. Mypy vérifiera quand
même TON code, juste pas les appels aux fonctions de ces libs.

## Une fois vérifié localement

**Décommenter** les lignes mypy dans `.github/workflows/tests.yml` :

```yaml
- name: Vérification des types (mypy)
  run: mypy scripts/ --ignore-missing-imports
```

## Limites connues

- **Notebooks Jupyter** : mypy ne peut pas les vérifier directement. Utiliser
  `nbqa mypy notebook.ipynb` si nécessaire.
- **Code legacy sans type hints** : mypy les ignore par défaut. Pour forcer
  une vérification stricte, ajouter `disallow_untyped_defs = true` dans
  `pyproject.toml` (mais attends que le code principal soit nettoyé d'abord).

## Ressources

- [Documentation mypy](https://mypy.readthedocs.io/)
- [Type hints Python](https://docs.python.org/3/library/typing.html)