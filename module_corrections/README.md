# module_corrections — prototype isolé

Ce dossier est **complètement autonome** : rien dans `scripts/` n'a été
modifié. Il implémente et teste la partie technique la plus nouvelle
(serveur local + pont vers Tkinter) avant de la brancher pour de vrai.

## Fichiers

| Fichier | Rôle |
|---|---|
| `serveur_corrections.py` | Serveur HTTP local (127.0.0.1 uniquement) qui reçoit les clics "Corriger" du rapport et les pousse dans une queue thread-safe |
| `formulaire_correction.py` | Le `Toplevel` à 4 champs (bague, espèce, sexe, couleur) — réutilisable pour le clic ponctuel *et* pour le futur écran batch (héritier de `09_ecran_identification.py`) |
| `ecrire_corrections.py` | Écrit dans `corrections.csv` (uniquement les champs changés) et `dictionnaire_identifications.json` |
| `demo_standalone.py` | Démo bout-en-bout, jouable telle quelle |

## Tester

```
python module_corrections/demo_standalone.py
```

Ça ouvre un faux rapport dans le navigateur + une petite fenêtre Tkinter
("Suivi Canards (démo)") qui doit rester ouverte. Clique sur un lien
"✏️ Corriger" sous un des 3 crops factices → une fenêtre de correction
doit apparaître au premier plan. Valide → regarde le contenu généré dans
`module_corrections/dictionnaire_demo/` (recréé à chaque lancement, sans
toucher au vrai `dictionnaire/` du dépôt).

*(Testé ici sans écran : la partie serveur + queue + écriture est
validée par un test automatisé sans Tkinter — voir historique de la
conversation. La partie purement visuelle du formulaire, elle, n'a pu
être vérifiée que par lecture du code — à tester en vrai sur une machine
avec affichage avant de considérer que c'est terminé.)*

## Ce qui reste à faire pour l'intégration réelle

Trois branchements dans le code existant, aucun autre changement :

**1. `scripts/interface_graphique.py`**
- Démarrer le serveur une fois, dans `_traitement_termine` (ou `__init__`
  si on veut qu'il tourne dès l'ouverture de l'appli) :
  `self.serveur_corrections, self.port_corrections = demarrer_serveur(self.log_queue)`
  → **réutiliser `self.log_queue` existante**, pas une nouvelle queue : le
  handler pousse déjà des tuples `(type, contenu)`, ajouter juste un type
  `"correction"` de plus au `if/elif` de `_traiter_queue`.
- Dans `_traiter_queue`, ajouter le cas :
  `elif type_msg == "correction": ouvrir_formulaire_correction(self, contenu, self._on_correction_validee)`
- `_on_correction_validee` appelle `traiter_validation_formulaire(dossier_dictionnaire, valeurs)`.
- Arrêt propre : `self.serveur_corrections.shutdown()` dans le handler de
  fermeture de fenêtre (`protocol("WM_DELETE_WINDOW", ...)`, à vérifier
  s'il existe déjà).

**2. `scripts/utils/report.py`**
- Passer `port_corrections` en paramètre de la fonction qui génère le
  tableau.
- Pour chaque ligne, ajouter une cellule avec le lien produit par
  `construire_lien_correction(port, crop_filename, espece=..., sexe=..., couleur=..., bague=...)`.

**3. `scripts/07_fusionner_dictionnaire.py`**
- `ecrire_corrections.enregistrer_bague()` est **volontairement
  simplifié** pour la démo (mapping direct crop → bague). La vraie
  gestion des doublons / bagues `ILLISIBLE` / suggestions déjà proposées
  vit dans ce script — à l'intégration, appeler sa logique plutôt que de
  la dupliquer.

## Bug corrigé au passage : `COULEURS_CLASSES` manquant

`utils/pipeline.py` importait `COULEURS_CLASSES`, `ESPECES_CLASSES` et
`SEXES_CLASSES` depuis `utils/config.py` — **aucune des trois n'y était
définie**. `ImportError` garanti dès le premier accès à
`.ia_couleur`/`.ia_espece`/`.ia_sexe` (donc dès qu'un modèle IA couleur/
espèce/sexe est réellement utilisé en inférence).

Corrigé directement dans `scripts/utils/config.py` (voir commentaire
ajouté sur place pour le détail). Valeurs déduites du reste du dépôt :

| Constante | Valeur | Source |
|---|---|---|
| `ESPECES_CLASSES` | `["milouin", "morillon"]` | `ESPECE_CODE_TABLEUR`, `00_migrate_annotations.py` |
| `SEXES_CLASSES` | `["male", "femelle", "indetermine_juvenile"]` | `SEXE_CODE_TABLEUR` |
| `COULEURS_CLASSES` | `["bleu", "marron", "indeterminable"]` | `triage_rapide.py` (`COLOR_KEYS`) |

Important à retenir (déjà documenté dans `_load_classifier()`, mais facile
à oublier) : ces listes ne sont qu'un **filet de sécurité**. L'ordre réel
des classes utilisé pour l'inférence vient du fichier `classes_*.json`
sauvegardé à côté de chaque modèle — ces constantes ne servent que si ce
fichier est absent.

`module_corrections/formulaire_correction.py` importe maintenant
`ESPECES_CLASSES`/`SEXES_CLASSES`/`COULEURS_CLASSES` **directement depuis
`scripts/utils/config.py`** (avec un filet de sécurité en dur si le
dossier est testé isolément, sans `scripts/` à côté) — une seule source de
vérité, partagée avec `utils/pipeline.py`, au lieu d'une copie manuelle
qui aurait pu diverger avec le temps.
