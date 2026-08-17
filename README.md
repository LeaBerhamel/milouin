# 🦆 Canard Detection - Workflow complet

## 📊 État actuel du projet

Architecture à 4 IA (voir [ANNOTATION_GUIDE.md](ANNOTATION_GUIDE.md) pour le détail
des règles d'annotation CVAT) :

```
Photo
  → YOLO détection (marque_nasale, canard)
      → crop marque_nasale → IA Couleur (bleu/marron/indeterminable)
      → crop canard        → IA Espèce (milouin/morillon)
                                → IA Sexe (male/femelle/indetermine_juvenile)
  → Matching temporel avec le tableur (filtré par espèce puis sexe si ambigu)
  → Si plusieurs canards détectés sur la même photo : chaque canard est
    filtré/matché INDIVIDUELLEMENT avec sa propre classification espèce/sexe
    (voir "Désambiguïsation multi-canards" ci-dessous)
```

- ✅ **IA 1 (Détection, YOLO)** : schéma à 2 classes (`marque_nasale`, `canard`)
- ✅ **IA 2 (Couleur)** : entraînée (`models/couleur/modele_couleur.pt`), 3 classes
  (`bleu`/`marron`/`indeterminable` — cette dernière couvre surtout les photos de
  nuit en infrarouge, sans information de couleur exploitable), accuracy 0.93 sur
  la validation. Voir "IA Couleur : comment la construire" ci-dessous si tu veux
  élargir le jeu de classes ou ré-entraîner sur plus de données.
- ✅ **IA 3 (Espèce)** : entraînée (`models/espece/modele_espece.pt`)
- ✅ **IA 4 (Sexe)** : entraînée (`models/sexe/modele_sexe.pt`)
- ✅ **Pipeline complet** : les 4 IA sont chargées en lazy loading et utilisées
  dès qu'on passe leur chemin en argument à `02_process_rafales.py` ; sans
  `--model-couleur`, le pipeline tourne quand même (juste sans classification
  de couleur)
- ✅ **Dictionnaire d'identification** (crop → bague) : export, relecture
  manuelle des non-identifiés, banque de référence visuelle, suggestions de
  correspondance et fusion multi-lots — voir "Étape 4" ci-dessous

Si tu as des annotations issues de l'ancien schéma à 5 classes
(`marque_nasale`/`canard_male`/`canard_femelle`/`milouin`/`morillon`),
utilise `scripts/00_migrate_annotations.py` pour les convertir automatiquement
vers le nouveau schéma (détection 2 classes + datasets de classification
espèce/sexe partiels).

---

## 📁 Structure du projet

```
canard-detection/
├── dataset/                    # Dataset pour IA 1 (détection)
├── dataset_espece/, dataset_sexe/   # Datasets pour IA classification
├── models/
│   ├── detection/best.pt         # Modèle YOLO entraîné
│   ├── couleur/modele_couleur.pt # IA Couleur (marque nasale)
│   ├── espece/modele_espece.pt   # IA Espèce
│   └── sexe/modele_sexe.pt     # IA Sexe
├── data/
│   ├── photos/                 # Toutes les photos brutes à traiter
│   └── observations.xlsx       # Tableur : DATE, HEURE, BAGUE, ESPECE, COULEUR_MARQUE, SEXE
├── scripts/
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── burst_detector.py   # Groupement des photos en rafales (EXIF)
│   │   ├── pipeline.py         # Détection + classification + matching
│   │   ├── confiance.py        # Calcul de la confiance globale (composite)
│   │   ├── annotate.py         # Dessin des bbox + tri des photos annotées
│   │   ├── synthese.py         # Chiffres agrégés du lot + graphique
│   │   ├── qualite.py          # Détection des incohérences tableur / IA
│   │   ├── journal.py          # Explication en clair des cas à vérifier
│   │   └── report.py           # Génération du rapport HTML consolidé
│   ├── interface_graphique.py          # Interface principale (lancée par Lancer.bat / Lancer.command)
│   ├── interface_preparation_ia.py     # Interface graphique : tri manuel + réentraînement (nouvelles photos + corrections)
│   ├── 00_migrate_annotations.py       # Migration des annotations (5 classes legacy -> 2 classes)
│   ├── 00_prepare_dataset_couleur.py   # Prépare le dataset couleur depuis les photos brutes
│   ├── 00_prepare_dataset_espece_sexe.py # Prépare les datasets espèce/sexe depuis les photos brutes
│   ├── 01_archiver_resultats.py        # Archive output/ existant avant un nouveau lot de photos
│   ├── 02_process_rafales.py           # Script principal (détection + classification + matching)
│   ├── 03_export_json_dictionnaire.py  # Résultats -> dictionnaire {crop: identité}
│   ├── 04_preparer_dossier_relecture.py# Dossier + zip des crops NON identifiés (relecteur humain)
│   ├── 04_copier_crops_valides.py      # Copie des crops IDENTIFIÉS -> crops_valides/<identifiant>/
│   ├── 05_construire_banque_reference.py # Empreintes visuelles par individu
│   ├── 06_proposer_correspondances_visuelles.py # Suggestions pour les non identifiés
│   ├── 07_fusionner_dictionnaire.py    # Fusionne un dictionnaire (nouveau lot ou archive) dans le maître
│   ├── 08_importer_identifications_manuelles.py # Importe des identifications faites à la main
│   ├── 09_ecran_identification.py      # Écran de relecture/correction avec suggestions visuelles
│   ├── 10_fusionner_corrections_dataset.py # Fusionne corrections.csv dans les datasets d'entraînement
│   ├── 11_reentrainer_modele.py        # Réentraîne un modèle (espèce/sexe/couleur) versionné
│   ├── 12_valider_et_deployer.py       # Valide un nouveau modèle avant déploiement en production
│   ├── 13_exporter_annotations_detection.py # Exporte les annotations de détection (relecture/audit)
│   ├── 14_bilan_matrice_confusion.py   # Matrice de confusion + bilan des identifications
│   ├── 15_fusionner_corrections.py     # Fusionne corrections.csv (+ crops) d'une source dans le maître
│   ├── gestion_modeles.py              # Versionnement / enregistrement des modèles entraînés
│   ├── nettoyer_non_identifies.py      # Nettoie les crops non identifiés obsolètes
│   ├── train_classification_ia.py      # Boucle d'entraînement générique (espèce/sexe/couleur)
│   └── triage_rapide.py                # Tri manuel rapide de crops par classe (ex. dataset couleur)
├── module_corrections/         # Fenêtre de correction (espèce/sexe/couleur, bagues), lancée depuis le rapport HTML
├── output/                     # Résultats générés (voir "Sorties" ci-dessous)
├── output_archives/
│   ├── <date>/                 # Un ancien output/ archivé par 01_archiver_resultats.py à chaque nouveau lot
│   └── dictionnaire_maitre/    # Dictionnaire + corrections cumulés de tous les lots (voir 4.6 et 4.7)
├── requirements.txt
└── README.md
```

---

## 🚀 Installation

```bash
pip install -r requirements.txt
```

---

## 🔧 Utilisation

### 🖱️ Option A — Interface graphique (recommandée, aucune ligne de commande)

1. Placer les photos dans `data/photos/`
2. Placer le tableur dans `data/observations.xlsx` (ou `.csv`)
3. Double-cliquer sur **`Lancer.bat`** à la racine du projet
   - Le lanceur installe automatiquement les dépendances manquantes au
     premier lancement (peut prendre quelques minutes), puis ouvre la
     fenêtre.
4. Dans la fenêtre : vérifier/choisir le dossier photos et le tableur, puis
   cliquer sur **"🚀 Lancer l'analyse"**.
5. À la fin, cliquer sur **"📄 Ouvrir le rapport"** pour voir les résultats
   dans le navigateur.

Les IA déjà entraînées (`models/<nom>/`) sont détectées et cochées
automatiquement ; celles qui manquent sont grisées (le traitement fonctionne
quand même sans elles).

#### 📥 Glisser-déposer (encore plus rapide)

Au lieu de placer les fichiers dans `data/` puis les re-choisir dans la
fenêtre, tu peux glisser directement ton dossier de photos (et le fichier
`.xlsx`/`.csv` d'observations, en même temps si tu veux) **sur l'icône
`Lancer.bat`**. La fenêtre s'ouvre avec les champs déjà pré-remplis.

Un simple double-clic sans rien déposer fonctionne toujours comme avant
(les champs restent à compléter à la main ou reprennent les valeurs par
défaut de `data/`).

### ⌨️ Option B — Ligne de commande (pour un usage scripté/automatisé)

### Étape 1 : Préparer les données

1. Placer les photos dans `data/photos/`
2. Placer le tableur dans `data/observations.xlsx` (ou `.csv`)
3. Placer les modèles entraînés dans `models/<nom>/`

### Étape 2 : Traiter les rafales

**PowerShell** — tout sur une seule ligne, ou avec le backtick `` ` `` en
fin de ligne pour continuer sur la ligne suivante :

```powershell
python scripts/02_process_rafales.py --photos data/photos --tableur data/observations.xlsx --model-detection models/detection/best.pt --model-couleur models/couleur/modele_couleur.pt --model-espece models/espece/modele_espece.pt --model-sexe models/sexe/modele_sexe.pt
```

Les IA de classification sont **optionnelles** : n'indique pas `--model-couleur` /
`--model-espece` / `--model-sexe` (ou ajoute `--skip-ia-couleur` / `--skip-ia-espece`
/ `--skip-ia-sexe`) pour ne lancer que la détection + matching temporel seul.

⚠️ Si un chemin de modèle est indiqué mais que le fichier n'existe pas, le
script plantera — ne passe que les modèles que tu as effectivement entraînés.

### Étape 3 : Analyser les résultats

Ouvre **`output/rapport.html`** dans un navigateur : c'est le point d'entrée —
vignettes des photos annotées, chiffres clés, tableau filtrable (Identifiées /
À vérifier / Non concluantes) et triable par colonne.

Le dossier `output/` contient aussi :

```
output/
├── rapport.html                 ← à ouvrir en premier
├── resultats_par_rafale.json    ← détail complet, machine-readable
├── resultats_par_photo.csv      ← une ligne par (photo, canard)
├── cas_ambigus.json             ← rafales à vérifier, format brut
├── journal_cas_limites.txt      ← EN LANGAGE CLAIR : pourquoi chaque cas coince
├── synthese_lot.csv             ← chiffres du lot (par espèce, sexe, statut...)
├── graphiques_synthese.png      ← histogrammes (statuts / espèces / sexes)
├── incoherences.csv             ← écarts tableur vs IA (absent si aucun écart)
└── photos_annotees/
    ├── identifies/               ← identification fiable, rien à faire
    ├── a_verifier/                ← ambigu, doublon, ou confiance faible
    └── non_concluant/              ← aucun canard détecté / pas de match tableur
```

Chaque photo annotée affiche, par canard détecté : la bague, l'espèce, le
sexe et la confiance globale, avec un code couleur (vert = confiance haute,
orange = moyenne, rouge = faible ou doublon).

---

## 📖 Étape 4 : Construire le dictionnaire d'identification (crop → bague)

`output/resultats_par_rafale.json` (Étape 2) contient déjà, pour chaque
marque nasale détectée, l'identité du canard **si** un match temporel a été
trouvé dans le tableur. Cette section transforme ces résultats bruts en un
**dictionnaire réutilisable** `{crop_filename: identité}`, avec une chaîne
d'outils pour boucher les trous : crops non identifiés à relire à la main,
banque de référence visuelle pour proposer des correspondances, et fusion
multi-lots.

⚠️ Cette partie n'a **pas d'interface graphique** pour l'instant : tout se
fait en ligne de commande (Option B).

```
resultats_par_rafale.json
        │
        ▼
03_export_json_dictionnaire.py
        │
        ├──► dictionnaire_identifications.json   (crops identifiés)
        └──► crops_non_identifies.csv            (crops sans match)
                    │                                     │
                    ▼                                     ▼
        04_copier_crops_valides.py          04_preparer_dossier_relecture.py
                    │                          (dossier + zip pour relecture humaine)
                    ▼
        05_construire_banque_reference.py
                    │
                    ▼
        06_proposer_correspondances_visuelles.py
          (suggestions à vérifier, jamais automatique)
```

### 4.1 Exporter le dictionnaire

```bash
python scripts/03_export_json_dictionnaire.py
```

Lit `output/resultats_par_rafale.json` et produit dans `output/` :
- **`dictionnaire_identifications.json`** : `{crop_filename: {identifiant, date, heure, sexe, espece}}`
  pour chaque marque nasale rattachée à un canard `IDENTIFIE`.
- **`crops_non_identifies.csv`** : une ligne par crop non rattaché, avec un
  `diagnostic` en clair (ex. *"Aucune ligne du tableur ne correspond"*).

### 4.2 (Optionnel) Préparer le dossier de relecture manuelle

```bash
python scripts/04_preparer_dossier_relecture.py
```

Copie uniquement les crops **non identifiés** dans
`output/a_identifier_manuellement/` (+ un zip prêt à envoyer), avec une
colonne `identifiant_manuel` vide à côté de chaque crop pour qu'un relecteur
humain y note la bague qu'il reconnaît à l'œil.

### 4.3 Copier les crops identifiés dans crops_valides/

```bash
python scripts/04_copier_crops_valides.py
```

Copie chaque crop du dictionnaire dans `output/crops_valides/<identifiant>/`
— un sous-dossier par individu. C'est cette base qui alimente la banque de
référence visuelle à l'étape suivante. **Ce dossier n'est jamais vidé** : le
relancer après un nouveau lot fusionné (voir 4.6) l'enrichit simplement.

### 4.4 Construire la banque de référence visuelle

```bash
python scripts/05_construire_banque_reference.py
```

Calcule une empreinte visuelle (vecteur ResNet18, 512 dimensions) par crop
confirmé de `crops_valides/` → `output/banque_reference/`. C'est la
"mémoire visuelle" des individus déjà identifiés.

### 4.5 Proposer des correspondances visuelles pour les non identifiés

```bash
python scripts/06_proposer_correspondances_visuelles.py
```

Compare chaque crop de `crops_non_identifies.csv` à la banque de référence
et propose jusqu'à 3 candidats par crop avec un score de similarité →
`output/propositions_visuelles.csv`.

⚠️ **Ce ne sont que des suggestions.** Un CNN générique pré-entraîné sur
ImageNet (jamais ré-entraîné sur vos marques nasales) donne un indice, pas
une preuve — deux marques ressemblantes peuvent être deux individus
différents. Chaque proposition est à vérifier photo à l'appui avant
correction du dictionnaire ; ne jamais écrire directement dans
`dictionnaire_identifications.json` à partir de ce fichier.

### 4.6 Ajouter de nouvelles photos au dictionnaire (nouveau lot)

`02_process_rafales.py` **écrase** `resultats_par_rafale.json` à chaque
lancement — si tu relances sur de nouvelles photos avec le même `--output`,
tu perds les identifications du lot précédent. `output/` reste donc
toujours le dossier de sortie par défaut (rien à changer, aucun nom de lot
à inventer), et c'est l'archivage qui protège le lot précédent avant de
traiter le suivant.

**Avec l'interface graphique (recommandé)** : rien à faire à la main. Le
bouton d'archivage (🗄️, accessible depuis le rapport HTML ou l'interface)
déplace `output/` existant vers `output_archives/<date>/`, recrée un
`output/` vide, et fusionne **automatiquement** ce lot fraîchement archivé
— dictionnaire d'identification **et** corrections (voir 4.7) — dans un
dossier maître cumulatif, `output_archives/dictionnaire_maitre/`, qui
n'est lui-même jamais archivé. Retraite les nouvelles photos normalement
par-dessus (toujours avec `output/` par défaut) : rien n'est perdu, et
"🔁 Réentraîner depuis les corrections…" (4.7) comme les étapes 4.3/4.4
retrouvent l'historique complet via ce maître.

**En ligne de commande pure** (sans passer par l'interface graphique) :

```bash
# 1. Archive output/ existant dans output_archives/<date>/ et recrée un output/ vide
python scripts/01_archiver_resultats.py

# 2. Traiter les nouvelles photos comme d'habitude, TOUJOURS avec le
#    dossier de sortie par défaut "output/"
python scripts/02_process_rafales.py \
    --photos data/photos --tableur data/observations.xlsx \
    --model-detection models/detection/best.pt \
    --model-couleur models/couleur/modele_couleur.pt \
    --model-espece models/espece/modele_espece.pt \
    --model-sexe models/sexe/modele_sexe.pt

python scripts/03_export_json_dictionnaire.py

# 3. Fusionner l'ANCIEN dictionnaire (celui qui vient d'être archivé) dans le maître
python scripts/07_fusionner_dictionnaire.py \
    --nouveau output_archives/<date>/dictionnaire_identifications.json

# 4. Idem pour les corrections manuelles de ce lot archivé (voir 4.7)
python scripts/15_fusionner_corrections.py \
    --source output_archives/<date> --maitre output

# 5. Régénérer les crops validés + la banque de référence à partir du maître à jour
python scripts/04_copier_crops_valides.py
python scripts/05_construire_banque_reference.py
```

Le chemin exact de l'archive à utiliser aux étapes 3 et 4 est affiché par
`01_archiver_resultats.py` au moment de l'archivage — copie-colle la ligne
affichée plutôt que de le retaper à la main.

**Règles de fusion** appliquées par `07_fusionner_dictionnaire.py` :
- une identification du nouveau lot est ajoutée au maître (ou remplace
  l'entrée existante si le même crop y était déjà) ;
- un crop marqué "non identifié" dans le maître mais identifié dans le
  nouveau lot (ex. le tableur a été complété entre-temps et tu as retraité
  les mêmes photos) est **retiré** des non-identifiés du maître ;
- un crop non identifié du nouveau lot n'est ajouté au maître que s'il n'y
  est pas déjà identifié.

Comme les noms de crop sont uniques par photo source (voir
`scripts/utils/crop_export.py`), fusionner plusieurs lots ne peut pas
mélanger deux crops différents sous le même nom.

### 4.7 Corriger les identifications manuellement et réentraîner l'IA (08 à 12, 15)

Quand l'IA se trompe (espèce, sexe ou couleur mal classés), les corrections
faites à la main peuvent servir à réentraîner les modèles concernés :

```bash
# 1. Relire/corriger les identifications à l'écran
python scripts/09_ecran_identification.py

# 2. (Optionnel) importer des identifications faites en dehors de cet écran
python scripts/08_importer_identifications_manuelles.py --source mes_corrections.csv

# 3. Fusionner les corrections dans les datasets d'entraînement
python scripts/10_fusionner_corrections_dataset.py --corrections output/corrections.csv

# 4. Réentraîner un modèle (versionné, ex. v2) à partir du dataset enrichi
python scripts/11_reentrainer_modele.py --type espece --version v2 --epochs 10

# 5. Valider le nouveau modèle avant de le déployer (jamais automatique)
python scripts/12_valider_et_deployer.py \
    --nouveau models/espece/v2/modele_espece.pt \
    --actuel models/espece/modele_espece.pt \
    --test-dir dataset_espece_test
```

`09_ecran_identification.py` propose des suggestions visuelles cliquables
pendant la relecture et écrit les corrections dans `output/corrections.csv`
(colonnes `espece_corrigee`, `sexe_corrige`, `couleur_corrigee`).

**Les corrections d'un lot archivé ne sont pas perdues** : comme pour le
dictionnaire (4.6), `output/corrections.csv` suit `output/` à chaque
archivage. `scripts/15_fusionner_corrections.py --source <archive>
--maitre <dossier>` les intègre dans un `corrections.csv` cumulatif, en
retrouvant chaque image par son nom de fichier plutôt que par le chemin
enregistré (devenu obsolète après l'archivage). Avec l'interface
graphique, cette fusion a lieu automatiquement en même temps que celle du
dictionnaire (voir 4.6) — rien à lancer à la main.

**Plus simple : tout ça est accessible depuis l'interface graphique**, via
le bouton "🔁 Réentraîner depuis les corrections…" (écran `EcranCorrections`
dans `scripts/interface_preparation_ia.py`). L'interface analyse un
fichier de corrections (par défaut `output_archives/dictionnaire_maitre/corrections.csv`,
le maître cumulé — donc les corrections des lots déjà archivés y sont déjà,
pas seulement celles du lot courant), affiche combien de corrections sont
exploitables pour chacun des 3 types (espèce/sexe/couleur), et ne propose
que les modèles qui ont effectivement des corrections disponibles. Deux
façons de lancer, une fois les types choisis :
- **🚀 Fusionner + entraîner ici** : fusionne puis entraîne directement sur
  cet ordinateur (nécessite PyTorch installé localement) ;
- **📦 Fusionner puis préparer un zip pour Colab** : fusionne puis prépare
  un zip du dataset à uploader dans un notebook Colab (pas d'entraînement
  local dans ce cas) — utile si l'ordinateur n'a pas de carte graphique ou
  pas PyTorch installé.

Dans les deux cas la fusion a lieu d'abord, systématiquement : un zip
préparé pour Colab contient donc toujours les corrections qui viennent
d'être cochées, jamais un dataset périmé.

`12_valider_et_deployer.py` ne déploie **jamais automatiquement** : le
nouveau modèle n'est promu en production que si ses métriques dépassent un
seuil minimal configurable sur un jeu de test dédié.

---

## 🦆 Désambiguïsation multi-canards

Quand plusieurs canards sont détectés sur une même photo, **chacun est classifié
et matché individuellement** (son propre crop → sa propre espèce/sexe → son
propre matching temporel dans le tableur), plutôt que d'abandonner en bloc.

Résultat possible pour une rafale à plusieurs canards :
- **IDENTIFIE_MULTIPLE** : tous les canards ont été résolus individuellement
- **PARTIELLEMENT_IDENTIFIE** : certains oui, d'autres non
- **MULTIPLE_CANARDS** : aucun n'a pu être résolu automatiquement
- Un canard individuel peut aussi être marqué **AMBIGU_DOUBLON** si deux
  canards de la même photo ont été appariés à la même bague (signe que la
  classification espèce/sexe n'a pas suffi à les distinguer)

## 🎨 IA Couleur : comment la (re)construire

L'IA Couleur est entraînée (`models/couleur/modele_couleur.pt`, 3 classes
`bleu`/`marron`/`indeterminable`). Cette section décrit le circuit complet, utile
si tu veux ré-entraîner sur plus de données ou élargir le jeu de couleurs
reconnues (ex. ajouter jaune/vert/rouge si ces marques apparaissent sur le
terrain) :

**1. Extraire les crops de marques nasales depuis tes photos**

```bash
python scripts/00_prepare_dataset_couleur.py --photos data/photos --model-detection models/detection/best.pt
```

Ça crée `dataset_couleur/crops/` avec un crop par marque nasale détectée, plus
`dataset_couleur/metadata.json`. Plus tu donnes de photos en entrée (idéalement
plusieurs centaines, couvrant toutes les couleurs utilisées sur le terrain), mieux
le futur modèle généralisera — un dataset de moins d'une vingtaine d'images par
couleur ne donnera pas un modèle fiable.

**2. Trier les crops par couleur, à l'œil**

```bash
python scripts/triage_rapide.py --crops dataset_couleur/crops --classes jaune vert rouge bleu noir blanc orange
```

Ouvre chaque crop en grand et déplace-le dans le bon dossier couleur d'une simple
touche (1 à 7 selon le nombre de classes passées à `--classes`). C'est un tri
manuel volontairement : les crops sont petits et peuvent prêter à confusion
(jaune/orange, noir/bleu foncé selon la luminosité), seul un œil humain —
éventuellement aidé du tableur d'observations pour recouper — peut trancher de
façon fiable.

Pense aussi à ajouter une classe `indeterminable` pour les crops sans couleur
exploitable (essentiellement les photos de nuit en infrarouge, quasi grises) :
mélanger ces cas avec de vraies couleurs perturbe l'entraînement plus qu'autre
chose. C'est la classe `--classes bleu marron indeterminable` utilisée pour le
modèle actuel — adapte la liste à tes propres couleurs de terrain.

**3. Entraîner le modèle sur Google Colab**

Ouvre `models/couleur/IA_couleur.ipynb` dans [Google Colab](https://colab.research.google.com/)
(gratuit, pas d'installation), suis les instructions dans le notebook (compresser
`dataset_couleur` trié en zip, l'uploader, lancer les cellules dans l'ordre). Il
te renverra `modele_couleur.pt` à placer dans `models/couleur/`. Ensuite, ajoute
simplement `--model-couleur models/couleur/modele_couleur.pt` à la commande du
pipeline (voir "Étape 2" ci-dessus) — plus besoin de `--skip-ia-couleur`.

---

## 🎯 Comment est calculée la confiance affichée

`confiance_globale` est une moyenne pondérée de 3 composantes disponibles :

| Composante | Poids | Signification |
|---|---|---|
| Temporelle | 0.50 | Qualité du match dans le tableur (0.95 si match ≤30s et univoque, 0.85 si ≤180s, 0.90/0.75 pour les mêmes cas en contexte multi-canards) |
| Espèce | 0.25 | Probabilité softmax de l'IA Espèce sur le crop du canard |
| Sexe | 0.25 | Probabilité softmax de l'IA Sexe sur le crop du canard |

Si une IA de classification n'a pas été utilisée, son poids est retiré et les
poids restants sont renormalisés (la confiance temporelle seule redevient
alors 100 % du score). Formule et détails : `scripts/utils/confiance.py`.

---

## 📋 Format du tableur

Le fichier `data/observations.xlsx` (ou `.csv`) doit contenir :

```csv
DATE,HEURE,BAGUE,ESPECE,COULEUR_MARQUE,SEXE
2024-03-15,14:23:45,A12345,Colvert,Jaune,M
2024-03-15,14:25:12,B67890,Souchet,Vert,F
```

**Colonnes obligatoires** : DATE, HEURE, BAGUE, ESPECE, COULEUR_MARQUE
**Colonne optionnelle** : SEXE — si présente, utilisée pour affiner le matching
temporel en cas d'ambiguïté (voir `match_exact_time`).

---

## 📊 Statuts possibles

| Statut | Signification |
|---|---|
| `IDENTIFIE` | Un seul canard détecté, identifié automatiquement |
| `IDENTIFIE_MULTIPLE` | Plusieurs canards détectés, tous identifiés individuellement |
| `PARTIELLEMENT_IDENTIFIE` | Plusieurs canards détectés, certains identifiés, d'autres non |
| `AUCUN_CANARD` | Pas de canard détecté sur la photo |
| `AMBIGU` | Un seul canard, mais plusieurs observations du tableur correspondent encore au créneau (même après filtrage espèce/sexe) |
| `MULTIPLE_CANARDS` | Plusieurs canards détectés, aucun résolu automatiquement |
| `NON_TROUVE_TABLEUR` | Aucune observation correspondante dans le tableur |

Statuts à l'échelle d'un canard individuel (dans `resultats_par_rafale.json`,
clé `canards`) : `IDENTIFIE`, `NON_RESOLU`, `AMBIGU_DOUBLON`.

---

## 🐛 Dépannage

### Erreur "No module named 'utils'"
Vérifier que `scripts/utils/__init__.py` existe, et lancer la commande depuis
la racine du projet (`python scripts/02_process_rafales.py ...`), pas depuis
le dossier `scripts/`.

### Erreur "EXIF data not found"
Certaines photos n'ont pas de métadonnées EXIF, elles seront ignorées.

### PowerShell : "unrecognized arguments"
C'est un souci de continuation de ligne : `\` est la syntaxe bash, PowerShell
utilise `` ` `` (sans espace après). Le plus simple reste de mettre toute la
commande sur une seule ligne.

### `ModuleNotFoundError: No module named 'matplotlib'`
`pip install matplotlib` (nécessaire pour `graphiques_synthese.png`).

### Performances lentes
Réduire le nombre de photos par lot, ou utiliser un GPU.