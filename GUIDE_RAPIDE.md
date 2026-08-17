# Suivi Canards — Démarrage rapide

*Ce guide ne suppose aucune connaissance en informatique.*

## Ce dont tu as besoin
- Un ordinateur Windows
- Tes photos de canards, dans un dossier
- Ton fichier d'observations (Excel), si tu en as un

## Les 3 étapes

### 1. Mets tes fichiers au bon endroit
- Glisse tes photos dans le dossier `data\photos`
- Glisse ton fichier Excel dans le dossier `data` (garde son nom tel quel,
  ou renomme-le `observations.xlsx`)

*(Astuce : tu peux aussi sauter cette étape et glisser directement tes
fichiers sur l'icône `Lancer.bat` à l'étape 2 — voir plus bas.)*

### 2. Double-clique sur `Lancer.bat`
C'est le fichier à l'icône 🦆 à la racine du dossier du projet.

- **La toute première fois**, une fenêtre noire s'ouvre et installe ce
  dont le programme a besoin. **Ça peut prendre 10 à 20 minutes** — c'est
  normal, ne ferme pas la fenêtre même si elle semble ne plus bouger par
  moments.
- **Les fois suivantes**, ça s'ouvre en quelques secondes.

*(Astuce glisser-déposer : au lieu de l'étape 1, tu peux directement
déposer ton dossier de photos — et ton fichier Excel — sur l'icône
`Lancer.bat`. Les champs se remplissent tout seuls.)*

### 3. Clique sur "🚀 Lancer l'analyse"
Une fenêtre s'ouvre avec tes dossiers déjà indiqués. Vérifie qu'ils sont
corrects, puis clique sur le bouton. Une barre de progression te montre où
en est le traitement.

À la fin, clique sur **"📄 Ouvrir le rapport"** : ça ouvre un tableau dans
ton navigateur avec toutes les photos analysées, triables et filtrables.

---

## Si quelque chose ne marche pas

**"Python n'est pas installé"**
→ Va sur https://www.python.org/downloads/, installe Python, et
**coche bien la case "Add python.exe to PATH"** pendant l'installation
(c'est l'erreur la plus fréquente). Puis recommence l'étape 2.

**La fenêtre se ferme toute seule / message d'erreur**
→ Une fenêtre noire (terminal) reste normalement ouverte avec le message
d'erreur en rouge. Fais une capture d'écran de ce message si tu demandes
de l'aide — c'est ce qui permet de comprendre le problème.

**L'installation semble bloquée**
→ Regarde s'il y a du texte qui continue de défiler dans la fenêtre noire
(ça veut dire que ça travaille toujours). Une vraie coupure ne montre
plus rien bouger pendant plusieurs minutes ET affiche un message
`[ERREUR]`.

---

## Pour aller plus loin
Ce guide couvre juste le lancement d'une analyse. Pour tout le reste
(gérer plusieurs lots de photos, identifier les marques manquantes,
ré-entraîner les IA...), voir le `README.md` complet à la racine du projet.

---

> 📸 **Note pour la personne qui finalise ce guide** : ajouter ici 2-3
> captures d'écran (fenêtre de lancement, bouton "Lancer l'analyse",
> rapport ouvert dans le navigateur) — impossible à générer sans lancer le
> programme sur une vraie machine Windows.
