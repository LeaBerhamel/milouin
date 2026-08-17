# Guide d'annotation CVAT — Canard Detection

Ce document définit le schéma d'annotation à utiliser dans CVAT pour toute nouvelle
annotation, et sert de référence pour la migration des annotations existantes
(voir `scripts/00_migrate_annotations.py`).

## Architecture cible

```
Photo
  → YOLO détection (marque_nasale, canard)
      → crop marque_nasale → IA Couleur (jaune/vert/rouge/...)
      → crop canard        → IA Espèce (milouin/morillon)
                                → IA Sexe (male/femelle/indetermine_juvenile)
  → Matching temporel avec le tableur (filtré par espèce puis sexe si ambigu)
```

Une seule IA de détection (YOLO) trouve les objets sur la photo. Tout le reste
(espèce, sexe, couleur de marque) est déterminé par des IAs de classification
séparées qui tournent sur les crops extraits par la détection — ce ne sont
**pas** des classes YOLO.

## Classes CVAT (bounding boxes)

Deux classes seulement :

| Classe          | Description                                   |
|------------------|------------------------------------------------|
| `marque_nasale`  | La marque nasale colorée portée par le canard  |
| `canard`         | Le canard entier (corps + tête)                |

### Attributs de la classe `canard`

| Attribut | Valeurs possibles                          | Obligatoire ? |
|----------|---------------------------------------------|----------------|
| `espece` | `milouin`, `morillon`                       | Oui, si visible |
| `sexe`   | `male`, `femelle`, `indetermine_juvenile`   | Oui |

La classe `marque_nasale` n'a pas d'attribut : sa couleur est traitée par
l'IA Couleur en aval, pas annotée manuellement dans CVAT.

## Règles d'annotation

1. **Espèce toujours renseignée si visible.** Même si le sexe ne peut pas être
   déterminé, renseigner `espece` dès que l'espèce est identifiable sur la photo.
2. **Sexe uniquement si dimorphisme visible et individu adulte.** Le
   dimorphisme sexuel (plumage, couleur de l'œil, etc.) n'est fiable que chez
   l'adulte. Ne pas deviner.
3. **Juvénile → `sexe = indetermine_juvenile`.** Dès qu'un individu est
   identifié comme juvénile, mettre `sexe` à `indetermine_juvenile`,
   indépendamment de ce que son plumage pourrait suggérer. `espece` reste
   renseignée normalement si visible.
4. **Une bounding box `marque_nasale` par marque visible**, même si le canard
   associé n'est pas entièrement visible ou pas annoté.
5. **Un canard non identifiable** (trop loin, flou, coupé par le cadre) :
   annoter quand même la classe `canard` si le corps est reconnaissable, mais
   laisser `espece` et `sexe` non renseignés plutôt que de deviner.

## Historique — ancien schéma (5 classes, déprécié)

Les annotations faites avant cette refonte utilisaient 5 classes distinctes :
`marque_nasale`, `canard_male`, `canard_femelle`, `milouin`, `morillon`.
Ce schéma mélangeait détection et classification dans les classes YOLO
elles-mêmes, ce qui empêchait par exemple de distinguer une femelle milouin
d'un mâle morillon avec une seule classe.

Ces annotations existantes ne sont pas perdues : le script
`scripts/00_migrate_annotations.py` les convertit automatiquement vers le
nouveau schéma (voir sa docstring pour le mapping exact).
