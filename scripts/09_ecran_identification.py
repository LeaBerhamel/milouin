#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Écran d'identification manuelle avec suggestions visuelles.

Remplace la relecture "à l'aveugle" (Excel/LibreOffice) par une fenêtre qui
affiche, pour chaque crop non identifié, le crop à identifier ET jusqu'à 3
individus déjà connus qui lui ressemblent visuellement (calculés par
scripts/06_proposer_correspondances_visuelles.py). Un clic sur une
suggestion PRÉ-REMPLIT le champ texte, il ne valide RIEN tout seul -- il
faut toujours cliquer "Valider" (ou Entrée) pour enregistrer, exactement
comme si la bague avait été tapée à la main.

⚠️ Les suggestions sont un INDICE (similarité visuelle par CNN générique,
jamais ré-entraîné sur vos marques), pas une preuve -- voir l'avertissement
en permanence à l'écran, et scripts/06_proposer_correspondances_visuelles.py.

Prérequis :
    python scripts/04_preparer_dossier_relecture.py
    python scripts/06_proposer_correspondances_visuelles.py   (optionnel --
        si absent, l'écran fonctionne quand même, juste sans suggestions)

Le CSV édité est celui produit par 04_preparer_dossier_relecture.py
(colonne "identifiant_manuel"), directement compatible avec :
    python scripts/08_importer_identifications_manuelles.py --csv <ce_csv>

Raccourcis clavier :
    1 / 2 / 3   -> pré-remplir avec la suggestion correspondante (pas de validation)
    Entrée      -> valider le contenu actuel du champ texte comme vraie bague
    I           -> marquer ILLISIBLE
    P           -> marquer PAS_MARQUE (fausse détection)
    Espace      -> passer (laisse vide, réapparaîtra plus tard)
    Retour ⌫    -> revenir au crop précédent

Usage :
    python scripts/09_ecran_identification.py
    python scripts/09_ecran_identification.py --csv output/a_identifier_manuellement/crops_non_identifies.csv
"""

import argparse
import csv
import subprocess
import sys
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

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
from utils.config import BASE_DIR  # noqa: E402
from utils.propositions import charger_propositions, libelle_pour_score  # noqa: E402

BG = "#222222"
FG = "white"
BLEU = "#4C8BF5"
VERT = "#3fae5b"
ROUGE = "#c0392b"
GRIS = "#555555"

TAILLE_CROP_PRINCIPAL = 320
TAILLE_CROP_SUGGESTION = 140


def lire_lignes_et_entetes(path: Path):
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)
    return rows, fieldnames


def sauvegarder_csv(path: Path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class EcranIdentification(tk.Tk):
    def __init__(self, csv_path, crops_dir, propositions_path, crops_valides_dir):
        super().__init__()
        self.title("Identification des crops -- SuiviCanards")
        self.configure(bg=BG)
        self.geometry("1000x720")

        self.csv_path = csv_path
        self.crops_dir = Path(crops_dir)
        self.crops_valides_dir = Path(crops_valides_dir)

        self.rows, self.fieldnames = lire_lignes_et_entetes(self.csv_path)
        if "identifiant_manuel" not in (self.fieldnames or []):
            messagebox.showerror(
                "Fichier inattendu",
                f"'identifiant_manuel' absent de {self.csv_path}\n\n"
                "Ce n'est pas un CSV produit par 04_preparer_dossier_relecture.py ?")
            self.destroy()
            return

        self.propositions = charger_propositions(propositions_path)

        # Seulement les lignes pas encore relues
        self.todo = [i for i, r in enumerate(self.rows) if not (r.get("identifiant_manuel") or "").strip()]
        self.historique = []
        self.position = 0  # index dans self.todo
        self.photos_refs = []  # garde une référence forte aux ImageTk.PhotoImage affichées

        self._construire_interface()
        if not self.todo:
            self._afficher_tout_traite()
        else:
            self._afficher_courant()

    # -- construction --

    def _construire_interface(self):
        pad = {"padx": 14, "pady": 6}

        self.lbl_compteur = tk.Label(self, bg=BG, fg=FG, font=("Helvetica", 11, "bold"))
        self.lbl_compteur.pack(anchor="w", **pad)

        tk.Label(
            self, bg=BG, fg="#e0a030", font=("Helvetica", 9, "italic"),
            text="⚠️ Les suggestions sont un indice de ressemblance visuelle, "
                 "PAS une preuve d'identité -- à vérifier photo à l'appui.",
        ).pack(anchor="w", padx=14)

        corps = tk.Frame(self, bg=BG)
        corps.pack(fill="both", expand=True, padx=14, pady=6)

        # -- colonne gauche : crop à identifier --
        cadre_crop = tk.Frame(corps, bg=BG)
        cadre_crop.pack(side="left", fill="y", padx=(0, 20))
        tk.Label(cadre_crop, text="Crop à identifier", bg=BG, fg=FG,
                 font=("Helvetica", 10, "bold")).pack(anchor="w")
        self.lbl_crop_principal = tk.Label(cadre_crop, bg="#111111")
        self.lbl_crop_principal.pack(pady=6)
        self.lbl_meta = tk.Label(cadre_crop, bg=BG, fg="#bbbbbb", justify="left", font=("Helvetica", 9))
        self.lbl_meta.pack(anchor="w")

        # -- colonne droite : suggestions --
        cadre_droite = tk.Frame(corps, bg=BG)
        cadre_droite.pack(side="left", fill="both", expand=True)
        tk.Label(cadre_droite, text="Individus déjà connus qui ressemblent",
                 bg=BG, fg=FG, font=("Helvetica", 10, "bold")).pack(anchor="w")
        self.cadre_suggestions = tk.Frame(cadre_droite, bg=BG)
        self.cadre_suggestions.pack(fill="x", pady=6)

        # -- champ manuel, toujours visible et prioritaire --
        cadre_saisie = tk.Frame(self, bg=BG)
        cadre_saisie.pack(fill="x", padx=14, pady=(10, 4))
        tk.Label(cadre_saisie, text="Identifiant (bague) :", bg=BG, fg=FG).pack(side="left")
        self.var_saisie = tk.StringVar()
        self.entree = tk.Entry(cadre_saisie, textvariable=self.var_saisie, font=("Helvetica", 12), width=20)
        self.entree.pack(side="left", padx=8)
        self.entree.bind("<Return>", lambda e: self._valider())

        # -- boutons --
        cadre_boutons = tk.Frame(self, bg=BG)
        cadre_boutons.pack(fill="x", padx=14, pady=(4, 12))
        tk.Button(cadre_boutons, text="✓ Valider  (Entrée)", bg=VERT, fg="white",
                  font=("Helvetica", 10, "bold"), command=self._valider).pack(side="left", padx=4)
        tk.Button(cadre_boutons, text="Illisible  (I)", bg=GRIS, fg="white",
                  command=self._illisible).pack(side="left", padx=4)
        tk.Button(cadre_boutons, text="Pas une marque  (P)", bg=GRIS, fg="white",
                  command=self._pas_marque).pack(side="left", padx=4)
        tk.Button(cadre_boutons, text="Passer  (Espace)", bg=GRIS, fg="white",
                  command=self._passer).pack(side="left", padx=4)
        tk.Button(cadre_boutons, text="← Précédent  (⌫)", bg=GRIS, fg="white",
                  command=self._precedent).pack(side="left", padx=4)
        
        # Bouton Étape 5 : Réentraînement IA
        tk.Button(cadre_boutons, text="🔄 Réentraîner l'IA", bg=BLEU, fg="white",
                  font=("Helvetica", 10, "bold"), command=self._lancer_reentrainement_ia).pack(side="left", padx=12)

        self.lbl_progression = tk.Label(cadre_boutons, bg=BG, fg="#888888")
        self.lbl_progression.pack(side="right")

        # raccourcis clavier globaux
        self.bind("<Key-1>", lambda e: self._choisir_suggestion(0))
        self.bind("<Key-2>", lambda e: self._choisir_suggestion(1))
        self.bind("<Key-3>", lambda e: self._choisir_suggestion(2))
        self.bind("<space>", self._espace_si_pas_saisie)
        self.bind("<BackSpace>", self._retour_si_pas_saisie)
        self.bind("<i>", self._i_si_pas_saisie)
        self.bind("<p>", self._p_si_pas_saisie)

    # -- garde-fous pour ne pas voler les touches pendant la frappe --

    def _dans_le_champ_texte(self):
        return self.focus_get() is self.entree

    def _espace_si_pas_saisie(self, event):
        if not self._dans_le_champ_texte():
            self._passer()

    def _retour_si_pas_saisie(self, event):
        if not self._dans_le_champ_texte():
            self._precedent()

    def _i_si_pas_saisie(self, event):
        if not self._dans_le_champ_texte():
            self._illisible()

    def _p_si_pas_saisie(self, event):
        if not self._dans_le_champ_texte():
            self._pas_marque()

    # -- affichage --

    def _row_courante(self):
        return self.rows[self.todo[self.position]]

    def _afficher_courant(self):
        if self.position >= len(self.todo):
            self._afficher_tout_traite()
            return

        row = self._row_courante()
        crop = row["crop_filename"]
        self.var_saisie.set("")
        self.photos_refs.clear()

        n_restant = len(self.todo) - self.position
        self.lbl_compteur.config(text=f"{crop}   —   {n_restant} crop(s) restant(s) à identifier")
        self.lbl_progression.config(text=f"{self.position + 1} / {len(self.todo)}")
        self.lbl_meta.config(text=(
            f"Date : {row.get('date', '?')}    Heure : {row.get('heure', '?')}\n"
            f"Espèce détectée : {row.get('espece_detectee', '?')}    "
            f"Sexe détecté : {row.get('sexe_detectee', '?')}"
        ))

        self._afficher_image(self.lbl_crop_principal, self.crops_dir / crop, TAILLE_CROP_PRINCIPAL)
        self._afficher_suggestions(crop)
        self.entree.focus_set()

    def _afficher_image(self, label_widget, path: Path, taille):
        if not path.exists():
            label_widget.config(image="", text="⚠️ image introuvable", fg=FG, bg="#111111")
            return
        img = Image.open(path)
        img.thumbnail((taille, taille))
        photo = ImageTk.PhotoImage(img)
        self.photos_refs.append(photo)
        label_widget.config(image=photo, text="")

    def _afficher_suggestions(self, crop):
        for w in self.cadre_suggestions.winfo_children():
            w.destroy()

        candidats, statut = self.propositions.get(crop, ([], "AUCUNE_REFERENCE"))
        if not candidats:
            tk.Label(
                self.cadre_suggestions, bg=BG, fg="#999999", justify="left",
                text="Aucune correspondance visuelle trouvée dans la banque de référence.\n"
                     "(banque encore vide, ou aucun individu connu ne ressemble assez --\n"
                     "identifie ce crop directement dans le champ ci-dessous.)",
            ).pack(anchor="w", padx=4, pady=10)
            return

        for i, (identifiant, crop_ref, score) in enumerate(candidats):
            cellule = tk.Frame(self.cadre_suggestions, bg="#333333", cursor="hand2")
            cellule.pack(side="left", padx=6, pady=4)

            img_label = tk.Label(cellule, bg="#333333")
            img_label.pack(padx=6, pady=(6, 2))
            ref_path = self.crops_valides_dir / identifiant / crop_ref
            self._afficher_image(img_label, ref_path, TAILLE_CROP_SUGGESTION)

            tk.Label(cellule, text=f"[{i + 1}]  {identifiant}", bg="#333333", fg="white",
                     font=("Helvetica", 10, "bold")).pack()

            libelle = libelle_pour_score(score)
            tk.Label(cellule, text=libelle[0], bg="#333333", fg=libelle[1],
                     font=("Helvetica", 9)).pack(pady=(0, 8))

            for widget in (cellule, img_label):
                widget.bind("<Button-1>", lambda e, idx=i: self._choisir_suggestion(idx))

    def _afficher_tout_traite(self):
        for w in self.winfo_children():
            w.destroy()
        tk.Label(self, bg=BG, fg=FG, font=("Helvetica", 13, "bold"),
                 text="✅ Tous les crops de ce fichier sont traités.").pack(pady=30)
        tk.Label(self, bg=BG, fg="#bbbbbb", justify="left", text=(
            f"Réintègre maintenant avec :\n\n"
            f"    python scripts/08_importer_identifications_manuelles.py --csv {self.csv_path}"
        )).pack()

    # -- actions --

    def _choisir_suggestion(self, index):
        if self.position >= len(self.todo):
            return
        row = self._row_courante()
        crop = row["crop_filename"]
        candidats, _ = self.propositions.get(crop, ([], ""))
        if index >= len(candidats):
            return
        identifiant = candidats[index][0]
        self.var_saisie.set(identifiant)
        self.entree.focus_set()
        self.entree.icursor("end")

    def _ecrire_valeur(self, valeur):
        row = self._row_courante()
        row["identifiant_manuel"] = valeur
        sauvegarder_csv(self.csv_path, self.rows, self.fieldnames)
        self.historique.append(self.position)
        self.position += 1
        self._afficher_courant()

    def _valider(self):
        valeur = self.var_saisie.get().strip()
        if not valeur:
            messagebox.showinfo("Champ vide", "Tape une bague, clique une suggestion, ou utilise "
                                              "Illisible / Pas une marque / Passer.")
            return
        self._ecrire_valeur(valeur)

    def _illisible(self):
        self._ecrire_valeur("ILLISIBLE")

    def _pas_marque(self):
        self._ecrire_valeur("PAS_MARQUE")

    def _passer(self):
        self.historique.append(self.position)
        self.position += 1
        self._afficher_courant()

    def _precedent(self):
        if not self.historique:
            return
        self.position = self.historique.pop()
        self._afficher_courant()

    # -- réentraînement IA (Étape 5) --

    def _lancer_reentrainement_ia(self):
        reponse = messagebox.askyesno(
            "Réentraînement de l'IA",
            "Voulez-vous réentraîner l'IA avec les corrections actuelles ?\n\n"
            "Cette procédure va :\n"
            "1. Fusionner les corrections récents dans les datasets.\n"
            "2. Réentraîner une nouvelle version (v2) du modèle.\n"
            "3. Valider le modèle avant remplacement en production.\n\n"
            "Cette opération peut prendre quelques minutes.",
        )
        if not reponse:
            return

        try:
            # 1. Fusionner les corrections (script 10)
            subprocess.run([sys.executable, str(BASE_DIR / "scripts" / "10_fusionner_corrections_dataset.py")], check=True)

            # 2. Réentraîner le modèle (script 11)
            subprocess.run([
                sys.executable, str(BASE_DIR / "scripts" / "11_reentrainer_modele.py"),
                "--type", "espece", "--version", "v2"
            ], check=True)

            # 3. Valider et déployer en production (script 12)
            subprocess.run([
                sys.executable, str(BASE_DIR / "scripts" / "12_valider_et_deployer.py"),
                "--nouveau", str(BASE_DIR / "models" / "espece" / "v2" / "modele_espece.pt"),
                "--actuel", str(BASE_DIR / "models" / "espece" / "modele_espece.pt"),
                "--test-dir", str(BASE_DIR / "dataset_espece_test")
            ], check=True)

            messagebox.showinfo("Succès", "Le réentraînement et le déploiement de l'IA ont été effectués avec succès !")

        except subprocess.CalledProcessError as err:
            messagebox.showerror("Erreur de réentraînement", f"Une erreur s'est produite lors du processus :\n{err}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", default=str(BASE_DIR / "output" / "a_identifier_manuellement" / "crops_non_identifies.csv"),
                         help="CSV produit par 04_preparer_dossier_relecture.py")
    parser.add_argument("--crops-dir", default=None,
                         help="Dossier contenant les crops à identifier (par défaut : dossier du --csv)")
    parser.add_argument("--propositions", default=str(BASE_DIR / "output" / "propositions_visuelles.csv"),
                         help="CSV produit par 06_proposer_correspondances_visuelles.py (optionnel)")
    parser.add_argument("--crops-valides-dir", default=str(BASE_DIR / "output" / "crops_valides"),
                         help="Dossier des crops de référence par identifiant (pour afficher les vignettes des suggestions)")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise SystemExit(
            f"❌ {csv_path} introuvable. Lance d'abord :\n"
            "   python scripts/04_preparer_dossier_relecture.py"
        )
    crops_dir = Path(args.crops_dir) if args.crops_dir else csv_path.parent

    app = EcranIdentification(
        csv_path=csv_path,
        crops_dir=crops_dir,
        propositions_path=Path(args.propositions),
        crops_valides_dir=Path(args.crops_valides_dir),
    )
    app.mainloop()


if __name__ == "__main__":
    main()