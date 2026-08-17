# -*- coding: utf-8 -*-
"""
Démo autonome et testable du flux complet "clic dans le rapport -> pont
serveur -> formulaire Tkinter -> écriture dans dictionnaire/".

Ne touche à AUCUN fichier du dépôt existant (scripts/). Tout se passe
dans un dossier temporaire, recréé à chaque lancement.

Usage :
    python module_corrections/demo_standalone.py

Ça ouvre :
  1. Un faux rapport.html (3 crops factices) dans le navigateur par défaut.
  2. Une petite fenêtre Tkinter "Suivi Canards (démo)" qui reste ouverte
     et fait office de pont -- c'est elle qui doit rester active pour que
     les liens "Corriger" du rapport fonctionnent (exactement comme
     interface_graphique.py devra le faire une fois intégré).

Clique sur "✏️ Corriger" sous n'importe quel crop du rapport -> une
fenêtre de correction doit s'ouvrir au premier plan. Valide -> va
regarder le contenu de dictionnaire_demo/corrections.csv et
dictionnaire_demo/dictionnaire_identifications.json à côté de ce fichier.
"""

import queue
import shutil
import sys
import tkinter as tk
import webbrowser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ecrire_corrections import traiter_validation_formulaire
from formulaire_correction import ouvrir_formulaire_correction
from serveur_corrections import construire_lien_correction, demarrer_serveur

DOSSIER_DEMO = Path(__file__).resolve().parent / "dictionnaire_demo"

CROPS_FACTICES = [
    {"crop": "IMG_0042_crop1.jpg", "espece": "colvert", "sexe": "male", "couleur": "bleu", "bague": ""},
    {"crop": "IMG_0043_crop1.jpg", "espece": "morillon", "sexe": "femelle", "couleur": "", "bague": ""},
    {"crop": "IMG_0044_crop2.jpg", "espece": "inconnu", "sexe": "indetermine_juvenile", "couleur": "rouge", "bague": "AB123"},
]


def generer_faux_rapport(port, chemin_html):
    lignes = ""
    for crop in CROPS_FACTICES:
        lien = construire_lien_correction(
            port,
            crop["crop"],
            espece=crop["espece"],
            sexe=crop["sexe"],
            couleur=crop["couleur"],
            bague=crop["bague"],
        )
        lignes += f"""
        <tr>
          <td>{crop['crop']}</td>
          <td>{crop['espece']}</td>
          <td>{crop['sexe']}</td>
          <td>{crop['couleur'] or '—'}</td>
          <td>{crop['bague'] or '—'}</td>
          <td><a href="{lien}" target="_blank">✏️ Corriger</a></td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html lang="fr">
<head><meta charset="utf-8"><title>Faux rapport (démo)</title>
<style>
  body {{ font-family: sans-serif; margin: 24px; }}
  table {{ border-collapse: collapse; width: 100%; }}
  th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
  th {{ background: #f5f5f5; }}
</style>
</head>
<body>
  <h1>Rapport factice — démo module_corrections</h1>
  <p>Le serveur de correction tourne sur le port {port}. Clique sur un
  lien "Corriger" ci-dessous.</p>
  <table>
    <tr><th>Crop</th><th>Espèce</th><th>Sexe</th><th>Couleur</th><th>Bague</th><th></th></tr>
    {lignes}
  </table>
</body>
</html>"""
    chemin_html.write_text(html, encoding="utf-8")


def main():
    if DOSSIER_DEMO.exists():
        shutil.rmtree(DOSSIER_DEMO)
    DOSSIER_DEMO.mkdir(parents=True)

    file_attente = queue.Queue()
    serveur, port = demarrer_serveur(file_attente)
    print(f"[demo] Serveur de correction démarré sur 127.0.0.1:{port}")

    racine = tk.Tk()
    racine.title("Suivi Canards (démo module_corrections)")
    racine.geometry("420x160")

    compteur_var = tk.StringVar(value="0 correction reçue")
    tk.Label(racine, text="Cette fenêtre doit rester ouverte.\n"
                          "Les liens 'Corriger' du rapport passent par elle.",
             justify="left", padx=16, pady=16).pack(anchor="w")
    tk.Label(racine, textvariable=compteur_var, font=("", 10, "bold"), padx=16).pack(anchor="w")

    compteur = {"n": 0}

    def on_valider(valeurs):
        resultat = traiter_validation_formulaire(DOSSIER_DEMO, valeurs)
        compteur["n"] += 1
        compteur_var.set(f"{compteur['n']} correction(s) reçue(s) — "
                          f"dernière: correction_ecrite={resultat['correction_ecrite']}, "
                          f"bague_ecrite={resultat['bague_ecrite']}")
        print(f"[demo] Validé pour {valeurs['crop']} -> {resultat}")

    def traiter_file_attente():
        try:
            while True:
                type_msg, contenu = file_attente.get_nowait()
                if type_msg == "correction":
                    print(f"[demo] Requête de correction reçue : {contenu}")
                    ouvrir_formulaire_correction(racine, contenu, on_valider)
        except queue.Empty:
            pass
        racine.after(100, traiter_file_attente)

    racine.after(100, traiter_file_attente)

    chemin_html = DOSSIER_DEMO / "rapport.html"
    generer_faux_rapport(port, chemin_html)
    webbrowser.open(chemin_html.as_uri())

    def a_la_fermeture():
        serveur.shutdown()
        racine.destroy()

    racine.protocol("WM_DELETE_WINDOW", a_la_fermeture)
    racine.mainloop()


if __name__ == "__main__":
    main()
