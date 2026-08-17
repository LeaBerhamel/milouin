# -*- coding: utf-8 -*-
"""
Serveur HTTP local (127.0.0.1 uniquement) qui reçoit les clics "Corriger"
et "Archiver" depuis le rapport HTML et les pousse dans une queue
thread-safe, pour que la fenêtre Tkinter les traite dans son propre thread.

Reprend le même pattern que QueueLogHandler / RedirectPrintToQueue déjà
utilisés dans scripts/interface_graphique.py (queue de tuples
(type_message, contenu), consommée par un self.after() côté Tkinter) --
voir README.md de ce dossier pour le branchement exact.

Ce fichier est autonome et ne modifie rien dans scripts/. Il peut être
testé seul via demo_standalone.py.
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

PAGE_CONFIRMATION = """<!DOCTYPE html>
<html lang="fr">
<head><meta charset="utf-8"><title>Correction envoyée</title>
<style>
  body { font-family: -apple-system, Segoe UI, Arial, sans-serif;
         display: flex; align-items: center; justify-content: center;
         height: 100vh; margin: 0; background: #f7f7f8; color: #222; }
  .carte { background: white; padding: 24px 32px; border-radius: 8px;
           box-shadow: 0 1px 3px rgba(0,0,0,0.15); text-align: center; }
</style>
</head>
<body>
  <div class="carte">
    <p>✅ Fenêtre de correction ouverte dans Suivi Canards.</p>
    <p>Tu peux fermer cet onglet.</p>
  </div>
</body>
</html>"""

PAGE_ERREUR_SERVEUR_FERME = """<!DOCTYPE html>
<html lang="fr">
<head><meta charset="utf-8"><title>Suivi Canards fermé</title></head>
<body style="font-family: sans-serif; padding: 40px;">
  <p>⚠️ Ce lien ne fonctionne que si l'application Suivi Canards est
  encore ouverte.</p>
  <p>Relance <code>Lancer.bat</code>, puis rouvre ce rapport depuis le
  bouton "📂 Ouvrir le dossier de sortie".</p>
</body>
</html>"""

PAGE_ARCHIVAGE_LANCE = """<!DOCTYPE html>
<html lang="fr">
<head><meta charset="utf-8"><title>Archivage en cours</title>
<style>
  body { font-family: -apple-system, Segoe UI, Arial, sans-serif;
         display: flex; align-items: center; justify-content: center;
         height: 100vh; margin: 0; background: #f7f7f8; color: #222; }
  .carte { background: white; padding: 24px 32px; border-radius: 8px;
           box-shadow: 0 1px 3px rgba(0,0,0,0.15); text-align: center; }
</style>
</head>
<body>
  <div class="carte">
    <p>🗄️ Demande d'archivage envoyée à Suivi Canards.</p>
    <p>Une fenêtre de confirmation s'est ouverte dans l'application --
    reviens-y pour valider.</p>
    <p>Ce rapport ne reflétera plus le contenu du dossier une fois
    l'archivage confirmé : tu peux fermer cet onglet.</p>
  </div>
</body>
</html>"""


PAGE_ERREUR_SERIE_VIDE = """<!DOCTYPE html>
<html lang="fr">
<head><meta charset="utf-8"><title>Rien à vérifier</title></head>
<body style="font-family: sans-serif; padding: 40px;">
  <p>⚠️ Paramètre 'rafales' manquant ou vide -- rien à ouvrir.</p>
</body>
</html>"""


class GestionnaireCorrections(BaseHTTPRequestHandler):
    """Handler HTTP minimal : trois routes.
    - /corriger?rafale=<id> : pousse une demande de correction pour UNE
      rafale (voir fenetre_rafale.py).
    - /corriger_serie?rafales=<id1>,<id2>,... : pousse une demande de
      correction en série -- l'appli enchaîne automatiquement les
      fenêtres les unes après les autres au fil des validations, sans
      revenir cliquer dans le rapport entre deux rafales (voir
      InterfaceCanard._ouvrir_correction_serie / fenetre_rafale.py,
      paramètres serie_position / on_serie_suivante). Utilisé par le
      bouton "🔎 Vérifier toutes les rafales à vérifier" du rapport HTML
      (voir utils/report.py).
    - /archiver : pousse une demande d'archivage (voir
      01_archiver_resultats.py, appelé côté appli par
      InterfaceCanard._archiver_resultats_courants -- une confirmation
      est TOUJOURS demandée côté appli avant tout déplacement de
      fichiers, ce endpoint ne fait qu'annoncer la demande)."""

    # Injectée par demarrer_serveur() avant que le serveur ne tourne.
    queue_cible = None

    def do_GET(self):
        chemin = urlparse(self.path)

        if chemin.path == "/archiver":
            self.queue_cible.put(("archiver", {}))
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(PAGE_ARCHIVAGE_LANCE.encode("utf-8"))
            return

        if chemin.path == "/corriger_serie":
            params_bruts = parse_qs(chemin.query)
            valeur_brute = params_bruts.get("rafales", [""])[0]
            # Tolère les espaces/entrées vides qu'un copier-coller pourrait
            # introduire (ex. "1, 2, 3" ou une virgule finale) plutôt que
            # de planter sur un int("") -- un identifiant invalide isolé
            # est juste ignoré, pas fatal pour toute la série.
            rafale_ids = []
            for morceau in valeur_brute.split(","):
                morceau = morceau.strip()
                if not morceau:
                    continue
                try:
                    rafale_ids.append(int(morceau))
                except ValueError:
                    continue

            if not rafale_ids:
                self.send_response(400)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(PAGE_ERREUR_SERIE_VIDE.encode("utf-8"))
                return

            self.queue_cible.put(("correction_serie", {"rafale_ids": rafale_ids}))

            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(PAGE_CONFIRMATION.encode("utf-8"))
            return

        if chemin.path != "/corriger":
            self.send_response(404)
            self.end_headers()
            return

        params_bruts = parse_qs(chemin.query)
        # parse_qs renvoie des listes ; on ne garde que la première valeur
        # de chaque paramètre, plus pratique à consommer côté fenêtre.
        params = {cle: valeurs[0] for cle, valeurs in params_bruts.items()}

        if "rafale" not in params:
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b"Parametre 'rafale' manquant.")
            return

        try:
            rafale_id = int(params["rafale"])
        except ValueError:
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b"Parametre 'rafale' invalide (attendu : un entier).")
            return

        self.queue_cible.put(("correction", {"rafale_id": rafale_id}))

        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(PAGE_CONFIRMATION.encode("utf-8"))

    def log_message(self, format, *args):
        # Coupe le log par défaut de BaseHTTPRequestHandler (une ligne par
        # requête sur stderr) : on n'en a pas besoin, l'appli a déjà son
        # propre journal.
        pass


def demarrer_serveur(queue_cible):
    """Démarre le serveur sur un port libre choisi par l'OS (port=0), dans
    un thread daemon (s'arrête automatiquement si le process principal se
    termine, pas besoin d'un arrêt explicite si l'appli est fermée
    brutalement -- mais on expose quand même serveur.shutdown() pour un
    arrêt propre, voir README.md).

    Retourne (serveur, port).
    """
    GestionnaireCorrections.queue_cible = queue_cible

    serveur = ThreadingHTTPServer(("127.0.0.1", 0), GestionnaireCorrections)
    port = serveur.server_address[1]

    thread = threading.Thread(target=serveur.serve_forever, daemon=True)
    thread.start()

    return serveur, port


def construire_lien_correction(port, rafale_id):
    """Construit l'URL http://127.0.0.1:<port>/corriger?rafale=<id> à
    insérer dans le rapport HTML pour une rafale donnée. Un seul lien par
    rafale (pas par canard) : la fenêtre "photo + liste" ouverte côté
    appli affiche tous les canards de la photo, bbox déjà dessinées, le
    chercheur choisit lequel corriger une fois la fenêtre ouverte -- le
    serveur n'a donc besoin de transmettre que l'identifiant de la
    rafale, tout le reste (espèce/sexe/couleur/bague de chaque canard)
    est retrouvé côté appli depuis les résultats déjà en mémoire.
    """
    return f"http://127.0.0.1:{port}/corriger?rafale={rafale_id}"


def construire_lien_correction_serie(port, rafale_ids):
    """Construit l'URL http://127.0.0.1:<port>/corriger_serie?rafales=<id1>,<id2>,...
    à insérer dans le rapport HTML pour un bouton "🔎 Vérifier toutes" --
    voir utils/report.py. rafale_ids : liste d'entiers, dans l'ordre où les
    fenêtres doivent s'enchaîner (typiquement l'ordre du tableau du
    rapport, donc l'ordre des rafales)."""
    ids_txt = ",".join(str(rid) for rid in rafale_ids)
    return f"http://127.0.0.1:{port}/corriger_serie?rafales={ids_txt}"


def construire_lien_archivage(port):
    """Construit l'URL http://127.0.0.1:<port>/archiver à insérer une
    seule fois dans le rapport HTML (bouton "🗄️ Archiver", pas par
    rafale)."""
    return f"http://127.0.0.1:{port}/archiver"