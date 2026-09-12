#!/usr/bin/env python3
"""
PIXCHRON - MATCH OR PASS : INTERFACE WEB ULTRA-RAPIDE DE REVUE PIXEL ART

Fonctionnalités :
- Démarrage instantané (0.1s).
- Affichage centré avec zoom fluide et grille fine séparant chaque pixel individuel.
- Fond damier transparent discret pour visualiser l'Alpha.
- Barre d'espace / Flèche droite : Valider (Keep / Match).
- Bouton Rouge / Flèche bas / Suppr : Rejeter (Envoyer en quarantaine).
- Flèche gauche / Touche Z : Précédent (Annuler le rejet et restaurer le fichier).
- Raccourci 'G' : Afficher/Masquer la grille de pixels.
- Zéro dépendance externe (tourne avec le serveur HTTP standard de Python).

Utilisation :
    python review_ui.py --dir ./datasets_gold_pass1 --quarantine ./datasets_purge_pass1 --port 7860
"""

import os
import sys
import json
import shutil
import argparse
import urllib.parse
from http.server import HTTPServer, BaseHTTPRequestHandler
from PIL import Image
import numpy as np

SUPPORTED_EXTS = {'.png', '.jpg', '.jpeg', '.webp', '.bmp', '.gif'}

HTML_PAGE = """<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Pixchron - Match or Pass Reviewer</title>
    <style>
        :root {
            --bg: #0f1117;
            --panel: #1a1d27;
            --border: #2a2e3d;
            --text: #e1e4ea;
            --text-dim: #8b92a5;
            --accent-green: #10b981;
            --accent-green-hover: #059669;
            --accent-red: #ef4444;
            --accent-red-hover: #dc2626;
            --accent-blue: #3b82f6;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            user-select: none;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, monospace;
            background: var(--bg);
            color: var(--text);
            height: 100vh;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }

        /* En-tête */
        header {
            background: var(--panel);
            border-bottom: 1px solid var(--border);
            padding: 10px 24px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            height: 60px;
        }

        .title-group {
            display: flex;
            align-items: center;
            gap: 12px;
        }

        .logo {
            font-size: 20px;
            font-weight: 800;
            background: linear-gradient(135deg, #10b981, #3b82f6);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .badge {
            background: #222634;
            padding: 4px 10px;
            border-radius: 6px;
            font-size: 13px;
            color: var(--text-dim);
            border: 1px solid var(--border);
        }

        .stats {
            display: flex;
            gap: 16px;
            font-size: 14px;
        }

        .stat-item {
            display: flex;
            align-items: center;
            gap: 6px;
        }

        .stat-val {
            font-weight: 700;
        }

        .val-green { color: var(--accent-green); }
        .val-red { color: var(--accent-red); }

        /* Conteneur principal */
        main {
            flex: 1;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            position: relative;
            padding: 20px;
        }

        /* Métadonnées de l'image */
        .meta-card {
            margin-bottom: 14px;
            display: flex;
            gap: 16px;
            background: var(--panel);
            padding: 8px 18px;
            border-radius: 30px;
            border: 1px solid var(--border);
            font-size: 13px;
        }

        .meta-pill {
            display: flex;
            align-items: center;
            gap: 6px;
        }

        .meta-label { color: var(--text-dim); }
        .meta-val { font-weight: 600; color: #fff; }

        /* Visualiseur Canvas */
        .viewport {
            background-color: #161922;
            background-image: 
                linear-gradient(45deg, #1e222e 25%, transparent 25%), 
                linear-gradient(-45deg, #1e222e 25%, transparent 25%), 
                linear-gradient(45deg, transparent 75%, #1e222e 75%), 
                linear-gradient(-45deg, transparent 75%, #1e222e 75%);
            background-size: 20px 20px;
            background-position: 0 0, 0 10px, 10px -10px, -10px 0px;
            border: 2px solid var(--border);
            border-radius: 12px;
            box-shadow: 0 12px 36px rgba(0,0,0,0.5);
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 10px;
            max-width: 85vw;
            max-height: 60vh;
            overflow: auto;
            position: relative;
        }

        canvas {
            display: block;
            image-rendering: pixelated;
        }

        /* Barre d'outils de contrôle */
        .controls {
            margin-top: 24px;
            display: flex;
            align-items: center;
            gap: 20px;
        }

        .btn {
            border: none;
            cursor: pointer;
            font-family: inherit;
            font-weight: 700;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 10px;
            transition: all 0.15s ease;
            outline: none;
        }

        .btn:active {
            transform: scale(0.95);
        }

        .btn-prev {
            width: 54px;
            height: 54px;
            border-radius: 50%;
            background: #252a3a;
            color: var(--text);
            border: 1px solid var(--border);
            font-size: 22px;
        }

        .btn-prev:hover {
            background: #31374a;
        }

        .btn-action {
            height: 64px;
            padding: 0 36px;
            border-radius: 32px;
            font-size: 17px;
            box-shadow: 0 8px 24px rgba(0,0,0,0.3);
        }

        .btn-reject {
            background: var(--accent-red);
            color: #fff;
        }

        .btn-reject:hover {
            background: var(--accent-red-hover);
        }

        .btn-accept {
            background: var(--accent-green);
            color: #fff;
        }

        .btn-accept:hover {
            background: var(--accent-green-hover);
        }

        .kbd {
            background: rgba(0,0,0,0.25);
            padding: 3px 8px;
            border-radius: 6px;
            font-size: 11px;
            font-weight: 500;
            letter-spacing: 0.5px;
        }

        /* Pied de page et raccourcis */
        footer {
            background: var(--panel);
            border-top: 1px solid var(--border);
            padding: 8px 24px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            font-size: 12px;
            color: var(--text-dim);
        }

        .shortcuts-hint {
            display: flex;
            gap: 18px;
        }

        .hint-item {
            display: flex;
            align-items: center;
            gap: 6px;
        }

        /* Barre de progression */
        .progress-bar-container {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 3px;
            background: rgba(255,255,255,0.05);
        }

        .progress-bar-fill {
            height: 100%;
            background: linear-gradient(90deg, #10b981, #3b82f6);
            width: 0%;
            transition: width 0.2s ease;
        }

        /* Toast notification */
        .toast {
            position: absolute;
            bottom: 30px;
            background: #252a3a;
            border: 1px solid var(--border);
            padding: 8px 16px;
            border-radius: 8px;
            font-size: 13px;
            opacity: 0;
            transform: translateY(10px);
            transition: all 0.2s ease;
            pointer-events: none;
        }

        .toast.show {
            opacity: 1;
            transform: translateY(0);
        }
    </style>
</head>
<body>

    <div class="progress-bar-container">
        <div class="progress-bar-fill" id="progressFill"></div>
    </div>

    <header>
        <div class="title-group">
            <span class="logo">PIXCHRON</span>
            <span class="badge" id="sourceDirBadge">Dossier</span>
        </div>
        <div class="stats">
            <div class="stat-item">
                <span>Progression :</span>
                <span class="stat-val" id="progressText">0 / 0</span>
            </div>
            <div class="stat-item">
                <span>Conservés :</span>
                <span class="stat-val val-green" id="keptCount">0</span>
            </div>
            <div class="stat-item">
                <span>Rejetés :</span>
                <span class="stat-val val-red" id="rejectedCount">0</span>
            </div>
        </div>
    </header>

    <main>
        <div class="meta-card" id="metaCard">
            <div class="meta-pill">
                <span class="meta-label">Fichier :</span>
                <span class="meta-val" id="fileName">...</span>
            </div>
            <div class="meta-pill">
                <span class="meta-label">Catégorie :</span>
                <span class="meta-val" id="fileCategory">...</span>
            </div>
            <div class="meta-pill">
                <span class="meta-label">Résolution :</span>
                <span class="meta-val" id="fileRes">...</span>
            </div>
            <div class="meta-pill">
                <span class="meta-label">Palette :</span>
                <span class="meta-val" id="fileColors">...</span>
            </div>
        </div>

        <div class="viewport" id="viewport">
            <canvas id="pixelCanvas"></canvas>
        </div>

        <div class="controls">
            <button class="btn btn-prev" id="btnUndo" title="Précédent / Annuler (Flèche Gauche ou Z)">
                ←
            </button>
            <button class="btn btn-action btn-reject" id="btnReject" title="Rejeter vers la quarantaine (Suppr ou Flèche Bas)">
                ✕ REJETER <span class="kbd">Suppr / ↓</span>
            </button>
            <button class="btn btn-action btn-accept" id="btnAccept" title="Valider et passer à la suivante (Espace ou Flèche Droite)">
                ✓ VALIDER <span class="kbd">ESPACE / →</span>
            </button>
        </div>

        <div class="toast" id="toast"></div>
    </main>

    <footer>
        <div class="shortcuts-hint">
            <div class="hint-item"><span class="kbd">ESPACE</span> ou <span class="kbd">→</span> Valider</div>
            <div class="hint-item"><span class="kbd">SUPPR</span> ou <span class="kbd">↓</span> Rejeter</div>
            <div class="hint-item"><span class="kbd">←</span> ou <span class="kbd">Z</span> Précédent (Undo)</div>
            <div class="hint-item"><span class="kbd">G</span> Grille de Pixels (On/Off)</div>
            <div class="hint-item"><span class="kbd">+</span> / <span class="kbd">-</span> Zoom</div>
        </div>
        <div id="zoomText">Zoom : Auto</div>
    </footer>

    <script>
        let imagesList = [];
        let currentIndex = 0;
        let kept = 0;
        let rejected = 0;
        let undoStack = [];
        let showGrid = true;
        let manualZoom = 0;

        const canvas = document.getElementById("pixelCanvas");
        const ctx = canvas.getContext("2d");
        const toast = document.getElementById("toast");

        function showToast(msg) {
            toast.textContent = msg;
            toast.classList.add("show");
            setTimeout(() => toast.classList.remove("show"), 1200);
        }

        async function init() {
            try {
                const res = await fetch("/api/init");
                const data = await res.json();
                document.getElementById("sourceDirBadge").textContent = data.input_dir;
                imagesList = data.images;
                if (imagesList.length === 0) {
                    alert("Aucune image trouvée dans le répertoire !");
                    return;
                }
                loadCurrentImage();
            } catch (err) {
                console.error("Erreur init:", err);
            }
        }

        function loadCurrentImage() {
            if (currentIndex >= imagesList.length) {
                document.getElementById("fileName").textContent = "Revue terminée !";
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                showToast("Félicitations, toutes les images ont été traitées !");
                return;
            }

            const item = imagesList[currentIndex];
            
            document.getElementById("fileName").textContent = item.filename;
            document.getElementById("fileCategory").textContent = item.category;
            document.getElementById("fileRes").textContent = "...";
            document.getElementById("fileColors").textContent = "...";

            const progressPct = (currentIndex / imagesList.length) * 100;
            document.getElementById("progressFill").style.width = `${progressPct}%`;
            document.getElementById("progressText").textContent = `${currentIndex + 1} / ${imagesList.length}`;
            document.getElementById("keptCount").textContent = kept;
            document.getElementById("rejectedCount").textContent = rejected;

            const img = new Image();
            img.src = `/api/image?path=${encodeURIComponent(item.rel_path)}&t=${Date.now()}`;
            img.onload = () => {
                document.getElementById("fileRes").textContent = `${img.naturalWidth}x${img.naturalHeight}`;
                renderImage(img);
                countColorsFast(img);
            };
        }

        function countColorsFast(img) {
            try {
                const offCanvas = document.createElement("canvas");
                offCanvas.width = img.naturalWidth;
                offCanvas.height = img.naturalHeight;
                const offCtx = offCanvas.getContext("2d");
                offCtx.drawImage(img, 0, 0);
                const pData = offCtx.getImageData(0, 0, offCanvas.width, offCanvas.height).data;
                const colorSet = new Set();
                for (let i = 0; i < pData.length; i += 4) {
                    if (pData[i+3] === 0) {
                        colorSet.add(0);
                    } else {
                        colorSet.add((pData[i] << 16) | (pData[i+1] << 8) | pData[i+2]);
                    }
                    if (colorSet.size > 256) break;
                }
                document.getElementById("fileColors").textContent = `${colorSet.size} col`;
            } catch (e) {
                document.getElementById("fileColors").textContent = "N/A";
            }
        }

        function renderImage(img) {
            const maxViewW = Math.min(window.innerWidth * 0.75, 650);
            const maxViewH = Math.min(window.innerHeight * 0.52, 480);

            let zoom = manualZoom;
            if (zoom <= 0) {
                const zoomX = Math.floor(maxViewW / img.naturalWidth);
                const zoomY = Math.floor(maxViewH / img.naturalHeight);
                zoom = Math.max(1, Math.min(zoomX, zoomY));
                if (zoom > 24) zoom = 24;
            }

            document.getElementById("zoomText").textContent = `Zoom : ${zoom}x`;

            const cw = img.naturalWidth * zoom;
            const ch = img.naturalHeight * zoom;

            canvas.width = cw;
            canvas.height = ch;

            ctx.imageSmoothingEnabled = false;
            ctx.drawImage(img, 0, 0, cw, ch);

            // Grille fine de pixels
            if (showGrid && zoom >= 4) {
                ctx.strokeStyle = "rgba(100, 115, 140, 0.40)";
                ctx.lineWidth = 1;
                ctx.beginPath();
                
                for (let x = 0; x <= cw; x += zoom) {
                    ctx.moveTo(x + 0.5, 0);
                    ctx.lineTo(x + 0.5, ch);
                }
                for (let y = 0; y <= ch; y += zoom) {
                    ctx.moveTo(0, y + 0.5);
                    ctx.lineTo(cw, y + 0.5);
                }
                ctx.stroke();
            }
        }

        async function handleAccept() {
            if (currentIndex >= imagesList.length) return;
            const item = imagesList[currentIndex];
            undoStack.push({ action: "accept", index: currentIndex, item: item });
            kept++;
            currentIndex++;
            showToast("✓ Image Validée");
            loadCurrentImage();
        }

        async function handleReject() {
            if (currentIndex >= imagesList.length) return;
            const item = imagesList[currentIndex];
            
            try {
                const res = await fetch("/api/reject", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ rel_path: item.rel_path })
                });
                const data = await res.json();
                if (data.ok) {
                    undoStack.push({ action: "reject", index: currentIndex, item: item });
                    rejected++;
                    currentIndex++;
                    showToast("✕ Déplacé en Quarantaine");
                    loadCurrentImage();
                }
            } catch (err) {
                console.error("Erreur rejet:", err);
            }
        }

        async function handleUndo() {
            if (undoStack.length === 0) {
                showToast("Aucune action à annuler");
                return;
            }

            const lastAction = undoStack.pop();
            
            if (lastAction.action === "reject") {
                try {
                    const res = await fetch("/api/undo", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ rel_path: lastAction.item.rel_path })
                    });
                    const data = await res.json();
                    if (data.ok) {
                        rejected--;
                        currentIndex = lastAction.index;
                        showToast("↺ Rejet Annulé (Fichier Restauré)");
                        loadCurrentImage();
                    }
                } catch (err) {
                    console.error("Erreur undo:", err);
                }
            } else {
                kept--;
                currentIndex = lastAction.index;
                showToast("↺ Retour à l'image précédente");
                loadCurrentImage();
            }
        }

        document.getElementById("btnAccept").addEventListener("click", handleAccept);
        document.getElementById("btnReject").addEventListener("click", handleReject);
        document.getElementById("btnUndo").addEventListener("click", handleUndo);

        window.addEventListener("keydown", (e) => {
            if (e.code === "Space" || e.code === "ArrowRight") {
                e.preventDefault();
                handleAccept();
            } else if (e.code === "Delete" || e.code === "Backspace" || e.code === "ArrowDown") {
                e.preventDefault();
                handleReject();
            } else if (e.code === "ArrowLeft" || e.key.toLowerCase() === "z") {
                e.preventDefault();
                handleUndo();
            } else if (e.key.toLowerCase() === "g") {
                showGrid = !showGrid;
                showToast(showGrid ? "Grille de pixels : ACTIVE" : "Grille : DESACTIVEE");
                loadCurrentImage();
            } else if (e.key === "+" || e.key === "=") {
                manualZoom = (manualZoom || 8) + 2;
                loadCurrentImage();
            } else if (e.key === "-" || e.key === "_") {
                manualZoom = Math.max(2, (manualZoom || 8) - 2);
                loadCurrentImage();
            }
        });

        init();
    </script>
</body>
</html>
"""

class ReviewServerHandler(BaseHTTPRequestHandler):
    input_dir = "./datasets_gold_pass1"
    quarantine_dir = "./datasets_purge_pass1"
    images_meta = []

    def log_message(self, format, *args):
        pass

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        
        if parsed.path in ("/", "/index.html"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode("utf-8"))
            
        elif parsed.path == "/api/init":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            data = {
                "input_dir": self.input_dir,
                "quarantine_dir": self.quarantine_dir,
                "images": self.images_meta
            }
            self.wfile.write(json.dumps(data).encode("utf-8"))
            
        elif parsed.path == "/api/image":
            query = urllib.parse.parse_qs(parsed.query)
            rel_path = query.get("path", [""])[0]
            full_path = os.path.join(self.input_dir, rel_path)
            
            if not os.path.exists(full_path):
                full_path = os.path.join(self.quarantine_dir, rel_path)
                
            if os.path.exists(full_path):
                ext = os.path.splitext(full_path)[1].lower()
                mime = "image/png" if ext == ".png" else "image/jpeg"
                self.send_response(200)
                self.send_header("Content-Type", mime)
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                with open(full_path, "rb") as f:
                    self.wfile.write(f.read())
            else:
                self.send_response(404)
                self.end_headers()
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        content_len = int(self.headers.get("Content-Length", 0))
        post_body = self.rfile.read(content_len)
        data = json.loads(post_body.decode("utf-8")) if post_body else {}

        if parsed.path == "/api/reject":
            rel_path = data.get("rel_path", "")
            src = os.path.join(self.input_dir, rel_path)
            dst = os.path.join(self.quarantine_dir, rel_path)
            
            if os.path.exists(src):
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.move(src, dst)
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"ok": True, "action": "rejected"}).encode("utf-8"))
            else:
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"ok": False, "error": "Fichier source introuvable"}).encode("utf-8"))

        elif parsed.path == "/api/undo":
            rel_path = data.get("rel_path", "")
            src_in_quarantine = os.path.join(self.quarantine_dir, rel_path)
            dst_in_gold = os.path.join(self.input_dir, rel_path)
            
            if os.path.exists(src_in_quarantine):
                os.makedirs(os.path.dirname(dst_in_gold), exist_ok=True)
                shutil.move(src_in_quarantine, dst_in_gold)
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"ok": True, "action": "restored"}).encode("utf-8"))
            else:
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"ok": True, "action": "noop"}).encode("utf-8"))

def scan_images_fast(input_dir: str):
    images_meta = []
    print(f"🔍 Indexation instantanée des images dans '{input_dir}'...")
    
    for root, _, files in os.walk(input_dir):
        for f in sorted(files):
            ext = os.path.splitext(f)[1].lower()
            if ext in SUPPORTED_EXTS:
                full_p = os.path.join(root, f)
                rel_p = os.path.relpath(full_p, input_dir)
                cat = os.path.dirname(rel_p) or "Racine"
                images_meta.append({
                    "rel_path": rel_p,
                    "filename": f,
                    "category": cat
                })
                    
    print(f"✅ {len(images_meta)} images prêtes pour la revue.")
    return images_meta

def main():
    parser = argparse.ArgumentParser(description="Pixchron Match-or-Pass Web Reviewer")
    parser.add_argument("--dir", type=str, default="./datasets_gold_pass1", help="Répertoire à réviser (défaut: ./datasets_gold_pass1 ou ./datasets_cleaned)")
    parser.add_argument("--quarantine", type=str, default="./datasets_purge_pass1", help="Répertoire de quarantaine vers lequel déplacer les rejets")
    parser.add_argument("--port", type=int, default=7860, help="Port HTTP (défaut: 7860)")

    args = parser.parse_args()

    # Si datasets_gold_pass1 n'existe pas encore, fallback automatique sur datasets_cleaned
    if not os.path.exists(args.dir) and os.path.exists("./datasets_cleaned"):
        print(f"ℹ️ '{args.dir}' n'existe pas encore. Utilisation de './datasets_cleaned'.")
        args.dir = "./datasets_cleaned"
        args.quarantine = "./datasets_cleaned_anomalies"

    images_meta = scan_images_fast(args.dir)

    ReviewServerHandler.input_dir = args.dir
    ReviewServerHandler.quarantine_dir = args.quarantine
    ReviewServerHandler.images_meta = images_meta

    server = HTTPServer(("0.0.0.0", args.port), ReviewServerHandler)
    print("=" * 75)
    print("  🎮 PIXCHRON - MATCH OR PASS REVIEWER PRÊT !")
    print("=" * 75)
    print(f"  • Dossier revu       : {args.dir}")
    print(f"  • Dossier Quarantaine: {args.quarantine}")
    print(f"  • Ouvrir dans le web : http://localhost:{args.port}")
    print("=" * 75)
    print("👉 Contrôles :")
    print("   [ESPACE] ou [→]  : Valider l'image (Keep / Match)")
    print("   [SUPPR]  ou [↓]  : Bouton Rouge (Rejeter vers la quarantaine)")
    print("   [←]      ou [Z]  : Flèche Précédent (Annuler et restaurer le fichier)")
    print("   [G]              : Activer / Désactiver la grille de pixels")
    print("   [+] / [-]        : Zoom avant / arrière")
    print("=" * 75)
    print("Appuyez sur Ctrl+C pour arrêter le serveur.")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nArrêt du serveur. À bientôt !")

if __name__ == "__main__":
    main()
