#!/usr/bin/env python3
"""
PIXCHRON - MATCH OR PASS : INTERFACE WEB ULTRA-RAPIDE DE REVUE PIXEL ART

Fonctionnalités :
- Démarrage instantané (0.1s).
- Affichage centré avec zoom fluide et grille fine séparant chaque pixel individuel.
- Fond damier transparent discret pour visualiser l'Alpha.
- Barre d'espace / Flèche droite : Valider (Keep / Match).
- Bouton Rouge / Flèche bas / Suppr : Rejeter (Envoyer en quarantaine).
- Flèche gauche / Touche Z : Précédent (Annuler le rejet/découpage et restaurer le fichier).
- Outil Découpage interactif (Crop & Slice) :
  • Tracé de boîtes de cadrage à la souris directement sur l'image.
  • Support multi-cadres (découper plusieurs sprites d'une même planche).
  • Bouton ⚡ Cadres Auto (Touche A) : Détection automatique des sprites par composantes connexes.
  • Bouton ✂️ DÉCOUPER (Touche C ou Entrée) : Découpe 1:1, sauvegarde _crop_01.png, etc., archive l'original.
  • Touche Échap : Effacer les cadres.
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
            --accent-cyan: #38bdf8;
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
            height: 56px;
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
        .val-cyan { color: var(--accent-cyan); }

        /* Conteneur principal */
        main {
            flex: 1;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            position: relative;
            padding: 12px 20px;
        }

        /* Métadonnées de l'image */
        .meta-card {
            margin-bottom: 10px;
            display: flex;
            gap: 16px;
            background: var(--panel);
            padding: 6px 18px;
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
            max-height: 52vh;
            overflow: auto;
            position: relative;
        }

        canvas {
            display: block;
            image-rendering: pixelated;
            cursor: crosshair;
        }

        /* Barre d'outils de découpage (Crop) */
        .crop-toolbar {
            margin-top: 10px;
            display: flex;
            align-items: center;
            gap: 10px;
            background: var(--panel);
            padding: 6px 16px;
            border-radius: 20px;
            border: 1px solid var(--border);
            max-width: 85vw;
            flex-wrap: wrap;
            justify-content: center;
        }

        .btn-crop-sub {
            background: #252a3a;
            color: var(--text);
            padding: 6px 12px;
            border-radius: 12px;
            font-size: 13px;
            border: 1px solid var(--border);
            cursor: pointer;
            display: flex;
            align-items: center;
            gap: 6px;
            font-weight: 600;
        }
        .btn-crop-sub:hover {
            background: #31374a;
        }

        .btn-crop-action {
            background: linear-gradient(135deg, #0284c7, #2563eb);
            color: #fff;
            padding: 6px 16px;
            border-radius: 12px;
            font-size: 13px;
            border: 1px solid rgba(255,255,255,0.15);
            box-shadow: 0 2px 8px rgba(37,99,235,0.3);
            cursor: pointer;
            display: flex;
            align-items: center;
            gap: 6px;
            font-weight: 700;
            transition: all 0.15s ease;
        }
        .btn-crop-action:hover:not(:disabled) {
            background: linear-gradient(135deg, #0369a1, #1d4ed8);
            transform: scale(1.02);
        }
        .btn-crop-action:disabled {
            opacity: 0.40;
            cursor: not-allowed;
            box-shadow: none;
            background: #374151;
        }

        .box-chips {
            display: flex;
            gap: 6px;
            align-items: center;
            flex-wrap: wrap;
        }

        .box-chip {
            background: rgba(56, 189, 248, 0.15);
            border: 1px solid rgba(56, 189, 248, 0.4);
            color: #38bdf8;
            padding: 2px 8px;
            border-radius: 10px;
            font-size: 11px;
            display: flex;
            align-items: center;
            gap: 6px;
            font-weight: 600;
        }

        .box-chip-del {
            cursor: pointer;
            color: #ef4444;
            font-weight: bold;
            font-size: 13px;
            line-height: 1;
        }
        .box-chip-del:hover {
            color: #ff6b6b;
        }

        .crop-hint {
            font-size: 12px;
            color: var(--text-dim);
            margin-left: 6px;
        }

        /* Barre d'outils de contrôle */
        .controls {
            margin-top: 14px;
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
            width: 50px;
            height: 50px;
            border-radius: 50%;
            background: #252a3a;
            color: var(--text);
            border: 1px solid var(--border);
            font-size: 20px;
        }

        .btn-prev:hover {
            background: #31374a;
        }

        .btn-action {
            height: 56px;
            padding: 0 32px;
            border-radius: 28px;
            font-size: 16px;
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
            padding: 2px 7px;
            border-radius: 5px;
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
            gap: 14px;
            flex-wrap: wrap;
        }

        .hint-item {
            display: flex;
            align-items: center;
            gap: 5px;
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
            bottom: 24px;
            background: #252a3a;
            border: 1px solid var(--border);
            padding: 8px 18px;
            border-radius: 8px;
            font-size: 13px;
            font-weight: 600;
            opacity: 0;
            transform: translateY(10px);
            transition: all 0.2s ease;
            pointer-events: none;
            box-shadow: 0 8px 24px rgba(0,0,0,0.5);
            z-index: 100;
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
                <span>Découpés :</span>
                <span class="stat-val val-cyan" id="croppedCount">0</span>
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

        <div class="crop-toolbar" id="cropToolbar">
            <button class="btn-crop-sub" id="btnAutoDetect" title="Détecter automatiquement les sprites distincts (Touche A)">
                ⚡ Cadres Auto <span class="kbd">A</span>
            </button>
            <button class="btn-crop-action" id="btnCrop" title="Découper les cadres actifs (Touche C ou Entrée)" disabled>
                ✂️ DÉCOUPER <span id="cropCountBadge"></span> <span class="kbd">C / ↵</span>
            </button>
            <button class="btn-crop-sub" id="btnClearBoxes" title="Effacer tous les cadres (Échap)" style="display: none;">
                ✕ Effacer cadres <span class="kbd">Échap</span>
            </button>
            <div class="box-chips" id="boxChips"></div>
            <span class="crop-hint" id="cropHint">💡 Glissez sur l'image pour tracer un cadre</span>
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
            <div class="hint-item"><span class="kbd">Glisser</span> Cadrer</div>
            <div class="hint-item"><span class="kbd">C</span> / <span class="kbd">↵</span> Découper</div>
            <div class="hint-item"><span class="kbd">A</span> Auto-cadres</div>
            <div class="hint-item"><span class="kbd">Échap</span> Effacer cadres</div>
            <div class="hint-item"><span class="kbd">G</span> Grille</div>
            <div class="hint-item"><span class="kbd">+</span> / <span class="kbd">-</span> Zoom</div>
        </div>
        <div id="zoomText">Zoom : Auto</div>
    </footer>

    <script>
        let imagesList = [];
        let currentIndex = 0;
        let kept = 0;
        let rejected = 0;
        let croppedTotal = 0;
        let undoStack = [];
        let showGrid = true;
        let manualZoom = 0;
        let currentZoom = 1;
        let currentLoadedImg = null;

        // État de cadrage (Crop)
        let boxes = [];
        let isDrawing = false;
        let dragStart = { x: 0, y: 0 };
        let currentDrag = null;

        const canvas = document.getElementById("pixelCanvas");
        const ctx = canvas.getContext("2d");
        const toast = document.getElementById("toast");
        const btnAutoDetect = document.getElementById("btnAutoDetect");
        const btnCrop = document.getElementById("btnCrop");
        const btnClearBoxes = document.getElementById("btnClearBoxes");
        const cropCountBadge = document.getElementById("cropCountBadge");
        const boxChips = document.getElementById("boxChips");
        const cropHint = document.getElementById("cropHint");

        function showToast(msg) {
            toast.textContent = msg;
            toast.classList.add("show");
            setTimeout(() => toast.classList.remove("show"), 1400);
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
            boxes = [];
            currentDrag = null;
            isDrawing = false;
            updateBoxesUI();

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
            document.getElementById("croppedCount").textContent = croppedTotal;

            const img = new Image();
            img.src = `/api/image?path=${encodeURIComponent(item.rel_path)}&t=${Date.now()}`;
            img.onload = () => {
                currentLoadedImg = img;
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
            const maxViewH = Math.min(window.innerHeight * 0.46, 440);

            let zoom = manualZoom;
            if (zoom <= 0) {
                const zoomX = Math.floor(maxViewW / img.naturalWidth);
                const zoomY = Math.floor(maxViewH / img.naturalHeight);
                zoom = Math.max(1, Math.min(zoomX, zoomY));
                if (zoom > 24) zoom = 24;
            }

            currentZoom = zoom;
            document.getElementById("zoomText").textContent = `Zoom : ${zoom}x`;

            const cw = img.naturalWidth * zoom;
            const ch = img.naturalHeight * zoom;

            canvas.width = cw;
            canvas.height = ch;

            redrawCanvas();
        }

        function redrawCanvas() {
            if (!currentLoadedImg) return;
            const img = currentLoadedImg;
            const cw = img.naturalWidth * currentZoom;
            const ch = img.naturalHeight * currentZoom;

            ctx.clearRect(0, 0, cw, ch);
            ctx.imageSmoothingEnabled = false;
            ctx.drawImage(img, 0, 0, cw, ch);

            // Grille fine de pixels
            if (showGrid && currentZoom >= 4) {
                ctx.strokeStyle = "rgba(100, 115, 140, 0.40)";
                ctx.lineWidth = 1;
                ctx.beginPath();
                for (let x = 0; x <= cw; x += currentZoom) {
                    ctx.moveTo(x + 0.5, 0);
                    ctx.lineTo(x + 0.5, ch);
                }
                for (let y = 0; y <= ch; y += currentZoom) {
                    ctx.moveTo(0, y + 0.5);
                    ctx.lineTo(cw, y + 0.5);
                }
                ctx.stroke();
            }

            // Dessin des boîtes de découpage confirmées
            boxes.forEach((b, i) => {
                const bx = b.x * currentZoom;
                const by = b.y * currentZoom;
                const bw = b.w * currentZoom;
                const bh = b.h * currentZoom;

                // Remplissage bleu translucide
                ctx.fillStyle = "rgba(56, 189, 248, 0.22)";
                ctx.fillRect(bx, by, bw, bh);

                // Bordure cyan pointillée
                ctx.strokeStyle = "#38bdf8";
                ctx.lineWidth = 2;
                ctx.setLineDash([4, 4]);
                ctx.strokeRect(bx + 0.5, by + 0.5, bw, bh);
                ctx.setLineDash([]);

                // Badge numéro et dimensions
                const label = `#${i + 1} (${b.w}x${b.h})`;
                ctx.font = "bold 11px monospace";
                const metrics = ctx.measureText(label);
                const badgeW = metrics.width + 10;
                const badgeH = 18;
                let badgeX = bx + 2;
                let badgeY = by + 2;
                if (badgeY + badgeH > ch) badgeY = Math.max(0, by - badgeH - 2);

                ctx.fillStyle = "rgba(15, 23, 42, 0.88)";
                if (ctx.roundRect) {
                    ctx.beginPath();
                    ctx.roundRect(badgeX, badgeY, badgeW, badgeH, 4);
                    ctx.fill();
                } else {
                    ctx.fillRect(badgeX, badgeY, badgeW, badgeH);
                }

                ctx.fillStyle = "#38bdf8";
                ctx.fillText(label, badgeX + 5, badgeY + 13);
            });

            // Dessin du cadre en cours de tracé
            if (currentDrag) {
                const dx = currentDrag.x * currentZoom;
                const dy = currentDrag.y * currentZoom;
                const dw = currentDrag.w * currentZoom;
                const dh = currentDrag.h * currentZoom;

                ctx.fillStyle = "rgba(16, 185, 129, 0.28)";
                ctx.fillRect(dx, dy, dw, dh);

                ctx.strokeStyle = "#10b981";
                ctx.lineWidth = 2;
                ctx.setLineDash([4, 4]);
                ctx.strokeRect(dx + 0.5, dy + 0.5, dw, dh);
                ctx.setLineDash([]);

                const label = `${currentDrag.w}x${currentDrag.h}`;
                ctx.font = "bold 11px monospace";
                const metrics = ctx.measureText(label);
                const badgeW = metrics.width + 10;
                const badgeH = 18;
                let badgeX = dx + 2;
                let badgeY = dy + 2;
                if (badgeY + badgeH > ch) badgeY = Math.max(0, dy - badgeH - 2);

                ctx.fillStyle = "rgba(15, 23, 42, 0.9)";
                if (ctx.roundRect) {
                    ctx.beginPath();
                    ctx.roundRect(badgeX, badgeY, badgeW, badgeH, 4);
                    ctx.fill();
                } else {
                    ctx.fillRect(badgeX, badgeY, badgeW, badgeH);
                }

                ctx.fillStyle = "#10b981";
                ctx.fillText(label, badgeX + 5, badgeY + 13);
            }
        }

        function getPixelCoords(e) {
            if (!currentLoadedImg) return { x: 0, y: 0 };
            const rect = canvas.getBoundingClientRect();
            const scaleX = canvas.width / rect.width;
            const scaleY = canvas.height / rect.height;
            const canvasX = (e.clientX - rect.left) * scaleX;
            const canvasY = (e.clientY - rect.top) * scaleY;

            let px = Math.floor(canvasX / currentZoom);
            let py = Math.floor(canvasY / currentZoom);

            px = Math.max(0, Math.min(currentLoadedImg.naturalWidth - 1, px));
            py = Math.max(0, Math.min(currentLoadedImg.naturalHeight - 1, py));
            return { x: px, y: py };
        }

        function updateBoxesUI() {
            if (boxes.length > 0) {
                btnCrop.disabled = false;
                cropCountBadge.textContent = `(${boxes.length})`;
                btnClearBoxes.style.display = "flex";
                cropHint.style.display = "none";
            } else {
                btnCrop.disabled = true;
                cropCountBadge.textContent = "";
                btnClearBoxes.style.display = "none";
                cropHint.style.display = "inline-block";
            }

            boxChips.innerHTML = "";
            boxes.forEach((b, i) => {
                const chip = document.createElement("div");
                chip.className = "box-chip";
                chip.innerHTML = `<span>#${i + 1} (${b.w}x${b.h})</span><span class="box-chip-del" title="Supprimer ce cadre" data-idx="${i}">✕</span>`;
                boxChips.appendChild(chip);
            });

            boxChips.querySelectorAll(".box-chip-del").forEach(btn => {
                btn.addEventListener("click", (e) => {
                    e.stopPropagation();
                    const idx = parseInt(btn.getAttribute("data-idx"), 10);
                    boxes.splice(idx, 1);
                    updateBoxesUI();
                    redrawCanvas();
                });
            });
        }

        function clearBoxes() {
            boxes = [];
            currentDrag = null;
            isDrawing = false;
            updateBoxesUI();
            redrawCanvas();
            showToast("Cadres effacés");
        }

        async function autoDetectSprites() {
            if (currentIndex >= imagesList.length) return;
            const item = imagesList[currentIndex];
            showToast("⚡ Détection des sprites...");
            try {
                const res = await fetch("/api/detect_sprites", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ rel_path: item.rel_path })
                });
                const data = await res.json();
                if (data.ok && data.boxes && data.boxes.length > 0) {
                    boxes = data.boxes;
                    updateBoxesUI();
                    redrawCanvas();
                    showToast(`⚡ ${boxes.length} sprites détectés !`);
                } else {
                    showToast("Aucun sprite distinct détecté");
                }
            } catch (err) {
                console.error("Erreur auto-detect:", err);
                showToast("Erreur lors de la détection");
            }
        }

        async function handleCrop() {
            if (currentIndex >= imagesList.length || boxes.length === 0) return;
            const item = imagesList[currentIndex];
            
            try {
                const res = await fetch("/api/crop", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        rel_path: item.rel_path,
                        boxes: boxes
                    })
                });
                const data = await res.json();
                if (data.ok) {
                    undoStack.push({
                        action: "crop",
                        index: currentIndex,
                        item: item,
                        created_files: data.created_files,
                        original_dst: data.original_dst,
                        boxes: [...boxes]
                    });
                    croppedTotal += (data.created_files || []).length;
                    boxes = [];
                    updateBoxesUI();
                    currentIndex++;
                    showToast(`✂️ ${data.created_files.length} sprites découpés et enregistrés !`);
                    loadCurrentImage();
                } else {
                    showToast("Erreur découpe: " + (data.error || "inconnue"));
                }
            } catch (err) {
                console.error("Erreur crop:", err);
                showToast("Erreur serveur lors de la découpe");
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
            
            if (lastAction.action === "crop") {
                try {
                    const res = await fetch("/api/undo_crop", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({
                            rel_path: lastAction.item.rel_path,
                            created_files: lastAction.created_files,
                            original_dst: lastAction.original_dst
                        })
                    });
                    const data = await res.json();
                    if (data.ok) {
                        croppedTotal -= (lastAction.created_files || []).length;
                        currentIndex = lastAction.index;
                        boxes = lastAction.boxes || [];
                        updateBoxesUI();
                        showToast("↺ Découpage Annulé (Planche Restaurée)");
                        loadCurrentImage();
                    }
                } catch (err) {
                    console.error("Erreur undo crop:", err);
                }
            } else if (lastAction.action === "reject") {
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
                        boxes = [];
                        updateBoxesUI();
                        showToast("↺ Rejet Annulé (Fichier Restauré)");
                        loadCurrentImage();
                    }
                } catch (err) {
                    console.error("Erreur undo:", err);
                }
            } else {
                kept--;
                currentIndex = lastAction.index;
                boxes = [];
                updateBoxesUI();
                showToast("↺ Retour à l'image précédente");
                loadCurrentImage();
            }
        }

        // Événements Souris Canvas (Tracé de Cadres)
        canvas.addEventListener("mousedown", (e) => {
            if (!currentLoadedImg || e.button !== 0) return;
            isDrawing = true;
            dragStart = getPixelCoords(e);
            currentDrag = { x: dragStart.x, y: dragStart.y, w: 1, h: 1 };
            redrawCanvas();
        });

        canvas.addEventListener("mousemove", (e) => {
            if (!currentLoadedImg || !isDrawing) return;
            const p = getPixelCoords(e);
            const x1 = Math.min(dragStart.x, p.x);
            const y1 = Math.min(dragStart.y, p.y);
            const x2 = Math.max(dragStart.x, p.x);
            const y2 = Math.max(dragStart.y, p.y);
            currentDrag = { x: x1, y: y1, w: x2 - x1 + 1, h: y2 - y1 + 1 };
            redrawCanvas();
        });

        window.addEventListener("mouseup", (e) => {
            if (!isDrawing) return;
            isDrawing = false;
            if (currentDrag && currentDrag.w >= 3 && currentDrag.h >= 3) {
                boxes.push(currentDrag);
                updateBoxesUI();
            }
            currentDrag = null;
            redrawCanvas();
        });

        // Boutons
        document.getElementById("btnAccept").addEventListener("click", handleAccept);
        document.getElementById("btnReject").addEventListener("click", handleReject);
        document.getElementById("btnUndo").addEventListener("click", handleUndo);
        btnCrop.addEventListener("click", handleCrop);
        btnAutoDetect.addEventListener("click", autoDetectSprites);
        btnClearBoxes.addEventListener("click", clearBoxes);

        // Raccourcis Clavier
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
            } else if (e.key.toLowerCase() === "c" || e.code === "Enter" || e.code === "NumpadEnter") {
                if (boxes.length > 0) {
                    e.preventDefault();
                    handleCrop();
                }
            } else if (e.key.toLowerCase() === "a") {
                e.preventDefault();
                autoDetectSprites();
            } else if (e.code === "Escape") {
                e.preventDefault();
                clearBoxes();
            } else if (e.key.toLowerCase() === "g") {
                showGrid = !showGrid;
                showToast(showGrid ? "Grille : ACTIVÉE" : "Grille : DÉSACTIVÉE");
                redrawCanvas();
            } else if (e.key === "+" || e.key === "=") {
                manualZoom = (currentZoom || 8) + 2;
                if (currentLoadedImg) renderImage(currentLoadedImg);
            } else if (e.key === "-" || e.key === "_") {
                manualZoom = Math.max(2, (currentZoom || 8) - 2);
                if (currentLoadedImg) renderImage(currentLoadedImg);
            }
        });

        init();
    </script>
</body>
</html>
"""

def detect_sprites(img: Image.Image, min_size: int = 6):
    """
    Détecte automatiquement les boîtes englobantes des sprites distincts
    présents sur l'image (sur fond transparent ou couleur de fond uniforme).
    """
    arr = np.array(img)
    H, W = arr.shape[:2]
    if arr.shape[2] == 4:
        mask = arr[:, :, 3] > 10
    else:
        corners = [arr[0,0], arr[0,-1], arr[-1,0], arr[-1,-1]]
        bg = corners[0]
        diff = np.abs(arr[:, :, :3].astype(int) - bg[:3].astype(int))
        mask = diff.sum(axis=2) > 15

    fg_ratio = np.count_nonzero(mask) / (H * W)
    if fg_ratio < 0.001 or fg_ratio > 0.98:
        return []

    visited = np.zeros((H, W), dtype=bool)
    boxes = []

    for y in range(H):
        for x in range(W):
            if mask[y, x] and not visited[y, x]:
                min_x, max_x = x, x
                min_y, max_y = y, y
                q = [(y, x)]
                visited[y, x] = True
                while q:
                    cy, cx = q.pop()
                    if cx < min_x: min_x = cx
                    elif cx > max_x: max_x = cx
                    if cy < min_y: min_y = cy
                    elif cy > max_y: max_y = cy
                    for dy, dx in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
                        ny, nx = cy + dy, cx + dx
                        if 0 <= ny < H and 0 <= nx < W and mask[ny, nx] and not visited[ny, nx]:
                            visited[ny, nx] = True
                            q.append((ny, nx))
                w = max_x - min_x + 1
                h = max_y - min_y + 1
                if w >= min_size and h >= min_size:
                    boxes.append({'x': int(min_x), 'y': int(min_y), 'w': int(w), 'h': int(h)})

    # Tri naturel (de gauche à droite en regroupant par ligne)
    boxes.sort(key=lambda b: (round(b['y'] / 24), b['x']))
    return boxes

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

        elif parsed.path == "/api/detect_sprites":
            rel_path = data.get("rel_path", "")
            full_path = os.path.join(self.input_dir, rel_path)
            if not os.path.exists(full_path):
                full_path = os.path.join(self.quarantine_dir, rel_path)

            if os.path.exists(full_path):
                try:
                    with Image.open(full_path) as img:
                        boxes = detect_sprites(img)
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps({"ok": True, "boxes": boxes}).encode("utf-8"))
                except Exception as e:
                    self.send_response(500)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps({"ok": False, "error": str(e)}).encode("utf-8"))
            else:
                self.send_response(404)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"ok": False, "error": "Image introuvable"}).encode("utf-8"))

        elif parsed.path == "/api/crop":
            rel_path = data.get("rel_path", "")
            boxes = data.get("boxes", [])
            src = os.path.join(self.input_dir, rel_path)

            if not os.path.exists(src) or not boxes:
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"ok": False, "error": "Source introuvable ou aucun cadre"}).encode("utf-8"))
                return

            try:
                created_files = []
                dir_name = os.path.dirname(src)
                base_name = os.path.splitext(os.path.basename(src))[0]
                ext = os.path.splitext(src)[1]
                if not ext:
                    ext = ".png"

                with Image.open(src) as img:
                    W, H = img.size
                    for i, b in enumerate(boxes):
                        x = max(0, min(W - 1, int(b.get('x', 0))))
                        y = max(0, min(H - 1, int(b.get('y', 0))))
                        w = max(1, min(W - x, int(b.get('w', 1))))
                        h = max(1, min(H - y, int(b.get('h', 1))))

                        cropped = img.crop((x, y, x + w, y + h))

                        crop_filename = f"{base_name}_crop_{i+1:02d}{ext}"
                        crop_full_path = os.path.join(dir_name, crop_filename)
                        
                        idx = i + 1
                        while os.path.exists(crop_full_path):
                            idx += 1
                            crop_filename = f"{base_name}_crop_{idx:02d}{ext}"
                            crop_full_path = os.path.join(dir_name, crop_filename)

                        cropped.save(crop_full_path)
                        created_files.append(crop_full_path)

                dst_archive = os.path.join(self.quarantine_dir, "sliced_composites", rel_path)
                os.makedirs(os.path.dirname(dst_archive), exist_ok=True)
                shutil.move(src, dst_archive)

                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({
                    "ok": True,
                    "created_files": created_files,
                    "original_dst": dst_archive
                }).encode("utf-8"))
            except Exception as e:
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"ok": False, "error": str(e)}).encode("utf-8"))

        elif parsed.path == "/api/undo_crop":
            rel_path = data.get("rel_path", "")
            created_files = data.get("created_files", [])
            original_dst = data.get("original_dst", "")
            original_src = os.path.join(self.input_dir, rel_path)

            for f in created_files:
                if os.path.exists(f):
                    try:
                        os.remove(f)
                    except Exception:
                        pass

            if os.path.exists(original_dst):
                os.makedirs(os.path.dirname(original_src), exist_ok=True)
                shutil.move(original_dst, original_src)

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"ok": True, "action": "crop_undone"}).encode("utf-8"))

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
    print("  🎮 PIXCHRON - MATCH OR PASS REVIEWER AVEC DÉCOUPAGE PRÊT !")
    print("=" * 75)
    print(f"  • Dossier revu       : {args.dir}")
    print(f"  • Dossier Quarantaine: {args.quarantine}")
    print(f"  • Ouvrir dans le web : http://localhost:{args.port}")
    print("=" * 75)
    print("👉 Contrôles de Revue :")
    print("   [ESPACE] ou [→]  : Valider l'image (Keep / Match)")
    print("   [SUPPR]  ou [↓]  : Bouton Rouge (Rejeter vers la quarantaine)")
    print("   [←]      ou [Z]  : Flèche Précédent (Annuler et restaurer)")
    print("👉 Outils de Cadrage & Découpage :")
    print("   [Glisser Souris] : Dessiner un cadre de découpe sur l'image")
    print("   [A]              : ⚡ Cadres Auto (détection automatique de sprites)")
    print("   [C] ou [ENTRÉE]  : ✂️ Découper les cadres actifs et archiver la planche")
    print("   [ÉCHAP]          : Effacer tous les cadres")
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
