#!/usr/bin/env bash
# Script pour lancer automatiquement le serveur vLLM dans un conteneur Docker optimisé pour NVIDIA DGX
set -e

MODEL_NAME=${1:-"Qwen/Qwen2.5-VL-7B-Instruct"}
PORT=${2:-8000}
GPU_MEM_UTIL=${3:-0.80}

echo "=================================================================="
echo "🚀 Lancement du serveur vLLM (Docker) pour DGX Grace Blackwell"
echo "=================================================================="
echo "• Modèle          : $MODEL_NAME"
echo "• Port API        : $PORT"
echo "• VRAM allouée    : $(echo "$GPU_MEM_UTIL * 100" | bc 2>/dev/null || echo "80")% des 120 Go"
echo "• Cache Modèles   : $HOME/.cache/huggingface"
echo "=================================================================="

# Vérifier si Docker et le support NVIDIA sont disponibles
if ! command -v docker &> /dev/null; then
    echo "❌ Erreur: Docker n'est pas installé sur cette machine."
    exit 1
fi

# Arrêter un éventuel conteneur vllm déjà actif sur le même port
if docker ps -q --filter "name=pixchron_vllm" | grep -q .; then
    echo "⚠️ Arrêt du conteneur vLLM existant..."
    docker stop pixchron_vllm > /dev/null 2>&1 || true
    docker rm pixchron_vllm > /dev/null 2>&1 || true
fi

echo "📦 Démarrage du conteneur vllm/vllm-openai:latest..."

docker run -d \
    --name pixchron_vllm \
    --gpus all \
    --ipc=host \
    --restart unless-stopped \
    -p ${PORT}:8000 \
    -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
    -e HF_TOKEN="${HF_TOKEN:-}" \
    vllm/vllm-openai:latest \
    --model "$MODEL_NAME" \
    --dtype bfloat16 \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --max-model-len 4096 \
    --trust-remote-code \
    --limit-mm-per-prompt image=1

echo "⏳ Attente du chargement du modèle vLLM en VRAM..."
until curl -s "http://localhost:${PORT}/health" > /dev/null 2>&1; do
    echo -n "."
    sleep 3
done

echo -e "\n✅ Serveur vLLM prêt et opérationnel sur http://localhost:${PORT}/v1 !"
