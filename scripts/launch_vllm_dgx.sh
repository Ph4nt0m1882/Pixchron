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

# Nettoyer l'ancien conteneur
echo "⚠️ Nettoyage de l'ancien conteneur pixchron_vllm..."
docker stop pixchron_vllm > /dev/null 2>&1 || true
docker rm pixchron_vllm > /dev/null 2>&1 || true

echo "📦 Démarrage du conteneur vllm/vllm-openai:latest..."

docker run -d \
    --name pixchron_vllm \
    --gpus all \
    --ipc=host \
    -p ${PORT}:8000 \
    -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
    -e HF_TOKEN="${HF_TOKEN:-}" \
    vllm/vllm-openai:latest \
    "$MODEL_NAME" \
    --dtype bfloat16 \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --max-model-len 4096 \
    --trust-remote-code

echo "⏳ Initialisation du conteneur..."
echo "💡 Vous pouvez suivre la progression en temps réel avec :"
echo "   docker logs -f pixchron_vllm"
echo ""

ATTEMPTS=0
MAX_ATTEMPTS=300

while [ $ATTEMPTS -lt $MAX_ATTEMPTS ]; do
    # Vérifier si le conteneur est toujours actif
    if ! docker ps -q -f "name=pixchron_vllm" -f "status=running" | grep -q .; then
        echo -e "\n❌ Le conteneur vLLM s'est arrêté ! Voici les dernières lignes de logs :"
        echo "------------------------------------------------------------------"
        docker logs --tail 30 pixchron_vllm || true
        echo "------------------------------------------------------------------"
        exit 1
    fi

    # Vérifier si l'API est prête
    if curl -s "http://localhost:${PORT}/health" > /dev/null 2>&1; then
        echo -e "\n✅ Serveur vLLM opérationnel sur http://localhost:${PORT}/v1 !"
        exit 0
    fi

    echo -n "."
    ATTEMPTS=$((ATTEMPTS + 1))
    sleep 3
done

echo -e "\n⚠️ Délai d'attente dépassé. Vérifiez les logs avec : docker logs pixchron_vllm"
