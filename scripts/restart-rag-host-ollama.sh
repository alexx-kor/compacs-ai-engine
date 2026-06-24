#!/usr/bin/env bash
# Restart RAG host-ollama stack (Git Bash).
set -euo pipefail
cd "$(dirname "$0")/.."

PORT="${RAG_GATEWAY_PORT:-3090}"
export RAG_GATEWAY_PORT="$PORT"

docker compose -f rag-compose.host-ollama.yml up -d --remove-orphans
echo "Gateway: http://localhost:${PORT}/health"
curl -sf "http://localhost:${PORT}/health" | head -c 200
echo
