#!/usr/bin/env bash
# Ollama for Docker on Windows — run in Git Bash (no PowerShell).
set -euo pipefail

echo "=== 1. Stop Ollama (quit tray app if still running) ==="
taskkill //F //IM ollama.exe 2>/dev/null || true
sleep 2

echo "=== 2. Start Ollama on 0.0.0.0 (Docker can reach it) ==="
export OLLAMA_HOST=0.0.0.0
nohup ollama serve >/tmp/ollama-serve.log 2>&1 &
sleep 3

echo "=== 3. Host check ==="
curl -sf http://127.0.0.1:11434/api/tags | head -c 120
echo
echo

echo "=== 4. Quick chat test (not full warm-up) ==="
curl -sf http://127.0.0.1:11434/api/chat \
  -H 'Content-Type: application/json' \
  -d '{"model":"llama3.2:3b","messages":[{"role":"user","content":"ok"}],"stream":false}' \
  | head -c 200 || echo "(model still loading — wait 30s and retry curl)"
echo
echo

CONTAINER=$(docker ps --filter name=rag-engine --format '{{.Names}}' 2>/dev/null | head -1)
if [[ -n "$CONTAINER" ]]; then
  echo "=== 5. Docker check from $CONTAINER ==="
  docker exec "$CONTAINER" python -c "
import ollama
c = ollama.Client(host='http://host.docker.internal:11434', timeout=90)
t = c.chat(model='llama3.2:3b', messages=[{'role':'user','content':'ok'}])['message']['content']
print('container:', repr(t))
import sys
sys.exit(0 if str(t).strip() else 1)
"
  echo "OK: Docker -> Ollama works"
else
  echo "rag-engine not running yet — start compose next"
fi

echo
echo "=== Next (same Git Bash) ==="
echo "  cd ~/Desktop/compacs-ai-engine-develop"
echo "  RAG_GATEWAY_PORT=3090 docker compose -f rag-compose.host-ollama.yml up -d"
echo "  curl http://localhost:3090/health"
echo "  RAG_GATEWAY_URL=http://localhost:3090 python scripts/manual_api_check.py --skip-slow"
