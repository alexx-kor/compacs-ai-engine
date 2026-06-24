# Bind Ollama on all interfaces so Docker (host.docker.internal) can run /api/chat.
# PowerShell: powershell -ExecutionPolicy Bypass -File scripts/setup-ollama-docker-windows.ps1

$ErrorActionPreference = "Stop"

Write-Host "Setting user env OLLAMA_HOST=0.0.0.0 ..."
[Environment]::SetEnvironmentVariable("OLLAMA_HOST", "0.0.0.0", "User")
$env:OLLAMA_HOST = "0.0.0.0"

Write-Host "Stopping Ollama processes ..."
Get-Process -Name "ollama*" -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 2

$ollama = Get-Command ollama -ErrorAction SilentlyContinue
if (-not $ollama) {
    Write-Warning "ollama not in PATH. Quit Ollama from the tray, set OLLAMA_HOST=0.0.0.0, restart Ollama."
    exit 1
}

Write-Host "Starting: ollama serve (OLLAMA_HOST=0.0.0.0) ..."
Start-Process -FilePath $ollama.Source -ArgumentList "serve" -WindowStyle Hidden

$ready = $false
for ($i = 0; $i -lt 30; $i++) {
    try {
        $null = Invoke-RestMethod -Uri "http://127.0.0.1:11434/api/tags" -TimeoutSec 2
        $ready = $true
        break
    } catch {
        Start-Sleep -Seconds 1
    }
}
if (-not $ready) {
    Write-Error "Ollama did not start on :11434 within 30s"
    exit 1
}
Write-Host "Ollama is up."

Write-Host "Host chat smoke test (60s timeout) ..."
$body = '{"model":"llama3.2:3b","messages":[{"role":"user","content":"Say ok"}],"stream":false}'
try {
    $resp = Invoke-RestMethod -Uri "http://127.0.0.1:11434/api/chat" -Method Post -Body $body -ContentType "application/json" -TimeoutSec 60
    $text = $resp.message.content
    Write-Host "Host chat:" $text
    if (-not $text) { throw "empty host chat response" }
} catch {
    Write-Warning "Host chat test failed. Try: ollama run llama3.2:3b ok"
    Write-Warning $_.Exception.Message
}

$container = (docker ps --filter "name=rag-engine" --format "{{.Names}}" 2>$null | Select-Object -First 1)
if ($container) {
    Write-Host "Docker chat check from $container ..."
    $py = "import ollama; c=ollama.Client(host='http://host.docker.internal:11434', timeout=90); t=c.chat(model='llama3.2:3b', messages=[{'role':'user','content':'Say ok'}])['message']['content']; print('container chat:', repr(t)); import sys; sys.exit(0 if str(t).strip() else 1)"
    docker exec $container python -c $py
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Docker cannot chat with Ollama. Restart Ollama tray app after OLLAMA_HOST=0.0.0.0."
        exit 1
    }
    Write-Host "OK: Ollama reachable from Docker for chat."
} else {
    Write-Host "rag-engine not running. Start stack after this script."
}

Write-Host ""
Write-Host "Next:"
Write-Host "  RAG_GATEWAY_PORT=3090 docker compose -f rag-compose.host-ollama.yml up -d"
Write-Host "  RAG_GATEWAY_URL=http://localhost:3090 python scripts/manual_api_check.py --skip-slow"
