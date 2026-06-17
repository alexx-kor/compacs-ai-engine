"""External gateway (port 3080): UI + proxy to RAG engine (port 8080)."""

from __future__ import annotations

import html
import json
import os
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

import httpx
from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse

ENGINE_URL = os.getenv("RAG_ENGINE_URL", "http://127.0.0.1:8080").rstrip("/")

_NAV = """
<nav>
  <a href="/">Чат</a>
  <a href="/#load">Загрузка</a>
  <a href="/sources">Источники</a>
  <a href="/metrics">Метрики</a>
  <a href="/export">Экспорт</a>
</nav>
"""

_BASE_STYLE = """
<style>
  :root { font-family: system-ui, sans-serif; color: #1a1a1a; background: #f6f7fb; }
  body { max-width: 960px; margin: 0 auto; padding: 24px; }
  nav { display: flex; gap: 16px; margin-bottom: 24px; flex-wrap: wrap; }
  nav a { color: #1d4ed8; text-decoration: none; font-weight: 600; }
  .card { background: #fff; border-radius: 12px; padding: 20px; box-shadow: 0 1px 4px rgba(0,0,0,.08); margin-bottom: 20px; }
  textarea, input, select, button { font: inherit; }
  textarea, input, select { width: 100%; box-sizing: border-box; margin: 8px 0 16px; padding: 10px; }
  button { background: #1d4ed8; color: #fff; border: 0; border-radius: 8px; padding: 10px 16px; cursor: pointer; }
  button.danger { background: #dc2626; }
  pre { white-space: pre-wrap; background: #0f172a; color: #e2e8f0; padding: 16px; border-radius: 8px; }
  .muted { color: #64748b; font-size: 14px; }
  .row { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
  table { width: 100%; border-collapse: collapse; }
  th, td { text-align: left; padding: 8px; border-bottom: 1px solid #e2e8f0; font-size: 14px; }
  @media (max-width: 700px) { .row { grid-template-columns: 1fr; } }
</style>
"""


def _page(title: str, body: str) -> str:
    return (
        f"<!DOCTYPE html><html><head><meta charset='utf-8'>"
        f"<title>{title}</title>{_BASE_STYLE}</head><body>{_NAV}{body}</body></html>"
    )


@asynccontextmanager
async def _lifespan(_app: FastAPI) -> AsyncIterator[None]:
    _app.state.http = httpx.AsyncClient(base_url=ENGINE_URL, timeout=120.0)
    yield
    await _app.state.http.aclose()


app_gateway = FastAPI(title="COMPACS RAG Gateway", version="2.0", lifespan=_lifespan)


async def _engine_request(
    request: Request,
    method: str,
    path: str,
    *,
    json_body: Any | None = None,
    content: bytes | None = None,
    headers: dict[str, str] | None = None,
    params: dict[str, str] | None = None,
) -> httpx.Response:
    client: httpx.AsyncClient = request.app.state.http
    try:
        return await client.request(
            method,
            path,
            json=json_body,
            content=content,
            headers=headers,
            params=params,
        )
    except httpx.RequestError as error:
        raise HTTPException(
            status_code=503,
            detail=f"RAG engine unavailable at {ENGINE_URL}: {error}",
        ) from error


def _proxy_response(response: httpx.Response) -> Response:
    if response.headers.get("content-type", "").startswith("application/json"):
        try:
            return JSONResponse(status_code=response.status_code, content=response.json())
        except json.JSONDecodeError:
            pass
    return Response(
        content=response.content,
        status_code=response.status_code,
        media_type=response.headers.get("content-type"),
        headers={
            key: value
            for key, value in response.headers.items()
            if key.lower() in {"content-disposition", "content-length", "content-type"}
        },
    )


@app_gateway.get("/health")
async def gateway_health(request: Request) -> dict[str, Any]:
    engine = await _engine_request(request, "GET", "/health")
    payload = engine.json() if engine.headers.get("content-type", "").startswith("application/json") else {}
    return {
        "status": "healthy" if engine.status_code == 200 else "degraded",
        "gateway_port": int(os.getenv("GATEWAY_PORT", "3080")),
        "engine_url": ENGINE_URL,
        "engine": payload,
    }


@app_gateway.get("/", response_class=HTMLResponse)
async def chat_page() -> HTMLResponse:
    body = """
<div class="card">
<h1>COMPACS RAG — чат</h1>
<p class="muted">Клиент → :3080 (gateway) → :8080 (engine + Ollama)</p>
<label>Активные папки (через запятую, пусто = все)</label>
<input id="folders" placeholder="ui-ext, operator-manual">
<label>Вопрос</label>
<textarea id="question" rows="4" placeholder="Задайте вопрос по документации..."></textarea>
<button id="ask">Спросить</button>
<pre id="answer">Ответ появится здесь.</pre>
</div>
<div class="card" id="load">
<h2>Загрузка документов</h2>
<p class="muted">POST /load — ingestion и переиндексация в выбранную папку.</p>
<div class="row">
  <div>
    <h3>Новая папка</h3>
    <input id="newId" placeholder="id (ops-manual)">
    <input id="newName" placeholder="Название">
    <button id="createFolder">Создать</button>
  </div>
  <div>
    <h3>Выбор папок для RAG</h3>
    <select id="selection" multiple size="6"></select>
    <button id="saveSelection">Применить выбор</button>
  </div>
</div>
<select id="uploadFolder"></select>
<input id="file" type="file" accept=".pdf,.txt,.md,.rst">
<button id="uploadBtn">Загрузить (POST /load)</button>
<pre id="loadLog"></pre>
</div>
<script>
async function refreshCollections() {
  const res = await fetch('/v1/collections');
  const data = await res.json();
  const selected = new Set(data.selected_collection_ids || []);
  const sel = document.getElementById('selection');
  const upload = document.getElementById('uploadFolder');
  sel.innerHTML = '';
  upload.innerHTML = '';
  for (const c of data.collections || []) {
    const opt = document.createElement('option');
    opt.value = c.id;
    opt.textContent = `${c.name} (${c.document_count})`;
    opt.selected = selected.has(c.id);
    sel.appendChild(opt);
    upload.appendChild(opt.cloneNode(true));
  }
}
document.getElementById('ask').onclick = async () => {
  const question = document.getElementById('question').value.trim();
  const foldersRaw = document.getElementById('folders').value.trim();
  if (!question) return;
  const payload = { question, stream: true };
  if (foldersRaw) payload.collection_ids = foldersRaw.split(',').map(s => s.trim()).filter(Boolean);
  const answerEl = document.getElementById('answer');
  answerEl.textContent = 'Поиск контекста...';
  const res = await fetch('/api/chat', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
  if (!res.ok) { answerEl.textContent = (await res.json()).detail || res.statusText; return; }
  if (!res.body) { answerEl.textContent = await res.text(); return; }
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let answer = '';
  let sources = [];
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const parts = buffer.split('\\n\\n');
    buffer = parts.pop() || '';
    for (const block of parts) {
      const lines = block.split('\\n');
      let event = 'message';
      let data = '';
      for (const line of lines) {
        if (line.startsWith('event: ')) event = line.slice(7);
        if (line.startsWith('data: ')) data = line.slice(6);
      }
      if (!data) continue;
      const parsed = JSON.parse(data);
      if (event === 'status' && parsed.phase === 'retrieval') answerEl.textContent = 'Генерация ответа...';
      if (event === 'token') { answer += parsed.text || ''; answerEl.textContent = answer; }
      if (event === 'done') {
        answer = parsed.answer || answer;
        sources = parsed.sources || [];
      }
    }
  }
  const srcText = sources.map(s => `- ${s.source || s[0]}, p.${s.page || s[1]}`).join('\\n');
  answerEl.textContent = answer + (srcText ? '\\n\\nSources:\\n' + srcText : '');
};
document.getElementById('createFolder').onclick = async () => {
  const id = document.getElementById('newId').value.trim();
  const name = document.getElementById('newName').value.trim();
  const res = await fetch('/v1/collections', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ id: id || undefined, name: name || id }) });
  document.getElementById('loadLog').textContent = JSON.stringify(await res.json(), null, 2);
  await refreshCollections();
};
document.getElementById('saveSelection').onclick = async () => {
  const ids = [...document.getElementById('selection').selectedOptions].map(o => o.value);
  const res = await fetch('/v1/collections/selection', { method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ collection_ids: ids }) });
  document.getElementById('loadLog').textContent = JSON.stringify(await res.json(), null, 2);
};
document.getElementById('uploadBtn').onclick = async () => {
  const file = document.getElementById('file').files[0];
  const collectionId = document.getElementById('uploadFolder').value;
  if (!file || !collectionId) return;
  const form = new FormData();
  form.append('file', file);
  const res = await fetch(`/load?collection_id=${encodeURIComponent(collectionId)}&background=true`, { method: 'POST', body: form });
  const data = await res.json();
  if (res.status === 202 && data.job_id) {
    document.getElementById('loadLog').textContent = 'Индексация в фоне: ' + data.job_id;
    const poll = async () => {
      const st = await fetch('/load/' + encodeURIComponent(data.job_id));
      const body = await st.json();
      document.getElementById('loadLog').textContent = JSON.stringify(body, null, 2);
      if (body.status === 'pending' || body.status === 'running') setTimeout(poll, 1500);
    };
    poll();
  } else {
    document.getElementById('loadLog').textContent = JSON.stringify(data, null, 2);
  }
  await refreshCollections();
};
refreshCollections();
</script>
"""
    return HTMLResponse(_page("COMPACS RAG", body))


@app_gateway.post("/api/chat")
async def chat_api(request: Request) -> Response:
    payload = await request.json()
    if payload.get("stream"):
        client: httpx.AsyncClient = request.app.state.http

        async def event_stream() -> AsyncIterator[bytes]:
            try:
                async with client.stream(
                    "POST",
                    "/v1/query",
                    json=payload,
                    timeout=120.0,
                ) as response:
                    if response.status_code >= 400:
                        body = await response.aread()
                        raise HTTPException(
                            status_code=response.status_code,
                            detail=body.decode(errors="replace"),
                        )
                    async for chunk in response.aiter_bytes():
                        yield chunk
            except httpx.RequestError as error:
                raise HTTPException(status_code=503, detail=str(error)) from error

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    response = await _engine_request(request, "POST", "/v1/query", json_body=payload)
    return JSONResponse(status_code=response.status_code, content=response.json())


@app_gateway.post("/load")
async def load_upload(
    request: Request,
    collection_id: str = Query(...),
    background: bool = Query(default=False, description="202 + job_id; poll GET /load/{job_id}"),
    file: UploadFile = File(...),
) -> JSONResponse:
    """Upload document → engine ingestion → reindex."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="filename is required")
    content = await file.read()
    files = {"file": (file.filename, content, file.content_type or "application/octet-stream")}
    client: httpx.AsyncClient = request.app.state.http
    try:
        response = await client.post(
            f"/v1/collections/{collection_id}/documents",
            files=files,
            params={"background": str(background).lower()},
        )
    except httpx.RequestError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    status = 202 if background and response.status_code == 200 else response.status_code
    return JSONResponse(status_code=status, content=response.json())


@app_gateway.get("/load/{job_id}")
async def load_job_status(job_id: str, request: Request) -> JSONResponse:
    """Poll background ingestion job started via POST /load?background=true."""
    response = await _engine_request(request, "GET", f"/v1/jobs/{job_id}")
    return JSONResponse(status_code=response.status_code, content=response.json())


@app_gateway.get("/sources")
async def sources_list(request: Request, format: str = Query(default="html")) -> Response:
    """List sources — JSON API or HTML UI."""
    response = await _engine_request(request, "GET", "/sources")
    if response.status_code >= 400:
        return _proxy_response(response)
    data = response.json()
    if format == "json" or "application/json" in request.headers.get("accept", ""):
        return JSONResponse(content=data)
    rows = "".join(
        f"<tr><td>{html.escape(s['filename'] or s['source'])}</td>"
        f"<td>{s['chunk_count']}</td>"
        f"<td>{html.escape(s.get('collection_id') or '-')}</td>"
        f"<td><a href='/sources/{html.escape(s['id'])}/download'>скачать</a></td>"
        f"<td><button class='danger' onclick=\"deleteSource('{html.escape(s['id'])}')\">удалить</button></td></tr>"
        for s in data.get("sources", [])
    )
    body = f"""
<div class="card">
<h1>Источники</h1>
<p class="muted">GET /sources — JSON: <a href="/sources?format=json">/sources?format=json</a></p>
<table><thead><tr><th>Файл</th><th>Chunks</th><th>Папка</th><th></th><th></th></tr></thead><tbody>{rows or '<tr><td colspan=5>нет источников</td></tr>'}</tbody></table>
</div>
<script>
async function deleteSource(id) {{
  if (!confirm('Удалить источник и переиндексировать?')) return;
  const res = await fetch('/sources/' + encodeURIComponent(id), {{ method: 'DELETE' }});
  alert(JSON.stringify(await res.json()));
  location.reload();
}}
</script>
"""
    return HTMLResponse(_page("Источники", body))


@app_gateway.get("/sources/{source_id}/download")
async def sources_download(source_id: str, request: Request) -> Response:
    response = await _engine_request(request, "GET", f"/sources/{source_id}/download")
    return _proxy_response(response)


@app_gateway.delete("/sources/{source_id}")
async def sources_delete(source_id: str, request: Request) -> Response:
    response = await _engine_request(request, "DELETE", f"/sources/{source_id}")
    return _proxy_response(response)


@app_gateway.get("/export")
async def export_index(request: Request) -> Response:
    response = await _engine_request(request, "GET", "/v1/export", params={"format": "jsonl"})
    return _proxy_response(response)


@app_gateway.get("/metrics")
async def metrics_page(request: Request, format: str = Query(default="html")) -> Response:
    response = await _engine_request(request, "GET", "/v1/metrics")
    if response.status_code >= 400:
        raise HTTPException(status_code=response.status_code, detail=response.text)
    data = response.json()
    if format == "json" or "application/json" in request.headers.get("accept", ""):
        return JSONResponse(content=data)
    pretty = html.escape(json.dumps(data, ensure_ascii=False, indent=2))
    quality = data.get("quality", {})
    verdict = quality.get("overall_verdict", "unavailable")
    body = f"""<div class="card"><h1>Метрики</h1>
<p><strong>PSI / деградация:</strong> <code>{html.escape(str(verdict))}</code></p>
<pre>{pretty}</pre>
<p class="muted">Offline: <code>python scripts/monitor_data_drift.py --preset splits</code> · JSON: <a href="/metrics?format=json">/metrics?format=json</a></p></div>"""
    return HTMLResponse(_page("Метрики", body))


@app_gateway.post("/upgrade")
async def upgrade_pro(request: Request) -> Response:
    payload = await request.json()
    response = await _engine_request(request, "POST", "/upgrade", json_body=payload)
    return _proxy_response(response)


@app_gateway.api_route("/v1/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def proxy_v1(path: str, request: Request) -> Response:
    body = await request.body()
    headers = {
        key: value
        for key, value in request.headers.items()
        if key.lower() not in {"host", "content-length"}
    }
    response = await _engine_request(
        request,
        request.method,
        f"/v1/{path}",
        content=body if body else None,
        headers=headers,
        params=dict(request.query_params),
    )
    return _proxy_response(response)


@app_gateway.get("/{full_path:path}")
async def unknown_get(full_path: str) -> None:
    raise HTTPException(status_code=404, detail="not found")
