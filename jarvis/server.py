"""
JARVIS - HTTP server
====================
Zero-dependency server (Python standard library). It serves the web UI from
``public/`` and exposes a small JSON API:

    GET  /                 -> the web app
    GET  /api/config       -> current LLM config (key masked)
    POST /api/config       -> update LLM config
    POST /api/chat         -> {message, config?, history?} -> {reply, mode}

LLM behaviour: Jarvis answers with the LOCAL brain by default. If an
OpenAI-compatible API key is set (via env, config endpoint, or per-request
config) it calls the chosen provider for rich conversational answers.

Run:  PORT=8000 python -m jarvis.server   (or: python main.py)
"""

from __future__ import annotations

import json
import os
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import brain

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
_HOST = os.environ.get("HOST", "0.0.0.0")
_PORT = int(os.environ.get("PORT", "8000"))

DEFAULT_PROVIDERS = {
    "openai": {
        "label": "OpenAI",
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o-mini",
        "key_env": "OPENAI_API_KEY",
    },
    "openrouter": {
        "label": "OpenRouter",
        "base_url": "https://openrouter.ai/api/v1",
        "model": "openai/gpt-4o-mini",
        "key_env": "OPENROUTER_API_KEY",
    },
    "groq": {
        "label": "Groq",
        "base_url": "https://api.groq.com/openai/v1",
        "model": "llama-3.3-70b-versatile",
        "key_env": "GROQ_API_KEY",
    },
    "custom": {
        "label": "Custom (OpenAI-compatible)",
        "base_url": "http://localhost:11434/v1",
        "model": "llama3.2",
        "key_env": "CUSTOM_OPENAI_BASE_URL",
    },
}

SYSTEM_PROMPT = (
    "You are JARVIS — Just A Rather Very Intelligent System — a polished, "
    "witty, helpful AI assistant. You address the user as 'sir' or 'madam' "
    "with warmth and precision. Be concise, correct and helpful. If you "
    "cannot know something, say so plainly rather than guessing."
)

# --------------------------------------------------------------------------- #
# In-memory state (seeded from env + optional config file)
# --------------------------------------------------------------------------- #
_CFG_FILE = os.environ.get("JARVIS_CONFIG", os.path.join(os.path.dirname(__file__), "..", "config.json"))
_cfg_lock = threading.Lock()


def _load_config():
    base = dict(provider="openai", base_url="", model="", api_key="")
    # seed from env keys
    for name, p in DEFAULT_PROVIDERS.items():
        val = os.environ.get(p["key_env"])
        if val:
            base["provider"] = name
            base["api_key"] = val
            base["base_url"] = p["base_url"]
            base["model"] = p["model"]
            break
    try:
        with open(_CFG_FILE) as fh:
            saved = json.load(fh)
        base.update(saved)
    except Exception:  # noqa: BLE001 - no config file yet
        pass
    return base


_config = _load_config()


def _save_config():
    try:
        with open(_CFG_FILE, "w") as fh:
            json.dump(_config, fh, indent=2)
    except Exception:  # noqa: BLE001
        pass


def get_config(public=False):
    with _cfg_lock:
        c = dict(_config)
    if public:
        c["api_key"] = "••• set •••" if c.get("api_key") else ""
    return c


def set_config(patch: dict):
    with _cfg_lock:
        provider = patch.get("provider")
        if provider in DEFAULT_PROVIDERS:
            _config["provider"] = provider
            if provider != "custom":
                p = DEFAULT_PROVIDERS[provider]
                if not patch.get("base_url") or patch.get("base_url") == _config.get("base_url"):
                    _config["base_url"] = p["base_url"]
                if not patch.get("model"):
                    _config["model"] = p["model"]
        for key in ("api_key", "base_url", "model"):
            if key in patch and patch[key] is not None:
                _config[key] = patch[key].strip()
        _save_config()
        return dict(_config)


def _resolve_llm(per_request):
    """Merge per-request config on top of server config; return (base_url, model, key) or None."""
    with _cfg_lock:
        base = dict(_config)
    if per_request:
        provider = per_request.get("provider") or base.get("provider")
        merged = dict(base)
        if provider in DEFAULT_PROVIDERS:
            merged["provider"] = provider
            if not per_request.get("base_url"):
                merged["base_url"] = DEFAULT_PROVIDERS[provider]["base_url"]
            if not per_request.get("model"):
                merged["model"] = DEFAULT_PROVIDERS[provider]["model"]
        for key in ("api_key", "base_url", "model"):
            if per_request.get(key):
                merged[key] = per_request[key].strip()
        base = merged
    key = (base.get("api_key") or "").strip()
    url = (base.get("base_url") or "").strip()
    if not key or not url:
        return None
    return url, (base.get("model") or "").strip(), key


def _chat_via_llm(url, model, key, user_message, history):
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        # keep last ~10 turns of context
        for h in history[-10:]:
            role = "user" if h.get("role") == "user" else "assistant"
            msgs.append({"role": role, "content": h.get("content", "")})
    msgs.append({"role": "user", "content": user_message})
    endpoint = url.rstrip("/") + "/chat/completions"
    payload = json.dumps({
        "model": model,
        "messages": msgs,
        "temperature": 0.7,
        "max_tokens": 700,
    }).encode("utf-8")
    req = urllib.request.Request(
        endpoint, data=payload,
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {key}",
                 "User-Agent": "JARVIS/1.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read().decode("utf-8", "replace"))
    return data["choices"][0]["message"]["content"].strip()


# --------------------------------------------------------------------------- #
# Request handler
# --------------------------------------------------------------------------- #
class _Handler(BaseHTTPRequestHandler):
    server_version = "JARVIS/1.0"

    def log_message(self, fmt, *args):  # quieter logs
        if self.path.startswith("/api/"):
            print("[jarvis] %s - %s" % (self.address_string(), fmt % args))

    def _send(self, code, body=b"", ctype="application/json; charset=utf-8",
              extra_headers=None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
        for k, v in (extra_headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD" and body:
            self.wfile.write(body)

    def do_OPTIONS(self):
        self._send(204)

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path in ("/api/config", "/api/config/"):
            self._send(200, json.dumps(get_config(public=True)).encode())
            return
        # static assets (only from public/, and safe)
        if path == "/":
            path = "/index.html"
        base = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "public")
        rel = os.path.normpath(path.lstrip("/"))
        full = os.path.join(base, rel)
        if not full.startswith(os.path.abspath(base)) or not os.path.isfile(full):
            self._send(404, b"not found", "text/plain; charset=utf-8")
            return
        ctype = {
            ".html": "text/html; charset=utf-8",
            ".css": "text/css; charset=utf-8",
            ".js": "application/javascript; charset=utf-8",
            ".png": "image/png", ".jpg": "image/jpeg", ".svg": "image/svg+xml",
            ".ico": "image/x-icon", ".json": "application/json",
            ".woff2": "font/woff2",
        }.get(os.path.splitext(full)[1].lower(), "application/octet-stream")
        with open(full, "rb") as fh:
            body = fh.read()
        self._send(200, body, ctype)

    def do_POST(self):
        path = self.path.split("?", 1)[0]
        try:
            raw = self.rfile.read(int(self.headers.get("Content-Length") or 0))
            data = json.loads(raw.decode("utf-8")) if raw else {}
        except Exception:  # noqa: BLE001
            data = {}
        if path in ("/api/config", "/api/config/"):
            cfg = set_config(data)
            self._send(200, json.dumps(cfg | {"api_key": "••• set •••" if cfg.get("api_key") else ""}).encode())
            return
        if path in ("/api/chat", "/api/chat/"):
            self._handle_chat(data)
            return
        self._send(404, json.dumps({"error": "not found"}).encode())

    def _handle_chat(self, data):
        message = (data.get("message") or "").strip()
        if not message:
            self._send(400, json.dumps({"reply": "Please say something first.", "mode": "local"}).encode())
            return
        per_request = data.get("config") or {}
        history = data.get("history") or []
        llm = _resolve_llm(per_request)
        mode = "local"
        try:
            if llm:
                base, model, key = llm
                reply = _chat_via_llm(base, model, key, message, history)
                mode = "llm"
            else:
                reply = brain.respond(message)
        except Exception as exc:  # noqa: BLE001
            reply = ("I tried to reach my neural core but hit an error: "
                     f"{exc}. Please check your API key / endpoint in Settings.")
            mode = "local-error"
        body = json.dumps({"reply": reply, "mode": mode}).encode()
        self._send(200, body)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main():
    server = ThreadingHTTPServer((_HOST, _PORT), _Handler)
    print(f"[jarvis] JARVIS online at http://{_HOST}:{_PORT}")
    print(f"[jarvis] LLM: {'configured' if get_config().get('api_key') else 'local mode (connect a key in Settings for full AI)'}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[jarvis] Shutting down.")


if __name__ == "__main__":
    main()
