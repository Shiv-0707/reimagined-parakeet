# J.A.R.V.I.S — Just A Rather Very Intelligent System

A lightweight, self-contained, *deployable* AI assistant web app in the spirit
of Tony Stark's JARVIS. It runs a fast, **offline-capable** local brain out of
the box (no API key needed), and can be upgraded to **full conversational AI**
by connecting any OpenAI-compatible API key (OpenAI, OpenRouter, Groq, or a
custom endpoint such as Ollama / LM Studio / vLLM).

![stack](https://img.shields.io/badge/Python-3.9%2B-3df0ff)
![deps](https://img.shields.io/badge/deps-zero-blue)
![web](https://img.shields.io/badge/UI-Web-green)

---

## What Jarvis can do — right now, with zero keys

| Skill | Try saying |
| --- | --- |
| Greetings / identity | `hello`, `who are you` |
| Time & date | `what time is it`, `what's the date` |
| Math (safe parser) | `what is sqrt(2) + 3`, `calculate 15% of 2400` |
| Unit conversion | `convert 5 miles to km`, `how many kg in 200 pounds` |
| Temperature conversion | `convert 100 fahrenheit to celsius` |
| **Live weather** (no key) | `weather in London` |
| **Currency FX** (no key) | `convert 100 USD to EUR` |
| **Wikipedia summaries** | `tell me about the Apollo program` |
| Fun | `tell me a joke` |
| Help | `help` |

The offline brain never executes your input — math is parsed into a safe AST
(no `eval`), and all network tools gracefully degrade if offline.

## Full conversational AI

For open-ended, deep conversation, open the **⚙ Settings** panel and connect a
key (OpenAI / OpenRouter / Groq / custom). You can also set the key once as an
environment variable on the server:

```bash
export OPENAI_API_KEY=sk-...
# or
export OPENROUTER_API_KEY=...
export GROQ_API_KEY=...
```

The server stores the key for the process lifetime (and locally in
`config.json`, which is git-ignored). Per-browser settings persist in
`localStorage` and are sent with each request.

---

## Run it

**No dependencies** — just Python 3.9+:

```bash
python main.py            # serves on http://0.0.0.0:8000
```

or

```bash
PORT=8080 python main.py
```

Open the printed URL in a browser.

## With Docker

```bash
docker build -t jarvis .
docker run -p 8000:8000 -e PORT=8000 jarvis
```

## Deploy to a host

Because it's a single self-contained Python service with zero dependencies, it
drops onto nearly any host:

* **Render / Railway / Fly.io / Heroku** — set the run command to `python main.py`
  (or `gunicorn`-free; use `python`) and set `PORT` to the platform-provided
  port. Add your `OPENAI_API_KEY` as a secret for full AI mode.
* **Any VPS** — `nohup python main.py &` behind a reverse proxy.

See `Dockerfile` for a minimal container reference.

---

## Project layout

```
main.py            # entry point  ->  python main.py
jarvis/
  server.py        # zero-dep HTTP server + /api/chat + /api/config
  brain.py         # local Jarvis brain (tools + router)
public/
  index.html       # the web UI (dark, arc-reactor themed)
config.json        # runtime LLM settings (created, git-ignored)
Dockerfile         # minimal deployable container
```

## API

* `POST /api/chat` — body `{"message":"...", "config":{...}, "history":[...]}` →
  `{"reply":"...", "mode":"llm"|"local"}`
* `GET/POST /api/config` — read/update the server LLM config (key masked on GET)

## Security notes

* Math is evaluated via a hand-written recursive-descent parser → **no `eval`/`exec`**.
* Static file serving is restricted to the `public/` directory (path traversal blocked).
* Your API key is masked on the wire back to the browser and never committed to git.

---

Built as a clean, hackable base — extend the `brain.py` skill table or point
the UI at any model you like.
