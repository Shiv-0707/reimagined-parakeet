"""
JARVIS - Local brain
====================
A dependency-free, Jarvis-style assistant core. It understands a set of
intents and answers with real tools (time, date, math, weather, currency,
Wikipedia, etc.). No API key is required for these local skills.

If an OpenAI-compatible API key is supplied (via settings or env), Jarvis
switches to full conversational AI mode handled in `server.py`; this module
remains the offline fallback / always-available skill engine.
"""

from __future__ import annotations

import datetime as _dt
import json
import math
import operator
import random
import re
import urllib.parse
import urllib.request

# --------------------------------------------------------------------------- #
# Small HTTP helper (stdlib only, so we run anywhere).
# --------------------------------------------------------------------------- #
_USER_AGENT = "JARVIS/1.0 (local assistant)"


def _http_json(url: str, timeout: float = 8.0):
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8", "replace"))


def _get(url: str, timeout: float = 8.0):
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", "replace")


# --------------------------------------------------------------------------- #
# Tool: math (safe AST evaluator - never eval()s raw user input)
# --------------------------------------------------------------------------- #
_ALLOWED = {
    "Add": operator.add, "Sub": operator.sub, "Mul": operator.mul,
    "Div": operator.truediv, "Pow": operator.pow, "Mod": operator.mod,
    "Neg": operator.neg, "Pos": operator.pos, "Abs": operator.abs,
    "FloorDiv": operator.floordiv, "sqrt": math.sqrt, "sin": math.sin,
    "cos": math.cos, "tan": math.tan, "pi": math.pi, "e": math.e,
    "log": math.log, "log10": math.log10, "floor": math.floor,
    "ceil": math.ceil, "round": round,
}


def evaluate_math(expr: str):
    """Return (ok, result_string). Never executes arbitrary code."""
    tree = _parse_math(expr)
    if tree is None:
        return False, None
    try:
        value = _eval_node(tree)
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)
    if isinstance(value, float):
        if not math.isfinite(value):
            return True, "undefined (division result is not finite)"
        value = round(value, 10)
        if value == int(value):
            value = int(value)
    return True, str(value)


def _parse_math(expr: str):
    """Parse a math expression into a node tree without executing code."""
    expr = expr.strip()
    if not expr:
        return None
    if not re.fullmatch(r"[0-9+\-*/%().,\s a-zA-Z_]*", expr):
        return None
    if re.search(r"[a-zA-Z_]+", expr) and " " not in expr.strip():
        # could still be a function call like sqrt(4) - allow; tokenise below
        pass
    # Reject anything that looks like Python import/attribute/function attack.
    for bad in ["__", "import", "lambda", "exec", "eval", "open", "globals"]:
        if bad in expr:
            return None
    try:
        return _make_parser(expr).parse_expression()
    except Exception:  # noqa: BLE001
        return None


def _make_parser(text: str):
    from io import StringIO

    class _P:
        def __init__(self, s):
            self.toks = list(re.finditer(r"\d+\.?\d*|[A-Za-z_]\w*|[()+\-*/%,]", s))
            self.i = 0

        def peek(self):
            return self.toks[self.i].group() if self.i < len(self.toks) else None

        def next(self):
            t = self.peek()
            self.i += 1
            return t

        def parse_expression(self):
            node = self.parse_term()
            while self.peek() in ("+", "-"):
                op = self.next()
                rhs = self.parse_term()
                node = (op, node, rhs)
            return node

        def parse_term(self):
            node = self.parse_factor()
            while self.peek() in ("*", "/", "%"):
                op = self.next()
                rhs = self.parse_factor()
                node = (op, node, rhs)
            return node

        def parse_factor(self):
            t = self.next()
            if t == "-":
                return ("Neg", self.parse_factor())
            if t == "+":
                return ("Pos", self.parse_factor())
            if t == "(":
                node = self.parse_expression()
                if self.peek() == ")":
                    self.next()
                return node
            if t is None:
                raise ValueError("empty")
            if re.fullmatch(r"\d+\.?\d*", t):
                return float(t) if "." in t else int(t)
            # function call name
            if self.peek() == "(":
                self.next()  # consume '('
                args = []
                if self.peek() != ")":
                    args.append(self.parse_expression())
                    while self.peek() == ",":
                        self.next()
                        args.append(self.parse_expression())
                if self.peek() == ")":
                    self.next()
                return ("Call", t, args)
            if t in _ALLOWED and not isinstance(_ALLOWED[t], (int, float)):
                return ("ConstFn", t)
            raise ValueError("unknown token %r" % t)

    return _P(text)


def _eval_node(node):
    if isinstance(node, (int, float)):
        return node
    if isinstance(node, tuple):
        if node[0] in ("+", "-", "*", "/", "%"):
            _, a, b = node
            op = {"+": operator.add, "-": operator.sub, "*": operator.mul,
                  "/": operator.truediv, "%": operator.mod}[node[0]]
            return op(_eval_node(a), _eval_node(b))
        if node[0] == "Neg":
            return -_eval_node(node[1])
        if node[0] == "Pos":
            return +_eval_node(node[1])
        if node[0] == "Call":
            fn = _ALLOWED[node[1]]
            if isinstance(fn, (int, float)):  # constant called => just constant
                return fn
            return fn(*[_eval_node(a) for a in node[2]])
        if node[0] == "ConstFn":
            c = _ALLOWED[node[1]]
            return c if isinstance(c, (int, float)) else c
    raise ValueError("cannot evaluate %r" % (node,))


# --------------------------------------------------------------------------- #
# Tool: unit conversions
# --------------------------------------------------------------------------- #
_UNITS = {
    "length": {"m": 1, "meter": 1, "meters": 1, "km": 1000, "kilometer": 1000,
               "kilometers": 1000, "cm": 0.01, "centimeter": 0.01,
               "mm": 0.001, "mile": 1609.344, "miles": 1609.344,
               "foot": 0.3048, "feet": 0.3048, "ft": 0.3048,
               "inch": 0.0254, "inches": 0.0254, "yard": 0.9144, "yd": 0.9144},
    "mass": {"kg": 1, "kilogram": 1, "kilograms": 1, "g": 0.001, "gram": 0.001,
             "grams": 0.001, "mg": 1e-6, "lb": 0.45359237, "lbs": 0.45359237,
             "pound": 0.45359237, "pounds": 0.45359237, "ounce": 0.0283495,
             "ounces": 0.0283495, "oz": 0.0283495, "ton": 1000, "tonne": 1000},
    "temperature": {"c": 1, "celsius": 1, "f": 2, "fahrenheit": 2,
                    "k": 3, "kelvin": 3},
    "time": {"second": 1, "seconds": 1, "s": 1, "minute": 60, "minutes": 60,
             "min": 60, "hour": 3600, "hours": 3600, "hr": 3600,
             "day": 86400, "days": 86400, "week": 604800, "weeks": 604800},
}
_UNIT_ALIAS = {}
for _cat, _m in _UNITS.items():
    for _k in _m:
        _UNIT_ALIAS[_k] = (_cat, _k)


def try_convert(text: str):
    """e.g. 'convert 5 miles to km' or 'how many kg in 200 pounds'."""
    low = text.lower()
    # Pattern A: "<num> <from> to|in|into <to>"
    m = re.search(r"(-?\d+(?:\.\d+)?)\s*([a-zA-Z]+)\s+(?:to|in|into)\s+([a-zA-Z]+)", low)
    if m:
        num, unit_from, unit_to = float(m.group(1)), m.group(2), m.group(3)
    else:
        # Pattern B: "how many <to> (are|is) in <num> <from>"
        m = re.search(r"how many\s+([a-zA-Z]+)\s+(?:are|is)?\s*(?:there\s+)?in\s+(-?\d+(?:\.\d+)?)\s+([a-zA-Z]+)", low)
        if not m:
            return None
        unit_to, num, unit_from = m.group(1), float(m.group(2)), m.group(3)
    return run_convert(num, unit_from, unit_to)


def run_convert(num, unit_from, unit_to):
    a = _UNIT_ALIAS.get(unit_from)
    b = _UNIT_ALIAS.get(unit_to)
    if not a or not b or a[0] != b[0]:
        return None
    cat = a[0]
    if cat == "temperature":
        return convert_temperature(num, unit_from, unit_to)
    base = num * _UNITS[cat][_canon(unit_from)]
    out = base / _UNITS[cat][_canon(unit_to)]
    return f"{_fmt(num)} {unit_from} = {_fmt(out)} {unit_to}"


def _canon(u):
    return _UNIT_ALIAS[u][1]


_TEMP_SHORT = {"c": "c", "celsius": "c", "f": "f", "fahrenheit": "f",
               "k": "k", "kelvin": "k"}


def convert_temperature(num, fu, tu):
    f, t = _TEMP_SHORT[_canon(fu)], _TEMP_SHORT[_canon(tu)]
    def to_c(x, u):
        return {"c": x, "f": (x - 32) * 5 / 9, "k": x - 273.15}[u]
    def from_c(x, u):
        return {"c": x, "f": x * 9 / 5 + 32, "k": x + 273.15}[u]
    return f"{_fmt(num)} {fu} = {_fmt(from_c(to_c(num, f), t))} {tu}"


def _fmt(n):
    n = float(n)
    if abs(n - round(n)) < 1e-9:
        return str(int(round(n)))
    return ("%.4f" % n).rstrip("0").rstrip(".")


# --------------------------------------------------------------------------- #
# Tools requiring the network (all gracefully degrade if offline)
# --------------------------------------------------------------------------- #
def current_time():
    now = _dt.datetime.now()
    return now.strftime("%I:%M %p").lstrip("0") + " — " + now.strftime("%A, %B %d, %Y")


def weather(city: str):
    """Live weather via Open-Meteo (no API key)."""
    try:
        g = _http_json("https://geocoding-api.open-meteo.com/v1/search?count=1&language=en&format=json&name=" + urllib.parse.quote(city))
        if not g.get("results"):
            return f"I couldn't find a city called '{city}'. Could you check the spelling?"
        loc = g["results"][0]
        lat, lon = loc["latitude"], loc["longitude"]
        name = loc.get("name", city) + (", " + loc.get("country_code", "") if loc.get("country_code") else "")
        w = _http_json(
            f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
            "&current_weather=true&daily=temperature_2m_max,temperature_2m_min&timezone=auto")
        cur = w["current_weather"]
        temp = cur["temperature"]
        code = cur["weathercode"]
        desc = _wmo(code)
        daily = w["daily"]
        hi, lo = daily["temperature_2m_max"][0], daily["temperature_2m_min"][0]
        wind = cur.get("windspeed", 0)
        return (f"Conditions in {name} right now: {desc}, {temp:.0f}°C "
                f"(high {hi:.0f}°C / low {lo:.0f}°C). Wind speed {wind:.0f} km/h.")
    except Exception:  # noqa: BLE001
        return "I couldn't reach the weather service. Please try again in a moment."


def _wmo(code):
    table = {0: "clear sky", 1: "mainly clear", 2: "partly cloudy",
             3: "overcast", 45: "foggy", 48: "icy fog", 51: "light drizzle",
             61: "light rain", 63: "moderate rain", 65: "heavy rain",
             71: "light snow", 80: "rain showers", 95: "a thunderstorm"}
    return table.get(int(code), f"conditions code {code}")


def currency(amount, code_from, code_to):
    """FX via the European Central Bank (Frankfurter), no key needed."""
    code_from = code_from.upper()
    code_to = code_to.upper()
    try:
        r = _http_json(f"https://api.frankfurter.app/latest?amount={amount}&from={code_from}&to={code_to}")
        rate = r["rates"].get(code_to)
        if rate is None:
            return (f"I could not convert {code_from} to {code_to}. "
                    "Note: the free service only covers major currencies.")
        return f"{_fmt(amount)} {code_from} is about {_fmt(rate)} {code_to} (as of {r.get('date','today')})."
    except Exception:  # noqa: BLE001
        return "I couldn't fetch live exchange rates. Please try again shortly."


def wikipedia(topic: str, lang="en"):
    """Short intro summary from Wikipedia REST API (no key)."""
    try:
        data = _http_json(f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote(topic)}", timeout=6)
        if data.get("type") == "disambiguation":
            return f"'{topic}' has several meanings. Try being more specific."
        extract = data.get("extract") or data.get("description") or ""
        url = data.get("content_urls", {}).get("desktop", {}).get("page", "")
        out = extract.strip()
        if url:
            out += f"\n(Read more: {url})"
        return out or f"I couldn't find anything on '{topic}'."
    except Exception:  # noqa: BLE001
        return f"I couldn't fetch information on '{topic}' right now."


# --------------------------------------------------------------------------- #
# Local conversational content
# --------------------------------------------------------------------------- #
def _rand(items):
    return random.choice(items)


_JOKES = [
    "Why did the AI go to therapy? It had too many unresolved for-loops.",
    "I would tell you a UDP joke, but you might not get it.",
    "There are only 10 types of people: those who understand binary and those who don't.",
    "Why do programmers prefer dark mode? Because light attracts bugs.",
    "I would make a joke about recursion, but to understand it you'd first have to understand recursion.",
]

_HELP = (
    "Here's what I can do right now, sir:\n"
    "• General conversation — ask me anything.\n"
    "• \u201cWhat time is it?\u201d or \u201cWhat's the date?\u201d\n"
    "• Math — \u201ccalculate 15% of 2400\u201d, \u201cwhat is sqrt(2)+3?\u201d\n"
    "• Unit conversion — \u201cconvert 5 miles to km\u201d, \u201chow many kg in 200 pounds?\u201d\n"
    "• Weather — \u201cweather in London\u201d.\n"
    "• Currency — \u201cconvert 100 USD to EUR\u201d.\n"
    "• Wikipedia — \u201ctell me about the Apollo program\u201d.\n"
    "• Fun — \u201ctell me a joke\u201d.\n"
    "For deep, open-ended answers, open Settings and connect an LLM API key — "
    "then I become a full conversational AI. Try saying \u201chello\u201d or \u201cwho are you\u201d first."
)


def _greeting():
    return _rand([
        "At your service, sir. Systems online and fully operational.",
        "Good to hear from you. All systems are nominal. How may I assist?",
        "Online and ready, sir. What shall we work on today?",
    ])


def _whoami():
    return ("I am JARVIS — Just A Rather Very Intelligent System. "
            "I'm a lightweight AI assistant you can deploy anywhere. "
            "Right now I run on a local, offline-capable brain, but connect an "
            "LLM API key in Settings and I'll be far more conversational.")


def _thanks():
    return _rand(["Anytime, sir.", "Always a pleasure.", "You're most welcome."])


def _byebye():
    return _rand(["Goodbye, sir. I'll be standing by.", "Powering down gracefully. Farewell."])


def _fallback(text):
    return _rand([
        "I understand the words, sir, but I'm currently in local mode and that question "
        "needs a full language model. Connect an API key in Settings and I'll handle it.",
        "An interesting request — I'd need my neural core for that. "
        "Open Settings and add an OpenAI-compatible API key to unlock full conversation.",
        "I've noted that down, sir. For richer answers, connect an LLM API key in Settings. "
        "Meanwhile, ask me the time, the weather, a calculation, or 'help'.",
    ])


# --------------------------------------------------------------------------- #
# Main router: text -> reply
# --------------------------------------------------------------------------- #
def respond(text: str):
    """Return a string reply from the LOCAL brain (no LLM)."""
    t = re.sub(r"[!?.,]+$", "", text.strip().lower())
    if not t:
        return _fallback(text)

    if re.search(r"\b(hi|hello|hey|greetings|good (morning|afternoon|evening))\b", t):
        return _greeting()
    if re.search(r"\bwho are you|what are you|your name|introduce yourself|about you\b", t):
        return _whoami()
    if re.search(r"\bthank(s| you)\b", t):
        return _thanks()
    if re.search(r"\b(bye|goodbye|see you|good night)\b", t):
        return _byebye()
    if re.search(r"\b(help|what can you do|commands|capabilities)\b", t) and not re.search(r"help me", t):
        return _HELP
    if re.search(r"\bjoke|laugh|funny\b", t):
        return _rand(_JOKES)
    if re.search(r"\b(how are you|how do you feel|status|systems?|all good)\b", t):
        return _rand([
            "All systems running at peak efficiency, sir. Memory banks clear, "
            "processing cores at a steady hum. Ready when you are.",
            "Operating normally. Power at full capacity and diagnostics clean.",
        ])
    if re.search(r"\bwho (made|created|built) you|your creator|who built\b", t):
        return ("I was engineered as a lightweight, deployable AI assistant. "
                "You can call me your project — this codebase is designed to be "
                "run, customised and shipped anywhere you like.")
    if re.search(r"\b(time|clock|hour|o'clock)\b", t) and re.search(r"\b(what|current|now)\b", t):
        return "The time is " + _dt.datetime.now().strftime("%I:%M %p").lstrip("0") + "."
    if re.search(r"\bdate|today|what day|day is it\b", t):
        return "Today is " + _dt.datetime.now().strftime("%A, %B %d, %Y") + "."

    # weather in <city>
    m = re.search(r"\bweather|temperature|forecast\b", t)
    if m and re.search(r"\bin\s+\w+", t):
        city = re.search(r"in\s+([A-Za-z ]+?)$", t).group(1).strip()
        if city:
            return weather(city.title())

    # currency conversion 100 USD to EUR
    m = re.search(r"(?:convert\s+)?([\d.,]+)\s*([a-zA-Z]{3})\s*(?:to|in|into)\s*([a-zA-Z]{3})$", t)
    if m:
        return currency(_fmt_num(m.group(1)), m.group(2), m.group(3))

    # percentage phrase: X% of Y  (must precede unit conversion / wikipedia)
    pm = re.search(r"(\d+(?:\.\d+)?)\s*%\s*of\s*(\d+(?:\.\d+)?)", t)
    if pm:
        return f"{_fmt(float(pm.group(1)) / 100 * float(pm.group(2)))}"

    # unit conversions
    if "convert" in t or "how many" in t:
        r = try_convert(t)
        if r:
            return r

    # math: plain number expression (also catches 'what is sqrt(2)+3' etc.)
    if re.search(r"\d", t):
        cleaned = re.sub(r"[^0-9+\-*/%()., x^a-zA-Z ]", "", t)
        cleaned = (cleaned.replace("x", "*").replace("^", "**")
                   .replace("what is ", "").replace("calculate ", "")
                   .replace("compute ", "").replace("=", "")
                   .replace(",", ""))
        if re.search(r"[\d].*[\d+*/%().\-.]", cleaned):
            ok, res = evaluate_math(cleaned)
            if ok and res is not None:
                return f"Computing... {res}."

    # wikipedia / tell me about  (only word-based topics, not arithmetic)
    m = re.search(r"\b(tell me about|what is|who is|define|about|explain|wikipedia|info on)\s+(.+)", t)
    if m and not re.search(r"\b(you|me|help)\b", m.group(2)):
        if not re.search(r"[\d%+\-*/()]|sqrt|sin|cos|log|\bof\s+\d", m.group(2)):
            return wikipedia(m.group(2).strip().title())

    return _fallback(text)


def _fmt_num(s):
    try:
        return float(s.replace(",", ""))
    except ValueError:
        return 0.0
