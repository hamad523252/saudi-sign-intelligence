"""Tiny server: serves web/, composes sentences with an LLM served on AMD GPUs (AMD Token Factory),
and proxies Arabic text-to-speech to ElevenLabs. All keys stay server-side.

    AMD_API_KEY=rc-... ELEVENLABS_API_KEY=... python3 server/server.py

    ELEVENLABS_API_KEY=... ELEVEN_VOICE_ID=... python3 server/server.py   ->  http://localhost:8000
Without a key, /api/tts returns 503 and the page falls back to the browser's Arabic voice.
Only short phrases from the fixed vocabulary are spoken; no video or landmarks leave the browser.
"""
import json, os, urllib.request
from functools import lru_cache
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

WEB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "web")
KEY = os.environ.get("ELEVENLABS_API_KEY", "")
VOICE = os.environ.get("ELEVEN_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")
with open(os.path.join(WEB, "vocab.json"), encoding="utf-8") as fh:
    ALLOWED = {v["ar"] for v in json.load(fh)} | {"يُرجى استدعاء مترجم لغة إشارة"}


AMD_KEY = os.environ.get("AMD_API_KEY", "")
AMD_BASE = os.environ.get("AMD_BASE_URL", "https://developer.amd.com.cn/radeon/api/v1")
AMD_MODEL = os.environ.get("AMD_MODEL", "Qwen3.8-27B")
VOCAB_IDS = {}
with open(os.path.join(WEB, "vocab.json"), encoding="utf-8") as fh:
    for v in json.load(fh):
        VOCAB_IDS[v["id"]] = v

COMPOSE_PROMPT = (
    "You help a Deaf patient talk to hospital staff. You receive a short list of Saudi Sign Language "
    "signs the patient made, in order. Rewrite them as ONE short, natural first-person sentence. "
    "Do not add symptoms, diagnoses, advice or any fact that is not in the signs. "
    'Reply only with JSON: {"ar": "<Arabic sentence>", "en": "<English sentence>"}')
SIMPLIFY_PROMPT = (
    "You help hospital staff talk to a Deaf patient. Rewrite the staff message as very short, plain Arabic "
    "(max 12 words) and pick the matching sign ids from this list, in order, only if they clearly apply: {ids}. "
    "Never change the medical meaning. "
    'Reply only with JSON: {{"ar": "<simple Arabic>", "signs": ["ID", ...]}}')


def amd_chat(system, user):
    """OpenAI-compatible chat completion on AMD Token Factory (models served on AMD GPUs)."""
    req = urllib.request.Request(
        AMD_BASE.rstrip("/") + "/chat/completions",
        data=json.dumps({"model": AMD_MODEL, "temperature": 0.1, "max_tokens": 200,
                         "messages": [{"role": "system", "content": system},
                                      {"role": "user", "content": user}]}).encode(),
        headers={"Authorization": f"Bearer {AMD_KEY}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        txt = json.loads(r.read())["choices"][0]["message"]["content"]
    s, e = txt.find("{"), txt.rfind("}")
    return json.loads(txt[s:e + 1])


@lru_cache(maxsize=256)
def tts(text):
    req = urllib.request.Request(
        f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE}",
        data=json.dumps({"text": text, "model_id": "eleven_multilingual_v2"}).encode(),
        headers={"xi-api-key": KEY, "Content-Type": "application/json", "Accept": "audio/mpeg"})
    with urllib.request.urlopen(req, timeout=20) as r:
        return r.read()


class H(SimpleHTTPRequestHandler):
    def __init__(self, *a, **k):
        super().__init__(*a, directory=WEB, **k)

    def _json(self, obj, code=200):
        body = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0) or 0)
        payload = json.loads(self.rfile.read(min(n, 4000)) or b"{}")
        if self.path == "/api/compose":           # signs -> sentence (AMD-hosted LLM)
            ids = [i for i in payload.get("signs", []) if i in VOCAB_IDS][:8]
            if not ids:
                return self._json({"error": "no signs"}, 400)
            fallback = {"ar": " ، ".join(VOCAB_IDS[i]["ar"] for i in ids),
                        "en": ", ".join(VOCAB_IDS[i]["en"] for i in ids), "source": "rule"}
            if not AMD_KEY:
                return self._json(fallback)
            try:
                out = amd_chat(COMPOSE_PROMPT, " | ".join(f'{i} ({VOCAB_IDS[i]["ar"]})' for i in ids))
                ALLOWED.add(out["ar"])
                return self._json({"ar": out["ar"], "en": out["en"], "source": "amd:" + AMD_MODEL})
            except Exception:
                return self._json(fallback)
        if self.path == "/api/simplify":          # staff text -> plain Arabic + signs to show
            text = str(payload.get("text", ""))[:500]
            if not text or not AMD_KEY:
                return self._json({"ar": text, "signs": [], "source": "none"})
            try:
                out = amd_chat(SIMPLIFY_PROMPT.format(ids=", ".join(VOCAB_IDS)), text)
                signs = [s for s in out.get("signs", []) if s in VOCAB_IDS]
                ALLOWED.add(out.get("ar", text))
                return self._json({"ar": out.get("ar", text), "signs": signs, "source": "amd:" + AMD_MODEL})
            except Exception:
                return self._json({"ar": text, "signs": [], "source": "error"})
        if self.path != "/api/tts":
            return self.send_error(404)
        text = payload.get("text", "")
        if not KEY or text not in ALLOWED:
            return self.send_error(503 if not KEY else 400)
        try:
            audio = tts(text)
        except Exception:
            return self.send_error(502)
        self.send_response(200)
        self.send_header("Content-Type", "audio/mpeg")
        self.send_header("Content-Length", str(len(audio)))
        self.end_headers()
        self.wfile.write(audio)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"Saudi Sign Intelligence on http://localhost:{port}  (ElevenLabs TTS: {'on' if KEY else 'off'}, AMD LLM: {AMD_MODEL if AMD_KEY else 'off'})")
    ThreadingHTTPServer(("0.0.0.0", port), H).serve_forever()
