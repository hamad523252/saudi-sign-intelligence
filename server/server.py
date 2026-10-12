"""Tiny server: serves web/ and proxies Arabic text-to-speech to ElevenLabs (key stays server-side).

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

    def do_POST(self):
        if self.path != "/api/tts":
            return self.send_error(404)
        n = int(self.headers.get("Content-Length", 0) or 0)
        text = json.loads(self.rfile.read(min(n, 2000)) or b"{}").get("text", "")
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
    print(f"Saudi Sign Intelligence on http://localhost:{port}  (ElevenLabs TTS: {'on' if KEY else 'off'})")
    ThreadingHTTPServer(("0.0.0.0", port), H).serve_forever()
