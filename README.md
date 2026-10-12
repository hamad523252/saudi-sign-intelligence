# 🤟 Saudi Sign Intelligence

**Real-time Saudi Sign Language → Arabic text & speech, so Deaf patients can be understood in hospitals.**
AMD Developer Hackathon ACT III · Health & Wellbeing track · Team *Saudi Sign Intelligence* (Riyadh)

## The problem
Deaf patients in Saudi hospitals often wait for a human sign-language interpreter to say even
"I am in pain" or "I have an allergy". There is no AI system for **Saudi** Sign Language.

## What it does
1. The patient signs in front of any laptop/tablet camera.
2. **MediaPipe** extracts hand + upper-body landmarks **inside the browser** (no video leaves the device).
3. A small **Transformer** trained on **AMD Instinct GPUs (ROCm)** recognises the sign.
4. The phrase appears in Arabic + English and is spoken in natural Arabic (**ElevenLabs**; browser voice fallback).
5. Staff confirm ✔ or press ✖ to request a human interpreter.

V1 vocabulary: 16 hospital phrases (pain, doctor, nurse, medicine, water, help, toilet, emergency,
head, stomach, dizzy, allergy, family, yes, no, thank you) — `web/vocab.json`.
The dataset is recorded by a certified Saudi Sign Language interpreter (signer `H001`) with the built-in recorder.

## Safety (health track)
- Communication aid, **not** a medical device and **no diagnosis**.
- Confidence is always shown; below 75 % nothing is spoken and the UI asks to repeat or call an interpreter.
- Human-in-the-loop confirm/reject; fixed phrase list (no free-text generation, no hallucinated medical content).
- Privacy: only anonymous landmark numbers are processed; raw video never leaves the browser.
  The TTS server only voices phrases from the fixed vocabulary; the API key stays server-side.
- Known limits: isolated signs only, small vocabulary, one signer in V1 → signer-independent evaluation is the next step.

## Where AMD is used
`train/train.py` trains the recogniser on **AMD Instinct MI300X with ROCm** (ROCm exposes AMD GPUs through
`torch.cuda`); the script logs the GPU name and ROCm version into `train/report.json` and exports `web/model.onnx`.

## Run
```bash
# 1) record: open the app, tab "تسجيل البيانات", 10+ samples per sign, export JSON into data/
python3 server/server.py                      # http://localhost:8000
# 2) train on AMD Developer Cloud (rocm/pytorch image)
pip install onnx && python3 train/train.py --epochs 120
# 3) serve with Arabic voice
ELEVENLABS_API_KEY=... python3 server/server.py
```
Pipeline self-test without a camera: `python3 tools/make_synthetic.py && python3 train/train.py --epochs 30`.

## Structure
`web/` browser app (MediaPipe + onnxruntime-web) · `train/` features + Transformer + ONNX export ·
`server/` static server + ElevenLabs proxy · `data/` recorded JSON samples.

MIT License.
