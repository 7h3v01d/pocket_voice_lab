# Pocket Voice Lab v1.1.0

Local voice capture, conditioning, testing, and voice-library desktop app built on
[Kyutai Pocket TTS](https://github.com/kyutai-labs/pocket-tts). Single-file PyQt6,
CPU-only, everything stays on your machine.

Copyright 2026 Leon Priest (7h3v01d) — Apache License 2.0.

## Pipeline

| Tab | Function |
|---|---|
| **CAPTURE** | Record 15–30 s of reference speech from a local mic (device select, RMS meter, peak/clip warnings), save WAV, one-click hand-off to conditioning |
| **CONDITION** | Run reference audio through `get_state_for_audio_prompt()` and export the voice state via `export_model_state()` to `.safetensors` + JSON sidecar (name, source, SHA-256, timing) |
| **TEST** | Synthesize arbitrary text with any library voice, realtime-factor readout, auto-play / auto-save renders, `frames_after_eos` control |
| **LIBRARY** | Table of all conditioned voices — rename, delete, open folder, jump to test |

Slow work (audio → voice state) happens once; every later generation loads the
`.safetensors` near-instantly, with an in-worker state cache keyed on mtime.

## Install

```bat
pip install pocket-tts PyQt6 sounddevice numpy
```

Requires Python 3.10–3.14 and PyTorch 2.5+ (pulled in by pocket-tts; CPU build is fine —
GPU gives no speedup). First model load needs Hugging Face access:
accept the terms at `huggingface.co/kyutai/pocket-tts`, then `hf auth login`
with a Read token. The app surfaces a hint in the status bar if auth fails.

## Run

```bat
python pocket_voice_lab.py
```

1. Top strip: pick language / temp / steps / eos, press **LOAD MODEL** (first run downloads weights).
2. CAPTURE a take → **SAVE + SEND TO CONDITION**.
3. Tick the consent box, **CONDITION VOICE**.
4. TEST tab → pick voice, type text, **GENERATE**.

Library lives at `%LOCALAPPDATA%\PocketVoiceLab\` (`captures/`, `voices/`, `renders/`).

## Logging

Rotating log at `%LOCALAPPDATA%\PocketVoiceLab\logs\pvl.log` (5 x 1 MB). Every worker
job (load / condition / generate) logs start, timing, and full tracebacks on failure;
Qt warnings and uncaught exceptions are captured too. Startup records python/OS
version and whether HF_TOKEN is present (never its value). LOG button in the header
opens the file; `--verbose` echoes DEBUG lines to the console.

## Smoke test

```bat
python pocket_voice_lab.py --smoke
```

Constructs the full UI offscreen (QT_QPA_PLATFORM=offscreen), verifies tabs, worker
thread, directories, and chip state, then exits 0/1.

## Architecture notes

- One persistent `ModelWorker(QThread)` owns the `TTSModel` and processes a job queue
  (load / condition / generate). The model is never touched from the GUI thread.
- `QThread` subclassing throughout (no `moveToThread`); workers pinned in
  `MainWindow._worker_refs` for Windows GC-safety; `@pyqtSlot` on cross-thread slots.
- `pocket_tts` / `torch` / `sounddevice` import lazily — the window opens instantly and
  degrades gracefully if a dependency is missing.
- Playback via PortAudio's own callback thread (`sounddevice.play`), no Qt multimedia.
- Model generation parameters (`temp`, `lsd_decode_steps`, `eos_threshold`, `quantize`)
  are load-time in Pocket TTS, hence they live on the model strip with RELOAD.

## Consent

The Kyutai model terms prohibit voice cloning without explicit, lawful consent.
The conditioning tab gates on an explicit consent checkbox by design — leave it in.
