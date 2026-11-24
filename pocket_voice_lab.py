#!/usr/bin/env python3
# =============================================================================
#  Pocket Voice Lab v1.1.0
#  Local voice capture / conditioning / testing / library — built on Pocket TTS
#
#  v1.1.0 — rotating file log (logs/pvl.log), full tracebacks for every worker
#           failure, Qt message capture, global excepthook, LOG button in the
#           header, HF-auth hint on conditioning failures.
#
#  Copyright 2026 Leon Priest (7h3v01d)
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# =============================================================================
#
#  Pipeline:
#    CAPTURE   — record reference audio from a local microphone (WAV, mono)
#    CONDITION — extract a Pocket TTS voice state and export .safetensors
#    TEST      — synthesize arbitrary text with any library voice, audition it
#    LIBRARY   — manage the voice collection (rename / delete / inspect)
#
#  Engineering notes:
#    * The TTS model lives on ONE persistent QThread subclass (ModelWorker)
#      with a job queue. Model load, conditioning and generation all run
#      there — the model is never touched from the GUI thread.
#    * QThread subclassing (not moveToThread) for Windows GC-safety.
#      All workers are pinned in MainWindow._worker_refs.
#    * pocket_tts / torch / sounddevice are imported lazily so the window
#      appears instantly and the app degrades gracefully if a dependency
#      is missing.
#    * Voice library layout (under %LOCALAPPDATA%\PocketVoiceLab):
#        captures/  raw reference WAVs
#        voices/    <slug>.safetensors + <slug>.json sidecar metadata
#        renders/   synthesized test output WAVs
#
#  Offscreen smoke test:  python pocket_voice_lab.py --smoke
# =============================================================================

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import logging.handlers
import os
import platform
import queue
import re
import sys
import time
import traceback
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from PyQt6.QtCore import Qt, QThread, QTimer, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFrame,
    QGridLayout, QGroupBox, QHBoxLayout, QHeaderView, QInputDialog, QLabel,
    QLineEdit, QMainWindow, QMessageBox, QPlainTextEdit, QProgressBar,
    QPushButton, QSpinBox, QStatusBar, QTableWidget, QTableWidgetItem,
    QTabWidget, QVBoxLayout, QWidget,
)

APP_NAME = "Pocket Voice Lab"
APP_VERSION = "1.1.0"
ORG_TAG = "7h3v01d"

# ---------------------------------------------------------------------------
# Palette — dark-industrial
# ---------------------------------------------------------------------------
C_BG      = "#0b0f14"   # obsidian
C_PANEL   = "#10161d"
C_PANEL2  = "#141c25"
C_EDGE    = "#1e2933"
C_TEXT    = "#c9d4dc"
C_DIM     = "#5c6b78"
C_TEAL    = "#2fd6c3"
C_GREEN   = "#4be08a"
C_AMBER   = "#ffb454"
C_RED     = "#ff5c66"

MONO = "JetBrains Mono, Consolas, monospace"

QSS = f"""
* {{ font-family: {MONO}; font-size: 12px; }}
QMainWindow, QWidget {{ background: {C_BG}; color: {C_TEXT}; }}
QTabWidget::pane {{ border: 1px solid {C_EDGE}; top: -1px; }}
QTabBar::tab {{
    background: {C_PANEL}; color: {C_DIM}; padding: 7px 22px;
    border: 1px solid {C_EDGE}; border-bottom: none; margin-right: 2px;
}}
QTabBar::tab:selected {{ background: {C_PANEL2}; color: {C_TEAL};
    border-top: 2px solid {C_TEAL}; }}
QGroupBox {{
    border: 1px solid {C_EDGE}; margin-top: 14px; padding-top: 10px;
    background: {C_PANEL}; font-weight: bold; color: {C_TEAL};
}}
QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 4px; }}
QLabel {{ background: transparent; }}
QLabel#chip_ok    {{ color: {C_GREEN};  border: 1px solid {C_GREEN};  padding: 2px 10px; }}
QLabel#chip_warn  {{ color: {C_AMBER};  border: 1px solid {C_AMBER};  padding: 2px 10px; }}
QLabel#chip_err   {{ color: {C_RED};    border: 1px solid {C_RED};    padding: 2px 10px; }}
QLabel#chip_idle  {{ color: {C_DIM};    border: 1px solid {C_EDGE};   padding: 2px 10px; }}
QLabel#hdr_title  {{ color: {C_TEAL}; font-size: 15px; font-weight: bold; }}
QLabel#hdr_sub    {{ color: {C_DIM}; }}
QLineEdit, QPlainTextEdit, QComboBox, QSpinBox, QDoubleSpinBox {{
    background: {C_PANEL2}; border: 1px solid {C_EDGE}; padding: 4px 6px;
    color: {C_TEXT}; border-radius: 0;
    selection-background-color: {C_TEAL}; selection-color: {C_BG};
}}
QLineEdit:focus, QPlainTextEdit:focus, QComboBox:focus {{ border: 1px solid {C_TEAL}; }}
QComboBox QAbstractItemView {{
    background: {C_PANEL2}; border: 1px solid {C_EDGE};
    selection-background-color: {C_TEAL}; selection-color: {C_BG};
}}
QPushButton {{
    background: {C_PANEL2}; border: 1px solid {C_EDGE}; padding: 6px 16px;
    color: {C_TEXT}; border-radius: 0;
}}
QPushButton:hover {{ border: 1px solid {C_TEAL}; color: {C_TEAL}; }}
QPushButton:pressed {{ background: {C_EDGE}; }}
QPushButton:disabled {{ color: {C_DIM}; border-color: {C_EDGE}; }}
QPushButton#btn_go     {{ border: 1px solid {C_TEAL};  color: {C_TEAL}; font-weight: bold; }}
QPushButton#btn_rec    {{ border: 1px solid {C_RED};   color: {C_RED};  font-weight: bold; }}
QPushButton#btn_danger {{ border: 1px solid {C_RED};   color: {C_RED}; }}
QProgressBar {{
    background: {C_PANEL2}; border: 1px solid {C_EDGE}; height: 14px;
    text-align: center; color: {C_BG}; border-radius: 0;
}}
QProgressBar::chunk {{ background: {C_GREEN}; border-radius: 0; }}
QTableWidget {{
    background: {C_PANEL}; border: 1px solid {C_EDGE}; gridline-color: {C_EDGE};
    alternate-background-color: {C_PANEL2};
}}
QTableWidget::item:selected {{ background: {C_TEAL}; color: {C_BG}; }}
QHeaderView::section {{
    background: {C_PANEL2}; color: {C_TEAL}; border: 1px solid {C_EDGE};
    padding: 4px 8px; font-weight: bold;
}}
QStatusBar {{ background: {C_PANEL}; color: {C_DIM}; border-top: 1px solid {C_EDGE}; }}
QCheckBox::indicator {{
    width: 13px; height: 13px; border: 1px solid {C_EDGE}; background: {C_PANEL2};
}}
QCheckBox::indicator:checked {{ background: {C_TEAL}; border: 1px solid {C_TEAL}; }}
QFrame#hdr {{ background: {C_PANEL}; border-bottom: 1px solid {C_EDGE}; }}
QScrollBar:vertical {{ background: {C_PANEL}; width: 10px; }}
QScrollBar::handle:vertical {{ background: {C_EDGE}; min-height: 24px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; }}
"""

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


def app_root() -> Path:
    base = os.environ.get("LOCALAPPDATA") or str(Path.home())
    return Path(base) / "PocketVoiceLab"


DIR_ROOT = app_root()
DIR_CAPTURES = DIR_ROOT / "captures"
DIR_VOICES = DIR_ROOT / "voices"
DIR_RENDERS = DIR_ROOT / "renders"

for _d in (DIR_CAPTURES, DIR_VOICES, DIR_RENDERS):
    _d.mkdir(parents=True, exist_ok=True)

LANGUAGES = [
    "english", "english_2026-04", "english_2026-01",
    "french_24l", "german_24l", "portuguese_24l", "italian_24l", "spanish_24l",
]

# ---------------------------------------------------------------------------
# Logging — rotating file log so failures are copy-pasteable
# ---------------------------------------------------------------------------

DIR_LOGS = DIR_ROOT / "logs"
LOG_PATH = DIR_LOGS / "pvl.log"
log = logging.getLogger("pvl")


def setup_logging(verbose: bool = False) -> Path:
    DIR_LOGS.mkdir(parents=True, exist_ok=True)
    log.setLevel(logging.DEBUG)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(threadName)-12s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.handlers.RotatingFileHandler(
        LOG_PATH, maxBytes=1_000_000, backupCount=5, encoding="utf-8")
    fh.setFormatter(fmt)
    fh.setLevel(logging.DEBUG)
    log.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    ch.setLevel(logging.DEBUG if verbose else logging.INFO)
    log.addHandler(ch)

    # capture anything the rest of the process throws
    def _excepthook(exc_type, exc, tb) -> None:
        log.critical("UNCAUGHT: %s",
                     "".join(traceback.format_exception(exc_type, exc, tb)))
        sys.__excepthook__(exc_type, exc, tb)

    sys.excepthook = _excepthook

    # route Qt's own warnings (bad connects, style issues) into the log
    try:
        from PyQt6.QtCore import qInstallMessageHandler

        def _qt_handler(_mode, ctx, message) -> None:
            log.debug("QT: %s (%s:%s)", message, ctx.file, ctx.line)

        qInstallMessageHandler(_qt_handler)
    except Exception:  # noqa: BLE001 — logging must never take the app down
        pass

    log.info("=" * 72)
    log.info("%s v%s starting | python %s | %s | pid %d",
             APP_NAME, APP_VERSION, platform.python_version(),
             platform.platform(), os.getpid())
    log.info("library root: %s", DIR_ROOT)
    log.info("HF_TOKEN present in environment: %s",
             "yes" if os.environ.get("HF_TOKEN") else "no")
    return LOG_PATH

# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def slugify(name: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9_-]+", "_", name.strip()).strip("_").lower()
    return s or f"voice_{int(time.time())}"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def write_wav_int16(path: Path, audio_f32: np.ndarray, samplerate: int) -> None:
    a = np.clip(audio_f32.astype(np.float32), -1.0, 1.0)
    i16 = (a * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(samplerate)
        w.writeframes(i16.tobytes())


def read_wav_f32(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as w:
        sr = w.getframerate()
        n = w.getnframes()
        ch = w.getnchannels()
        raw = w.readframes(n)
    data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if ch > 1:
        data = data.reshape(-1, ch).mean(axis=1)
    return data, sr


def fmt_dur(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


def voice_meta_path(safetensors: Path) -> Path:
    return safetensors.with_suffix(".json")


def load_voice_entries() -> list[dict]:
    entries = []
    for st in sorted(DIR_VOICES.glob("*.safetensors")):
        meta = {}
        mp = voice_meta_path(st)
        if mp.exists():
            try:
                meta = json.loads(mp.read_text(encoding="utf-8"))
            except Exception:
                meta = {}
        entries.append({
            "path": st,
            "name": meta.get("name", st.stem),
            "source": meta.get("source", "?"),
            "created": meta.get("created", "?"),
            "sha256": meta.get("sha256", ""),
            "source_seconds": meta.get("source_seconds", 0.0),
            "size": st.stat().st_size,
        })
    return entries


# ---------------------------------------------------------------------------
# Model worker — owns the Pocket TTS model, processes a job queue
# ---------------------------------------------------------------------------


@dataclass
class ModelParams:
    language: str = "english"
    temp: float = 0.7
    lsd_decode_steps: int = 1
    eos_threshold: float = -4.0
    quantize: bool = False


class ModelWorker(QThread):
    sig_status = pyqtSignal(str)                 # transient status text
    sig_model_ready = pyqtSignal(int, str)       # sample_rate, summary
    sig_model_error = pyqtSignal(str)
    sig_conditioned = pyqtSignal(str)            # safetensors path
    sig_condition_error = pyqtSignal(str)
    sig_generated = pyqtSignal(object, int, float, str)  # audio f32, sr, secs, voice name
    sig_generate_error = pyqtSignal(str)
    sig_busy = pyqtSignal(bool)

    def __init__(self) -> None:
        super().__init__()
        self._q: "queue.Queue[Optional[dict]]" = queue.Queue()
        self._model = None
        self._params = ModelParams()
        self._state_cache: dict[str, tuple[float, object]] = {}

    # -- public API (GUI thread) ------------------------------------------
    def submit_load(self, params: ModelParams) -> None:
        self._q.put({"kind": "load", "params": params})

    def submit_condition(self, source_wav: str, dest_st: str, meta: dict) -> None:
        self._q.put({"kind": "condition", "src": source_wav,
                     "dest": dest_st, "meta": meta})

    def submit_generate(self, voice_st: str, voice_name: str, text: str,
                        frames_after_eos: Optional[int]) -> None:
        self._q.put({"kind": "generate", "voice": voice_st,
                     "voice_name": voice_name, "text": text,
                     "frames_after_eos": frames_after_eos})

    def shutdown(self) -> None:
        self._q.put(None)

    # -- thread body -------------------------------------------------------
    def run(self) -> None:
        while True:
            job = self._q.get()
            if job is None:
                break
            self.sig_busy.emit(True)
            kind = job.get("kind", "?")
            log.info("job start: %s %s", kind,
                     {k: v for k, v in job.items()
                      if k not in ("kind", "text")} or "")
            if kind == "generate":
                log.debug("generate text (%d chars): %r",
                          len(job.get("text", "")), job.get("text", "")[:500])
            t0 = time.monotonic()
            try:
                if kind == "load":
                    self._do_load(job["params"])
                elif kind == "condition":
                    self._do_condition(job)
                elif kind == "generate":
                    self._do_generate(job)
                log.info("job done: %s (%.2fs)", kind, time.monotonic() - t0)
            except Exception as exc:  # noqa: BLE001 — surfaced to GUI
                msg = f"{type(exc).__name__}: {exc}"
                log.error("job FAILED: %s (%.2fs) — %s\n%s",
                          kind, time.monotonic() - t0, msg,
                          traceback.format_exc())
                if kind == "load":
                    self.sig_model_error.emit(msg)
                elif kind == "condition":
                    self.sig_condition_error.emit(msg)
                else:
                    self.sig_generate_error.emit(msg)
            finally:
                self.sig_busy.emit(False)
        log.info("model worker shut down")

    def _do_load(self, params: ModelParams) -> None:
        self.sig_status.emit(f"loading model [{params.language}] — first run downloads weights…")
        t0 = time.monotonic()
        from pocket_tts import TTSModel  # lazy: heavy import
        self._model = TTSModel.load_model(
            language=params.language,
            temp=params.temp,
            lsd_decode_steps=params.lsd_decode_steps,
            eos_threshold=params.eos_threshold,
            quantize=params.quantize,
        )
        self._params = params
        self._state_cache.clear()
        dt = time.monotonic() - t0
        summary = (f"{params.language} | temp={params.temp} "
                   f"steps={params.lsd_decode_steps} eos={params.eos_threshold}"
                   f"{' int8' if params.quantize else ''} | {dt:.1f}s")
        self.sig_model_ready.emit(int(self._model.sample_rate), summary)

    def _require_model(self):
        if self._model is None:
            raise RuntimeError("model not loaded — press LOAD MODEL first")
        return self._model

    def _do_condition(self, job: dict) -> None:
        model = self._require_model()
        src, dest = Path(job["src"]), Path(job["dest"])
        self.sig_status.emit(f"conditioning voice from {src.name} — this is the slow step…")
        t0 = time.monotonic()
        from pocket_tts import export_model_state  # lazy
        state = model.get_state_for_audio_prompt(str(src))
        export_model_state(state, str(dest))
        meta = dict(job["meta"])
        meta.update({
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "sha256": sha256_file(dest),
            "conditioned_seconds": round(time.monotonic() - t0, 2),
            "model_language": self._params.language,
            "app": f"{APP_NAME} v{APP_VERSION}",
        })
        voice_meta_path(dest).write_text(
            json.dumps(meta, indent=2), encoding="utf-8")
        self._state_cache[str(dest)] = (dest.stat().st_mtime, state)
        self.sig_conditioned.emit(str(dest))

    def _get_voice_state(self, voice_st: str):
        model = self._require_model()
        mtime = Path(voice_st).stat().st_mtime
        cached = self._state_cache.get(voice_st)
        if cached and cached[0] == mtime:
            return cached[1]
        state = model.get_state_for_audio_prompt(voice_st)
        self._state_cache[voice_st] = (mtime, state)
        return state

    def _do_generate(self, job: dict) -> None:
        model = self._require_model()
        self.sig_status.emit(f"synthesizing with [{job['voice_name']}]…")
        t0 = time.monotonic()
        state = self._get_voice_state(job["voice"])
        audio_t = model.generate_audio(
            state, job["text"],
            frames_after_eos=job["frames_after_eos"],
            copy_state=True,
        )
        audio = audio_t.detach().cpu().numpy().astype(np.float32).reshape(-1)
        self.sig_generated.emit(audio, int(model.sample_rate),
                                time.monotonic() - t0, job["voice_name"])


# ---------------------------------------------------------------------------
# Recorder worker — microphone capture via sounddevice
# ---------------------------------------------------------------------------


class RecorderWorker(QThread):
    sig_level = pyqtSignal(float)          # 0..1 RMS
    sig_elapsed = pyqtSignal(float)
    sig_finished = pyqtSignal(object, int)  # float32 mono, samplerate
    sig_error = pyqtSignal(str)

    def __init__(self, device_index: Optional[int], samplerate: int) -> None:
        super().__init__()
        self._device = device_index
        self._sr = samplerate
        self._stop = False

    @pyqtSlot()
    def stop(self) -> None:
        self._stop = True

    def run(self) -> None:
        try:
            import sounddevice as sd  # lazy
        except Exception as exc:  # noqa: BLE001
            log.error("recorder: sounddevice import failed\n%s",
                      traceback.format_exc())
            self.sig_error.emit(f"sounddevice unavailable: {exc}")
            return
        log.info("recorder start: device=%s rate=%d", self._device, self._sr)
        frames: list[np.ndarray] = []

        def cb(indata, _frames, _time, status) -> None:
            if status:
                pass  # over/underruns are non-fatal for reference capture
            mono = indata[:, 0].copy()
            frames.append(mono)
            rms = float(np.sqrt(np.mean(np.square(mono), dtype=np.float64)))
            self.sig_level.emit(min(1.0, rms * 4.0))

        try:
            t0 = time.monotonic()
            with sd.InputStream(device=self._device, channels=1,
                                samplerate=self._sr, dtype="float32",
                                callback=cb):
                while not self._stop:
                    sd.sleep(50)
                    self.sig_elapsed.emit(time.monotonic() - t0)
        except Exception as exc:  # noqa: BLE001
            log.error("recorder stream FAILED\n%s", traceback.format_exc())
            self.sig_error.emit(f"{type(exc).__name__}: {exc}")
            return
        if not frames:
            log.warning("recorder: no audio captured")
            self.sig_error.emit("no audio captured")
            return
        audio = np.concatenate(frames)
        log.info("recorder done: %.2fs @ %d Hz, peak %.3f",
                 audio.size / self._sr, self._sr,
                 float(np.max(np.abs(audio))) if audio.size else 0.0)
        self.sig_finished.emit(audio, self._sr)


# ---------------------------------------------------------------------------
# Playback (PortAudio's own thread; safe to drive from GUI)
# ---------------------------------------------------------------------------


def play_audio(audio_f32: np.ndarray, samplerate: int) -> bool:
    try:
        import sounddevice as sd
        sd.stop()
        sd.play(audio_f32, samplerate)
        return True
    except Exception:
        return False


def stop_audio() -> None:
    try:
        import sounddevice as sd
        sd.stop()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Chip label
# ---------------------------------------------------------------------------


class Chip(QLabel):
    def set_state(self, text: str, state: str) -> None:
        self.setText(f" {text} ")
        self.setObjectName({
            "ok": "chip_ok", "warn": "chip_warn",
            "err": "chip_err",
        }.get(state, "chip_idle"))
        self.style().unpolish(self)
        self.style().polish(self)


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------


class CaptureTab(QWidget):
    sig_capture_saved = pyqtSignal(str)   # wav path — feeds Condition tab

    def __init__(self, main: "MainWindow") -> None:
        super().__init__()
        self.main = main
        self._recorder: Optional[RecorderWorker] = None
        self._last_audio: Optional[np.ndarray] = None
        self._last_sr = 48000

        lay = QVBoxLayout(self)

        gb_dev = QGroupBox("INPUT DEVICE")
        g = QGridLayout(gb_dev)
        self.cmb_device = QComboBox()
        self.cmb_rate = QComboBox()
        self.cmb_rate.addItems(["48000", "44100", "24000", "16000"])
        btn_refresh = QPushButton("RESCAN")
        btn_refresh.clicked.connect(self.refresh_devices)
        g.addWidget(QLabel("device"), 0, 0)
        g.addWidget(self.cmb_device, 0, 1)
        g.addWidget(btn_refresh, 0, 2)
        g.addWidget(QLabel("rate"), 1, 0)
        g.addWidget(self.cmb_rate, 1, 1)
        g.setColumnStretch(1, 1)
        lay.addWidget(gb_dev)

        gb_rec = QGroupBox("CAPTURE")
        g2 = QGridLayout(gb_rec)
        self.btn_record = QPushButton("● RECORD")
        self.btn_record.setObjectName("btn_rec")
        self.btn_record.clicked.connect(self._toggle_record)
        self.lbl_elapsed = QLabel("00:00")
        self.lbl_elapsed.setStyleSheet(f"color: {C_AMBER}; font-size: 22px;")
        self.meter = QProgressBar()
        self.meter.setRange(0, 100)
        self.meter.setTextVisible(False)
        self.lbl_hint = QLabel("aim for 15–30 s of clean, natural speech — "
                               "one speaker, minimal room noise")
        self.lbl_hint.setStyleSheet(f"color: {C_DIM};")
        g2.addWidget(self.btn_record, 0, 0)
        g2.addWidget(self.lbl_elapsed, 0, 1)
        g2.addWidget(self.meter, 0, 2)
        g2.addWidget(self.lbl_hint, 1, 0, 1, 3)
        g2.setColumnStretch(2, 1)
        lay.addWidget(gb_rec)

        gb_save = QGroupBox("LAST TAKE")
        g3 = QGridLayout(gb_save)
        self.ed_name = QLineEdit()
        self.ed_name.setPlaceholderText("take name  e.g. leon_ref_01")
        self.btn_play = QPushButton("PLAY")
        self.btn_play.clicked.connect(self._play_last)
        self.btn_save = QPushButton("SAVE TO CAPTURES")
        self.btn_save.setObjectName("btn_go")
        self.btn_save.clicked.connect(self._save_last)
        self.btn_send = QPushButton("SAVE + SEND TO CONDITION →")
        self.btn_send.setObjectName("btn_go")
        self.btn_send.clicked.connect(lambda: self._save_last(send=True))
        self.lbl_take = QLabel("no take yet")
        self.lbl_take.setStyleSheet(f"color: {C_DIM};")
        for b in (self.btn_play, self.btn_save, self.btn_send):
            b.setEnabled(False)
        g3.addWidget(QLabel("name"), 0, 0)
        g3.addWidget(self.ed_name, 0, 1, 1, 3)
        g3.addWidget(self.btn_play, 1, 0)
        g3.addWidget(self.btn_save, 1, 1)
        g3.addWidget(self.btn_send, 1, 2)
        g3.addWidget(self.lbl_take, 2, 0, 1, 4)
        g3.setColumnStretch(3, 1)
        lay.addWidget(gb_save)
        lay.addStretch(1)

        self.refresh_devices()

    # ------------------------------------------------------------------
    def refresh_devices(self) -> None:
        self.cmb_device.clear()
        try:
            import sounddevice as sd
            default_in = sd.default.device[0]
            for idx, dev in enumerate(sd.query_devices()):
                if dev.get("max_input_channels", 0) > 0:
                    tag = "  [default]" if idx == default_in else ""
                    self.cmb_device.addItem(f"{idx}: {dev['name']}{tag}", idx)
        except Exception as exc:  # noqa: BLE001
            self.cmb_device.addItem(f"sounddevice unavailable: {exc}", None)

    def _toggle_record(self) -> None:
        if self._recorder is not None:
            self._recorder.stop()
            return
        dev = self.cmb_device.currentData()
        if dev is None:
            self.main.flash("no input device — install sounddevice / check mic", err=True)
            return
        sr = int(self.cmb_rate.currentText())
        stop_audio()
        rec = RecorderWorker(dev, sr)
        rec.sig_level.connect(self._on_level)
        rec.sig_elapsed.connect(self._on_elapsed)
        rec.sig_finished.connect(self._on_finished)
        rec.sig_error.connect(self._on_error)
        rec.finished.connect(lambda: self.main.release_worker(rec))
        self.main.pin_worker(rec)
        self._recorder = rec
        self.btn_record.setText("■ STOP")
        self.main.flash("recording…")
        rec.start()

    @pyqtSlot(float)
    def _on_level(self, level: float) -> None:
        self.meter.setValue(int(level * 100))

    @pyqtSlot(float)
    def _on_elapsed(self, secs: float) -> None:
        self.lbl_elapsed.setText(fmt_dur(secs))

    @pyqtSlot(object, int)
    def _on_finished(self, audio: object, sr: int) -> None:
        self._recorder = None
        self.btn_record.setText("● RECORD")
        self.meter.setValue(0)
        self._last_audio = np.asarray(audio, dtype=np.float32)
        self._last_sr = sr
        peak = float(np.max(np.abs(self._last_audio))) if self._last_audio.size else 0.0
        dur = self._last_audio.size / sr
        self.lbl_take.setText(
            f"take: {dur:.1f}s @ {sr} Hz | peak {peak:.2f}"
            + ("  ⚠ low level" if peak < 0.1 else "")
            + ("  ⚠ clipping" if peak >= 0.999 else ""))
        for b in (self.btn_play, self.btn_save, self.btn_send):
            b.setEnabled(True)
        self.main.flash(f"capture complete — {dur:.1f}s")

    @pyqtSlot(str)
    def _on_error(self, msg: str) -> None:
        self._recorder = None
        self.btn_record.setText("● RECORD")
        self.meter.setValue(0)
        self.main.flash(f"capture error: {msg}", err=True)

    def _play_last(self) -> None:
        if self._last_audio is not None:
            play_audio(self._last_audio, self._last_sr)

    def _save_last(self, send: bool = False) -> None:
        if self._last_audio is None:
            return
        name = slugify(self.ed_name.text() or f"take_{time.strftime('%Y%m%d_%H%M%S')}")
        path = DIR_CAPTURES / f"{name}.wav"
        n = 1
        while path.exists():
            path = DIR_CAPTURES / f"{name}_{n}.wav"
            n += 1
        write_wav_int16(path, self._last_audio, self._last_sr)
        self.main.flash(f"saved {path.name}")
        if send:
            self.sig_capture_saved.emit(str(path))
            self.main.tabs.setCurrentIndex(1)


class ConditionTab(QWidget):
    sig_library_changed = pyqtSignal()

    def __init__(self, main: "MainWindow") -> None:
        super().__init__()
        self.main = main

        lay = QVBoxLayout(self)
        gb = QGroupBox("VOICE CONDITIONING — reference audio → .safetensors voice state")
        g = QGridLayout(gb)

        self.ed_source = QLineEdit()
        self.ed_source.setPlaceholderText("reference WAV / MP3 path")
        btn_browse = QPushButton("BROWSE")
        btn_browse.clicked.connect(self._browse)
        self.ed_voice_name = QLineEdit()
        self.ed_voice_name.setPlaceholderText("voice name  e.g. dad_reading")
        self.chk_consent = QCheckBox(
            "I have explicit consent to clone this voice (Kyutai model terms)")
        self.btn_condition = QPushButton("CONDITION VOICE")
        self.btn_condition.setObjectName("btn_go")
        self.btn_condition.clicked.connect(self._condition)
        self.lbl_state = QLabel("idle")
        self.lbl_state.setStyleSheet(f"color: {C_DIM};")

        g.addWidget(QLabel("source"), 0, 0)
        g.addWidget(self.ed_source, 0, 1)
        g.addWidget(btn_browse, 0, 2)
        g.addWidget(QLabel("voice"), 1, 0)
        g.addWidget(self.ed_voice_name, 1, 1, 1, 2)
        g.addWidget(self.chk_consent, 2, 1, 1, 2)
        g.addWidget(self.btn_condition, 3, 1)
        g.addWidget(self.lbl_state, 4, 0, 1, 3)
        g.setColumnStretch(1, 1)
        lay.addWidget(gb)

        note = QLabel(
            "notes: conditioning runs the full audio through the model — expect it to be slow.\n"
            "the exported .safetensors reloads near-instantly for every future generation.\n"
            "best source: 15–30 s, one speaker, dry signal, consistent tone.")
        note.setStyleSheet(f"color: {C_DIM};")
        lay.addWidget(note)
        lay.addStretch(1)

    @pyqtSlot(str)
    def set_source(self, path: str) -> None:
        self.ed_source.setText(path)
        if not self.ed_voice_name.text():
            self.ed_voice_name.setText(Path(path).stem)

    def _browse(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Reference audio", str(DIR_CAPTURES),
            "Audio (*.wav *.mp3 *.flac *.ogg);;All files (*)")
        if path:
            self.set_source(path)

    def _condition(self) -> None:
        src = Path(self.ed_source.text().strip().strip('"'))
        if not src.is_file():
            self.main.flash("source file not found", err=True)
            return
        if not self.chk_consent.isChecked():
            self.main.flash("consent checkbox is required before conditioning", err=True)
            return
        if not self.main.model_ready:
            self.main.flash("load the model first (top bar)", err=True)
            return
        name = slugify(self.ed_voice_name.text() or src.stem)
        dest = DIR_VOICES / f"{name}.safetensors"
        if dest.exists():
            if QMessageBox.question(
                    self, "Overwrite",
                    f"voice '{name}' exists — overwrite?") != QMessageBox.StandardButton.Yes:
                return
        secs = 0.0
        if src.suffix.lower() == ".wav":
            try:
                audio, sr = read_wav_f32(src)
                secs = round(audio.size / sr, 2)
            except Exception:
                secs = 0.0
        meta = {"name": name, "source": src.name, "source_seconds": secs}
        self.btn_condition.setEnabled(False)
        self.lbl_state.setText(f"conditioning [{name}]…")
        self.lbl_state.setStyleSheet(f"color: {C_AMBER};")
        self.main.model_worker.submit_condition(str(src), str(dest), meta)

    @pyqtSlot(str)
    def on_conditioned(self, dest: str) -> None:
        self.btn_condition.setEnabled(True)
        self.lbl_state.setText(f"OK → {Path(dest).name}")
        self.lbl_state.setStyleSheet(f"color: {C_GREEN};")
        self.sig_library_changed.emit()
        self.main.flash(f"voice conditioned: {Path(dest).stem}")

    @pyqtSlot(str)
    def on_condition_error(self, msg: str) -> None:
        self.btn_condition.setEnabled(True)
        low = msg.lower()
        if (("cloning" in low and "download" in low)
                or "401" in low or "403" in low or "gated" in low):
            msg += ("  [hint: accept terms at huggingface.co/kyutai/pocket-tts, "
                    "`hf auth login` with a Read token or set HF_TOKEN, "
                    "open a NEW terminal, retry]")
        self.lbl_state.setText(f"FAIL: {msg}")
        self.lbl_state.setStyleSheet(f"color: {C_RED};")
        self.main.flash(f"conditioning failed: {msg}", err=True)


class TestTab(QWidget):
    def __init__(self, main: "MainWindow") -> None:
        super().__init__()
        self.main = main
        self._last_render: Optional[tuple[np.ndarray, int, str]] = None

        lay = QVBoxLayout(self)
        gb = QGroupBox("SYNTHESIS TEST")
        g = QGridLayout(gb)

        self.cmb_voice = QComboBox()
        btn_reload = QPushButton("↻")
        btn_reload.setFixedWidth(34)
        btn_reload.clicked.connect(self.refresh_voices)
        self.txt = QPlainTextEdit()
        self.txt.setPlaceholderText("text to synthesize…")
        self.txt.setPlainText("Pocket Voice Lab online. Local synthesis path verified.")
        self.spn_frames = QSpinBox()
        self.spn_frames.setRange(-1, 50)
        self.spn_frames.setValue(-1)
        self.spn_frames.setSpecialValueText("auto")
        self.chk_autoplay = QCheckBox("auto-play")
        self.chk_autoplay.setChecked(True)
        self.chk_autosave = QCheckBox("auto-save render")
        self.btn_generate = QPushButton("GENERATE")
        self.btn_generate.setObjectName("btn_go")
        self.btn_generate.clicked.connect(self._generate)
        self.btn_play = QPushButton("REPLAY")
        self.btn_play.clicked.connect(self._replay)
        self.btn_stop = QPushButton("STOP")
        self.btn_stop.clicked.connect(stop_audio)
        self.btn_save = QPushButton("SAVE RENDER")
        self.btn_save.clicked.connect(self._save_render)
        for b in (self.btn_play, self.btn_save):
            b.setEnabled(False)
        self.lbl_result = QLabel("no render yet")
        self.lbl_result.setStyleSheet(f"color: {C_DIM};")

        g.addWidget(QLabel("voice"), 0, 0)
        g.addWidget(self.cmb_voice, 0, 1, 1, 3)
        g.addWidget(btn_reload, 0, 4)
        g.addWidget(self.txt, 1, 0, 1, 5)
        g.addWidget(QLabel("frames after EOS"), 2, 0)
        g.addWidget(self.spn_frames, 2, 1)
        g.addWidget(self.chk_autoplay, 2, 2)
        g.addWidget(self.chk_autosave, 2, 3)
        g.addWidget(self.btn_generate, 3, 0)
        g.addWidget(self.btn_play, 3, 1)
        g.addWidget(self.btn_stop, 3, 2)
        g.addWidget(self.btn_save, 3, 3)
        g.addWidget(self.lbl_result, 4, 0, 1, 5)
        g.setColumnStretch(3, 1)
        lay.addWidget(gb)
        lay.addStretch(1)
        self.refresh_voices()

    def refresh_voices(self, preselect: Optional[str] = None) -> None:
        self.cmb_voice.clear()
        for e in load_voice_entries():
            self.cmb_voice.addItem(e["name"], str(e["path"]))
        if preselect is not None:
            i = self.cmb_voice.findData(preselect)
            if i >= 0:
                self.cmb_voice.setCurrentIndex(i)

    def _generate(self) -> None:
        voice = self.cmb_voice.currentData()
        if voice is None:
            self.main.flash("no voice in library — condition one first", err=True)
            return
        if not self.main.model_ready:
            self.main.flash("load the model first (top bar)", err=True)
            return
        text = self.txt.toPlainText().strip()
        if not text:
            self.main.flash("nothing to synthesize", err=True)
            return
        frames = self.spn_frames.value()
        self.btn_generate.setEnabled(False)
        self.lbl_result.setText("generating…")
        self.lbl_result.setStyleSheet(f"color: {C_AMBER};")
        self.main.model_worker.submit_generate(
            voice, self.cmb_voice.currentText(), text,
            None if frames < 0 else frames)

    @pyqtSlot(object, int, float, str)
    def on_generated(self, audio: object, sr: int, secs: float, voice: str) -> None:
        self.btn_generate.setEnabled(True)
        a = np.asarray(audio, dtype=np.float32)
        dur = a.size / sr
        rtf = dur / secs if secs > 0 else 0.0
        self._last_render = (a, sr, voice)
        self.lbl_result.setText(
            f"[{voice}] {dur:.2f}s audio in {secs:.2f}s  ({rtf:.1f}x realtime)")
        self.lbl_result.setStyleSheet(f"color: {C_GREEN};")
        for b in (self.btn_play, self.btn_save):
            b.setEnabled(True)
        if self.chk_autoplay.isChecked():
            play_audio(a, sr)
        if self.chk_autosave.isChecked():
            self._save_render(silent=True)

    @pyqtSlot(str)
    def on_generate_error(self, msg: str) -> None:
        self.btn_generate.setEnabled(True)
        self.lbl_result.setText(f"FAIL: {msg}")
        self.lbl_result.setStyleSheet(f"color: {C_RED};")
        self.main.flash(f"generation failed: {msg}", err=True)

    def _replay(self) -> None:
        if self._last_render:
            play_audio(self._last_render[0], self._last_render[1])

    def _save_render(self, silent: bool = False) -> None:
        if not self._last_render:
            return
        a, sr, voice = self._last_render
        path = DIR_RENDERS / f"{slugify(voice)}_{time.strftime('%Y%m%d_%H%M%S')}.wav"
        write_wav_int16(path, a, sr)
        if not silent:
            self.main.flash(f"render saved: {path.name}")


class LibraryTab(QWidget):
    def __init__(self, main: "MainWindow") -> None:
        super().__init__()
        self.main = main

        lay = QVBoxLayout(self)
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(
            ["VOICE", "CREATED", "SOURCE", "REF LEN", "SIZE"])
        self.table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch)
        self.table.setAlternatingRowColors(True)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        lay.addWidget(self.table)

        row = QHBoxLayout()
        btn_refresh = QPushButton("REFRESH")
        btn_refresh.clicked.connect(self.refresh)
        btn_test = QPushButton("TEST →")
        btn_test.setObjectName("btn_go")
        btn_test.clicked.connect(self._test_selected)
        btn_rename = QPushButton("RENAME")
        btn_rename.clicked.connect(self._rename)
        btn_delete = QPushButton("DELETE")
        btn_delete.setObjectName("btn_danger")
        btn_delete.clicked.connect(self._delete)
        btn_open = QPushButton("OPEN FOLDER")
        btn_open.clicked.connect(self._open_folder)
        for b in (btn_refresh, btn_test, btn_rename, btn_delete, btn_open):
            row.addWidget(b)
        row.addStretch(1)
        lay.addLayout(row)
        self.refresh()

    def _selected(self) -> Optional[Path]:
        r = self.table.currentRow()
        if r < 0:
            return None
        item = self.table.item(r, 0)
        return Path(item.data(Qt.ItemDataRole.UserRole)) if item else None

    @pyqtSlot()
    def refresh(self) -> None:
        entries = load_voice_entries()
        self.table.setRowCount(len(entries))
        for r, e in enumerate(entries):
            cells = [
                e["name"], e["created"], e["source"],
                f"{e['source_seconds']:.1f}s" if e["source_seconds"] else "?",
                f"{e['size'] / 1024:.0f} KiB",
            ]
            for c, text in enumerate(cells):
                item = QTableWidgetItem(text)
                if c == 0:
                    item.setData(Qt.ItemDataRole.UserRole, str(e["path"]))
                self.table.setItem(r, c, item)

    def _test_selected(self) -> None:
        p = self._selected()
        if p is None:
            return
        self.main.test_tab.refresh_voices(preselect=str(p))
        self.main.tabs.setCurrentIndex(2)

    def _rename(self) -> None:
        p = self._selected()
        if p is None:
            return
        new, ok = QInputDialog.getText(self, "Rename voice", "new name:", text=p.stem)
        if not ok or not new.strip():
            return
        slug = slugify(new)
        dest = DIR_VOICES / f"{slug}.safetensors"
        if dest.exists():
            self.main.flash("a voice with that name already exists", err=True)
            return
        mp_old = voice_meta_path(p)
        p.rename(dest)
        if mp_old.exists():
            try:
                meta = json.loads(mp_old.read_text(encoding="utf-8"))
            except Exception:
                meta = {}
            meta["name"] = slug
            voice_meta_path(dest).write_text(
                json.dumps(meta, indent=2), encoding="utf-8")
            mp_old.unlink(missing_ok=True)
        self.refresh()
        self.main.test_tab.refresh_voices()
        self.main.flash(f"renamed → {slug}")

    def _delete(self) -> None:
        p = self._selected()
        if p is None:
            return
        if QMessageBox.question(
                self, "Delete voice",
                f"delete '{p.stem}' permanently?") != QMessageBox.StandardButton.Yes:
            return
        p.unlink(missing_ok=True)
        voice_meta_path(p).unlink(missing_ok=True)
        self.refresh()
        self.main.test_tab.refresh_voices()
        self.main.flash(f"deleted {p.stem}")

    def _open_folder(self) -> None:
        try:
            os.startfile(str(DIR_VOICES))  # type: ignore[attr-defined]  # Windows
        except AttributeError:
            import subprocess
            subprocess.Popen(["xdg-open", str(DIR_VOICES)])


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(f"{APP_NAME} v{APP_VERSION} — {ORG_TAG}")
        self.resize(920, 640)
        self.model_ready = False
        self._worker_refs: set[QThread] = set()

        # persistent model worker
        self.model_worker = ModelWorker()
        self.pin_worker(self.model_worker)
        self.model_worker.start()

        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        self.setCentralWidget(central)

        # -- header ---------------------------------------------------------
        hdr = QFrame()
        hdr.setObjectName("hdr")
        h = QHBoxLayout(hdr)
        title = QLabel("POCKET VOICE LAB")
        title.setObjectName("hdr_title")
        sub = QLabel(f"local capture · conditioning · synthesis  //  pocket-tts")
        sub.setObjectName("hdr_sub")
        self.chip_model = Chip()
        self.chip_model.set_state("MODEL: NOT LOADED", "warn")
        btn_log = QPushButton("LOG")
        btn_log.setToolTip(str(LOG_PATH))
        btn_log.clicked.connect(self._open_log)
        h.addWidget(title)
        h.addWidget(sub)
        h.addStretch(1)
        h.addWidget(self.chip_model)
        h.addWidget(btn_log)
        root.addWidget(hdr)

        # -- model control strip ---------------------------------------------
        strip = QFrame()
        strip.setObjectName("hdr")
        s = QHBoxLayout(strip)
        self.cmb_lang = QComboBox()
        self.cmb_lang.addItems(LANGUAGES)
        self.spn_temp = QDoubleSpinBox()
        self.spn_temp.setRange(0.0, 2.0)
        self.spn_temp.setSingleStep(0.05)
        self.spn_temp.setValue(0.7)
        self.spn_steps = QSpinBox()
        self.spn_steps.setRange(1, 16)
        self.spn_steps.setValue(1)
        self.spn_eos = QDoubleSpinBox()
        self.spn_eos.setRange(-10.0, 0.0)
        self.spn_eos.setSingleStep(0.5)
        self.spn_eos.setValue(-4.0)
        self.chk_quant = QCheckBox("int8")
        self.btn_load = QPushButton("LOAD MODEL")
        self.btn_load.setObjectName("btn_go")
        self.btn_load.clicked.connect(self._load_model)
        for w_label, w in (("lang", self.cmb_lang), ("temp", self.spn_temp),
                           ("steps", self.spn_steps), ("eos", self.spn_eos)):
            s.addWidget(QLabel(w_label))
            s.addWidget(w)
        s.addWidget(self.chk_quant)
        s.addWidget(self.btn_load)
        s.addStretch(1)
        root.addWidget(strip)

        # -- tabs -------------------------------------------------------------
        self.tabs = QTabWidget()
        self.capture_tab = CaptureTab(self)
        self.condition_tab = ConditionTab(self)
        self.test_tab = TestTab(self)
        self.library_tab = LibraryTab(self)
        self.tabs.addTab(self.capture_tab, "CAPTURE")
        self.tabs.addTab(self.condition_tab, "CONDITION")
        self.tabs.addTab(self.test_tab, "TEST")
        self.tabs.addTab(self.library_tab, "LIBRARY")
        root.addWidget(self.tabs, 1)

        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage(f"library: {DIR_ROOT}")

        # -- wiring ------------------------------------------------------------
        mw = self.model_worker
        mw.sig_status.connect(self.flash)
        mw.sig_model_ready.connect(self._on_model_ready)
        mw.sig_model_error.connect(self._on_model_error)
        mw.sig_conditioned.connect(self.condition_tab.on_conditioned)
        mw.sig_condition_error.connect(self.condition_tab.on_condition_error)
        mw.sig_generated.connect(self.test_tab.on_generated)
        mw.sig_generate_error.connect(self.test_tab.on_generate_error)
        self.capture_tab.sig_capture_saved.connect(self.condition_tab.set_source)
        self.condition_tab.sig_library_changed.connect(self.library_tab.refresh)
        self.condition_tab.sig_library_changed.connect(
            lambda: self.test_tab.refresh_voices())

    # -- worker pinning (Windows GC-safety) ---------------------------------
    def pin_worker(self, w: QThread) -> None:
        self._worker_refs.add(w)

    def release_worker(self, w: QThread) -> None:
        self._worker_refs.discard(w)

    # -- model lifecycle ------------------------------------------------------
    def _load_model(self) -> None:
        params = ModelParams(
            language=self.cmb_lang.currentText(),
            temp=self.spn_temp.value(),
            lsd_decode_steps=self.spn_steps.value(),
            eos_threshold=self.spn_eos.value(),
            quantize=self.chk_quant.isChecked(),
        )
        self.model_ready = False
        self.btn_load.setEnabled(False)
        self.chip_model.set_state("MODEL: LOADING…", "warn")
        self.model_worker.submit_load(params)

    @pyqtSlot(int, str)
    def _on_model_ready(self, sample_rate: int, summary: str) -> None:
        self.model_ready = True
        self.btn_load.setEnabled(True)
        self.btn_load.setText("RELOAD MODEL")
        self.chip_model.set_state(f"MODEL: READY @ {sample_rate} Hz", "ok")
        self.flash(f"model ready — {summary}")

    @pyqtSlot(str)
    def _on_model_error(self, msg: str) -> None:
        self.model_ready = False
        self.btn_load.setEnabled(True)
        self.chip_model.set_state("MODEL: ERROR", "err")
        hint = ""
        low = msg.lower()
        if "401" in low or "403" in low or "gated" in low or "auth" in low:
            hint = ("  [hint: accept terms at huggingface.co/kyutai/pocket-tts "
                    "then `hf auth login`]")
        self.flash(f"model load failed: {msg}{hint}", err=True)

    # -- status helper -----------------------------------------------------
    @pyqtSlot(str)
    def flash(self, msg: str, err: bool = False) -> None:
        (log.error if err else log.info)("status: %s", msg)
        self.statusBar().setStyleSheet(
            f"color: {C_RED if err else C_DIM};")
        self.statusBar().showMessage(msg, 12000)

    def _open_log(self) -> None:
        try:
            os.startfile(str(LOG_PATH))  # type: ignore[attr-defined]  # Windows
        except AttributeError:
            import subprocess
            subprocess.Popen(["xdg-open", str(LOG_PATH)])
        except OSError:
            self.flash(f"log: {LOG_PATH}")

    # -- shutdown ------------------------------------------------------------
    def closeEvent(self, event) -> None:  # noqa: N802 (Qt override)
        stop_audio()
        if self.capture_tab._recorder is not None:
            self.capture_tab._recorder.stop()
        self.model_worker.shutdown()
        self.model_worker.wait(3000)
        super().closeEvent(event)


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=f"{APP_NAME} v{APP_VERSION}")
    parser.add_argument("--smoke", action="store_true",
                        help="offscreen construct-and-exit smoke test")
    parser.add_argument("--verbose", action="store_true",
                        help="echo DEBUG log lines to the console")
    args = parser.parse_args()

    setup_logging(verbose=args.verbose)

    if args.smoke:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    app = QApplication(sys.argv)
    app.setStyleSheet(QSS)
    app.setFont(QFont("JetBrains Mono", 9))

    win = MainWindow()

    if args.smoke:
        checks = {
            "tabs": win.tabs.count() == 4,
            "model_worker_alive": win.model_worker.isRunning(),
            "dirs": all(d.is_dir() for d in (DIR_CAPTURES, DIR_VOICES, DIR_RENDERS)),
            "chip": "NOT LOADED" in win.chip_model.text(),
            "log_file": LOG_PATH.is_file() and LOG_PATH.stat().st_size > 0,
        }
        win.close()
        ok = all(checks.values())
        log.info("smoke result: %s", checks)
        print(f"SMOKE {'PASS' if ok else 'FAIL'}: {checks}")
        return 0 if ok else 1

    win.show()
    rc = app.exec()
    log.info("clean exit rc=%d", rc)
    return rc


if __name__ == "__main__":
    sys.exit(main())
