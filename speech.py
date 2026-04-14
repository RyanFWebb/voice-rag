"""
speech.py — Speech-to-text (Whisper) and text-to-speech (Kokoro).

STT:  faster-whisper, CPU, int8 quantized
TTS:  Kokoro-82M, 24kHz, streaming output to speakers via sounddevice

Platform note:
  - Mac:     brew install espeak-ng portaudio ffmpeg
  - Windows: install eSpeak NG to default path; ffmpeg via winget or scoop
  - Linux:   sudo apt install espeak-ng portaudio19-dev ffmpeg
"""

import os
import platform
import shutil
import numpy as np
import soundfile as sf
import sounddevice as sd
from faster_whisper import WhisperModel
from kokoro import KPipeline

from config import (
    WHISPER_MODEL_SIZE,
    WHISPER_DEVICE,
    WHISPER_COMPUTE,
    TTS_VOICE,
    TTS_SPEED,
    TTS_SAMPLE_RATE,
    RECORD_SECONDS,
    RECORD_SAMPLE_RATE,
    IS_WINDOWS,
    WINDOWS_ESPEAK_DIR,
)


def _ensure_espeak_on_path() -> None:
    """Add eSpeak NG to PATH on Windows if needed."""
    if IS_WINDOWS and os.path.isdir(WINDOWS_ESPEAK_DIR):
        if WINDOWS_ESPEAK_DIR not in os.environ.get("PATH", ""):
            os.environ["PATH"] += os.pathsep + WINDOWS_ESPEAK_DIR


def check_dependencies() -> dict[str, str | None]:
    """Return availability of required external tools."""
    _ensure_espeak_on_path()
    return {
        "ffmpeg":    shutil.which("ffmpeg"),
        "espeak-ng": shutil.which("espeak-ng"),
    }


# Model loading

_whisper_model: WhisperModel | None = None
_tts_pipeline:  KPipeline | None = None


def load_models() -> None:
    """Load Whisper and Kokoro models into module-level singletons."""
    global _whisper_model, _tts_pipeline

    _ensure_espeak_on_path()

    if _whisper_model is None:
        print(f"Loading Whisper ({WHISPER_MODEL_SIZE}, {WHISPER_DEVICE}, {WHISPER_COMPUTE})...")
        _whisper_model = WhisperModel(
            WHISPER_MODEL_SIZE, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE
        )
        print("Whisper ready.")

    if _tts_pipeline is None:
        print("Loading Kokoro TTS...")
        _tts_pipeline = KPipeline(lang_code="a")
        print("Kokoro ready.")


# STT

def transcribe(audio_path: str) -> str:
    """
    Transcribe a WAV/MP3 file to text using faster-whisper.
    Calls load_models() automatically if not already loaded.
    """
    if _whisper_model is None:
        load_models()
    segments, _ = _whisper_model.transcribe(audio_path)
    return " ".join(seg.text.strip() for seg in segments).strip()


def record(
    seconds: int = RECORD_SECONDS,
    sample_rate: int = RECORD_SAMPLE_RATE,
    output_path: str = "rag_question.wav",
) -> str:
    """
    Record audio from the default microphone and save to a WAV file.
    Returns the path to the saved file.
    """
    print(f"Recording for {seconds}s... (speak now)")
    recording = sd.rec(
        int(seconds * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
    )
    sd.wait()
    sf.write(output_path, recording, sample_rate)
    print(f"Saved to {output_path}")
    return output_path


# TTS

def speak(
    text: str,
    voice: str = TTS_VOICE,
    speed: float = TTS_SPEED,
    output_path: str | None = None,
) -> np.ndarray:
    """
    Convert text to speech and play it through speakers in real time.
    Each chunk is played as it is generated (streaming).

    Optionally saves the full audio to output_path (.wav).
    Returns the full audio as a numpy float32 array.
    """
    if _tts_pipeline is None:
        load_models()

    audio_chunks = []
    stream = sd.OutputStream(
        samplerate=TTS_SAMPLE_RATE, channels=1, dtype="float32"
    )
    stream.start()

    for _, _, audio in _tts_pipeline(text, voice=voice, speed=speed):
        stream.write(audio.reshape(-1, 1))
        audio_chunks.append(audio)

    stream.stop()
    stream.close()

    full_audio = (
        np.concatenate(audio_chunks) if audio_chunks else np.zeros(2400, dtype=np.float32)
    )
    if output_path:
        sf.write(output_path, full_audio, TTS_SAMPLE_RATE)
    return full_audio


def synthesize(
    text: str,
    voice: str = TTS_VOICE,
    speed: float = TTS_SPEED,
) -> np.ndarray:
    """
    Generate TTS audio without playing it (non-streaming, for file export).
    Returns full audio array.
    """
    if _tts_pipeline is None:
        load_models()

    chunks = []
    for _, _, audio in _tts_pipeline(text, voice=voice, speed=speed):
        chunks.append(audio)
    return np.concatenate(chunks) if chunks else np.zeros(2400, dtype=np.float32)
