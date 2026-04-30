from __future__ import annotations

import os
import shutil
from functools import lru_cache


def load_env_file_if_available() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv()


load_env_file_if_available()


@lru_cache(maxsize=1)
def load_whisper_model():
    """Load the Whisper model once per Python process."""
    try:
        import whisper
    except ImportError as exc:
        raise RuntimeError(
            "openai-whisper is not installed. Run: pip install -r requirements.txt"
        ) from exc

    model_name = os.getenv("WHISPER_MODEL", "base").strip() or "base"
    try:
        return whisper.load_model(model_name)
    except Exception as exc:
        raise RuntimeError(
            f"Could not load Whisper model '{model_name}'. "
            "Check your internet connection for the first download, or set WHISPER_MODEL=base."
        ) from exc


def _check_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "FFmpeg was not found. Install FFmpeg and add it to PATH, then restart the terminal."
        )


def transcribe_audio(audio_path: str) -> str:
    """Transcribe a local audio file with Whisper and return clean text."""
    _check_ffmpeg()
    model = load_whisper_model()

    try:
        result = model.transcribe(audio_path, fp16=False)
    except Exception as exc:
        raise RuntimeError(
            "Whisper could not transcribe this audio. Try a clearer WAV/MP3/M4A file."
        ) from exc

    transcript = (result.get("text") or "").strip()
    if not transcript:
        raise RuntimeError("No speech was detected in the audio file.")
    return " ".join(transcript.split())
