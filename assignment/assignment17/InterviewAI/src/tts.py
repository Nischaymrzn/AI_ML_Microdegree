from __future__ import annotations

from pathlib import Path


def text_to_speech(text: str, output_path: str) -> str:
    """Convert feedback text to speech using offline pyttsx3 and save a WAV file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import pyttsx3
    except ImportError as exc:
        raise RuntimeError("pyttsx3 is not installed. Run: pip install -r requirements.txt") from exc

    try:
        engine = pyttsx3.init()
        engine.setProperty("rate", 165)
        engine.setProperty("volume", 0.95)
        engine.save_to_file(text, str(path))
        engine.runAndWait()
        engine.stop()
    except Exception as exc:
        raise RuntimeError(
            "Could not generate speech with pyttsx3. Check that an offline system voice is installed."
        ) from exc

    if not path.exists() or path.stat().st_size == 0:
        raise RuntimeError("TTS finished but no audio file was created.")
    return str(path)
