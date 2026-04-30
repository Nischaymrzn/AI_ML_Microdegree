from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import BinaryIO


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
AUDIO_DIR = OUTPUTS_DIR / "audio"
TRANSCRIPTS_DIR = OUTPUTS_DIR / "transcripts"


def ensure_output_dirs() -> None:
    """Create output folders used by the app."""
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def safe_filename(name: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_.-]+", "_", name).strip("._")
    return cleaned or "audio"


def save_audio_file(file_obj: BinaryIO, filename: str, question_number: int) -> str:
    """Save recorded or uploaded audio and return the local path."""
    ensure_output_dirs()
    suffix = Path(filename).suffix or ".wav"
    output_name = f"question_{question_number}_{timestamp()}_{safe_filename(Path(filename).stem)}{suffix}"
    output_path = AUDIO_DIR / output_name

    data = file_obj.getvalue() if hasattr(file_obj, "getvalue") else file_obj.read()
    output_path.write_bytes(data)
    return str(output_path)


def save_text_file(content: str, prefix: str) -> str:
    """Save text content in outputs/transcripts and return the path."""
    ensure_output_dirs()
    output_path = TRANSCRIPTS_DIR / f"{safe_filename(prefix)}_{timestamp()}.txt"
    output_path.write_text(content, encoding="utf-8")
    return str(output_path)


def format_transcript_export(qa_pairs: list[dict]) -> str:
    lines = ["InterviewAI Transcript", ""]
    for index, item in enumerate(qa_pairs, start=1):
        lines.append(f"Question {index}: {item.get('question', '')}")
        lines.append(f"Answer: {item.get('answer', '').strip() or '[No transcript]'}")
        lines.append("")
    return "\n".join(lines).strip() + "\n"
