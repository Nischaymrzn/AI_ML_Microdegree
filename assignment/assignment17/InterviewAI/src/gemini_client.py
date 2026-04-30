from __future__ import annotations

import os


def load_env_file_if_available() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv()


load_env_file_if_available()


def _api_key() -> str:
    return os.getenv("GEMINI_API_KEY", "").strip()


def is_gemini_available() -> bool:
    return bool(_api_key())


def _build_prompt(qa_pairs: list[dict]) -> str:
    qa_text = "\n\n".join(
        f"Question {index}: {item['question']}\nAnswer: {item['answer']}"
        for index, item in enumerate(qa_pairs, start=1)
    )
    return f"""You are a friendly interview coach. Analyze the following 3 interview answers. Give a realistic score out of 100, a performance level, strengths, weaknesses, and practical suggestions. Be concise, supportive, and specific. Do not be overly harsh. Encourage the candidate to use the STAR method where useful.

Questions and Answers:
{qa_text}

Return the response exactly in this structure:
Overall Score: __/100
Performance Level:
Strengths:
-
Weaknesses:
-
Suggestions:
-
Final Advice:
"""


def generate_gemini_feedback(qa_pairs: list[dict]) -> str:
    """Generate structured interview feedback with Gemini."""
    if not is_gemini_available():
        raise RuntimeError("GEMINI_API_KEY is missing.")

    try:
        import google.generativeai as genai
    except ImportError as exc:
        raise RuntimeError(
            "google-generativeai is not installed. Run: pip install -r requirements.txt"
        ) from exc

    try:
        genai.configure(api_key=_api_key())
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(_build_prompt(qa_pairs))
    except Exception as exc:
        raise RuntimeError(f"Gemini feedback generation failed: {exc}") from exc

    text = (getattr(response, "text", "") or "").strip()
    if not text:
        raise RuntimeError("Gemini returned an empty response.")
    return text
