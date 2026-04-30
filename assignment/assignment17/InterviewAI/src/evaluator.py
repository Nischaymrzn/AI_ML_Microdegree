from __future__ import annotations

import re

from src.gemini_client import generate_gemini_feedback, is_gemini_available


FILLER_WORDS = {
    "um",
    "uh",
    "like",
    "basically",
    "actually",
    "literally",
    "you know",
    "sort of",
    "kind of",
}

EXAMPLE_WORDS = {
    "project",
    "example",
    "experience",
    "team",
    "role",
    "challenge",
    "problem",
    "solution",
    "result",
    "impact",
}

ACTION_WORDS = {
    "built",
    "created",
    "developed",
    "designed",
    "implemented",
    "improved",
    "solved",
    "led",
    "managed",
    "learned",
    "delivered",
}

STAR_WORDS = {
    "situation",
    "task",
    "action",
    "result",
    "learned",
    "improved",
    "built",
    "solved",
}


def _extract_score(feedback_text: str) -> int | None:
    match = re.search(r"Overall Score:\s*(\d{1,3})\s*/\s*100", feedback_text, re.IGNORECASE)
    if not match:
        match = re.search(r"score[^\d]{0,12}(\d{1,3})", feedback_text, re.IGNORECASE)
    if not match:
        return None
    return max(0, min(100, int(match.group(1))))


def _level_from_score(score: int) -> str:
    if score <= 39:
        return "Beginner"
    if score <= 59:
        return "Developing"
    if score <= 79:
        return "Good"
    return "Excellent"


def _count_words(text: str) -> int:
    return len(re.findall(r"\b[\w'-]+\b", text.lower()))


def _contains_any(text: str, words: set[str]) -> bool:
    lowered = text.lower()
    return any(word in lowered for word in words)


def _count_fillers(text: str) -> int:
    lowered = text.lower()
    return sum(len(re.findall(rf"\b{re.escape(word)}\b", lowered)) for word in FILLER_WORDS)


def rule_based_feedback(qa_pairs: list[dict]) -> dict:
    combined_answers = " ".join(item.get("answer", "") for item in qa_pairs)
    total_words = _count_words(combined_answers)
    score = 50

    if total_words >= 120:
        score += 15
    elif total_words >= 75:
        score += 10
    elif total_words >= 45:
        score += 5
    else:
        score -= 15

    complete_answers = sum(1 for item in qa_pairs if _count_words(item.get("answer", "")) >= 20)
    score += complete_answers * 5
    if complete_answers < 3:
        score -= (3 - complete_answers) * 5

    if _contains_any(combined_answers, EXAMPLE_WORDS):
        score += 8
    if _contains_any(combined_answers, ACTION_WORDS):
        score += 8
    if _contains_any(combined_answers, STAR_WORDS):
        score += 7

    filler_count = _count_fillers(combined_answers)
    if filler_count >= 10:
        score -= 10
    elif filler_count >= 5:
        score -= 5

    score = max(0, min(100, score))
    level = _level_from_score(score)

    strengths = []
    weaknesses = []
    suggestions = []

    if complete_answers >= 2:
        strengths.append("You gave enough detail in most answers.")
    else:
        weaknesses.append("Some answers were too short to show your skills clearly.")

    if _contains_any(combined_answers, ACTION_WORDS):
        strengths.append("You used action-focused language that shows ownership.")
    else:
        suggestions.append("Use stronger action words such as built, improved, solved, led, or delivered.")

    if _contains_any(combined_answers, EXAMPLE_WORDS):
        strengths.append("You included signs of real examples or project experience.")
    else:
        weaknesses.append("The answers need more specific examples from your work, study, or projects.")

    if filler_count >= 5:
        weaknesses.append("There were several filler words, which can reduce clarity.")
        suggestions.append("Pause briefly instead of using filler words like um, uh, or like.")

    suggestions.extend(
        [
            "Structure project answers with the STAR method: Situation, Task, Action, and Result.",
            "Mention measurable results or learning outcomes when possible.",
            "Practice a 45 to 60 second answer for each common interview question.",
        ]
    )

    if not strengths:
        strengths.append("You completed the practice session and have a clear starting point to improve.")
    if not weaknesses:
        weaknesses.append("The main opportunity is to add more measurable outcomes and sharper structure.")

    feedback_text = f"""Overall Score: {score}/100
Performance Level: {level}
Strengths:
{_format_bullets(strengths)}
Weaknesses:
{_format_bullets(weaknesses)}
Suggestions:
{_format_bullets(suggestions)}
Final Advice:
Keep your answers clear, specific, and example-based. For role motivation, connect your skills to the job. For project questions, use STAR so the interviewer can follow what happened, what you did, and what changed because of your work.
"""

    return {
        "score": score,
        "level": level,
        "feedback_text": feedback_text.strip(),
        "source": "rule-based",
    }


def _format_bullets(items: list[str]) -> str:
    return "\n".join(f"- {item}" for item in items)


def generate_feedback(qa_pairs: list[dict]) -> dict:
    """Try Gemini first, then fall back to local rule-based scoring."""
    if is_gemini_available():
        try:
            feedback_text = generate_gemini_feedback(qa_pairs)
            score = _extract_score(feedback_text)
            if score is None:
                fallback = rule_based_feedback(qa_pairs)
                score = fallback["score"]
            return {
                "score": score,
                "level": _level_from_score(score),
                "feedback_text": feedback_text,
                "source": "gemini",
            }
        except Exception as exc:
            fallback = rule_based_feedback(qa_pairs)
            fallback["feedback_text"] += (
                "\n\nNote: Gemini feedback was unavailable, so local rule-based feedback was used. "
                f"Reason: {exc}"
            )
            return fallback

    return rule_based_feedback(qa_pairs)
