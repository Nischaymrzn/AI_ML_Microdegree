from __future__ import annotations

from io import BytesIO
from pathlib import Path

import streamlit as st

from src.asr import transcribe_audio
from src.evaluator import generate_feedback
from src.tts import text_to_speech
from src.utils import (
    AUDIO_DIR,
    ensure_output_dirs,
    format_transcript_export,
    save_audio_file,
    save_text_file,
    timestamp,
)


QUESTIONS = [
    "Tell me about yourself.",
    "Why do you want this role?",
    "Describe a project you worked on and what you learned from it.",
]


def load_env() -> None:
    """Load .env only if python-dotenv is installed."""
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass


def setup_page() -> None:
    st.set_page_config(page_title="InterviewAI", layout="centered")
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

        html, body, [class*="css"] {
            font-family: "Outfit", "Segoe UI", Arial, sans-serif;
        }

        .stApp {
            background: linear-gradient(180deg, #f7fdf5 0%, #f4fbf2 100%);
            color: #17351f;
        }

        h1, h2, h3 {
            font-family: "Space Grotesk", "Outfit", "Segoe UI", Arial, sans-serif;
            color: #17351f;
        }

        .stButton > button,
        .stDownloadButton > button {
            border-radius: 8px;
            border: 1px solid #2f8a4c;
            background-color: #2f8a4c;
            color: white;
            font-weight: 600;
        }

        .stButton > button:hover,
        .stDownloadButton > button:hover {
            border-color: #22683a;
            background-color: #22683a;
            color: white;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def init_state() -> None:
    defaults = {
        "question_index": 0,
        "transcripts": ["", "", ""],
        "audio_paths": ["", "", ""],
        "feedback": "",
        "score": None,
        "level": "",
        "source": "",
        "feedback_audio": "",
        "transcript_file": "",
        "feedback_file": "",
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def reset_app() -> None:
    st.session_state.clear()
    init_state()


def get_qa_pairs() -> list[dict]:
    return [
        {"question": question, "answer": st.session_state.transcripts[index].strip()}
        for index, question in enumerate(QUESTIONS)
    ]


def get_recorded_audio(question_index: int):
    """Return recorded audio from streamlit-mic-recorder, or native Streamlit fallback."""
    try:
        from streamlit_mic_recorder import mic_recorder

        recorded = mic_recorder(
            start_prompt="Start recording",
            stop_prompt="Stop recording",
            just_once=True,
            use_container_width=True,
            format="wav",
            key=f"recorder_{question_index}",
        )
        if recorded and recorded.get("bytes"):
            audio = BytesIO(recorded["bytes"])
            audio.name = f"recorded_question_{question_index + 1}.wav"
            audio.size = len(recorded["bytes"])
            return audio
    except Exception:
        pass

    if hasattr(st, "audio_input"):
        st.caption("Using Streamlit's built-in recorder fallback.")
        return st.audio_input(
            "Record answer",
            sample_rate=16000,
            key=f"native_recorder_{question_index}",
        )

    st.info("Microphone recording is not available. Please upload audio or type your answer.")
    return None


def show_audio_input(question_index: int):
    input_method = st.radio(
        "Answer input method",
        ["Upload audio", "Record audio", "Type answer"],
        horizontal=True,
        key=f"input_method_{question_index}",
    )

    if input_method == "Upload audio":
        return (
            input_method,
            st.file_uploader(
                "Upload WAV, MP3, or M4A",
                type=["wav", "mp3", "m4a"],
                key=f"upload_{question_index}",
            ),
        )

    if input_method == "Record audio":
        st.caption(
            "If recording fails in your browser, use upload mode or type the answer manually."
        )
        return input_method, get_recorded_audio(question_index)

    typed_answer = st.text_area(
        "Type your answer",
        height=140,
        key=f"typed_answer_{question_index}",
        placeholder="Type what you would say in the interview...",
    )
    if st.button("Save Typed Answer", type="primary"):
        save_typed_answer(question_index, typed_answer)
    return input_method, None


def audio_has_content(audio_file) -> bool:
    if audio_file is None:
        st.warning("Please record or upload an audio file first.")
        return False

    size = getattr(audio_file, "size", None)
    if size is None and hasattr(audio_file, "getvalue"):
        size = len(audio_file.getvalue())

    if size is not None and size < 1024:
        st.warning("The audio is empty or too short. Please try again.")
        return False

    return True


def transcribe_answer(question_index: int, audio_file) -> None:
    if not audio_has_content(audio_file):
        return

    filename = getattr(audio_file, "name", f"answer_{question_index + 1}.wav")

    try:
        with st.spinner("Transcribing with local Whisper..."):
            audio_path = save_audio_file(audio_file, filename, question_index + 1)
            transcript = transcribe_audio(audio_path)
    except Exception as exc:
        st.error(str(exc))
        return

    st.session_state.audio_paths[question_index] = audio_path
    st.session_state.transcripts[question_index] = transcript
    st.success("Answer transcribed.")
    go_next_question()


def save_typed_answer(question_index: int, text: str) -> None:
    text = " ".join(text.strip().split())
    if not text:
        st.warning("Please type your answer first.")
        return

    st.session_state.transcripts[question_index] = text
    st.session_state.audio_paths[question_index] = ""
    st.success("Typed answer saved.")
    go_next_question()


def go_next_question() -> None:
    if st.session_state.question_index < len(QUESTIONS) - 1:
        st.session_state.question_index += 1
        st.rerun()


def save_transcript_edits() -> None:
    for index in range(len(QUESTIONS)):
        key = f"transcript_{index}"
        if key in st.session_state:
            st.session_state.transcripts[index] = " ".join(st.session_state[key].strip().split())


def build_feedback() -> None:
    save_transcript_edits()

    if any(not answer.strip() for answer in st.session_state.transcripts):
        st.warning("Please answer all 3 questions first.")
        return

    qa_pairs = get_qa_pairs()
    transcript_text = format_transcript_export(qa_pairs)

    try:
        with st.spinner("Generating interview feedback..."):
            result = generate_feedback(qa_pairs)
    except Exception as exc:
        st.error(f"Could not generate feedback: {exc}")
        return

    st.session_state.score = result["score"]
    st.session_state.level = result["level"]
    st.session_state.source = result["source"]
    st.session_state.feedback = result["feedback_text"]
    st.session_state.transcript_file = save_text_file(transcript_text, "interview_transcript")
    st.session_state.feedback_file = save_text_file(result["feedback_text"], "interview_feedback")

    make_feedback_audio(result["feedback_text"])
    st.success("Final feedback is ready.")


def make_feedback_audio(feedback_text: str) -> None:
    audio_path = AUDIO_DIR / f"feedback_{timestamp()}.wav"

    try:
        with st.spinner("Creating spoken feedback..."):
            st.session_state.feedback_audio = text_to_speech(feedback_text, str(audio_path))
    except Exception as exc:
        st.session_state.feedback_audio = ""
        st.warning(f"Text feedback was created, but audio feedback failed: {exc}")


def show_question_section() -> None:
    st.header("1. Interview Questions")

    index = st.session_state.question_index
    st.progress((index + 1) / len(QUESTIONS))
    st.subheader(f"Question {index + 1} of {len(QUESTIONS)}")
    st.write(f"**{QUESTIONS[index]}**")

    input_method, audio_file = show_audio_input(index)

    if audio_file is not None:
        st.audio(audio_file.getvalue())

    if input_method != "Type answer" and st.button("Transcribe Answer", type="primary"):
        transcribe_answer(index, audio_file)

    col1, col2 = st.columns(2)
    with col1:
        if index > 0 and st.button("Previous Question"):
            st.session_state.question_index -= 1
            st.rerun()
    with col2:
        if index < len(QUESTIONS) - 1 and st.button("Next Question"):
            st.session_state.question_index += 1
            st.rerun()

    if st.session_state.transcripts[index]:
        st.text_area(
            "Review or edit transcript",
            value=st.session_state.transcripts[index],
            height=120,
            key=f"transcript_{index}",
        )


def show_transcripts_section() -> None:
    st.header("2. Transcripts")

    for index, question in enumerate(QUESTIONS):
        with st.expander(f"Question {index + 1}: {question}"):
            transcript = st.session_state.transcripts[index].strip()
            st.write(transcript or "No transcript yet.")

    if all(st.session_state.transcripts):
        st.download_button(
            "Download Transcript",
            format_transcript_export(get_qa_pairs()),
            file_name="interviewai_transcript.txt",
            mime="text/plain",
        )


def show_feedback_section() -> None:
    st.header("3. Final Feedback")

    if st.button("Generate Final Feedback", type="primary"):
        build_feedback()

    if not st.session_state.feedback:
        return

    col1, col2, col3 = st.columns(3)
    col1.metric("Score", f"{st.session_state.score}/100")
    col2.metric("Level", st.session_state.level)
    col3.metric("Source", st.session_state.source)

    st.text_area("Feedback", st.session_state.feedback, height=300)
    st.download_button(
        "Download Feedback",
        st.session_state.feedback,
        file_name="interviewai_feedback.txt",
        mime="text/plain",
    )

    if st.session_state.transcript_file:
        st.caption(f"Transcript saved to: {st.session_state.transcript_file}")
    if st.session_state.feedback_file:
        st.caption(f"Feedback saved to: {st.session_state.feedback_file}")


def show_spoken_feedback_section() -> None:
    st.header("4. Spoken Feedback")

    audio_path = st.session_state.feedback_audio
    if not audio_path or not Path(audio_path).exists():
        st.caption("Spoken feedback will appear here after feedback is generated.")
        return

    audio_bytes = Path(audio_path).read_bytes()
    st.audio(audio_bytes, format="audio/wav")
    st.download_button(
        "Download Feedback Audio",
        audio_bytes,
        file_name=Path(audio_path).name,
        mime="audio/wav",
    )


def main() -> None:
    load_env()
    ensure_output_dirs()
    setup_page()
    init_state()

    st.title("InterviewAI")
    st.subheader("Voice-based interview practice bot")
    st.write(
        "Answer three interview questions, transcribe audio locally with Whisper, "
        "and get feedback from Gemini or the built-in rule-based evaluator."
    )

    if st.button("Reset Interview"):
        reset_app()
        st.rerun()

    show_question_section()
    st.divider()
    show_transcripts_section()
    st.divider()
    show_feedback_section()
    st.divider()
    show_spoken_feedback_section()


if __name__ == "__main__":
    main()
