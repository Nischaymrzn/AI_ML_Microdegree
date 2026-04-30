# InterviewAI

Voice-based interview practice bot built with Python and Streamlit.

InterviewAI asks exactly three interview questions, accepts spoken answers through microphone recording or uploaded audio files, transcribes answers locally with Whisper, generates interview feedback with optional Gemini AI, falls back to rule-based scoring when Gemini is not configured, and converts the final feedback into offline speech with pyttsx3.

## Features

- Simple Streamlit web interface.
- Light green UI theme with Space Grotesk and Outfit font styling when Google Fonts can load.
- Three fixed interview practice questions.
- Microphone recording with a free Streamlit recorder component and native `st.audio_input` fallback.
- Audio upload fallback for WAV, MP3, and M4A files.
- Manual transcript fallback for demo situations where browser microphone recording is unavailable.
- Local Whisper automatic speech recognition.
- Transcript display for every answer.
- Optional Gemini feedback using `GEMINI_API_KEY`.
- Free local rule-based feedback fallback.
- Overall score out of 100.
- Performance level: Beginner, Developing, Good, or Excellent.
- Strengths, weaknesses, suggestions, and final advice.
- STAR method improvement guidance.
- Offline text-to-speech using pyttsx3.
- Local transcript, feedback, and audio saving.
- Download buttons for transcript, feedback, and feedback audio.

## Architecture Diagram

```text
User speaks, uploads audio, or types transcript
        |
        v
Streamlit app.py
        |
        +--> Save audio in outputs/audio/
        |
        +--> Local Whisper ASR
        |        |
        |        v
        |    Transcript
        |
        +--> Feedback generator
        |        |
        |        +--> Gemini API if GEMINI_API_KEY exists
        |        |
        |        +--> Rule-based evaluator if Gemini is missing or fails
        |
        +--> pyttsx3 offline TTS
        |
        v
Text feedback + playable WAV audio
```

More details are available in [docs/architecture.md](docs/architecture.md).

## Tech Stack

- Python
- Streamlit
- streamlit-mic-recorder
- openai-whisper
- google-generativeai
- pyttsx3
- pydub
- python-dotenv
- FFmpeg

## Why This Project Is Free-Cost

This project uses local and open-source tools wherever possible:

- Speech-to-text uses local Whisper, not a paid hosted STT service.
- Text-to-speech uses offline pyttsx3, not ElevenLabs, Azure, AWS, or other paid TTS services.
- Gemini is optional. If no API key is provided, the app uses a local rule-based evaluator.
- Output files are saved locally on your machine.

## Installation

Clone or download the project, then open a terminal in the project folder.

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## FFmpeg Installation on Windows

Whisper needs FFmpeg to read and convert audio files.

### Option 1: Install with winget

```powershell
winget install Gyan.FFmpeg
```

Close and reopen your terminal after installation.

### Option 2: Manual installation

1. Go to https://www.gyan.dev/ffmpeg/builds/
2. Download a release build.
3. Extract the ZIP file.
4. Add the `bin` folder to your Windows PATH.
5. Restart your terminal.

Check installation:

```powershell
ffmpeg -version
```

## Gemini API Key Setup

Gemini is optional. The app works without it using rule-based feedback.

To use Gemini:

1. Visit https://aistudio.google.com/
2. Sign in with a Google account.
3. Create an API key.
4. Copy `.env.example` to `.env`.
5. Paste your key:

```env
GEMINI_API_KEY=your_gemini_api_key_here
WHISPER_MODEL=base
```

`WHISPER_MODEL=base` is a good default for student laptops. You can try `small` for better quality if your computer can handle it.

## Run the App

```powershell
streamlit run app.py
```

Then open the local URL shown by Streamlit.

## How to Use

1. Open the app.
2. Read Question 1.
3. Choose an input method: upload audio, record with microphone, or type a transcript manually.
4. Click **Transcribe Answer** for audio, or **Save Typed Answer** for manual text.
5. Repeat for Questions 2 and 3.
6. Click **Generate Final Feedback**.
7. Review your score, level, strengths, weaknesses, suggestions, and final advice.
8. Play the spoken feedback inside the app.
9. Download the transcript, text feedback, or WAV feedback audio if needed.

## Microphone Troubleshooting

If the microphone recorder shows **An error has occurred, please try again** after clicking Stop:

- Use Chrome or Edge.
- Open the app from `http://localhost:8501`, not from a file path.
- Allow microphone permission in the browser.
- Close other apps that may be using the microphone.
- Refresh the page and try again.
- Use **Upload audio file** or **Type transcript manually** if the browser recorder still fails.

The upload and manual transcript modes are included so the project can still be demonstrated even when the browser blocks microphone recording.

## Interview Questions

1. Tell me about yourself.
2. Why do you want this role?
3. Describe a project you worked on and what you learned from it.

## Folder Structure

```text
InterviewAI/
|
+-- app.py
+-- requirements.txt
+-- README.md
+-- .env.example
+-- .gitignore
+-- .streamlit/
|   +-- config.toml
|
+-- src/
|   +-- __init__.py
|   +-- asr.py
|   +-- evaluator.py
|   +-- gemini_client.py
|   +-- tts.py
|   +-- utils.py
|
+-- outputs/
|   +-- audio/
|   +-- transcripts/
|
+-- docs/
    +-- architecture.md
```

## Screenshots

Add screenshots here after running the app:

- Home screen
- Transcript section
- Final feedback section
- Spoken feedback section

## Future Improvements

- Add interview categories such as HR, technical, and behavioral.
- Add more question sets.
- Add answer timing and speaking pace analysis.
- Add confidence score based on transcript clarity.
- Export a complete PDF report.
- Add support for multiple languages supported by Whisper.

## License

This project is provided for learning and academic use. You may use, modify, and share it freely.
