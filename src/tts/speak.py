# src/tts/speak.py
from gtts import gTTS
import os
import tempfile
import playsound


def speak_text(text: str, lang: str = "en"):
    """
    Convert text to speech and play it.
    """
    try:
        # Generate speech with gTTS
        tts = gTTS(text=text, lang=lang)

        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
            temp_path = fp.name
            tts.save(temp_path)

        # Play the audio
        playsound.playsound(temp_path)

        # Cleanup
        os.remove(temp_path)
    except Exception as e:
        print(f"[TTS ERROR] {e}")
