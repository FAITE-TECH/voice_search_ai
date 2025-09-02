import whisper
from typing import Dict

_model = None

def load_whisper(model_name: str = "base"):
    global _model
    if _model is None:
        _model = whisper.load_model(model_name)
    return _model

def transcribe_audio(audio_path: str, model_name: str = "base") -> Dict:
    model = load_whisper(model_name)
    result = model.transcribe(audio_path)
    return result