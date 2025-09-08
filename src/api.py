from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from typing import Optional
import tempfile
import shutil
import os

from src.full_pipeline import run_full_pipeline

app = FastAPI(
    title="Voice Search AI API",
    description=(
        "Upload audio, get transcription + intent + FAQ matches + voice-ready response.\n\n"
        "Optional FAQ CSV can be uploaded, otherwise the server uses `data/brand_faq.csv`.\n"
    ),
    version="1.0",
)


@app.get("/health", summary="Health Check")
async def health_check():
    """Health check endpoint. Returns 200 if the API is alive."""
    return {"status": "ok"}


@app.post("/query", summary="Query Endpoint")
async def query_endpoint(
    audio: UploadFile = File(..., description="Audio file (wav, mp3)"),
    faq: Optional[UploadFile] = File(None, description="Optional FAQ CSV; if omitted server uses default"),
    whisper_model: str = Form("base", description="Whisper model size (tiny, base, small...)"),
    k: int = Form(3, description="Number of top FAQ matches to return"),
):
    """
    Main Voice Search AI endpoint.

    Steps:
    - Accepts an audio file (wav/mp3)
    - Optional FAQ CSV (if none provided, uses default `data/brand_faq.csv`)
    - Runs full pipeline: STT → NLP → FAQ search → Response generation → TTS
    """
    try:
        # Save audio to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(audio.filename)[-1]) as tmp_audio:
            shutil.copyfileobj(audio.file, tmp_audio)
            tmp_audio_path = tmp_audio.name

        # If FAQ uploaded, save it, else fallback to default
        if faq is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_faq:
                shutil.copyfileobj(faq.file, tmp_faq)
                faq_csv = tmp_faq.name
        else:
            faq_csv = os.path.join(os.path.dirname(__file__), "../data/brand_faq.csv")

        # Run pipeline
        result = run_full_pipeline(
            audio_path=tmp_audio_path,
            faq_csv=faq_csv,
            whisper_model=whisper_model,
            k=k,
        )

        # Cleanup temp files
        os.unlink(tmp_audio_path)
        if faq is not None and os.path.exists(faq_csv):
            os.unlink(faq_csv)

        return JSONResponse(content=result)

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
