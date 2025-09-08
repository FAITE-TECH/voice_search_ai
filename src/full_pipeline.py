# src/full_pipeline.py
import argparse
import os
import json
from src.stt.transcribe import transcribe_audio
from src.nlp.nlp_intent import extract_intent_entities
from src.nlp.response_generator import generate_response
from src.knowledge_base.faiss_search import FAQSearch
from src.tts.speak import speak_text

# Load knowledge_base once to pass into response generator
DATA_PATH = os.path.join(os.path.dirname(__file__), "../data/sample_data.json")
with open(DATA_PATH, "r", encoding="utf-8") as f:
    knowledge_base = json.load(f)


def run_full_pipeline(audio_path: str, faq_csv: str, whisper_model: str = "base", k: int = 3):
    # 1) STT
    print(f"[STT] Transcribing: {audio_path}")
    stt_result = transcribe_audio(audio_path, model_name=whisper_model)
    text = stt_result.get("text", "").strip()
    print(f"[STT] Text: {text!r}")

    # 2) NLP
    print("[NLP] Extracting intent & entities...")
    nlp_result = extract_intent_entities(text)
    intent = nlp_result["intent"]
    entities = nlp_result["entities"]
    print(f"[NLP] Intent: {intent}, Entities: {entities}")

    # 3) KB Search for multiple FAQ matches
    print("[KB] Loading FAQ index and searching...")
    faq_search = FAQSearch(faq_csv)
    top_results = faq_search.search(text, k=k)
    faq_responses = []
    for idx, dist in top_results:
        question = faq_search.get_question(idx)
        answer = faq_search.get_answer(idx)
        if question and answer and str(answer).lower() != "nan":
            faq_responses.append(f"{question}: {answer}")
    print(f"[KB] Top {k} FAQ matches:")
    for resp in faq_responses:
        print("-", resp)

    # 4) Generate response
    print("[NLP] Generating final response...")
    response = generate_response(nlp_result, faq_responses, knowledge_base=knowledge_base)
    print(f"[BOT] {response}")

    # 5) TTS
    print("[TTS] Speaking response...")
    speak_text(response)

    return {
        "transcription": text,
        "intent": intent,
        "entities": entities,
        "faq_matches": faq_responses,
        "response": response,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="End-to-end STT -> NLP -> FAQ -> TTS pipeline")
    parser.add_argument("--audio", required=True, help="Path to audio file (wav/mp3)")
    parser.add_argument("--faq", default="data/brand_faq.csv", help="Path to FAQ CSV")
    parser.add_argument("--whisper_model", default="base", help="Whisper model (tiny, base, small...)")
    parser.add_argument("-k", type=int, default=3, help="Number of top FAQ matches to return")
    args = parser.parse_args()

    if not os.path.exists(args.audio):
        raise FileNotFoundError(f"Audio file not found: {args.audio}")
    if not os.path.exists(args.faq):
        raise FileNotFoundError(f"FAQ CSV not found: {args.faq}")

    result = run_full_pipeline(args.audio, args.faq, whisper_model=args.whisper_model, k=args.k)

    # Save output JSON
    out_path = "full_pipeline_output.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"[DONE] Saved pipeline output to {out_path}")
