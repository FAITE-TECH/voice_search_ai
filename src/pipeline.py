# src/pipeline.py
import argparse
from src.stt.transcribe import transcribe_audio
from src.nlp.processor import extract_entities_and_basic_features
from src.knowledge_base.faiss_search import FAQSearch
import os
import json

def run_pipeline(audio_path: str, faq_csv: str, whisper_model: str = "base", k: int = 1):
    # 1) STT
    print(f"[STT] Transcribing: {audio_path}")
    stt_result = transcribe_audio(audio_path, model_name=whisper_model)
    text = stt_result.get("text", "").strip()
    print(f"[STT] Text: {text!r}")

    # 2) NLP
    print("[NLP] Extracting entities and tokens...")
    nlp_info = extract_entities_and_basic_features(text)
    print(f"[NLP] Entities: {nlp_info['entities']}")
    # 3) KB Search
    print("[KB] Building/Loading FAQ index...")
    faq_search = FAQSearch(faq_csv)
    print("[KB] Querying FAQ index...")
    results = faq_search.search(text, k=k)
    print("[KB] Top results (index, distance):", results)
    for idx, dist in results:
        question = faq_search.get_question(idx)
        answer = faq_search.get_answer(idx)
        print("----")
        print(f"[MATCH] FAQ question: {question}")
        print(f"[MATCH] Answer: {answer}")
    return {
        "transcription": text,
        "nlp": nlp_info,
        "matches": [{"index": int(idx), "distance": float(dist),
                     "question": faq_search.get_question(idx),
                     "answer": faq_search.get_answer(idx)} for idx, dist in results]
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="End-to-end STT -> NLP -> FAQ Search pipeline")
    parser.add_argument("--audio", required=True, help="Path to audio file (wav/mp3)")
    parser.add_argument("--faq", default="data/brand_faq.csv", help="Path to FAQ CSV")
    parser.add_argument("--whisper_model", default="base", help="Whisper model name (tiny, base, small...)")
    parser.add_argument("-k", type=int, default=1, help="Number of top FAQ matches to return")
    args = parser.parse_args()

    if not os.path.exists(args.audio):
        raise FileNotFoundError(f"Audio file not found: {args.audio}")
    if not os.path.exists(args.faq):
        raise FileNotFoundError(f"FAQ CSV not found: {args.faq}")

    out = run_pipeline(args.audio, args.faq, whisper_model=args.whisper_model, k=args.k)
    # optional: save output json
    out_path = "pipeline_output.json"
    with open(out_path, "w", encoding="utf-8") as f:
        import json
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"[DONE] Saved pipeline output to {out_path}")
