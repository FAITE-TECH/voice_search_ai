import json
import os
from nlp_intent import extract_intent_entities
from response_generator import generate_response

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "../../data/sample_data.json")

with open(DATA_PATH, "r", encoding="utf-8") as f:
    knowledge_base = json.load(f)

def handle_query(query: str) -> str:
    parsed = extract_intent_entities(query)

    response = generate_response(parsed)
    return response

if __name__ == "__main__":
    test_queries = [
        "What are your vegan options?",
        "Do you have gluten free items?",
        "What time do you open?",
        "Tell me the price of coffee",
        "Do you delivery"
    ]

    for q in test_queries:
        print(f"Usser: {q}")
        print(f"AI: {handle_query(q)}\n")