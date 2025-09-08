# src/nlp/nlp_intent.py
import re

def extract_intent_entities(query: str) -> dict:
    """
    Extracts intent and entities from a user query.
    Handles multi-variation realistic voice queries.
    Returns:
        {
            "intent": <str>,
            "entities": {"diet": <str>, "info": <str>, "service": <str>}
        }
    """
    query = query.lower().strip()
    entities = {}

    # --- Diet / Menu entities ---
    if re.search(r"\bvegan\b", query):
        entities["diet"] = "vegan"
    elif re.search(r"\b(gluten[- ]?free|gf)\b", query):
        entities["diet"] = "gluten-free"
    elif re.search(r"\bvegetarian\b", query):
        entities["diet"] = "vegetarian"

    # --- Info / price entities ---
    if re.search(r"\b(price|cost|how much)\b", query):
        entities["info"] = "price"

    # --- Service entities ---
    if re.search(r"\b(deliver|delivery|home service|shipping)\b", query):
        entities["service"] = "delivery"

    # --- Determine intent ---
    menu_keywords = r"\b(menu|options|food|drink|vegan|gluten[- ]?free|vegetarian)\b"
    faq_keywords = r"\b(open|close|hours|location|delivery|reservation|book)\b"

    if re.search(menu_keywords, query):
        intent = "menu_query"
    elif re.search(faq_keywords, query):
        intent = "faq_query"
    else:
        intent = "other"

    return {
        "intent": intent,
        "entities": entities
    }

# --- Quick test ---
if __name__ == "__main__":
    test_queries = [
        "What are your vegan options?",
        "Do you have gluten free items?",
        "What time do you open?",
        "Tell me the price of coffee",
        "Do you deliver?",
        "Can I make a reservation?",
        "What vegetarian dishes do you have?"
    ]

    for q in test_queries:
        print(q, "->", extract_intent_entities(q))
