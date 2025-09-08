# src/nlp/response_generator.py
import json
import os
import math

DATA_PATH = os.path.join(os.path.dirname(__file__), "../../data/sample_data.json")

# Load sample data (menu, FAQs, etc.)
with open(DATA_PATH, "r", encoding="utf-8") as f:
    knowledge_base = json.load(f)

def generate_response(nlp_result: dict, faq_responses: list = None, knowledge_base: dict = None) -> str:
    """
    Generate a response based on NLP results and optional FAQ matches.
    - nlp_result: dict with 'intent' and 'entities'
    - faq_responses: list of FAQ strings
    - knowledge_base: dictionary with menu, faq, etc.
    """

    if knowledge_base is None:
        raise ValueError("knowledge_base must be provided")

    intent = nlp_result.get("intent")
    entities = nlp_result.get("entities", {})

    # 1) Main response
    main_response = ""
    if intent == "menu_query":
        diet = entities.get("diet")
        if diet:
            items = knowledge_base["menu"].get(diet, [])
            if items:
                main_response = f"{diet.capitalize()} options: {', '.join(items)}."
            else:
                main_response = f"Sorry, we don’t currently have {diet} options."
        else:
            main_response = f"Our menu includes: {', '.join(knowledge_base['menu']['all'])}."
    elif intent == "faq_query":
        service = entities.get("service")
        if service == "delivery":
            main_response = knowledge_base["faq"].get("delivery", "We offer delivery options. Please check our website.")
        else:
            main_response = knowledge_base["faq"].get("hours", "We are open from 8 AM to 9 PM Monday-Saturday, 9 AM to 6 PM Sunday.")
    elif intent == "other":
        info = entities.get("info")
        if info == "price":
            main_response = "The price of coffee is $3.50."
        else:
            main_response = "Could you please clarify your request?"
    else:
        main_response = "Sorry, I didn’t understand that."

    # 2) Filter FAQ responses
    filtered_faqs = []
    if faq_responses:
        for f in faq_responses:
            if f and f.strip() != "" and not (isinstance(f, float) and math.isnan(f)) and str(f).lower() != "nan":
                filtered_faqs.append(f)

    # 3) Build final response with top 2 FAQs
    faq_text = ""
    max_faq = 2
    if filtered_faqs:
        faq_text = "\nYou might also find these helpful:\n"
        faq_text += "\n".join([f"{i+1}. {f}" for i, f in enumerate(filtered_faqs[:max_faq])])

    final_response = f"{main_response}{faq_text}"

    return final_response


# Quick test
if __name__ == "__main__":
    test_inputs = [
        {"intent": "menu_query", "entities": {"diet": "vegan"}},
        {"intent": "faq_query", "entities": {"service": "delivery"}},
        {"intent": "other", "entities": {"info": "price"}},
    ]

    for t in test_inputs:
        print(t, "->", generate_response(t, faq_responses=["salads: and plant-based smoothies.", "nan", None], knowledge_base=knowledge_base))
