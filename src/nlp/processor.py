import spacy
from typing import Dict, List

_nlp = None

def load_spacy(model: str = "en_code_web_sm"):
    global _nlp
    if _nlp is None:
        try:
            _nlp = spacy.load(model)
        except OSError:
            from spacy.cli import download
            download(model)
            _nlp = spacy.load(model)
    return _nlp


def extract_entities_and_basic_features(text: str, model: str = "en_core_web_sm") -> Dict:
    nlp = load_spacy(model)
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    lemmas = [token.lemma_ for token in doc if not token.is_punct and not token.is_space]
    tokens = [token.text for token in doc if not token.is_punct and not token.is_space]
    return {"text": text, "entities": entities, "lemmas": lemmas, "tokens": tokens}

