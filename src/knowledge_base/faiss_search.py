import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Tuple

class FAQSearch:
    def __init__(self, faq_csv_path: str, embedding_model_name: str = "all-MiniLM-L6-v2"):
        self.faq_csv_path = faq_csv_path
        self.embedding_model_name = embedding_model_name
        self.faq_df = pd.read_csv(faq_csv_path)

        self.faq_df["question"] = self.faq_df["question"].astype(str)
        self.faq_df["answer"] = self.faq_df["answer"].astype(str)
        self.model = SentenceTransformer(self.embedding_model_name)
        self.index = None
        self._build_index()


    def _build_index(self):
        questions = self.faq_df["question"].tolist()
        embeddings = self.model.encode(questions, convert_to_numpy=True)
        dim = embeddings.shape[1]

        self.index = faiss.IndexFlatL2(dim)
        self.index.add(np.array(embeddings).astype("float32"))
        self.embeddings = embeddings

    def search(self, query: str, k: int = 1) -> List[Tuple[int, float]]:
        q_emb = self.model.encode([query], convert_to_numpy=True).astype("float32")
        D, I = self.index.search(q_emb, k)

        results = []
        for idx, dist in zip(I[0], D[0]):
            results.append((int(idx), float(dist)))
        return results
    
    def get_answer(self, index: int) -> str:
        return self.faq_df.iloc[index]["answer"]
    

    def get_question(self, index: int) -> str:
        return self.faq_df.iloc[index]["question"]
    
    





