from typing import List
from openai import OpenAI

class Embedder:
    def __init__(self, client: OpenAI, model: str):
        self.client = client
        self.model = model

    def embed_one(self, text: str) -> List[float]:
        t = (text or "").strip()
        if not t:
            return []
        resp = self.client.embeddings.create(model=self.model, input=t)
        return resp.data[0].embedding
