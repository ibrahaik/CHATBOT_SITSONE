from typing import Any, Dict, List, Optional
import os

class PineconeGateway:
    def __init__(self, index_name: str):
        api_key = (os.getenv("PINECONE_API_KEY") or "").strip()
        if not api_key:
            raise RuntimeError("Falta PINECONE_API_KEY en el entorno.")

        from pinecone import Pinecone
        pc = Pinecone(api_key=api_key)
        self.index = pc.Index(index_name)

    def query(
        self,
        namespace: str,
        embedding: List[float],
        top_k: int,
        flt: Optional[Dict[str, Any]] = None,
        include_metadata: bool = True,
    ) -> List[Dict[str, Any]]:
        res = self.index.query(
            namespace=namespace,
            vector=embedding,
            top_k=top_k,
            include_metadata=include_metadata,
            filter=flt or None,
        )
        matches = res.get("matches") or []
        out: List[Dict[str, Any]] = []
        for m in matches:
            out.append({
                "id": m.get("id"),
                "score": float(m.get("score") or 0.0),
                "metadata": m.get("metadata") or {},
            })
        return out

    def fetch(self, namespace: str, ids: List[str]) -> Dict[str, Any]:
        return self.index.fetch(namespace=namespace, ids=ids)
