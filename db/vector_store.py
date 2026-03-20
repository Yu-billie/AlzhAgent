"""db/vector_store.py - ChromaDB 듀얼 컬렉션 (literature + research)"""
import hashlib, logging
from datetime import datetime
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from config import CONFIG

logger = logging.getLogger("VectorStore")


class VectorStore:
    _instance = None

    @classmethod
    def get(cls) -> "VectorStore":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self.client = chromadb.PersistentClient(
            path=CONFIG["chroma_persist_dir"],
            settings=Settings(anonymized_telemetry=False),
        )
        self.embedder = SentenceTransformer(CONFIG["embedding_model"])
        self.lit = self.client.get_or_create_collection(name=CONFIG["literature_collection"])
        self.res = self.client.get_or_create_collection(name=CONFIG["research_collection"])
        logger.info(f"VectorStore: lit={self.lit.count()}, res={self.res.count()}")

    def _embed(self, texts: list[str]) -> list[list[float]]:
        return self.embedder.encode(texts, show_progress_bar=False).tolist()

    def _id(self, t: str) -> str:
        return hashlib.md5(t.encode()).hexdigest()

    # ── Literature DB ──
    def add_literature(self, texts: list[str], metas: list[dict] | None = None) -> int:
        if not texts: return 0
        ids = [self._id(t) for t in texts]
        embs = self._embed(texts)
        metas = metas or [{"source": "unknown", "at": datetime.now().isoformat()}] * len(texts)
        self.lit.upsert(ids=ids, embeddings=embs, documents=texts, metadatas=metas)
        return len(texts)

    def search_literature(self, q: str, k: int = 5) -> list[dict]:
        if self.lit.count() == 0: return []
        r = self.lit.query(query_embeddings=self._embed([q]), n_results=min(k, self.lit.count()))
        return self._fmt(r)

    # ── Research DB ──
    def add_research(self, texts: list[str], metas: list[dict] | None = None) -> int:
        if not texts: return 0
        ids = [self._id(t) for t in texts]
        embs = self._embed(texts)
        metas = metas or [{"type": "research", "at": datetime.now().isoformat()}] * len(texts)
        self.res.upsert(ids=ids, embeddings=embs, documents=texts, metadatas=metas)
        return len(texts)

    def search_research(self, q: str, k: int = 5) -> list[dict]:
        if self.res.count() == 0: return []
        r = self.res.query(query_embeddings=self._embed([q]), n_results=min(k, self.res.count()))
        return self._fmt(r)

    # ── 듀얼 검색 ──
    def search_all(self, q: str, k: int = 5) -> dict:
        return {"literature": self.search_literature(q, k), "research": self.search_research(q, k)}

    def stats(self) -> dict:
        return {"literature": self.lit.count(), "research": self.res.count()}

    @staticmethod
    def _fmt(r: dict) -> list[dict]:
        out = []
        if not r or not r.get("documents"): return out
        for i, doc in enumerate(r["documents"][0]):
            out.append({
                "text": doc,
                "metadata": r["metadatas"][0][i] if r.get("metadatas") else {},
                "distance": r["distances"][0][i] if r.get("distances") else None,
            })
        return out