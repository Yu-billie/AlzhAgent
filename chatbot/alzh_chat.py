"""chatbot/alzh_chat.py - AlzhChat: 듀얼 RAG 챗봇 (Research DB + Literature DB)"""
import logging
from agents.llm_client import LLMClient
from agents.bio_ner import BioNER
from db.vector_store import VectorStore

logger = logging.getLogger("AlzhChat")

SYS = """You are AlzhChat, an AI research assistant for Alzheimer's drug discovery.

You have access to two knowledge bases:
- Research DB: AI agent's analysis results (targets, compounds, validation reports)
- Literature DB: Original scientific papers (PubMed abstracts, BindingDB data)

Rules:
- Cite sources: [Research DB] for agent results, [Literature] for original papers
- Include metrics (QED, LogP, MW) when discussing compounds
- Be honest about uncertainty
- Suggest 1-2 follow-up questions
- Match the user's language (Korean or English)"""


class AlzhChat:
    def __init__(self, api_key: str):
        self.llm = LLMClient(api_key)
        self.vs = VectorStore.get()
        self.ner = BioNER()

    def answer(self, question: str, history: list[dict] | None = None, model_key: str = "chatbot") -> dict:
        """듀얼 RAG 답변 생성"""
        ents = self.ner.extract(question)
        boost = " ".join(e.normalized for e in ents[:3])
        sq = f"{question} {boost}".strip()

        results = self.vs.search_all(sq, k=5)
        r_hits = results["research"]
        l_hits = results["literature"]

        msgs = [{"role": "system", "content": SYS}]
        if history:
            msgs.extend(history[-10:])

        msgs.append({"role": "user", "content": f"""Question: {question}

Entities: {', '.join(e.normalized for e in ents) or 'None'}

=== RESEARCH DB (Agent results) ===
{self._fmt(r_hits, "Research")}

=== LITERATURE DB (Papers) ===
{self._fmt(l_hits, "Literature")}

=== DB Stats ===
{self.vs.stats()}

Answer with citations. Suggest follow-up questions."""})

        answer = self.llm.chat(msgs, model_key=model_key, temperature=0.5)

        sources = []
        for h in r_hits:
            m = h.get("metadata", {})
            sources.append(f"[Research] {m.get('type','')} {m.get('target','')}")
        for h in l_hits:
            m = h.get("metadata", {})
            s = "[Literature]"
            if m.get("pmid"): s += f" PMID:{m['pmid']}"
            if m.get("title"): s += f" {m['title'][:60]}"
            sources.append(s)

        return {
            "answer": answer,
            "sources": sources,
            "entities": [{"text": e.text, "label": e.label, "normalized": e.normalized} for e in ents],
            "stats": {"research_hits": len(r_hits), "literature_hits": len(l_hits)},
        }

    def _fmt(self, hits: list[dict], label: str) -> str:
        if not hits: return "No results."
        parts = []
        for i, h in enumerate(hits):
            m = h.get("metadata", {})
            d = h.get("distance")
            rel = f" (rel:{1-d:.2f})" if d is not None else ""
            hdr = f"[{label} #{i+1}]{rel}"
            if m.get("type"): hdr += f" type={m['type']}"
            if m.get("target"): hdr += f" target={m['target']}"
            if m.get("pmid"): hdr += f" PMID={m['pmid']}"
            parts.append(f"{hdr}\n{h['text'][:400]}")
        return "\n\n".join(parts)