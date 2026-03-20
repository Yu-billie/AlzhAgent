"""agents/literature.py - Literature Agent: RAG + NER → 타깃 가설 수립"""
import logging
from datetime import datetime
from agents.llm_client import LLMClient
from agents.bio_ner import BioNER
from db.vector_store import VectorStore

logger = logging.getLogger("LitAgent")

SYS = """You are a Literature Agent for Alzheimer's drug discovery.
Analyze retrieved literature to identify promising drug targets.
Focus on: Tau aggregation inhibitors, TREM2 agonists, BACE1 inhibitors, neuroinflammation modulators.
Cite specific findings. Respond in structured format."""


class LiteratureAgent:
    def __init__(self, llm: LLMClient, vs: VectorStore):
        self.llm = llm
        self.vs = vs
        self.ner = BioNER()

    def analyze(self, query: str) -> dict:
        logger.info(f"Literature: {query}")
        hits = self.vs.search_literature(query, k=8)
        ctx = "\n\n---\n\n".join(
            f"[{h.get('metadata',{}).get('source','?')}] "
            f"{h.get('metadata',{}).get('title','')}\n{h['text']}"
            for h in hits
        )
        all_text = " ".join(h["text"] for h in hits)
        ents = self.ner.extract(all_text)
        ent_text = self.ner.entities_to_text(ents)

        msgs = [
            {"role": "system", "content": SYS},
            {"role": "user", "content": f"""Research Query: {query}

Retrieved Literature:
{ctx[:6000]}

Extracted Entities: {ent_text}

Provide:
1. TOP 3 drug targets with evidence summary
2. Mechanism, evidence strength (High/Medium/Low) for each
3. Proposed hypothesis for the best target
4. Compound design strategy"""},
        ]
        analysis = self.llm.chat(msgs, model_key="literature", temperature=0.4)

        targets = self._extract_targets(analysis, ents)

        # DB 축적
        res_texts, res_metas = [], []
        res_texts.append(f"[Literature Analysis] Query: {query}\n\n{analysis}")
        res_metas.append({"type": "lit_analysis", "query": query, "agent": "literature", "at": datetime.now().isoformat()})
        for t in targets:
            res_texts.append(f"[Drug Target] {t['name']}\nMechanism: {t.get('mechanism','N/A')}\nEvidence: {t.get('evidence','N/A')}")
            res_metas.append({"type": "drug_target", "target": t["name"], "agent": "literature", "at": datetime.now().isoformat()})
        self.vs.add_research(res_texts, res_metas)

        return {
            "analysis": analysis,
            "entities": [{"text": e.text, "label": e.label, "normalized": e.normalized} for e in ents],
            "targets": targets,
            "num_sources": len(hits),
        }

    def _extract_targets(self, analysis: str, ents) -> list[dict]:
        msgs = [
            {"role": "system", "content": "Extract drug targets as JSON array. No markdown."},
            {"role": "user", "content": f"""Extract top targets as JSON: [{{"name":"...","mechanism":"...","evidence":"High/Medium/Low","strategy":"..."}}]

Analysis:
{analysis[:3000]}"""},
        ]
        r = self.llm.chat_json(msgs, model_key="literature", temperature=0.2)
        if isinstance(r, dict) and "items" in r:
            return r["items"][:5]
        if isinstance(r, list):
            return r[:5]
        # 폴백
        prots = [e for e in ents if e.label == "PROTEIN"]
        return [{"name": e.normalized, "mechanism": "See analysis", "evidence": "Medium", "strategy": "TBD"} for e in prots[:3]]