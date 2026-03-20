"""agents/supervisor.py - Supervisor: ToT 계획 수립 + 에이전트 오케스트레이션"""
import json, logging
from datetime import datetime
from agents.llm_client import LLMClient
from agents.literature import LiteratureAgent
from agents.design import DesignAgent
from agents.critic import CriticAgent
from db.vector_store import VectorStore
from config import CONFIG

logger = logging.getLogger("Supervisor")

PLAN_SYS = """You are a Supervisor Agent for Alzheimer's drug discovery.
Create a research plan using Tree of Thoughts reasoning.
Output JSON:
{
  "query_analysis": "...",
  "research_paths": [
    {"path_id":1,"target_focus":"...","rationale":"...","search_queries":["..."],"design_strategy":"...","priority":"high/medium/low"}
  ],
  "evaluation_criteria": ["..."]
}"""

DEFAULT_PLAN = {
    "query_analysis": "Alzheimer's drug discovery",
    "research_paths": [
        {"path_id": 1, "target_focus": "Tau aggregation", "rationale": "Primary AD pathology",
         "search_queries": ["Tau aggregation inhibitor Alzheimer small molecule", "Tau fibril structure drug design"],
         "design_strategy": "Structure-based design targeting Tau fibril", "priority": "high"},
        {"path_id": 2, "target_focus": "TREM2", "rationale": "Neuroinflammation via innate immunity",
         "search_queries": ["TREM2 agonist small molecule Alzheimer", "TREM2 mechanism neuroinflammation"],
         "design_strategy": "Fragment-based agonist design", "priority": "high"},
    ],
    "evaluation_criteria": ["QED > 0.5", "BBB: LogP 1-3, TPSA < 90", "SA < 5"],
}


class SupervisorAgent:
    def __init__(self, api_key: str):
        self.llm = LLMClient(api_key)
        self.vs = VectorStore.get()
        self.lit = LiteratureAgent(self.llm, self.vs)
        self.design = DesignAgent(self.llm, self.vs)
        self.critic = CriticAgent(self.llm, self.vs)

    def run(self, query: str, progress_cb=None) -> dict:
        """전체 파이프라인 실행. progress_cb(step, total, message)"""
        t0 = datetime.now()

        def _p(step, msg):
            logger.info(f"[{step}/4] {msg}")
            if progress_cb:
                progress_cb(step, 4, msg)

        # Step 1: Plan
        _p(1, "연구 계획 수립 (Tree of Thoughts)...")
        plan = self._plan(query)
        self.vs.add_research(
            [f"[Plan] {query}\n{json.dumps(plan, ensure_ascii=False, indent=2)}"],
            [{"type": "plan", "query": query, "agent": "supervisor", "at": datetime.now().isoformat()}],
        )

        # Step 2: Literature
        _p(2, "문헌 분석 및 타깃 발굴...")
        paths = plan.get("research_paths", [{"search_queries": [query]}])
        high = [p for p in paths if p.get("priority") == "high"] or paths[:2]

        lit_results = []
        all_targets = []
        for path in high:
            for q in path.get("search_queries", [query])[:2]:
                r = self.lit.analyze(q)
                lit_results.append(r)
                all_targets.extend(r.get("targets", []))

        targets = self._merge_targets(all_targets)
        merged_lit = {
            "analysis": "\n\n---\n\n".join(r.get("analysis", "") for r in lit_results),
            "targets": targets,
            "entities": sum((r.get("entities", []) for r in lit_results), []),
        }

        # Step 3: Design
        _p(3, f"후보물질 설계 ({len(targets[:2])}개 타깃)...")
        design_results = []
        for t in targets[:2]:
            r = self.design.design(t, CONFIG["max_compounds"])
            design_results.append(r)

        merged_design = {
            "all": sum((r.get("all", []) for r in design_results), []),
            "drug_like": sum((r.get("drug_like", []) for r in design_results), []),
            "total": sum(r.get("total", 0) for r in design_results),
            "total_drug_like": sum(r.get("total_drug_like", 0) for r in design_results),
        }

        # Step 4: Critic
        _p(4, "검증 및 리포트 생성...")
        critic_result = self.critic.validate(merged_lit, merged_design)

        elapsed = (datetime.now() - t0).total_seconds()

        result = {
            "query": query,
            "plan": plan,
            "literature": {"targets": targets, "num_entities": len(merged_lit["entities"])},
            "design": {
                "total": merged_design["total"],
                "total_drug_like": merged_design["total_drug_like"],
                "top_compounds": sorted(merged_design["drug_like"], key=lambda c: c.get("qed", 0), reverse=True)[:5],
            },
            "critic": critic_result,
            "elapsed": round(elapsed, 1),
            "db_stats": self.vs.stats(),
        }

        self.vs.add_research(
            [f"[Summary] {query} | Targets: {len(targets)} | Compounds: {merged_design['total']} ({merged_design['total_drug_like']} drug-like) | {elapsed:.0f}s"],
            [{"type": "summary", "query": query, "agent": "supervisor", "at": datetime.now().isoformat()}],
        )
        return result

    def _plan(self, query: str) -> dict:
        msgs = [
            {"role": "system", "content": PLAN_SYS},
            {"role": "user", "content": f"Research Query: {query}"},
        ]
        p = self.llm.chat_json(msgs, model_key="supervisor", temperature=0.4)
        if "raw" in p or not p.get("research_paths"):
            return {**DEFAULT_PLAN, "query_analysis": f"Research: {query}"}
        return p

    @staticmethod
    def _merge_targets(targets: list[dict]) -> list[dict]:
        seen = {}
        for t in targets:
            k = t.get("name", "?").lower().replace(" ", "")
            if k not in seen:
                seen[k] = t
            else:
                for f in ("mechanism", "evidence", "strategy"):
                    if t.get(f) and not seen[k].get(f):
                        seen[k][f] = t[f]
        return list(seen.values())