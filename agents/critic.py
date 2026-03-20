"""agents/critic.py - Critic Agent: 검증 + 인과 추론 + 보안 + 리포트"""
import re, logging
from datetime import datetime
from agents.llm_client import LLMClient
from db.vector_store import VectorStore

logger = logging.getLogger("CriticAgent")

SYS = """You are a Critic Agent for Alzheimer's drug discovery.
Validate research results with scientific rigor.
1. VALIDATE targets (evidence strength, confidence 0-100%)
2. ASSESS compounds (drug-likeness, novelty, issues)
3. CAUSAL ANALYSIS: Drug → Target → Mechanism → Effect → Side Effects
4. RISK ASSESSMENT
5. RECOMMENDATION: Go / No-Go / Conditional
6. NEXT STEPS"""

INJECTION_PATTERNS = [
    r"ignore\s+(previous|above|all)\s+(instructions?|prompts?)",
    r"you\s+are\s+now\s+",
    r"disregard\s+(?:your|the|all)\s+",
    r"new\s+instructions?:\s*",
    r"forget\s+(?:everything|all|your)",
    r"\[INST\]|\[/INST\]|<<SYS>>",
    r"<\|(?:im_start|im_end|system|user)\|>",
]


class CriticAgent:
    def __init__(self, llm: LLMClient, vs: VectorStore):
        self.llm = llm
        self.vs = vs

    def validate(self, lit_result: dict, design_result: dict) -> dict:
        logger.info("Critic: validating...")

        sec = self._security_scan(str(lit_result) + str(design_result))

        compounds = design_result.get("drug_like") or design_result.get("all", [])
        cpd_summary = self._cpd_summary(compounds)
        targets = lit_result.get("targets", [])
        tgt_text = "\n".join(f"- {t.get('name','?')}: {t.get('mechanism','?')} ({t.get('evidence','?')})" for t in targets) or "None"

        msgs = [
            {"role": "system", "content": SYS},
            {"role": "user", "content": f"""== TARGETS ==
{tgt_text}

== COMPOUNDS ==
{cpd_summary}

== LITERATURE ANALYSIS (excerpt) ==
{lit_result.get('analysis','N/A')[:2500]}

== SECURITY ==
{f"WARNINGS: {sec}" if sec else "Clean."}

Generate FINAL REPORT."""},
        ]
        report = self.llm.chat(msgs, model_key="critic", temperature=0.3)

        conf = self._get_confidence(report)

        self.vs.add_research(
            [f"[Critic Report]\n{report}", f"[Confidence] {conf}"],
            [{"type": "critic_report", "agent": "critic", "at": datetime.now().isoformat()},
             {"type": "confidence", "agent": "critic", "at": datetime.now().isoformat()}],
        )
        return {"report": report, "confidence": conf, "security": sec}

    def _security_scan(self, text: str) -> list[str]:
        tl = text.lower()
        return [p for p in INJECTION_PATTERNS if re.search(p, tl)]

    def _cpd_summary(self, cpds: list[dict]) -> str:
        if not cpds: return "No compounds."
        s = sorted(cpds, key=lambda c: c.get("qed", 0), reverse=True)
        lines = []
        for i, c in enumerate(s[:8]):
            lines.append(f"#{i+1} {c.get('smiles','?')[:60]}\n   QED={c.get('qed',0):.3f} LogP={c.get('logp',0):.2f} MW={c.get('mw',0):.0f} Lip={'Y' if c.get('lipinski') else 'N'}")
        return f"Total: {len(cpds)}\n" + "\n".join(lines)

    def _get_confidence(self, report: str) -> dict:
        msgs = [
            {"role": "system", "content": "Extract scores as JSON only."},
            {"role": "user", "content": f"""From this report extract: {{"overall":0-100,"target":0-100,"compound":0-100,"recommendation":"Go/No-Go/Conditional"}}
{report[:1500]}"""},
        ]
        r = self.llm.chat_json(msgs, model_key="critic", temperature=0.1)
        if "raw" in r:
            return {"overall": 50, "recommendation": "Conditional"}
        return r