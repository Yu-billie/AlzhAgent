"""agents/design.py - Design Agent: LLM 분자 설계 + RDKit 이중 검증 루프"""
import re, logging
from datetime import datetime
from agents.llm_client import LLMClient
from db.vector_store import VectorStore
from chem.mol_utils import validate_smiles, evaluate, MolReport, filter_drug_like
from config import CONFIG

logger = logging.getLogger("DesignAgent")

SYS = """You are a Drug Design Agent for Alzheimer's targets.
Design novel small molecules as SMILES strings.
Rules:
- Output ONLY valid SMILES, one per line, prefixed with "SMILES: "
- Optimize for CNS: MW 200-450, LogP 1-3, TPSA < 90
- Use heterocyclic scaffolds common in CNS drugs
- Follow Lipinski's Rule of Five"""


class DesignAgent:
    def __init__(self, llm: LLMClient, vs: VectorStore):
        self.llm = llm
        self.vs = vs

    def design(self, target: dict, n: int = None) -> dict:
        n = n or CONFIG["max_compounds"]
        name = target.get("name", "Unknown")
        logger.info(f"Design: {name}, target {n} compounds")

        refs = self.vs.search_literature(f"{name} inhibitor SMILES binding", k=3)
        ref_ctx = "\n".join(r["text"][:200] for r in refs) or "No references."

        reports: list[MolReport] = []
        for attempt in range(CONFIG["max_retries"]):
            remaining = n - len(reports)
            if remaining <= 0:
                break
            msgs = [
                {"role": "system", "content": SYS},
                {"role": "user", "content": f"""Design {remaining + 3} molecules for {name}.
Mechanism: {target.get('mechanism','Unknown')}
Strategy: {target.get('strategy','Small molecule')}

References:
{ref_ctx}

{"Already generated (avoid duplicates):" + chr(10) + chr(10).join(r.smiles for r in reports[:5]) if reports else ""}

Generate SMILES now:"""},
            ]
            resp = self.llm.chat(msgs, model_key="design", temperature=0.8)
            for smi in self._extract_smiles(resp):
                if not validate_smiles(smi):
                    continue
                rep = evaluate(smi)
                if rep.valid and rep.smiles not in [r.smiles for r in reports]:
                    reports.append(rep)
                    if len(reports) >= n:
                        break

        drug_like = filter_drug_like(reports)
        logger.info(f"Design done: {len(reports)} total, {len(drug_like)} drug-like")

        # DB 축적
        texts, metas = [], []
        for r in reports:
            texts.append(f"[Compound] Target: {name}\n{r.to_text()}")
            metas.append({
                "type": "compound", "target": name, "smiles": r.smiles,
                "qed": r.qed, "lipinski": r.lipinski, "drug_like": r in drug_like,
                "agent": "design", "at": datetime.now().isoformat(),
            })
        texts.append(f"[Design Summary] {name}: {len(reports)} generated, {len(drug_like)} drug-like")
        metas.append({"type": "design_summary", "target": name, "agent": "design", "at": datetime.now().isoformat()})
        self.vs.add_research(texts, metas)

        return {
            "target": name,
            "all": [r.to_dict() for r in reports],
            "drug_like": [r.to_dict() for r in drug_like],
            "total": len(reports),
            "total_drug_like": len(drug_like),
        }

    @staticmethod
    def _extract_smiles(text: str) -> list[str]:
        smiles = []
        for m in re.finditer(r"SMILES:\s*([^\s\n]+)", text):
            s = m.group(1).strip().rstrip(".,;)")
            if len(s) >= 5: smiles.append(s)
        for m in re.finditer(r"`([^`]{5,200})`", text):
            c = m.group(1).strip()
            if "C" in c and any(ch in c for ch in "()=#@"): smiles.append(c)
        for m in re.finditer(r"\d+[.)]\s*([A-Za-z0-9@+\-\[\]\(\)\\/=#$%:.]{5,200})", text):
            c = m.group(1).strip()
            if "C" in c and not c.isalpha(): smiles.append(c)
        return list(dict.fromkeys(smiles))  # 순서 유지 중복 제거