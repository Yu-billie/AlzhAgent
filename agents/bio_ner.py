"""agents/bio_ner.py - 바이오 도메인 NER/NEN (룰 기반, 경량)"""
import re
from dataclasses import dataclass, field

@dataclass
class BioEntity:
    text: str
    label: str       # PROTEIN, COMPOUND, DISEASE, MECHANISM
    normalized: str
    start: int = 0
    end: int = 0

PROTEIN_NORM = {
    "tau": "Tau (MAPT)", "tau protein": "Tau (MAPT)", "mapt": "Tau (MAPT)",
    "trem2": "TREM2", "trem-2": "TREM2",
    "bace1": "BACE1", "bace-1": "BACE1", "beta-secretase": "BACE1",
    "ache": "AChE", "acetylcholinesterase": "AChE",
    "gsk3β": "GSK-3β", "gsk-3beta": "GSK-3β", "gsk3b": "GSK-3β",
    "app": "APP", "apoe": "ApoE", "apoe4": "ApoE4",
    "cd33": "CD33", "nlrp3": "NLRP3",
    "presenilin": "Presenilin", "psen1": "PSEN1", "psen2": "PSEN2",
}
DISEASE_NORM = {
    "alzheimer": "Alzheimer's Disease", "alzheimer's": "Alzheimer's Disease",
    "alzheimer's disease": "Alzheimer's Disease",
    "dementia": "Dementia", "tauopathy": "Tauopathy",
    "neurodegeneration": "Neurodegeneration",
}
MECHANISM_NORM = {
    "tau aggregation": "Tau Aggregation",
    "tau phosphorylation": "Tau Hyperphosphorylation",
    "amyloid beta": "Amyloid-β Pathway", "amyloid-β": "Amyloid-β Pathway",
    "neuroinflammation": "Neuroinflammation",
    "oxidative stress": "Oxidative Stress",
    "cholinergic": "Cholinergic Pathway",
}
COMPOUND_PATTERNS = [
    r"\b[A-Z]{2,4}-\d{3,6}\b",
    r"\b(?:donepezil|galantamine|rivastigmine|memantine|aducanumab|lecanemab)\b",
]


class BioNER:
    def extract(self, text: str) -> list[BioEntity]:
        ents = []
        ents += self._match(text, PROTEIN_NORM, "PROTEIN")
        ents += self._match(text, DISEASE_NORM, "DISEASE")
        ents += self._match(text, MECHANISM_NORM, "MECHANISM")
        for pat in COMPOUND_PATTERNS:
            for m in re.finditer(pat, text, re.IGNORECASE):
                ents.append(BioEntity(m.group(), "COMPOUND", m.group().upper(), m.start(), m.end()))
        seen = set()
        unique = []
        for e in ents:
            k = (e.normalized.lower(), e.label)
            if k not in seen:
                seen.add(k)
                unique.append(e)
        return unique

    def entities_to_text(self, ents: list[BioEntity]) -> str:
        by_label: dict[str, list[str]] = {}
        for e in ents:
            by_label.setdefault(e.label, []).append(e.normalized)
        return " | ".join(f"{l}: {', '.join(set(n))}" for l, n in by_label.items())

    @staticmethod
    def _match(text: str, d: dict, label: str) -> list[BioEntity]:
        ents = []
        tl = text.lower()
        for k, v in d.items():
            for m in re.finditer(r"\b" + re.escape(k) + r"\b", tl):
                ents.append(BioEntity(text[m.start():m.end()], label, v, m.start(), m.end()))
        return ents