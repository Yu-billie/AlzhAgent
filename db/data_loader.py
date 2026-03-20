"""db/data_loader.py - PubMed 논문 초록 수집 및 Literature DB 적재"""
import logging
from datetime import datetime
from Bio import Entrez, Medline
from db.vector_store import VectorStore
from config import CONFIG

logger = logging.getLogger("DataLoader")
Entrez.email = "alzh.agent@research.ai"


def load_pubmed(vs: VectorStore) -> int:
    """PubMed에서 알츠하이머 관련 논문 초록 수집 → Literature DB"""
    q = CONFIG["pubmed_query"]
    mx = CONFIG["pubmed_max_results"]
    logger.info(f"PubMed 검색: '{q}' (max {mx})")

    try:
        h = Entrez.esearch(db="pubmed", term=q, retmax=mx, sort="relevance")
        ids = Entrez.read(h).get("IdList", [])
        h.close()
        if not ids:
            logger.warning("PubMed 결과 없음")
            return 0

        h = Entrez.efetch(db="pubmed", id=ids, rettype="medline", retmode="text")
        recs = list(Medline.parse(h))
        h.close()

        texts, metas = [], []
        for rec in recs:
            title = rec.get("TI", "")
            abstract = rec.get("AB", "")
            if not abstract:
                continue
            chunk = f"Title: {title}\n\nAbstract: {abstract}"
            texts.append(chunk)
            metas.append({
                "source": "pubmed",
                "pmid": rec.get("PMID", ""),
                "title": title[:200],
                "authors": ", ".join(rec.get("AU", [])[:3]),
                "date": rec.get("DP", ""),
                "at": datetime.now().isoformat(),
            })

        added = vs.add_literature(texts, metas)
        logger.info(f"PubMed: {added}개 청크 추가")
        return added
    except Exception as e:
        logger.error(f"PubMed 실패: {e}")
        return 0


def load_seed_targets(vs: VectorStore) -> int:
    """알츠하이머 핵심 타깃 시드 데이터"""
    seeds = [
        ("Tau (MAPT)", "Tau aggregation is a hallmark of Alzheimer's. Tau fibrils (PDB 5O3L) are targets for small molecule inhibitors. Tau hyperphosphorylation leads to neurofibrillary tangles."),
        ("TREM2", "TREM2 is a receptor on microglia involved in neuroinflammation. TREM2 agonists like VG-3927 enhance microglial clearance of amyloid. Loss-of-function TREM2 variants increase AD risk."),
        ("BACE1", "BACE1 (Beta-secretase 1) cleaves APP to produce amyloid-beta. BACE1 inhibitors reduce amyloid plaque formation. Past clinical failures due to off-target effects."),
        ("AChE", "Acetylcholinesterase inhibitors (donepezil, galantamine, rivastigmine) are approved symptomatic treatments. They increase acetylcholine levels in cholinergic synapses."),
        ("GSK-3β", "GSK-3β phosphorylates tau protein. Inhibition reduces tau hyperphosphorylation. Dual-target strategies combining GSK-3β and tau aggregation inhibition are being explored."),
    ]
    texts, metas = [], []
    for name, desc in seeds:
        texts.append(f"Target: {name}\n{desc}\nSource: BindingDB / Literature Summary")
        metas.append({"source": "seed", "target": name, "type": "target_summary", "at": datetime.now().isoformat()})
    return vs.add_literature(texts, metas)


def load_if_empty(vs: VectorStore) -> dict:
    """DB가 비어있으면 초기 데이터 로딩"""
    s = vs.stats()
    if s["literature"] > 0:
        return {"status": "already_loaded", **s}
    n1 = load_seed_targets(vs)
    n2 = load_pubmed(vs)
    return {"status": "loaded", "seeds": n1, "pubmed": n2, **vs.stats()}