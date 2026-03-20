"""chem/mol_utils.py - 화학 유틸리티 (RDKit 있으면 정밀, 없으면 휴리스틱 폴백)"""
import re, logging
from dataclasses import dataclass, field

logger = logging.getLogger("Chem")

try:
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Descriptors, QED, FilterCatalog
    from rdkit.Chem.FilterCatalog import FilterCatalogParams
    RDLogger.DisableLog("rdApp.*")
    HAS_RDKIT = True
    logger.info("RDKit 로드 완료 — 정밀 평가 모드")
except ImportError:
    HAS_RDKIT = False
    logger.warning("RDKit 미설치 — 휴리스틱 폴백 모드 (Python 3.12 이하에서 pip install rdkit-pypi)")


@dataclass
class MolReport:
    smiles: str
    valid: bool
    qed: float = 0.0
    logp: float = 0.0
    mw: float = 0.0
    hbd: int = 0
    hba: int = 0
    tpsa: float = 0.0
    sa_score: float = 0.0
    lipinski: bool = False
    tox_alerts: list = field(default_factory=list)
    error: str = ""
    mode: str = "rdkit"  # "rdkit" or "heuristic"

    def to_text(self) -> str:
        if not self.valid:
            return f"INVALID: {self.smiles} ({self.error})"
        tag = "[heuristic]" if self.mode == "heuristic" else ""
        return (f"SMILES: {self.smiles} {tag}\n"
                f"QED: {self.qed:.3f} | LogP: {self.logp:.2f} | MW: {self.mw:.1f}\n"
                f"HBD: {self.hbd} | HBA: {self.hba} | TPSA: {self.tpsa:.1f}\n"
                f"Lipinski: {'PASS' if self.lipinski else 'FAIL'} | Tox: {len(self.tox_alerts)}")

    def to_dict(self) -> dict:
        return {
            "smiles": self.smiles, "valid": self.valid, "qed": round(self.qed, 3),
            "logp": round(self.logp, 2), "mw": round(self.mw, 1),
            "hbd": self.hbd, "hba": self.hba, "tpsa": round(self.tpsa, 1),
            "sa_score": round(self.sa_score, 2), "lipinski": self.lipinski,
            "tox_alerts": self.tox_alerts, "error": self.error, "mode": self.mode,
        }


# ══════════════════════════════════════════
# SMILES 유효성 검증
# ══════════════════════════════════════════
def validate_smiles(smiles: str) -> bool:
    if HAS_RDKIT:
        try:
            return Chem.MolFromSmiles(smiles) is not None
        except Exception:
            return False
    else:
        return _heuristic_validate(smiles)


def _heuristic_validate(smiles: str) -> bool:
    """RDKit 없을 때 SMILES 문자열 기반 기본 검증"""
    if not smiles or len(smiles) < 3:
        return False
    # 최소 탄소 원자 포함
    if "C" not in smiles and "c" not in smiles:
        return False
    # 괄호 짝 맞는지
    if smiles.count("(") != smiles.count(")"):
        return False
    if smiles.count("[") != smiles.count("]"):
        return False
    # 허용 문자만 포함하는지
    allowed = set("ABCDEFGHIKLMNOPRSTUVWYZabcdefghiklmnoprstuvwyz0123456789@+-.=#\\/()[]%:$")
    if not all(c in allowed for c in smiles):
        return False
    # 비정상적으로 긴/짧은 것 필터
    if len(smiles) > 300:
        return False
    # 일반 영어 단어 필터 (SMILES가 아닌 텍스트 걸러내기)
    if smiles.isalpha() and smiles.islower():
        return False
    return True


# ══════════════════════════════════════════
# 분자 평가
# ══════════════════════════════════════════
def evaluate(smiles: str) -> MolReport:
    if HAS_RDKIT:
        return _evaluate_rdkit(smiles)
    else:
        return _evaluate_heuristic(smiles)


def _evaluate_rdkit(smiles: str) -> MolReport:
    """RDKit 정밀 평가"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return MolReport(smiles=smiles, valid=False, error="Invalid SMILES")

        qed_v = QED.qed(mol)
        logp = Descriptors.MolLogP(mol)
        mw = Descriptors.MolWt(mol)
        hbd = Descriptors.NumHDonors(mol)
        hba = Descriptors.NumHAcceptors(mol)
        tpsa = Descriptors.TPSA(mol)
        lip = (mw <= 500 and logp <= 5 and hbd <= 5 and hba <= 10)
        sa = min(10.0, 1.0 + mol.GetNumHeavyAtoms() * 0.1 + mol.GetRingInfo().NumRings() * 0.3)
        tox = _pains_rdkit(mol)

        return MolReport(
            smiles=Chem.MolToSmiles(mol), valid=True, qed=qed_v, logp=logp,
            mw=mw, hbd=hbd, hba=hba, tpsa=tpsa, sa_score=sa, lipinski=lip,
            tox_alerts=tox, mode="rdkit",
        )
    except Exception as e:
        return MolReport(smiles=smiles, valid=False, error=str(e))


def _evaluate_heuristic(smiles: str) -> MolReport:
    """RDKit 없을 때 SMILES 문자열 기반 휴리스틱 평가"""
    if not _heuristic_validate(smiles):
        return MolReport(smiles=smiles, valid=False, error="Heuristic validation failed", mode="heuristic")

    # 원자 개수 추정 (대문자 = 무거운 원자)
    heavy_atoms = len(re.findall(r"[A-Z]", smiles))
    # 고리 추정 (숫자 쌍)
    ring_digits = re.findall(r"\d", smiles)
    est_rings = len(ring_digits) // 2

    # MW 추정: 평균 무거운 원자 질량 ~12-14
    est_mw = heavy_atoms * 13.5 + est_rings * 2
    # LogP 추정: 탄소 비율 기반
    c_count = smiles.count("C") + smiles.count("c")
    n_count = smiles.count("N") + smiles.count("n")
    o_count = smiles.count("O") + smiles.count("o")
    polar = n_count + o_count
    est_logp = (c_count * 0.5 - polar * 0.8)
    est_logp = max(-2, min(8, est_logp))

    # HBD: OH, NH 패턴
    est_hbd = len(re.findall(r"[ON]H?\b", smiles))
    est_hbd = min(est_hbd, smiles.count("O") + smiles.count("N"))
    # HBA: N, O 개수
    est_hba = n_count + o_count
    # TPSA 추정
    est_tpsa = polar * 20.0
    # QED 추정 (단순 복합 스코어)
    qed_est = 0.0
    if 200 < est_mw < 500: qed_est += 0.25
    if 0 < est_logp < 5: qed_est += 0.25
    if est_hbd <= 5: qed_est += 0.15
    if est_hba <= 10: qed_est += 0.15
    if est_rings >= 1: qed_est += 0.1
    if len(smiles) > 10: qed_est += 0.1
    qed_est = min(1.0, qed_est)

    # Lipinski
    lip = (est_mw <= 500 and est_logp <= 5 and est_hbd <= 5 and est_hba <= 10)

    # SA 추정
    sa_est = min(10.0, 1.0 + heavy_atoms * 0.1 + est_rings * 0.3)

    # 독성 패턴 (문자열 기반)
    tox = []
    if "[N+](=O)[O-]" in smiles: tox.append("Nitro")
    if "C1OC1" in smiles: tox.append("Epoxide")
    if smiles.endswith("=O") and smiles[-3:-2] not in ("C", "S"): tox.append("Aldehyde (possible)")

    return MolReport(
        smiles=smiles, valid=True, qed=qed_est, logp=round(est_logp, 2),
        mw=round(est_mw, 1), hbd=est_hbd, hba=est_hba, tpsa=round(est_tpsa, 1),
        sa_score=round(sa_est, 2), lipinski=lip, tox_alerts=tox, mode="heuristic",
    )


def _pains_rdkit(mol) -> list[str]:
    """RDKit PAINS 필터"""
    alerts = []
    try:
        p = FilterCatalogParams()
        p.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
        cat = FilterCatalog.FilterCatalog(p)
        entry = cat.GetFirstMatch(mol)
        if entry:
            alerts.append(f"PAINS: {entry.GetDescription()}")
    except Exception:
        pass
    for name, smarts in [("Nitro", "[N+](=O)[O-]c"), ("Epoxide", "C1OC1"), ("Aldehyde", "[CH]=O")]:
        try:
            pat = Chem.MolFromSmarts(smarts)
            if pat and mol.HasSubstructMatch(pat):
                alerts.append(name)
        except Exception:
            pass
    return alerts


def filter_drug_like(reports: list[MolReport]) -> list[MolReport]:
    return [r for r in reports if r.valid and r.qed >= 0.4 and r.lipinski and len(r.tox_alerts) <= 2]