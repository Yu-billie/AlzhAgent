# AlzhAgent: LLM 멀티 에이전트 기반 알츠하이머 신약 리서치 자동화 + 대화형 연구 챗봇

> **Autonomous Multi-Agent Research Pipeline with Persistent Knowledge DB & Interactive RAG Chatbot for Alzheimer's Drug Discovery**

---

## 1. 프로젝트 개요

### 1.1 기술명

**AlzhAgent** — LLM 멀티 에이전트가 자율적으로 알츠하이머 신약 타깃을 리서치하고, 축적된 연구 DB를 제약 연구원이 자연어로 탐색·대화할 수 있는 통합 플랫폼

### 1.2 개발 목표

본 프로젝트는 **두 개의 연결된 시스템**을 구축한다:

**① 자율 리서치 에이전트 (AlzhAgent Pipeline)**
- PubMed/BindingDB 등 바이오 문헌을 자동 수집·분석하여 타깃 선정 → 후보물질 설계 → 약물성 평가까지 수행
- 리서치 결과가 **지속적으로 Knowledge DB에 축적**됨

**② 연구원용 RAG 챗봇 (AlzhChat)**
- 축적된 Knowledge DB + 원본 문헌 DB를 실시간 참조하여 제약 연구원과 자연어 대화
- 연구원이 에이전트의 리서치 결과를 질의·검증·확장할 수 있는 인터페이스

**핵심 가치:**
- 에이전트가 리서치할수록 DB가 풍부해지고, 챗봇 응답 품질도 함께 향상되는 **자기강화 루프(Self-Reinforcing Loop)**
- 비전공 연구원도 자연어로 복잡한 화합물·타깃 분석에 접근 가능
- **GPU 없이 OpenRouter 무료 API만으로 전체 파이프라인 운영**

### 1.3 팀 구성

| 역할 | 담당 | 핵심 역량 |
|---|---|---|
| 1인 개발 | 김유민 | NLP/RAG 시스템 3년, EMNLP·COLING 등 6편 발표, LLM 보안(Indirect Prompt Injection 방어) 연구 |

### 1.4 컴퓨팅 자원

| 자원 | 상세 |
|---|---|
| **LLM 추론** | OpenRouter 무료 API (DeepSeek-R1, Llama-3-70B, Qwen-2.5-72B 등 무료 모델 활용) |
| **임베딩** | 로컬 Sentence Transformers (all-MiniLM-L6-v2 등, CPU 구동) |
| **벡터 DB** | ChromaDB (로컬, 서버리스) |
| **하드웨어** | 개인 PC (RTX 4090 — 임베딩 가속용, 필수 아님) |

> **비용: $0** — 전체 파이프라인이 무료 API + 로컬 오픈소스 도구로 구동됨

---

## 2. 시스템 아키텍처

### 2.1 전체 구조

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 PART A: 자율 리서치 에이전트 (AlzhAgent Pipeline)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[사용자 리서치 명령]
    ↓
┌───────────────────────────────────┐
│  Supervisor Agent                 │ ← 연구 계획 수립 & 에이전트 조율
│  (DeepSeek-R1 via OpenRouter)     │   Tree of Thoughts (ToT) 추론
└──────────┬────────────────────────┘
           ↓
    ┌──────┴──────┬────────────────┐
    ↓             ↓                ↓
┌────────┐  ┌──────────┐   ┌────────────┐
│Literatu│  │ Design   │   │  Critic    │
│re Agent│  │ Agent    │   │  Agent     │
│(RAG+NER│  │(분자설계) │   │(검증+보안)  │
└───┬────┘  └────┬─────┘   └─────┬──────┘
    ↓            ↓               ↓
 PubMed/     RDKit 코드       약물성 평가
 BindingDB   생성 & 실행     + 독성 필터링
 검색·분석                   + 보안 검증
    ↓            ↓               ↓
    └────────────┴───────┬───────┘
                         ↓
              ┌─────────────────────┐
              │   Knowledge DB      │ ← 리서치 결과 지속 축적
              │   (ChromaDB)        │
              │  ・타깃 분석 결과     │
              │  ・후보물질 SMILES    │
              │  ・약물성 평가 데이터  │
              │  ・인과관계 분석 로그  │
              │  ・원본 문헌 청크     │
              └──────────┬──────────┘
                         ↓
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 PART B: 연구원용 RAG 챗봇 (AlzhChat)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[연구원 자연어 질의]
    ↓
┌───────────────────────────────────┐
│  AlzhChat (RAG Chatbot)           │
│  (Qwen-2.5-72B via OpenRouter)    │
│                                   │
│  ① Knowledge DB 검색 (리서치 결과) │
│  ② 원본 문헌 DB 검색 (PubMed 등)  │
│  ③ 두 소스 통합하여 근거 기반 답변  │
│  ④ 출처·신뢰도 스코어 함께 제시    │
└───────────────────────────────────┘
    ↓
 [연구원에게 답변 + 출처 + 후속 질문 제안]
```

### 2.2 자기강화 루프 (Self-Reinforcing Loop)

```
에이전트 리서치 실행
       ↓
  Knowledge DB 축적  ←──────────┐
       ↓                       │
  챗봇 응답 품질 향상            │
       ↓                       │
  연구원 피드백 & 새 질의        │
       ↓                       │
  에이전트에 새 리서치 과제 입력 ─┘
```

에이전트가 리서치를 수행할수록 DB가 풍부해지고, 연구원이 챗봇을 사용할수록 새로운 리서치 과제가 생성되어 시스템 전체의 지식이 지속적으로 성장한다.

---

## 3. 에이전트 상세 기능

### 3.1 Literature Agent — 지식 기반 타깃 발굴

| 항목 | 내용 |
|---|---|
| **핵심 기능** | PubMed·BindingDB·특허 문헌을 RAG로 검색, 최신 연구 동향 기반 타깃 가설 수립 |
| **기술 스택** | ChromaDB (벡터 검색), OpenRouter API (요약·추출), Sentence Transformers (임베딩) |
| **NER/NEN** | 바이오 도메인 특화 개체명 인식 및 정규화: 단백질명(Tau, TREM2, BACE1), 화합물(SMILES), 질환명, 기전 용어 → 정규화된 ID로 매핑하여 Knowledge DB에 구조화 저장 |
| **DB 축적** | 분석된 문헌 청크 + 추출된 엔티티 + 타깃-근거 매핑이 자동으로 Knowledge DB에 누적 |

### 3.2 Design Agent — 구조 기반 분자 설계

| 항목 | 내용 |
|---|---|
| **핵심 기능** | 타깃 단백질에 맞춰 약물성(QED, LogP)과 합성 용이성(SA Score)이 최적화된 저분자 화합물 SMILES 생성 |
| **기술 스택** | LLM 코드 생성 (OpenRouter) → RDKit 로컬 실행 (화학적 유효성 검증) |
| **환각 방지** | LLM 생성 SMILES를 **RDKit으로 즉시 파싱** → 유효하지 않으면 자동 폐기 & 재생성 루프 |
| **DB 축적** | 생성된 유효 화합물 SMILES + QED/LogP/SA Score 지표가 Knowledge DB에 누적 |

### 3.3 Critic Agent — 결과 검증 및 보고

| 항목 | 내용 |
|---|---|
| **핵심 기능** | 독성 작용기(Toxicophore) 검사, Lipinski Rule 검증, 최종 후보물질 리포트 자동 생성 |
| **보안** | EMNLP 2025 논문(Keep Security!) 기반 Indirect Prompt Injection 방어 필터 |
| **인과 추론** | "타깃 A 억제 → 부작용 C 발생 확률" 등 인과관계 로그를 투명하게 제시 |
| **DB 축적** | 검증 결과 + 인과 분석 로그 + 최종 랭킹이 Knowledge DB에 누적 |

### 3.4 AlzhChat — 연구원용 RAG 챗봇

| 항목 | 내용 |
|---|---|
| **핵심 기능** | 연구원이 자연어로 질의하면 Knowledge DB + 원본 문헌 DB를 동시 검색하여 근거 기반 답변 생성 |
| **듀얼 RAG** | **① Research DB 검색** (에이전트 리서치 결과: 타깃 분석, 화합물, 평가 데이터) + **② Literature DB 검색** (원본 PubMed/BindingDB 문헌 청크) → 두 소스를 통합하여 답변 |
| **출처 추적** | 모든 답변에 근거 문헌/리서치 결과의 출처와 신뢰도 스코어를 함께 제시 |
| **대화 흐름** | 멀티턴 대화 지원, 후속 질문 자동 제안, 연구원의 피드백을 새 리서치 과제로 변환 |
| **기술 스택** | Qwen-2.5-72B (OpenRouter 무료), ChromaDB 듀얼 컬렉션 검색, Streamlit Chat UI |

**예시 대화:**
```
연구원: "현재까지 에이전트가 발견한 Tau 타깃 후보물질 중 QED가 가장 높은 건?"
AlzhChat: "Research DB 기준, 3번째 리서치 사이클에서 생성된 화합물 
          SMILES: CC1=CC(=O)... 이 QED 0.78로 가장 높습니다.
          근거: [PubMed #38291045] Tau fibril 구조 PDB 5O3L 기반 설계.
          SA Score: 3.2, Lipinski 5개 규칙 모두 충족.
          → 추가로 TREM2 교차 활성 가능성을 분석해볼까요?"
```

---

## 4. 핵심 기술적 차별점

### 4.1 이중 검증 루프 (Anti-Hallucination)

LLM이 생성한 화합물을 **(1) RDKit 화학적 유효성 검증** + **(2) 약물성 지표 정량 평가**로 이중 필터링. 존재하지 않는 화합물 생성 문제를 원천 차단한다.

### 4.2 Tree of Thoughts (ToT) 연구 경로 탐색

Supervisor Agent가 단선적 CoT를 넘어, 여러 연구 경로(타깃 A vs B, 기전 조합 등)를 병렬 탐색하고 Critic Agent 피드백으로 최적 경로를 선택한다.

### 4.3 보안 Guardrail

EMNLP 2025 팀 자체 연구를 적용하여 외부 데이터(논문, 웹) 검색 시 포함될 수 있는 Indirect Prompt Injection을 사전 탐지·차단한다.

### 4.4 Zero-Cost 아키텍처

GPU 클러스터 없이 **OpenRouter 무료 API + 로컬 RDKit + ChromaDB**만으로 전체 파이프라인이 동작하는 경량 설계. 중소 연구소도 즉시 도입 가능하다.

---

## 5. 기술 스택 요약

| 분류 | 기술 | 비용 |
|---|---|---|
| **LLM (Supervisor)** | DeepSeek-R1 via OpenRouter | 무료 |
| **LLM (Literature/Design)** | Llama-3-70B via OpenRouter | 무료 |
| **LLM (Chatbot)** | Qwen-2.5-72B via OpenRouter | 무료 |
| **LLM (Critic)** | DeepSeek-R1 via OpenRouter | 무료 |
| **임베딩** | Sentence Transformers (로컬 CPU) | 무료 |
| **벡터 DB** | ChromaDB (로컬) | 무료 |
| **화학 도구** | RDKit (로컬) | 무료 |
| **NER/NEN** | spaCy + scispaCy 바이오 모델 (로컬) | 무료 |
| **에이전트 프레임워크** | LangGraph (에이전트 오케스트레이션) | 무료 |
| **데이터** | PubMed (공개), BindingDB (공개) | 무료 |
| **UI** | Streamlit (챗봇 + 에이전트 대시보드) | 무료 |
| **총 비용** | | **$0** |

---

## 6. 구현 계획 (3일 스프린트)

### Day 1 — 데이터 파이프라인 & Knowledge DB 구축

- PubMed/BindingDB 알츠하이머 문헌·활성 데이터 수집 및 전처리
- ChromaDB 듀얼 컬렉션 설계: `literature_db` (원본 문헌) + `research_db` (리서치 결과)
- 임베딩 파이프라인 구축 (Sentence Transformers)
- Literature Agent: RAG 검색 + scispaCy 기반 바이오 NER/NEN 파이프라인

### Day 2 — 멀티 에이전트 코어 + 챗봇

- Supervisor Agent: DeepSeek-R1 기반 ToT 추론 + LangGraph 오케스트레이션
- Design Agent: LLM → RDKit 화학 검증 루프
- Critic Agent: 약물성 평가 + 독성 필터링 자동화
- **AlzhChat 챗봇:** 듀얼 RAG (Research DB + Literature DB) 통합 검색 + 답변 생성

### Day 3 — 통합 테스트 & 데모 UI

- 에이전트 → DB 축적 → 챗봇 참조의 전체 루프 통합 테스트
- 벤치마크: 기존 약물(Donepezil 등) 재발견 테스트
- Streamlit UI: 에이전트 실행 대시보드 + 챗봇 인터페이스 통합
- 데모 시나리오 작성 및 패키징

### 정량적 평가 지표

| 지표 | 목표 |
|---|---|
| 에이전트 코드 실행 성공률 | > 95% |
| 챗봇 답변 출처 정확도 | > 90% (출처 추적 가능한 답변 비율) |
| Prompt Injection 방어율 | > 95% |
| 생성 화합물 약물성 | QED ≥ 0.6, SA Score ≤ 4.0 |
| 기존 약물 재발견 | Donepezil 등 ChemBench 기준 평가 |

---

## 7. 파급효과

- **연구 자동화:** 문헌 검색·데이터 전처리·초기 스크리닝 반복 업무 80% 이상 단축
- **지식 축적:** 에이전트가 리서치할수록 DB가 성장하여 조직 전체의 연구 자산으로 활용
- **접근성:** 자연어 챗봇으로 비전공 연구원도 복잡한 화합물·타깃 데이터에 즉시 접근
- **도입 비용 $0:** GPU 없이 무료 API + 오픈소스만으로 운영, 중소 제약사·연구소 즉시 도입 가능
- **글로벌 시장:** AI 신약 시장 2026년 32억 달러 규모, 알츠하이머 특화 에이전트 수요 높음

---

## References

1. Large Language Model-Driven Prioritization of Alzheimer's Disease Drug Targets Across Multidimensional Criteria | medRxiv (2025)
2. Therapeutic Modalities Targeting Tau Protein in Alzheimer's Disease - MDPI (2025)
3. De novo drug design by iterative multiobjective deep reinforcement learning - Oxford Academic (2023)
4. Exploring Modularity of Agentic Systems for Drug Discovery - arXiv (2025)
5. DruGagent: Multi-Agent LLM-Based Reasoning for Drug-Target Interaction Prediction - PMC (2025)
6. Agentic AI for Scientific Discovery: A Survey - arXiv (2025)
7. Accelerating scientific breakthroughs with an AI co-scientist - Google Research (2025)