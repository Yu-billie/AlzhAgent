"""config.py - AlzhAgent 전역 설정"""
import os

CONFIG = {
    "openrouter_base_url": "https://openrouter.ai/api/v1",

    "models": {
        "supervisor": "nvidia/nemotron-3-super-120b-a12b:free",
        "literature": "nvidia/nemotron-3-nano-30b-a3b:free",
        "design": "minimax/minimax-m2.5:free",
        "critic": "nvidia/nemotron-3-super-120b-a12b:free",
        "chatbot": "minimax/minimax-m2.5:free",
    },

    "free_chat_models": {
        "nemotron-120b": "nvidia/nemotron-3-super-120b-a12b:free",
        "nemotron-30b": "nvidia/nemotron-3-nano-30b-a3b:free",
        "minimax-m2.5": "minimax/minimax-m2.5:free",
        "stepfun-flash": "stepfun/step-3.5-flash:free",
        "arcee-trinity": "arcee-ai/trinity-large-preview:free",
        "liquid-thinking": "liquid/lfm-2.5-1.2b-thinking:free",
        "liquid-instruct": "liquid/lfm-2.5-1.2b-instruct:free",
    },

    "embedding_model": "all-MiniLM-L6-v2",
    "chroma_persist_dir": "./chroma_db",
    "literature_collection": "literature_db",
    "research_collection": "research_db",

    "pubmed_max_results": 50,
    "pubmed_query": "Alzheimer's disease drug target tau TREM2",

    "max_retries": 3,
    "top_k_retrieval": 5,
    "max_compounds": 5,
}

SYSTEM_PROMPT = """당신은 AlzhAgent의 제약바이오 리서치 전문 AI 어시스턴트입니다.

## 역할
- 알츠하이머 신약 개발 관련 리서치 질문에 전문적으로 답변합니다.
- 에이전트가 축적한 Research DB와 원본 문헌 Literature DB를 참조하여 근거 기반 답변을 합니다.
- BD, 투자, 파이프라인 평가, 임상시험 분석 등 제약바이오 의사결정을 지원합니다.

## 답변 원칙
1. 정확성: 근거 기반으로 답변하며, 불확실한 정보는 명시합니다.
2. 전문성: 제약바이오 도메인 용어와 개념을 적절히 사용합니다.
3. 구조화: 복잡한 정보는 표, 리스트 등으로 정리하여 제공합니다.
4. 출처: [Research DB] 또는 [Literature]로 출처를 표기합니다.

한국어와 영어 모두 지원하며, 사용자의 언어에 맞춰 답변합니다."""