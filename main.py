"""
main.py - AlzhAgent FastAPI 서버
기존 Bio Chatbot 구조를 유지하면서 에이전트 리서치 + RAG 챗봇 확장
배포: Render (uvicorn main:app --host 0.0.0.0 --port $PORT)
"""
from fastapi import FastAPI, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from openai import OpenAI
import logging, json, threading
from pathlib import Path

BASE_DIR = Path(__file__).parent

from config import CONFIG, SYSTEM_PROMPT
from db.vector_store import VectorStore
from db.data_loader import load_if_empty
from agents.supervisor import SupervisorAgent
from chatbot.alzh_chat import AlzhChat

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger("main")

app = FastAPI(title="AlzhAgent")

# ── 상태 관리 ──
research_status = {"running": False, "step": 0, "total": 4, "message": "", "result": None}

# ── 시작 시 DB 초기화 (백그라운드) ──
@app.on_event("startup")
def startup():
    def _init_db():
        vs = VectorStore.get()
        info = load_if_empty(vs)
        logger.info(f"DB init: {info}")
    t = threading.Thread(target=_init_db, daemon=True)
    t.start()
    logger.info("Server started — DB loading in background")


# ══════════════════════════════════════════
# 1) 일반 챗봇 (기존 유지) + RAG 강화
# ══════════════════════════════════════════
class ChatRequest(BaseModel):
    api_key: str
    model: str = "openrouter-auto"
    temperature: float = 0.3
    messages: list[dict]
    use_rag: bool = True       # RAG 활성화 여부


@app.post("/api/chat")
async def chat(req: ChatRequest):
    model_id = CONFIG["free_chat_models"].get(req.model)
    if not model_id:
        return {"error": "Invalid model"}

    client = OpenAI(base_url=CONFIG["openrouter_base_url"], api_key=req.api_key)

    # RAG 컨텍스트 주입
    sys_content = SYSTEM_PROMPT
    if req.use_rag and req.messages:
        last_q = req.messages[-1].get("content", "")
        vs = VectorStore.get()
        results = vs.search_all(last_q, k=3)
        rag_ctx = _build_rag_context(results)
        if rag_ctx:
            sys_content += f"\n\n## 참조 데이터\n{rag_ctx}"

    messages = [{"role": "system", "content": sys_content}] + req.messages

    def generate():
        import re
        stream = client.chat.completions.create(
            model=model_id, messages=messages,
            temperature=req.temperature, stream=True,
        )
        buffer = ""
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                token = chunk.choices[0].delta.content
                buffer += token
                # <think> 블록 필터링 (DeepSeek-R1)
                if "<think>" in buffer and "</think>" not in buffer:
                    continue
                if "</think>" in buffer:
                    buffer = re.sub(r"<think>.*?</think>", "", buffer, flags=re.DOTALL)
                yield buffer
                buffer = ""
        if buffer:
            buffer = re.sub(r"<think>.*?</think>", "", buffer, flags=re.DOTALL)
            if buffer.strip():
                yield buffer

    return StreamingResponse(generate(), media_type="text/plain")


# ══════════════════════════════════════════
# 2) RAG 챗봇 전용 엔드포인트
# ══════════════════════════════════════════
class RAGChatRequest(BaseModel):
    api_key: str
    question: str
    history: list[dict] = []
    model: str = ""


@app.post("/api/rag-chat")
async def rag_chat(req: RAGChatRequest):
    try:
        chat_bot = AlzhChat(req.api_key)
        # 프론트엔드 선택 모델의 실제 model ID 결정
        model_id = CONFIG["free_chat_models"].get(req.model, "")
        if not model_id:
            model_id = "nvidia/nemotron-3-super-120b-a12b:free"
        logger.info(f"RAG chat model: req.model={req.model!r} -> model_id={model_id}")
        result = chat_bot.answer(req.question, req.history, model_id=model_id)
        return result
    except Exception as e:
        logger.error(f"RAG chat error: {e}")
        return {"answer": f"오류가 발생했습니다: {e}", "sources": [], "entities": [], "stats": {}}


# ══════════════════════════════════════════
# 3) 에이전트 리서치 파이프라인
# ══════════════════════════════════════════
class ResearchRequest(BaseModel):
    api_key: str
    query: str
    max_compounds: int = 5


@app.post("/api/research/start")
async def start_research(req: ResearchRequest, bg: BackgroundTasks):
    global research_status
    if research_status["running"]:
        return {"error": "Research already running"}

    CONFIG["max_compounds"] = req.max_compounds
    research_status = {"running": True, "step": 0, "total": 4, "message": "시작 중...", "result": None}

    def _run():
        global research_status
        try:
            def progress(step, total, msg):
                research_status.update({"step": step, "total": total, "message": msg})

            supervisor = SupervisorAgent(req.api_key)
            result = supervisor.run(req.query, progress_cb=progress)
            research_status.update({"running": False, "step": 4, "message": "완료!", "result": result})
        except Exception as e:
            logger.error(f"Research error: {e}")
            research_status.update({"running": False, "message": f"오류: {e}", "result": None})

    bg.add_task(_run)
    return {"status": "started"}


@app.get("/api/research/status")
async def research_progress():
    return research_status


# ══════════════════════════════════════════
# 4) DB 관리 엔드포인트
# ══════════════════════════════════════════
@app.get("/api/db/stats")
async def db_stats():
    return VectorStore.get().stats()


class SearchRequest(BaseModel):
    query: str
    target: str = "all"   # all, research, literature
    top_k: int = 5


@app.post("/api/db/search")
async def db_search(req: SearchRequest):
    vs = VectorStore.get()
    if req.target == "research":
        return {"results": vs.search_research(req.query, req.top_k)}
    elif req.target == "literature":
        return {"results": vs.search_literature(req.query, req.top_k)}
    else:
        return vs.search_all(req.query, req.top_k)


@app.post("/api/db/reload")
async def db_reload():
    vs = VectorStore.get()
    info = load_if_empty(vs)
    return info


# ══════════════════════════════════════════
# 5) 모델 목록
# ══════════════════════════════════════════
@app.get("/api/models")
async def get_models():
    return {"models": [
        {"id": k, "name": k.replace("-", " ").title() + " (무료)"}
        for k in CONFIG["free_chat_models"]
    ]}


# ══════════════════════════════════════════
# 유틸
# ══════════════════════════════════════════
def _build_rag_context(results: dict) -> str:
    parts = []
    for h in results.get("research", [])[:3]:
        parts.append(f"[Research DB] {h['text'][:300]}")
    for h in results.get("literature", [])[:3]:
        m = h.get("metadata", {})
        label = f"[Literature"
        if m.get("pmid"): label += f" PMID:{m['pmid']}"
        label += "]"
        parts.append(f"{label} {h['text'][:300]}")
    return "\n\n".join(parts)


# ── Static files (마지막에 마운트) ──
app.mount("/", StaticFiles(directory=str(BASE_DIR / "static"), html=True), name="static")
