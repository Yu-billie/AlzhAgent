"""agents/llm_client.py - OpenRouter 무료 API 클라이언트"""
import json, re, logging
from openai import OpenAI
from config import CONFIG

logger = logging.getLogger("LLM")


class LLMClient:
    def __init__(self, api_key: str):
        self.client = OpenAI(
            base_url=CONFIG["openrouter_base_url"],
            api_key=api_key,
        )

    def chat(self, messages: list[dict], model_key: str = "supervisor",
             temperature: float = 0.7, max_tokens: int = 4096) -> str:
        model = CONFIG["models"].get(model_key, CONFIG["models"]["supervisor"])
        try:
            resp = self.client.chat.completions.create(
                model=model, messages=messages,
                temperature=temperature, max_tokens=max_tokens,
            )
            content = resp.choices[0].message.content or ""
            content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
            return content
        except Exception as e:
            logger.error(f"LLM error: {e}")
            return f"[LLM Error: {e}]"

    def chat_direct(self, messages: list[dict], model_id: str,
                    temperature: float = 0.7, max_tokens: int = 4096) -> str:
        """model ID를 직접 지정하여 호출 (CONFIG 매핑 우회)"""
        logger.info(f"chat_direct calling model: {model_id}")
        try:
            resp = self.client.chat.completions.create(
                model=model_id, messages=messages,
                temperature=temperature, max_tokens=max_tokens,
            )
            content = resp.choices[0].message.content or ""
            content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
            return content
        except Exception as e:
            logger.error(f"LLM error: {e}")
            return f"[LLM Error: {e}]"

    def chat_json(self, messages: list[dict], model_key: str = "supervisor",
                  temperature: float = 0.3) -> dict:
        if messages and messages[0]["role"] == "system":
            messages[0]["content"] += (
                "\n\nIMPORTANT: Respond ONLY with valid JSON. "
                "No markdown fences, no preamble."
            )
        raw = self.chat(messages, model_key=model_key, temperature=temperature)
        raw = re.sub(r"```json\s*", "", raw)
        raw = re.sub(r"```\s*", "", raw).strip()
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            if m:
                try:
                    return json.loads(m.group())
                except json.JSONDecodeError:
                    pass
            m = re.search(r"\[.*\]", raw, re.DOTALL)
            if m:
                try:
                    return {"items": json.loads(m.group())}
                except json.JSONDecodeError:
                    pass
            return {"raw": raw}