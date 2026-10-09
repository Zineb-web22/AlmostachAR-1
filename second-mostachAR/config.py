# إعدادات المشروع المركزية وتحميل متغيرات البيئة

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import litellm

# ضمان ضبط مسار المشروع الأساسي في sys.path لتعمل جميع الاستيرادات بسلاسة
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

# تحميل متغيرات البيئة من ملف .env
load_dotenv()

_original_completion = litellm.completion
_original_acompletion = litellm.acompletion


def _strip_cache_control(messages):
    if not messages:
        return messages
    for msg in messages:
        if isinstance(msg, dict):
            msg.pop("cache_control", None)
            content = msg.get("content")
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict):
                        part.pop("cache_control", None)
    return messages


def _patched_completion(*args, **kwargs):
    if "messages" in kwargs:
        kwargs["messages"] = _strip_cache_control(kwargs["messages"])
    return _original_completion(*args, **kwargs)


async def _patched_acompletion(*args, **kwargs):
    if "messages" in kwargs:
        kwargs["messages"] = _strip_cache_control(kwargs["messages"])
    return await _original_acompletion(*args, **kwargs)


litellm.completion = _patched_completion
litellm.acompletion = _patched_acompletion

from crewai import LLM

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError(
        "GROQ_API_KEY غير موجود في ملف .env. "
        "تأكدي من إضافة السطر: GROQ_API_KEY=gsk_..."
    )

DEFAULT_MODEL_NAME = "groq/openai/gpt-oss-120b"


def get_llm(model_name: str = DEFAULT_MODEL_NAME, temperature: float = 0.3):
    """
    يرجع نسخة مهيأة من crewai.LLM جاهزة للاستخدام في أي وكيل.
    """
    return LLM(
        model=model_name,
        api_key=GROQ_API_KEY,
        temperature=temperature,
    )


DATA_RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
DATA_PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
KNOWLEDGE_BASE_DIR = os.path.join(BASE_DIR, "knowledge_base")
