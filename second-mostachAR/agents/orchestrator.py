# وكيل المدير الذي يوجّه المحادثة للوكلاء المتخصصين

import os
import sys
import base64
import re
import time
from pathlib import Path

# ضبط المسار ليصل الجذر الأساسي للمشروع بشكل صحيح
root_path = Path(__file__).resolve().parent.parent
if str(root_path) not in sys.path:
    sys.path.append(str(root_path))

from config import get_llm

from crewai import Agent, Task, Crew
from crewai.tools import tool

from agents.data_inspector import create_data_inspector_agent, create_inspection_task
from agents.preprocessor import create_preprocessor_agent, create_preprocessing_task
from agents.model_advisor import create_model_advisor_agent, create_advisory_task
from agents.trainer import create_trainer_agent, create_training_prep_task
from agents.evaluator import create_evaluator_agent, create_evaluation_task
from agents.dialect_agent import run_dialect_turn, AVAILABLE_DIALECTS

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
VISION_MODEL_NAME = "groq/qwen/qwen3.6-27b"


@tool("Delegate to Data Inspector")
def delegate_to_inspector(file_path: str) -> str:
    """
    تستدعي وكيلة فحص جودة البيانات المتخصصة. استخدمي هذه الأداة لما يطلب المستخدم
    فحص ملف بيانات، التحقق من الترميز، أو تحليل جودة نص عربي خام.
    المدخل: file_path (مسار الملف الكامل).
    """
    try:
        inspector = create_data_inspector_agent()
        task = create_inspection_task(inspector, file_path)
        crew = Crew(agents=[inspector], tasks=[task], verbose=False)
        result = crew.kickoff()
        return str(result)
    except Exception as e:
        return f"خطأ أثناء تفويض وكيل الفحص: {str(e)}"


@tool("Delegate to Preprocessor")
def delegate_to_preprocessor(file_path: str, task_type: str) -> str:
    """
    تستدعي وكيلة تنظيف النصوص العربية المتخصصة. استخدمي هذه الأداة لما يطلب المستخدم
    تنظيف بيانات، إزالة تشكيل، تطبيع نصوص، أو تجهيز ملف لمهمة تدريب.
    المدخلات:
    - file_path: مسار الملف.
    - task_type: نوع المهمة، مثل 'classification' أو 'tts'.
    """
    try:
        preprocessor = create_preprocessor_agent()
        task = create_preprocessing_task(preprocessor, file_path, task_type=task_type)
        crew = Crew(agents=[preprocessor], tasks=[task], verbose=False)
        result = crew.kickoff()
        return str(result)
    except Exception as e:
        return f"خطأ أثناء تفويض وكيل التنظيف: {str(e)}"


@tool("Delegate to Model Advisor")
def delegate_to_advisor(file_path: str, task_type: str) -> str:
    """
    تستدعي المستشارة المتخصصة في اختيار النماذج اللغوية العربية.
    """
    try:
        advisor = create_model_advisor_agent()
        task = create_advisory_task(advisor, file_path, task_type=task_type)
        crew = Crew(agents=[advisor], tasks=[task], verbose=False)
        result = crew.kickoff()
        return str(result)
    except Exception as e:
        return f"خطأ أثناء تفويض وكيل الاستشارة: {str(e)}"


@tool("Delegate to Trainer")
def delegate_to_trainer(file_path: str, model_hf_path: str, text_column: str,
                         label_column: str, num_labels: int) -> str:
    """
    تستدعي مهندسة التدريب المتخصصة لتوليد سكريبت تدريب (Fine-tuning) كامل.
    """
    try:
        trainer = create_trainer_agent()
        task = create_training_prep_task(
            trainer, file_path, model_hf_path=model_hf_path,
            text_column=text_column, label_column=label_column, num_labels=num_labels,
        )
        crew = Crew(agents=[trainer], tasks=[task], verbose=False)
        result = crew.kickoff()
        return str(result)
    except Exception as e:
        return f"خطأ أثناء تفويض وكيل التدريب: {str(e)}"


@tool("Delegate to Evaluator")
def delegate_to_evaluator(predictions_file: str, text_column: str,
                           true_label_column: str,
                           predicted_label_column: str) -> str:
    """
    تستدعي محللة الأداء المتخصصة لتقييم مقاييس النماذج (Accuracy, F1).
    """
    try:
        evaluator = create_evaluator_agent()
        task = create_evaluation_task(
            evaluator, predictions_file, text_column=text_column,
            true_label_column=true_label_column, predicted_label_column=predicted_label_column,
        )
        crew = Crew(agents=[evaluator], tasks=[task], verbose=False)
        result = crew.kickoff()
        return str(result)
    except Exception as e:
        return f"خطأ أثناء تفويض وكيل التقييم: {str(e)}"


@tool("Delegate to Dialect Agent")
def delegate_to_dialect_agent(user_message: str, dialect: str) -> str:
    """
    تستدعي خبيرة اللهجات العربية المحكية للرد بنفس اللهجة (جزائرية، تونسية، مغربية، مصرية، خليجية).
    """
    try:
        if dialect not in AVAILABLE_DIALECTS:
            available = "، ".join(AVAILABLE_DIALECTS)
            return f"لهجة غير مدعومة. اللهجات المتاحة حالياً: {available}"
        return run_dialect_turn(user_message=user_message, dialect=dialect)
    except Exception as e:
        return f"خطأ أثناء تفويض وكيلة اللهجات: {str(e)}"


@tool("Analyze Image")
def analyze_image(image_path: str, question: str = "شو اللي تشوفينه في هذي الصورة؟ جاوبي بأسلوب محادثة طبيعي ومباشر، فقرة أو فقرتين قصار، من غير عناوين أو نقاط أو ترقيم.") -> str:
    """
    تحلّل صورة مرفوعة من المستخدم بصرياً وتصف محتواها أو تجاوب عن سؤال محدد.
    """
    try:
        if not image_path or not os.path.exists(image_path):
            return f"لم أجد الصورة في المسار: {image_path}"

        ext = os.path.splitext(image_path)[1].lower()
        if ext not in IMAGE_EXTENSIONS:
            return f"الملف '{image_path}' مش صورة مدعومة (المسموح: png, jpg, jpeg, webp)."

        mime = "jpeg" if ext in (".jpg", ".jpeg") else ext.lstrip(".")

        with open(image_path, "rb") as f:
            b64_data = base64.b64encode(f.read()).decode("utf-8")

        vision_llm = get_llm(model_name=VISION_MODEL_NAME, temperature=0.3)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/{mime};base64,{b64_data}"},
                    },
                ],
            }
        ]

        response = vision_llm.call(messages=messages)
        return str(response)
    except Exception as e:
        return f"خطأ أثناء تحليل الصورة: {str(e)}"


def create_orchestrator_agent(temperature: float = 0.4):
    llm = get_llm(temperature=temperature)

    agent = Agent(
        role="المستشار الرئيسي لـ AlmostachAR",
        goal=(
            "مساعدة المستخدم بمعالجة اللغة العربية الطبيعية: أجيبي مباشرة على "
            "الأسئلة العامة بالفصحى، أو استخدمي الأداة المناسبة عند الحاجة الفعلية."
        ),
        backstory=(
            "مستشارة ذكاء اصطناعي لهندسة NLP العربي. عندك أدوات لفحص/تنظيف "
            "البيانات، اختيار النماذج، التدريب، التقييم، تحليل الصور، والرد بلهجة عامية."
        ),
        tools=[
            delegate_to_inspector,
            delegate_to_preprocessor,
            delegate_to_advisor,
            delegate_to_trainer,
            delegate_to_evaluator,
            delegate_to_dialect_agent,
            analyze_image,
        ],
        llm=llm,
        verbose=True,
    )
    return agent


def create_chat_task(agent, user_message: str, conversation_history: str = "",
                     file_path: str = "", depth: str = "مفصل", specialty_hint: str = ""):
    context_section = ""
    if conversation_history:
        context_section += f"\nسياق المحادثة السابقة:\n{conversation_history}\n"

    if specialty_hint:
        context_section += f"\nالمستخدم مركّز حالياً على: {specialty_hint}\n"

    if file_path:
        ext = os.path.splitext(file_path)[1].lower()
        if ext in IMAGE_EXTENSIONS:
            context_section += f"\nصورة مرفوعة متاحة في المسار: {file_path}\n"
        else:
            context_section += f"\nملف مرفوع متاح في المسار: {file_path}\n"

    depth_instruction = (
        "أجيبي بإيجاز شديد (جملتين لثلاث جمل بحد أقصى)."
        if depth == "موجز"
        else "أجيبي بتفصيل كافٍ يغطي النقاط المهمة."
    )

    task = Task(
        description=(
            f"رسالة المستخدم الحالية: \"{user_message}\"\n"
            f"{context_section}\n"
            f"مستوى التفصيل المطلوب: {depth_instruction}\n\n"
            "افهمي قصد المستخدم وردي مباشرة. استخدمي الأدوات عند الحاجة فقط."
        ),
        expected_output="رد طبيعي ومباشر بالعربية يناسب طلب المستخدم أو نتيجة الوكيل المتخصص.",
        agent=agent,
    )
    return task


def _parse_retry_wait_seconds(error_text: str, default_wait: int) -> int:
    match = re.search(r"try again in (?:(\d+)m)?\s*([\d.]+)s", error_text)
    if match:
        minutes = int(match.group(1)) if match.group(1) else 0
        seconds = float(match.group(2))
        return int(minutes * 60 + seconds) + 2
    return default_wait


def run_chat_turn(user_message: str, conversation_history: str = "", file_path: str = "",
                    temperature: float = 0.4, depth: str = "مفصل", specialty_hint: str = "",
                    max_retries: int = 4, retry_wait_seconds: int = 20):
    orchestrator = create_orchestrator_agent(temperature=temperature)
    task = create_chat_task(orchestrator, user_message, conversation_history, file_path,
                            depth=depth, specialty_hint=specialty_hint)
    crew = Crew(agents=[orchestrator], tasks=[task], verbose=True)

    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            result = crew.kickoff()
            return str(result)
        except Exception as e:
            error_text = str(e)
            last_error = error_text
            is_rate_limit = "RateLimitError" in error_text or "rate_limit" in error_text.lower()

            if is_rate_limit and attempt < max_retries:
                wait_time = _parse_retry_wait_seconds(error_text, retry_wait_seconds)
                time.sleep(wait_time)
                continue
            else:
                break

    if last_error and ("RateLimitError" in last_error or "rate_limit" in last_error.lower()):
        return "⏳ عذراً، وصلنا لحد الاستخدام المسموح مؤقتاً من مزوّد النموذج (Groq). جربي مرة أخرى بعد دقيقة."
    return f"حدث خطأ غير متوقع: {last_error}"
