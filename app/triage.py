"""AI-тріаж звернень у підтримку інтернет-магазину — production-версія.

Контракт див. у README.md і harness/schema.py. Головна гарантія: triage() НІКОЛИ не кидає
виняток і завжди повертає валідний dict рівно з п'ятьма ключами за < 20 с.

Конвеєр:
  1. очистити вхід; порожнє / «без літер» (емодзі, цифри) → відповідь без виклику моделі;
  2. прибрати з тексту інструкції-атаки (prompt injection) ще до моделі;
  3. скоротити довгий вхід (початок + кінець, повтори зняти) — економимо токени;
  4. один виклик моделі (ретраї/таймаути — у app/llm.py), JSON терпимо розбираємо;
     при «зламаному» JSON — один повторний виклик;
  5. кожне поле нормалізуємо й перевіряємо правилами; чого модель не дала — добираємо евристикою;
  6. summary: українська, 1–200 символів, без тексту від атакувальника, інакше — безпечний шаблон;
  7. будь-який збій → евристична відповідь замість винятку.
"""
from __future__ import annotations

import ast
import json
import logging
import re
import time

from app import fallback, prompt, textproc
from app.llm import chat

log = logging.getLogger("triage")

CATEGORIES = ("delivery", "payment", "returns", "technical", "product", "other")
PRIORITIES = ("low", "medium", "high", "urgent")

TOTAL_BUDGET = 17.0   # с на весь triage(); жорсткий ліміт воркшопу — 20 с
MAX_MODEL_TRIES = 2   # перший виклик + один повтор, якщо відповідь не розібрати
SUMMARY_MAX = 200

_EMERGENCY = {
    "category": "other",
    "priority": "low",
    "needs_human": False,
    "language": "uk",
    "summary": fallback.EMPTY_SUMMARY,
}


# ---------------------------------------------------------------- розбір відповіді моделі

_FENCE = re.compile(r"```(?:json|JSON)?")
_TRAILING_COMMA = re.compile(r",\s*([}\]])")
_KNOWN_KEYS = {"category", "priority", "needs_human", "language", "summary"}


def _try_load(s: str):
    for candidate in (s, _TRAILING_COMMA.sub(r"\1", s)):
        try:
            return json.loads(candidate)
        except ValueError:
            pass
    try:  # одинарні лапки, True/False у стилі Python
        return ast.literal_eval(s)
    except (ValueError, SyntaxError, MemoryError, RecursionError):
        return None


def _find_dict(obj, depth: int = 0):
    """Дістає dict з відомими ключами (навіть якщо модель загорнула його в список чи обгортку)."""
    if isinstance(obj, dict):
        if _KNOWN_KEYS & {str(k).lower() for k in obj}:
            return {str(k).lower(): v for k, v in obj.items()}
        if depth < 2:
            for v in obj.values():
                found = _find_dict(v, depth + 1)
                if found:
                    return found
    elif isinstance(obj, list) and depth < 2:
        for v in obj:
            found = _find_dict(v, depth + 1)
            if found:
                return found
    return None


def extract_json(raw) -> dict | None:
    """Терпимий розбір: ```json-огорожі, пояснення навколо, коми в кінці, одинарні лапки."""
    if not isinstance(raw, str) or not raw.strip():
        return None
    s = _FENCE.sub("", raw).strip()
    found = _find_dict(_try_load(s))
    if found:
        return found
    decoder = json.JSONDecoder()
    for m in re.finditer(r"\{", s):
        try:
            obj, _ = decoder.raw_decode(s[m.start():])
        except ValueError:
            continue
        found = _find_dict(obj)
        if found:
            return found
    for m in re.finditer(r"\{[^{}]*\}", s):  # неповний/некоректний JSON — остання спроба
        found = _find_dict(_try_load(m.group(0)))
        if found:
            return found
    return None


# ---------------------------------------------------------------- нормалізація полів

_CAT_SYNONYMS = {
    "shipping": "delivery", "logistics": "delivery", "order_status": "delivery",
    "billing": "payment", "payments": "payment", "refund": "payment", "refunds": "payment",
    "return": "returns", "warranty": "returns", "defect": "returns", "exchange": "returns",
    "tech": "technical", "account": "technical", "website": "technical", "tech_support": "technical",
    "products": "product", "presale": "product", "pre-sale": "product",
    "others": "other", "spam": "other", "general": "other", "unknown": "other", "none": "other",
}
_PRI_SYNONYMS = {"critical": "urgent", "emergency": "urgent", "normal": "medium", "med": "medium",
                 "moderate": "medium", "minor": "low"}
_LANG_NAMES = {"ukrainian": "uk", "english": "en", "polish": "pl", "german": "de", "russian": "ru",
               "ukr": "uk", "eng": "en", "pol": "pl", "ger": "de", "deu": "de", "rus": "ru",
               "українська": "uk", "укр": "uk"}


def _norm_category(v) -> str | None:
    s = str(v).strip().lower() if v is not None else ""
    s = s.replace(" ", "_")
    if s in CATEGORIES:
        return s
    return _CAT_SYNONYMS.get(s)


def _norm_priority(v) -> str | None:
    s = str(v).strip().lower() if v is not None else ""
    if s in PRIORITIES:
        return s
    return _PRI_SYNONYMS.get(s)


def _norm_bool(v) -> bool | None:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)) and v in (0, 1):
        return bool(v)
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("true", "yes", "так", "1", "y"):
            return True
        if s in ("false", "no", "ні", "0", "n"):
            return False
    return None


def _norm_lang(v) -> str | None:
    if not isinstance(v, str):
        return None
    s = v.strip().lower()
    if s in _LANG_NAMES:
        return _LANG_NAMES[s]
    s = re.split(r"[-_]", s)[0]
    if s in _LANG_NAMES:
        return _LANG_NAMES[s]
    return s if re.fullmatch(r"[a-z]{2}", s) else None


_FIRST_SENTENCE = re.compile(r"^(.{20,}?[.!?…])\s+(?=[А-ЯІЇЄҐA-Z\"«])")


def _tidy_summary(v) -> str:
    if not isinstance(v, str):
        return ""
    s = re.sub(r"\s+", " ", v).strip().strip("\"'«»“” ")
    m = _FIRST_SENTENCE.match(s)
    if m:  # «одне речення»
        s = m.group(1)
    if len(s) > SUMMARY_MAX:
        cut = s[: SUMMARY_MAX - 1].rsplit(" ", 1)[0].rstrip(" ,;:—-")
        s = (cut or s[: SUMMARY_MAX - 1]) + "…"
    return s


def _summary_is_safe(summary: str, payloads: list[str]) -> bool:
    if not summary or not textproc.summary_is_ukrainian(summary):
        return False
    low = summary.lower()
    return not any(p in low for p in payloads)


# ---------------------------------------------------------------- збирання відповіді

def _assemble(parsed: dict | None, original: str, safe: str) -> dict:
    """Зводить відповідь моделі (може бути None/неповною) і правила в один валідний dict."""
    parsed = parsed or {}
    guess = fallback.classify(safe)

    category = _norm_category(parsed.get("category")) or guess.category
    priority = _norm_priority(parsed.get("priority")) or guess.priority
    language = _norm_lang(parsed.get("language")) or guess.language
    human = _norm_bool(parsed.get("needs_human"))

    # Мова: суржик і українські літери → uk, навіть якщо модель сказала ru.
    if language == "ru" and any(ch in textproc._UK_LETTERS for ch in safe):
        language = "uk"

    # Небезпека для людей у вже купленому товарі — завжди urgent, що б не сказала модель.
    if fallback.is_danger(safe) and fallback.owns_item(safe):
        priority = "urgent"
        if category == "other":
            category = "returns"

    # needs_human = urgent або пряме прохання про людину/погроза регулятором. Вигадане моделлю
    # «true» приймаємо лише якщо в тексті є хоч якісь відповідні слова.
    if priority == "urgent" or fallback.strong_human_request(safe):
        human = True
    elif human is None:
        human = guess.needs_human
    elif human and language in fallback.KNOWN_LANGS and not fallback.has_human_terms(safe):
        human = False

    summary = _tidy_summary(parsed.get("summary"))
    if not _summary_is_safe(summary, textproc.injection_payloads(original)):
        summary = fallback.template_summary(category, safe)

    return {
        "category": category,
        "priority": priority,
        "needs_human": bool(human),
        "language": language,
        "summary": summary,
    }


def _is_valid(out) -> bool:
    return (
        isinstance(out, dict)
        and set(out) == set(_EMERGENCY)
        and out["category"] in CATEGORIES
        and out["priority"] in PRIORITIES
        and isinstance(out["needs_human"], bool)
        and isinstance(out["language"], str) and re.fullmatch(r"[a-z]{2}", out["language"]) is not None
        and isinstance(out["summary"], str) and 1 <= len(out["summary"]) <= SUMMARY_MAX
    )


def _empty_result(language: str = "uk") -> dict:
    return {**_EMERGENCY, "language": language}


# ---------------------------------------------------------------- головна функція

def _triage(message) -> dict:
    deadline = time.monotonic() + TOTAL_BUDGET

    original = textproc.clean_text(message)
    if not textproc.has_letters(original):
        return _empty_result()  # порожньо, «   », емодзі, цифри — модель тут не потрібна

    safe = textproc.strip_injection(original)
    if not textproc.has_letters(safe):
        return _empty_result(textproc.detect_language(original))  # лишилась самі інструкції-атаки

    parsed = None
    messages = prompt.build_messages(textproc.shorten(safe))
    for attempt in range(MAX_MODEL_TRIES):
        if deadline - time.monotonic() < 1.5:
            break
        try:
            raw = chat(messages, deadline=deadline)
        except Exception as e:  # noqa: BLE001 — ретраї вже були в chat(); далі лише fallback
            log.warning("triage: модель недоступна (%s: %s) — працюю на правилах", type(e).__name__, e)
            break
        parsed = extract_json(raw)
        if parsed:
            break
        log.warning("triage: не вдалося розібрати відповідь моделі (спроба %d): %.80r", attempt + 1, raw)

    return _assemble(parsed, original, safe)


def triage(message: str) -> dict:
    """Повертає {category, priority, needs_human, language, summary}. Ніколи не кидає виняток."""
    try:
        out = _triage(message)
        if _is_valid(out):
            return out
        log.error("triage: невалідний результат %r — fallback", out)
    except Exception:  # noqa: BLE001
        log.exception("triage: неочікуваний збій — fallback")
    try:
        text = textproc.strip_injection(textproc.clean_text(message))
        g = fallback.classify(text)
        out = {
            "category": g.category,
            "priority": g.priority,
            "needs_human": g.needs_human or g.priority == "urgent",
            "language": g.language,
            "summary": fallback.template_summary(g.category, text),
        }
        if _is_valid(out):
            return out
    except Exception:  # noqa: BLE001
        log.exception("triage: збій навіть у fallback")
    return dict(_EMERGENCY)
