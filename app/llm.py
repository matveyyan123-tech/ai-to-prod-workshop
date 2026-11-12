"""Обгортка над LLM API (Gemini через OpenAI-сумісний endpoint) з production-поведінкою.

Правила воркшопу: лише офіційний `openai` SDK, без streaming; модель — з LLM_MODEL;
ключ — лише зі змінної середовища GEMINI_API_KEY.

Що тут зроблено і чому:
  * `max_retries=0` + власний цикл: SDK за замовчуванням чекає до 10 хв і ретраїть лише двічі,
    а нам треба вкластися в жорсткий дедлайн і пережити 3+ поспіль 429;
  * кожен запит має власний короткий таймаут, а весь виклик — спільний дедлайн;
  * 429 → чекаємо стільки, скільки просить сервер (retry-after), але не довше за дедлайн;
    5xx / таймаути / обриви / порожні відповіді → експоненційний backoff з jitter;
  * 401/403/404 і помилки конфігурації не ретраїмо — це марно;
  * необов'язкові параметри (reasoning_effort, response_format, temperature) вимикаються самі,
    якщо модель відповіла 400 «не підтримується» — і більше не надсилаються.
"""
from __future__ import annotations

import logging
import os
import random
import threading
import time

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

log = logging.getLogger("triage.llm")

MODEL = os.getenv("LLM_MODEL", "gemini-3.5-flash-lite")
BASE_URL = os.getenv("LLM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")

PER_TRY_TIMEOUT = 7.0   # с на один HTTP-запит
DEFAULT_BUDGET = 14.0   # с на весь виклик chat(), якщо дедлайн не передали
MAX_ATTEMPTS = 5
MIN_TRY_TIME = 1.0      # якщо лишилось менше — нового запиту вже не починаємо
MAX_SLEEP = 3.0

# Необов'язкові параметри в порядку «від найімовірніше непідтримуваного».
# reasoning_effort=none — не платити токенами за «роздуми» там, де класифікація їх не потребує.
_OPTIONAL = [
    ("reasoning_effort", "none"),
    ("response_format", {"type": "json_object"}),
    ("temperature", 0),
]
_dropped: set[str] = set()

_client = None
_client_lock = threading.Lock()


class LLMError(RuntimeError):
    """Модель так і не дала відповіді (після всіх ретраїв / у межах дедлайну)."""


def get_client() -> OpenAI:
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                _client = OpenAI(
                    api_key=os.environ["GEMINI_API_KEY"],
                    base_url=BASE_URL,
                    max_retries=0,
                    timeout=PER_TRY_TIMEOUT,
                )
    return _client


def _status(exc: BaseException):
    code = getattr(exc, "status_code", None)
    if not isinstance(code, int):
        code = getattr(getattr(exc, "response", None), "status_code", None)
    return code if isinstance(code, int) else None


def _retry_after(exc: BaseException):
    try:
        headers = exc.response.headers  # type: ignore[attr-defined]
        ms = headers.get("retry-after-ms")
        if ms:
            return float(ms) / 1000.0
        sec = headers.get("retry-after")
        if sec:
            return float(sec)
    except Exception:  # noqa: BLE001 — заголовка може не бути або він у форматі дати
        pass
    return None


def _drop_unsupported(exc: BaseException) -> bool:
    """Після 400 вимикає один необов'язковий параметр. True, якщо було що вимикати."""
    msg = str(exc).lower()
    active = [name for name, _ in _OPTIONAL if name not in _dropped]
    if not active:
        return False
    named = [n for n in active if n in msg or n.replace("_", " ") in msg]
    victim = named[0] if named else active[0]
    _dropped.add(victim)
    log.warning("LLM: параметр %s не підтримується, вимикаю", victim)
    return True


def _text_of(resp) -> str:
    try:
        content = resp.choices[0].message.content
    except (AttributeError, IndexError, TypeError):
        return ""
    if isinstance(content, list):  # деякі провайдери віддають список частин
        content = "".join(str(p.get("text", "")) if isinstance(p, dict) else str(p) for p in content)
    return content.strip() if isinstance(content, str) else ""


# Помилки конфігурації/коду (немає ключа, зламаний імпорт) ретраїти марно. ValueError сюди не входить:
# JSONDecodeError від «кривої» відповіді сервера — це якраз тимчасовий збій.
_CONFIG_ERRORS = (KeyError, TypeError, AttributeError, ImportError)


def chat(messages, *, deadline: float | None = None, **kwargs) -> str:
    """Надсилає повідомлення моделі й повертає непорожній текст відповіді.

    deadline — момент часу за time.monotonic(), після якого не можна нічого починати.
    Кидає LLMError, якщо за цей час відповіді не отримано.
    """
    if deadline is None:
        deadline = time.monotonic() + DEFAULT_BUDGET
    client = get_client()

    last_err: BaseException | None = None
    attempt = 0
    while attempt < MAX_ATTEMPTS:
        remaining = deadline - time.monotonic()
        if remaining < MIN_TRY_TIME:
            break
        timeout = max(1.0, min(PER_TRY_TIMEOUT, remaining - 0.3))
        params = {k: v for k, v in _OPTIONAL if k not in _dropped and k not in kwargs}
        params.update(kwargs)
        attempt += 1
        delay = 0.0
        try:
            resp = client.chat.completions.create(model=MODEL, messages=messages, timeout=timeout, **params)
            text = _text_of(resp)
            if text:
                return text
            last_err = LLMError("порожня відповідь моделі")
            delay = 0.2
        except Exception as e:  # noqa: BLE001 — класифікуємо за статусом, а не за класом
            last_err = e
            status = _status(e)
            if status in (400, 422):
                if _drop_unsupported(e):
                    attempt -= 1  # це не збій сервісу, а наш параметр — пробуємо без нього одразу
                    continue
                raise LLMError(f"запит відхилено ({status}): {e}") from e
            if status in (401, 403, 404) or (status is None and isinstance(e, _CONFIG_ERRORS)):
                raise LLMError(f"неповторювана помилка: {type(e).__name__}: {e}") from e
            backoff = 0.4 * (2 ** (attempt - 1)) + random.uniform(0.0, 0.2)
            wait = _retry_after(e) if status == 429 else None
            delay = min(wait if wait is not None else backoff, MAX_SLEEP)
            log.info("LLM: спроба %d не вдалась (%s %s), чекаю %.1f с", attempt, status, type(e).__name__, delay)

        sleep_for = min(delay, deadline - time.monotonic() - MIN_TRY_TIME)
        if sleep_for > 0:
            time.sleep(sleep_for)

    what = type(last_err).__name__ if last_err else "дедлайн"
    raise LLMError(f"немає відповіді за {attempt} спроб: {what}") from last_err
