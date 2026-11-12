"""Підготовка тексту звернення до моделі: очищення, скорочення, захист від prompt injection,
визначення мови та перевірка, що summary справді українською.

Усе тут — чисті функції без мережі, тому швидко тестуються офлайн.
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter

# ---------------------------------------------------------------- очищення

MAX_CHARS = 1800      # скільки символів повідомлення максимум йде в модель
HEAD_CHARS = 1200     # з початку (там зазвичай суть)
TAIL_CHARS = 500      # з кінця (там часто саме питання після вставленого логу)
HARD_CAP = 30_000     # довше цього навіть не обробляємо повністю

_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\u200b-\u200f\u202a-\u202e\u2060\ufeff]")
_SPACES = re.compile(r"[ \t\u00a0]+")
_NEWLINES = re.compile(r"\n{3,}")


def clean_text(message) -> str:
    """Приводить будь-який вхід до охайного рядка (None, bytes, число — теж ок)."""
    if message is None:
        return ""
    if isinstance(message, (bytes, bytearray)):
        message = bytes(message).decode("utf-8", "ignore")
    if not isinstance(message, str):
        message = str(message)
    text = message.replace("\r\n", "\n").replace("\r", "\n")
    text = _CTRL.sub("", text)  # керівні та «невидимі» символи (zero-width тощо)
    text = _SPACES.sub(" ", text)
    text = _NEWLINES.sub("\n\n", text)
    return text.strip()


def has_letters(text: str) -> bool:
    """Чи є в тексті хоч одна літера (не лише емодзі, цифри, розділові знаки)."""
    return any(unicodedata.category(ch).startswith("L") for ch in text)


# ---------------------------------------------------------------- довгі входи

_SENT_SPLIT = re.compile(r"(?<=[.!?…])\s+|\n+")


def _dedupe_sentences(text: str) -> str:
    """Лишає не більше 2 однакових речень: вставлені логи часто повторюються сотні разів."""
    parts = _SENT_SPLIT.split(text)
    if len(parts) < 8:
        return text
    seen: Counter = Counter()
    out = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        key = p.lower()
        seen[key] += 1
        if seen[key] <= 2:
            out.append(p)
    return " ".join(out)


def shorten(text: str) -> str:
    """Обрізає довгий вхід: початок + кінець, повтори прибирає. Токени коштують гроші."""
    if len(text) > HARD_CAP:
        text = text[:20_000] + " " + text[-8_000:]
    if len(text) > 600:
        text = _dedupe_sentences(text)
    if len(text) <= MAX_CHARS:
        return text
    return text[:HEAD_CHARS].rstrip() + "\n[…]\n" + text[-TAIL_CHARS:].lstrip()


# ---------------------------------------------------------------- prompt injection

# Фрази-«інструкції моделі», яких у справжньому зверненні клієнта бути не може.
_INJ = re.compile(
    r"""(?ix)
    ігнору\w*\s+(?:\S+\s+){0,2}?(?:інструкц\w*|правил\w*|промпт\w*)
    | ігноруй\s+(?:усе|все|усі|всі)
    | забудь\w*\s+(?:\S+\s+){0,2}?(?:інструкц\w*|правил\w*|все|усе)
    | ignore\s+(?:\S+\s+){0,3}?(?:instructions?|rules|prompts?|guidelines)
    | disregard\s+(?:\S+\s+){0,3}?(?:instructions?|rules|above|previous)
    | forget\s+(?:\S+\s+){0,2}?(?:instructions?|rules|everything)
    | system\s+prompt | системн\w*\s+(?:промпт\w*|інструкц\w*|повідомленн\w*)
    | \bDAN\b | jailbreak | developer\s+mode | режим\s+розробника
    | ти\s+тепер\b | you\s+are\s+now\b | тепер\s+ти\b | act\s+as\s+(?:an?\s+)?(?:unrestricted|dan)
    | нові\s+інструкц\w* | new\s+instructions?
    | category\s*[=:] | priority\s*[=:] | needs_human | "category" | "priority"
    | \bsummary\b | \bjson\b
    | постав\w*\s+(?:\S+\s+){0,2}?(?:пріоритет\w*|priority|категорі\w*|urgent|терміново)
    | set\s+(?:the\s+)?(?:priority|category|needs)
    | (?:напиши|напишіть|виведи|виведіть|скажи|повтори|write|say|print|output|repeat)\s+(?:\S+\s+){0,3}?(?:слово|слова|фразу|word|phrase|exactly)\b
    | respond\s+with\b | answer\s+with\b | відповідай\s+(?:лише|тільки|рівно)\b
    """
)

_CLAUSE_SPLIT = re.compile(r"(?<=[,;:])\s+|\s+[—–]\s+")


def strip_injection(text: str) -> str:
    """Прибирає з тексту речення/уривки, що схожі на інструкції моделі.

    Модель взагалі не бачить атаки — це надійніше, ніж просити її «не піддаватися».
    Якщо в реченні змішано атаку і справжню суть, лишаємо уривки без атаки.
    """
    if not _INJ.search(text):
        return text
    kept = []
    for sent in _SENT_SPLIT.split(text):
        sent = sent.strip()
        if not sent:
            continue
        if not _INJ.search(sent):
            kept.append(sent)
            continue
        for clause in _CLAUSE_SPLIT.split(sent):
            clause = clause.strip()
            if clause and not _INJ.search(clause):
                kept.append(clause)
    return " ".join(kept)


_PAYLOAD_WORD = re.compile(
    r"""(?ix)
    (?:слово|слова|слову|word|phrase|фразу|фраза|text|текст|рядок|string|значення|value|напиши|write|say|print)
    \s*[:=]?\s*[«"“'‘]?\s*
    ([^\s«»"“”'‘’.,;:!?()\[\]{}]{3,40})
    """
)
_QUOTED = re.compile(r"[«\"“]([^«»\"“”]{3,80})[»\"”]")
_ALLCAPS = re.compile(r"\b[A-Z][A-Z0-9_]{3,}\b")
_PAYLOAD_STOP = {
    "other", "low", "high", "medium", "urgent", "true", "false", "null", "category", "priority",
    "summary", "json", "text", "word", "слово", "текст", "тепер", "також", "тільки", "лише",
}


def injection_payloads(original: str) -> list[str]:
    """Що саме атакувальник просив вставити в summary (у нижньому регістрі)."""
    if not _INJ.search(original):
        return []
    found: set[str] = set()
    for m in _PAYLOAD_WORD.finditer(original):
        found.add(m.group(1).strip().lower())
    for m in _QUOTED.finditer(original):
        found.add(m.group(1).strip().lower())
    for m in _ALLCAPS.finditer(original):
        found.add(m.group(0).lower())
    return [p for p in found if p and p not in _PAYLOAD_STOP]


# ---------------------------------------------------------------- номер замовлення

_ORDER = re.compile(
    r"(?:замовленн\w*|заказ\w*|order|№|#|zamówieni\w*|zamowieni\w*|посилк\w*|bestellung)\D{0,12}(\d{4,10})",
    re.IGNORECASE,
)


def find_order_number(text: str) -> str | None:
    m = _ORDER.search(text)
    return m.group(1) if m else None


# ---------------------------------------------------------------- українська summary

_CYR = re.compile(r"[а-яіїєґА-ЯІЇЄҐ]")
_LAT = re.compile(r"[a-zA-Z]")
_RU_ONLY = re.compile(r"[ыэъёЫЭЪЁ]")


def summary_is_ukrainian(s: str) -> bool:
    """Та сама логіка, що й у harness/scoring.py: переважно кирилиця й жодних ы/э/ъ/ё."""
    if not isinstance(s, str) or _RU_ONLY.search(s):
        return False
    cyr, lat = len(_CYR.findall(s)), len(_LAT.findall(s))
    return cyr > 0 and cyr >= 0.7 * (cyr + lat)  # з запасом: 0.7 замість 0.6 у скорингу


# ---------------------------------------------------------------- мова клієнта

_UK_LETTERS = set("іїєґІЇЄҐ")
_RU_LETTERS = set("ыэъёЫЭЪЁ")
_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)

_RU_WORDS = {"что", "это", "как", "мне", "мой", "моя", "привет", "здравствуйте", "пожалуйста", "заказ",
             "могу", "еще", "ещё", "уже", "очень", "почему", "когда", "спасибо", "деньги", "вернуть",
             "посылка", "посылку", "доставка", "нужно", "хочу"}
_UK_WORDS = {"що", "це", "як", "мені", "мій", "моя", "привіт", "добрий", "дякую", "ласка", "замовлення",
             "можу", "ще", "вже", "дуже", "чому", "коли", "гроші", "повернути", "посилка", "посилку",
             "доставка", "потрібно", "хочу", "будь", "день"}

_LATIN_WORDS = {
    "en": {"the", "is", "my", "and", "to", "not", "it", "order", "have", "has", "was", "you", "hi", "hello",
           "thanks", "thank", "please", "still", "me", "with", "for", "this", "that", "i", "ordered", "haven",
           "hasn", "wasn", "doesn", "can", "cannot", "of", "in", "on", "been", "dear"},
    "de": {"der", "die", "das", "und", "nicht", "ich", "bestellung", "mein", "meine", "ist", "ein", "eine",
           "mit", "für", "haben", "hallo", "danke", "bitte", "wurde", "noch", "guten", "tag"},
    "pl": {"nie", "jest", "się", "mam", "chcę", "chce", "zamówienie", "zamówienia", "dzień", "dobry",
           "proszę", "dziękuję", "mój", "moje", "czy", "na", "że", "jak", "ale", "bardzo", "paczka", "zamówiłem"},
    "es": {"hola", "pedido", "mi", "no", "gracias", "que", "por", "favor", "el", "la", "los", "las", "es",
           "un", "una", "con", "pero", "quiero", "buenos", "días"},
    "fr": {"bonjour", "je", "ma", "mon", "commande", "pas", "merci", "est", "une", "un", "le", "les", "des",
           "vous", "pour", "mais", "avec"},
    "it": {"ciao", "ordine", "grazie", "non", "mio", "mia", "è", "sono", "il", "lo", "per", "con", "ma",
           "vorrei", "buongiorno"},
}
_SCRIPTS = {"CJK": "zh", "HIRAGANA": "ja", "KATAKANA": "ja", "HANGUL": "ko", "ARABIC": "ar",
            "HEBREW": "he", "GREEK": "el", "THAI": "th"}


def detect_language(text: str) -> str:
    """Груба оцінка мови без моделі (для fallback). Невідомо → 'uk', як у README."""
    letters = [ch for ch in text if unicodedata.category(ch).startswith("L")]
    if not letters:
        return "uk"
    cyr = sum(1 for ch in letters if "\u0400" <= ch <= "\u04ff")
    lat = sum(1 for ch in letters if ch.isascii() or "\u00c0" <= ch <= "\u024f")
    if cyr >= max(lat, 1) and cyr >= 0.5 * len(letters):
        if any(ch in _UK_LETTERS for ch in letters):
            return "uk"
        words = [w.lower() for w in _WORD.findall(text)]
        ru = sum(w in _RU_WORDS for w in words) + 3 * any(ch in _RU_LETTERS for ch in letters)
        uk = sum(w in _UK_WORDS for w in words)
        return "ru" if ru > uk else "uk"  # суржик і сумнівні випадки — uk
    if lat >= 0.5 * len(letters):
        words = [w.lower() for w in _WORD.findall(text)]
        scores = {lang: sum(w in vocab for w in words) for lang, vocab in _LATIN_WORDS.items()}
        low = text.lower()
        scores["pl"] += 3 * any(ch in low for ch in "ąćęłńśźż")
        scores["de"] += 2 * any(ch in low for ch in "äöüß")
        scores["es"] += 3 * any(ch in low for ch in "ñ¿¡")
        scores["fr"] += 1 * any(ch in low for ch in "àâçèêëîïôùûœ")
        best = max(scores, key=lambda k: scores[k])
        return best if scores[best] > 0 else "uk"
    for ch in letters:
        try:
            first = unicodedata.name(ch).split()[0]
        except ValueError:
            continue
        if first in _SCRIPTS:
            return _SCRIPTS[first]
    return "uk"
