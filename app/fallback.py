"""Правила без моделі: graceful degradation і страховка поверх відповіді моделі.

Тут три різні речі:
  * classify()   — повноцінна відповідь, коли модель так і не дала валідного JSON;
  * is_danger(), strong_human_request(), has_human_terms() — детектори, які перевіряють модель;
  * template_summary() — безпечне українське резюме, коли резюме моделі непридатне.
"""
from __future__ import annotations

import re
from typing import NamedTuple

from app import textproc

APOS = r"['’ʼ`]"


class Guess(NamedTuple):
    category: str
    priority: str
    needs_human: bool
    language: str


# ---------------------------------------------------------------- небезпека

_DANGER = re.compile(
    r"""(?ix)
    \bдим(?!ов)\w* | задим\w* | \bіскр\w* | загор\w* | займан\w* | \bпалає | \bпалаю\w* | здув\w* | здул\w* | здут\w* | роздув\w*
    | \bвибух\w* | \bвогон\w* | б['’ʼ]є\s+струмом | вдарив\w*\s+струмом | удар\w*\s+струмом
    | \bsmok(?:e|es|ed|ing|y)\b | \bsparks?\b | \bsparking\b | (?:caught|catch\w*)\s+fire | \bon\s+fire\b
    | \bswollen\b | \bbulg(?:e|es|ed|ing)\b | \bexplod\w+ | \bexploded\b | electric\s+shock | \bburning\s+smell
    | \brauch\w* | \bfunken\b | \bbrennt\b | aufgebläht
    | \bdym\w* | \biskr\w* | \bzapal\w* | \bpożar\w* | \bspuchn\w* | \bwybuch\w*
    """
)
_NEG_BEFORE = re.compile(
    r"(?:\bне\b|\bні\b|\bбез\b|\bno\b|\bnot\b|n['’]t\b|\bnie\b|\bkein\w*\b|\bnicht\b)\W{0,3}\w{0,12}\W{0,2}$",
    re.IGNORECASE,
)


_OWNER = re.compile(
    r"(?i)купив\w*|купила|придбав\w*|замовив\w*|замовила|\bмій\b|\bмоя\b|\bмоє\b|\bмої\b|\bмене\b|\bmy\b|\bbought\b|\bpurchased\b|\bordered\b|\bI\s+have\b|zamówi\w*|kupi\w*|\bmój\b|\bmoja\b|\bmeine?\b|\bgekauft\b"
)


def owns_item(text: str) -> bool:
    """Є ознака, що товар уже куплений/належить клієнту (а не питання «чи є в магазині»)."""
    return _OWNER.search(text) is not None


def is_danger(text: str) -> bool:
    """Дим, іскри, займання, здуття батареї, удар струмом — без заперечення перед словом."""
    for m in _DANGER.finditer(text):
        before = text[max(0, m.start() - 18): m.start()]
        if _NEG_BEFORE.search(before):
            continue
        return True
    return False


# ---------------------------------------------------------------- списання, яких клієнт не робив

_FRAUD = re.compile(
    r"""(?ix)
    \bне\s+я\b | \bне\s+(?:мною|робив\w*|робила|замовляв\w*|замовляла|авторизув\w*|дозволяв\w*|здійснюв\w*)\b
    | без\s+мо(?:єї|го|єю)\s+(?:згоди|відома|дозволу) | несанкціонов\w* | шахрай\w* | вкрал\w* | викрал\w*
    | \bзламали | \bзлам\s+(?:акаунт\w*|облік\w*) | акаунт\w*\s+(?:було\s+)?зламан\w* | не\s+маю\s+(?:у\s+вас\s+)?акаунт\w*
    | \bfraud\w* | unauthori[sz]ed | didn['’]?t\s+(?:order|authori[sz]e|make|buy)
    | did\s+not\s+(?:order|authori[sz]e|make|buy) | without\s+my\s+(?:consent|knowledge|permission)
    | \bhack\w* | \bstolen\b | never\s+ordered
    | nie\s+zamawia\w* | bez\s+mojej\s+zgody | oszust\w* | \bwłam\w* | ohne\s+mein\w* | betrug\w* | gehackt
    """
)
_HACK = re.compile(r"(?i)\bзламали|\bзлам\s+(?:акаунт\w*|облік\w*)|акаунт\w*\s+(?:було\s+)?зламан\w*|\bhack\w*|gehackt|\bwłam\w*|несанкціонован\w*\s+вхід|unauthori[sz]ed\s+(?:login|access)")
_MONEY = re.compile(
    r"(?i)\bспис(?:ан|ал|ую|уют|ува)\w*|картк\w*|карт[аиу]\b|гривн\w*|\bгрн\b|кошти|гроші|charged?|\bcard\b|[$€£]|\busd\b|\beur\b|\bpln\b|\bzł\b|\bbank\w*|karte|karty|abgebucht"
)

# ---------------------------------------------------------------- категорії

_PAY = re.compile(
    r"""(?ix)
    \bспис(?:ан|ал|ую|уют|ува)\w* | \bдвічі\b | оплат\w* | платіж\w* | платеж\w* | картк\w*
    | поверн\w*\s+(?:кошт\w*|грош\w*) | повернути\s+(?:кошти|гроші) | повернення\s+коштів | \bкошти\b | промокод\w* | купон\w*
    | \brefund | \bcharged?\b | \bpayment | \bpaid\b | invoice | receipt | \bpromo\w* | billing | double\s+charg\w*
    | zwrot\s+pieni\w* | płatno\w* | zapłac\w* | opłac\w* | zahlung\w* | bezahl\w* | rückerstattung | erstattung | abgebucht
    """
)
_RETURNS = re.compile(
    r"""(?ix)
    поверн\w* | обмін\w* | обміня\w* | \bбрак\w* | несправн\w* | пошкодж\w* | зламан\w* | не\s+той\s+товар | гаранті\w*
    | не\s+працю\w* | \breturn\w* | \bexchange\w* | defect\w* | \bbroken\b | damaged | faulty | wrong\s+(?:item|product)
    | malfunction | doesn['’]?t\s+work | not\s+working
    | zwróc\w* | zwrot\w* | wymian\w* | uszkodz\w* | \bwad[aąy]\b | rückgabe | zurückgeben | umtausch | defekt\w* | kaputt | beschädigt
    """
)
_DELIVERY = re.compile(
    r"""(?ix)
    доставк\w* | доставл\w* | посилк\w* | кур['’ʼ]?єр\w* | відділення | нова\s+пошта | укрпошта | \bтрек\w* | відстеж\w*
    | мало\s+прийти | не\s+прийшл\w* | в\s+дорозі | відправ\w*
    | \bshipp?\w* | \bdeliver\w* | \bparcel | \bpackage | tracking | courier | \barriv\w*
    | dostaw\w* | paczk\w* | przesył\w* | kurier | lieferung | \bpaket | versand | geliefert
    """
)
_TECH = re.compile(
    r"""(?ix)
    сайт\w* | застосун\w* | додат\w* | акаунт\w* | аккаунт\w* | \bлогін\w* | пароль | \bвхід | увійти | кабінет | кнопк\w*
    | \bapp\b | website | \blogin | log\s*in | sign\s*in | password | \baccount | \berror | помилк\w* | \bsms\b
    | strona | aplikacj\w* | \bkonto | hasło | anmeld\w* | passwort
    """
)
_PRODUCT = re.compile(
    r"""(?ix)
    наявн\w* | характеристик\w* | сумісн\w* | підійде | підходить\s+до | гарантійн\w*\s+термін
    | compatib\w* | in\s+stock | \bspecs?\b | specification\w* | dimensions | warranty\s+(?:period|term|length)
    | dostępn\w* | kompatybil\w* | verfügbar | kompatibel
    """
)
_THANKS = re.compile(r"(?i)^\W*(?:дякую|дякуємо|спасибі|вдячн\w*|thank\w*|thx|dziękuj\w*|danke|merci|gracias)")
_PROBLEM = re.compile(
    r"(?i)\bале\b|\bпроте\b|\bоднак\b|\bне\b|\bні\b|проблем\w*|\bbut\b|however|\bnot\b|problem|issue|\bnie\b|\baber\b|\bnicht\b"
)
_PURCHASE = re.compile(
    r"(?i)замовив\w*|замовила|замовлення|купив\w*|купила|придбав\w*|\bordered\b|\border\b|\bbought\b|purchased|zamówi\w*|kupi\w*|bestell\w*|gekauft|\d{4,}"
)
_CANT = re.compile(
    r"(?i)не\s+можу|не\s+вдаєтьс\w*|не\s+виходить|взагалі|неможливо|\bcannot\b|\bcan['’]?t\b|unable|impossible|nie\s+mogę|kann\s+nicht"
)
_ACTION = re.compile(
    r"(?i)увійти|\bвхід|залогін\w*|авториз\w*|оформити|оформлен\w*|оплатити|замовлен\w*|log\s*in|sign\s*in|checkout|place\s+(?:an\s+)?order|\bpay\b|zalogow\w*|anmelden"
)
_LOST = re.compile(r"(?i)загуб\w*|втрачен\w*|\blost\b|zgubi\w*|verloren|zagin\w*|не\s+знайд\w*")
_DAYS = re.compile(
    r"(?i)(\d{1,3})\s*(дн\w*|день|дні|days?|dni|dzień|tage?n?|тиж\w*|weeks?|tyg\w*|wochen?)"
)
_WEEKS_PLURAL = re.compile(r"(?i)\bтижні\b|\bтижнів\b|\bweeks\b|\btygodni\w*|\bwochen\b|\bмісяц\w*|\bmonths?\b")
_SIMPLE_REFUND_CANCEL = re.compile(r"(?i)скасован\w*")

# ---------------------------------------------------------------- потрібна людина

_HUMAN_STRONG = re.compile(
    r"""(?ix)
    # --- прохання про живу людину
    (?:живого|реального|справжнього|живу)\s+(?:оператор\w*|людин\w*|менеджер\w*|спеціаліст\w*)
    | (?:дайте|покличте|переключіть|з['’ʼ]?єднайте|підключіть|перевед\w*)\s+(?:мене\s+|мені\s+)?(?:з\s+|на\s+)?(?:оператор\w*|людин\w*|менеджер\w*|начальник\w*|керівник\w*)
    | (?:хочу|прошу|потрібно|потрібен|потрібна|вимагаю|можна)\s+(?:\S+\s+){0,2}?(?:поговорити|говорити|розмовляти|зв['’ʼ]язатися|зв['’ʼ]язатись)\s+(?:\S+\s+)?(?:з|із)\s+(?:живою\s+)?(?:людиною|оператором|менеджером|керівником|спеціалістом|консультантом)
    | (?:хочу|прошу|потрібен|потрібна|вимагаю)\s+(?:\S+\s+)?(?:оператор\w*|менеджер\w*|людин\w*)\b
    | \b(?:real|live|human)\s+(?:person|agent|operator|being|support)\b
    | (?:speak|talk|connect|transfer|escalate)\w*\s+(?:me\s+)?(?:to|with)\s+(?:a\s+|an\s+|the\s+|your\s+)?(?:human|manager|agent|person|supervisor|operator|representative)
    | (?:want|need)\s+(?:to\s+(?:speak|talk)\s+(?:to|with)\s+)?(?:a\s+|an\s+|the\s+)?(?:human|manager|supervisor|operator|representative)
    | (?:chcę|chce|proszę\s+o|potrzebuję)\s+(?:\S+\s+){0,2}?(?:człowiek\w*|konsultant\w*|menedżer\w*|kierownik\w*)
    | (?:sprechen|verbinden|reden)\s+(?:\S+\s+){0,4}?(?:mensch\w*|mitarbeiter\w*|manager\w*|vorgesetzt\w*)
    | (?:echten|richtigen)\s+(?:mensch\w*|mitarbeiter\w*)
    # --- погрози судом, поліцією, регулятором
    | (?:подам|піду|звернуся|звернусь|звернуться|буду\s+звертатися|напишу|заявлю|подаю)\s+(?:\S+\s+){0,3}?(?:суд\w*|поліці\w*|прокуратур\w*|держспоживзахист\w*|захист\w*\s+прав\w*\s+споживач\w*|адвокат\w*|юрист\w*)
    | (?:до|в|у)\s+суд(?:у|і)\b | (?:до|в|у)\s+поліці\w* | держспоживзахист\w* | захисту\s+прав\s+споживачів
    | \b(?:sue|lawsuit|attorney|lawyer|solicitor)\b | (?:call|contact|report\s+(?:you\s+)?to)\s+(?:the\s+)?police
    | (?:go|going)\s+to\s+(?:court|the\s+police)\b | consumer\s+(?:protection|rights|authority) | trading\s+standards
    | (?:pójdę|zgłoszę|złożę)\s+(?:\S+\s+){0,3}?(?:sąd\w*|policj\w*|uokik\w*|prokuratur\w*) | \bUOKiK\b | \bprawnik\w*
    | \banwalt\b | \bklage\b | polizei | verbraucherschutz | \bgericht\b
    """
)

# Широкий словник: підтверджує «true» від моделі (щоб не вірити вигаданому needs_human).
_HUMAN_TERMS = re.compile(
    r"""(?ix)
    оператор\w* | людин\w* | живо\w* | менеджер\w* | керівник\w* | начальник\w* | адвокат\w* | юрист\w* | \bсуд\w* | поліці\w* | прокуратур\w*
    | держспоживзахист\w* | споживач\w* | регулятор\w* | скарг\w* | претензі\w*
    | \bhuman\b | \bagent\b | \bmanager\b | supervisor | operator | representative | person | \bsue\b | lawsuit | lawyer | attorney
    | court | police | regulator | consumer | complain\w* | escalat\w*
    | człowiek\w* | konsultant\w* | menedżer\w* | kierownik\w* | prawnik\w* | \bsąd\w* | policj\w* | uokik | skarg\w*
    | mensch\w* | mitarbeiter\w* | vorgesetzt\w* | anwalt | klage | gericht | polizei | beschwerde
    """
)
KNOWN_LANGS = {"uk", "ru", "en", "pl", "de"}  # мови, на які розрахований словник вище


def strong_human_request(text: str) -> bool:
    """Клієнт прямо просить живу людину/менеджера або погрожує судом/поліцією/регулятором."""
    return _HUMAN_STRONG.search(text) is not None


def has_human_terms(text: str) -> bool:
    return _HUMAN_TERMS.search(text) is not None


# ---------------------------------------------------------------- повна евристична відповідь

def _days_over_week(text: str) -> bool:
    for m in _DAYS.finditer(text):
        n, unit = int(m.group(1)), m.group(2).lower()
        if unit.startswith(("тиж", "week", "tyg", "woche")):
            n *= 7
        if n > 7:
            return True
    return bool(_WEEKS_PLURAL.search(text))


def classify(text: str) -> Guess:
    """Правила замість моделі. Очікує текст уже без атак (див. textproc.strip_injection)."""
    lang = textproc.detect_language(text)
    if not textproc.has_letters(text):
        return Guess("other", "low", False, lang)

    human = strong_human_request(text)
    low = text.lower()

    if is_danger(text):
        return Guess("returns", "urgent", True, lang)

    if _FRAUD.search(text):
        if _HACK.search(text) and not _MONEY.search(text):
            return Guess("technical", "urgent", True, lang)
        return Guess("payment", "urgent", True, lang)

    if _THANKS.search(text) and not _PROBLEM.search(text):
        return Guess("other", "low", human, lang)

    purchase = bool(_PURCHASE.search(text))

    if _PAY.search(text):
        high = bool(re.search(r"(?i)двічі|double|twice|podwójn|doppelt|двічі\s+спис", text))
        pri = "high" if high else ("medium" if purchase else "low")
        return Guess("payment", pri, human, lang)

    if _RETURNS.search(text):
        return Guess("returns", "medium" if purchase else "low", human, lang)

    if _DELIVERY.search(text):
        high = bool(_LOST.search(text)) or _days_over_week(text)
        pri = "high" if high else ("medium" if purchase else "low")
        return Guess("delivery", pri, human, lang)

    if _TECH.search(text):
        blocked = bool(_CANT.search(text) and _ACTION.search(text))
        return Guess("technical", "high" if blocked else "medium", human, lang)

    if _PRODUCT.search(text):
        return Guess("product", "low", human, lang)

    return Guess("other", "low", human, lang)


# ---------------------------------------------------------------- шаблонні резюме (українською)

_TEMPLATES = {
    "delivery": "Клієнт звертається щодо доставки{order}.",
    "payment": "Клієнт звертається щодо оплати чи повернення коштів{order}.",
    "returns": "Клієнт звертається щодо повернення, обміну або несправності товару{order}.",
    "technical": "Клієнт повідомляє про технічну проблему із сайтом чи акаунтом.",
    "product": "Клієнт ставить питання про товар до покупки.",
    "other": "Звернення без конкретного запиту до підтримки.",
}
EMPTY_SUMMARY = "Порожнє або незрозуміле повідомлення без тексту."


def template_summary(category: str, text: str = "") -> str:
    """Безпечне резюме українською: без жодного тексту від клієнта, окрім номера замовлення."""
    num = textproc.find_order_number(text) if text else None
    order = f" за замовленням {num}" if num and category in ("delivery", "payment", "returns") else ""
    return _TEMPLATES.get(category, _TEMPLATES["other"]).format(order=order)
