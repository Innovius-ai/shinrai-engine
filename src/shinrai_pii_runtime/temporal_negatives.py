"""Deterministic DATE / AGE negatives for records whose corpus scope leaves the two heads ``unknown`` (2026-09-19).

Problem (G5 readout §3, expansion-v1 README): every legacy source declares DATE / AGE ``unknown``, so the two new heads
never see a general-text negative. They then fire on proper nouns, codenames, house numbers and versions (DATE precision
37-42 % on chat text; 11 of 115 a4p STREET spans broken by a DATE hit).

Rule (``training.objective.temporal_negatives: true``): for a record whose resolved DATE (or AGE) state is ``unknown``,
the head becomes ``complete`` for THIS record, with unknown regions over everything that could be a date or an age:

  DATE  every DATE-SHAPED substring: a numeric date (12.03.1998, 3/4/98, 2016-03-12), a partial numeric date
        (03/2016, 9.10.), a year 1500-2099, a month name of the 15 fleet languages with its adjacent number, and the
        CJK / RTL date forms - each widened by ``PAD`` characters;
  AGE   every 1-3 digit number or decade word within ``AGE_PAD`` characters of an age cue of the record's language
        (year / Jahr / ano / лет / 歳 / سنة / שנה ..., "old", "aged"), and the cue itself.

  v2 (2026-09-20, after G7): a digit run that is NOT date- or age-shaped is an O target, not an unknown region. v1
  masked every digit run, so firing DATE on a phone number or a house number stayed unpenalised: G7 lost the inline
  PHONE frame (141 -> 110 of 150 whole numbers) and chat STREET precision (95 -> 85) against G6 while gaining DATE
  precision. v2 keeps the gain and makes non-date digits negatives.

Characters inside a labelled entity of another head are trusted negatives (a PHONE, STREET, POSTAL_CODE, CUSTOMER_ID or
DOB span is by policy never a DATE / AGE) and are cut out of the regions. Everything else - words, names, codenames -
becomes an O target. The rule only ever ADDS O targets on text that carries no digit, month or age cue nearby; it never
creates a positive and never touches a head that is already complete / positives_only.
"""
from __future__ import annotations

import hashlib
import re

VERSION = "temporal-negatives-v2.1"  # 2.1 (2026-09-22): month words need a left word boundary («marschiert», «decimal» no longer mask)
FRACTION_VERSION = "dose-v1"
PAD = 4          # covers «3rd», «12.», «de», «of» glue around a digit run or month word
AGE_PAD = (18, 6)  # number words stand BEFORE the unit («twenty-five years», «mitte dreißig»)
HEADS = ("DATE", "AGE")

_MONTHS = (
    # en de fr es it pt pl (stems, case-insensitive, word-initial)
    "janu jan. feb febr mar mär märz apr may mai maj jun jul aug sep sept oct okt nov dec dez déc "
    "janv févr fevr mars avr juin juil août aout "
    "ener enero febrero marzo abril mayo junio julio agosto septiembre setiembre octubre noviembre diciembre "
    "genn gennaio febbraio aprile maggio giugno luglio settembre ottobre novembre dicembre "
    "janeiro fevereiro março marco abril maio junho julho setembro outubro novembro dezembro "
    "stycz luty lutego marzec marca kwie czerw lipiec lipca sierp wrze paźdz pazdz listopad grud "
    # tr
    "ocak şubat subat mart nisan mayıs mayis haziran temmuz ağustos agustos eylül eylul ekim kasım kasim aralık aralik "
    # ru uk
    "январ феврал март апрел мая май июн июл август сентябр октябр ноябр декабр "
    "січ лют берез квіт трав черв лип серп верес жовт листопад груд "
).split()
_MONTH_RX = re.compile(r"(?i)(?<![^\W\d_])(?:" + "|".join(sorted({re.escape(m) for m in _MONTHS}, key=len, reverse=True)) + r")\w*")
_MONTH_NOSPACE = re.compile(  # ar he ja ko: month names / date characters without Latin word boundaries
    "يناير|فبراير|مارس|أبريل|ابريل|مايو|يونيو|يوليو|أغسطس|اغسطس|سبتمبر|أكتوبر|اكتوبر|نوفمبر|ديسمبر|كانون|شباط|آذار|نيسان|أيار|حزيران|تموز|آب|أيلول|تشرين"
    "|ינואר|פברואר|מרץ|מרס|אפריל|מאי|יוני|יולי|אוגוסט|ספטמבר|אוקטובר|נובמבר|דצמבר"
    "|[〇一二三四五六七八九十百]+\\s*[年月日歳才]|[年月日]|년|월|일")
# explicit whole-word month names and abbreviations of the ten Latin/Cyrillic fleet languages (v2.1: the stem+\w* form
# masked «marschiert», «decimal», «Marke» as months and cost negatives)
_MONTH_FULL = r"(?:październiku|października|październik|listopadzie|septiembre|diciembre|fevereiro|листопада|listopada|september|settembre|setiembre|листопаді|septembre|noviembre|dezembro|листопад|february|sierpień|wrzesień|kwietnia|czerwiec|grudzień|novembre|березень|dicembre|сентября|kwiecień|novembro|december|dezember|sierpniu|сентябрь|stycznia|decembre|listopad|febbraio|kwietniu|września|november|setembro|wrześniu|sierpnia|вересень|styczniu|décembre|сентябре|czerwcu|березні|ağustos|octobre|janvier|октября|styczeń|декабрь|декабре|августа|вересня|февраля|октябре|februar|octubre|czerwca|oktober|january|février|juillet|gennaio|февраль|fevrier|вересні|febrero|грудень|grudnia|березня|ottobre|janeiro|жовтень|октябрь|october|grudniu|agustos|травень|outubro|квітень|haziran|декабря|феврале|серпень|червень|августе|январь|января|temmuz|maggio|лютому|giugno|aralik|aprile|жовтня|ноябре|august|lipiec|januar|червня|ноябрь|лютого|апреле|январе|marzec|травні|август|грудня|luglio|жовтні|грудні|квітня|lutego|липень|agosto|ноября|серпні|апрель|апреля|квітні|серпня|aralık|червні|травня|січень|julio|şubat|lipca|липня|mayıs|april|mayis|lipcu|marco|enero|lutym|eylül|abril|march|kasim|marzo|nisan|subat|січня|марта|лютий|марте|junho|січні|março|junio|marcu|липні|kasım|eylul|avril|marca|julho|ekim|июль|july|juli|maju|июле|июнь|maio|aout|june|июня|июне|март|août|luty|maja|mayo|märz|juin|mars|ocak|mart|juni|июля|май|мае|may|мая|mai|maj)"
# abbreviations collide across languages (pl «sie» = de «Sie», es «mar» = sea): they count only with a period
# or beside a day number
_MONTH_ABBR = r"(?:févr|fevr|sept|janv|сент|juil|ago|cze|ott|сен|déc|wrz|lut|jul|mag|apr|giu|dec|set|авг|oct|фев|fev|окт|lip|дек|мар|dez|ene|янв|lis|avr|paź|sty|aug|июн|sie|out|апр|abr|nov|feb|okt|gru|mar|lug|gen|июл|kwi|ноя|jan|dic|sep|jun|mär)"
_MONTH_WORD = r"(?:" + _MONTH_FULL + r"|" + _MONTH_ABBR + r")\.?"          # day-adjacent forms
_MONTH_BARE = r"(?:" + _MONTH_FULL + r"|" + _MONTH_ABBR + r"(?=\.))\.?"     # a month standing alone
_DATE_SHAPE = re.compile(
    r"(?i)\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b"          # 12.03.1998, 3/4/98
    r"|\b\d{4}-\d{1,2}-\d{1,2}\b"                        # 2016-03-12
    r"|\b\d{1,2}[./]\d{4}\b|\b\d{4}[./]\d{1,2}\b"      # 03/2016, 2016/03
    r"|\b\d{1,2}\.\s?\d{1,2}\.(?!\d)"                   # 9.10.  (day-month, no year)
    r"|(?<![\w-])(?:1[5-9]\d{2}|20\d{2})(?![\w-])"           # a year 1500-2099, but not inside «INC-2023-4567»
    r"|\b\d{1,2}(?:st|nd|rd|th|\.)?\s*(?:de\s|of\s|d[ei]\s)?(?<![^\W\d_])(?:" + _MONTH_WORD + r")(?![^\W\d_])"   # 15 May, 12. Oktober, 12 de marzo
    r"|(?<![^\W\d_])(?:" + _MONTH_WORD + r")(?![^\W\d_])\s*,?\s*\d{1,2}(?:st|nd|rd|th)?\b"                           # May 15
    r"|(?<![^\W\d_])" + _MONTH_BARE + r"(?![^\W\d_])")                                                             # a bare month name
_DATE_NOSPACE = re.compile(
    "يناير|فبراير|مارس|أبريل|ابريل|مايو|يونيو|يوليو|أغسطس|اغسطس|سبتمبر|أكتوبر|اكتوبر|نوفمبر|ديسمبر|كانون|شباط|آذار|نيسان|أيار|حزيران|تموز|أيلول|تشرين"
    "|ינואר|פברואר|מרץ|מרס|אפריל|מאי|יוני|יולי|אוגוסט|ספטמבר|אוקטובר|נובמבר|דצמבר"
    "|[〇一二三四五六七八九十百\\d]+\\s*[年月日]|\\d+\\s*[년월일]")
_SMALL_NUM = re.compile(r"(?<![\d.,:/-])\d{1,3}(?![\d.,:/-])")

_AGE_WORDS = {  # per language: short cue words collide across languages («an» de/fr, «alt», «mes», «lat», «eta»)
    "en": r"years?|yrs?|y/o|yo|old|aged?|months?|weeks?|teen\w*|twenties|thirties|forties|fifties|sixties|seventies|eighties|nineties",
    "de": r"jahr\w*|jähr\w*|monat\w*|woche\w*|alt|alter|zwanzig\w*|dreißig\w*|dreissig\w*|vierzig\w*|fünfzig\w*|sechzig\w*|siebzig\w*|achtzig\w*|neunzig\w*",
    "fr": r"ans?|âgée?s?|agée?s?|mois|semaines?|vingtaine|trentaine|quarantaine|cinquantaine|soixantaine",
    "es": r"años?|edad|meses|mes|semanas?|\w+genari[oa]s?|veinteañer\w*|treintañer\w*",
    "it": r"anni|anno|età|eta|mesi|mese|settimane|\w+enne|\w+genari[oa]",
    "pt": r"anos?|idade|meses|mês|mes|semanas?|\w+genári[oa]s?",
    "pl": r"lat|lata|roku?|wiek\w*|miesi\w*|tygodn\w*|\w+latek|\w+letni\w*",
    "tr": r"yaş\w*|yas\w*|aylık|aylik|haftalık",
    "ru": r"лет|год\w*|возраст\w*|месяц\w*|недел\w*|\w+летн\w*",
    "uk": r"рок\w*|років|вік\w*|місяц\w*|тижн\w*|\w+річн\w*",
}
_AGE_NOSPACE = re.compile("歳|才|か月|ヶ月|週間|세|살|개월|سنة|سنوات|عام|عاما|عمر\\w*|شهر|أشهر|اشهر|שנה|שנים|בן|בת|גיל|חודש\\w*".replace("\\\\", "\\"))
_AGE_RX = {k: re.compile(r"(?i)(?<![^\W\d_])(?:" + v + r")(?![^\W\d_])") for k, v in _AGE_WORDS.items()}


def _age_cues(text: str, lang: str | None):
    base = (lang or "").split("-")[0].lower()
    rxs = [_AGE_RX[base]] if base in _AGE_RX else list(_AGE_RX.values())   # unknown language: every list (conservative)
    return [(m.start(), m.end()) for rx in (*rxs, _AGE_NOSPACE) for m in rx.finditer(text)]

def _merge(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for s, e in sorted(spans):
        if out and s <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], e))
        else:
            out.append((s, e))
    return out


def _cut(spans: list[tuple[int, int]], holes: list[tuple[int, int]]) -> list[tuple[int, int]]:
    out = []
    for s, e in spans:
        cur = s
        for hs, he in holes:
            if he <= cur or hs >= e:
                continue
            if hs > cur:
                out.append((cur, hs))
            cur = max(cur, he)
        if cur < e:
            out.append((cur, e))
    return [(s, e) for s, e in out if e > s]


def temporal_unknown_regions(text: str, entities: list[dict], lang: str | None = None) -> dict[str, list[tuple[int, int]]]:
    """Unknown regions per head for a record that has NO trusted DATE / AGE annotation."""
    n = len(text)
    dates = [(m.start(), m.end()) for rx in (_DATE_SHAPE, _DATE_NOSPACE) for m in rx.finditer(text)]
    cues = _age_cues(text, lang)
    # an age is a small number (or a decade word) near a cue of this language; the cue itself stays unknown too
    # the cue keeps a small left margin so a preceding modifier ("mitte dreißig", "early forties",
    # "Anfang 30") is not split into an O target and an unknown region
    ages = [(max(0, cs - 10), min(n, ce + 4)) for cs, ce in cues]
    # every STANDALONE 1-3 digit number stays unknown for AGE: a bare age has no cue («Helen (47)», «Alter: 72»,
    # «turning 40»). Long digit runs (phones, ids, postal codes) and decimals are excluded by the pattern, so they
    # stay O targets - which is what repairs the v1 PHONE / STREET regression.
    ages += [(m.start(), m.end()) for m in _SMALL_NUM.finditer(text)]
    holes = _merge([tuple(e["span"]) for e in entities if e.get("type") not in HEADS])
    def pad(spans, left, right=None):
        right = left if right is None else right
        return [(max(0, s - left), min(n, e + right)) for s, e in spans]

    return {
        "DATE": _cut(_merge(pad(dates, PAD)), holes),
        "AGE": _cut(_merge(ages), holes),
    }


def record_in_dose(record_id: str, fraction: float) -> bool:
    """Deterministic, order-independent membership for ``training.objective.temporal_negatives_fraction``.

    Measured on G7 / G8 (readout 15639415 §3): the rule lifts the DATE / AGE share of all supervised targets from
    2.4 % to ~9 %, and the inline PHONE frame, gold-v2 ja / ko and DATE recall all track that share. The dose knob
    applies the rule to a stable subset of the eligible records so the share can be placed between the two points
    without changing what the rule masks. 1.0 = every eligible record (the G7 / G8 behaviour).
    """
    if fraction >= 1.0:
        return True
    if fraction <= 0.0:
        return False
    h = hashlib.blake2b(str(record_id).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(h, "big") / 2**64 < fraction
