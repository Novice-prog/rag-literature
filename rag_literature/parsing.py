"""Разбор итогового ответа LLM в числовой код варианта."""

import re

# Соответствие буквы варианта числовому коду ответа в датасете.
LETTER_TO_CODE = {"a": 1, "b": 2, "c": 3, "d": 4}
CODE_TO_LETTER = {1: "A", 2: "B", 3: "C", 4: "D"}

# Модель может отвечать кириллицей вместо латиницы.
_CYR_TO_LAT = {"а": "a", "б": "b", "в": "c", "г": "d"}

_ANSWER_RE = re.compile(r"ответ\s*[:\-]\s*\[?\s*([abcdабвг])")


def extract_answer(text: str) -> int:
    """Достаёт код ответа (1..4) из текста LLM, 0 — если не распознан.

    Требуем разделитель после «ответ» (двоеточие/тире), чтобы не путать с
    оборотами вроде «варианты ответа A», и берём последнее совпадение —
    итоговый ответ в формате [ОТВЕТ: X] стоит в конце рассуждения.
    """
    matches = _ANSWER_RE.findall(text.lower())
    if not matches:
        return 0
    letter = matches[-1]
    letter = _CYR_TO_LAT.get(letter, letter)
    return LETTER_TO_CODE.get(letter, 0)
