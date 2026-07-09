"""Оркестрация RAG-пайплайна: один вопрос и прогон целого датасета."""

import pandas as pd

from . import config
from .parsing import extract_answer
from .prompts import answer_prompt
from .providers import get_llm
from .retrieval import get_context


def answer_question(book: str, question: str, answers: dict) -> tuple:
    """Полный RAG-пайплайн для одного вопроса.

    answers — dict с ключами 'a','b','c','d'.
    Возвращает (код ответа 1..4 или 0, текст рассуждения LLM).
    """
    context = get_context(book, question, answers)
    response = get_llm().invoke(answer_prompt(question, answers, context))
    return extract_answer(response.content), response.content


def answer_dataset(
    questions_csv: str = config.QUESTIONS_CSV,
    output_csv: str = config.OUTPUT_CSV,
) -> pd.DataFrame:
    """Прогоняет все вопросы из CSV, инкрементально сохраняя ответы."""
    lab = pd.read_csv(questions_csv, index_col=0)
    final = pd.DataFrame(columns=["answer"])

    for i in range(len(lab)):
        row = lab.iloc[i]
        answers = {
            "a": row["answer a"],
            "b": row["answer b"],
            "c": row["answer c"],
            "d": row["answer d"],
        }
        code, reasoning = answer_question(row["book"], row["question"], answers)
        final.loc[i, "answer"] = code
        final.to_csv(output_csv)

        print(reasoning)
        print(i, code)

    return final
