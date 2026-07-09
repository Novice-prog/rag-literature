"""Подсчёт точности: сравнение предсказанных ответов с эталонными."""

import pandas as pd

from . import config


def evaluate(
    predictions_csv: str = config.OUTPUT_CSV,
    answers_csv: str = config.ANSWERS_CSV,
) -> int:
    """Считает и печатает число совпадений с эталонными ответами."""
    preds = pd.read_csv(predictions_csv, index_col=0)
    truth = pd.read_csv(answers_csv, index_col=0)
    correct = sum(
        preds.loc[i, "answer"] == truth.loc[i, "answer"] for i in range(len(truth))
    )
    print(f"Верных ответов: {correct} из {len(truth)}")
    return correct
