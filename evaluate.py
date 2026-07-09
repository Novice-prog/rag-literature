"""
Оценка точности: сравнивает предсказанные ответы с эталонными.

Пример:  python evaluate.py
"""

import sys

import pandas as pd


def evaluate(predictions_csv="answers.csv", answers_csv="LR2_answer.csv"):
    preds = pd.read_csv(predictions_csv, index_col=0)
    truth = pd.read_csv(answers_csv, index_col=0)
    correct = sum(
        preds.loc[i, "answer"] == truth.loc[i, "answer"] for i in range(len(truth))
    )
    print(f"Верных ответов: {correct} из {len(truth)}")
    return correct


if __name__ == "__main__":
    pred = sys.argv[1] if len(sys.argv) > 1 else "answers.csv"
    truth = sys.argv[2] if len(sys.argv) > 2 else "LR2_answer.csv"
    evaluate(pred, truth)
