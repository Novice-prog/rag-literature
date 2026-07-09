"""
RAG-система для ответа на вопросы с вариантами (A/B/C/D) по русской литературе.

Пайплайн:
1. По вопросу LLM выделяет ключевые слова/фразы.
2. По ним + вариантам ответа ищем релевантные фрагменты в FAISS-индексе книги.
3. LLM рассуждает по контексту и выбирает единственный вариант.

Ключ к Groq API берётся из переменной окружения GROQ_API_KEY (см. .env.example).
Перед запуском нужно построить индексы: python build_index.py
"""

import os
import re

import pandas as pd
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

MODEL_NAME = "llama3-70b-8192"
EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
VECTOR_DIR = "vectorization"

# Книги, разбитые на несколько частей при векторизации (ищем по всем частям).
MULTIPART_BOOKS = {"Война и мир": ["Война и мир 1", "Война и мир 2"]}

# Тяжёлые объекты (LLM-клиент и модель эмбеддингов) инициализируются лениво —
# чтобы import main не требовал ключа и не качал модель (важно для тестов).
_llm = None
_emb_model = None


def get_llm():
    global _llm
    if _llm is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "Не задан GROQ_API_KEY. Скопируй .env.example в .env и впиши свой ключ "
                "(получить: https://console.groq.com/keys)."
            )
        _llm = ChatGroq(api_key=api_key, model=MODEL_NAME)
    return _llm


def get_emb_model():
    global _emb_model
    if _emb_model is None:
        _emb_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return _emb_model


def load_stores(book):
    """Возвращает список FAISS-хранилищ для книги (учитывая многотомники)."""
    folders = MULTIPART_BOOKS.get(book, [book])
    return [
        FAISS.load_local(
            folder_path=os.path.join(VECTOR_DIR, folder),
            embeddings=get_emb_model(),
            allow_dangerous_deserialization=True,
        )
        for folder in folders
    ]


def get_context(book, question, answers, k=10):
    """Ищет в индексе книги фрагменты, релевантные вопросу и вариантам ответа."""
    query = get_llm().invoke(
        "Найди в вопросе ключевые слова, фразы или предложения, чтобы потом "
        f"искать по ним в книге нужные предложения для ответа:\nВопрос: {question}"
    ).content

    search_text = f"{query}, {answers['a']}, {answers['b']}, {answers['c']}, {answers['d']}"
    context_docs = []
    for store in load_stores(book):
        context_docs += store.similarity_search(
            search_text, k=k, search_kwargs={"score_threshold": 0.70}
        )
    return "\n".join(doc.page_content for doc in context_docs)


def build_prompt(question, answers, context):
    return [
        (
            "system",
            "Ты — эксперт по русской литературе. Твоя задача:\n"
            "1. Тщательно проанализировать вопрос\n"
            "2. Изучить предоставленный контекст из произведения\n"
            "3. Последовательно рассуждать, сравнивая варианты ответов\n"
            "4. Выбрать единственный правильный вариант, обосновав выбор\n\n"
            "Правила работы:\n"
            "- Если контекст противоречит вариантам ответа или не относится к вопросу — игнорируй его\n"
            "- Обязательно рассматривай все варианты перед выбором\n"
            "- Логическая цепочка рассуждений должна быть четкой и последовательной\n"
            "- Ответ должен основываться на содержании произведения и литературном анализе",
        ),
        (
            "human",
            f"Анализируемый вопрос:\n{question}\n\nКонтекст из произведения:\n{context}",
        ),
        (
            "human",
            "Требуется выполнить:\n\n"
            "1. Пошаговый анализ вопроса\n"
            "2. Сравнение каждого варианта с контекстом\n"
            "3. Критическую оценку соответствия вариантов сюжету и персонажам\n"
            "4. Логическое обоснование выбора\n\n"
            "Варианты ответов:\n"
            f"A) {answers['a']}\n"
            f"B) {answers['b']}\n"
            f"C) {answers['c']}\n"
            f"D) {answers['d']}\n\n"
            "Формат ответа:\n"
            "- Подробный анализ каждого варианта\n"
            "- Четкое обоснование выбора\n"
            "- Итоговый ответ строго в формате: [ОТВЕТ: X]",
        ),
    ]


# Соответствие буквы варианта числовому коду ответа в датасете.
LETTER_TO_CODE = {"a": 1, "b": 2, "c": 3, "d": 4}


def extract_answer(text):
    """Достаёт код ответа (1..4) из текста LLM, 0 — если не распознан.

    Требуем разделитель после «ответ» (двоеточие/тире), чтобы не путать с
    оборотами вроде «варианты ответа A», и берём последнее совпадение —
    итоговый ответ в формате [ОТВЕТ: X] стоит в конце рассуждения.
    """
    matches = re.findall(r"ответ\s*[:\-]\s*\[?\s*([abcdабвг])", text.lower())
    if not matches:
        return 0
    # Модель может отвечать латиницей (a/b/c/d) или кириллицей (а/б/в/г).
    cyr_to_lat = {"а": "a", "б": "b", "в": "c", "г": "d"}
    letter = matches[-1]
    letter = cyr_to_lat.get(letter, letter)
    return LETTER_TO_CODE.get(letter, 0)


def answer_question(book, question, answers):
    """Полный RAG-пайплайн для одного вопроса.

    answers — dict с ключами 'a','b','c','d'.
    Возвращает (код ответа 1..4 или 0, текст рассуждения LLM).
    """
    context = get_context(book, question, answers)
    response = get_llm().invoke(build_prompt(question, answers, context))
    return extract_answer(response.content), response.content


def run(questions_csv="LR2.csv", answers_csv="LR2_answer.csv", output_csv="answers.csv"):
    lab = pd.read_csv(questions_csv, index_col=0)
    final = pd.DataFrame(columns=["answer"])

    for i in range(len(lab)):
        row = lab.iloc[i]
        book = row["book"]
        question = row["question"]
        answers = {
            "a": row["answer a"],
            "b": row["answer b"],
            "c": row["answer c"],
            "d": row["answer d"],
        }

        code, reasoning = answer_question(book, question, answers)
        final.loc[i, "answer"] = code
        final.to_csv(output_csv)

        print(reasoning)
        print(i, code)

    if os.path.exists(answers_csv):
        evaluate(output_csv, answers_csv)


def evaluate(predictions_csv="answers.csv", answers_csv="LR2_answer.csv"):
    """Считает и печатает число совпадений с эталонными ответами."""
    preds = pd.read_csv(predictions_csv, index_col=0)
    truth = pd.read_csv(answers_csv, index_col=0)
    correct = sum(
        preds.loc[i, "answer"] == truth.loc[i, "answer"] for i in range(len(truth))
    )
    print(f"Верных ответов: {correct} из {len(truth)}")
    return correct


if __name__ == "__main__":
    run()
