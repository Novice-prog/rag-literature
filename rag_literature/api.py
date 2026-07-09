"""FastAPI-сервис над RAG-пайплайном.

Запуск:  uvicorn rag_literature.api:app --reload
"""

from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .parsing import CODE_TO_LETTER
from .pipeline import answer_question

app = FastAPI(
    title="RAG QA по русской литературе",
    description="Отвечает на вопросы с вариантами A/B/C/D, используя RAG над книгами.",
    version="1.0.0",
)


class AskRequest(BaseModel):
    book: str = Field(..., description="Название книги, как в индексе vectorization/")
    question: str = Field(..., description="Текст вопроса")
    a: str = Field(..., description="Вариант ответа A")
    b: str = Field(..., description="Вариант ответа B")
    c: str = Field(..., description="Вариант ответа C")
    d: str = Field(..., description="Вариант ответа D")

    model_config = {
        "json_schema_extra": {
            "example": {
                "book": "Роковые яйца",
                "question": "Что украли у профессора Персикова?",
                "a": "Его красный луч",
                "b": "Драгоценности",
                "c": "Лабораторные записи",
                "d": "Очки",
            }
        }
    }


class AskResponse(BaseModel):
    code: int = Field(..., description="Код ответа: 1=A, 2=B, 3=C, 4=D, 0=не распознан")
    letter: Optional[str] = Field(..., description="Буква варианта или null, если не распознан")
    reasoning: str = Field(..., description="Рассуждение модели")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    answers = {"a": req.a, "b": req.b, "c": req.c, "d": req.d}
    try:
        code, reasoning = answer_question(req.book, req.question, answers)
    except (FileNotFoundError, RuntimeError) as exc:
        raise HTTPException(
            status_code=404,
            detail=f"Не удалось обработать вопрос по книге «{req.book}»: {exc}",
        )
    return AskResponse(code=code, letter=CODE_TO_LETTER.get(code), reasoning=reasoning)
