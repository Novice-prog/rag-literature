"""Тесты FastAPI-эндпоинтов с замоканным пайплайном (без реального LLM)."""

from fastapi.testclient import TestClient

from rag_literature import api as app_module

client = TestClient(app_module.app)

VALID_PAYLOAD = {
    "book": "Роковые яйца",
    "question": "Что украли у профессора Персикова?",
    "a": "Его красный луч",
    "b": "Драгоценности",
    "c": "Лабораторные записи",
    "d": "Очки",
}


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_ask_ok(monkeypatch):
    monkeypatch.setattr(
        app_module, "answer_question", lambda b, q, a: (1, "рассуждение [ОТВЕТ: A]")
    )
    resp = client.post("/ask", json=VALID_PAYLOAD)
    assert resp.status_code == 200
    body = resp.json()
    assert body["code"] == 1
    assert body["letter"] == "A"
    assert "рассуждение" in body["reasoning"]


def test_ask_unrecognized_answer(monkeypatch):
    monkeypatch.setattr(app_module, "answer_question", lambda b, q, a: (0, "без метки"))
    resp = client.post("/ask", json=VALID_PAYLOAD)
    assert resp.status_code == 200
    body = resp.json()
    assert body["code"] == 0
    assert body["letter"] is None


def test_ask_passes_all_options(monkeypatch):
    captured = {}

    def fake(book, question, answers):
        captured["answers"] = answers
        return 2, "ok"

    monkeypatch.setattr(app_module, "answer_question", fake)
    client.post("/ask", json=VALID_PAYLOAD)
    assert captured["answers"] == {
        "a": "Его красный луч",
        "b": "Драгоценности",
        "c": "Лабораторные записи",
        "d": "Очки",
    }


def test_ask_validation_error():
    payload = dict(VALID_PAYLOAD)
    del payload["d"]
    resp = client.post("/ask", json=payload)
    assert resp.status_code == 422


def test_ask_missing_index_returns_404(monkeypatch):
    def boom(book, question, answers):
        raise FileNotFoundError("нет индекса")

    monkeypatch.setattr(app_module, "answer_question", boom)
    resp = client.post("/ask", json=VALID_PAYLOAD)
    assert resp.status_code == 404
    assert "Роковые яйца" in resp.json()["detail"]
