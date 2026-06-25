import app.main as app_main
from app.config import settings
from app.main import SlidingWindowRateLimiter
from tests.conftest import RAG_ANSWER, SUMMARY_TEXT, make_pdf


def upload(name: str, data: bytes, content_type: str = "application/octet-stream"):
    return {"file": (name, data, content_type)}


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_analyze_text_document(client):
    response = client.post(
        "/documents/analyze",
        files=upload("report.txt", b"Quarterly revenue grew by 12%.", "text/plain"),
    )
    assert response.status_code == 200
    body = response.json()
    assert body["filename"] == "report.txt"
    assert body["classification"]["category"] == "report"
    assert body["summary"] == SUMMARY_TEXT
    assert body["extracted_data"]["title"] == "Quarterly Report"


def test_analyze_pdf_document(client):
    pdf = make_pdf("Quarterly revenue grew by 12 percent")
    response = client.post(
        "/documents/analyze", files=upload("report.pdf", pdf, "application/pdf")
    )
    assert response.status_code == 200
    assert response.json()["classification"]["category"] == "report"


def test_classify_endpoint(client):
    response = client.post(
        "/documents/classify", files=upload("doc.txt", b"some text", "text/plain")
    )
    assert response.status_code == 200
    assert response.json()["classification"]["confidence"] == 0.92


def test_summarize_endpoint(client):
    response = client.post(
        "/documents/summarize", files=upload("doc.txt", b"some text", "text/plain")
    )
    assert response.status_code == 200
    assert response.json()["summary"] == SUMMARY_TEXT


def test_extract_endpoint(client):
    response = client.post(
        "/documents/extract", files=upload("doc.txt", b"some text", "text/plain")
    )
    assert response.status_code == 200
    assert response.json()["extracted_data"]["author"] == "Jane Doe"


def test_empty_document_rejected(client):
    response = client.post(
        "/documents/analyze", files=upload("empty.txt", b"   ", "text/plain")
    )
    assert response.status_code == 422


def test_rag_ingest_and_query(client):
    response = client.post(
        "/rag/ingest",
        files=upload("report.txt", b"Revenue grew by 12 percent in Q1.", "text/plain"),
    )
    assert response.status_code == 200
    assert response.json()["chunks_indexed"] >= 1

    response = client.post(
        "/rag/query", json={"question": "How much did revenue grow?", "top_k": 1}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == RAG_ANSWER
    assert len(body["sources"]) == 1
    assert body["sources"][0]["metadata"]["source"] == "report.txt"


def test_rag_query_before_ingest_conflict(client):
    response = client.post("/rag/query", json={"question": "anything"})
    assert response.status_code == 409


def test_providers_endpoint(client):
    response = client.get("/providers")
    assert response.status_code == 200
    body = response.json()
    assert body["providers"] == ["anthropic"]
    assert body["default"] == "anthropic"


def test_rag_query_with_explicit_available_provider(client):
    client.post(
        "/rag/ingest",
        files=upload("report.txt", b"Revenue grew by 12 percent in Q1.", "text/plain"),
    )
    response = client.post(
        "/rag/query",
        json={"question": "How much did revenue grow?", "top_k": 1, "provider": "anthropic"},
    )
    assert response.status_code == 200
    assert response.json()["answer"] == RAG_ANSWER


def test_rag_query_unavailable_provider_returns_400(client):
    client.post(
        "/rag/ingest",
        files=upload("report.txt", b"Revenue grew by 12 percent in Q1.", "text/plain"),
    )
    response = client.post(
        "/rag/query",
        json={"question": "anything", "provider": "openai"},
    )
    assert response.status_code == 400


def test_rag_demo_mode_queries_seeded_index(client, rag_demo_service):
    from app.services.documents import chunk_text

    rag_demo_service.ingest(
        chunk_text(
            "Amin Skenderi studies computer engineering at TU Berlin.",
            filename="cv.pdf",
        )
    )
    response = client.post(
        "/rag/query",
        json={"question": "What does Amin study?", "mode": "demo"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == RAG_ANSWER
    assert body["sources"][0]["metadata"]["source"] == "cv.pdf"


def test_rag_demo_and_upload_indexes_are_separate(client):
    # A document uploaded to the upload index must not be visible in demo mode.
    client.post(
        "/rag/ingest",
        files=upload("report.txt", b"Revenue grew by 12 percent in Q1.", "text/plain"),
    )
    response = client.post(
        "/rag/query", json={"question": "anything", "mode": "demo"}
    )
    assert response.status_code == 409


def test_rag_query_validation(client):
    response = client.post("/rag/query", json={"question": ""})
    assert response.status_code == 422


def test_oversized_upload_rejected(client, monkeypatch):
    monkeypatch.setattr(settings, "max_upload_bytes", 100)
    response = client.post(
        "/documents/summarize", files=upload("big.txt", b"x" * 200, "text/plain")
    )
    assert response.status_code == 413


def test_rate_limit_kicks_in(client, monkeypatch):
    monkeypatch.setattr(settings, "rate_limit_per_minute", 2)
    monkeypatch.setattr(app_main, "rate_limiter", SlidingWindowRateLimiter())
    for _ in range(2):
        response = client.post(
            "/documents/summarize", files=upload("doc.txt", b"some text", "text/plain")
        )
        assert response.status_code == 200
    response = client.post(
        "/documents/summarize", files=upload("doc.txt", b"some text", "text/plain")
    )
    assert response.status_code == 429
    # GET endpoints like /health stay unaffected
    assert client.get("/health").status_code == 200


def test_cors_preflight_for_site_origin(client):
    response = client.options(
        "/rag/query",
        headers={
            "Origin": "https://aminskenderi.me",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "https://aminskenderi.me"
