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


def test_rag_query_validation(client):
    response = client.post("/rag/query", json={"question": ""})
    assert response.status_code == 422
