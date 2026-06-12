from app.schemas import ClassificationResult, ExtractionResult
from tests.conftest import RAG_ANSWER, SUMMARY_TEXT


async def test_classify(analysis_service):
    result = await analysis_service.classify("Quarterly revenue grew by 12%.")
    assert isinstance(result, ClassificationResult)
    assert result.category == "report"
    assert result.confidence == 0.92


async def test_summarize(analysis_service):
    summary = await analysis_service.summarize("Quarterly revenue grew by 12%.")
    assert summary == SUMMARY_TEXT


async def test_extract(analysis_service):
    result = await analysis_service.extract("Quarterly revenue grew by 12%.")
    assert isinstance(result, ExtractionResult)
    assert result.title == "Quarterly Report"
    assert result.author == "Jane Doe"
    assert "Acme Corp" in result.key_entities


async def test_analyze_runs_all_three(analysis_service):
    classification, summary, extracted = await analysis_service.analyze("Some text.")
    assert classification.category == "report"
    assert summary == SUMMARY_TEXT
    assert extracted.language == "en"
    # concurrent execution must not cross-wire the chain responses
    assert summary != RAG_ANSWER


async def test_input_is_truncated(fake_llm):
    from app.services.analysis import AnalysisService

    service = AnalysisService(fake_llm, max_input_chars=10)
    assert service._truncate("x" * 100) == "x" * 10
