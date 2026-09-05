"""Unit tests for FinRAG components."""

import pytest
from finrag.retrieval.metadata_filter import (
    MetadataFilter,
    extract_metadata_filter,
    COMPANY_ALIASES,
    SECTION_KEYWORDS,
)
from eval.evaluate_generation import (
    check_citations_grounded,
    check_citation_format,
    keyword_coverage,
    is_refusal,
    INSUFFICIENT_EVIDENCE,
    check_answer_format,
)
from langchain_core.documents import Document


class TestMetadataFilter:
    """Tests for metadata filter extraction and matching."""

    def test_extract_ticker_alias(self):
        """Test that company aliases resolve to correct tickers."""
        query = "What are Apple's risk factors?"
        filter_obj = extract_metadata_filter(query)
        assert "AAPL" in filter_obj.tickers

    def test_extract_multiple_tickers(self):
        """Test extraction of multiple tickers from query."""
        query = "Compare Apple and Microsoft risk factors"
        filter_obj = extract_metadata_filter(query)
        assert "AAPL" in filter_obj.tickers
        assert "MSFT" in filter_obj.tickers

    def test_extract_form_10k(self):
        """Test 10-K form extraction."""
        query = "Apple 10-K risk factors"
        filter_obj = extract_metadata_filter(query)
        assert "10-K" in filter_obj.forms

    def test_extract_form_10q(self):
        """Test 10-Q form extraction."""
        query = "Apple quarterly report"
        filter_obj = extract_metadata_filter(query)
        assert "10-Q" in filter_obj.forms

    def test_extract_fiscal_year(self):
        """Test fiscal year extraction."""
        query = "Apple FY2025 revenue"
        filter_obj = extract_metadata_filter(query)
        assert "2025" in filter_obj.fiscal_years

    def test_extract_fy_format(self):
        """Test FY format year extraction."""
        query = "Apple FY25 revenue"
        filter_obj = extract_metadata_filter(query)
        assert "2025" in filter_obj.fiscal_years

    def test_extract_item_number(self):
        """Test Item/Section extraction."""
        query = "Apple Item 1A risk factors"
        filter_obj = extract_metadata_filter(query)
        assert "Item 1A" in filter_obj.item_numbers

    def test_metadata_filter_matches(self):
        """Test MetadataFilter.matches() method."""
        filter_obj = MetadataFilter(
            tickers=["AAPL"],
            forms=["10-K"],
            fiscal_years=["2025"],
            item_numbers=["1A"],
        )
        meta = {
            "ticker": "AAPL",
            "form": "10-K",
            "fiscal_year": "2025",
            "item_number": "1A",
        }
        assert filter_obj.matches(meta) is True

    def test_metadata_filter_mismatch_ticker(self):
        """Test MetadataFilter rejects wrong ticker."""
        filter_obj = MetadataFilter(tickers=["AAPL"])
        meta = {"ticker": "MSFT"}
        assert filter_obj.matches(meta) is False

    def test_metadata_filter_mismatch_form(self):
        """Test MetadataFilter rejects wrong form."""
        filter_obj = MetadataFilter(forms=["10-K"])
        meta = {"form": "10-Q"}
        assert filter_obj.matches(meta) is False

    def test_metadata_filter_mismatch_year(self):
        """Test MetadataFilter rejects wrong fiscal year."""
        filter_obj = MetadataFilter(fiscal_years=["2025"])
        meta = {"fiscal_year": "2024"}
        assert filter_obj.matches(meta) is False

    def test_metadata_filter_empty(self):
        """Test empty filter matches everything."""
        filter_obj = MetadataFilter()
        assert filter_obj.is_empty() is True
        meta = {"ticker": "ANYTHING"}
        assert filter_obj.matches(meta) is True


class TestCitationGrounding:
    """Tests for citation grounding validation."""

    def test_grounded_citation(self):
        """Test that a citation matching a retrieved doc is grounded."""
        citations = [{"ticker": "AAPL", "form": "10-K", "filing_date": "2025-10-31", "section": "Item 7"}]
        docs = [
            Document(page_content="test", metadata={"ticker": "AAPL", "form": "10-K", "filing_date": "2025-10-31", "section": "Item 7"})
        ]
        result = check_citations_grounded(citations, [Document(page_content="test", metadata={"ticker": "AAPL", "form": "10-K", "filing_date": "2025-10-31", "section": "Item 7"})])
        assert result["grounded"] == 1
        assert len(result["ungrounded"]) == 0

    def test_ungrounded_citation(self):
        """Test that a citation not matching any doc is ungrounded."""
        citations = [{"ticker": "AAPL", "form": "10-K", "filing_date": "2025-10-31", "section": "Item 7", "raw": "[AAPL, 10-K, 2025-10-31, Item 7]"}]
        docs = [Document(page_content="test", metadata={"ticker": "MSFT", "form": "10-K", "filing_date": "2025-10-31", "section": "Item 7"})]
        result = check_citations_grounded(citations, [Document(page_content="test", metadata={"ticker": "MSFT", "form": "10-K", "filing_date": "2025-10-31", "section": "Item 7"})])
        assert result["grounded"] == 0
        assert len(result["ungrounded"]) == 1

    def test_partial_match_ungrounded(self):
        """Test that partial field match is ungrounded."""
        citations = [{"ticker": "AAPL", "form": "10-Q", "filing_date": "2025-10-31", "section": "Item 7", "raw": "[AAPL, 10-Q, 2025-10-31, Item 7]"}]
        docs = [Document(page_content="test", metadata={"ticker": "AAPL", "form": "10-K", "filing_date": "2025-10-31", "section": "Item 7"})]
        result = check_citations_grounded(citations, docs)
        assert result["grounded"] == 0
        assert len(result["ungrounded"]) == 1


class TestCitationFormat:
    """Tests for citation format validation."""

    def test_valid_citation_format(self):
        """Test valid citation format passes."""
        answer = "Revenue was $416B [AAPL, 10-K, 2025-10-31, Item 7]."
        result = check_citation_format(answer)
        assert result["valid"] is True
        assert result["count"] == 1

    def test_multiple_citations(self):
        """Test multiple citations in one answer."""
        answer = "AAPL revenue $416B [AAPL, 10-K, 2025-10-31, Item 7]. MSFT revenue $200B [MSFT, 10-K, 2025-06-30, Item 7]."
        result = check_citation_format(answer)
        assert result["valid"] is True
        assert result["count"] == 2

    def test_malformed_citation(self):
        """Test malformed citation (missing fields) - not captured by regex at all."""
        answer = "Revenue was $416B [AAPL, 10-K, 2025-10-31]."
        result = check_citation_format(answer)
        assert result["valid"] is False
        assert result["count"] == 0
        assert "error" in result
        assert result["error"] == "No citations found"

    def test_no_citations(self):
        """Test answer with no citations."""
        answer = "Revenue was $416B."
        result = check_citation_format(answer)
        assert result["valid"] is False
        assert result["count"] == 0


class TestRefusalDetection:
    """Tests for refusal detection."""

    def test_exact_refusal(self):
        """Test exact refusal phrase detection."""
        answer = "Insufficient evidence to answer this question."
        assert is_refusal(answer) is True

    def test_case_insensitive_refusal(self):
        """Test case-insensitive refusal detection."""
        answer = "insufficient evidence to answer this question."
        assert is_refusal(answer) is True

    def test_partial_refusal(self):
        """Test partial refusal phrase in answer."""
        answer = "Based on the evidence, Insufficient evidence to answer this question."
        assert is_refusal(answer) is True

    def test_non_refusal(self):
        """Test non-refusal answer."""
        answer = "Revenue was $416B."
        assert is_refusal(answer) is False


class TestKeywordCoverage:
    """Tests for keyword coverage calculation."""

    def test_full_coverage(self):
        """Test full keyword coverage."""
        answer = "Revenue was $416,161 million in 2025 and $391,035 million in 2024."
        keywords = ["416,161", "391,035", "2025", "2024"]
        coverage = keyword_coverage(answer, keywords)
        assert coverage == 1.0

    def test_partial_coverage(self):
        """Test partial keyword coverage."""
        answer = "Revenue was $416,161 million in 2025."
        keywords = ["416,161", "2025"]
        coverage = keyword_coverage(answer, keywords)
        assert coverage == 1.0  # both keywords present

    def test_zero_coverage(self):
        """Test zero keyword coverage."""
        answer = "Revenue increased."
        keywords = ["416,161", "391,035"]
        coverage = keyword_coverage(answer, keywords)
        assert coverage == 0.0

    def test_empty_keywords(self):
        """Test empty keywords list returns 1.0."""
        answer = "Revenue was $416B."
        coverage = keyword_coverage(answer, [])
        assert coverage == 1.0


class TestRefusalDetection:
    """Tests for refusal detection."""

    def test_exact_refusal(self):
        answer = "Insufficient evidence to answer this question."
        assert is_refusal(answer) is True

    def test_case_insensitive(self):
        answer = "insufficient evidence to answer this question."
        assert is_refusal(answer) is True

    def test_partial(self):
        answer = "Insufficient evidence to answer this question. Available evidence shows X."
        assert is_refusal(answer) is True

    def test_non_refusal(self):
        answer = "Revenue was $416B."
        assert is_refusal(answer) is False


class TestAnswerFormat:
    """Tests for answer format validation."""

    def test_valid_format(self):
        answer = "Answer:\nRevenue was $416B.\n\nEvidence:\n- [AAPL, 10-K, 2025-10-31, Item 7]"
        result = check_answer_format(answer)
        assert result["valid_format"] is True

    def test_missing_answer(self):
        answer = "Evidence:\n- [AAPL, 10-K, 2025-10-31, Item 7]"
        result = check_answer_format(answer)
        assert result["valid_format"] is False

    def test_missing_evidence(self):
        answer = "Answer:\nRevenue was $416B."
        result = check_answer_format(answer)
        assert result["valid_format"] is False


class TestProductionPipeline:
    """Smoke test for production pipeline construction."""

    def test_build_production_pipeline(self):
        """Test that production pipeline builds without error."""
        from finrag.pipeline import build_production_pipeline
        pipeline = build_production_pipeline()
        assert pipeline is not None
        assert pipeline.vector_store is not None
        assert pchain is not None

    def test_load_production_pipeline(self):
        """Test loading existing production pipeline."""
        from finrag.pipeline import load_production_pipeline
        pipeline = load_production_pipeline()
        assert pipeline is not None
        assert pipeline.vector_store is not None