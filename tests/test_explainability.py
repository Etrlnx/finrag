"""Unit tests for FinRAG explainability module."""

import pytest
from langchain_core.documents import Document

from finrag.explainability import (
    _chunk_id,
    extract_citations_from_answer,
    extract_claims_from_answer,
    verify_claim_against_chunk,
    attribute_claims_to_evidence,
    build_explainable_result,
)
from finrag.explainability.models import (
    EvidenceChunk,
    ClaimTrace,
    ExplainableResult,
    RetrievalScore,
    VerificationStatus,
)


class TestChunkId:
    def test_chunk_id_format(self):
        doc = Document(
            page_content="test content",
            metadata={"ticker": "AAPL", "form": "10-K", "filing_date": "2025-10-31", "section": "Item 7"}
        )
        cid = _chunk_id(doc)
        assert "AAPL" in cid
        assert "10-K" in cid
        assert "2025-10-31" in cid
        assert "Item 7" in cid


class TestCitationExtraction:
    def test_extract_single_citation(self):
        answer = "Revenue was $416B [AAPL, 10-K, 2025-10-31, Item 7]."
        citations = extract_citations_from_answer(answer)
        assert len(citations) == 1
        assert citations[0]["ticker"] == "AAPL"
        assert citations[0]["form"] == "10-K"
        assert citations[0]["filing_date"] == "2025-10-31"
        assert citations[0]["section"] == "Item 7"

    def test_extract_multiple_citations(self):
        answer = "AAPL: $416B [AAPL, 10-K, 2025-10-31, Item 7]. MSFT: $200B [MSFT, 10-K, 2025-06-30, Item 7]."
        citations = extract_citations_from_answer(answer)
        assert len(citations) == 2
        assert citations[0]["ticker"] == "AAPL"
        assert citations[1]["ticker"] == "MSFT"

    def test_extract_no_citations(self):
        answer = "Revenue was $416B."
        citations = extract_citations_from_answer(answer)
        assert len(citations) == 0


class TestClaimExtraction:
    def test_extract_claims_from_answer(self):
        answer = """Answer:
Revenue was $416B [AAPL, 10-K, 2025-10-31, Item 7].
Net income increased [AAPL, 10-K, 2025-10-31, Item 7].

Evidence:
- [AAPL, 10-K, 2025-10-31, Item 7]"""
        claims = extract_claims_from_answer(answer)
        assert len(claims) == 2
        assert "$416B" in claims[0][0]
        assert "increased" in claims[1][0]
        assert claims[0][1] is not None
        assert claims[1][1] is not None


class TestClaimVerification:
    def test_verified_claim(self):
        claim = "Revenue was $416,161 million."
        chunk = Document(
            page_content="Total net sales were $416,161 million for fiscal 2025.",
            metadata={"ticker": "AAPL", "form": "10-K", "filing_date": "2025-10-31", "section": "Item 7"}
        )
        citation = {"ticker": "AAPL", "form": "10-K", "filing_date": "2025-10-31", "section": "Item 7"}
        status, conf, matched, missing = verify_claim_against_chunk(claim, chunk, citation)
        assert status == VerificationStatus.VERIFIED
        assert conf >= 0.65
        assert "416,161" in matched

    def test_partial_claim(self):
        claim = "Revenue was $416,161 million and profit was $100 million."
        chunk = Document(
            page_content="Total net sales were $416,161 million for fiscal 2025.",
            metadata={"ticker": "AAPL", "form": "10-K", "filing_date": "2025-10-31", "section": "Item 7"}
        )
        citation = {"ticker": "AAPL", "form": "10-K", "filing_date": "2025-10-31", "section": "Item 7"}
        status, conf, matched, missing = verify_claim_against_chunk(claim, chunk, citation)
        assert status in (VerificationStatus.VERIFIED, VerificationStatus.PARTIAL)
        assert "416,161" in matched
        assert "100" in missing

    def test_hallucinated_claim(self):
        claim = "Revenue was $999,999 million."
        chunk = Document(
            page_content="Total net sales were $416,161 million for fiscal 2025.",
            metadata={"ticker": "AAPL", "form": "10-K", "filing_date": "2025-10-31", "section": "Item 7"}
        )
        citation = {"ticker": "AAPL", "form": "10-K", "filing_date": "2025-10-31", "section": "Item 7"}
        status, conf, matched, missing = verify_claim_against_chunk(claim, chunk, citation)
        assert status in (VerificationStatus.HALLUCINATED, VerificationStatus.UNGROUNDED)
        assert conf < 0.5

    def test_ungrounded_citation_mismatch(self):
        claim = "Revenue was $416,161 million."
        chunk = Document(
            page_content="Total net sales were $416,161 million for fiscal 2025.",
            metadata={"ticker": "MSFT", "form": "10-K", "filing_date": "2025-06-30", "section": "Item 7"}
        )
        citation = {"ticker": "AAPL", "form": "10-K", "filing_date": "2025-10-31", "section": "Item 7"}
        status, conf, matched, missing = verify_claim_against_chunk(claim, chunk, citation)
        assert status == VerificationStatus.UNGROUNDED
        assert conf == 0.0


class TestClaimAttribution:
    def test_attribute_multiple_claims(self):
        answer = """Answer:
Revenue was $416,161 million [AAPL, 10-K, 2025-10-31, Item 7].
Assets were $359,241 million [AAPL, 10-K, 2025-10-31, Item 8].

Evidence:
- [AAPL, 10-K, 2025-10-31, Item 7]
- [AAPL, 10-K, 2025-10-31, Item 8]"""
        
        chunk1 = EvidenceChunk(
            chunk_id="AAPL_10-K_2025-10-31_Item 7_1234",
            content="Total net sales were $416,161 million.",
            metadata={"ticker": "AAPL", "form": "10-K", "filing_date": "2025-10-31", "section": "Item 7"},
            scores=RetrievalScore(),
        )
        chunk2 = EvidenceChunk(
            chunk_id="AAPL_10-K_2025-10-31_Item 8_5678",
            content="Total assets were $359,241 million.",
            metadata={"ticker": "AAPL", "form": "10-K", "filing_date": "2025-10-31", "section": "Item 8"},
            scores=RetrievalScore(),
        )
        
        traces, ungrounded = attribute_claims_to_evidence(answer, [chunk1, chunk2])
        assert len(traces) == 2
        assert traces[0].verification_status == VerificationStatus.VERIFIED
        assert traces[1].verification_status == VerificationStatus.VERIFIED
        assert len(ungrounded) == 0

    def test_attribute_with_ungrounded_citation(self):
        answer = """Answer:
Revenue was $416,161 million [AAPL, 10-K, 2025-10-31, Item 7].
Fake number $999,999 million [AAPL, 10-K, 2025-10-31, Item 7].

Evidence:
- [AAPL, 10-K, 2025-10-31, Item 7]"""
        
        chunk = EvidenceChunk(
            chunk_id="AAPL_10-K_2025-10-31_Item 7_1234",
            content="Total net sales were $416,161 million.",
            metadata={"ticker": "AAPL", "form": "10-K", "filing_date": "2025-10-31", "section": "Item 7"},
            scores=RetrievalScore(),
        )
        
        traces, ungrounded = attribute_claims_to_evidence(answer, [chunk])
        assert len(traces) == 2
        assert traces[0].verification_status == VerificationStatus.VERIFIED
        assert traces[1].verification_status in (VerificationStatus.HALLUCINATED, VerificationStatus.UNGROUNDED)
        assert len(ungrounded) >= 1


class TestExplainableResult:
    def test_to_dict(self):
        result = ExplainableResult(
            answer="Test answer",
            is_refusal=False,
            claim_traces=[],
            evidence_chunks=[],
            ungrounded_citations=[],
        )
        d = result.to_dict()
        assert d["answer"] == "Test answer"
        assert d["is_refusal"] is False
        assert "claim_traces" in d
        assert "evidence_chunks" in d

    def test_to_string(self):
        result = ExplainableResult(
            answer="Test answer",
            is_refusal=False,
            claim_traces=[],
            evidence_chunks=[],
            ungrounded_citations=[],
        )
        s = result.to_string()
        assert "EXPLAINABLE RESULT" in s
        assert "Test answer" in s


if __name__ == "__main__":
    pytest.main([__file__, "-v"])