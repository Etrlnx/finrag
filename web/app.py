"""FinRAG Streamlit Frontend - Financial Intelligence & Evidence Trace."""

from __future__ import annotations

import sys
import os
from pathlib import Path

# Add src to path for imports
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

import streamlit as st
import pandas as pd
import time
from typing import Optional

from finrag.pipeline import load_production_pipeline
from finrag.explainability.models import ExplainableResult, VerificationStatus

# Page config
st.set_page_config(
    page_title="FinRAG — Financial Intelligence & Evidence Trace",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better financial UI
st.markdown("""
<style>
    .stApp { background-color: #f8f9fa; }
    .main-header { 
        background: linear-gradient(90deg, #1e3a5f 0%, #2c5282 100%);
        color: white; padding: 1rem; border-radius: 8px; margin-bottom: 1rem;
    }
    .status-pill {
        display: inline-block; padding: 0.25rem 0.75rem; border-radius: 9999px;
        font-size: 0.75rem; font-weight: 600; margin-right: 0.5rem;
    }
    .pill-grounded { background-color: #d4edda; color: #155724; }
    .pill-partial { background-color: #fff3cd; color: #856404; }
    .pill-refusal { background-color: #f8d7da; color: #721c24; }
    .pill-hallucinated { background-color: #f5c6cb; color: #721c24; }
    .claim-card {
        border: 1px solid #dee2e6; border-radius: 8px; padding: 1rem;
        margin: 0.5rem 0; background: white;
    }
    .citation-badge {
        background: #e9ecef; border: 1px solid #ced4da;
        border-radius: 4px; padding: 0.125rem 0.5rem;
        font-family: monospace; font-size: 0.8rem; cursor: pointer;
    }
    .evidence-chunk { 
        border-left: 3px solid #2c5282; padding: 0.75rem; 
        background: #f8f9fa; margin: 0.5rem 0; font-size: 0.9rem;
    }
    .metric-card { background: white; border: 1px solid #dee2e6; border-radius: 8px; padding: 1rem; }
</style>
""", unsafe_allow_html=True)


# Cache the pipeline loading
@st.cache_resource(show_spinner="Loading FinRAG pipeline...")
def get_pipeline():
    return load_production_pipeline()


# Preset example questions
PRESET_QUESTIONS = {
    "Apple FY2025 Revenue": "What was Apple's total net sales in fiscal year 2025?",
    "Apple FY2025 vs FY2024": "What were Apple's total net sales for fiscal 2025 versus fiscal 2024?",
    "Apple Q3 2026 Revenue": "What was Apple's revenue in Q3 2026?",
    "Microsoft Risk Factors": "What are Microsoft's main risk factors regarding complex datacenters and hardware products?",
    "Microsoft Cloud Revenue": "How did Microsoft Cloud revenue and Intelligent Cloud operating income change year-over-year?",
    "NVIDIA R&D Spending": "How much did NVIDIA spend on research and development in the most recent quarter versus the prior-year quarter?",
    "NVIDIA Data Center Revenue": "How did NVIDIA's Data Center revenue compare between fiscal year 2025 and fiscal year 2026?",
    "Google Total Revenue": "What was Alphabet's total revenues for the year ended December 31, 2025 compared to 2024?",
    "Amazon AWS Revenue": "What were the net sales and operating income for AWS (Amazon Web Services)?",
    "Meta Reality Labs Loss": "What was the operating loss reported by Meta's Reality Labs segment?",
}


def render_status_pill(status: str) -> str:
    """Render a color-coded status pill."""
    status_map = {
        "VERIFIED": ("🟢 Grounded", "pill-grounded"),
        "PARTIAL": ("🟡 Partial Evidence", "pill-partial"),
        "UNGROUNDED": ("🔴 Unverified", "pill-refusal"),
        "HALLUCINATED": ("🚫 Hallucinated", "pill-hallucinated"),
    }
    label, cls = status_map.get(status, ("⚪ Unknown", ""))
    return f'<span class="status-pill {cls}">{label}</span>'


def render_citation_badge(citation: dict) -> str:
    """Render a clickable citation badge."""
    if not citation:
        return ""
    return (
        f'<span class="citation-badge" title="Click to view source">'
        f'[{citation.get("ticker", "")}, {citation.get("form", "")}, '
        f'{citation.get("filing_date", "")}, {citation.get("section", "")}]'
        f'</span>'
    )


def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1 style="margin: 0;">🏛️ FinRAG — Financial Intelligence & Evidence Trace</h1>
        <p style="margin: 0.5rem 0 0; opacity: 0.9;">SEC EDGAR 10-K/10-Q Analysis with Grounded Citations & Retrieval Diagnostics</p>
    </div>
    """, unsafe_allow_html=True)

    # Load pipeline
    try:
        pipeline = get_pipeline()
    except Exception as e:
        st.error(f"Failed to load pipeline: {e}")
        st.stop()

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Controls")

        st.subheader("📊 Filing Filters")
        col1, col2 = st.columns(2)
        with col1:
            ticker_filter = st.selectbox(
                "Company",
                ["All", "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "JPM", "GS", "BAC", "WMT", "COST", "JNJ", "UNH", "XOM"],
                index=0,
            )
        with col2:
            form_filter = st.selectbox("Filing Type", ["All", "10-K", "10-Q"], index=0)

        st.subheader("🔧 Retrieval Settings")
        use_reranker = st.toggle("Cross-Encoder Reranker (top-20 → 5)", value=True)
        use_filtering = st.toggle("Metadata Filtering", value=True)

        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.0, max_value=1.0, value=0.5, step=0.05,
            help="Minimum verification confidence for claim badges"
        )

        show_explainable = st.toggle("Show Evidence Trace", value=True)
        st.caption("Shows claim-by-claim verification, evidence chunks, and retrieval scores")

    # Main area
    col_main, col_sidebar_info = st.columns([3, 1])

    with col_main:
        # Query input
        st.subheader("💬 Query")

        # Preset buttons
        st.write("**Quick Examples:**")
        preset_cols = st.columns(4)
        selected_preset = None
        for i, (label, question) in enumerate(PRESET_QUESTIONS.items()):
            with preset_cols[i % 4]:
                if st.button(label, key=f"preset_{i}", use_container_width=True):
                    selected_preset = question

        # Query input
        default_query = selected_preset if selected_preset else ""
        query = st.text_input(
            "Your Question",
            value=default_query,
            placeholder="e.g., What was Apple's total net sales in fiscal 2025?",
            label_visibility="collapsed",
        )

        run_button = st.button("🚀 Run Analysis", type="primary", use_container_width=True)

        # Process query
        if run_button and query:
            with st.spinner("Running analysis..."):
                start = time.perf_counter()
                try:
                    if show_explainable:
                        result: ExplainableResult = pipeline.query_explainable(query)
                        answer = result.answer
                        is_refusal = result.is_refusal
                    else:
                        answer = pipeline.query(query)
                        is_refusal = "Insufficient evidence to answer this question" in answer
                        result = None

                    latency = (time.perf_counter() - start) * 1000

                    # Display answer
                    st.divider()
                    st.subheader("📝 Grounded Answer")

                    # Status indicator
                    if is_refusal:
                        st.markdown(
                            render_status_pill("UNGROUNDED"),
                            unsafe_allow_html=True,
                        )
                    elif result and result.claim_traces:
                        statuses = [ct.verification_status.value for ct in result.claim_traces]
                        if all(s == "VERIFIED" for s in statuses):
                            st.markdown(render_status_pill("VERIFIED"), unsafe_allow_html=True)
                        elif any(s == "HALLUCINATED" for s in statuses):
                            st.markdown(render_status_pill("HALLUCINATED"), unsafe_allow_html=True)
                        elif any(s in ("PARTIAL", "UNGROUNDED") for s in statuses):
                            st.markdown(render_status_pill("PARTIAL"), unsafe_allow_html=True)
                        else:
                            st.markdown(render_status_pill("VERIFIED"), unsafe_allow_html=True)
                    else:
                        st.markdown(render_status_pill("VERIFIED"), unsafe_allow_html=True)

                    # Answer text
                    st.markdown(answer)

                    # Latency
                    st.caption(f"⏱️ Latency: {latency:.0f} ms")

                    # Explainable evidence trace
                    if show_explainable and result:
                        st.divider()
                        st.subheader("🔍 Evidence & Audit Trace")

                        tab1, tab2, tab3 = st.tabs([
                            "📄 Supporting Passages",
                            "📊 Financial Tables",
                            "📈 Retrieval Diagnostics"
                        ])

                        with tab1:
                            st.write("**Claim-by-Claim Verification**")
                            for i, ct in enumerate(result.claim_traces, 1):
                                with st.expander(f"Claim {i}: {ct.claim_text[:100]}...", expanded=False):
                                    st.markdown(render_status_pill(ct.verification_status.value), unsafe_allow_html=True)
                                    st.write(f"**Confidence:** {ct.confidence:.2f}")
                                    if ct.citation:
                                        st.markdown(f"**Citation:** {render_citation_badge(ct.citation)}", unsafe_allow_html=True)
                                    if ct.matched_chunk_id:
                                        st.write(f"**Matched Chunk:** `{ct.matched_chunk_id}`")
                                    if ct.matched_tokens:
                                        st.write(f"✅ **Verified:** {', '.join(ct.matched_tokens)}")
                                    if ct.missing_tokens:
                                        st.write(f"❌ **Missing:** {', '.join(ct.missing_tokens)}")

                            st.write("**Top-5 Retrieved Evidence Chunks**")
                            for ec in result.evidence_chunks:
                                with st.container():
                                    meta = ec.metadata
                                    st.markdown(
                                        f'<div class="evidence-chunk">'
                                        f'<strong>{meta.get("ticker", "")} • {meta.get("form", "")} • '
                                        f'{meta.get("filing_date", "")} • {meta.get("section", "")}</strong>'
                                        f'{" 📊 Table" if ec.is_table else ""}<br>'
                                        f'Rank: {ec.scores.final_rank} | '
                                        f'Dense: {ec.scores.dense_score:.3f}' if ec.scores.dense_score else 'Rank: N/A'
                                        f' | BM25: {ec.scores.bm25_rank}' if ec.scores.bm25_rank else ''
                                        f' | Rerank: {ec.scores.rerank_score:.3f}' if ec.scores.rerank_score else ''
                                        f'<br><small>{ec.content[:300]}...</small>'
                                        f'</div>',
                                        unsafe_allow_html=True,
                                    )

                        with tab2:
                            table_chunks = [ec for ec in result.evidence_chunks if ec.is_table]
                            if table_chunks:
                                st.write(f"Found {len(table_chunks)} table chunks in top-5 evidence")
                                for ec in table_chunks:
                                    with st.expander(f"Table: {ec.metadata.get('ticker')} • {ec.metadata.get('filing_date')} • {ec.metadata.get('section')}", expanded=False):
                                        st.markdown(ec.content)
                            else:
                                st.info("No financial tables in top-5 retrieved evidence for this query.")

                        with tab3:
                            st.write("**Multi-Stage Retrieval Scores**")
                            diag = result.retrieval_diagnostics
                            cols = st.columns(4)
                            cols[0].metric("Total Retrieved", diag.get("total_retrieved", 0))
                            cols[1].metric("Dense Scored", diag.get("dense_scored", 0))
                            cols[2].metric("BM25 Scored", diag.get("bm25_scored", 0))
                            cols[3].metric("Rerank Scored", diag.get("rerank_scored", 0))

                            st.write("**Per-Chunk Score Breakdown**")
                            score_data = []
                            for ec in result.evidence_chunks:
                                score_data.append({
                                    "Chunk ID": ec.chunk_id,
                                    "Ticker": ec.metadata.get("ticker", ""),
                                    "Section": ec.metadata.get("section", ""),
                                    "Final Rank": ec.scores.final_rank,
                                    "Dense Score": f"{ec.scores.dense_score:.3f}" if ec.scores.dense_score else "N/A",
                                    "BM25 Rank": ec.scores.bm25_rank if ec.scores.bm25_rank else "N/A",
                                    "RRF Rank": ec.scores.rrf_rank if ec.scores.rrf_rank else "N/A",
                                    "Rerank Score": f"{ec.scores.rerank_score:.3f}" if ec.scores.rerank_score else "N/A",
                                    "Is Table": "✅" if ec.is_table else "",
                                })
                            if score_data:
                                df = pd.DataFrame(score_data)
                                st.dataframe(df, use_container_width=True, hide_index=True)

                            if result.ungrounded_citations:
                                st.warning(f"⚠️ {len(result.ungrounded_citations)} ungrounded citation(s) detected")
                                for uc in result.ungrounded_citations:
                                    st.code(uc)

                except Exception as e:
                    st.error(f"Error during analysis: {e}")

    # Sidebar info panel
    with col_sidebar_info:
        st.subheader("📊 System Status")
        st.success("Pipeline: **Loaded**")
        st.info("Index: **phase6_table_aware** (16,626 chunks, 4,678 tables)")
        st.info("Embeddings: **BAAI/bge-base-en-v1.5**")
        st.info("Reranker: **ms-marco-MiniLM-L-6-v2**")
        st.info("LLM: **Ollama llama3.2**")

        st.divider()
        st.caption("FinRAG v0.1.0 | Built on LangChain + Streamlit")


if __name__ == "__main__":
    main()