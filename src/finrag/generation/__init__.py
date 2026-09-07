from __future__ import annotations

import os
import time
import warnings
from typing import Any, Protocol, runtime_checkable

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential_jitter,
    retry_if_exception_type,
)

from finrag.config import config, LLMConfig

# Optional imports - only loaded when needed
warnings.filterwarnings("ignore", message="You are sending unauthenticated requests to the HF Hub")


# Single source of truth for the refusal string. Evaluation matches against this
# exact phrase, so the prompt and the checker must not drift apart.
INSUFFICIENT_EVIDENCE = "Insufficient evidence to answer this question."


RAG_PROMPT_HARDENED = ChatPromptTemplate.from_template("""You are a financial analyst assistant answering STRICTLY from SEC filing evidence passages provided below.

EVIDENCE PASSAGES:
{context}

QUESTION: {question}

RULES ΓÇö VIOLATION MEANS YOUR ANSWER WILL BE REJECTED:
1. SOURCE OF TRUTH: Use ONLY the numbered evidence passages above. Do NOT use external knowledge, financial training data, or plausible inference. If a fact is not in the passages, you do not know it.

2. CITATIONS REQUIRED: Every factual or numeric claim MUST be immediately followed by a citation in the exact format: [TICKER, FORM, FILING_DATE, SECTION]. Use the citation exactly as printed on the passage ΓÇö do NOT invent or modify it. Place the citation AFTER the claim it supports, not at the end of the answer.

3. PRESERVE SCALE & PERIOD: If a passage header states "in millions" or "in thousands", include that scale with the figure. Always state which fiscal period a figure belongs to (fiscal year, quarter, or "as of" date).

4. DIRECT VS CALCULATED: Distinguish directly stated values from calculated/derived values. If you compute something (e.g., growth rate, margin), state it explicitly as "calculated from..." with citations for each input.

5. PARTIAL EVIDENCE: If evidence covers only part of the question, answer the supported part and explicitly state which part is not covered by the evidence.

6. REFUSAL IS MANDATORY: If the evidence does not address the question, or asks for a future prediction, or sources conflict without reconciliation context, or a requested number/period/company is absent ΓÇö you MUST reply with EXACTLY:
{insufficient}
Then add ONE sentence naming what the evidence DOES contain instead.

7. NO SPECULATION: Never provide forward-looking figures, guidance, estimates, forecasts, or predictions not printed in the evidence. This includes implied trends beyond the stated periods.

OUTPUT FORMAT ΓÇö FOLLOW EXACTLY:

Answer:
<concise answer with every claim cited immediately after the claim using [TICKER, FORM, FILING_DATE, SECTION]>

Evidence:
- [TICKER, FORM, FILING_DATE, SECTION]
- [TICKER, FORM, FILING_DATE, SECTION]
...

FOR INSUFFICIENT EVIDENCE:
Insufficient evidence to answer this question.
Available evidence does not state <missing fact>.


Answer:""")

RAG_PROMPT = RAG_PROMPT_HARDENED


def is_refusal(answer: str) -> bool:
    """True when the model declined to answer for lack of evidence."""
    if not answer:
        return False
    return INSUFFICIENT_EVIDENCE.lower() in str(answer).lower()


def build_rag_prompt() -> ChatPromptTemplate:
    """Bind the refusal string into the prompt template."""
    return RAG_PROMPT.partial(insufficient=INSUFFICIENT_EVIDENCE)


def format_docs(docs) -> str:
    formatted = []
    for i, doc in enumerate(docs):
        meta = doc.metadata
        citation = f"[{meta.get('ticker', 'N/A')}, {meta.get('form', 'N/A')}, {meta.get('filing_date', 'N/A')}, {meta.get('section', 'N/A')}]"
        formatted.append(f"[{i+1}] {citation}\n{doc.page_content}")
    return "\n\n---\n\n".join(formatted)


def is_rate_limit_error(e: Exception) -> bool:
    """Check if exception is a rate limit / quota error."""
    error_str = str(e).lower()
    return any(kw in error_str for kw in ["429", "rate limit", "quota", "resource_exhausted", "too many requests"])


# =============================================================================
# Provider Protocol & Factory
# =============================================================================

@runtime_checkable
class LLMProvider(Protocol):
    """Protocol for LLM providers to implement."""
    def invoke(self, input: Any, config: Any = None) -> Any: ...


class BaseLLMProvider:
    """Base class for LLM providers with rate limiting and retry logic."""
    
    def __init__(self, model_name: str, temperature: float = 0.1, max_tokens: int = 4096):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
    
    def invoke(self, input: Any, config: Any = None) -> Any:
        raise NotImplementedError


class GeminiProvider(BaseLLMProvider):
    """Google Gemini provider."""
    
    def __init__(self, model_name: str, temperature: float = 0.1, max_tokens: int = 4096):
        super().__init__(model_name, temperature, max_tokens)
        from langchain_google_genai import ChatGoogleGenerativeAI
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not set in .env")
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
    
    def invoke(self, input: Any, config: Any = None) -> Any:
        return self.llm.invoke(input, config)


class AnthropicProvider(BaseLLMProvider):
    """Anthropic Claude provider."""
    
    def __init__(self, model_name: str, temperature: float = 0.1, max_tokens: int = 4096):
        super().__init__(model_name, temperature, max_tokens)
        from langchain_anthropic import ChatAnthropic
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not set in .env")
        self.llm = ChatAnthropic(
            model=model_name,
            anthropic_api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    
    def invoke(self, input: Any, config: Any = None) -> Any:
        return self.llm.invoke(input, config)


class OpenAIProvider(BaseLLMProvider):
    """OpenAI ChatGPT provider."""
    
    def __init__(self, model_name: str, temperature: float = 0.1, max_tokens: int = 4096):
        super().__init__(model_name, temperature, max_tokens)
        from langchain_openai import ChatOpenAI
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set in .env")
        self.llm = ChatOpenAI(
            model=model_name,
            openai_api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    
    def invoke(self, input: Any, config: Any = None) -> Any:
        return self.llm.invoke(input, config)


class OllamaProvider(BaseLLMProvider):
    """Local Ollama provider."""
    
    def __init__(self, model_name: str, temperature: float = 0.1, max_tokens: int = 4096):
        super().__init__(model_name, temperature, max_tokens)
        from langchain_ollama import ChatOllama
        self.llm = ChatOllama(
            model=model_name,
            temperature=temperature,
            num_predict=max_tokens,
        )
    
    def invoke(self, input: Any, config: Any = None) -> Any:
        return self.llm.invoke(input, config)


# Provider registry
_PROVIDER_REGISTRY = {
    "gemini": GeminiProvider,
    "anthropic": AnthropicProvider,
    "openai": OpenAIProvider,
    "ollama": OllamaProvider,
}


def get_provider_class(provider: str) -> type:
    """Get provider class by name."""
    provider = provider.lower()
    if provider not in _PROVIDER_REGISTRY:
        raise ValueError(f"Unknown provider: {provider}. Available: {list(_PROVIDER_REGISTRY.keys())}")
    return _PROVIDER_REGISTRY[provider]


def create_llm_provider(cfg: LLMConfig | None = None) -> BaseLLMProvider:
    """Factory function to create LLM provider from config."""
    cfg = cfg or config.llm
    provider_class = get_provider_class(cfg.provider)
    return provider_class(
        model_name=cfg.model_name,
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
    )


def is_rate_limit_error(e: Exception) -> bool:
    """Check if exception is a rate limit / quota error."""
    error_str = str(e).lower()
    return any(kw in error_str for kw in ["429", "rate limit", "quota", "resource_exhausted", "too many requests"])


def create_rate_limited_llm(provider: BaseLLMProvider, rpm: int = 10) -> RunnableLambda:
    """
    Create a rate-limited LLM with exponential backoff retry for quota errors.
    
    Respects RPM limit locally, and retries with exponential backoff on 429 errors.
    """
    min_interval = 60.0 / rpm
    last_called = [0.0]

    @retry(
        wait=wait_exponential_jitter(initial=1, max=60),
        stop=stop_after_attempt(5),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    def _invoke_with_retry(input: Any, config: Any = None) -> Any:
        # Local RPM throttle
        elapsed = time.time() - last_called[0]
        wait_time = min_interval - elapsed
        if wait_time > 0:
            time.sleep(wait_time)
        last_called[0] = time.time()
        
        try:
            return provider.invoke(input, config)
        except Exception as e:
            if is_rate_limit_error(e):
                print(f"Rate limit hit, retrying with backoff: {e}")
                raise
            raise

    return RunnableLambda(_invoke_with_retry)


def get_llm(cfg: LLMConfig | None = None) -> BaseLLMProvider:
    """Factory: get LLM provider instance from config."""
    cfg = cfg or config.llm
    return create_llm_provider(cfg)


def get_rate_limited_llm(cfg: LLMConfig | None = None) -> RunnableLambda:
    """Get LLM with rate limiting as a Runnable."""
    cfg = cfg or config.llm
    provider = get_llm(cfg)
    return create_rate_limited_llm(provider, rpm=cfg.rpm)
