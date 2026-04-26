"""Single chokepoint for LLM calls that must return a Pydantic-validated payload.

Phase 2 of the LangGraph→Claude Agent SDK migration: every caller that used to
chain ``llm.with_structured_output(Model).with_config(run_name=...).invoke(prompt)``
goes through :func:`invoke_structured` instead. This consolidates four
divergent call patterns onto one API so a future phase can swap the
implementation (Anthropic native ``tool_use``, or Claude Agent SDK forced
single-tool via ``app.pipeline.sdk_runtime.build_structured_options``) without
touching the call sites.

For now, the implementation delegates to the existing
``StructuredOutputClient`` machinery in :mod:`app.services.llm_client`,
preserving byte-identical behavior across every supported provider
(Anthropic, OpenAI, Bedrock, OpenRouter, Gemini, NVIDIA, MiniMax). The only
observable change at the call site is the API surface.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar, cast

from pydantic import BaseModel

from app.services.llm_client import get_llm_for_reasoning

if TYPE_CHECKING:
    from app.services.llm_client import BedrockLLMClient, LLMClient, OpenAILLMClient

    LlmLike = LLMClient | OpenAILLMClient | BedrockLLMClient


M = TypeVar("M", bound=BaseModel)


def invoke_structured(  # noqa: UP047
    prompt: str,
    model_cls: type[M],
    *,
    llm: LlmLike | None = None,
    run_name: str | None = None,
) -> M:
    """Invoke the LLM and return a validated instance of ``model_cls``.

    Parameters:
        prompt: User-facing prompt. The underlying client is responsible for
            wrapping it with any schema-conditioning text.
        model_cls: Pydantic model class to validate the response against.
        llm: Optional pre-built LLM client. Defaults to the project's
            reasoning-tier client (``get_llm_for_reasoning()``).
        run_name: Optional LangSmith / tracing label (mirrors the
            ``with_config(run_name=...)`` chain that callers used previously).
    """
    client = llm or get_llm_for_reasoning()
    structured = client.with_structured_output(model_cls)
    if run_name:
        structured = structured.with_config(run_name=run_name)
    return cast(M, structured.invoke(prompt))


__all__ = ["invoke_structured"]
