"""Tests for the :mod:`app.services.structured_llm` chokepoint.

The function exists so the four ``with_structured_output`` callers in the codebase
share one entry point. These tests use a fake LLM client to assert the wiring
without making any API calls.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from app.services import structured_llm
from app.services.structured_llm import invoke_structured


class _Reply(BaseModel):
    answer: str
    score: int = 0


class _FakeStructured:
    def __init__(self, payload: BaseModel) -> None:
        self._payload = payload
        self.run_name: str | None = None
        self.invocations: list[str] = []

    def with_config(self, **kwargs: Any) -> _FakeStructured:
        self.run_name = kwargs.get("run_name")
        return self

    def invoke(self, prompt: str) -> BaseModel:
        self.invocations.append(prompt)
        return self._payload


class _FakeLLM:
    def __init__(self, payload: BaseModel) -> None:
        self.last_model_cls: type[BaseModel] | None = None
        self.structured = _FakeStructured(payload)

    def with_structured_output(self, model_cls: type[BaseModel]) -> _FakeStructured:
        self.last_model_cls = model_cls
        return self.structured


def test_invoke_structured_uses_supplied_llm_and_returns_model_instance() -> None:
    fake = _FakeLLM(_Reply(answer="42", score=1))
    result = invoke_structured("what is the answer?", _Reply, llm=fake)
    assert isinstance(result, _Reply)
    assert result.answer == "42"
    assert result.score == 1
    assert fake.last_model_cls is _Reply
    assert fake.structured.invocations == ["what is the answer?"]
    # No run_name was provided, so with_config must not be called.
    assert fake.structured.run_name is None


def test_invoke_structured_passes_run_name_through_with_config() -> None:
    fake = _FakeLLM(_Reply(answer="ok"))
    invoke_structured("ping", _Reply, llm=fake, run_name="LLM – Test run")
    assert fake.structured.run_name == "LLM – Test run"


def test_invoke_structured_defaults_to_get_llm_for_reasoning(monkeypatch) -> None:
    fake = _FakeLLM(_Reply(answer="default-llm"))
    monkeypatch.setattr(structured_llm, "get_llm_for_reasoning", lambda: fake)
    result = invoke_structured("hi", _Reply)
    assert result.answer == "default-llm"
    assert fake.last_model_cls is _Reply
