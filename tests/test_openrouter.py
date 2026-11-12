#### Preamble ####
# Purpose: Test ask() without calling OpenRouter: a fake client stands in for the real one
# Author: Sean Murphy
# Date: 11 November 2026
# Pre-requisites: none
# Run with `uv run pytest`.

import httpx2
import openai
import pytest

from smiley_bench import openrouter, paths


class AlwaysTimesOut:
    """Stands in for the OpenAI client: every call times out, as if all retries had failed."""

    def __init__(self):
        self.calls = 0
        self.chat = self
        self.completions = self

    def create(self, **kwargs):
        self.calls += 1
        raise openai.APITimeoutError(request=httpx2.Request("POST", "https://openrouter.ai/api/v1/chat/completions"))


class RateLimited(AlwaysTimesOut):
    """Every call is refused with a rate limit (HTTP 429), as if all retries had failed."""

    def create(self, **kwargs):
        self.calls += 1
        request = httpx2.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
        raise openai.RateLimitError("rate limited", response=httpx2.Response(429, request=request), body=None)


@pytest.fixture
def fake_client(monkeypatch, tmp_path):
    """Save responses in a temporary folder, and let each test choose the fake client. OpenRouter's model list is
    left empty, so no test goes online (an unknown model is asked at TEMPERATURE)."""
    monkeypatch.setattr(paths, "RESPONSES", tmp_path)
    monkeypatch.setattr(openrouter, "openrouter_models", dict)

    def use(client):
        monkeypatch.setattr(openrouter, "openrouter_client", lambda: client)
        return client

    return use


def test_a_timeout_is_saved_as_a_non_answer(fake_client, tmp_path):
    client = fake_client(AlwaysTimesOut())
    record = openrouter.ask("some/model", "SMILES: CCO\nheavy_atoms: How many heavy atoms?", {"heavy_atoms": "number"})
    assert record["response"] is None
    assert record["finish_reason"] == "timeout"
    assert len(list(tmp_path.glob("*.json"))) == 1

    # Asked again, the saved non-answer is returned without calling the API.
    again = openrouter.ask("some/model", "SMILES: CCO\nheavy_atoms: How many heavy atoms?", {"heavy_atoms": "number"})
    assert again == record
    assert client.calls == 1


def test_a_rate_limit_is_raised_and_not_saved(fake_client, tmp_path):
    fake_client(RateLimited())
    with pytest.raises(openai.RateLimitError):
        openrouter.ask("some/model", "SMILES: CCO\nheavy_atoms: How many heavy atoms?", {"heavy_atoms": "number"})
    assert list(tmp_path.glob("*.json")) == []


def test_answer_schema_has_one_field_per_question():
    schema = openrouter.answer_schema({"chiral_atoms": "number", "is_chiral": "boolean"})
    assert schema["properties"] == {"chiral_atoms": {"type": "number"}, "is_chiral": {"type": "boolean"}}
    assert schema["required"] == ["chiral_atoms", "is_chiral"]
    assert schema["additionalProperties"] is False


def test_saved_record_has_the_date_and_fields(fake_client):
    fake_client(AlwaysTimesOut())
    record = openrouter.ask("some/model", "SMILES: CCO\nheavy_atoms: How many heavy atoms?", {"heavy_atoms": "number"})
    assert record["answer_types"] == {"heavy_atoms": "number"}
    assert len(record["date"]) == 10  # e.g. 2026-11-12


def test_budget_reached_stops_before_calling(fake_client, monkeypatch):
    client = fake_client(AlwaysTimesOut())
    monkeypatch.setattr(openrouter, "spent", openrouter.BUDGET_USD)
    with pytest.raises(openrouter.BudgetReached):
        openrouter.ask("some/model", "SMILES: CCO\nheavy_atoms: How many heavy atoms?", {"heavy_atoms": "number"})
    assert client.calls == 0


def test_ollama_models_go_to_the_local_client(monkeypatch):
    """A model named "ollama/..." is sent to Ollama without the prefix; any other model goes to OpenRouter."""
    monkeypatch.setattr(openrouter, "local_client", lambda: "local")
    monkeypatch.setattr(openrouter, "openrouter_client", lambda: "openrouter")
    assert openrouter.client_and_name("ollama/gemma4:12b") == ("local", "gemma4:12b")
    assert openrouter.client_and_name("google/gemma-4-31b-it:free") == ("openrouter", "google/gemma-4-31b-it:free")


class RateLimitedOnce(AlwaysTimesOut):
    """Rate-limited on the first call only; the retry times out (a stand-in for any reply that gets saved)."""

    def create(self, **kwargs):
        self.calls += 1
        request = httpx2.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
        if self.calls == 1:
            raise openai.RateLimitError("rate limited", response=httpx2.Response(429, request=request), body=None)
        raise openai.APITimeoutError(request=request)


def test_rate_limit_waits_and_retries(fake_client, monkeypatch):
    monkeypatch.setattr(openrouter, "RATE_LIMIT_WAIT_SECONDS", 0)
    client = fake_client(RateLimitedOnce())
    waits = []
    record = openrouter.ask_waiting("some/model", "SMILES: CCO", {"heavy_atoms": "number"}, on_wait=waits.append)
    assert record["finish_reason"] == "timeout"  # the retry got through and was saved
    assert waits == [1]
    assert client.calls == 2


def test_rate_limit_gives_up_after_the_last_retry(fake_client, monkeypatch, tmp_path):
    monkeypatch.setattr(openrouter, "RATE_LIMIT_WAIT_SECONDS", 0)
    client = fake_client(RateLimited())
    waits = []
    with pytest.raises(openai.RateLimitError):
        openrouter.ask_waiting("some/model", "SMILES: CCO", {"heavy_atoms": "number"}, on_wait=waits.append)
    assert waits == [1, 2, 3]
    assert client.calls == 1 + openrouter.RATE_LIMIT_RETRIES
    assert list(tmp_path.glob("*.json")) == []  # nothing saved


def test_other_errors_are_not_retried(fake_client, monkeypatch):
    """E.g. a model that is not available (error 403) fails at once, without waiting."""

    class Forbidden(AlwaysTimesOut):
        def create(self, **kwargs):
            self.calls += 1
            request = httpx2.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
            raise openai.PermissionDeniedError("forbidden", response=httpx2.Response(403, request=request), body=None)

    monkeypatch.setattr(openrouter, "RATE_LIMIT_WAIT_SECONDS", 0)
    client = fake_client(Forbidden())
    with pytest.raises(openai.PermissionDeniedError):
        openrouter.ask_waiting("some/model", "SMILES: CCO", {"heavy_atoms": "number"})
    assert client.calls == 1


def test_forget_timeout_deletes_only_a_saved_timeout(fake_client, tmp_path):
    fake_client(AlwaysTimesOut())
    fields = {"heavy_atoms": "number"}
    openrouter.ask("some/model", "SMILES: CCO", fields)  # saved as a timeout
    assert openrouter.forget_timeout("some/model", "SMILES: CCO", fields) is True
    assert list(tmp_path.glob("*.json")) == []
    assert openrouter.forget_timeout("some/model", "SMILES: CCO", fields) is False  # nothing left to delete


def test_forget_timeout_keeps_a_real_answer(tmp_path, monkeypatch):
    monkeypatch.setattr(openrouter.paths, "RESPONSES", tmp_path)
    fields = {"heavy_atoms": "number"}
    path = openrouter.response_path("some/model", "SMILES: CCO", fields, 1)
    path.write_text('{"finish_reason": "stop", "response": "{\\"heavy_atoms\\": 3}"}')
    assert openrouter.forget_timeout("some/model", "SMILES: CCO", fields) is False
    assert path.exists()


class ReplyWithProviderError(AlwaysTimesOut):
    """Returns a reply whose finish_reason is "error": a provider failure, not an answer."""

    def create(self, **kwargs):
        self.calls += 1
        message = type("Message", (), {"content": ""})()
        choice = type("Choice", (), {"message": message, "finish_reason": "error"})()
        return type("Response", (), {"choices": [choice], "usage": None, "model": "some/model"})()


def test_provider_error_inside_a_reply_is_raised_and_not_saved(fake_client, tmp_path):
    fake_client(ReplyWithProviderError())
    with pytest.raises(RuntimeError, match="Provider error"):
        openrouter.ask("some/model", "SMILES: CCO", {"heavy_atoms": "number"})
    assert list(tmp_path.glob("*.json")) == []


def test_temperature_is_left_out_for_models_that_do_not_accept_one(monkeypatch):
    models = {
        "a/takes-temperature": {"supported_parameters": ["temperature", "max_tokens"]},
        "b/no-temperature": {"supported_parameters": ["max_tokens", "reasoning"]},
    }
    monkeypatch.setattr(openrouter, "openrouter_models", lambda: models)
    assert openrouter.temperature_for("a/takes-temperature") == openrouter.TEMPERATURE
    assert openrouter.temperature_for("b/no-temperature") is None
    assert openrouter.temperature_for("ollama/gemma4:12b") == openrouter.TEMPERATURE
    assert "temperature" not in openrouter.request_body("b/no-temperature", "SMILES: CCO", {"heavy_atoms": "number"})
    body = openrouter.request_body("a/takes-temperature", "SMILES: CCO", {"heavy_atoms": "number"})
    assert body["temperature"] == 1.0


def test_claude_models_get_high_reasoning_effort(monkeypatch):
    monkeypatch.setattr(openrouter, "openrouter_models", dict)
    body = openrouter.request_body("anthropic/claude-opus-5.5", "SMILES: CCO", {"heavy_atoms": "number"})
    assert body["reasoning"] == {"effort": "high"}
    assert "reasoning" not in openrouter.request_body("a/other-model", "SMILES: CCO", {"heavy_atoms": "number"})
    record = openrouter.new_record("anthropic/claude-opus-5.5", "SMILES: CCO", {"heavy_atoms": "number"}, 1)
    assert record["reasoning_effort"] == "high"
