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
    """Save responses in a temporary folder, and let each test choose the fake client."""
    monkeypatch.setattr(paths, "RESPONSES", tmp_path)

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
