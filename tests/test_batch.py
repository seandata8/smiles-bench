#### Preamble ####
# Purpose: Test the batch waiting loop without calling OpenRouter: fake batches finish after a set number of checks
# Author: Sean Murphy
# Date: 12 November 2026
# Pre-requisites: none
# Run with `uv run pytest`.

import json

from smiley_bench import batch, openrouter, paths
from smiley_bench.properties import answer_types


def test_finish_all_shows_every_batch_and_finishes_each_when_done(monkeypatch, capsys):
    checks_until_done = {"id-haiku": 1, "id-opus": 3}
    checks = {"id-haiku": 0, "id-opus": 0}

    def fake_check(batch_id):
        checks[batch_id] += 1
        done = checks[batch_id] > checks_until_done[batch_id]
        return {"id": batch_id, "status": "completed" if done else "in_progress", "results": [] if done else None}

    finished = []
    monkeypatch.setattr(batch, "check", fake_check)
    monkeypatch.setattr(batch, "finish", lambda model, b, rows, trial: finished.append(model) or {"model": model})
    monkeypatch.setattr(batch, "POLL_SECONDS", 0)

    summary = batch.finish_all({"a/claude-opus": "id-opus", "a/claude-haiku": "id-haiku"}, rows=[], trial=1)

    assert finished == ["a/claude-haiku", "a/claude-opus"]  # each saved as soon as it is done
    assert [s["model"] for s in summary] == ["a/claude-opus", "a/claude-haiku"]  # in the order given
    first_line = capsys.readouterr().out.splitlines()[0]
    assert "claude-opus in_progress" in first_line and "claude-haiku in_progress" in first_line


def test_save_answers_keeps_reasoning_apart_from_the_response(monkeypatch, tmp_path):
    monkeypatch.setattr(paths, "RESPONSES", tmp_path / "responses")
    monkeypatch.setattr(openrouter, "openrouter_models", dict)
    monkeypatch.setattr(batch, "batch_price", lambda model: (1e-6, 2e-6))
    row = {"molecule_id": "atoms-1234abcd", "smiles_form": 1, "test": "atoms", "prompt": "SMILES: CCO"}
    message = {"content": '{"hydrogen_count": 6, "heavy_atoms": 3}', "reasoning": "Ethanol: C2H6O."}
    body = {
        "model": "a/model-20260101",
        "choices": [{"message": message, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 50, "completion_tokens": 40, "completion_tokens_details": {"reasoning_tokens": 30}},
    }
    fake_batch = {"id": "batch-1", "results": [{"custom_id": "atoms-1234abcd-form1", "response": {"body": body}}]}

    counts = batch.save_answers("a/model", fake_batch, [row], trial=1)

    assert counts["answered"] == 1
    path = openrouter.response_path("a/model", "SMILES: CCO", answer_types("atoms"), 1)
    record = json.loads(path.read_text())
    assert record["reasoning_tokens"] == 30 and "reasoning" not in record
    assert json.loads(openrouter.reasoning_path(path).read_text()) == {"reasoning": "Ethanol: C2H6O."}


def write_response(folder, name, model, prompt, tokens_out):
    """A saved response record with just what the cost estimate reads."""
    folder.mkdir(parents=True, exist_ok=True)
    record = {"model_requested": model, "prompt": prompt, "tokens_out": tokens_out}
    (folder / f"{name}.json").write_text(json.dumps(record))


def test_estimated_cost_uses_past_token_use(monkeypatch, tmp_path):
    """Per prompt: the model's own average on that molecule; a molecule it has not answered: the highest average of
    any model on it; a molecule no model has answered: MAX_TOKENS."""
    monkeypatch.setattr(paths, "RAW_DATA", tmp_path)
    monkeypatch.setattr(batch, "batch_price", lambda model: (0.0, 1e-6))  # output tokens only: $1 per million
    write_response(tmp_path / "pilot" / "responses", "a", "m/one", "big form 1", 30_000)
    write_response(tmp_path / "pilot" / "responses", "b", "m/one", "big form 3", 10_000)
    write_response(tmp_path / "max-run" / "responses", "c", "m/one", "small form 1", 1_000)
    write_response(tmp_path / "pilot" / "responses", "d", "m/one", "a prompt no longer asked", 99_000)
    write_response(tmp_path / "responses", "e", "m/other", "big form 1", 40_000)
    write_response(tmp_path / "responses", "f", "m/new", "sanity form 1", 100)
    molecule_of_prompt = {"big form 1": "big", "big form 3": "big", "small form 1": "small", "sanity form 1": "sanity"}
    use = batch.saved_token_use(molecule_of_prompt)
    assert use == {"m/one": {"big": 20_000, "small": 1_000}, "m/other": {"big": 40_000}, "m/new": {"sanity": 100}}
    rows = [{"molecule_id": "big"}, {"molecule_id": "small"}, {"molecule_id": "sanity"}, {"molecule_id": "unseen"}]
    assert batch.estimated_cost("m/one", rows[:2], use) == (20_000 + 1_000) / 1e6  # its own averages
    # a model that has only answered the sanity-check molecule: the most any model used on the others
    assert batch.estimated_cost("m/new", rows, use) == (40_000 + 1_000 + 100 + openrouter.MAX_TOKENS) / 1e6


def test_balance_is_credits_bought_minus_credits_used(monkeypatch):
    monkeypatch.setattr(batch, "call_api", lambda method, path: {"data": {"total_credits": 30.0, "total_usage": 15.89}})
    assert round(batch.balance(), 2) == 14.11


def test_batches_over_the_balance_are_refused_before_anything_is_sent():
    """Each batch is checked on its own against the balance, as OpenRouter does."""
    worst = {"a/haiku": 0.48, "a/sonnet": 9.72, "a/opus": 19.44}
    assert batch.refused_for_balance(worst, 14.11) == ["a/opus"]
    assert batch.refused_for_balance(worst, 20.0) == []
