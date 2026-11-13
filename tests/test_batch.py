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
