#### Preamble ####
# Purpose: Test the batch waiting loop without calling OpenRouter: fake batches finish after a set number of checks
# Author: Sean Murphy
# Date: 12 November 2026
# Pre-requisites: none
# Run with `uv run pytest`.

from smiley_bench import batch


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
