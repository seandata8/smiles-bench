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
    monkeypatch.setattr(
        batch, "finish", lambda model, test, b, rows, trial: finished.append(model) or {"model": model, "test": test}
    )
    monkeypatch.setattr(batch, "POLL_SECONDS", 0)

    batch_ids = {("a/claude-opus", "rings"): "id-opus", ("a/claude-haiku", "atoms"): "id-haiku"}
    summary = batch.finish_all(batch_ids, rows=[], trial=1)

    assert finished == ["a/claude-haiku", "a/claude-opus"]  # each saved as soon as it is done
    assert [(s["model"], s["test"]) for s in summary] == list(batch_ids)  # in the order given
    first_line = capsys.readouterr().out.splitlines()[0]
    assert first_line.endswith("claude-opus: rings ?/?; claude-haiku: atoms ?/?")


def test_progress_line_groups_each_models_batches():
    lines = [
        (
            "google/gemini-lite",
            batch.progress("mass", {"status": "in_progress", "request_counts": {"completed": 0, "total": 3}}),
        ),
        ("google/gemini-lite", batch.progress("rings", {"status": "validating"})),
        (
            "openai/luna",
            batch.progress("atoms", {"status": "in_progress", "request_counts": {"completed": 2, "total": 30}}),
        ),
    ]
    assert batch.progress_line(lines) == "gemini-lite: mass 0/3 | rings validating ?/?; luna: atoms 2/30"


def test_id_file_has_the_test_in_its_name_except_for_old_batches_of_every_test(monkeypatch, tmp_path):
    monkeypatch.setattr(batch, "folder", lambda: tmp_path)
    assert (
        batch.id_file("google/gemini-3.5-flash-lite", 1, "rings").name
        == "google_gemini-3.5-flash-lite_rings_trial1.json"
    )
    assert batch.id_file("minimax/minimax-m3", 1, "all").name == "minimax_minimax-m3_trial1.json"


def test_pending_gives_old_batches_the_test_all(monkeypatch, tmp_path):
    monkeypatch.setattr(batch, "folder", lambda: tmp_path)
    (tmp_path / "minimax_minimax-m3_trial1.json").write_text(
        '{"batch_id": "b1", "model": "minimax/minimax-m3", "trial": 1}'
    )
    (tmp_path / "x_m_rings_trial1.json").write_text('{"batch_id": "b2", "model": "x/m", "test": "rings", "trial": 1}')
    (tmp_path / "batch_b0.json").write_text("{}")  # a finished batch, kept: not pending
    assert sorted((p["model"], p["test"]) for p in batch.pending()) == [("minimax/minimax-m3", "all"), ("x/m", "rings")]


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
    """Per prompt: the model's own average on that molecule; a molecule it has not answered: the median of the other
    models' averages on it; a molecule no model has answered: MAX_TOKENS."""
    monkeypatch.setattr(paths, "RAW_DATA", tmp_path)
    monkeypatch.setattr(batch, "batch_price", lambda model: (0.0, 1e-6))  # output tokens only: $1 per million
    write_response(tmp_path / "pilot" / "responses", "a", "m/one", "big form 1", 30_000)
    write_response(tmp_path / "pilot" / "responses", "b", "m/one", "big form 3", 10_000)
    write_response(tmp_path / "max-run" / "responses", "c", "m/one", "small form 1", 1_000)
    write_response(tmp_path / "pilot" / "responses", "d", "m/one", "a prompt no longer asked", 99_000)
    write_response(tmp_path / "responses", "e", "m/other", "big form 1", 40_000)
    write_response(tmp_path / "responses", "g", "m/third", "big form 1", 5_000)
    write_response(tmp_path / "responses", "f", "m/new", "sanity form 1", 100)
    molecule_of_prompt = {"big form 1": "big", "big form 3": "big", "small form 1": "small", "sanity form 1": "sanity"}
    use = batch.saved_token_use(molecule_of_prompt)
    assert use == {
        "m/one": {"big": 20_000, "small": 1_000},
        "m/other": {"big": 40_000},
        "m/third": {"big": 5_000},
        "m/new": {"sanity": 100},
    }
    rows = [{"molecule_id": "big"}, {"molecule_id": "small"}, {"molecule_id": "sanity"}, {"molecule_id": "unseen"}]
    assert batch.estimated_cost("m/one", rows[:2], use) == (20_000 + 1_000) / 1e6  # its own averages
    # a model that has only answered the sanity-check molecule: the median of the other models on the others
    # (big: 20,000, 40,000 and 5,000, so 20,000; small: only m/one's 1,000)
    assert batch.estimated_cost("m/new", rows, use) == (20_000 + 1_000 + 100 + openrouter.MAX_TOKENS) / 1e6


def test_balance_is_credits_bought_minus_credits_used(monkeypatch):
    monkeypatch.setattr(batch, "call_api", lambda method, path: {"data": {"total_credits": 30.0, "total_usage": 15.89}})
    assert round(batch.balance(), 2) == 14.11


def test_batches_over_the_balance_are_refused_before_anything_is_sent():
    """Each batch is checked on its own against the balance, as OpenRouter does."""
    worst = {"a/haiku": 0.48, "a/sonnet": 9.72, "a/opus": 19.44}
    assert batch.refused_for_balance(worst, 14.11) == ["a/opus"]
    assert batch.refused_for_balance(worst, 20.0) == []


def test_check_waits_out_a_timeout(monkeypatch):
    """A status check that times out does not stop the run: the batch is checked again later."""

    def times_out(method, path):
        raise TimeoutError("The read operation timed out")

    monkeypatch.setattr(batch, "call_api", times_out)
    result = batch.check("batch-1")
    assert result == {"status": "no reply"} and not batch.is_done(result)
