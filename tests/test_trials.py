#### Preamble ####
# Purpose: Test the rule for which trial a run belongs to
# Author: Sean Murphy
# Date: 12 November 2026
# Pre-requisites: none
# Run with `uv run pytest`.

import polars as pl

from smiley_bench.trials import choose_trial, read_trials, record_trial


def trials(*rows):
    return pl.DataFrame(rows, schema={"trial": pl.Int64, "date": pl.String}, orient="row")


def test_first_run_ever_is_trial_1():
    assert choose_trial(trials(), "2026-11-12") == (1, True)


def test_running_again_the_same_day_continues_that_trial():
    assert choose_trial(trials((1, "2026-11-12")), "2026-11-12") == (1, False)


def test_a_new_day_starts_a_new_trial():
    assert choose_trial(trials((1, "2026-11-12")), "2026-11-13") == (2, True)


def test_a_new_trial_can_be_started_the_same_day():
    assert choose_trial(trials((1, "2026-11-12")), "2026-11-12", new_trial=True) == (2, True)


def test_continues_the_latest_trial():
    """Two trials on 12 November (one started on purpose); a third run that day continues trial 2."""
    assert choose_trial(trials((1, "2026-11-12"), (2, "2026-11-12")), "2026-11-12") == (2, False)


def test_trials_are_saved_and_read_back(tmp_path):
    path = tmp_path / "trials.csv"
    assert read_trials(path).height == 0
    record_trial(path, read_trials(path), 1, "2026-11-12")
    record_trial(path, read_trials(path), 2, "2026-11-13")
    assert read_trials(path).rows() == [(1, "2026-11-12"), (2, "2026-11-13")]
