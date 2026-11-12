#### Preamble ####
# Purpose: Decide which trial a run of the benchmark belongs to, and keep the list of trials and their dates
# Author: Sean Murphy
# Date: 12 November 2026
# Pre-requisites: none
#
# A trial is one run of the whole benchmark. The rule:
# - The first run on a new day starts a new trial.
# - Running again on the same day continues that day's trial: only questions without a saved response are asked
#   (e.g. after rate limits), and a model added later that day joins the same trial.
# - A new trial on the same day can be started on purpose (new_trial=True).
# The trial number is part of each saved response's file name, so a new trial makes new calls.

import polars as pl


def read_trials(path):
    """The trials so far (columns trial and date), or an empty table if there are none yet."""
    if not path.exists():
        return pl.DataFrame(schema={"trial": pl.Int64, "date": pl.String})
    return pl.read_csv(path, schema_overrides={"trial": pl.Int64, "date": pl.String})


def choose_trial(trials, today, new_trial=False):
    """The trial number for a run today, and whether it is a new trial. `today` is a date as text, e.g. "2026-11-12"."""
    if trials.height == 0:
        return 1, True
    last = trials.sort("trial").row(-1, named=True)
    if last["date"] == today and not new_trial:
        return last["trial"], False
    return last["trial"] + 1, True


def record_trial(path, trials, trial, today):
    """Add a new trial and its date to the list of trials."""
    added = pl.concat([trials, pl.DataFrame({"trial": [trial], "date": [today]})])
    path.parent.mkdir(parents=True, exist_ok=True)
    added.write_csv(path)
    return added
