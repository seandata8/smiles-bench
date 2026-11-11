#### Preamble ####
# Purpose: Check that the analysis recovers the known accuracies and differences of the fake models, choose the
#          interval method, and see how many questions we need
# Author: Sean Murphy
# Date: 11 November 2026
# Pre-requisites: run scripts/00-simulate_data.py
# Checks:
# 1. On the saved simulated data: the summary table, the comparisons between models, and the figure, with each
#    model's true accuracy marked.
# 2. Coverage of each interval method: over many simulated datasets, a 95% interval should contain the true
#    accuracy about 95% of the time. Too low means the interval is too narrow.
# 3. False positives of the comparison: two identical models should differ at p < 0.05 about 5% of the time.
# 4. Power of the comparison: how often it finds a real difference, by size of difference and number of questions.

import polars as pl

from smiley_bench import paths
from smiley_bench.analysis import INTERVAL_METHODS, compare_models, summarise
from smiley_bench.figures import plot_pass_at_1
from smiley_bench.simulation import FAKE_MODELS, simulate_results, true_accuracy

TRIALS = 5
SIMULATIONS_PER_SIZE = 100  # simulated datasets for each number of questions
QUESTIONS_PER_LEVEL_TO_TRY = [5, 10, 20, 50]
PERMUTATIONS = 2_000  # fewer than the default, to keep the many simulations quick
FIGURES = paths.ROOT / "other" / "validation"

# Two copies of the same model: any difference found between them is a false positive.
TWIN_MODELS = {"twin-a": FAKE_MODELS["fake-cheap"], "twin-b": FAKE_MODELS["fake-cheap"]}

truth = true_accuracy()

# The true difference in accuracy for every pair of fake models at each level, to check the comparisons against.
true_differences = (
    truth.join(truth, on="level", suffix="_b")
    .filter(pl.col("model") < pl.col("model_b"))  # each pair once, in the same order as compare_models
    .select(
        "level",
        model_a="model",
        model_b="model_b",
        true_difference=pl.col("accuracy") - pl.col("accuracy_b"),
    )
)


def pass_at_1_with_truth(results, method):
    """pass@1 and its interval for each model and level, with the true accuracy and whether the interval has it."""
    summary = summarise(results, k=TRIALS, method=method, metrics=["pass@1"])
    return summary.join(truth, on=["model", "level"]).with_columns(
        covered=(pl.col("lower") <= pl.col("accuracy")) & (pl.col("accuracy") <= pl.col("upper")),
        width=pl.col("upper") - pl.col("lower"),
    )


def size_of_difference(true_difference):
    """Group true differences in accuracy into bands, in percentage points."""
    gap = abs(true_difference)
    if gap < 0.15:
        return "1: under 15 points"
    if gap < 0.30:
        return "2: 15 to 30 points"
    return "3: 30 points or more"


# 1. The saved simulated data.
results = pl.read_csv(paths.SIMULATED_DATA / "simulated_results.csv")
summary = summarise(results, k=TRIALS)
comparisons = compare_models(results)
summary.write_csv(paths.SIMULATED_DATA / "simulated_summary.csv")
comparisons.write_csv(paths.SIMULATED_DATA / "simulated_comparisons.csv")
FIGURES.mkdir(parents=True, exist_ok=True)
plot_pass_at_1(summary, comparisons=comparisons, truth=truth).savefig(FIGURES / "simulated_pass_at_1.png", dpi=150)

with pl.Config(tbl_rows=-1, float_precision=3):
    print("Summary of the saved simulated data:")
    print(summary.pivot(on="metric", index=["model", "level"], values="estimate"))
    print(
        pass_at_1_with_truth(results, summary["interval"][0]).select(
            "model", "level", "accuracy", "estimate", "lower", "upper", "covered"
        )
    )
    print(comparisons.join(true_differences, on=["level", "model_a", "model_b"]))

# 2 to 4. Many simulated datasets of each size.
coverage_rows = []
false_positive_rows = []
power_rows = []
for questions_per_level in QUESTIONS_PER_LEVEL_TO_TRY:
    for simulation in range(SIMULATIONS_PER_SIZE):
        seed = 10_000 + simulation
        simulated = simulate_results(questions_per_level, TRIALS, seed)

        for method in INTERVAL_METHODS:
            checked = pass_at_1_with_truth(simulated, method)
            coverage_rows.append(
                {
                    "questions_per_level": questions_per_level,
                    "method": method,
                    "coverage": checked["covered"].mean(),
                    "average_width": checked["width"].mean(),
                }
            )

        compared = compare_models(simulated, n_permutations=PERMUTATIONS, seed=seed)
        for row in compared.join(true_differences, on=["level", "model_a", "model_b"]).iter_rows(named=True):
            power_rows.append(
                {
                    "questions_per_level": questions_per_level,
                    "size_of_difference": size_of_difference(row["true_difference"]),
                    "found": row["p_adjusted"] < 0.05,
                }
            )

        twins = compare_models(
            simulate_results(questions_per_level, TRIALS, seed, models=TWIN_MODELS),
            n_permutations=PERMUTATIONS,
            seed=seed,
        )
        false_positive_rows.extend(
            {"questions_per_level": questions_per_level, "false_positive": p < 0.05} for p in twins["p_value"]
        )

by_size = (
    pl.DataFrame(coverage_rows)
    .group_by("questions_per_level", "method")
    .agg(pl.col("coverage", "average_width").mean())
    .sort("questions_per_level", "method")
)
false_positives = (
    pl.DataFrame(false_positive_rows)
    .group_by("questions_per_level")
    .agg(false_positive_rate=pl.col("false_positive").mean())
    .sort("questions_per_level")
)
power = (
    pl.DataFrame(power_rows)
    .group_by("questions_per_level", "size_of_difference")
    .agg(power=pl.col("found").mean())
    .sort("questions_per_level", "size_of_difference")
    .pivot(on="size_of_difference", index="questions_per_level", values="power")
)
by_size.write_csv(paths.SIMULATED_DATA / "validation_by_number_of_questions.csv")
false_positives.write_csv(paths.SIMULATED_DATA / "validation_false_positives.csv")
power.write_csv(paths.SIMULATED_DATA / "validation_power.csv")

with pl.Config(float_precision=3):
    print(f"\nOver {SIMULATIONS_PER_SIZE} simulated datasets per size.")
    print("Coverage of the 95% interval (target 0.95):")
    print(by_size.pivot(on="method", index="questions_per_level", values="coverage"))
    print("Average width of the interval:")
    print(by_size.pivot(on="method", index="questions_per_level", values="average_width"))
    print("False positives, two identical models compared (target 0.05):")
    print(false_positives)
    print("Power: share of real differences found (adjusted p < 0.05), by size of true difference:")
    print(power)
