#### Preamble ####
# Purpose: The figures of the benchmark's results, shared by the check on simulated data and the website
# Author: Sean Murphy
# Date: 11 November 2026
# Pre-requisites: uv add matplotlib polars
# Follows the sketch in other/sketches/smiley-bench-results.png. Significant differences are marked with brackets
# the way GraphPad Prism does: only pairs with p < 0.05, stacked to use as little height as possible.

import math

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.colors import to_rgb
from matplotlib.patches import Rectangle
from matplotlib.transforms import Bbox
from scipy.stats import t

from smiley_bench.tables import LEVELS

SIGNIFICANCE = 0.05
BRACKET_BASE = 104  # height of the lowest row of brackets, just above 100%
BRACKET_STEP = 9  # height between rows of brackets
BRACKET_INSET = 0.08  # brackets stop just short of each model, so brackets that meet at a model can share a row


def p_value_label(p):
    """p < 0.001, or p rounded to 2 significant figures (p = 0.012, p = 0.034)."""
    return "p < 0.001" if p < 0.001 else f"p = {p:.2g}"


def stack_brackets(pairs):
    """The row (0 = lowest) for each bracket, given each as (left position, right position).

    Shortest brackets are placed first, so a bracket always sits below any longer bracket that spans it. Each bracket
    goes in the lowest row where it does not overlap a bracket already there."""
    rows_taken = []  # for each row, the (left, right) brackets already in it
    row_of = {}
    for left, right in sorted(pairs, key=lambda pair: (pair[1] - pair[0], pair[0])):
        row = 0
        while row < len(rows_taken) and any(
            left + BRACKET_INSET < other_right - BRACKET_INSET and other_left + BRACKET_INSET < right - BRACKET_INSET
            for other_left, other_right in rows_taken[row]
        ):
            row += 1
        if row == len(rows_taken):
            rows_taken.append([])
        rows_taken[row].append((left, right))
        row_of[(left, right)] = row
    return [row_of[pair] for pair in pairs]


def draw_bracket(ax, left, right, row, label):
    """A bracket between two models, in the given row, with its p-value above it."""
    height = BRACKET_BASE + row * BRACKET_STEP
    x_left, x_right = left + BRACKET_INSET, right - BRACKET_INSET
    ax.plot([x_left, x_left, x_right, x_right], [height - 2, height, height, height - 2], color="black", lw=0.8)
    ax.text((x_left + x_right) / 2, height + 0.5, label, ha="center", va="bottom", fontsize=8.4)


def significant_pairs(comparisons, level, position_of):
    """(left position, right position, adjusted p) for each pair of models at this level with p < 0.05."""
    significant = comparisons.filter((pl.col("level") == level) & (pl.col("p_adjusted") < SIGNIFICANCE))
    pairs = []
    for row in significant.iter_rows(named=True):
        a, b = position_of[row["model_a"]], position_of[row["model_b"]]
        pairs.append((min(a, b), max(a, b), row["p_adjusted"]))
    return pairs


def plot_pass_at_1(summary, comparisons=None, truth=None):
    """% correct (pass@1) with its 95% interval for each model, one panel per level.
    Models are in the same order in every panel: best to worst on the first level.
    `comparisons` (from compare_models) adds a bracket for every pair of models that differ at Holm-adjusted p < 0.05.
    `truth` (simulated data only) adds each model's true accuracy as a red cross."""
    pass_at_1 = summary.filter(pl.col("metric") == "pass@1")
    levels = [level for level in LEVELS if level in pass_at_1["level"].to_list()]
    model_order = pass_at_1.filter(pl.col("level") == levels[0]).sort("estimate", descending=True)["model"]
    positions = pl.DataFrame({"model": model_order, "position": range(len(model_order))})

    figure, axes = plt.subplots(1, len(levels), figsize=(4 * len(levels), 4.5), sharey=True, squeeze=False)
    most_rows = 0
    for ax, level in zip(axes[0], levels):
        one_level = pass_at_1.filter(pl.col("level") == level).join(positions, on="model").sort("position")
        models = one_level["model"].to_list()
        x = range(len(models))
        estimate = 100 * one_level["estimate"].to_numpy()
        below = estimate - 100 * one_level["lower"].to_numpy()
        above = 100 * one_level["upper"].to_numpy() - estimate
        ax.errorbar(x, estimate, yerr=[below, above], fmt="o", color="black", capsize=4)
        if truth is not None:
            true_level = one_level.join(truth, on=["model", "level"], how="left")
            ax.scatter(x, 100 * true_level["accuracy"].to_numpy(), marker="x", color="red", zorder=3)
        if comparisons is not None:
            pairs = significant_pairs(comparisons, level, {model: i for i, model in enumerate(models)})
            rows = stack_brackets([(left, right) for left, right, _ in pairs])
            for (left, right, p), row in zip(pairs, rows):
                draw_bracket(ax, left, right, row, p_value_label(p))
            most_rows = max(most_rows, max(rows, default=-1) + 1)
        ax.set_xticks(list(x), models, rotation=45, ha="right")
        ax.set_title(f"{level.capitalize()} ({one_level['questions'][0]} questions)", pad=10)
        ax.set_yticks(range(0, 101, 20))
        ax.spines[["top", "right"]].set_visible(False)
    axes[0][0].set_ylim(0, BRACKET_BASE + most_rows * BRACKET_STEP)  # shared y-axis: room for the tallest stack
    axes[0][0].set_ylabel("% correct (pass@1)")
    figure.tight_layout()
    return figure


# Overall results by price tier (other/notes/models.md). Every free model (":free") is in the free tier; each paid
# model's tier is by its batch output price: low under $2 per million tokens, med $3-6, high $10-25.
TIERS = ["free", "low", "med", "high"]
PAID_TIERS = {
    "low": [
        "deepseek/deepseek-v4.1-flash-1017",
        "z-ai/glm-5.3-flash",
        "anthropic/claude-haiku-5.5",
        "openai/gpt-6-luna",
        "minimax/minimax-m3",
        "qwen/qwen3.7-plus",
        "google/gemini-3.5-flash-lite",
        "particle-ai/particle-1",  # flex endpoint, no batch version (openrouter.FLEX_ENDPOINTS)
    ],
    "med": [
        "qwen/qwen3.8-max-0902",
        "mistralai/mistral-medium-3-5",
        "x-ai/grok-4.7",
        "anthropic/claude-sonnet-5.5",
        "google/gemini-4-carbon",
        "openai/gpt-6.1-sol",
        "moonshotai/kimi-k3.1-pro",
        "google/gemini-4-argon",  # added after the starting set (other/notes/models.md)
    ],
    "high": ["anthropic/claude-opus-5.5", "openai/gpt-5.5", "anthropic/claude-fable-5.2", "openai/gpt-6-astra",
             "google/gemini-4-neon"],  # batch $7.50 out: between med and high; Sean put it in high (other/notes/models.md)
}
# One hue per tier, from the categorical slots blue, orange, aqua and violet: checked with the dataviz skill's
# palette validator (lightness, chroma, colour-blind and normal-vision separation of neighbours all pass). The aqua
# is pale on white, so every bar carries its model's name and value as text.
TIER_COLORS = {"free": "#2a78d6", "low": "#eb6834", "med": "#1baf7a", "high": "#6250d6"}
# Tiers by what a model's answers to the whole benchmark actually cost, not by its price per token: a model that
# writes many more tokens can cost more than a pricier one. Same four tiers: "free" is under $0.10 (spend_tier()).


def spend_group_labels(costs):
    """The label over each group of models on the chart by what their answers cost (spend_tier()), e.g.
    "Cost $4–24": the top of the last group is the highest of `costs`, rounded up to the next dollar."""
    top = math.ceil(max(costs))
    return {"free": "Cost <10¢", "low": "Cost 10¢–\\$1", "med": "Cost \\$1–3", "high": f"Cost \\$4–{top}"}


def tier(model):
    """The price tier of an OpenRouter model id: free, low, med or high. An error for a model with no tier yet."""
    if model.endswith(":free"):
        return "free"
    for name, models in PAID_TIERS.items():
        if model in models:
            return name
    raise ValueError(f"{model} has no price tier: add it to PAID_TIERS in figures.py (see other/notes/models.md)")


def spend_tier(cost):
    """The tier of what a model's answers cost in dollars: free (under $0.10), low ($0.10-1.00), med ($1.01-3.00) or
    high (over $4.00)."""
    if cost < 0.10:
        return "free"
    if cost <= 1.00:
        return "low"
    if cost <= 3.00:
        return "med"
    if cost > 4.00:
        return "high"
    raise ValueError(f"${cost:.2f} falls between the med and high tiers ($3.00-4.00): choose a tier for it")


def shades(color, n):
    """n shades of one colour, darkest first, so the models of one tier share its hue but can be told apart: from
    30% darker to 40% lighter, the middle one the colour itself."""
    if n == 1:
        return [color]
    base = np.array(to_rgb(color))
    steps = np.linspace(-0.3, 0.4, n)  # below 0: mixed with black; above 0: mixed with white
    return [tuple(base * (1 + s) if s < 0 else base + (1 - base) * s) for s in steps]


# Shorter names for models whose OpenRouter names are long, as they appear on the charts.
DISPLAY_NAMES = {
    "cohere/north-mini-code:free": "north-mini",
    "dots-studio/dots-3-note-preview:free": "dots-3",
    "recursive/flint-v1.0:free": "flint-v1.0",
    "baidu/ernie-5.5-flash-preview:free": "ernie-5.5",
    "nvidia/nemotron-3.5-super-vl:free": "nemotron-3.5",
    "particle-ai/particle-1:free": "particle-1",
    "particle-ai/particle-1": "particle-1",
    "deepseek/deepseek-v4.1-flash-1017": "deepseek-v4.1",
    "z-ai/glm-5.3-flash": "glm-5.3",
    "google/gemini-3.5-flash-lite": "gemini-3.5-lite",
    "qwen/qwen3.8-max-0902": "qwen3.8-max",
    "mistralai/mistral-medium-3-5": "mistral-medium-3.5",
    "moonshotai/kimi-k3.1-pro": "kimi-k3.1",
}


def model_colors(models, tier_of=None):
    """A colour for each model: its tier's hue, in a shade given by its name in alphabetical order among the models of
    its tier here (see shades()). Both charts call this with the same models, so a model has the same colour in each.
    tier_of: {model: tier}, if not the price tiers (tier())."""
    tier_of = tier_of or {model: tier(model) for model in models}
    colors = {}
    for name in TIERS:
        in_tier = sorted(model for model in models if tier_of[model] == name)
        colors.update(zip(in_tier, shades(TIER_COLORS[name], len(in_tier))))
    return colors


def short_name(model):
    """The model's name on the charts: its name in DISPLAY_NAMES, or else its OpenRouter name without the company or
    ":free", e.g. "claude-opus-5.5" for anthropic/claude-opus-5.5."""
    return DISPLAY_NAMES.get(model, model.split("/")[-1].removesuffix(":free"))


COST_LINE = 9  # height on the y-axis (in % points) of the line of small type that gives a model's cost


def cost_label(cost):
    """A model's cost in dollars for the chart: "$0" for a free model, "<$0.01" below a cent, else e.g. "$0.83"."""
    if cost == 0:
        return "$0"
    return "<$0.01" if cost < 0.01 else f"${cost:.2f}"


def plot_overall(
    scores,
    questions,
    comparisons=None,
    tier_of=None,
    tier_labels=None,
    legend_title="Price tier",
    height=None,
    group_labels=None,
    score_marks=None,
    extra_notes=(),
    group_order=None,
):
    """A bar for each model's overall % correct (scores: model, score, answered, and optionally lower and upper, its
    95% interval in %, and cost, the dollars its answers cost, shown above the bar), grouped by tier from free to high, one hue per tier and a shade of it per model. Within a tier
    the bars go from lowest score to highest. Each model's shade is given by its name, in alphabetical order within
    its tier, not by its place in the order, so it keeps its shade as scores change. Each bar is labelled with its
    model and value (above its error bar, if it has one); a model that has answered fewer than all `questions` also
    is marked with * and its count of answered questions is given in a note below the chart.

    tier_of: {model: tier}, to group the models other than by price tier (e.g. by what they cost: spend_tier()), with
    tier_labels ({tier: legend label}) and legend_title for the legend. height: the figure's height in inches, if not
    the usual one.

    group_labels: {tier: label}, to colour the bars by company instead (company_color(), as on the cost-vs-score
    chart), with a labelled line over each group of bars in place of the tier legend, and a small legend of the
    companies.

    score_marks: {model: mark}, drawn above the model's score (e.g. "*"), and extra_notes: lines added to the note
    below the chart, e.g. to say what the mark means.

    group_order: the names of tier_of's groups, left to right, if they are not the price tiers (TIERS).

    `comparisons` (from compare_models, run within each tier: model_a, model_b, p_adjusted) adds a bracket for every
    pair of models that differ at Holm-adjusted p < 0.05, stacked as in plot_pass_at_1()."""
    tier_of = tier_of or {model: tier(model) for model in scores["model"]}
    tier_labels = tier_labels or {}
    scores = scores.with_columns(tier=pl.col("model").replace_strict(tier_of))
    has_intervals = "lower" in scores.columns
    has_costs = "cost" in scores.columns
    height = height or (6.4 if comparisons is not None else 5.6)
    figure, ax = plt.subplots(figsize=(1.0 * scores.height + 2.5, height))
    position, ticks, labels = 0.0, [], []
    position_of = {}
    if group_labels:
        color_of = {model: company_color(model) for model in scores["model"]}
    else:
        color_of = model_colors(scores["model"].to_list(), tier_of)
    group_ends = {}  # the first and last bar of each group: (left position, right position)
    for name in group_order or TIERS:
        in_tier = scores.filter(pl.col("tier") == name)
        if in_tier.height == 0:
            continue
        for row in in_tier.sort("score").iter_rows(named=True):
            model, score, answered = row["model"], row["score"], row["answered"]
            color = color_of[model]
            ax.bar(position, score, width=0.8, color=color, edgecolor="white", linewidth=2)
            top = score
            if has_intervals:
                below, above = score - row["lower"], row["upper"] - score
                ax.errorbar(position, score, yerr=[[below], [above]], color="#0b0b0b", capsize=4, linewidth=1)
                top = row["upper"]
            label = f"{score:.0f}%" if answered == questions else f"{score:.0f}%*"  # the count is in the note below
            if has_costs:  # the cost in small grey type just above the bar, the score above it
                ax.text(
                    position, top + 1.5, cost_label(row["cost"]), ha="center", va="bottom", fontsize=15.6, color="#5f5e5a"
                )
                top += COST_LINE
            ax.text(position, top + 1.5, label, ha="center", va="bottom", fontsize=17, color="#0b0b0b")
            if model in (score_marks or {}):  # one line higher, above the score
                ax.text(position, top + 1.5 + COST_LINE, score_marks[model], ha="center", va="bottom", fontsize=17,
                        color="#0b0b0b")
            position_of[model] = position
            ticks.append(position)
            labels.append(short_name(model))
            position += 1
        group_ends[name] = (position_of[in_tier.sort("score")["model"][0]], position - 1)
        if not group_labels:
            ax.bar(0, 0, color=TIER_COLORS[name], label=tier_labels.get(name, name))  # one legend entry per tier
        position += 0.6  # a gap between tiers
    ax.set_xticks(ticks, labels, rotation=30, ha="right", fontsize=17)
    top = 105 + (COST_LINE if has_costs else 0) + (COST_LINE if score_marks else 0)
    if comparisons is not None:
        drawn = comparisons.filter(
            pl.col("model_a").is_in(list(position_of)) & pl.col("model_b").is_in(list(position_of))
        )
        pairs = significant_pairs(drawn, "overall", position_of)
        rows = stack_brackets([(left, right) for left, right, _ in pairs])
        for (left, right, p), row in zip(pairs, rows):
            draw_bracket(ax, left, right, row, p_value_label(p))
        top = max(top, BRACKET_BASE + (max(rows, default=-1) + 1) * BRACKET_STEP)
    if group_labels:  # a line over each group of bars, with its label above it, all at one height
        line = top + 5
        for name, (left, right) in group_ends.items():
            ax.plot([left - 0.4, left - 0.4, right + 0.4, right + 0.4], [line - 2, line, line, line - 2],
                    color="#5f5e5a", lw=1.2)
            ax.text((left + right) / 2, line + 1.5, group_labels[name], ha="center", va="bottom", fontsize=19,
                    color="#0b0b0b")
        top = line + 11
        ax.spines["left"].set_bounds(0, 100)
    ax.set_ylim(0, top)
    ax.set_yticks(range(0, 101, 20))
    ax.set_ylabel("% correct (95% CI)" if has_intervals else "% correct", fontsize=18)
    ax.tick_params(axis="y", labelsize=17)
    ax.yaxis.grid(True, color="#e4e3df", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#8a8984")
    if group_labels:  # a small legend of the companies, in the empty space above the lowest-scoring bars
        for label, color in company_legend(scores["model"]).items():
            ax.bar(0, 0, color=color, label=label)
        ax.bar(0, 0, color=FREE_COLOR, label="Free models")
        legend_top = 100 + (COST_LINE if score_marks else 0)  # raised by the marks' line, clear of the low bars' marks
        ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.01, legend_top / top), ncol=2, fontsize=15.6,
                  columnspacing=1.2, handletextpad=0.4)
    else:  # the legend in one row above the bars, where it cannot cover a label
        ax.legend(
            title=legend_title, frameon=False, loc="lower left", bbox_to_anchor=(0, 1.02), ncol=len(TIERS),
            fontsize=17, title_fontsize=18, alignment="left",
        )
    notes = list(extra_notes)
    unfinished = scores.filter(pl.col("answered") < questions).sort("model")
    if unfinished.height:
        counts = ", ".join(
            f"{short_name(model)} {answered} of {questions}" for model, answered in unfinished.select("model", "answered").rows()
        )
        notes.append(f"* Unfinished, scored on the questions answered so far: {counts}")
    if comparisons is not None:
        notes.append(
            "Brackets: models in the same tier that differ (paired sign-flip test, Holm-adjusted within the tier)"
        )
    if notes:
        figure.text(0.01, 0.01, "\n".join(notes), ha="left", va="bottom", fontsize=14, color="#5f5e5a")
    figure.tight_layout(rect=(0, 0.02 + 0.03 * len(notes), 1, 1))
    return figure


def plot_groups_by_test(summary, groups, tests, width=9, wrong=False, error_bars=True, n_column="questions"):
    """% correct on each test (pass@1, with its 95% interval) for groups of models: the tests along the x-axis, one
    bar per group in each, labelled with its value.

    summary: from summarise(), with each group as the "model" and the test as the level.
    groups: {group name: colour}, in the order to draw them. tests: {test: label on the x-axis}, in order.
    width: the figure's width in inches. wrong: draw the % wrong instead (100 minus the % correct, its interval
    flipped the same way), with the y-axis only as high as the bars need, so small differences can be seen while
    the bars still start at 0. error_bars: False to leave out the 95% intervals.
    Each bar's n, from the summary's n_column (by default its number of questions, the unit of the interval), is
    shown: under each test's label when there is one group, or after each group's name in the legend when there are
    more (each group has the same n in every test)."""
    figure, ax = plt.subplots(figsize=(width, 4.8), layout="constrained")
    width = 0.8 / len(groups)
    highest = 0  # the top of the highest bar or error bar
    tick_labels = list(tests.values())
    for i, (group, color) in enumerate(groups.items()):
        one = summary.filter(pl.col("model") == group)
        n = dict(zip(one["level"], one[n_column]))
        if len(groups) == 1:
            tick_labels = [f"{label}\nn = {n[test]}" for test, label in tests.items()]
        else:
            if len(set(n.values())) > 1:
                raise ValueError(f"{group} has a different n in different tests: {n}")
            group_label = f"{group} (n = {n[next(iter(tests))]})"
        estimate = dict(zip(one["level"], 100 * one["estimate"]))
        lower = dict(zip(one["level"], 100 * one["lower"]))
        upper = dict(zip(one["level"], 100 * one["upper"]))
        if wrong:
            estimate = {test: 100 - value for test, value in estimate.items()}
            lower, upper = (
                {test: 100 - value for test, value in upper.items()},
                {test: 100 - value for test, value in lower.items()},
            )
        if not error_bars:  # each label goes just above its bar instead of above its error bar
            upper = estimate
        highest = max(highest, *upper.values())
        x = [j - 0.4 + width * (i + 0.5) for j in range(len(tests))]
        heights = [estimate[test] for test in tests]
        below = [estimate[test] - lower[test] for test in tests]
        above = [upper[test] - estimate[test] for test in tests]
        ax.bar(x, heights, width=width, color=color, edgecolor="white", linewidth=2,
               label=group if len(groups) == 1 else group_label)
        if error_bars:
            ax.errorbar(x, heights, yerr=[below, above], fmt="none", color="#0b0b0b", capsize=4, linewidth=1)
        for x_bar, height, test in zip(x, heights, tests):
            ax.annotate(f"{height:.0f}%", (x_bar, upper[test]), xytext=(0, 4), textcoords="offset points", ha="center",
                        va="bottom", fontsize=11, color="#0b0b0b")
    ax.set_xticks(range(len(tests)), tick_labels, fontsize=12)
    ax.tick_params(axis="y", labelsize=12)
    if wrong:
        top = 10 * math.ceil((highest + 4) / 10)  # room for the labels, up to the next 10%
        ax.set_ylim(0, top)
        ax.set_yticks(range(0, top + 1, 10))
        ax.set_ylabel("% wrong (95% CI)" if error_bars else "% wrong", fontsize=12)
    else:
        ax.set_ylim(0, 108)
        ax.set_yticks(range(0, 101, 20))
        ax.set_ylabel("% correct (95% CI)" if error_bars else "% correct", fontsize=12)
    ax.yaxis.grid(True, color="#e4e3df", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#8a8984")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=len(groups), fontsize=12)
    return figure


def plot_stacked_wrong(wrong, levels, kinds, n_labels, width=9):
    """% wrong at each level as one bar, split into segments by kind, coloured by kind: each segment is the kind's
    wrong answers as a % of all the level's answers, so the segments add up to the bar's % wrong.

    wrong: {(level, kind): % of the level's answers that are wrong and of that kind}. levels: {level: label on the
    x-axis}, in order. kinds: {kind: colour}, in the order to stack them, bottom up. n_labels: {level: text under
    its label}, e.g. "n = 11"."""
    figure, ax = plt.subplots(figsize=(width, 4.8), layout="constrained")
    bottoms = [0.0] * len(levels)
    for kind, color in kinds.items():
        heights = [wrong.get((level, kind), 0.0) for level in levels]
        ax.bar(range(len(levels)), heights, bottom=bottoms, width=0.6, color=color, edgecolor="white", linewidth=2,
               label=kind.replace("\n", " "))
        bottoms = [bottom + height for bottom, height in zip(bottoms, heights)]
    for x, total in enumerate(bottoms):
        ax.annotate(f"{total:.0f}%", (x, total), xytext=(0, 4), textcoords="offset points", ha="center", va="bottom",
                    fontsize=11, color="#0b0b0b")
    ax.set_xticks(range(len(levels)), [f"{label}\n{n_labels[level]}" for level, label in levels.items()], fontsize=12)
    ax.tick_params(axis="y", labelsize=12)
    top = 10 * math.ceil((max(bottoms) + 4) / 10)  # room for the labels, up to the next 10%
    ax.set_ylim(0, top)
    ax.set_yticks(range(0, top + 1, 10))
    ax.set_ylabel("% wrong", fontsize=12)
    ax.yaxis.grid(True, color="#e4e3df", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#8a8984")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=len(kinds), fontsize=12)
    return figure


# Each company's colour on the cost-vs-score chart: its corporate colour where one is known and the colours can still
# be told apart; every free model is grey, whoever makes it. DeepSeek's blue (#4D6BFE) is darkened so it is not taken
# for Google's, and Moonshot and Zhipu, whose brands are black and white, get a purple and a mustard. Every model
# named model is labelled, so it is not told by colour alone.
COMPANY_COLORS = {
    "anthropic": "#d97757",  # Claude's terracotta
    "openai": "#10a37f",  # ChatGPT green
    "google": "#4285f4",  # Google blue
    "x-ai": "#000000",
    "mistralai": "#fa520f",  # Mistral orange
    "minimax": "#e73562",
    "deepseek": "#2a3f9e",
    "moonshotai": "#7c3aed",
    "z-ai": "#c99a06",
    "particle-ai": "#2a3f9e",  # DeepSeek's: particle-1 is post-trained on DeepSeek V4.1 Flash (OpenRouter's listing)
}
FREE_COLOR = "#a8a7a2"


def company_color(model):
    """A model's colour: its company's (COMPANY_COLORS), or grey for a free model."""
    return FREE_COLOR if model.endswith(":free") else COMPANY_COLORS[model.split("/")[0]]

COMPANY_NAMES = {
    "anthropic": "Anthropic", "openai": "OpenAI", "google": "Google", "x-ai": "xAI", "mistralai": "Mistral",
    "minimax": "MiniMax", "deepseek": "DeepSeek", "moonshotai": "Moonshot", "z-ai": "Zhipu",
    "particle-ai": "Particle AI",
}


def company_legend(models):
    """The legend entries for the paid companies among `models`: {label: colour}, in alphabetical order. Particle AI
    has no entry of its own: its model is built on DeepSeek's and has DeepSeek's colour (COMPANY_COLORS)."""
    companies = sorted({model.split("/")[0] for model in models if not model.endswith(":free")} - {"particle-ai"},
                       key=lambda company: COMPANY_NAMES[company])
    names_of = {}
    for company in companies:
        names_of.setdefault(COMPANY_COLORS[company], []).append(COMPANY_NAMES[company])
    return {", ".join(names): color for color, names in names_of.items()}


# On the cost-vs-score chart, only the models that cost at least NAMED_FROM dollars or score over NAMED_ABOVE % are named
NAMED_FROM = 1.00
NAMED_ABOVE = 50
# The named models' names on the cost-vs-score chart, shorter than on the bar charts
SCATTER_NAMES = {
    "openai/gpt-6.1-sol": "sol",
    "google/gemini-4-carbon": "carbon",
    "openai/gpt-6-astra": "astra",
    "anthropic/claude-opus-5.5": "opus",
    "anthropic/claude-sonnet-5.5": "sonnet",
    "anthropic/claude-fable-5.2": "fable",
    "moonshotai/kimi-k3.1-pro": "kimi",
    "x-ai/grok-4.7": "grok",
    "anthropic/claude-haiku-5.5": "haiku",
    "deepseek/deepseek-v4.1-flash-1017": "deepseek",
    "particle-ai/particle-1:free": "particle",
    "particle-ai/particle-1": "particle",
    "google/gemini-4-argon": "argon",
    "openai/gpt-6-luna": "luna",
}
# Where a model's name goes, in points from its dot, where the default (just to the right) would overlap another
LABEL_OFFSETS = {
    "openai/gpt-6-astra": (-9, 0),
    "google/gemini-4-carbon": (9, -12),
    "x-ai/grok-4.7": (-9, 0),
    "google/gemini-4-argon": (9, 6),
    "deepseek/deepseek-v4.1-flash-1017": (9, -12),
}
# Models too close to others to name beside their dot: the name goes at this cost and % correct, with a thin line
# to the dot (free particle-1 to the right; haiku above and to the left, clear of particle-1)
LABEL_PLACES = {
    "particle-ai/particle-1:free": (1.5, 68),
    "anthropic/claude-haiku-5.5": (-0.75, 80),
}


DOT_SIZE = 230  # area of each dot, in points squared
# Two groups of dots that would sit on top of each other are spread sideways, evenly and centred on their real cost,
# this many dollars apart: the free models at $0, and the models that score 10-15%, all at $0-0.05.
SPREAD_STEPS = {"at $0": 0.2, "10-15%": 0.35}


def spread_sideways(scores):
    """Each model's place on the cost axis: its cost, plus a small sideways shift for the models in one of the two
    crowded groups (SPREAD_STEPS), highest score furthest left within the group."""
    in_band = pl.col("score").is_between(10, 15)
    groups = {"10-15%": scores.filter(in_band), "at $0": scores.filter(~in_band & (pl.col("cost") == 0))}
    x = dict(scores.select("model", "cost").rows())
    for name, group in groups.items():
        models = group.sort("score", descending=True)["model"].to_list()
        for i, model in enumerate(models):
            x[model] += (i - (len(models) - 1) / 2) * SPREAD_STEPS[name]
    return x


def plot_cost_vs_score(scores, questions):
    """Each model's overall % correct against what its answers cost, both axes linear, one colour per company
    (COMPANY_COLORS) and grey for every free model ($0). Only the models that cost at least NAMED_FROM dollars or
    score over NAMED_ABOVE % are named (particle-1 with a line to its dot, LABEL_PLACES): the others
    are too close together to name, and the legend gives their company. scores: model,
    score, answered, cost. A model that has answered fewer than all `questions` is marked with * and its count given
    in a note below the chart, as in plot_overall(). The dots that would overlap are spread sideways
    (spread_sideways()), as the note below the chart says."""
    figure, ax = plt.subplots(figsize=(16.9, 8.5))
    x_of = spread_sideways(scores)
    for row in scores.iter_rows(named=True):
        model = row["model"]
        ax.scatter(x_of[model], row["score"], s=DOT_SIZE, color=company_color(model), edgecolor="white", linewidth=1.5, zorder=3)
        name = SCATTER_NAMES.get(model, short_name(model)) + ("*" if row["answered"] < questions else "")
        if model in LABEL_PLACES:
            ax.annotate(name, (x_of[model], row["score"]), xytext=LABEL_PLACES[model], textcoords="data",
                        va="center", ha="left", fontsize=16.8, color="#0b0b0b",
                        arrowprops=dict(arrowstyle="-", color="#8a8984", linewidth=0.8, shrinkA=3, shrinkB=8))
        elif row["cost"] >= NAMED_FROM or row["score"] > NAMED_ABOVE:
            dx, dy = LABEL_OFFSETS.get(model, (9, 0))
            ax.annotate(name, (x_of[model], row["score"]), xytext=(dx, dy), textcoords="offset points", va="center",
                        ha="left" if dx > 0 else "right" if dx < 0 else "center", fontsize=16.8, color="#0b0b0b")
    ax.set_xlim(-0.8, scores["cost"].max() * 1.08)
    ax.xaxis.set_major_locator(plt.MultipleLocator(2))
    ax.xaxis.set_major_formatter(lambda value, _: f"\\${value:.0f}")
    ax.set_ylim(0, 105)
    ax.set_yticks(range(0, 101, 20))
    ax.tick_params(labelsize=16.8)
    ax.set_xlabel("Average cost to run the benchmark", fontsize=18)
    ax.set_ylabel("% correct", fontsize=18)
    ax.grid(True, color="#e4e3df", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#8a8984")
    for label, color in company_legend(scores["model"]).items():  # one legend entry per company, then the free models
        ax.scatter([], [], s=DOT_SIZE, color=color, label=label)
    ax.scatter([], [], s=DOT_SIZE, color=FREE_COLOR, label="Free models")
    ax.legend(frameon=False, loc="upper right", ncol=2, fontsize=15.6, columnspacing=1.2, handletextpad=0.3)  # two columns
    notes = ["Dots at \\$0, and at 10-15% correct, are spread sideways so each can be seen."]
    unfinished = scores.filter(pl.col("answered") < questions).sort("model")
    if unfinished.height:
        counts = ", ".join(f"{short_name(model)} {answered} of {questions}"
                           for model, answered in unfinished.select("model", "answered").rows())
        notes.append(f"* Unfinished, scored on the questions answered so far: {counts}")
    figure.text(0.01, 0.01, "\n".join(notes), ha="left", va="bottom", fontsize=14.4, color="#5f5e5a")
    figure.tight_layout(rect=(0, 0.035 * len(notes), 1, 1))
    return figure


# Names on the bands chart that go below their dot instead of above it, clear of a neighbour
BAND_LABELS_BELOW = {"google/gemini-4-carbon", "deepseek/deepseek-v4.1-flash-1017"}
BOX_SCALE = 1.2  # how much bigger than its dot and name a box round a model is
BAND_PAD = 2  # how far a band reaches past its group's lowest and highest score, in % points: the gaps show the background


def fit_on_axis(score_of, height_of, fit_models):
    """A least-squares line of height (on the axis as drawn) against score through `fit_models`, and for every other
    model whether it is far from the line: {model: (residual, p-value)}. The p-value is the two-sided t-test of a new
    point, with the residual over its prediction standard error (the line's residual spread, df = fitted models - 2,
    widened for a point far from the middle of the fitted scores)."""
    x = np.array([score_of[m] for m in fit_models])
    y = np.array([height_of[m] for m in fit_models])
    slope, intercept = np.polyfit(x, y, 1)
    n = len(x)
    spread = np.sqrt(((y - intercept - slope * x) ** 2).sum() / (n - 2))
    outside = {}
    for model in score_of:
        if model in fit_models:
            continue
        residual = height_of[model] - intercept - slope * score_of[model]
        se = spread * np.sqrt(1 + 1 / n + (score_of[model] - x.mean()) ** 2 / ((x - x.mean()) ** 2).sum())
        outside[model] = (residual, 2 * t.sf(abs(residual / se), n - 2))
    r_squared = np.corrcoef(x, y)[0, 1] ** 2
    return slope, intercept, r_squared, outside


def plot_bands(scores, groups, fit_models=(), boxes=None, box_notes=None):
    """% correct across, what the answers cost up (scores: model, score in %, cost), a dot for each model in its
    company's colour with its name standing above it, on a band for each of `groups` (lists of models, best first):
    from BAND_PAD below the group's lowest score to BAND_PAD above its highest, with the background between the
    bands. The cost axis is symlog, so the free models (cost 0) and the dearest model can be on one axis. Models
    that cost next to nothing and score within 2 points of each other in a run of three or more are named in a fan,
    each with a thin line to its dot. `fit_models`: a straight line is fitted through these models' dots (least squares,
    straight as it is drawn on the cost axis, so cost goes up by a fixed factor for each point of score), and drawn
    across them, and the others' distance from it tested (fit_on_axis(), printed). `boxes`: {model: colour}, a box
    of that colour round the model's dot and name, and `box_notes`: {model: (text, side)}, a note in the box's colour
    above or below the box ("above" or "below"), centred on it."""
    score_of = dict(scores.select("model", "score").rows())
    cost_of = dict(scores.select("model", "cost").rows())
    figure, ax = plt.subplots(figsize=(16.9, 8.5))
    ramp = plt.get_cmap("viridis")(np.linspace(0.95, 0.05, len(groups)))  # group 1 yellow ... the last group purple
    band_of = {}  # each model's band: its left and right edge, in % correct
    for number, members in enumerate(groups, start=1):
        left = max(min(score_of[m] for m in members) - BAND_PAD, 0)
        right = min(max(score_of[m] for m in members) + BAND_PAD, 100)
        band_of.update({m: (left, right) for m in members})
        ax.axvspan(left, right, color=ramp[number - 1], alpha=0.22, linewidth=0, zorder=0)
        ax.text((left + right) / 2, 1.01, f"Group {number}", transform=ax.get_xaxis_transform(), ha="center",
                va="bottom", fontsize=18, color="#0b0b0b")
    cheap = sorted((m for m in score_of if cost_of[m] < 0.05), key=score_of.get)  # along the bottom, left to right
    runs = []  # runs of cheap models each within 2 points of the one before
    for model in cheap:
        if runs and score_of[model] - score_of[runs[-1][-1]] < 2.2:
            runs[-1].append(model)
        else:
            runs.append([model])
    fan = {m: (i - (len(run) - 1) / 2) * 30 for run in runs if len(run) >= 3 for i, m in enumerate(run)}  # points across
    names = {}  # each model's name on the chart
    for model in score_of:
        ax.scatter(score_of[model], cost_of[model], s=DOT_SIZE, color=company_color(model), edgecolor="white",
                   linewidth=1.5, zorder=3)
        name = SCATTER_NAMES.get(model, short_name(model))
        point = (score_of[model], cost_of[model])
        if model in fan:
            names[model] = ax.annotate(name, point, xytext=(fan[model], 40), textcoords="offset points", rotation=90, ha="center",
                        va="bottom", fontsize=15.6, color="#0b0b0b",
                        arrowprops=dict(arrowstyle="-", color="#8a8984", linewidth=0.8, shrinkA=0, shrinkB=8))
        elif model in BAND_LABELS_BELOW:
            names[model] = ax.annotate(name, point, xytext=(0, -12), textcoords="offset points", rotation=90, ha="center",
                        va="top", fontsize=15.6, color="#0b0b0b")
        else:
            names[model] = ax.annotate(name, point, xytext=(0, 12), textcoords="offset points", rotation=90,
                                       ha="center", va="bottom", fontsize=15.6, color="#0b0b0b")
    ax.set_xlim(0, 100)
    ax.set_xticks(range(0, 101, 10))
    ax.set_yscale("symlog", linthresh=0.1, linscale=0.4)
    ax.set_ylim(-0.02, 70)
    if fit_models:  # fitted on the axis as it is drawn: the symlog of the cost
        axis = ax.yaxis.get_transform()
        height_of = dict(zip(score_of, axis.transform(np.array([cost_of[m] for m in score_of]))))
        slope, intercept, r_squared, outside = fit_on_axis(score_of, height_of, list(fit_models))
        print(f"Line through {len(fit_models)} models: drawn-axis height = {intercept:.3f} + {slope:.4f} x score; "
              f"R-squared {r_squared:.2f}")
        for model, (residual, p) in sorted(outside.items(), key=lambda item: item[1][1]):
            print(f"  {short_name(model)}: {residual:+.2f} from the line on the drawn axis, p = {p:.4f}")
        line = np.array([min(score_of[m] for m in fit_models), max(score_of[m] for m in fit_models)])
        ax.plot(line, axis.inverted().transform(intercept + slope * line), color="#bdbcb7", linewidth=2.5, linestyle="--",
                zorder=1)
    ax.set_yticks([0, 0.1, 0.5, 1, 2, 5, 10, 20])
    ax.set_yticklabels(["\\$0", "\\$0.10", "\\$0.50", "\\$1", "\\$2", "\\$5", "\\$10", "\\$20"])
    ax.yaxis.set_minor_locator(plt.NullLocator())
    ax.tick_params(labelsize=16.8)
    ax.set_xlabel("% correct", fontsize=18)
    ax.set_ylabel("Average cost to run the benchmark", fontsize=18)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#8a8984")
    for label, color in company_legend(scores["model"]).items():
        ax.scatter([], [], s=DOT_SIZE, color=color, label=label)
    ax.scatter([], [], s=DOT_SIZE, color=FREE_COLOR, label="Free models")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=5, fontsize=15.6, columnspacing=1.6,
              handletextpad=0.3)  # under the x-axis label, in two rows
    ax.grid(True, color="#e4e3df", linewidth=0.8)
    ax.set_axisbelow(True)
    figure.tight_layout()
    figure.canvas.draw()  # the names' places are known only once drawn
    for model, color in (boxes or {}).items():
        name_box = names[model].get_window_extent()
        x, y = ax.transData.transform((score_of[model], cost_of[model]))
        radius = DOT_SIZE ** 0.5 / 2 * figure.dpi / 72  # the dot's radius in pixels
        dot_box = Bbox.from_extents(x - radius, y - radius, x + radius, y + radius)
        box = Bbox.union([name_box, dot_box]).expanded(BOX_SCALE, BOX_SCALE)
        pad = 6
        inches = Bbox.from_extents(box.x0 - pad, box.y0 - pad, box.x1 + pad, box.y1 + pad).transformed(
            figure.dpi_scale_trans.inverted())
        figure.add_artist(Rectangle((inches.x0, inches.y0), inches.width, inches.height, transform=figure.dpi_scale_trans,
                                    fill=False, edgecolor=color, linewidth=2.5, zorder=5))
        if model in (box_notes or {}):
            text, side = box_notes[model]
            above = side == "above"
            note = figure.text(inches.x0 + inches.width / 2, inches.y1 + 0.08 if above else inches.y0 - 0.08, text,
                               transform=figure.dpi_scale_trans, ha="center", va="bottom" if above else "top",
                               fontsize=16.8, fontweight="bold", color=color, zorder=5)
            edges = [ax.transData.transform((edge, 0))[0] for edge in band_of[model]]  # the band's edges, in pixels
            extent = note.get_window_extent()
            shift = max(edges[0] + 4 - extent.x0, 0) - max(extent.x1 - (edges[1] - 4), 0)  # keep the note in its band
            note.set_x(note.get_position()[0] + shift / figure.dpi)
    return figure
