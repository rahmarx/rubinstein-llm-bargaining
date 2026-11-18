"""Deviation taxonomy write-up: aggregates `analysis.classify_deviation`
across all 7 game variants into one table and stacked-bar figure per
(model, prompt condition), answering "when models deviate from the SPE,
what does the deviation look like" -- the taxonomy write-up promised as
part of Phase 2 in the progress report.

Usage:
    python deviation_taxonomy.py                      # data/phase1_results.json -> report/figures/
    python deviation_taxonomy.py ../data/phase1_results.json --fig-dir ../report/figures
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analysis import classify_deviation, exclude_infra_errors, load_dialogues

CATEGORY_ORDER = [
    "SPE-consistent", "fairness-pull", "first-mover-exploitation",
    "second-mover-exploitation", "backward-induction-failure", "other", "breakdown",
]
CATEGORY_COLORS = {
    "SPE-consistent": "tab:blue",
    "fairness-pull": "tab:green",
    "first-mover-exploitation": "tab:orange",
    "second-mover-exploitation": "tab:orange",
    "backward-induction-failure": "tab:red",
    "other": "tab:gray",
    "breakdown": "black",
}
MODEL_LABELS = {
    "claude-sonnet-4.5": "Claude Sonnet 4.5",
    "gpt-5-mini": "GPT-5-mini",
    "gemini-2.5-flash": "Gemini 2.5 Flash",
}
MODEL_ORDER = list(MODEL_LABELS)


def aggregate(dialogues: list[dict]) -> dict:
    """Collapses all 7 game variants into one count per (model, explain_spe,
    category) -- the taxonomy is about the *shape* of deviation, pooled
    across variants, not a per-variant breakdown."""
    counts: dict[tuple[str, bool], dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for d in dialogues:
        key = (d["model_key"], d["explain_spe"])
        if d["agreed"]:
            cat = classify_deviation(d["first_mover_share"], d["spe_benchmark_share"])
        else:
            cat = "breakdown"
        counts[key][cat] += 1
    return counts


def print_table(counts: dict) -> None:
    categories = [c for c in CATEGORY_ORDER if any(counts[k].get(c, 0) for k in counts)]
    header = f"{'model':20s} {'cond':10s} {'n':>5s} " + " ".join(f"{c:>26s}" for c in categories)
    print(header)
    print("-" * len(header))
    for model in MODEL_ORDER:
        for explain_spe in (False, True):
            key = (model, explain_spe)
            if key not in counts:
                continue
            row = counts[key]
            n = sum(row.values())
            cells = []
            for c in categories:
                k = row.get(c, 0)
                pct = 100 * k / n if n else 0.0
                cells.append(f"{k:>3d} ({pct:4.1f}%){'':>13s}")
            cond = "explained" if explain_spe else "unexplained"
            print(f"{MODEL_LABELS.get(model, model):20s} {cond:10s} {n:5d} " + " ".join(cells))


def plot_taxonomy(counts: dict, fig_dir: Path) -> Path:
    categories = [c for c in CATEGORY_ORDER if any(counts[k].get(c, 0) for k in counts)]
    conditions = [(model, explain_spe) for model in MODEL_ORDER for explain_spe in (False, True)
                  if (model, explain_spe) in counts]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = list(range(len(conditions)))
    bottoms = [0.0] * len(conditions)

    for cat in categories:
        heights = []
        for (model, explain_spe) in conditions:
            row = counts[(model, explain_spe)]
            n = sum(row.values())
            heights.append(100 * row.get(cat, 0) / n if n else 0.0)
        ax.bar(x, heights, bottom=bottoms, color=CATEGORY_COLORS.get(cat, "tab:gray"), label=cat, width=0.65)
        bottoms = [b + h for b, h in zip(bottoms, heights)]

    labels = [f"{MODEL_LABELS.get(m, m)}\n{'explained' if e else 'unexplained'}" for m, e in conditions]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("share of all dialogues (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Deviation taxonomy, pooled across all 7 game variants\n(Phase 1, N=30/cell)")
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    out_path = fig_dir / "phase1_deviation_taxonomy.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    default_data = Path(__file__).resolve().parent.parent / "data" / "phase1_results.json"
    default_figs = Path(__file__).resolve().parent.parent / "report" / "figures"

    parser = argparse.ArgumentParser(description="Write up the deviation taxonomy, pooled across game variants.")
    parser.add_argument("data_path", nargs="?", default=str(default_data))
    parser.add_argument("--fig-dir", default=str(default_figs))
    args = parser.parse_args()

    data_path = Path(args.data_path)
    fig_dir = Path(args.fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    if not data_path.exists():
        raise SystemExit(f"No results file at {data_path} -- run run_experiments.py first.")

    all_dialogues = load_dialogues(data_path)
    clean, excluded = exclude_infra_errors(all_dialogues)
    if excluded:
        print(f"Note: excluded {len(excluded)}/{len(all_dialogues)} dialogues that hit a gateway "
              f"infrastructure error -- not real negotiation outcomes.")

    counts = aggregate(clean)
    print_table(counts)
    print()
    out_path = plot_taxonomy(counts, fig_dir)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
