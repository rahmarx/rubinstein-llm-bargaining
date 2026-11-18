"""Preliminary Phase-1 plots: distance-to-SPE and breakdown rate per game
variant, faceted by model and prompt condition (SPE explained vs. not).
This is the "preliminary distance-to-SPE plots" deliverable committed to
for the Week 11 progress report.

Usage:
    python plot_results.py                               # data/phase1_results.json -> report/figures/
    python plot_results.py ../data/phase1_results.json --fig-dir ../report/figures
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analysis import exclude_infra_errors, load_dialogues, summarize

MODEL_COLORS = {
    "claude-sonnet-4.5": "tab:blue",
    "gpt-5-mini": "tab:orange",
    "gemini-2.5-flash": "tab:green",
}
MODEL_LABELS = {
    "claude-sonnet-4.5": "Claude Sonnet 4.5",
    "gpt-5-mini": "GPT-5-mini",
    "gemini-2.5-flash": "Gemini 2.5 Flash",
}
MODEL_ORDER = list(MODEL_LABELS)

VARIANT_ORDER = [
    "infinite_d0.5", "infinite_d0.8", "infinite_d0.95",
    "finite_T2_d0.8", "finite_T3_d0.8", "finite_T5_d0.8",
    "asymmetric_d1-0.95_d2-0.5",
]
VARIANT_LABELS = {
    "infinite_d0.5": "infinite\nδ=0.5", "infinite_d0.8": "infinite\nδ=0.8",
    "infinite_d0.95": "infinite\nδ=0.95",
    "finite_T2_d0.8": "finite\nT=2", "finite_T3_d0.8": "finite\nT=3", "finite_T5_d0.8": "finite\nT=5",
    "asymmetric_d1-0.95_d2-0.5": "asymmetric\nδ1=.95,δ2=.5",
}


def _variant_sort_key(label: str) -> int:
    return VARIANT_ORDER.index(label) if label in VARIANT_ORDER else len(VARIANT_ORDER)


def _model_sort_key(model: str) -> int:
    return MODEL_ORDER.index(model) if model in MODEL_ORDER else len(MODEL_ORDER)


def plot_distance_and_breakdown(rows: list[dict], fig_dir: Path) -> list[Path]:
    written = []
    for explain_spe in (False, True):
        subset = [r for r in rows if r["explain_spe"] == explain_spe]
        if not subset:
            continue
        variants = sorted({r["config_label"] for r in subset}, key=_variant_sort_key)
        models = sorted({r["model_key"] for r in subset}, key=_model_sort_key)

        fig, (dist_ax, brk_ax) = plt.subplots(1, 2, figsize=(14, 5))
        x = list(range(len(variants)))
        width = 0.8 / max(len(models), 1)

        for i, model in enumerate(models):
            dists, dist_err_lo, dist_err_hi = [], [], []
            brks, brk_err_lo, brk_err_hi = [], [], []
            for v in variants:
                match = next((r for r in subset if r["config_label"] == v and r["model_key"] == model), None)
                if match is None or match["n_agreed"] == 0:
                    dists.append(0.0); dist_err_lo.append(0.0); dist_err_hi.append(0.0)
                else:
                    dists.append(match["distance_to_spe_mean"])
                    dist_err_lo.append(match["distance_to_spe_mean"] - match["distance_to_spe_ci"][0])
                    dist_err_hi.append(match["distance_to_spe_ci"][1] - match["distance_to_spe_mean"])
                if match is None:
                    brks.append(0.0); brk_err_lo.append(0.0); brk_err_hi.append(0.0)
                else:
                    brks.append(match["breakdown_rate"] * 100)
                    brk_err_lo.append((match["breakdown_rate"] - match["breakdown_rate_ci"][0]) * 100)
                    brk_err_hi.append((match["breakdown_rate_ci"][1] - match["breakdown_rate"]) * 100)

            offset = (i - (len(models) - 1) / 2) * width
            xs = [xi + offset for xi in x]
            color = MODEL_COLORS.get(model, f"C{i}")
            label = MODEL_LABELS.get(model, model)
            dist_ax.bar(xs, dists, width=width, color=color, label=label,
                        yerr=[dist_err_lo, dist_err_hi], capsize=3)
            brk_ax.bar(xs, brks, width=width, color=color, label=label,
                       yerr=[brk_err_lo, brk_err_hi], capsize=3)

        dist_ax.set_xticks(x)
        dist_ax.set_xticklabels([VARIANT_LABELS.get(v, v) for v in variants], fontsize=8)
        dist_ax.set_ylabel("distance to SPE (pie points)")
        dist_ax.set_title("Distance to SPE\n(mean ± 95% bootstrap CI, agreed dialogues only)")
        dist_ax.axhline(0, color="black", linewidth=0.8)
        dist_ax.grid(axis="y", alpha=0.3)
        dist_ax.legend(fontsize=8)

        brk_ax.set_xticks(x)
        brk_ax.set_xticklabels([VARIANT_LABELS.get(v, v) for v in variants], fontsize=8)
        brk_ax.set_ylabel("breakdown rate (%)")
        brk_ax.set_title("Breakdown rate\n(mean ± 95% Wilson CI)")
        brk_ax.set_ylim(0, 100)
        brk_ax.grid(axis="y", alpha=0.3)
        brk_ax.legend(fontsize=8)

        cond_label = "SPE explained in prompt" if explain_spe else "SPE not explained"
        fig.suptitle(f"Phase 1: LLM vs. scripted SPE opponent — {cond_label}")
        fig.tight_layout()
        suffix = "explained" if explain_spe else "unexplained"
        out_path = fig_dir / f"phase1_distance_breakdown_{suffix}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        written.append(out_path)
    return written


def main() -> None:
    default_data = Path(__file__).resolve().parent.parent / "data" / "phase1_results.json"
    default_figs = Path(__file__).resolve().parent.parent / "report" / "figures"

    parser = argparse.ArgumentParser(description="Plot Phase-1 distance-to-SPE and breakdown-rate figures.")
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
              f"infrastructure error (e.g. the daily budget cap) -- not real negotiation outcomes.")
    rows = summarize(clean)
    if not rows:
        raise SystemExit("No dialogues found in the results file.")

    written = plot_distance_and_breakdown(rows, fig_dir)
    for path in written:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
