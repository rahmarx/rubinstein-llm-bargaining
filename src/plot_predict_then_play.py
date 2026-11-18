"""Phase-2a figure: predicted vs. played distance-to-SPE, and a before/after
breakdown-rate comparison against Phase 1's unexplained condition (collected
under the old, too-tight token caps) -- the cleanest available evidence for
how much of Phase 1's breakdown rate was a token-truncation artifact.

Usage:
    python plot_predict_then_play.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analysis import bootstrap_ci, exclude_infra_errors, load_dialogues, wilson_ci
from bargaining_env import phase1_variants

MODEL_COLORS = {"claude-sonnet-4.5": "tab:blue", "gpt-5-mini": "tab:orange", "gemini-2.5-flash": "tab:green"}
MODEL_LABELS = {"claude-sonnet-4.5": "Claude Sonnet 4.5", "gpt-5-mini": "GPT-5-mini", "gemini-2.5-flash": "Gemini 2.5 Flash"}
MODEL_ORDER = list(MODEL_LABELS)

# Phase 1 unexplained-condition breakdown rates (old, too-tight token caps),
# hardcoded from the confirmed-clean N=30/cell Phase-1 dataset for the
# before/after comparison (see analysis.py / README for derivation).
PHASE1_BREAKDOWN_UNEXPLAINED = {"claude-sonnet-4.5": 0.071, "gpt-5-mini": 0.129, "gemini-2.5-flash": 0.519}


def main() -> None:
    data_dir = Path(__file__).resolve().parent.parent / "data"
    fig_dir = Path(__file__).resolve().parent.parent / "report" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    fixed_spe = {cfg.label(): cfg.spe_first_mover_share() * cfg.pie for cfg in phase1_variants()}

    all_dialogues = load_dialogues(data_dir / "phase2_predict_then_play.json")
    clean, excluded = exclude_infra_errors(all_dialogues)
    if excluded:
        print(f"Excluded {len(excluded)} infra-error dialogues")

    by_model: dict[str, list[dict]] = {m: [] for m in MODEL_ORDER}
    for d in clean:
        by_model[d["model_key"]].append(d)

    fig, (err_ax, brk_ax) = plt.subplots(1, 2, figsize=(12, 5))

    x = range(len(MODEL_ORDER))
    width = 0.35

    pred_means, pred_lo, pred_hi = [], [], []
    play_means, play_lo, play_hi = [], [], []
    for m in MODEL_ORDER:
        ds = by_model[m]
        pred_errs = [abs(d["prediction"] - fixed_spe[d["config_label"]]) for d in ds if d["prediction"] is not None]
        play_errs = [abs(d["first_mover_share"] - fixed_spe[d["config_label"]]) for d in ds if d["agreed"]]
        pm, plo, phi = bootstrap_ci(pred_errs)
        ym, ylo, yhi = bootstrap_ci(play_errs)
        pred_means.append(pm); pred_lo.append(pm - plo); pred_hi.append(phi - pm)
        play_means.append(ym); play_lo.append(ym - ylo); play_hi.append(yhi - ym)

    xs_pred = [xi - width / 2 for xi in x]
    xs_play = [xi + width / 2 for xi in x]
    err_ax.bar(xs_pred, pred_means, width=width, color="tab:gray", label="Stated prediction\n(before playing)",
               yerr=[pred_lo, pred_hi], capsize=3)
    err_ax.bar(xs_play, play_means, width=width, color="tab:purple", label="Realized play\n(agreed dialogues)",
               yerr=[play_lo, play_hi], capsize=3)
    err_ax.set_xticks(list(x))
    err_ax.set_xticklabels([MODEL_LABELS[m] for m in MODEL_ORDER], fontsize=9)
    err_ax.set_ylabel("distance to true SPE (pie points)")
    err_ax.set_title("Predicted vs. played distance to SPE\n(mean ± 95% bootstrap CI)")
    err_ax.legend(fontsize=8)
    err_ax.grid(axis="y", alpha=0.3)

    old_rates, new_rates, new_lo, new_hi = [], [], [], []
    for m in MODEL_ORDER:
        ds = by_model[m]
        n = len(ds)
        bk = sum(1 for d in ds if not d["agreed"])
        p, lo, hi = wilson_ci(bk, n)
        old_rates.append(PHASE1_BREAKDOWN_UNEXPLAINED[m] * 100)
        new_rates.append(p * 100)
        new_lo.append((p - lo) * 100)
        new_hi.append((hi - p) * 100)

    xs_old = [xi - width / 2 for xi in x]
    xs_new = [xi + width / 2 for xi in x]
    brk_ax.bar(xs_old, old_rates, width=width, color="firebrick",
               label="Phase 1 (old token caps:\nGemini 300, GPT-5-mini 2000)")
    brk_ax.bar(xs_new, new_rates, width=width, color="seagreen",
               label="Phase 2a (fixed caps:\nGemini 16384, GPT-5-mini 4000)",
               yerr=[new_lo, new_hi], capsize=3)
    brk_ax.set_xticks(list(x))
    brk_ax.set_xticklabels([MODEL_LABELS[m] for m in MODEL_ORDER], fontsize=9)
    brk_ax.set_ylabel("breakdown rate (%)")
    brk_ax.set_ylim(0, 100)
    brk_ax.set_title("Breakdown rate before/after the token-cap fix\n(unexplained condition)")
    brk_ax.legend(fontsize=7.5, loc="upper right")
    brk_ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Phase 2a: Predict-then-play (N=10/cell, unexplained condition)")
    fig.tight_layout()
    out_path = fig_dir / "phase2a_predict_then_play.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
