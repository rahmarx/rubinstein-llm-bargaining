"""Phase-2b figure: distance to SPE and breakdown rate by pairing type
(same-family vs. cross-family) and prompt condition, LLM-vs-LLM extension.

Usage:
    python plot_llm_vs_llm.py
"""
from __future__ import annotations

from pathlib import Path
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analysis import bootstrap_ci, exclude_infra_errors, load_dialogues, wilson_ci
from bargaining_env import phase1_variants


def main() -> None:
    data_dir = Path(__file__).resolve().parent.parent / "data"
    fig_dir = Path(__file__).resolve().parent.parent / "report" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    fixed_spe = {cfg.label(): cfg.spe_first_mover_share() * cfg.pie for cfg in phase1_variants()}

    all_dialogues = load_dialogues(data_dir / "phase2_llm_vs_llm.json")
    clean, excluded = exclude_infra_errors(all_dialogues)
    if excluded:
        print(f"Excluded {len(excluded)} infra-error dialogues")

    by_type_cond = defaultdict(list)
    for d in clean:
        by_type_cond[(d["pairing_type"], d["explain_spe"])].append(d)

    pairing_types = ["same-family", "cross-family"]
    conditions = [False, True]
    colors = {"same-family": "tab:cyan", "cross-family": "tab:pink"}

    fig, (dist_ax, brk_ax) = plt.subplots(1, 2, figsize=(12, 5))
    x = range(len(conditions))
    width = 0.35

    for i, ptype in enumerate(pairing_types):
        means, lo_err, hi_err = [], [], []
        for cond in conditions:
            ds = by_type_cond[(ptype, cond)]
            dists = [abs(d["first_mover_share"] - fixed_spe[d["config_label"]]) for d in ds if d["agreed"]]
            m, lo, hi = bootstrap_ci(dists) if dists else (0, 0, 0)
            means.append(m); lo_err.append(m - lo); hi_err.append(hi - m)
        offset = (i - 0.5) * width
        xs = [xi + offset for xi in x]
        dist_ax.bar(xs, means, width=width, color=colors[ptype], label=ptype,
                    yerr=[lo_err, hi_err], capsize=3)

    dist_ax.set_xticks(list(x))
    dist_ax.set_xticklabels(["unexplained", "explained"])
    dist_ax.set_ylabel("distance to SPE (pie points)")
    dist_ax.set_title("Distance to SPE, LLM-vs-LLM\n(mean ± 95% bootstrap CI, agreed only)")
    dist_ax.legend(fontsize=9)
    dist_ax.grid(axis="y", alpha=0.3)

    for i, ptype in enumerate(pairing_types):
        means, lo_err, hi_err = [], [], []
        for cond in conditions:
            ds = by_type_cond[(ptype, cond)]
            n = len(ds)
            bk = sum(1 for d in ds if not d["agreed"])
            p, lo, hi = wilson_ci(bk, n)
            means.append(p * 100); lo_err.append((p - lo) * 100); hi_err.append((hi - p) * 100)
        offset = (i - 0.5) * width
        xs = [xi + offset for xi in x]
        brk_ax.bar(xs, means, width=width, color=colors[ptype], label=ptype,
                   yerr=[lo_err, hi_err], capsize=3)

    brk_ax.set_xticks(list(x))
    brk_ax.set_xticklabels(["unexplained", "explained"])
    brk_ax.set_ylabel("breakdown rate (%)")
    brk_ax.set_ylim(0, 100)
    brk_ax.set_title("Breakdown rate, LLM-vs-LLM\n(mean ± 95% Wilson CI)")
    brk_ax.legend(fontsize=9)
    brk_ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Phase 2b: LLM-vs-LLM (N=10/cell, 2 variants x 6 pairings x 2 conditions)")
    fig.tight_layout()
    out_path = fig_dir / "phase2b_llm_vs_llm.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
