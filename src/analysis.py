"""Analysis utilities: distance-to-SPE statistics with confidence intervals,
breakdown rates, and the deviation taxonomy, applied per (game variant,
model, prompt condition) cell -- implementing the pre-specified protocol
from the progress-report feedback response.

CI method: a percentile bootstrap for the continuous distance-to-SPE mean
(robust to skew at N~30), and a Wilson score interval for the breakdown
rate (robust for rare events, unlike a normal approximation).
"""
from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from pathlib import Path

N_BOOTSTRAP = 2000
TOLERANCE = 2.0  # points on the 0-100 pie scale for "SPE-consistent"
FAIR_SHARE = 50.0


def bootstrap_ci(values: list[float], n_boot: int = N_BOOTSTRAP, seed: int = 0) -> tuple[float, float, float]:
    """Percentile bootstrap 95% CI for the mean of `values`. Returns (mean, lo, hi)."""
    if not values:
        return float("nan"), float("nan"), float("nan")
    n = len(values)
    mean = sum(values) / n
    if n == 1:
        return mean, mean, mean
    rng = random.Random(seed)
    boot_means = []
    for _ in range(n_boot):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        boot_means.append(sum(sample) / n)
    boot_means.sort()
    lo = boot_means[int(0.025 * n_boot)]
    hi = boot_means[min(n_boot - 1, int(0.975 * n_boot))]
    return mean, lo, hi


def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """Wilson score 95% CI for a proportion. Returns (p, lo, hi)."""
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = successes / n
    denom = 1 + z ** 2 / n
    center = (p + z ** 2 / (2 * n)) / denom
    half_width = (z * math.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2))) / denom
    return p, max(0.0, center - half_width), min(1.0, center + half_width)


def classify_deviation(first_mover_share: float, spe_share: float,
                        tolerance: float = TOLERANCE, fair: float = FAIR_SHARE) -> str:
    """Deviation taxonomy for an agreed dialogue, matching the categories
    named in the proposal: SPE-consistent, fairness-pull,
    first-mover-exploitation, backward-induction-failure.

    The T=2 finite variant is the one case among the 7 where the SPE itself
    favors the *second* mover (v_2 = 1 - delta < 50 for delta > 0.5); since
    `first_mover_share` is always what's reported, the "exploitation"
    category is labeled second-mover-exploitation there for clarity (the
    underlying logic -- deviating further in the direction the equilibrium
    already favors -- is identical).
    """
    diff = first_mover_share - spe_share
    if abs(diff) <= tolerance:
        return "SPE-consistent"
    if spe_share == fair:
        return "other"  # no fairness/advantage axis to classify against
    lo, hi = sorted((fair, spe_share))
    if lo <= first_mover_share <= hi:
        return "fairness-pull"
    same_side_as_spe = (first_mover_share - fair) * (spe_share - fair) > 0
    if same_side_as_spe:
        return "first-mover-exploitation" if spe_share > fair else "second-mover-exploitation"
    return "backward-induction-failure"


# Substrings of gateway-side infrastructure errors that must NOT be counted as
# genuine negotiation breakdowns -- these are API/account failures, not model
# behavior. Found empirically: the TAMU gateway enforces a shared daily USD
# budget across all models/students on the account, and returns this message
# (via an empty, unparseable completion) once it's exhausted. A dialogue that
# hit this is missing data, not a disagreement, and must be excluded from
# every statistic -- including the breakdown rate -- or it silently inflates
# breakdown rates for whichever cells happened to run late in the queue.
INFRA_ERROR_MARKERS = ("Budget Exceeded",)


def load_dialogues(path: Path) -> list[dict]:
    return json.loads(path.read_text())["dialogues"]


def is_infra_error(dialogue: dict) -> bool:
    for round_rec in dialogue.get("history", []):
        for raw_key in ("proposer_raw", "responder_raw"):
            raw = round_rec.get(raw_key)
            if not raw:
                continue
            error_text = str(raw.get("error")) if raw.get("error") else ""
            if any(marker in error_text for marker in INFRA_ERROR_MARKERS):
                return True
            # A second, silent failure signature observed empirically: the
            # gateway sometimes returns a completely empty SSE stream (zero
            # chunks, no exception) once the account-wide budget cap is hit,
            # rather than the explicit "Budget Exceeded" error body. A call
            # that actually reached the model always has nonzero
            # prompt_tokens (the input is billed even if output is cut off),
            # so zero prompt_tokens + zero completion_tokens + no explicit
            # error is this silent variant. (Not checking `text` here: with
            # --save-transcripts off, the text field is stripped from every
            # record regardless of outcome, so its absence alone doesn't
            # mean the call failed.)
            # Guard: only applies to LLM call results (which always carry a
            # "prompt_tokens" key, even if 0) -- the scripted agent's raw
            # dict has no such key and must not be caught by this heuristic.
            if "prompt_tokens" in raw and not raw.get("error") \
                    and raw.get("prompt_tokens", 0) == 0 and raw.get("completion_tokens", 0) == 0:
                return True
    return False


def exclude_infra_errors(dialogues: list[dict]) -> tuple[list[dict], list[dict]]:
    """Returns (clean, excluded). `excluded` dialogues are infrastructure
    failures (e.g. the gateway's daily budget cap) and should be re-run once
    the underlying condition clears (the budget error's own message says it
    "resets daily"), not treated as data."""
    clean, excluded = [], []
    for d in dialogues:
        (excluded if is_infra_error(d) else clean).append(d)
    return clean, excluded


def summarize(dialogues: list[dict]) -> list[dict]:
    cells = defaultdict(list)
    for d in dialogues:
        key = (d["config_label"], d["model_key"], d["explain_spe"])
        cells[key].append(d)

    rows = []
    for (config_label, model_key, explain_spe), records in sorted(cells.items()):
        n = len(records)
        agreed = [r for r in records if r["agreed"]]
        breakdowns = [r for r in records if not r["agreed"]]

        distances = [abs(r["first_mover_share"] - r["spe_benchmark_share"]) for r in agreed]
        dist_mean, dist_lo, dist_hi = bootstrap_ci(distances)

        breakdown_p, breakdown_lo, breakdown_hi = wilson_ci(len(breakdowns), n)

        taxonomy: dict[str, int] = defaultdict(int)
        for r in agreed:
            taxonomy[classify_deviation(r["first_mover_share"], r["spe_benchmark_share"])] += 1
        taxonomy["breakdown"] = len(breakdowns)

        rows.append({
            "config_label": config_label, "model_key": model_key, "explain_spe": explain_spe,
            "n": n, "n_agreed": len(agreed),
            "distance_to_spe_mean": dist_mean, "distance_to_spe_ci": [dist_lo, dist_hi],
            "breakdown_rate": breakdown_p, "breakdown_rate_ci": [breakdown_lo, breakdown_hi],
            "deviation_taxonomy": dict(taxonomy),
        })
    return rows


def print_summary(rows: list[dict]) -> None:
    header = f"{'cell':46s} {'n':>4s} {'dist ± 95% CI':>20s} {'breakdown %':>14s}"
    print(header)
    print("-" * len(header))
    for r in rows:
        cell = f"{r['config_label']}|{r['model_key']}|{'exp' if r['explain_spe'] else 'noexp'}"
        if math.isnan(r["distance_to_spe_mean"]):
            dist = "n/a (0 agreed)"
        else:
            dist = (f"{r['distance_to_spe_mean']:.1f} "
                    f"[{r['distance_to_spe_ci'][0]:.1f},{r['distance_to_spe_ci'][1]:.1f}]")
        brk = f"{r['breakdown_rate']*100:.1f}% [{r['breakdown_rate_ci'][0]*100:.0f},{r['breakdown_rate_ci'][1]*100:.0f}]"
        print(f"{cell:46s} {r['n']:4d} {dist:>20s} {brk:>14s}")


if __name__ == "__main__":
    import sys
    from collections import Counter
    default_path = Path(__file__).resolve().parent.parent / "data" / "phase1_results.json"
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else default_path
    if not path.exists():
        print(f"No results file at {path} yet -- run run_experiments.py first.")
    else:
        all_dialogues = load_dialogues(path)
        clean, excluded = exclude_infra_errors(all_dialogues)
        if excluded:
            by_cell = Counter((d["config_label"], d["model_key"], d["explain_spe"]) for d in excluded)
            print(f"Excluded {len(excluded)}/{len(all_dialogues)} dialogues: hit a gateway infrastructure "
                  f"error (e.g. the shared daily budget cap), not a real negotiation outcome. Re-run these "
                  f"cells once the underlying condition clears:")
            for (cfg, model, exp), count in sorted(by_cell.items()):
                print(f"  {cfg}|{model}|{'exp' if exp else 'noexp'}: {count} missing")
            print()
        print_summary(summarize(clean))
