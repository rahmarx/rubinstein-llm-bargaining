"""Phase-2 LLM-vs-LLM extension: replaces the scripted SPE opponent with a
second LLM, both same-family (a model against itself) and cross-family (two
different models), per the progress report's stated Phase-2 plan.

Scope, deliberately reduced from the full Phase-1 grid to keep this a
bounded "bonus" extension rather than a second full-size study: 2
representative game variants (one infinite-horizon, one finite-horizon, both
at the central delta=0.8 already used throughout Phase 1) x 2 prompt
conditions (both players see the same condition -- symmetric information,
to avoid an extra asymmetric-information confound) x 6 model pairings (3
same-family, 3 cross-family) x N dialogues/cell.

Player 0 is always whichever model is listed FIRST in the pairing, and
always proposes first (round 1) -- there is no first-mover counterbalancing
across the pairing order; this is a documented limitation, not an oversight
(see README). `first_mover_share` in the output is therefore always
player 0 / the first-listed model's share.

Usage:
    python llm_vs_llm.py                       # N=10/cell, all 6 pairings, 2 variants, 2 conditions
    python llm_vs_llm.py --n 10 --budget 5.0
"""
from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from itertools import combinations
from pathlib import Path

from bargaining_env import phase1_variants, run_dialogue
from llm_agent import MODEL_CONFIGS, build_client, make_llm_agent
from run_experiments import MODEL_COST_PER_1K, Budget

MODELS = list(MODEL_CONFIGS.keys())
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

VARIANT_LABELS = ["infinite_d0.8", "finite_T3_d0.8"]
PROMPT_CONDITIONS = [False, True]  # explain_spe, applied symmetrically to both players


def pairings() -> list[tuple[str, str]]:
    same_family = [(m, m) for m in MODELS]
    cross_family = list(combinations(MODELS, 2))
    return same_family + cross_family


def pairing_label(model_a: str, model_b: str) -> str:
    return f"{model_a}_vs_{model_b}"


def pairing_type(model_a: str, model_b: str) -> str:
    return "same-family" if model_a == model_b else "cross-family"


def cell_key(config_label: str, model_a: str, model_b: str, explain_spe: bool) -> str:
    return f"{config_label}|{pairing_label(model_a, model_b)}|{'explained' if explain_spe else 'unexplained'}"


def _two_llm_fns(propose_a, respond_a, propose_b, respond_b):
    """Routes player 0's turns to model A, player 1's turns to model B --
    mirrors run_experiments._mix_with_scripted but with a second LLM instead
    of the scripted agent on the player-1 side."""

    def propose_fn(config, proposer_idx, round_num, history):
        if proposer_idx == 0:
            return propose_a(config, proposer_idx, round_num, history)
        return propose_b(config, proposer_idx, round_num, history)

    def respond_fn(config, responder_idx, round_num, history, offer):
        if responder_idx == 0:
            return respond_a(config, responder_idx, round_num, history, offer)
        return respond_b(config, responder_idx, round_num, history, offer)

    return propose_fn, respond_fn


def run_cell(client, config, model_a: str, model_b: str, explain_spe: bool, n: int, budget: Budget,
             retries: int, sleep_s: float, rng: random.Random, save_transcripts: bool) -> list[dict]:
    propose_a, respond_a = make_llm_agent(client, model_a, explain_spe, retries, sleep_s)
    propose_b, respond_b = make_llm_agent(client, model_b, explain_spe, retries, sleep_s)
    propose_fn, respond_fn = _two_llm_fns(propose_a, respond_a, propose_b, respond_b)

    cost_in_a, cost_out_a = MODEL_COST_PER_1K[model_a]
    cost_in_b, cost_out_b = MODEL_COST_PER_1K[model_b]

    records = []
    for trial_index in range(n):
        if budget.exceeded():
            break
        result = run_dialogue(config, propose_fn, respond_fn, rng)

        for r in result.history:
            is_a_turn = r.proposer_idx == 0
            cost_in, cost_out = (cost_in_a, cost_out_a) if is_a_turn else (cost_in_b, cost_out_b)
            if r.proposer_raw and "prompt_tokens" in r.proposer_raw:
                budget.add_dollars(r.proposer_raw.get("prompt_tokens", 0) / 1000 * cost_in
                                    + r.proposer_raw.get("completion_tokens", 0) / 1000 * cost_out)
            # the responder in round N is the other player from the proposer
            resp_cost_in, resp_cost_out = (cost_in_b, cost_out_b) if is_a_turn else (cost_in_a, cost_out_a)
            if r.responder_raw and "prompt_tokens" in r.responder_raw:
                budget.add_dollars(r.responder_raw.get("prompt_tokens", 0) / 1000 * resp_cost_in
                                    + r.responder_raw.get("completion_tokens", 0) / 1000 * resp_cost_out)

        if not save_transcripts:
            for r in result.history:
                if r.proposer_raw:
                    r.proposer_raw = {k: v for k, v in r.proposer_raw.items() if k != "text"}
                if r.responder_raw:
                    r.responder_raw = {k: v for k, v in r.responder_raw.items() if k != "text"}

        d = asdict(result)
        d["model_a"] = model_a
        d["model_b"] = model_b
        d["pairing_type"] = pairing_type(model_a, model_b)
        d["explain_spe"] = explain_spe
        d["trial_index"] = trial_index
        records.append(d)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Phase-2 LLM-vs-LLM grid (same- and cross-family).")
    parser.add_argument("--n", type=int, default=10, help="Dialogues per cell (default: 10).")
    parser.add_argument("--budget", type=float, default=5.0, help="Max USD spend this run (default: 5.0).")
    parser.add_argument("--out", type=str, default=str(DATA_DIR / "phase2_llm_vs_llm.json"))
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--sleep", type=float, default=1.0)
    parser.add_argument("--save-transcripts", action="store_true")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    existing: list[dict] = []
    if out_path.exists():
        existing = json.loads(out_path.read_text())["dialogues"]

    client = build_client()
    budget = Budget(args.budget)
    rng = random.Random(args.seed)

    variants = [v for v in phase1_variants() if v.label() in VARIANT_LABELS]
    cells = [(v, a, b, e) for v in variants for (a, b) in pairings() for e in PROMPT_CONDITIONS]
    rng.shuffle(cells)

    all_records = list(existing)
    done_counts: dict[str, int] = {}
    for d in existing:
        k = cell_key(d["config_label"], d["model_a"], d["model_b"], d["explain_spe"])
        done_counts[k] = done_counts.get(k, 0) + 1

    for cell_idx, (config, model_a, model_b, explain_spe) in enumerate(cells):
        if budget.exceeded():
            print(f"Budget cap (${budget.max_cost:.2f}) reached; stopping after "
                  f"{cell_idx}/{len(cells)} cells. Re-run later to resume.")
            break
        key = cell_key(config.label(), model_a, model_b, explain_spe)
        already = done_counts.get(key, 0)
        remaining = max(0, args.n - already)
        if remaining == 0:
            continue
        print(f"[{cell_idx + 1}/{len(cells)}] {key}: running {remaining} more dialogue(s) "
              f"(spend so far ${budget.spent:.3f})")
        new_records = run_cell(client, config, model_a, model_b, explain_spe, remaining, budget,
                                args.retries, args.sleep, rng, args.save_transcripts)
        all_records.extend(new_records)
        out_path.write_text(json.dumps({"dialogues": all_records}, indent=2))

    out_path.write_text(json.dumps({"dialogues": all_records}, indent=2))
    print(f"Done. {len(all_records)} dialogues saved to {out_path}. Total spend this run: ${budget.spent:.3f}")


if __name__ == "__main__":
    main()
