"""Phase-1 experiment runner: LLM vs. scripted SPE opponent across the
7 game variants x 3 models x 2 prompt conditions (42 cells), N dialogues
per cell, resumable and budget-capped -- implementing the sample-size and
budget commitments made in response to the progress-report feedback.

Usage:
    python run_experiments.py --pilot              # 5 dialogues/cell (~210 total)
    python run_experiments.py --n 30                # full N=30/cell run
    python run_experiments.py --n 30 --budget 8.0   # stop early if $ spend hits cap
    python run_experiments.py --models gpt-5-mini    # restrict to one model
"""
from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path

from bargaining_env import phase1_variants, run_dialogue
from llm_agent import MODEL_CONFIGS, build_client, make_llm_agent
from scripted_agent import scripted_propose, scripted_respond

MODELS = list(MODEL_CONFIGS.keys())
PROMPT_CONDITIONS = [False, True]  # explain_spe

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# USD per 1,000 tokens (input, output). Claude Sonnet 4.5's price is from the
# TAMU API guide; GPT-5-mini / Gemini 2.5 Flash are not priced in the guide,
# so these use typical small-model list prices as a conservative
# placeholder -- reconcile against actual spend after the pilot run.
MODEL_COST_PER_1K = {
    "claude-sonnet-4.5": (0.003, 0.015),
    "gpt-5-mini": (0.00025, 0.002),
    "gemini-2.5-flash": (0.0003, 0.0025),
}


class Budget:
    def __init__(self, max_cost: float) -> None:
        self.max_cost = max_cost
        self.spent = 0.0

    def add_dollars(self, amount: float) -> None:
        self.spent += amount

    def exceeded(self) -> bool:
        return self.spent >= self.max_cost


def cell_key(config_label: str, model_key: str, explain_spe: bool) -> str:
    return f"{config_label}|{model_key}|{'explained' if explain_spe else 'unexplained'}"


def dialogue_to_dict(result, model_key: str, explain_spe: bool, trial_index: int) -> dict:
    d = asdict(result)
    d["model_key"] = model_key
    d["explain_spe"] = explain_spe
    d["trial_index"] = trial_index
    return d


def _mix_with_scripted(llm_propose_fn, llm_respond_fn):
    """Phase 1 is LLM (player 0, first mover) vs. the scripted SPE opponent
    (player 1) -- NOT LLM self-play. This wraps the LLM propose/respond
    functions so player 0's turns go to the LLM and player 1's turns go to
    the scripted agent, regardless of which index is proposing/responding
    in a given round."""

    def propose_fn(config, proposer_idx, round_num, history):
        if proposer_idx == 0:
            return llm_propose_fn(config, proposer_idx, round_num, history)
        return scripted_propose(config, proposer_idx, round_num, history)

    def respond_fn(config, responder_idx, round_num, history, offer):
        if responder_idx == 0:
            return llm_respond_fn(config, responder_idx, round_num, history, offer)
        return scripted_respond(config, responder_idx, round_num, history, offer)

    return propose_fn, respond_fn


def run_cell(client, config, model_key: str, explain_spe: bool, n: int, budget: Budget,
             retries: int, sleep_s: float, rng: random.Random, save_transcripts: bool) -> list[dict]:
    llm_propose_fn, llm_respond_fn = make_llm_agent(client, model_key, explain_spe, retries, sleep_s)
    propose_fn, respond_fn = _mix_with_scripted(llm_propose_fn, llm_respond_fn)
    cost_in, cost_out = MODEL_COST_PER_1K[model_key]
    records = []
    for trial_index in range(n):
        if budget.exceeded():
            break
        result = run_dialogue(config, propose_fn, respond_fn, rng)

        prompt_toks = completion_toks = 0
        for r in result.history:
            for raw in (r.proposer_raw, r.responder_raw):
                if raw and "prompt_tokens" in raw:
                    prompt_toks += raw.get("prompt_tokens", 0)
                    completion_toks += raw.get("completion_tokens", 0)
        budget.add_dollars(prompt_toks / 1000 * cost_in + completion_toks / 1000 * cost_out)

        if not save_transcripts:
            for r in result.history:
                if r.proposer_raw:
                    r.proposer_raw = {k: v for k, v in r.proposer_raw.items() if k != "text"}
                if r.responder_raw:
                    r.responder_raw = {k: v for k, v in r.responder_raw.items() if k != "text"}

        records.append(dialogue_to_dict(result, model_key, explain_spe, trial_index))
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Phase-1 bargaining grid (LLM vs. scripted SPE).")
    parser.add_argument("--n", type=int, default=30, help="Dialogues per cell (default: 30).")
    parser.add_argument("--pilot", action="store_true", help="Shortcut for --n 5.")
    parser.add_argument("--budget", type=float, default=8.0, help="Max USD spend this run (default: 8.0).")
    parser.add_argument("--out", type=str, default=str(DATA_DIR / "phase1_results.json"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--sleep", type=float, default=1.0)
    parser.add_argument("--models", type=str, default=",".join(MODELS),
                         help="Comma-separated subset of model keys to run.")
    parser.add_argument("--save-transcripts", action="store_true",
                         help="Keep full raw model text in the output file (larger; off by default).")
    args = parser.parse_args()

    n = 5 if args.pilot else args.n
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    for m in models:
        if m not in MODEL_CONFIGS:
            raise SystemExit(f"Unknown model key '{m}'. Choices: {', '.join(MODEL_CONFIGS)}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    existing: list[dict] = []
    if out_path.exists():
        existing = json.loads(out_path.read_text())["dialogues"]

    client = build_client()
    budget = Budget(args.budget)
    rng = random.Random(args.seed)

    variants = phase1_variants()
    cells = [(v, m, e) for v in variants for m in models for e in PROMPT_CONDITIONS]
    rng.shuffle(cells)

    all_records = list(existing)
    done_counts: dict[str, int] = {}
    for d in existing:
        k = cell_key(d["config_label"], d["model_key"], d["explain_spe"])
        done_counts[k] = done_counts.get(k, 0) + 1

    for cell_idx, (config, model_key, explain_spe) in enumerate(cells):
        if budget.exceeded():
            print(f"Budget cap (${budget.max_cost:.2f}) reached; stopping after "
                  f"{cell_idx}/{len(cells)} cells. Re-run later to resume.")
            break
        key = cell_key(config.label(), model_key, explain_spe)
        already = done_counts.get(key, 0)
        remaining = max(0, n - already)
        if remaining == 0:
            continue
        print(f"[{cell_idx + 1}/{len(cells)}] {key}: running {remaining} more dialogue(s) "
              f"(spend so far ${budget.spent:.3f})")
        new_records = run_cell(client, config, model_key, explain_spe, remaining, budget,
                                args.retries, args.sleep, rng, args.save_transcripts)
        all_records.extend(new_records)
        out_path.write_text(json.dumps({"dialogues": all_records}, indent=2))

    out_path.write_text(json.dumps({"dialogues": all_records}, indent=2))
    print(f"Done. {len(all_records)} dialogues saved to {out_path}. Total spend this run: ${budget.spent:.3f}")


if __name__ == "__main__":
    main()
