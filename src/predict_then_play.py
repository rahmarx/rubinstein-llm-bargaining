"""Phase-2 predict-then-play extension: before each dialogue, the LLM is
asked (as a side-channel question, outside the negotiation transcript) to
predict the rational/equilibrium first-mover share for the exact game it is
about to play -- THEN it plays that game against the scripted SPE opponent,
exactly as in Phase 1's unexplained condition.

This separates two things Phase 1's headline "knows vs. plays" gap
conflates: does the model fail to compute the SPE at all, or does it
compute it and then not act on it? Only meaningful under the unexplained
condition (explain_spe=True would just be re-reading the hint back).

Usage:
    python predict_then_play.py                       # N=10/cell, all 3 models, 7 variants
    python predict_then_play.py --n 10 --budget 3.0
"""
from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path

from bargaining_env import phase1_variants, run_dialogue
from llm_agent import MODEL_CONFIGS, build_client, make_llm_agent, make_predictor_fn
from run_experiments import MODEL_COST_PER_1K, Budget, _mix_with_scripted

MODELS = list(MODEL_CONFIGS.keys())
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def cell_key(config_label: str, model_key: str) -> str:
    return f"{config_label}|{model_key}"


def run_cell(client, config, model_key: str, n: int, budget: Budget,
             retries: int, sleep_s: float, rng: random.Random, save_transcripts: bool) -> list[dict]:
    predict_fn = make_predictor_fn(client, model_key, retries, sleep_s)
    llm_propose_fn, llm_respond_fn = make_llm_agent(client, model_key, explain_spe=False,
                                                      retries=retries, sleep_s=sleep_s)
    propose_fn, respond_fn = _mix_with_scripted(llm_propose_fn, llm_respond_fn)
    cost_in, cost_out = MODEL_COST_PER_1K[model_key]
    records = []

    for trial_index in range(n):
        if budget.exceeded():
            break

        pred_result = predict_fn(config)
        pred_raw = pred_result["raw"]
        budget.add_dollars(pred_raw.get("prompt_tokens", 0) / 1000 * cost_in
                            + pred_raw.get("completion_tokens", 0) / 1000 * cost_out)

        result = run_dialogue(config, propose_fn, respond_fn, rng)
        for r in result.history:
            for raw in (r.proposer_raw, r.responder_raw):
                if raw and "prompt_tokens" in raw:
                    budget.add_dollars(raw.get("prompt_tokens", 0) / 1000 * cost_in
                                        + raw.get("completion_tokens", 0) / 1000 * cost_out)

        if not save_transcripts:
            pred_raw = {k: v for k, v in pred_raw.items() if k != "text"}
            for r in result.history:
                if r.proposer_raw:
                    r.proposer_raw = {k: v for k, v in r.proposer_raw.items() if k != "text"}
                if r.responder_raw:
                    r.responder_raw = {k: v for k, v in r.responder_raw.items() if k != "text"}

        d = asdict(result)
        d["model_key"] = model_key
        d["trial_index"] = trial_index
        d["prediction"] = pred_result["prediction"]
        d["prediction_raw"] = pred_raw
        records.append(d)

    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the predict-then-play extension (unexplained condition only).")
    parser.add_argument("--n", type=int, default=10, help="Dialogues per cell (default: 10).")
    parser.add_argument("--budget", type=float, default=3.0, help="Max USD spend this run (default: 3.0).")
    parser.add_argument("--out", type=str, default=str(DATA_DIR / "phase2_predict_then_play.json"))
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--sleep", type=float, default=1.0)
    parser.add_argument("--models", type=str, default=",".join(MODELS))
    parser.add_argument("--save-transcripts", action="store_true")
    args = parser.parse_args()

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
    cells = [(v, m) for v in variants for m in models]
    rng.shuffle(cells)

    all_records = list(existing)
    done_counts: dict[str, int] = {}
    for d in existing:
        k = cell_key(d["config_label"], d["model_key"])
        done_counts[k] = done_counts.get(k, 0) + 1

    for cell_idx, (config, model_key) in enumerate(cells):
        if budget.exceeded():
            print(f"Budget cap (${budget.max_cost:.2f}) reached; stopping after "
                  f"{cell_idx}/{len(cells)} cells. Re-run later to resume.")
            break
        key = cell_key(config.label(), model_key)
        already = done_counts.get(key, 0)
        remaining = max(0, args.n - already)
        if remaining == 0:
            continue
        print(f"[{cell_idx + 1}/{len(cells)}] {key}: running {remaining} more dialogue(s) "
              f"(spend so far ${budget.spent:.3f})")
        new_records = run_cell(client, config, model_key, remaining, budget,
                                args.retries, args.sleep, rng, args.save_transcripts)
        all_records.extend(new_records)
        out_path.write_text(json.dumps({"dialogues": all_records}, indent=2))

    out_path.write_text(json.dumps({"dialogues": all_records}, indent=2))
    print(f"Done. {len(all_records)} dialogues saved to {out_path}. Total spend this run: ${budget.spent:.3f}")


if __name__ == "__main__":
    main()
