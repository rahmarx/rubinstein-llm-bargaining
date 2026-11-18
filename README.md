# Course Project: Distance to Subgame-Perfect Play in LLM Alternating-Offers Bargaining

CSCE 631-D, Fall 2026 | Raissa Marques da Silva | Seed 11

Implements the Phase-1 (progress-report) grid from the proposal: three LLM
families negotiating against a scripted Rubinstein-SPE opponent, across 7
game variants and 2 prompt conditions, with the sample size, confidence
intervals, call budget, and disagreement-scoring protocol specified in
response to the professor's proposal feedback.

## Setup

```bash
cd src
pip install -r requirements.txt
```

Set credentials — either export them, or copy `.env.example` to `.env` at
the project root and fill it in (`src/llm_agent.py` loads `.env`
automatically if present; real environment variables always take
precedence):

```bash
export TAMU_API_KEY=sk-...
export CF_COOKIE="CF_Authorization=eyJ..."
```

The cookie expires roughly every 24 hours — re-copy it from
`chat.tamu.ai` DevTools each session.

## Reproducing the results

1. **Pilot (recommended first)** — 5 dialogues/cell, ~210 dialogues, to
   measure real average rounds-to-resolution and reconcile the cost
   estimate below before committing to the full run:
   ```bash
   python run_experiments.py --pilot
   ```
2. **Full Phase-1 grid** — N=30 dialogues/cell:
   ```bash
   python run_experiments.py --n 30 --budget 8.0
   ```
   Resumable and budget-capped exactly like PA1/PA2's `run_experiments.py`
   / `llm_experiments.py`: cells are processed in a seeded-shuffle order
   (not one at a time), writes to `data/phase1_results.json` after every
   cell, and picks up where it left off if interrupted or the budget cap is
   hit. Use `--models gpt-5-mini,gemini-2.5-flash` to restrict to a subset.
3. **Analysis**:
   ```bash
   python analysis.py ../data/phase1_results.json
   ```
   Prints the per-cell distance-to-SPE mean with a 95% bootstrap CI, the
   breakdown rate with a 95% Wilson CI, and the deviation-taxonomy counts.

Quick offline checks (no API calls, no credentials needed) are built into
each module: `python bargaining_env.py`, `python scripted_agent.py`,
`python llm_agent.py` each run their own sanity assertions and print a
summary.

## Design, matching the professor's proposal feedback

**Sample size per cell.** 7 game variants (δ ∈ {0.5, 0.8, 0.95};
T ∈ {2, 3, 5}; one asymmetric-δ case) × 3 models × 2 prompt conditions = 42
cells, **N = 30 dialogues/cell = 1,260 dialogues** for Phase 1
(`bargaining_env.phase1_variants`).

**Confidence intervals** (`analysis.py`). Distance-to-SPE: percentile
bootstrap, 2,000 resamples (`bootstrap_ci`) — more robust than a normal
approximation at N=30, especially since distances are bounded below by 0 and
likely right-skewed. Breakdown/agreement rate: Wilson score interval
(`wilson_ci`) — robust when breakdowns are rare, unlike a normal
approximation near 0.

**Call budget.** Using a conservative round-cap-based upper bound (see
`bargaining_env.infinite_round_cap`): ~16,200 calls worst case, roughly $49
at a blended $0.003/call estimate. In practice, dialogues should mostly
resolve in 1–3 rounds (the SPE predicts immediate agreement), so real spend
is expected to be far lower — confirm with the `--pilot` run before scaling.
`run_experiments.py --budget` enforces a hard USD stop using per-model
token costs (`MODEL_COST_PER_1K`); **note the GPT-5-mini / Gemini 2.5 Flash
prices are placeholders** (the TAMU guide only prices Claude Sonnet 4.5) —
reconcile against actual spend after the pilot.

**Disagreement scoring** (`bargaining_env._breakdown`, `run_dialogue`). A
single rule applied uniformly across all three conditions: any dialogue that
ends without an accepted offer — hitting the finite-horizon deadline, the
infinite-horizon stochastic breakdown draw, or the safety-net round cap —
pays **(0, 0)** and is logged with a `breakdown_reason`
(`deadline` / `stochastic` / `round_cap` / `parse_failure`). Breakdowns are
excluded from the primary distance-to-SPE mean and reported separately as a
breakdown rate. For the **infinite** condition specifically, a breakdown is
additionally scored (for robustness, not the headline number) against the
finite-horizon SPE for its own realized stopping round — treating the random
stop as if it were a known deadline of that length — which is the explicit
link the feedback asked for between "stochastic continuation" and "the
finite-horizon benchmark."

**Agreements** are always scored against the fixed, overall-game SPE share
(e.g. 84% for T=3, δ=0.8), never a round-local renormalization — this
matters when agreement is reached later than round 1 (see "Bugs found"
below).

**Asymmetric-δ condition.** The breakdown-probability equivalence used for
the symmetric `infinite` condition does not have a clean two-discount-factor
generalization, so `asymmetric` uses deterministic continuation up to the
same safety-net round cap (50) rather than a stochastic draw — a documented
simplification, not the Binmore–Rubinstein–Wolinsky equivalence.

**Round cap for the infinite condition**
(`bargaining_env.infinite_round_cap`): chosen so that the "naturally still
running" tail probability is ≤1%, capped at 50 rounds regardless — for
δ=0.95 this caps the natural tail at ~7.7% rather than 1%, a documented
truncation bias traded off against worst-case cost.

**Deviation taxonomy** (`analysis.classify_deviation`): SPE-consistent
(within 2 points), fairness-pull (between the fair 50/50 split and the SPE
share), first-mover-exploitation (beyond the SPE share, same direction —
labeled second-mover-exploitation for the T=2 variant, the one case where
the SPE itself favors the second mover), or backward-induction-failure (the
realized split is on the opposite side of 50/50 from what the SPE predicts).

## Bugs found and fixed during live-API validation

Before any paid run, each module was validated against the real TAMU
gateway with single dialogues per model, which caught three real issues —
worth noting since none of them would show up in the scripted-vs-scripted
sanity check alone:

1. **`gpt-5-mini` returned empty text at `max_tokens=256–300`** despite the
   TAMU guide listing it under "standard parameters, small max_tokens" — it
   was silently spending its entire token budget on hidden reasoning tokens.
   Fixed by raising its `max_tokens` to 2000 (`llm_agent.MODEL_CONFIGS`).
2. **Scripted responder's final-round acceptance threshold was backwards**
   (`scripted_agent._responder_threshold`): at the literal last offer, it
   required ~100% of the pie instead of accepting any non-negative offer
   (rejecting is worthless either way once there's no round left). This
   would have made every finite-horizon dialogue that didn't resolve in
   round 1 falsely register as a deadline breakdown. Fixed, with a
   regression test in `scripted_agent.py`.
3. **Acceptance epsilon was too tight** (`1e-6`): SPE shares are often
   repeating decimals (e.g. 1/(1+0.5) = 66.666...7%), so an LLM that rounds
   to "67" instead of the exact value was being rejected over a third of a
   point — a rounding artifact, not a real strategic deviation. Widened to
   0.5 points (`scripted_agent.ACCEPT_EPS`), still well below the 2-point
   tolerance used for the "SPE-consistent" classification.

## Gateway infrastructure: shared daily budget cap (found during the real pilot run)

The first real pilot run (`--pilot`, N=5/cell, all 3 models) appeared to complete
cleanly ($1.32 spent, 210/210 dialogues) but turned out to have two problems,
both found by inspecting the data rather than trusting it:

1. **Response caching defeated the "independent trials" design.** Every cell
   showed exactly zero variance across all 5 trials (e.g. the same offer to
   the decimal, 5/5 times). Confirmed by sending the literal same prompt
   three times in a row and getting back byte-identical text, phrasing
   included — the gateway (or an underlying provider) caches responses for
   repeated identical requests. Since every trial in a cell starts from the
   same empty history, all 5 were effectively one cached draw. **Fixed** by
   appending a short per-call nonce to the user message
   (`llm_agent._nonce`), explicitly labeled as bookkeeping to ignore;
   verified live that this restores real variance (confirmed: same
   underlying decision, different phrasing, once the cache was defeated;
   separately confirmed genuine numeric variance across repeated calls after
   the fix, e.g. 99 / 0 / 0 / 0 / 0 across 5 trials of the same cell).
2. **A shared account-wide daily budget cap was hit mid-run**, after the
   first (cache-contaminated) pilot plus this session's live validation
   calls. The gateway returns a `429` with `"Budget Exceeded: Current
   spend: $20.00, Budget: $20.00"` — note this figure does not match the
   "$10/day" the professor mentioned raising it to; confirm the actual
   current cap with the professor or the TAMU usage dashboard rather than
   assuming either number. The cap is **account-wide, not per-model**: of
   40 affected dialogues in the re-run pilot, all three models hit it
   (20/70 Claude, 15/70 GPT-5-mini, 5/70 Gemini), roughly proportional to
   how late each cell fell in the shuffled work queue. A hit call returns
   empty text with no visible error unless the raw `error` field is
   inspected — left unfiltered, this silently masquerades as a 100%
   negotiation breakdown for whichever cells ran last, which is exactly
   what the first pass of this pilot's analysis showed for three Claude
   finite-horizon cells before this was caught.

   **Fixed** by excluding these dialogues from every statistic rather than
   counting them as breakdowns: `analysis.exclude_infra_errors` (used by
   both `analysis.py` and `plot_results.py`) detects the `"Budget
   Exceeded"` marker in any round's raw API error and treats that dialogue
   as missing data, not a disagreement. Both scripts print which cells are
   short before showing results. The invalidated first pilot is kept for
   the record at `data/phase1_results_INVALID_cached_pilot.json`
   (pre-dates the nonce fix — do not use for analysis).

   **A second, silent variant of the same cap was found on the next day**:
   a backfill attempt of the 8 missing cells reported `$0.000` spend across
   every cell, which turned out to mean the account was *still* fully
   blocked (confirmed live against all three models simultaneously) despite
   the error message's "resets daily" claim — this time with no exception
   and no error text at all, just a completely empty response stream (0
   chunks). `is_infra_error` was extended to also catch this signature:
   zero `prompt_tokens` and zero `completion_tokens` with no error, which
   only happens when the gateway never actually processed the request (a
   real call always bills at least the input tokens). Guarded against
   false-positiving on the scripted agent's own raw records, which have no
   `prompt_tokens` key at all. The cap did clear later that day; the 8
   cells were successfully backfilled ($0.42 spent) and the full 42-cell,
   210-dialogue pilot (`data/phase1_results.json`) is now 100% clean —
   `analysis.py`/`plot_results.py` print nothing excluded.

   **For the full N=30 run**: budget resets are real but can lag the
   stated "daily" cadence by at least several hours — don't assume a fresh
   calendar day means a fresh budget; do a cheap live check (see the
   one-liner in this section's git history, or just run `--pilot` on one
   cell) before committing to a long run. `run_experiments.py`'s resume
   logic counts raw record counts, not validity, so after any run that hits
   the cap, strip tainted records from `data/phase1_results.json` with
   `analysis.exclude_infra_errors` *before* resuming, or the resume logic
   will think those cells are already done. Spread the full grid across
   multiple days regardless. **Caveat**: one of the two apparent "budget cap"
   incidents during N=30 scale-up turned out on closer inspection to actually
   be an expired `CF_Authorization` cookie (the API gateway sits behind
   Cloudflare Access, whose session cookie expires roughly every 24h — see
   below), not the account spend cap. Both failure modes can look similar
   from the outside (calls fail or return nothing), so treat "the cap reset
   late" as one possible explanation among a few, not a confirmed mechanism.

## Client-side hang during the overnight N=30 run

The resumed N=30 run (after the first backfill) stalled silently for 11+
hours with no exception and no progress — the output file stopped updating
mid-run with the process still technically alive. Root cause: the
`openai.OpenAI` client in `llm_agent.build_client()` had no timeout
configured, so a single hung streaming connection (most likely triggered by
the laptop sleeping overnight) blocked the whole sequential run forever,
since there was nothing to time it out and raise an exception for the
existing retry logic to catch. **Fixed** by setting `timeout=90.0,
max_retries=0` on the client (the explicit `max_retries=0` avoids stacking
the client's own retry on top of `call_llm`'s existing retry loop, which
would otherwise multiply any future stall). Confirmed working: the run was
restarted for the 9 cells still short and completed the remaining 225
dialogues without further incident.

An important side note while diagnosing this: a separate expired-cookie
incident (not this hang) was found and fixed earlier the same day by
checking whether a plain non-streaming API call returned the literal
Cloudflare Access login page HTML instead of a parsed response — a quick
way to distinguish "cookie expired" from "budget cap hit" from "network
hung," since all three can otherwise look like generic failures.

**Final dataset**: all 42 cells reached N=30 (1,260 dialogues total,
confirmed clean — `analysis.py` reports zero excluded infra errors). One
dialogue hit the budget cap partway through and was stripped and
backfilled individually before the final run.

## Known limitations

- GPT-5-mini / Gemini 2.5 Flash per-token costs in `MODEL_COST_PER_1K` are
  estimates, not sourced from the TAMU guide (which only prices Claude
  Sonnet 4.5) — treat the budget projection as approximate until reconciled
  against a pilot run.
- An unparseable *response* (accept/reject) is treated as a reject-and-continue
  (flagged via `parse_failed` in the raw record) rather than a breakdown,
  whereas an unparseable *offer* ends the dialogue (`parse_failure`
  breakdown) since there's no numeric value to act on. This asymmetry is
  intentional but worth knowing when reading breakdown-rate numbers.
- Full model transcripts are stripped by default to keep `data/*.json`
  small; pass `--save-transcripts` to keep them (useful for qualitative
  read-through of a few cells, e.g. for the deviation-taxonomy write-up).

## Phase 2 (predict-then-play, LLM-vs-LLM) and the final report

Both Phase 2 extensions described as future work in the progress report are
now complete: `predict_then_play.py` (210/210 dialogues, all 21 unexplained
cells) and `llm_vs_llm.py` (240/240 dialogues, all 24 cells) ran on
2026-11-18 under the corrected token caps. Running the follow-up extensions
is in fact what surfaced the Phase-1 token-truncation bug documented above —
the gpt-5-mini/gemini-2.5-flash caps had already been raised for this reason
before Phase 2 data collection began, so none of the Phase 2 numbers are
affected by it, and Phase 2a's before/after breakdown-rate comparison
(Section 4.4 of the final report) is the cleanest evidence that the Phase-1
fix mattered.

`Final_Report.docx` (built by `build_final_report.py` from a markdown draft)
is the submitted writeup: 11 pages of main text plus references and a
reproducibility appendix, covering both phases, a deviation taxonomy, and
the methodological finding above. `plot_predict_then_play.py` and
`plot_llm_vs_llm.py` generate the Phase 2 figures the same way
`plot_results.py` generates the Phase 1 ones.
