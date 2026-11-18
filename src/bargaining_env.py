"""Alternating-offers bargaining environment with a known closed-form SPE.

Implements the Rubinstein (1982) alternating-offers game under three
conditions, matching the design in the Week 6 proposal:

  - **finite**: a hard deadline of T offers, common discount delta. The
    last offer is take-it-or-leave-it; if rejected, the dialogue ends in
    disagreement.
  - **infinite**: an infinite-horizon proxy implemented as a per-round
    *stochastic continuation* probability equal to delta (the standard
    risk-of-breakdown equivalence to time discounting: Binmore, Rubinstein
    and Wolinsky, 1986). After each rejection, the dialogue continues with
    probability delta and breaks down with probability (1 - delta). A
    round cap (see `infinite_round_cap`) bounds worst-case cost.
  - **asymmetric**: two distinct discount factors (delta_1 != delta_2). The
    breakdown-probability equivalence used for `infinite` does not have a
    clean two-discount-factor generalization, so this condition is
    implemented as *deterministic* continuation up to a safety-net round
    cap (documented simplification -- see README).

Disagreement convention (pre-specified, applied uniformly across all three
conditions, per the professor's progress-report feedback): any dialogue
that ends without an accepted offer -- deadline, stochastic breakdown, or
hitting the safety-net round cap -- pays (0, 0) to both players and is
logged with a `breakdown_reason`. Breakdown dialogues are excluded from the
primary distance-to-SPE statistics and reported separately as a breakdown
rate (see analysis.py).
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Callable, Literal, Optional

PIE = 100.0

ConditionType = Literal["finite", "infinite", "asymmetric"]


# --------------------------------------------------------------------------
# Closed-form SPE benchmarks
# --------------------------------------------------------------------------

def spe_share_infinite(delta: float) -> float:
    """First-mover share in the infinite-horizon, symmetric-delta SPE."""
    return 1.0 / (1.0 + delta)


@lru_cache(maxsize=None)
def spe_share_finite(delta: float, rounds_remaining: int) -> float:
    """Proposer's SPE share with `rounds_remaining` offers left (>=1), common delta.

    Backward induction: v_1 = 1 (last offer, proposer takes everything).
    v_k = 1 - delta * v_{k-1} for k >= 2 (the proposer offers the responder
    exactly their discounted continuation value of becoming proposer next
    round). This matches the closed form reported in the proposal, e.g.
    v_2 = 1 - delta for a two-round game.
    """
    if rounds_remaining <= 1:
        return 1.0
    v_prev = spe_share_finite(delta, rounds_remaining - 1)
    return 1.0 - delta * v_prev


def spe_share_asymmetric(delta_proposer: float, delta_other: float) -> float:
    """Stationary infinite-horizon SPE share for a proposer with `delta_proposer`
    facing a responder with `delta_other`: (1 - delta_other) / (1 - delta_proposer * delta_other).
    """
    return (1.0 - delta_other) / (1.0 - delta_proposer * delta_other)


def infinite_round_cap(delta: float, tail_prob: float = 0.01, hard_max: int = 50) -> int:
    """Safety-net round cap for the stochastic `infinite` condition.

    Chosen so that, absent agreement, the probability the dialogue would
    *naturally* still be running (i.e. P(continue)=delta each round) past
    the cap is at most `tail_prob`, subject to a hard ceiling `hard_max` on
    worst-case API cost. For delta=0.95 this caps the natural-breakdown
    tail probability at ~7.7% rather than 1% -- a documented truncation
    bias, traded off against call-budget control.
    """
    if delta <= 0:
        return 1
    natural_cap = math.ceil(math.log(tail_prob) / math.log(delta))
    return max(1, min(natural_cap, hard_max))


# --------------------------------------------------------------------------
# Game configuration and dialogue result
# --------------------------------------------------------------------------

@dataclass
class GameConfig:
    condition_type: ConditionType
    horizon: Optional[int] = None           # T, for "finite"
    delta: Optional[float] = None           # symmetric delta, for "finite" / "infinite"
    delta_1: Optional[float] = None         # asymmetric: first mover's delta
    delta_2: Optional[float] = None         # asymmetric: second mover's delta
    round_cap: Optional[int] = None         # override; computed if None
    pie: float = PIE

    def label(self) -> str:
        if self.condition_type == "finite":
            return f"finite_T{self.horizon}_d{self.delta}"
        if self.condition_type == "infinite":
            return f"infinite_d{self.delta}"
        return f"asymmetric_d1-{self.delta_1}_d2-{self.delta_2}"

    def effective_round_cap(self) -> int:
        if self.round_cap is not None:
            return self.round_cap
        if self.condition_type == "finite":
            return self.horizon
        if self.condition_type == "infinite":
            return infinite_round_cap(self.delta)
        return 50  # asymmetric safety-net default

    def spe_first_mover_share(self, rounds_remaining: Optional[int] = None) -> float:
        """SPE share (as a fraction of the pie, 0-1) for whoever proposes
        first in this configuration. For `finite`, this is evaluated at the
        full horizon unless `rounds_remaining` overrides it (used to score
        a dialogue against the benchmark for its *own* realized round, per
        the uniform disagreement-scoring rule)."""
        if self.condition_type == "finite":
            k = rounds_remaining if rounds_remaining is not None else self.horizon
            return spe_share_finite(self.delta, k)
        if self.condition_type == "infinite":
            return spe_share_infinite(self.delta)
        return spe_share_asymmetric(self.delta_1, self.delta_2)


BreakdownReason = Literal["deadline", "stochastic", "round_cap", "parse_failure"]


@dataclass
class RoundRecord:
    round_num: int
    proposer_idx: int
    offer: Optional[float]          # share (0-100) the proposer keeps for themselves
    accepted: Optional[bool]
    proposer_raw: Optional[dict] = None
    responder_raw: Optional[dict] = None


@dataclass
class DialogueResult:
    config_label: str
    agreed: bool
    agreement_round: Optional[int]
    first_mover_share: Optional[float]   # pie-share (0-100) realized by the round-1 proposer
    second_mover_share: Optional[float]
    rounds_played: int
    breakdown_reason: Optional[BreakdownReason]
    spe_benchmark_share: float           # benchmark the realized/attempted play is scored against
    history: list = field(default_factory=list)


# --------------------------------------------------------------------------
# Core engine (agent-agnostic: propose_fn / respond_fn are injected)
# --------------------------------------------------------------------------

ProposeFn = Callable[[GameConfig, int, int, list], dict]
# (config, proposer_idx, round_num, history) -> {"offer": float, "raw": dict}
RespondFn = Callable[[GameConfig, int, int, list, float], dict]
# (config, responder_idx, round_num, history, offer) -> {"accept": bool, "raw": dict}


def run_dialogue(
    config: GameConfig,
    propose_fn: ProposeFn,
    respond_fn: RespondFn,
    rng: random.Random,
) -> DialogueResult:
    """Run one alternating-offers dialogue. Player 0 proposes in round 1."""
    cap = config.effective_round_cap()
    history: list[RoundRecord] = []

    for round_num in range(1, cap + 1):
        proposer_idx = 0 if round_num % 2 == 1 else 1
        responder_idx = 1 - proposer_idx

        prop_result = propose_fn(config, proposer_idx, round_num, history)
        offer = prop_result.get("offer")

        if offer is None:
            # Unparseable proposal (model didn't follow the OFFER: <number> format,
            # or an API error occurred): treated as an immediate breakdown at this
            # round, tagged distinctly from running out of rounds.
            history.append(RoundRecord(round_num, proposer_idx, None, None, prop_result.get("raw")))
            return _breakdown(config, history, round_num, "parse_failure")

        resp_result = respond_fn(config, responder_idx, round_num, history, offer)
        accepted = resp_result.get("accept", False)
        history.append(RoundRecord(round_num, proposer_idx, offer, accepted,
                                    prop_result.get("raw"), resp_result.get("raw")))

        if accepted:
            return _agreement(config, history, round_num, proposer_idx, offer)

        # Rejected: decide whether the dialogue continues.
        if config.condition_type == "finite":
            if round_num == config.horizon:
                return _breakdown(config, history, round_num, "deadline")
            continue  # proceed to next round deterministically

        if config.condition_type == "infinite":
            if round_num == cap:
                return _breakdown(config, history, round_num, "round_cap")
            if rng.random() >= config.delta:
                return _breakdown(config, history, round_num, "stochastic")
            continue

        # asymmetric: deterministic continuation up to the safety-net cap
        if round_num == cap:
            return _breakdown(config, history, round_num, "round_cap")
        continue

    # Exhausted the loop without resolution (should not happen; safety net).
    return _breakdown(config, history, cap, "round_cap")


def _agreement(config: GameConfig, history: list, round_num: int,
               proposer_idx: int, offer: float) -> DialogueResult:
    proposer_share = offer
    responder_share = config.pie - offer
    first_mover_share = proposer_share if proposer_idx == 0 else responder_share
    second_mover_share = config.pie - first_mover_share
    # Primary distance-to-SPE benchmark: the FIXED, overall-game SPE
    # first-mover share (e.g. 84% for T=3, delta=0.8) -- not a round-local
    # renormalization -- regardless of which round agreement was actually
    # reached in. This is what "distance to SPE" means per the proposal's
    # problem statement; a round-dependent benchmark is used only for the
    # breakdown robustness check (see _breakdown below).
    benchmark = config.spe_first_mover_share()
    return DialogueResult(
        config_label=config.label(), agreed=True, agreement_round=round_num,
        first_mover_share=first_mover_share, second_mover_share=second_mover_share,
        rounds_played=round_num, breakdown_reason=None,
        spe_benchmark_share=benchmark * config.pie, history=history,
    )


def _breakdown(config: GameConfig, history: list, round_num: int,
               reason: BreakdownReason) -> DialogueResult:
    # Per the uniform disagreement rule: payoff (0, 0). The benchmark used for
    # robustness reporting is the finite-horizon SPE for the *realized* number
    # of rounds actually played (treating the stop as if it were a known
    # deadline of that length) -- see README / analysis.py.
    rounds_remaining_at_stop = max(1, round_num)
    if config.condition_type == "finite":
        benchmark = config.spe_first_mover_share(rounds_remaining=config.horizon - round_num + 1)
    elif config.condition_type == "infinite":
        benchmark = spe_share_finite(config.delta, rounds_remaining_at_stop)
    else:
        benchmark = config.spe_first_mover_share()
    return DialogueResult(
        config_label=config.label(), agreed=False, agreement_round=None,
        first_mover_share=0.0, second_mover_share=0.0,
        rounds_played=round_num, breakdown_reason=reason,
        spe_benchmark_share=benchmark * config.pie, history=history,
    )


# --------------------------------------------------------------------------
# The 7 game variants used in the Phase-1 (progress-report) grid
# --------------------------------------------------------------------------

def phase1_variants() -> list[GameConfig]:
    variants = []
    for delta in (0.5, 0.8, 0.95):
        variants.append(GameConfig(condition_type="infinite", delta=delta))
    for horizon in (2, 3, 5):
        # A single common delta is used for the finite-horizon sweep; 0.8 is
        # the mid-point of the infinite-horizon delta grid.
        variants.append(GameConfig(condition_type="finite", horizon=horizon, delta=0.8))
    variants.append(GameConfig(condition_type="asymmetric", delta_1=0.95, delta_2=0.5))
    return variants


if __name__ == "__main__":
    # Quick closed-form sanity checks (no API calls).
    assert abs(spe_share_infinite(0.5) - 2 / 3) < 1e-9
    assert abs(spe_share_finite(0.5, 2) - 0.5) < 1e-9       # v_2 = 1 - delta
    assert abs(spe_share_finite(0.8, 1) - 1.0) < 1e-9
    assert abs(spe_share_asymmetric(0.95, 0.5) - (1 - 0.5) / (1 - 0.95 * 0.5)) < 1e-9
    print("bargaining_env: closed-form checks passed.")
    for cfg in phase1_variants():
        print(f"  {cfg.label():28s} cap={cfg.effective_round_cap():3d}  "
              f"spe_first_mover_share={cfg.spe_first_mover_share():.3f}")
