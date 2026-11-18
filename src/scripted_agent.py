"""A scripted negotiator that plays the exact subgame-perfect equilibrium.

Used as the Phase-1 opponent (LLM vs. scripted SPE player) so that any
distance from equilibrium in a dialogue is attributable to the LLM, not to
noise from both sides. The scripted player always proposes its exact SPE
offer and always accepts iff the offer on the table meets or exceeds its
SPE continuation value.
"""
from __future__ import annotations

from bargaining_env import GameConfig, RoundRecord, spe_share_finite, spe_share_asymmetric

ACCEPT_EPS = 0.5
# Tolerance (points on the 0-100 pie) for the scripted responder's acceptance
# threshold. SPE shares are often repeating decimals (e.g. 1/(1+0.5) = 66.67%);
# a tolerance near zero would reject an LLM that rounds to "67" instead of
# "66.666...7" over a third of a point -- a rounding artifact, not a genuine
# strategic deviation. 0.5 is well below the 2.0-point tolerance used for the
# "SPE-consistent" deviation-taxonomy classification in analysis.py, so it
# does not materially affect the substantive distance-to-SPE measurements.


def _proposer_share(config: GameConfig, proposer_idx: int, round_num: int) -> float:
    """The scripted proposer's SPE self-share (0-1) at this round."""
    if config.condition_type == "finite":
        rounds_remaining = config.horizon - round_num + 1
        return spe_share_finite(config.delta, rounds_remaining)
    if config.condition_type == "infinite":
        return config.spe_first_mover_share()  # stationary
    # asymmetric: share depends on which physical player (and their delta) proposes
    delta_proposer = config.delta_1 if proposer_idx == 0 else config.delta_2
    delta_other = config.delta_2 if proposer_idx == 0 else config.delta_1
    return spe_share_asymmetric(delta_proposer, delta_other)


def _responder_threshold(config: GameConfig, responder_idx: int, round_num: int) -> float:
    """Minimum share (0-1) the scripted responder requires to accept."""
    if config.condition_type == "finite":
        rounds_remaining_next = config.horizon - round_num  # responder's offers left if they wait
        if rounds_remaining_next <= 0:
            # This is the last offer (round == horizon). Rejecting here pays (0, 0)
            # with certainty -- there is no further round -- so the responder
            # should accept any non-negative offer.
            return 0.0
        delta = config.delta
        return delta * spe_share_finite(delta, rounds_remaining_next)
    if config.condition_type == "infinite":
        delta = config.delta
        return delta * config.spe_first_mover_share()
    delta_responder = config.delta_1 if responder_idx == 0 else config.delta_2
    delta_other = config.delta_2 if responder_idx == 0 else config.delta_1
    own_share_as_proposer = spe_share_asymmetric(delta_responder, delta_other)
    return delta_responder * own_share_as_proposer


def scripted_propose(config: GameConfig, proposer_idx: int, round_num: int,
                      history: list[RoundRecord]) -> dict:
    share = _proposer_share(config, proposer_idx, round_num)
    offer = share * config.pie
    return {"offer": offer, "raw": {"agent": "scripted", "spe_share": share}}


def scripted_respond(config: GameConfig, responder_idx: int, round_num: int,
                      history: list[RoundRecord], offer: float) -> dict:
    offered_to_responder = config.pie - offer
    threshold = _responder_threshold(config, responder_idx, round_num) * config.pie
    accept = offered_to_responder >= threshold - ACCEPT_EPS
    return {"accept": accept, "raw": {"agent": "scripted", "threshold": threshold}}


if __name__ == "__main__":
    import random
    from bargaining_env import phase1_variants, run_dialogue

    rng = random.Random(0)
    print("Scripted-vs-scripted sanity check (should always agree in round 1):")
    for cfg in phase1_variants():
        result = run_dialogue(cfg, scripted_propose, scripted_respond, rng)
        ok = result.agreed and result.agreement_round == 1
        print(f"  {cfg.label():28s} agreed={result.agreed} round={result.agreement_round} "
              f"first_mover_share={result.first_mover_share:.2f}  {'OK' if ok else 'UNEXPECTED'}")
