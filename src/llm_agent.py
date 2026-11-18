"""LLM negotiator: wraps the TAMU API gateway and turns model output into
propose_fn / respond_fn callables usable by bargaining_env.run_dialogue.

Follows the same client/credential pattern as PA1/PA2 (TAMU_API_KEY +
CF_COOKIE env vars). Offers and decisions are elicited via an explicit
end-of-response tag (e.g. "OFFER: 55", "DECISION: ACCEPT"), the same
robust-parsing style used in PA2's Rock-Paper-Scissors harness -- this is
more tolerant of Claude's thinking-mode preamble than requiring strict JSON.
"""
from __future__ import annotations

import os
import re
import time
import uuid
from pathlib import Path

import openai

from bargaining_env import GameConfig, RoundRecord

BASE_URL = "https://chat.tamu.ai/api"


def _load_dotenv(path: Path) -> None:
    """Minimal .env loader (no external dependency): sets os.environ for any
    KEY=VALUE line not already present in the environment. Real environment
    variables always take precedence over the file."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if key and key not in os.environ:
            os.environ[key] = value.strip()


_load_dotenv(Path(__file__).resolve().parent.parent / ".env")

MODEL_CONFIGS: dict[str, dict] = {
    "claude-sonnet-4.5": {
        "model_id": "protected.Claude Sonnet 4.5",
        "temperature": 1.0,
        "max_tokens": 16384,   # thinking-mode minimum per the TAMU API guide
    },
    "gpt-5-mini": {
        "model_id": "protected.gpt-5-mini",
        "temperature": 0.7,
        # Despite the TAMU guide listing gpt-5-mini under "standard parameters,
        # small max_tokens", live testing showed it silently consumes its
        # entire token budget on hidden reasoning tokens. 2000 was enough for a
        # round-1 proposal with no history, but Phase 2 (LLM-vs-LLM) surfaced
        # cases where a longer prompt (growing round history) pushed it over
        # 2000 again, truncating before the OFFER/DECISION tag -- confirmed by
        # checking that 100% of Phase 1's gpt-5-mini "parse_failure" breakdowns
        # (30/30) landed at exactly completion_tokens=2000. Raised to 4000.
        "max_tokens": 4000,
    },
    "gemini-2.5-flash": {
        "model_id": "protected.gemini-2.5-flash",
        "temperature": 0.7,
        # Same truncation bug as gpt-5-mini above, far more severe here: 100%
        # of Phase 1's gemini "parse_failure" breakdowns (142/142) landed at
        # exactly completion_tokens=300 (the old cap), meaning Gemini's
        # elevated Phase-1 breakdown rate is partly a token-budget artifact,
        # not a genuine negotiation outcome. Unlike gpt-5-mini, this did NOT
        # resolve by raising the cap moderately: at 1000 it truncated at ~996,
        # at 3000 it truncated at ~2996 -- the truncation point tracks the cap
        # almost exactly, meaning Gemini is spending its ENTIRE budget on
        # hidden reasoning every time, not occasionally running short. Treated
        # like Claude's thinking-mode minimum (see claude-sonnet-4.5 above)
        # and given the same generous budget.
        "max_tokens": 16384,
    },
}

OFFER_RE = re.compile(r"OFFER:\s*(-?\d+(?:\.\d+)?)", re.IGNORECASE)
DECISION_RE = re.compile(r"DECISION:\s*(ACCEPT|REJECT)", re.IGNORECASE)
PREDICTION_RE = re.compile(r"PREDICTION:\s*(-?\d+(?:\.\d+)?)", re.IGNORECASE)


def build_client() -> openai.OpenAI:
    api_key = os.environ.get("TAMU_API_KEY")
    cf_cookie = os.environ.get("CF_COOKIE")
    if not api_key or not cf_cookie:
        raise SystemExit("Missing credentials. Set TAMU_API_KEY and CF_COOKIE environment variables "
                          "(see .env.example).")
    # Explicit timeout: the default client had none set, and a run stalled
    # for 11+ hours overnight on a single hung streaming connection (likely
    # the laptop sleeping mid-request) with no exception ever raised, since
    # there was nothing to time it out. 90s is generous enough for Claude's
    # thinking-mode latency but bounds the worst case. max_retries=0 because
    # call_llm already retries explicitly -- the client's own built-in retry
    # would otherwise stack on top of that, multiplying any stall.
    return openai.OpenAI(base_url=BASE_URL, api_key=api_key, default_headers={"Cookie": cf_cookie},
                          timeout=90.0, max_retries=0)


def call_llm(client: openai.OpenAI, model_key: str, system_prompt: str, user_message: str,
             retries: int = 3, sleep_s: float = 1.0) -> dict:
    cfg = MODEL_CONFIGS[model_key]
    last_error: str | None = None
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=cfg["model_id"],
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                temperature=cfg["temperature"],
                max_tokens=cfg["max_tokens"],
                stream=True,
                stream_options={"include_usage": True},
            )
            text, usage = "", None
            for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    text += chunk.choices[0].delta.content
                if getattr(chunk, "usage", None):
                    usage = chunk.usage
            time.sleep(sleep_s)
            return {
                "text": text,
                "prompt_tokens": getattr(usage, "prompt_tokens", 0) if usage else 0,
                "completion_tokens": getattr(usage, "completion_tokens", 0) if usage else 0,
                "error": None,
            }
        except Exception as exc:  # gateway errors are varied; retry uniformly
            last_error = str(exc)
            time.sleep(sleep_s * (attempt + 1))
    return {"text": "", "prompt_tokens": 0, "completion_tokens": 0, "error": last_error}


def parse_offer(text: str) -> float | None:
    matches = OFFER_RE.findall(text)
    if not matches:
        return None
    try:
        value = float(matches[-1])
    except ValueError:
        return None
    if 0.0 <= value <= 100.0:
        return value
    return None


def parse_decision(text: str) -> bool | None:
    matches = DECISION_RE.findall(text)
    if not matches:
        return None
    return matches[-1].upper() == "ACCEPT"


def parse_prediction(text: str) -> float | None:
    matches = PREDICTION_RE.findall(text)
    if not matches:
        return None
    try:
        value = float(matches[-1])
    except ValueError:
        return None
    if 0.0 <= value <= 100.0:
        return value
    return None


# --------------------------------------------------------------------------
# Prompting
# --------------------------------------------------------------------------

_BASE_RULES = (
    "You are negotiating over a pie worth {pie:.0f} points with one other player, using the "
    "alternating-offers protocol: players take turns proposing how to split the pie, and the other "
    "player accepts (the split is final) or rejects (the negotiation continues, as described below). "
    "You are player {you_label}; the other player is player {other_label}. "
)

_HORIZON_TEXT = {
    "finite": (
        "This negotiation ends after at most {horizon} offers. If the final offer (offer {horizon}) "
        "is rejected, the negotiation breaks down and BOTH players get 0 points. "
    ),
    "infinite": (
        "After any rejected offer, the negotiation continues to the next round with probability "
        "{delta:.2f} and breaks down (both players get 0 points) with probability {one_minus_delta:.2f}. "
        "This probability is the same after every rejection, so the negotiation could in principle "
        "continue for many rounds. "
    ),
    "asymmetric": (
        "This negotiation has no fixed deadline known to you in advance, but it will not continue "
        "forever; keep responses efficient. If no agreement is ever reached, both players get 0 points. "
    ),
}

_SPE_HINT = (
    "Game-theoretic note: this game has a unique subgame-perfect equilibrium. If you reason about it "
    "carefully, the rational, equilibrium outcome is for the first proposer to offer themselves a share "
    "of approximately {spe_pct:.1f}% of the pie, and for that offer to be accepted immediately. "
)

_PROPOSE_INSTR = (
    "It is round {round_num}. You are proposing now. State how many of the {pie:.0f} points you want "
    "to keep for yourself (the rest goes to the other player). "
    "End your response with exactly one line: OFFER: <number>"
)

_RESPOND_INSTR = (
    "It is round {round_num}. The other player has offered to keep {offer:.1f} points for themselves, "
    "leaving you {remainder:.1f} points. Decide whether to accept or reject. "
    "End your response with exactly one line: DECISION: ACCEPT or DECISION: REJECT"
)

_PREDICT_INSTR = (
    "Before the negotiation begins, state your best estimate of the game-theoretically rational "
    "(equilibrium) outcome of this exact game: specifically, what share of the {pie:.0f} points the "
    "first proposer would keep for themselves under fully rational play by both sides. This is only "
    "a prediction and does not commit you to anything in the negotiation that follows. "
    "End your response with exactly one line: PREDICTION: <number>"
)


def _history_text(history: list[RoundRecord]) -> str:
    if not history:
        return "No offers have been made yet."
    lines = ["History so far:"]
    for r in history:
        status = "accepted" if r.accepted else "rejected"
        lines.append(f"  Round {r.round_num}: player {'A' if r.proposer_idx == 0 else 'B'} proposed to "
                      f"keep {r.offer:.1f} for themselves; the other player {status} it.")
    return "\n".join(lines)


def _system_prompt(config: GameConfig, player_idx: int, explain_spe: bool) -> str:
    you_label = "A" if player_idx == 0 else "B"
    other_label = "B" if player_idx == 0 else "A"
    text = _BASE_RULES.format(pie=config.pie, you_label=you_label, other_label=other_label)
    if config.condition_type == "finite":
        text += _HORIZON_TEXT["finite"].format(horizon=config.horizon)
    elif config.condition_type == "infinite":
        text += _HORIZON_TEXT["infinite"].format(delta=config.delta, one_minus_delta=1 - config.delta)
    else:
        text += _HORIZON_TEXT["asymmetric"]
    if explain_spe:
        spe_share = config.spe_first_mover_share()
        text += _SPE_HINT.format(spe_pct=spe_share * 100)
    return text


def _nonce() -> str:
    """A short per-call token appended to the user message. The TAMU gateway
    (or an underlying provider) was found empirically to cache responses for
    byte-identical requests -- confirmed by sending the exact same prompt
    three times in a row and getting back identical text, phrasing included.
    Since every trial within a cell starts from the same empty history, all
    "independent" trials would otherwise collapse to a single cached draw
    (this is exactly what happened in the first real pilot run: every cell
    showed zero variance across 5 trials). The nonce is framed explicitly as
    bookkeeping to ignore, so it shouldn't bias the actual decision -- and
    empirically it doesn't change the substance of the answer, only defeats
    the cache (verified: same offer value, different phrasing, once added)."""
    return uuid.uuid4().hex[:10]


def make_llm_agent(client: openai.OpenAI, model_key: str, explain_spe: bool, retries: int, sleep_s: float):
    """Returns (propose_fn, respond_fn) bound to this model/prompt condition."""

    def propose_fn(config: GameConfig, proposer_idx: int, round_num: int, history: list) -> dict:
        system = _system_prompt(config, proposer_idx, explain_spe)
        user = (_history_text(history) + "\n\n" + _PROPOSE_INSTR.format(round_num=round_num, pie=config.pie)
                + f"\n\n(internal trial reference, not part of the negotiation: {_nonce()})")
        result = call_llm(client, model_key, system, user, retries, sleep_s)
        offer = None if result["error"] else parse_offer(result["text"])
        return {"offer": offer, "raw": result}

    def respond_fn(config: GameConfig, responder_idx: int, round_num: int, history: list, offer: float) -> dict:
        system = _system_prompt(config, responder_idx, explain_spe)
        remainder = config.pie - offer
        user = (_history_text(history) + "\n\n" + _RESPOND_INSTR.format(
                    round_num=round_num, offer=offer, remainder=remainder)
                + f"\n\n(internal trial reference, not part of the negotiation: {_nonce()})")
        result = call_llm(client, model_key, system, user, retries, sleep_s)
        decision = None if result["error"] else parse_decision(result["text"])
        accept = bool(decision) if decision is not None else False  # unparseable -> treat as reject
        return {"accept": accept, "raw": result, "parse_failed": decision is None}

    return propose_fn, respond_fn


def make_predictor_fn(client: openai.OpenAI, model_key: str, retries: int, sleep_s: float):
    """Predict-then-play extension: elicits the model's belief about the SPE
    share BEFORE it plays, as a side-channel question that never enters the
    negotiation transcript. Only meaningful under the unexplained condition
    (explain_spe=False) -- under explained, the SPE is just handed to it, so
    "predicting" it is circular. Always asked from player 0's seat, since
    player 0 is the one that goes on to actually play the first-mover role."""

    def predict_fn(config: GameConfig) -> dict:
        system = _system_prompt(config, player_idx=0, explain_spe=False)
        user = _PREDICT_INSTR.format(pie=config.pie) + \
            f"\n\n(internal trial reference, not part of the negotiation: {_nonce()})"
        result = call_llm(client, model_key, system, user, retries, sleep_s)
        prediction = None if result["error"] else parse_prediction(result["text"])
        return {"prediction": prediction, "raw": result}

    return predict_fn


if __name__ == "__main__":
    # Offline parser checks -- no API calls.
    assert parse_offer("I'll propose fairly.\nOFFER: 55") == 55.0
    assert parse_offer("OFFER: 61.5 points for me") == 61.5
    assert parse_offer("no tag here") is None
    assert parse_decision("Thinking...\nDECISION: ACCEPT") is True
    assert parse_decision("DECISION: REJECT") is False
    assert parse_decision("unclear") is None
    assert parse_prediction("I estimate...\nPREDICTION: 66.7") == 66.7
    assert parse_prediction("no tag here") is None
    print("llm_agent: parser checks passed.")
