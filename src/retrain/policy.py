"""Retrain decision policy: severity, detector agreement, persistence and cooldown."""

import json
import os
from dataclasses import asdict, dataclass, fields
from pathlib import Path


@dataclass(frozen=True)
class PolicyConfig:
    min_agreeing: int = 2  # detectors (of 3) whose severity is above none
    medium_persistence: int = 3  # consecutive drifted windows needed when severity never hits high
    cooldown_windows: int = 3  # windows ignored after a retrain attempt
    rollback_cooldown_windows: int = 10  # longer pause after a rollback


@dataclass
class PolicyState:
    last_window: int = -1
    streak: int = 0
    streak_first_id: int | None = None
    cooldown_left: int = 0
    pending: bool = False
    drift_first_id: int | None = None
    trigger_window: int | None = None
    trigger_reason: str | None = None


@dataclass(frozen=True)
class Decision:
    window: int
    trigger: bool
    reason: str


def load_state(path) -> PolicyState:
    p = Path(path)
    if not p.exists():
        return PolicyState()
    data = json.loads(p.read_text())
    known = {f.name for f in fields(PolicyState)}
    return PolicyState(**{k: v for k, v in data.items() if k in known})


def save_state(path, state: PolicyState) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(asdict(state), indent=2))
    os.replace(tmp, p)


def _reset_streak(state: PolicyState) -> None:
    state.streak = 0
    state.streak_first_id = None


def decide(a: dict, state: PolicyState, cfg: PolicyConfig) -> Decision:
    """Feed one analysed window (monitor output). Mutates state, returns the decision."""
    window, severity = a["window"], a["severity"]
    state.last_window = window

    if state.cooldown_left > 0:
        state.cooldown_left -= 1
        _reset_streak(state)
        return Decision(window, False, f"cooldown ({state.cooldown_left} more windows)")
    if severity == "none":
        _reset_streak(state)
        return Decision(window, False, "no drift")

    if state.streak == 0:
        state.streak_first_id = a["first_id"]
    state.streak += 1

    if state.pending:
        return Decision(window, False, "retrain already pending")
    if a["n_agreeing"] < cfg.min_agreeing:
        return Decision(
            window,
            False,
            f"{severity} drift but only {a['n_agreeing']} detector(s) agree, "
            f"need {cfg.min_agreeing}",
        )
    if severity == "high":
        reason = "high severity with detector agreement"
    elif state.streak >= cfg.medium_persistence:
        reason = f"drift for {state.streak} consecutive windows with detector agreement"
    else:
        return Decision(
            window, False, f"{severity} drift, streak {state.streak}/{cfg.medium_persistence}"
        )

    state.pending = True
    state.drift_first_id = state.streak_first_id
    state.trigger_window = window
    state.trigger_reason = reason
    return Decision(window, True, reason)


def advance(state: PolicyState, analyses: list[dict], cfg: PolicyConfig) -> list[Decision]:
    return [decide(a, state, cfg) for a in sorted(analyses, key=lambda a: a["window"])]


def on_retrain_done(state: PolicyState, cfg: PolicyConfig) -> None:
    """Called after a retrain attempt, whether the challenger was promoted or rejected."""
    state.pending = False
    state.drift_first_id = None
    state.trigger_window = None
    state.trigger_reason = None
    state.cooldown_left = cfg.cooldown_windows
    _reset_streak(state)


def on_rollback(state: PolicyState, cfg: PolicyConfig) -> None:
    state.cooldown_left = cfg.rollback_cooldown_windows
    _reset_streak(state)
