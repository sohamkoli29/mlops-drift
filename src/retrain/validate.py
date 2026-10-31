"""Champion vs challenger on the same recent labeled holdout."""

import time
from dataclasses import asdict, dataclass

from sklearn.metrics import f1_score


@dataclass(frozen=True)
class ValidationConfig:
    min_f1_gain: float = 0.01  # challenger F1 must beat the champion by at least this
    max_latency_ratio: float = 2.0
    latency_slack_ms: float = 0.2  # per row, so tiny absolute differences are not "regressions"


@dataclass(frozen=True)
class ValidationResult:
    promote: bool
    reason: str
    champion_f1: float
    challenger_f1: float
    f1_gain: float
    champion_ms_per_row: float
    challenger_ms_per_row: float
    n_eval: int

    def to_dict(self) -> dict:
        return asdict(self)


def f1_at_half(model, X, y) -> float:
    pred = (model.predict_proba(X)[:, 1] >= 0.5).astype(int)  # same rule as the API
    return float(f1_score(y, pred, zero_division=0))


def ms_per_row(model, X, repeats: int = 3) -> float:
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        model.predict_proba(X)
        best = min(best, time.perf_counter() - start)
    return best * 1000 / len(X)


def compare(champion, challenger, X, y, cfg: ValidationConfig | None = None) -> ValidationResult:
    cfg = cfg or ValidationConfig()
    champ_f1, chall_f1 = f1_at_half(champion, X, y), f1_at_half(challenger, X, y)
    gain = chall_f1 - champ_f1
    champ_ms, chall_ms = ms_per_row(champion, X), ms_per_row(challenger, X)
    latency_ok = chall_ms <= max(champ_ms * cfg.max_latency_ratio, champ_ms + cfg.latency_slack_ms)

    if gain < cfg.min_f1_gain - 1e-12:
        promote, reason = False, f"F1 gain {gain:+.4f} is below the required {cfg.min_f1_gain}"
    elif not latency_ok:
        promote, reason = False, f"latency regression ({chall_ms:.3f} vs {champ_ms:.3f} ms/row)"
    else:
        promote, reason = True, f"F1 gain {gain:+.4f} and latency acceptable"

    return ValidationResult(
        promote=promote,
        reason=reason,
        champion_f1=round(champ_f1, 4),
        challenger_f1=round(chall_f1, 4),
        f1_gain=round(gain, 4),
        champion_ms_per_row=round(champ_ms, 4),
        challenger_ms_per_row=round(chall_ms, 4),
        n_eval=len(X),
    )
