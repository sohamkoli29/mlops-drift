"""Window severity (none / medium / high) from per-feature detector results."""

from dataclasses import dataclass

from src.drift.base import FeatureScore
from src.drift.psi import PSI_HIGH, PSI_MEDIUM

LEVELS = ("none", "medium", "high")


@dataclass(frozen=True)
class SeverityConfig:
    medium_share: float = 0.15
    high_share: float = 0.40
    psi_medium: float = PSI_MEDIUM
    psi_high: float = PSI_HIGH


def level_rank(level: str) -> int:
    return LEVELS.index(level)


def share_flagged(results: dict[str, FeatureScore]) -> float:
    if not results:
        return 0.0
    return sum(bool(r.flagged) for r in results.values()) / len(results)


def window_severity(
    results: dict[str, FeatureScore],
    config: SeverityConfig | None = None,
    use_psi_rule: bool = False,
) -> str:
    """Severity of one detector's results on one window.

    use_psi_rule adds the max-PSI conditions; only meaningful for PSI scores.
    """
    cfg = config or SeverityConfig()
    if not results:
        return "none"
    share = share_flagged(results)
    max_psi = max(r.score for r in results.values()) if use_psi_rule else 0.0
    if share >= cfg.high_share or (use_psi_rule and max_psi > cfg.psi_high):
        return "high"
    if share >= cfg.medium_share or (use_psi_rule and max_psi >= cfg.psi_medium):
        return "medium"
    return "none"
