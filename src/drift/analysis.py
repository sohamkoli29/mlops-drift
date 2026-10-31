"""Per-window analysis: detector results -> severity, top-k features, explanation dict."""

import pandas as pd

from src.config import CATEGORICAL_FEATURES
from src.drift.base import Detector, FeatureScore, numeric_values
from src.drift.detectors import get_detectors
from src.drift.profile import MISSING, category_shares
from src.drift.severity import SeverityConfig, level_rank, share_flagged, window_severity


def _numeric_change(reference: pd.DataFrame, current: pd.DataFrame, col: str) -> dict:
    ref = numeric_values(reference, col)
    cur = numeric_values(current, col)
    std = float(ref.std(ddof=1)) if len(ref) > 1 else 0.0
    ref_mean, cur_mean = float(ref.mean()), float(cur.mean())
    return {
        "type": "numeric",
        "ref_mean": round(ref_mean, 3),
        "cur_mean": round(cur_mean, 3),
        "mean_shift_in_ref_std": round((cur_mean - ref_mean) / std, 3) if std > 0 else None,
    }


def _category_change(reference: pd.DataFrame, current: pd.DataFrame, col: str) -> dict:
    filled = reference[col].astype(object).where(reference[col].notna(), MISSING)
    categories = sorted(filled.unique())
    ref_sh = category_shares(reference[col], categories)
    cur_sh = category_shares(current[col], categories)
    biggest = max(ref_sh, key=lambda k: abs(cur_sh[k] - ref_sh[k]))
    return {
        "type": "categorical",
        "category": biggest,
        "ref_share": round(ref_sh[biggest], 3),
        "cur_share": round(cur_sh[biggest], 3),
    }


def _describe(feature: str, reference: pd.DataFrame, current: pd.DataFrame) -> dict:
    if feature in CATEGORICAL_FEATURES:
        return _category_change(reference, current, feature)
    return _numeric_change(reference, current, feature)


def _top_features(
    results: dict[str, dict[str, FeatureScore]],
    reference: pd.DataFrame,
    current: pd.DataFrame,
    k: int,
) -> list[dict]:
    ranking = results.get("psi") or next(iter(results.values()))
    flagged_by: dict[str, list[str]] = {}
    for det, res in results.items():
        for feature, r in res.items():
            if r.flagged:
                flagged_by.setdefault(feature, []).append(det)

    def strength(feature: str) -> float:
        return ranking[feature].score if feature in ranking else 0.0

    top = sorted(flagged_by, key=strength, reverse=True)[:k]
    entries = []
    for feature in top:
        scores: dict[str, float] = {}
        for det, res in results.items():
            if feature in res:
                scores[det] = round(res[feature].score, 4)
                if res[feature].p_value is not None:
                    scores[f"{det}_p_value"] = float(f"{res[feature].p_value:.3g}")
        entries.append(
            {
                "feature": feature,
                "flagged_by": sorted(flagged_by[feature]),
                "scores": scores,
                "change": _describe(feature, reference, current),
            }
        )
    return entries


def analyze_window(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    detectors: dict[str, Detector] | None = None,
    k: int = 3,
    config: SeverityConfig | None = None,
    window: int | None = None,
) -> dict:
    """Score one window with every detector; return a JSON-serialisable explanation."""
    detectors = detectors or get_detectors()
    results = {name: det.score(reference, current) for name, det in detectors.items()}

    per_detector = {}
    for name, res in results.items():
        flagged = sorted(f for f, r in res.items() if r.flagged)
        per_detector[name] = {
            "severity": window_severity(res, config, use_psi_rule=(name == "psi")),
            "n_flagged": len(flagged),
            "n_features": len(res),
            "flagged_features": flagged,
        }

    if "psi" in per_detector:
        severity = per_detector["psi"]["severity"]
        max_psi = round(max(r.score for r in results["psi"].values()), 4)
    else:
        severity = max((d["severity"] for d in per_detector.values()), key=level_rank)
        max_psi = None

    agreeing = sorted(n for n, d in per_detector.items() if d["severity"] != "none")
    top = _top_features(results, reference, current, k)
    primary = per_detector.get("psi") or next(iter(per_detector.values()))
    names = ", ".join(t["feature"] for t in top)
    summary = (
        "No significant drift"
        if severity == "none"
        else f"{severity.upper()} drift: {primary['n_flagged']}/{primary['n_features']} "
        f"features flagged, {len(agreeing)}/{len(per_detector)} detectors agree"
        + (f"; top features: {names}" if names else "")
    )

    psi_results = results.get("psi") or next(iter(results.values()))
    return {
        "window": window,
        "severity": severity,
        "share_flagged": round(share_flagged(psi_results), 4),
        "max_psi": max_psi,
        "n_agreeing": len(agreeing),
        "agreeing_detectors": agreeing,
        "detectors": per_detector,
        "top_features": top,
        "feature_scores": {
            name: {
                f: {"score": round(float(r.score), 4), "flagged": bool(r.flagged)}
                for f, r in res.items()
            }
            for name, res in results.items()
        },
        "summary": summary,
    }


def explanation_hit(analysis: dict, true_features: list[str], k: int = 3) -> dict | None:
    """Do the top-k reported features match the features the simulator actually shifted?"""
    if not true_features:
        return None
    predicted = [t["feature"] for t in analysis["top_features"]][:k]
    hits = len(set(predicted) & set(true_features))
    return {
        "precision_at_k": round(hits / len(predicted), 3) if predicted else 0.0,
        "recall_at_k": round(hits / min(len(true_features), k), 3),
    }
