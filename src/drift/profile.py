"""Reference profile: the numbers the drift detectors compare against.

Numeric features: 10 quantile bins built from the reference (unique inner edges; the outer
bins are implicitly -inf and +inf). A value x falls in bin i if edge[i-1] < x <= edge[i].
Categorical features: share of each reference category, plus an __OTHER__ bin for unseen
categories. Missing values are their own category (__MISSING__).
"""

import json

import numpy as np
import pandas as pd

from src.config import CATEGORICAL_FEATURES, NUMERIC_FEATURES, REPORTS_DIR

PROFILE_FILE = REPORTS_DIR / "reference_profile.json"
N_BINS = 10
MISSING = "__MISSING__"
OTHER = "__OTHER__"


def assign_bins(values, edges) -> np.ndarray:
    """Bin index for each value given the inner edges (right-inclusive intervals)."""
    return np.searchsorted(
        np.asarray(edges, dtype=float), np.asarray(values, dtype=float), side="left"
    )


def numeric_bin_shares(values, edges) -> np.ndarray:
    idx = assign_bins(values, edges)
    counts = np.bincount(idx, minlength=len(edges) + 1)
    return counts / counts.sum()


def category_shares(series: pd.Series, categories: list[str]) -> dict[str, float]:
    s = series.astype(object).where(series.notna(), MISSING)
    s = s.where(s.isin(categories), OTHER)
    counts = s.value_counts()
    total = counts.sum()
    out = {c: float(counts.get(c, 0)) / total for c in categories}
    out[OTHER] = float(counts.get(OTHER, 0)) / total
    return out


def build_profile(X: pd.DataFrame, n_bins: int = N_BINS) -> dict:
    profile: dict = {
        "n_rows": int(len(X)),
        "n_bins": n_bins,
        "numeric": {},
        "categorical": {},
    }

    quantiles = np.linspace(0, 1, n_bins + 1)[1:-1]
    for col in NUMERIC_FEATURES:
        x = X[col].dropna().astype(float).to_numpy()
        edges = np.unique(np.quantile(x, quantiles))
        profile["numeric"][col] = {
            "mean": float(x.mean()),
            "std": float(x.std(ddof=1)),
            "min": float(x.min()),
            "max": float(x.max()),
            "edges": edges.tolist(),
            "shares": numeric_bin_shares(x, edges).tolist(),
        }

    for col in CATEGORICAL_FEATURES:
        filled = X[col].astype(object).where(X[col].notna(), MISSING)
        categories = sorted(filled.unique())
        profile["categorical"][col] = {
            "categories": categories,
            "shares": category_shares(X[col], categories),
        }
    return profile


def save_profile(profile: dict, path=PROFILE_FILE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(profile, f, indent=2)


def load_profile(path=PROFILE_FILE) -> dict:
    with open(path) as f:
        return json.load(f)


def main() -> None:
    from src.data.loader import load_reference

    X_ref, _ = load_reference()
    profile = build_profile(X_ref)
    save_profile(profile)
    print(f"Saved {PROFILE_FILE}")
    for col, p in profile["numeric"].items():
        print(f"  {col:15s} bins={len(p['shares']):2d}  mean={p['mean']:.2f}  std={p['std']:.2f}")


if __name__ == "__main__":
    main()
