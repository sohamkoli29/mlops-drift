"""Plot max score per window for each detector and scenario, with drift onset marked."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from src.config import REPORTS_DIR  # noqa: E402

OFFLINE = REPORTS_DIR / "offline"


def main(seed: int = 42) -> None:
    df = pd.read_csv(OFFLINE / f"windows_seed{seed}.csv")
    scenarios = ["none", "sudden", "gradual"]
    detectors = ["psi", "ks", "wasserstein"]
    fig, axes = plt.subplots(len(detectors), len(scenarios), figsize=(15, 9), sharex=True)

    for r, det in enumerate(detectors):
        for c, scen in enumerate(scenarios):
            ax = axes[r][c]
            g = df[(df.scenario == scen) & (df.detector == det)]
            ax.plot(g.window, g.max_score, marker="o", ms=3)
            onset = g[g.drift_active].window.min()
            if pd.notna(onset):
                ax.axvline(onset, color="red", ls="--", lw=1)
            ax.set_yscale("log")
            if r == 0:
                ax.set_title(scen)
            if c == 0:
                ax.set_ylabel(f"{det}: max score")
            if r == len(detectors) - 1:
                ax.set_xlabel("window")
    plt.tight_layout()
    out = OFFLINE / "offline_scores.png"
    plt.savefig(out, dpi=120)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
