"""Quick EDA: summary stats, missing values, class balance, plots."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from src.config import (  # noqa: E402
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    RAW_FILE,
    REPORTS_DIR,
    TARGET,
)

OUT = REPORTS_DIR / "eda"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(RAW_FILE)

    print("Shape:", df.shape)
    print("\nDtypes:\n", df.dtypes)
    print("\nMissing values:\n", df.isna().sum()[df.isna().sum() > 0])
    print("\nTarget balance:\n", df[TARGET].value_counts(normalize=True).round(3))
    print("\nNumeric summary:\n", df[NUMERIC_FEATURES].describe().round(2))
    print("\nCategorical cardinality:\n", df[CATEGORICAL_FEATURES].nunique())

    # Numeric distributions
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    for ax, col in zip(axes.ravel(), NUMERIC_FEATURES, strict=False):
        sns.histplot(df[col], bins=40, ax=ax)
        ax.set_title(col)
    axes.ravel()[-1].axis("off")
    plt.tight_layout()
    plt.savefig(OUT / "numeric_distributions.png", dpi=120)
    plt.close()

    # Categorical counts
    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    for ax, col in zip(axes.ravel(), CATEGORICAL_FEATURES, strict=False):
        df[col].value_counts().head(8).plot(kind="barh", ax=ax)
        ax.set_title(col)
        ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(OUT / "categorical_counts.png", dpi=120)
    plt.close()

    # Target balance
    df[TARGET].value_counts().plot(kind="bar", title="Target (0: <=50K, 1: >50K)")
    plt.tight_layout()
    plt.savefig(OUT / "target_balance.png", dpi=120)
    plt.close()

    print(f"\nPlots saved to {OUT}")


if __name__ == "__main__":
    main()
