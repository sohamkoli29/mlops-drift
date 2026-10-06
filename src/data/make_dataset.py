"""Download UCI Adult Census Income (OpenML) and save a cleaned raw CSV."""

from sklearn.datasets import fetch_openml

from src.config import RAW_DIR, RAW_FILE, TARGET


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    bunch = fetch_openml("adult", version=2, as_frame=True, parser="auto")
    df = bunch.frame.copy()

    # snake_case column names, nicer for Pydantic and code
    df.columns = [c.replace("-", "_") for c in df.columns]
    df = df.rename(columns={"class": TARGET})
    df[TARGET] = (df[TARGET].astype(str).str.strip() == ">50K").astype(int)

    # categoricals -> plain strings (NaN stays NaN; imputed later in the pipeline)
    for col in df.select_dtypes(include=["category", "object"]).columns:
        df[col] = df[col].astype(object).str.strip()

    df.to_csv(RAW_FILE, index=False)
    print(f"Saved {RAW_FILE}  shape={df.shape}")
    print(df.head())


if __name__ == "__main__":
    main()
