"""Retraining flow: policy check, challenger training, validation, promotion, live verification.

Usage:
  python -m src.retrain.flow --once                     # one check, full JSON report
  python -m src.retrain.flow --loop --interval 15       # keep checking
  python -m src.retrain.flow --force                    # skip the policy, retrain on latest labels
"""

import argparse
import json
import logging
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import httpx
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from src.api.model_store import load_champion
from src.config import API_URL, CHALLENGER_ALIAS, FEATURES, MODEL_NAME, MONITOR_DIR, SEED
from src.data.loader import clean_frame, load_reference
from src.monitor.flow import flow
from src.retrain.policy import (
    PolicyConfig,
    PolicyState,
    advance,
    load_state,
    on_retrain_done,
    on_rollback,
    save_state,
)
from src.retrain.reference import set_reference
from src.retrain.rollback import PROMOTION_FILE, perform_rollback, promote
from src.retrain.validate import ValidationConfig, compare
from src.train.registry import configure, register_version
from src.train.train_register import train_and_log

log = logging.getLogger("retrain")


@dataclass(frozen=True)
class RetrainConfig:
    model: str = "gradient_boosting"
    min_rows: int = 4000  # labeled rows since drift onset needed before training
    max_rows: int = 6000
    eval_fraction: float = 0.3
    reference_rows: int = 5000  # old training rows mixed in (0 = recent data only)
    recent_weight: float = 5.0  # weight of each recent row relative to a reference row
    post_check_rows: int = 1000  # labeled rows served by the new champion before verifying it


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def make_fetch(api_url: str):
    def fetch(after_id: int = 0, limit: int = 6000) -> pd.DataFrame:
        with httpx.Client(base_url=api_url, timeout=60) as c:
            r = c.get("/log/labeled", params={"after_id": after_id, "limit": limit})
            r.raise_for_status()
        return pd.DataFrame(r.json()["records"])

    return fetch


def load_analyses(window_dir) -> list[dict]:
    files = sorted(Path(window_dir).glob("window_*.json"))
    return sorted((json.loads(p.read_text()) for p in files), key=lambda a: a["window"])


def append_history(monitor_dir, event: dict) -> None:
    path = Path(monitor_dir) / "retrain_history.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(event, default=str) + "\n")


def retrain_and_validate(
    rows: pd.DataFrame,
    api_url: str,
    monitor_dir,
    rcfg: RetrainConfig,
    vcfg: ValidationConfig,
    trigger: dict,
) -> dict:
    """Train a challenger on labeled rows, compare with the champion, promote or keep."""
    monitor_dir = Path(monitor_dir)
    if len(rows) < 200:
        return {"promoted": False, "skipped": True, "reason": f"only {len(rows)} labeled rows"}
    X = clean_frame(rows[FEATURES]).reset_index(drop=True)
    y = rows["label"].astype(int).reset_index(drop=True)
    if y.nunique() < 2 or y.value_counts().min() < 2:
        return {"promoted": False, "skipped": True, "reason": "labels contain a single class"}

    X_train, X_eval, y_train, y_eval = train_test_split(
        X, y, test_size=rcfg.eval_fraction, stratify=y, random_state=SEED
    )

    client = configure()
    champion = load_champion()

    X_fit, y_fit, weights = X_train, y_train, None
    if rcfg.reference_rows > 0:
        X_ref, y_ref = load_reference()
        ref_idx = X_ref.sample(min(rcfg.reference_rows, len(X_ref)), random_state=SEED).index
        X_fit = pd.concat([X_ref.loc[ref_idx], X_train], ignore_index=True)
        y_fit = pd.concat([y_ref.loc[ref_idx], y_train], ignore_index=True)
        weights = np.concatenate([np.ones(len(ref_idx)), np.full(len(X_train), rcfg.recent_weight)])

    run_id, train_metrics = train_and_log(
        rcfg.model,
        X_fit,
        y_fit,
        X_eval,
        y_eval,
        run_name=f"retrain-{_now()}",
        tags={
            "stage": "retrain",
            "train_data": "recent_labeled+reference_sample",
            "eval_data": "recent_labeled_holdout",
            "trigger": str(trigger.get("reason")),
        },
        sample_weight=weights,
    )

    mv = register_version(client, run_id, alias=CHALLENGER_ALIAS)
    challenger = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/{mv.version}")
    result = compare(champion.model, challenger, X_eval, y_eval, vcfg)

    verdict = "promoted" if result.promote else "rejected"
    client.log_metric(run_id, "validation_champion_f1", result.champion_f1)
    client.log_metric(run_id, "validation_challenger_f1", result.challenger_f1)
    client.log_metric(run_id, "validation_f1_gain", result.f1_gain)
    client.set_tag(run_id, "validation", verdict)
    client.set_model_version_tag(MODEL_NAME, mv.version, "validation", verdict)

    out = {
        "promoted": result.promote,
        "champion_version": champion.version,
        "challenger_version": str(mv.version),
        "n_train": len(X_fit),
        "n_train_recent": len(X_train),
        "n_eval": len(X_eval),
        "validation": result.to_dict(),
        "train_metrics": train_metrics,
        "trigger": trigger,
    }

    if result.promote:
        old_version = promote(client, mv.version)
        try:
            r = httpx.post(f"{api_url}/reload", timeout=60)
            r.raise_for_status()
            out["reload"] = r.json()
        except httpx.HTTPError as exc:
            out["reload_error"] = str(exc)
        set_reference(X_train, monitor_dir)
        record = {
            "status": "monitoring",
            "version": str(mv.version),
            "previous_version": old_version,
            "champion_f1": result.champion_f1,
            "challenger_f1": result.challenger_f1,
            "promoted_at": _now(),
            "trigger": trigger,
        }
        (monitor_dir / PROMOTION_FILE).write_text(json.dumps(record, indent=2, default=str))

    append_history(monitor_dir, {"ts": _now(), "event": verdict, **out})
    return out


def run_check(monitor_dir, pcfg: PolicyConfig, rcfg: RetrainConfig, fetch, retrain) -> dict:
    """Read new monitor windows, apply the policy, retrain when a trigger has enough data."""
    monitor_dir = Path(monitor_dir)
    state_path = monitor_dir / "policy_state.json"
    state = load_state(state_path)

    analyses = load_analyses(monitor_dir / "windows")
    if analyses and analyses[-1]["window"] < state.last_window:
        log.warning("Window numbering restarted (monitor was reset); resetting policy state")
        state = PolicyState()
    new = [a for a in analyses if a["window"] > state.last_window]
    decisions = advance(state, new, pcfg)

    report: dict = {
        "new_windows": len(new),
        "decisions": [asdict(d) for d in decisions][-10:],
        "pending": state.pending,
    }
    if state.pending:
        rows = fetch(after_id=(state.drift_first_id or 1) - 1, limit=rcfg.max_rows)
        if len(rows) >= rcfg.min_rows:
            trigger = {
                "window": state.trigger_window,
                "reason": state.trigger_reason,
                "drift_first_id": state.drift_first_id,
            }
            report["retrain"] = retrain(rows, trigger)
            on_retrain_done(state, pcfg)
            report["pending"] = False
        else:
            report["waiting"] = (
                f"{len(rows)}/{rcfg.min_rows} labeled rows since drift onset "
                f"(trigger: window {state.trigger_window})"
            )
    save_state(state_path, state)
    return report


def post_promotion_check(monitor_dir, fetch, do_rollback, min_rows: int = 1000) -> dict | None:
    """Verify a fresh champion on live labeled traffic; roll back if worse than the old one."""
    path = Path(monitor_dir) / PROMOTION_FILE
    if not path.exists():
        return None
    rec = json.loads(path.read_text())
    if rec.get("status") != "monitoring":
        return None

    rows = fetch(after_id=0, limit=8000)
    if not rows.empty:
        rows = rows[rows["model_version"] == rec["version"]]
    if len(rows) < min_rows:
        return {"status": "waiting", "rows": len(rows), "need": min_rows}

    live_f1 = float(
        f1_score(rows["label"].astype(int), rows["prediction"].astype(int), zero_division=0)
    )
    baseline = float(rec["champion_f1"])
    if live_f1 < baseline:
        reason = f"live F1 {live_f1:.3f} is below the old champion's {baseline:.3f}"
        do_rollback(reason=reason)
        status = "rolled_back"
    else:
        reason = f"live F1 {live_f1:.3f} is at least the old champion's {baseline:.3f}"
        status = "confirmed"
    rec.update(status=status, live_f1=round(live_f1, 4), live_rows=len(rows), verdict=reason)
    path.write_text(json.dumps(rec, indent=2, default=str))
    return {"status": status, "live_f1": round(live_f1, 4), "rows": len(rows), "reason": reason}


@flow(name="retrain-check", log_prints=True)
def retrain_check_flow(
    api_url: str = API_URL, monitor_dir: str = str(MONITOR_DIR), force: bool = False
) -> dict:
    mdir = Path(monitor_dir)
    pcfg, rcfg, vcfg = PolicyConfig(), RetrainConfig(), ValidationConfig()
    fetch = make_fetch(api_url)

    def retrain(rows, trigger):
        return retrain_and_validate(rows, api_url, mdir, rcfg, vcfg, trigger)

    def do_rollback(reason):
        return perform_rollback(api_url, mdir, reason=reason)

    if force:
        rows = fetch(after_id=0, limit=rcfg.max_rows)
        report = {"forced": True, "retrain": retrain(rows, {"window": None, "reason": "forced"})}
    else:
        report = run_check(mdir, pcfg, rcfg, fetch, retrain)

    post = post_promotion_check(mdir, fetch, do_rollback, rcfg.post_check_rows)
    if post and post.get("status") == "rolled_back":
        state_path = mdir / "policy_state.json"
        state = load_state(state_path)
        on_rollback(state, pcfg)
        save_state(state_path, state)
    report["post_check"] = post
    return report


def one_line(report: dict) -> str:
    parts = [f"{report.get('new_windows', 0)} new window(s)"]
    if report.get("decisions"):
        d = report["decisions"][-1]
        parts.append(f"w{d['window']}: {d['reason']}")
    if report.get("waiting"):
        parts.append(report["waiting"])
    r = report.get("retrain")
    if r:
        v = r.get("validation")
        parts.append(
            "RETRAIN "
            + (
                f"v{r['challenger_version']} {'PROMOTED' if r['promoted'] else 'rejected'}: "
                f"{v['reason']}"
                if v
                else r.get("reason", "skipped")
            )
        )
    p = report.get("post_check")
    if p:
        parts.append(f"post-check {p['status']}" + (f" ({p['reason']})" if "reason" in p else ""))
    return " | ".join(parts)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-url", default=API_URL)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--interval", type=float, default=15)
    args = parser.parse_args()

    while True:
        try:
            report = retrain_check_flow(
                api_url=args.api_url, monitor_dir=str(MONITOR_DIR), force=args.force
            )
            if args.loop and not args.force:
                print(one_line(report), flush=True)
                if report.get("retrain"):
                    print(json.dumps(report["retrain"], indent=2, default=str), flush=True)
            else:
                print(json.dumps(report, indent=2, default=str))
        except Exception:
            log.exception("Check failed; will retry" if args.loop else "Check failed")
            if not args.loop:
                raise
        if not args.loop:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
