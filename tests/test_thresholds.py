import json

from src.drift.detectors import DEFAULT_THRESHOLDS, get_detectors, load_thresholds
from src.drift.wasserstein import WassersteinDetector
from tests.test_analysis import make_frame


def test_missing_file_gives_defaults(tmp_path):
    assert load_thresholds(tmp_path / "nope.json") == DEFAULT_THRESHOLDS


def test_file_overrides_only_what_it_sets(tmp_path):
    p = tmp_path / "t.json"
    p.write_text(json.dumps({"ks": {"alpha": 0.01}, "wasserstein": {"per_feature": {"age": 0.2}}}))
    t = load_thresholds(p)
    assert t["ks"]["alpha"] == 0.01
    assert t["psi"] == DEFAULT_THRESHOLDS["psi"]
    assert t["wasserstein"]["default"] == 0.10
    dets = get_detectors(t)
    assert dets["ks"].alpha == 0.01
    assert dets["wasserstein"].per_feature == {"age": 0.2}


def test_per_feature_threshold_controls_the_flag():
    ref, cur = make_frame(20_000, 0), make_frame(1000, 1, age_shift=0.5)
    assert WassersteinDetector().score(ref, cur)["age"].flagged
    assert not WassersteinDetector(0.1, {"age": 1.0}).score(ref, cur)["age"].flagged
