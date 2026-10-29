from src.drift.base import FeatureScore
from src.drift.severity import SeverityConfig, level_rank, window_severity


def results(n_total: int, n_flagged: int, flagged_score=0.15, other_score=0.01):
    return {
        f"f{i}": FeatureScore(flagged_score if i < n_flagged else other_score, i < n_flagged)
        for i in range(n_total)
    }


def test_nothing_flagged_is_none():
    assert window_severity(results(13, 0), use_psi_rule=True) == "none"


def test_single_psi_feature_above_threshold_is_medium():
    assert window_severity(results(13, 1), use_psi_rule=True) == "medium"


def test_single_huge_psi_is_high():
    assert window_severity(results(13, 1, flagged_score=1.5), use_psi_rule=True) == "high"


def test_many_flagged_features_is_high():
    assert window_severity(results(13, 6), use_psi_rule=True) == "high"  # 6/13 = 0.46


def test_share_rule_for_ks_without_psi_rule():
    assert window_severity(results(5, 0)) == "none"
    assert window_severity(results(5, 1)) == "medium"  # 0.2 >= 0.15
    assert window_severity(results(5, 2)) == "high"  # 0.4 >= 0.40
    # a large score alone must not escalate a non-PSI detector
    assert window_severity(results(5, 0, other_score=5.0)) == "none"


def test_empty_results_is_none():
    assert window_severity({}) == "none"


def test_custom_config_and_ranking():
    strict = SeverityConfig(medium_share=0.5, high_share=0.9)
    assert window_severity(results(10, 4), strict) == "none"
    assert level_rank("none") < level_rank("medium") < level_rank("high")
