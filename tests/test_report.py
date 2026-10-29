import pytest

pytest.importorskip("evidently")

from src.drift.report import save_html_report  # noqa: E402
from tests.test_analysis import make_frame  # noqa: E402


def test_html_report_is_written(tmp_path):
    ref = make_frame(3000, seed=0)
    cur = make_frame(500, seed=1, age_shift=2.0)
    out = save_html_report(ref, cur, tmp_path / "r.html")
    assert out.exists() and out.stat().st_size > 10_000
