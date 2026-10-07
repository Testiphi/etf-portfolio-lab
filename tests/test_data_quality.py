"""质量展示不把库存存在、局部缺失或空库误报为完整有效。"""

import pandas as pd

from etf_lab.data import repo
from etf_lab.reports import quality, static_site


def test_quality_counts_null_and_nonfinite_without_claiming_calendar_coverage():
    con = repo.connect(":memory:")
    try:
        con.execute("INSERT INTO etf_price (symbol, date, close_adj) VALUES "
                    "('A', '2020-01-01', 1), ('A', '2020-01-03', NULL), "
                    "('B', '2020-01-03', 'NaN'::DOUBLE)")
        rows = repo.data_quality_summary(con)
        price = rows[0]
        assert price["rows"] == 3 and price["missing"] == 2
        assert price["missing_ratio"] == 2 / 3
        assert price["start"] == "2020-01-01" and price["end"] == "2020-01-03"
        assert rows[1]["missing_ratio"] is None
        html = quality.dataset_html(rows)
        assert "未保存" in html and "不能据此断言数据完整" in html
    finally:
        con.close()


def test_coverage_counts_union_loss_and_all_missing_asset():
    dates = pd.date_range("2020-01-01", periods=4)
    panel = pd.DataFrame({"A": [1, 2, 3, 4], "B": [None, 2, None, 4]}, index=dates)
    result = quality.coverage_summary(panel, panel.dropna().iloc[1:])
    assert result["observed_dates"] == 4 and result["used_dates"] == 1
    assert result["excluded_dates"] == 3
    assert result["assets"][1]["missing_ratio"] == 0.5
    assert result["assets"][1]["start"] == "2020-01-02"
    panel["B"] = None
    assert quality.coverage_summary(panel, panel.dropna())["assets"][1]["start"] is None


def test_about_uses_actual_override_and_escapes_labels():
    html = static_site.render_about(counts={}, data_version="test", results=[{
        "name": "<example>", "rf_annual": 0.03, "rf_source": "override",
        "start": "2020-01-01", "end": "2020-12-31",
    }])
    assert "3.00%" in html and "显式覆盖" in html and "&lt;example&gt;" in html
    assert "尚未接入真实期货" in html
    assert "风险敞口将改用" not in html
