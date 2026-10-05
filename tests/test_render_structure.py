"""渲染结构的回归测试。

这里守住的是**课堂上不会讲、但会让页面直接白掉**的一类错误：
首屏把多个组合的仪表盘内联到同一页，任何重复的 DOM id 都会让
``document.getElementById`` 只命中第一个元素，于是后面几档的图全部画不出来。
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from etf_lab.data import repo
from etf_lab.presets import PortfolioSpec
from etf_lab.reports import static_site, theme

FIGURE_SUFFIXES = (
    "fig-nav",
    "fig-attrib",
    "fig-risk",
    "fig-dca",
    "fig-roll",
    "fig-asset",
    "fig-expo",
    "fig-vol",
    "fig-prot",
    "fig-epi",
)


def _seed_db(path: Path) -> None:
    """造一个最小的可计算数据仓（两只标的、约三年半日线）。"""
    con = repo.connect(path)
    rng = np.random.default_rng(7)
    dates = pd.date_range("2018-01-01", periods=900, freq="B")

    for symbol, drift, vol in (("AAA", 0.0003, 0.010), ("BBB", 0.0001, 0.003)):
        prices = 1.0 * np.exp(np.cumsum(rng.normal(drift, vol, len(dates))))
        frame = pd.DataFrame(
            {
                "symbol": symbol,
                "date": dates,
                "open": prices,
                "high": prices * 1.01,
                "low": prices * 0.99,
                "close": prices,
                "volume": 1_000_000.0,
                "amount": 10_000_000.0,
                "adj_factor": 1.0,
                "close_adj": prices,
            }
        )
        repo.upsert(
            con,
            "etf_price",
            frame,
            ["symbol", "date", "open", "high", "low", "close", "volume", "amount", "adj_factor", "close_adj"],
        )

    meta = pd.DataFrame(
        [
            {
                "symbol": "AAA",
                "name": "宽基A",
                "exchange": "SH",
                "asset_class": "broad",
                "underlying_index": "000300",
                "list_date": dates[0].date(),
                "mgmt_fee": None,
                "custodian_fee": None,
                "currency": "CNY",
                "is_cross_border": False,
                "t_plus": 1,
                "price_limit": None,
                "notes": None,
            },
            {
                "symbol": "BBB",
                "name": "债券B",
                "exchange": "SH",
                "asset_class": "bond",
                "underlying_index": None,
                "list_date": dates[0].date(),
                "mgmt_fee": None,
                "custodian_fee": None,
                "currency": "CNY",
                "is_cross_border": False,
                "t_plus": 0,
                "price_limit": None,
                "notes": None,
            },
        ]
    )
    repo.upsert(
        con,
        "etf_meta",
        meta,
        [
            "symbol",
            "name",
            "exchange",
            "asset_class",
            "underlying_index",
            "list_date",
            "mgmt_fee",
            "custodian_fee",
            "currency",
            "is_cross_border",
            "t_plus",
            "price_limit",
            "notes",
        ],
    )
    repo.log_data_version(con, source="synthetic", notes="测试数据")
    con.close()


def _spec(key: str) -> PortfolioSpec:
    return PortfolioSpec(
        key=key,
        name=f"组合{key}",
        question="测试用组合",
        weights={"AAA": 0.6, "BBB": 0.4},
        dca={"amount": 1000.0, "freq": "monthly", "mode": "fixed", "day": None},
    )


def test_compute_preset_runs_end_to_end_on_synthetic_data(tmp_path: Path) -> None:
    db = tmp_path / "lab.duckdb"
    _seed_db(db)
    con = repo.connect(db)
    result = static_site.compute_preset(con, _spec("one"))
    con.close()

    assert result["n_obs"] > 500
    assert result["metrics"]["annualized_return"] is not None
    assert set(result["risk_contribution"]["component_var_share"]) == {"AAA", "BBB"}
    # 对数贡献与组合实际对数收益之差即再平衡效应，Jensen 不等式保证非负
    risk = result["risk_contribution"]
    assert set(risk["log_contribution"]) == {"AAA", "BBB"}
    assert risk["rebalancing_effect"] >= 0
    assert risk["log_total_return"] == pytest.approx(risk["log_contribution_sum"] + risk["rebalancing_effect"], abs=1e-6)
    # 没有净值数据时折溢价应为空字典，而不是抛错或填 0
    assert result["premium_discount"] == {}
    # 组合结构识别正确 → 债券类模块应当被触发
    assert result["composition"]["has_bond"] is True


def test_gear_panes_have_unique_figure_ids(tmp_path: Path) -> None:
    """三个组合内联到一页时，图表容器 id 必须全局唯一。

    这是实际发生过的故障：id 重复 → 浏览器只命中第一个 → 后两档图表全空白。
    """
    db = tmp_path / "lab.duckdb"
    _seed_db(db)
    con = repo.connect(db)
    results = [static_site.compute_preset(con, _spec(key)) for key in ("one", "two", "three")]
    counts = repo.table_counts(con)
    version = repo.latest_data_version(con)
    con.close()

    dashboards: dict[str, str] = {}
    figs_by_key: dict[str, dict] = {}
    for result in results:
        key = str(result["key"])
        figs: dict = {}
        dashboards[key] = static_site.render_dashboard(result, prefix=f"{key}-", figs=figs)
        figs_by_key[key] = figs

    html = static_site.render_index(results, dashboards=dashboards, counts=counts, data_version=version)

    ids = re.findall(r'id="([^"]+)"', html)
    duplicates = sorted({value for value in ids if ids.count(value) > 1})
    assert not duplicates, f"存在重复 DOM id：{duplicates}"

    # 每一档都要各自拥有全套图表容器
    for key in ("one", "two", "three"):
        for suffix in FIGURE_SUFFIXES:
            assert f'id="{key}-{suffix}"' in html, f"缺少 {key}-{suffix}"

    # 图表数据必须全部外置到数据文件里（页面本身不再内联 Plotly 数据）
    assert "Plotly.newPlot(" not in html, "页面里不应再内联绘图脚本"
    all_fig_ids = {fid for figs in figs_by_key.values() for fid in figs}
    assert all_fig_ids == {f"{key}-{suffix}" for key in ("one", "two", "three") for suffix in FIGURE_SUFFIXES}
    data_js = theme.figure_data_js(figs_by_key["one"])
    assert '"one-fig-nav"' in data_js


def test_curves_are_downsampled_but_keep_extremes() -> None:
    """抽稀必须保留极值点：等距抽稀有可能恰好丢掉最深的那一天。"""
    index = pd.date_range("2020-01-01", periods=1000, freq="B")
    values = np.linspace(1.0, 2.0, 1000)
    values[777] = 0.5  # 一个孤立的最深点
    series = pd.Series(values, index=index)

    pairs = static_site._series_to_pairs(series, step=10, keep_extremes=True)
    dates = [row[0] for row in pairs]
    assert len(pairs) < 200, "应当被抽稀"
    assert str(index[777].date()) in dates, "极值点不能被抽稀丢掉"
    assert str(index[0].date()) in dates and str(index[-1].date()) in dates

    without = [row[0] for row in static_site._series_to_pairs(series, step=10, keep_extremes=False)]
    assert str(index[777].date()) not in without, "对照组：不保留极值时确实会丢"


def test_standalone_page_ids_are_unique(tmp_path: Path) -> None:
    db = tmp_path / "lab.duckdb"
    _seed_db(db)
    con = repo.connect(db)
    result = static_site.compute_preset(con, _spec("solo"))
    con.close()

    figs: dict = {}
    dashboard = static_site.render_dashboard(result, prefix=f"{result['key']}-", figs=figs)
    html = static_site.render_preset_page(result, dashboard=dashboard)

    ids = re.findall(r'id="([^"]+)"', html)
    assert len(ids) == len(set(ids))
    assert 'id="solo-fig-nav"' in html
    # 独立页要自己装载数据文件
    assert 'data/solo.figs.js' in html
    assert 'assets/lab.js' in html
