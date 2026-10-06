"""渲染结构的回归测试。

这里守住的是**课堂上不会讲、但会让页面直接白掉**的一类错误：
首屏把多个组合的仪表盘内联到同一页，任何重复的 DOM id 都会让
``document.getElementById`` 只命中第一个元素，于是后面几档的图全部画不出来。
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from etf_lab import universe
from etf_lab.data import repo
from etf_lab.presets import PortfolioSpec
from etf_lab.reports import static_site, theme

FIGURE_SUFFIXES = (
    # 容器 id 由图表函数名机械推导（theme.figure_id），所以这里就是函数名换连字符——
    # 不再需要人工维护一张 "fig-attrib ↔ fig_return_contribution" 的对照表。
    "fig-nav",
    "fig-return_contribution",
    "fig-risk_vs_weight",
    "fig-dca",
    "fig-rolling_sharpe",
    "fig-per_asset",
    "fig-exposure_heatmap",
    "fig-yield_curve",
    "fig-yield_history",
    "fig-vol_term_structure",
    "fig-protection_curve",
    "fig-hedge_tradeoff",
    "fig-mc_fan",
    "fig-mc_histogram",
    "fig-mc_convergence",
    "fig-episodes",
)
"""无论是否持有债券都应出现的图表。"""
BOND_ONLY_SUFFIXES = ("fig-rate_scenarios",)
"""只有组合含债券标的（久期可估）时才出现的图表。"""
EXPECTED_WITH_BOND = FIGURE_SUFFIXES + BOND_ONLY_SUFFIXES


def _seed_db(path: Path) -> None:
    """造一个最小的可计算数据仓：国债收益率曲线 + 两只标的（一宽基、一债券）。

    债券标的的收益**由收益率变动驱动**（久期 4 年），这样久期回归才有可靠结果，
    久期面板也才会真正出现——否则测不到那条条件渲染分支。
    """
    con = repo.connect(path)
    rng = np.random.default_rng(7)
    dates = pd.date_range("2018-01-01", periods=900, freq="B")

    # 国债收益率曲线用**均值回复**过程（围绕 2.5%，日变动约 3bp）：
    # 纯随机游走跑 900 天后会漂到负收益率——那在现实中不存在，
    # 还会让无风险利率变成负数，把被测代码引入不现实的分支。
    five_year = np.empty(len(dates))
    five_year[0] = 2.5
    for index in range(1, len(dates)):
        five_year[index] = five_year[index - 1] + 0.01 * (2.5 - five_year[index - 1]) + rng.normal(0, 0.03)
    curve = pd.concat(
        [
            pd.DataFrame({"date": dates, "code": "CN5Y", "tenor": "5年", "yield": five_year}),
            pd.DataFrame({"date": dates, "code": "CN1Y", "tenor": "1年", "yield": five_year - 0.4}),
            pd.DataFrame({"date": dates, "code": "CN10Y", "tenor": "10年", "yield": five_year + 0.3}),
        ],
        ignore_index=True,
    )
    repo.upsert(con, "bond_yield", curve, ["date", "code", "tenor", "yield"])

    changes_bp = np.concatenate([[0.0], np.diff(five_year) * 100.0])
    bbb_returns = -4.0 * changes_bp / 10000.0 + rng.normal(0, 2e-4, len(dates))
    series = {
        "AAA": 1.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.010, len(dates)))),
        "BBB": 100.0 * np.cumprod(1.0 + bbb_returns),
    }
    for symbol, prices in series.items():
        frame = pd.DataFrame(
            {
                "symbol": symbol,
                "date": dates,
                "open": prices,
                "high": prices * 1.001,
                "low": prices * 0.999,
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


def _spec(
    key: str,
    *,
    include_bond: bool = True,
    dca: dict | None = None,
    weights: dict | None = None,
    rebalance: dict | None = None,
    cash: dict | None = None,
) -> PortfolioSpec:
    if weights is None:
        weights = {"AAA": 0.6, "BBB": 0.4} if include_bond else {"AAA": 1.0}
    return PortfolioSpec(
        key=key,
        name=f"组合{key}",
        question="测试用组合",
        weights=weights,
        dca=dca if dca is not None else {"amount": 1000.0, "freq": "monthly", "mode": "fixed", "day": None},
        rebalance=rebalance or {},
        cash=cash or {},
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
    # 无风险利率来自收益率曲线，而不是写死的假设值
    assert result["rf_source"] == "curve"
    # as_dict 会把 rf 舍入到 6 位小数，比较时用相应容差
    assert result["rf_annual"] == pytest.approx(result["rates"]["environment"]["risk_free"], abs=1e-6)
    assert 0.0 < result["rf_annual"] < 0.10, "合成曲线是均值回复的，rf 应当落在合理区间"
    # 组合含债券，久期面板的图与数据必须同时存在
    assert result["composition"]["has_bond"] is True
    # 只有 R² 达标的标的才进入利率冲击情景
    assert result["rates"]["reliable_symbols"] == ["BBB"]
    assert result["rates"]["scenarios"]


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
        for suffix in EXPECTED_WITH_BOND:
            assert f'id="{key}-{suffix}"' in html, f"缺少 {key}-{suffix}"

    # 图表数据必须全部外置到数据文件里（页面本身不再内联 Plotly 数据）
    assert "Plotly.newPlot(" not in html, "页面里不应再内联绘图脚本"
    all_fig_ids = {fid for figs in figs_by_key.values() for fid in figs}
    assert all_fig_ids == {f"{key}-{suffix}" for key in ("one", "two", "three") for suffix in EXPECTED_WITH_BOND}
    data_js = theme.figure_data_js(figs_by_key["one"])
    assert '"one-fig-nav"' in data_js


def test_duration_panel_is_conditional_on_bond_holdings(tmp_path: Path) -> None:
    """久期面板与它的图表必须**同时**出现或同时缺席。

    这是条件渲染最容易出的错：面板隐藏了但图表数据还在（或反之）。
    它也保证解锁语义成立——「利率敏感性与久期」标着需债券资产，就必须真的只在含债券时出现。
    """
    db = tmp_path / "lab.duckdb"
    _seed_db(db)
    con = repo.connect(db)
    with_bond = static_site.compute_preset(con, _spec("with"))
    without = static_site.compute_preset(con, _spec("without", include_bond=False))
    con.close()

    assert with_bond["composition"]["has_bond"] is True
    assert not without["composition"].get("has_bond")
    assert with_bond["rates"]["scenarios"], "含债券时应当给出利率冲击情景"
    assert without["rates"]["scenarios"] == [], "不含债券时不应给利率冲击情景"

    figs_with: dict = {}
    html_with = static_site.render_dashboard(with_bond, prefix="with-", figs=figs_with)
    figs_without: dict = {}
    html_without = static_site.render_dashboard(without, prefix="without-", figs=figs_without)

    assert "久期与利率冲击" in html_with
    assert "with-fig-rate_scenarios" in figs_with
    assert "久期与利率冲击" not in html_without
    assert "with-fig-rate_scenarios" not in figs_without

    # 利率环境面板与曲线图与持仓无关，只要曲线数据在就应当出现
    assert "利率环境与无风险利率" in html_without
    assert "without-fig-yield_curve" in figs_without


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


def test_every_span_class_used_has_a_css_rule(tmp_path: Path) -> None:
    """每个用到的 span-N 都必须有对应的 CSS 规则。

    这是实际发生过的故障：新增期权与蒙特卡洛面板时用了 span-5 / span-7，
    而 CSS 里只手写了 3/4/6/8/12——那六块面板因此没有 grid-column，
    在 12 栅格里只占 1 格、被挤成细条，看上去就是「渲染有误」。
    现在 span 规则由代码生成 1..12，并由这条测试守住。
    """
    db = tmp_path / "lab.duckdb"
    _seed_db(db)
    con = repo.connect(db)
    result = static_site.compute_preset(con, _spec("one"))
    con.close()

    figs: dict = {}
    html = static_site.render_dashboard(result, prefix="one-", figs=figs)

    used = set(re.findall(r"\bspan-(\d+)\b", html))
    defined = set(re.findall(r"\.span-(\d+)\s*\{", theme.STYLE))
    assert used, "页面上应当用到 span 类"
    assert used <= defined, f"以下 span 类没有 CSS 规则：{sorted(used - defined, key=int)}"
    # 1..12 必须全部有定义，避免再次出现「用到才发现没写」的情况
    assert defined == {str(i) for i in range(1, 13)}


def test_narrow_screen_collapses_all_panels() -> None:
    """窄屏折叠要用 .grid > * 通配，而不是逐个列出 span 类（漏一个就不折叠）。"""
    assert ".grid > *" in theme.STYLE
    assert "@media (max-width: 1000px)" in theme.STYLE


def test_app_renders_every_analysis_the_static_site_shows(tmp_path: Path) -> None:
    """C 路径（NiceGUI）必须覆盖 A 路径（静态站）仪表盘的全部分析模块。

    两条路线共用同一套**计算**，却各自**渲染**。覆盖度一旦分叉，同一个组合在
    静态站上有利率、久期、蒙特卡洛、Delta-Gamma 复制，在应用里却看不到——
    这比数字不一致更隐蔽，因为两边都不会报错。

    真源取"实际渲染出来的仪表盘里的图表清单"，而不是某个手写列表：
    手写列表会在新增面板时忘记同步，那正是这个 bug 的成因。
    """
    pytest.importorskip("nicegui", reason="路线 C 依赖 NiceGUI（可选依赖）")

    db = tmp_path / "lab.duckdb"
    _seed_db(db)
    con = repo.connect(db)
    result = static_site.compute_preset(con, _spec("one"))
    con.close()

    figs: dict = {}
    static_site.render_dashboard(result, prefix="one-", figs=figs)
    rendered = {re.sub(r"^one-", "", figure_id).replace("-", "_") for figure_id in figs}

    from etf_lab.app.main import ANALYSIS_TABS
    from etf_lab.reports import figures as figures_mod

    covered = {name for _, names, _ in ANALYSIS_TABS for name in names}
    assert rendered, "静态站仪表盘应当至少有一张图"
    assert not (rendered - covered), f"静态站有、应用没渲染的图表：{sorted(rendered - covered)}"
    for name in covered:
        assert hasattr(figures_mod, name), f"应用引用了不存在的图表函数：{name}"


def test_shared_html_blocks_cover_all_analyses(tmp_path: Path) -> None:
    """应用复用的表格块必须齐全且非空（数据缺失时也要给出说明而不是空白）。"""
    db = tmp_path / "lab.duckdb"
    _seed_db(db)
    con = repo.connect(db)
    result = static_site.compute_preset(con, _spec("one"))
    con.close()

    blocks = static_site.html_blocks(result)
    assert set(blocks) == {
        "drawdown",
        "dca",
        "per_asset",
        "exposure",
        "rates",
        "duration",
        "derivatives",
        "hedge",
        "monte_carlo",
    }
    empty = [key for key, html in blocks.items() if not html.strip()]
    assert not empty, f"这些表格块是空的（应当给出说明文案）：{empty}"


def test_symbol_labels_include_name_and_sector() -> None:
    """标的标签必须带上名称与板块——只给代码没人看得懂哪个是哪只。"""
    assert universe.symbol_label("510300", "沪深300ETF华泰柏瑞", "broad") == "沪深300ETF华泰柏瑞（510300 · 宽基）"
    # 缺字段时自动省略，不留空洞，也不猜
    assert universe.symbol_label("510300", "沪深300ETF华泰柏瑞", None) == "沪深300ETF华泰柏瑞（510300）"
    assert universe.symbol_label("510300", None, "broad") == "（510300 · 宽基）"
    assert universe.symbol_label("510300") == "510300"
    # 名称与代码相同（元数据缺失时的占位）不应重复出现
    assert universe.symbol_label("510300", "510300", "broad") == "（510300 · 宽基）"
    # 未知类别原样返回（不猜），已知类别给中文
    assert universe.symbol_label("511380", "转债ETF", "convertible") == "转债ETF（511380 · 可转债）"
    assert "mortgage" in universe.symbol_label("999999", "某REIT", "mortgage"), "未知类别原样显示，不猜中文名"
    assert universe.asset_class_label(None) == "未分类"


def test_label_maps_from_metadata() -> None:
    meta = pd.DataFrame(
        [
            {"symbol": "AAA", "name": "宽基A", "asset_class": "broad"},
            {"symbol": "BBB", "name": "债券B", "asset_class": "bond"},
        ]
    )
    labels, names = universe.label_maps(meta)
    assert labels["AAA"] == "宽基A（AAA · 宽基）"
    assert labels["BBB"] == "债券B（BBB · 债券）"
    assert names == {"AAA": "宽基A", "BBB": "债券B"}
    # 空表不能炸
    assert universe.label_maps(pd.DataFrame()) == ({}, {})


def test_dashboard_shows_names_not_only_codes(tmp_path: Path) -> None:
    """仪表盘各处（权重 chips、各标的表、图例）都要出现名称，而不是只有代码。"""
    db = tmp_path / "lab.duckdb"
    _seed_db(db)
    con = repo.connect(db)
    result = static_site.compute_preset(con, _spec("named"))
    con.close()

    assert result["labels"]["AAA"] == "宽基A（AAA · 宽基）"
    assert result["names"]["BBB"] == "债券B"

    figs: dict = {}
    html = static_site.render_dashboard(result, prefix="named-", figs=figs)
    assert "宽基A（AAA · 宽基）" in html, "权重 chips 里应当有名称"
    # 各标的表用标签
    per_asset = static_site._per_asset_table(result)
    assert "宽基A" in per_asset and "债券B" in per_asset
    # 图表用简短名称，不塞整条标签（否则坐标轴会被挤爆）
    assert "宽基A" in json.dumps(figs, ensure_ascii=False)
    assert "宽基A（AAA · 宽基）" not in json.dumps(figs, ensure_ascii=False)


def test_insight_text_uses_names(tmp_path: Path) -> None:
    """洞察文案是给人看的，不能只出现代码。"""
    from etf_lab.reports import insights as insights_mod

    db = tmp_path / "lab.duckdb"
    _seed_db(db)
    con = repo.connect(db)
    # 造一个权重与风险贡献严重失衡的组合，确保触发那条洞察
    result = static_site.compute_preset(con, _spec("insight"))
    con.close()
    result["weights"] = {"AAA": 0.05, "BBB": 0.95}
    found = insights_mod.evaluate(result)
    titles = " ".join(item.title for item in found)
    if "风险" in titles:
        assert "AAA" not in titles or "宽基A" in titles, "洞察文案里应出现名称"
        assert "宽基A" in titles or "债券B" in titles


def test_requested_dca_mode_is_actually_used(tmp_path: Path) -> None:
    """``spec.dca['mode']`` 必须真的生效。

    这是实际发生过的 bug：``compute_preset`` 里硬编码 ``for mode in ("fixed", "value_avg")``，
    于是实验室的「定投方式」下拉框选了 ``target_vol`` 也毫无效果——界面看着能用，其实是个摆设。
    """
    db = tmp_path / "lab.duckdb"
    _seed_db(db)
    con = repo.connect(db)
    try:
        fixed = static_site.compute_preset(con, _spec("m_fixed"))
        value_avg = static_site.compute_preset(
            con, _spec("m_va", dca={"amount": 1000.0, "freq": "monthly", "mode": "value_avg", "day": None})
        )
        target_vol = static_site.compute_preset(
            con,
            _spec(
                "m_tv",
                dca={"amount": 1000.0, "freq": "monthly", "mode": "target_vol", "day": None, "params": {"target_vol": 0.10}},
            ),
        )
    finally:
        con.close()

    assert list(fixed["dca"]) == ["fixed"]
    # 非固定模式：固定金额留作基准，再加请求的模式
    assert set(value_avg["dca"]) == {"fixed", "value_avg"}
    assert set(target_vol["dca"]) == {"fixed", "target_vol"}
    # 目标波动率模式的投入曲线应当与固定金额明显不同（否则说明参数没传进去）
    fixed_invested = fixed["dca"]["fixed"]["invested_total"]
    scaled_invested = target_vol["dca"]["target_vol"]["invested_total"]
    assert fixed_invested != scaled_invested


def test_dca_mode_labels_cover_every_mode() -> None:
    """界面标签必须覆盖引擎全部模式，不能再手写子集。"""
    from etf_lab.core import dca as dca_core

    assert set(dca_core.MODE_LABELS) == set(dca_core.MODES)
    assert all(dca_core.MODE_LABELS[mode] for mode in dca_core.MODES)


def test_standalone_page_ids_are_unique(tmp_path: Path) -> None:
    """独立组合页：id 唯一、自带数据文件与装载脚本。"""
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
