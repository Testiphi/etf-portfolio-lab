"""洞察规则引擎：让**数据自己**决定该浮现哪条知识。

设计意图
--------
这个项目不是"知识点的讲义"，而是"数据的仪表盘"。因此知识不能靠导航去翻，而要满足两个条件：

1. **从算出来的数字里长出来**——规则读的是组合的指标、权重、风险贡献、相关性、
   复权事件等真实结果，不是预设要讲什么；
2. **默认不出现**——只有触发条件成立时才浮出一条洞察条，点开才展开公式与陷阱。

规则是**纯函数**：输入 ``compute_preset`` 的结果字典，输出洞察条列表。
因此它可以独立测试，也被静态站（路线 A）与 NiceGUI（路线 C）共用。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

# 触发阈值统一放在这里，便于调参和在教学页上如实说明"这条提醒是按什么标准冒出来的"
RISK_WEIGHT_GAP = 0.10
"""权重与风险贡献之差超过 10 个百分点 → 提示风险贡献失衡。"""
HIGH_CORRELATION = 0.90
"""两个标的相关系数高于 0.9 → 提示分散化失效。"""
DEEP_DRAWDOWN = -0.30
SLOW_RECOVERY_DAYS = 365
DCA_ALGORITHM_GAP = 0.01
ARITHMETIC_GEOMETRIC_GAP = 0.01
"""算术年化与复合年化之差超过 1 个百分点 → 提示波动拖累。"""
SMALL_SAMPLE_DAYS = 250
CONCENTRATION_SHARE = 0.80
SKEW_THRESHOLD = -0.30
VAR_METHOD_GAP = 0.20
"""历史法与参数法 VaR 的相对差超过 20% → 提示分布并非正态。"""
PREMIUM_DISCOUNT_ALERT = 0.01
"""组合加权折溢价绝对值超过 1% → 提示买贵/买便宜的幅度。"""


@dataclass(frozen=True)
class Insight:
    """一条洞察条。"""

    key: str
    """稳定标识，便于测试与去重。"""
    level: str
    """``warn``（数据看着有问题）/ ``info``（值得看一眼）/ ``unlock``（配出某种结构才出现）。"""
    title: str
    """一句话结论，**以数字为主**——它出现在仪表盘上，不展开。"""
    card: str
    """关联的教学卡片 key；点开就地展开公式与「何时会骗人」。"""
    evidence: Mapping[str, Any] = field(default_factory=dict)
    """支撑这条结论的原始数字，展开时一并显示，便于核对。"""


def _f(result: Mapping[str, Any], *path: str, default: Any = None) -> Any:
    """安全取值：``_f(result, "metrics", "sharpe")``。"""
    node: Any = result
    for key in path:
        if not isinstance(node, Mapping) or key not in node:
            return default
        node = node[key]
    return node


# --------------------------------------------------------------------------- #
# 单条规则
# --------------------------------------------------------------------------- #
def _label(result: Mapping[str, Any], symbol: Any) -> str:
    """标的的人读标签。

    洞察文案是给**人**看的，不该只出现 ``510300`` 这样的代码——
    没人记得住哪个代码对应哪只 ETF。机器可读的原值仍放在 ``evidence`` 里。
    """
    key = str(symbol)
    return str((result.get("labels") or {}).get(key, key))


def rule_risk_contribution_gap(result: Mapping[str, Any]) -> Insight | None:
    """权重与风险贡献失衡——最容易让人误解自己持仓结构的一条。"""
    shares = _f(result, "risk_contribution", "component_var_share", default={}) or {}
    weights = _f(result, "weights", default={}) or {}
    worst: tuple[str, float, float] | None = None
    for symbol, share in shares.items():
        if share is None:
            continue
        gap = float(share) - float(weights.get(symbol, 0.0))
        if worst is None or abs(gap) > abs(worst[1]):
            worst = (symbol, gap, float(share))
    if worst is None or abs(worst[1]) < RISK_WEIGHT_GAP:
        return None
    symbol, gap, share = worst
    direction = "高于" if gap > 0 else "低于"
    return Insight(
        key="risk_contribution_gap",
        level="warn",
        title=f"{_label(result, symbol)} 占 {weights.get(symbol, 0):.0%} 权重，却贡献 {share:.0%} 风险（{direction}权重 {abs(gap):.0%}）",
        card="risk_contribution",
        evidence={"symbol": symbol, "weight": weights.get(symbol), "risk_share": share, "gap": gap},
    )


def rule_high_correlation(result: Mapping[str, Any]) -> Insight | None:
    pair = _f(result, "diagnostics", "max_corr_pair") or {}
    rho = pair.get("rho")
    if rho is None or float(rho) < HIGH_CORRELATION:
        return None
    return Insight(
        key="high_correlation",
        level="warn",
        title=f"{_label(result, pair.get('a'))} 与 {_label(result, pair.get('b'))} 相关系数 {float(rho):.3f}，几乎同涨同跌",
        card="correlation",
        evidence=dict(pair),
    )


def rule_slow_drawdown_recovery(result: Mapping[str, Any]) -> Insight | None:
    depth = _f(result, "max_drawdown_info", "depth")
    recovery_days = _f(result, "metrics", "max_drawdown_recovery_days")
    if depth is None or float(depth) > DEEP_DRAWDOWN:
        return None
    if recovery_days is None or int(recovery_days) < SLOW_RECOVERY_DAYS:
        return None
    return Insight(
        key="slow_recovery",
        level="warn",
        title=f"最深回撤 {float(depth):.1%}，用了 {int(recovery_days)} 天（{int(recovery_days) / 365:.1f} 年）才回本",
        card="max_drawdown",
        evidence={"depth": depth, "recovery_days": recovery_days, **_f(result, "max_drawdown_info", default={})},
    )


def rule_dca_algorithm_gap(result: Mapping[str, Any]) -> Insight | None:
    dca = _f(result, "dca", default={}) or {}
    best: tuple[str, float, float, float] | None = None
    for mode, payload in dca.items():
        if not isinstance(payload, Mapping) or "error" in payload:
            continue
        naive = payload.get("naive_annualized_return")
        xirr = payload.get("xirr")
        if naive is None or xirr is None:
            continue
        gap = abs(float(naive) - float(xirr))
        if best is None or gap > best[1]:
            best = (mode, gap, float(naive), float(xirr))
    if best is None or best[1] < DCA_ALGORITHM_GAP:
        return None
    mode, gap, naive, xirr = best
    label = "固定金额" if mode == "fixed" else "价值平均"
    tw = _f(result, "time_weighted_annualized")
    return Insight(
        key="dca_gap",
        level="info",
        title=f"{label}定投：「按累计收益折年化」得 {naive:.2%}，按资金加权 XIRR 只有 {xirr:.2%}，差 {gap:.2%}"
        + (f"（组合本身时间加权年化 {float(tw):.2%}）" if tw is not None else ""),
        card="xirr",
        evidence={"mode": mode, "naive": naive, "xirr": xirr, "gap": gap, "time_weighted": tw},
    )


def rule_arithmetic_vs_geometric_gap(result: Mapping[str, Any]) -> Insight | None:
    """算术平均年化与复合年化的差距 = 波动拖累。

    这是最普遍的收益认知偏差：宣传口径常用算术平均，而投资者实际拿到的是复合收益。
    """
    arithmetic = _f(result, "metrics", "arithmetic_annualized_return")
    geometric = _f(result, "metrics", "annualized_return")
    drag = _f(result, "metrics", "volatility_drag")
    if arithmetic is None or geometric is None or drag is None:
        return None
    if abs(float(drag)) < ARITHMETIC_GEOMETRIC_GAP:
        return None
    vol = _f(result, "metrics", "annualized_volatility")
    theory = (float(vol) ** 2 / 2) if vol is not None else None
    extra = f"，理论近似 σ²/2 = {theory:.2%}" if theory is not None else ""
    return Insight(
        key="arithmetic_geometric_gap",
        level="info",
        title=(
            f"算术平均年化 {float(arithmetic):.2%} 比复合年化 {float(geometric):.2%} "
            f"高出 {float(drag):.2%}{extra}——这就是波动拖累"
        ),
        card="arithmetic_vs_geometric",
        evidence={"arithmetic": arithmetic, "geometric": geometric, "drag": drag, "sigma_squared_over_2": theory},
    )


def rule_premium_discount(result: Mapping[str, Any]) -> Insight | None:
    """组合加权折溢价——买入时多付/少付了多少钱。"""
    block = _f(result, "premium_discount") or {}
    latest = block.get("weighted_latest")
    if latest is None or abs(float(latest)) < PREMIUM_DISCOUNT_ALERT:
        return None
    direction = "溢价" if float(latest) > 0 else "折价"
    action = "买入即多付" if float(latest) > 0 else "买入即少付"
    worst = block.get("max_abs_symbol")
    extra = ""
    if worst:
        per = (block.get("per_symbol") or {}).get(worst) or {}
        if per.get("latest") is not None:
            extra = f"；{worst} 单只 {float(per['latest']):.2%}"
    # 跨境 ETF 的 QDII 净值披露有滞后（反映上一交易日境外收盘），
    # 直接把它当"多付了多少钱"会高估——必须在这条提醒里说清楚。
    has_cross_border = bool((_f(result, "composition") or {}).get("has_cross_border"))
    caveat = "（含跨境 ETF：其净值披露有时滞，估算偏高）" if has_cross_border else ""
    return Insight(
        key="premium_discount",
        level="warn" if float(latest) > 0 else "info",
        title=f"组合加权{direction} {float(latest):.2%}（{action}）{extra}{caveat}",
        card="premium_discount",
        evidence={
            "weighted_latest": latest,
            "cross_border_timing_lag": has_cross_border,
            **{k: v for k, v in block.items() if k not in ("per_symbol",)},
        },
    )


def rule_small_sample(result: Mapping[str, Any]) -> Insight | None:
    """样本少于一年时提示估计不稳定。"""
    n_obs = _f(result, "n_obs")
    if n_obs is None or int(n_obs) >= SMALL_SAMPLE_DAYS:
        return None
    return Insight(
        key="small_sample",
        level="warn",
        title=f"样本只有 {int(n_obs)} 个交易日（不足一年），相关性与夏普都极不稳定",
        card="beta",
        evidence={"n_obs": n_obs},
    )


def rule_adjustment_events(result: Mapping[str, Any]) -> Insight | None:
    events = _f(result, "diagnostics", "adjustment_events")
    if not events:
        return None
    symbols = _f(result, "diagnostics", "adjustment_symbols", default=[]) or []
    return Insight(
        key="adjustment_events",
        level="info",
        title=f"样本区间内有 {int(events)} 天发生分红/份额折算（涉及 {len(symbols)} 只标的）——已用前复权价计收益",
        card="adjustment",
        evidence={"days": events, "symbols": list(symbols)},
    )


def rule_return_concentration(result: Mapping[str, Any]) -> Insight | None:
    concentration = _f(result, "diagnostics", "concentration") or {}
    share = concentration.get("return_share")
    if share is None or float(share) < CONCENTRATION_SHARE:
        return None
    return Insight(
        key="return_concentration",
        level="warn",
        title=f"组合累计收益的 {float(share):.0%} 来自单一标的 {_label(result, concentration.get('symbol'))}",
        card="diversification_ratio",
        evidence=dict(concentration),
    )


def rule_var_method_divergence(result: Mapping[str, Any]) -> Insight | None:
    hist = _f(result, "metrics", "var_95_historical")
    para = _f(result, "metrics", "var_95_parametric")
    if not hist or not para:
        return None
    gap = abs(float(para) - float(hist)) / float(hist)
    if gap < VAR_METHOD_GAP:
        return None
    relation = "高于" if float(para) > float(hist) else "低于"
    return Insight(
        key="var_method_gap",
        level="info",
        title=f"参数法 VaR {float(para):.2%} {relation}历史法 {float(hist):.2%}（相差 {gap:.0%}），说明收益分布偏离正态",
        card="var",
        evidence={"historical": hist, "parametric": para, "relative_gap": gap},
    )


def rule_left_tail(result: Mapping[str, Any]) -> Insight | None:
    skew = _f(result, "metrics", "skewness")
    if skew is None or float(skew) > SKEW_THRESHOLD:
        return None
    return Insight(
        key="left_tail",
        level="info",
        title=f"偏度 {float(skew):.2f}：左尾比右尾更长，大跌比大涨更容易出现",
        card="cvar",
        evidence={"skewness": skew},
    )


# --------------------------------------------------------------------------- #
# 解锁：配出特定结构才出现的模块
# --------------------------------------------------------------------------- #
# ``implemented`` 标记该模块的功能是否已经上线。**不假装覆盖**：
# 组合结构满足但功能未上线时，界面显示「已触发 · 待接入」，并可先展开概念卡片。
UNLOCK_CATALOG: tuple[dict[str, Any], ...] = (
    {
        "key": "fx_exposure",
        "title": "汇率贡献分解",
        "card": "fx_exposure",
        "trigger": "组合含跨境 ETF",
        "requirement": "cross_border",
        "implemented": False,
        "pending": "需要汇率数据（USDCNH/HKDCNY）",
    },
    {
        "key": "duration",
        "title": "利率敏感性与久期",
        "card": "duration",
        "trigger": "组合含债券 ETF",
        "requirement": "bond",
        "implemented": True,
        "pending": "久期用收益对收益率变动的回归反推；R² 不足的标的会标注为无参考价值",
    },
    {
        "key": "greeks",
        "title": "期权 Greeks 与保护成本",
        "card": "greeks",
        "trigger": "任何组合（保护成本是通用问题）",
        "requirement": "always",
        "implemented": True,
        "pending": "隐含波动率需要期权行情（尚未接入）；当前用历史波动率给理论值",
    },
    {
        "key": "hedge_simulation",
        "title": "Delta-Gamma 复制与再平衡频率",
        "card": "delta_gamma_replication",
        "trigger": "任何组合（没有期权市场时的替代方案）",
        "requirement": "always",
        "implemented": True,
        "pending": "认沽用 Black-Scholes 理论定价（无期权行情）；标的按 GBM 生成，厚尾下误差尾部会被低估",
    },
    {
        "key": "basis",
        "title": "股指期货基差与展期成本",
        "card": "hedging",
        "trigger": "组合布置股指期货对冲",
        "requirement": "future_hedge",
        "implemented": False,
        "pending": "需要股指期货行情",
    },
    {
        "key": "exposure_matrix",
        "title": "因子/行业敞口矩阵（RBSA）",
        "card": "beta",
        "trigger": "任何组合（有指数数据即可）",
        "requirement": "always",
        "implemented": True,
        "pending": "",
    },
    {
        "key": "monte_carlo",
        "title": "蒙特卡洛分布与收敛诊断",
        "card": "monte_carlo",
        "trigger": "任何组合（四个模型并排对比）",
        "requirement": "always",
        "implemented": True,
        "pending": "GARCH 为自实现（方差目标化 MLE）；协整/多资产联合模拟尚未接入",
    },
)

REQUIREMENT_LABELS = {
    "cross_border": "跨境资产",
    "bond": "债券资产",
    "gold": "黄金资产",
    "broad": "宽基资产",
    "option_hedge": "期权头寸",
    "future_hedge": "期货头寸",
    "lab": "实验室开关",
    "always": "无",
}


def evaluate_unlocks(composition: Mapping[str, Any] | None, extra: Mapping[str, bool] | None = None) -> list[dict[str, Any]]:
    """判断每个模块是否被当前组合结构触发。"""
    composition = composition or {}
    extra = extra or {}
    unlocked: list[dict[str, Any]] = []
    for item in UNLOCK_CATALOG:
        requirement = str(item["requirement"])
        if requirement == "always":
            triggered = True
        elif requirement in ("option_hedge", "future_hedge", "lab"):
            triggered = bool(extra.get(requirement, False))
        else:
            triggered = bool(composition.get(f"has_{requirement}", False))
        unlocked.append({**item, "triggered": triggered, "requirement_label": REQUIREMENT_LABELS.get(requirement, requirement)})
    return unlocked


# --------------------------------------------------------------------------- #
# 汇总
# --------------------------------------------------------------------------- #
RULES: tuple[Callable[[Mapping[str, Any]], Insight | None], ...] = (
    rule_risk_contribution_gap,
    rule_high_correlation,
    rule_slow_drawdown_recovery,
    rule_dca_algorithm_gap,
    rule_premium_discount,
    rule_return_concentration,
    rule_var_method_divergence,
    rule_left_tail,
    rule_arithmetic_vs_geometric_gap,
    rule_adjustment_events,
    rule_small_sample,
)


def evaluate(result: Mapping[str, Any]) -> list[Insight]:
    """跑全部规则，返回洞察条（按严重程度排序，warn 在前）。

    规则内部全部做了取空保护：缺少某个字段时返回 ``None`` 而不是抛异常——
    仪表盘少一条提醒可以接受，整页打不开不行。
    """
    insights: list[Insight] = []
    for rule in RULES:
        try:
            insight = rule(result)
        except Exception:  # noqa: BLE001 - 单条规则出错不应影响整页
            continue
        if insight is not None:
            insights.append(insight)
    order = {"warn": 0, "info": 1, "unlock": 2}
    return sorted(insights, key=lambda item: (order.get(item.level, 9), item.key))
