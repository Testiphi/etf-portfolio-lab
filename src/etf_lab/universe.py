"""标的清单与资产属性。

这里的代码只做**分类**，不写费率、不写历史名称——那些必须从数据源读，
否则就是把记忆当事实写进产品（本项目的一条纪律）。
"""

from __future__ import annotations

from typing import Any

# symbol → 资产属性。
ETF_PRESET: dict[str, dict[str, Any]] = {
    "510300": {"asset_class": "broad", "underlying_index": "000300"},
    "510500": {"asset_class": "broad", "underlying_index": "000905"},
    "512100": {"asset_class": "broad", "underlying_index": "000852"},
    "159915": {"asset_class": "broad", "underlying_index": "399006"},
    "515180": {"asset_class": "broad", "underlying_index": "000922"},
    "511010": {"asset_class": "bond", "underlying_index": None},
    "511260": {"asset_class": "bond", "underlying_index": None},
    "518880": {"asset_class": "gold", "underlying_index": None},
    "513100": {"asset_class": "cross_border", "underlying_index": None},
    "159920": {"asset_class": "cross_border", "underlying_index": None},
}

# 指数不可直接交易，仅用于补充 ETF 上市时间过短导致的样本不足。
INDEX_PRESET: dict[str, str] = {
    "000300": "沪深300",
    "000905": "中证500",
    "000852": "中证1000",
    "399006": "创业板指",
    "000922": "中证红利",
    "000016": "上证50",
    "000012": "上证国债指数",
    "000985": "中证全指",
}

# 行业指数用于 RBSA 收益法敞口回归（**不需要**指数成分股名单）。
# 这些代码来自申万行业分类，东财接口是否提供**必须实测**，见 fetch.probe_sources()。
INDUSTRY_INDEX_PRESET: dict[str, str] = {
    "801010": "农林牧渔(申万)",
    "801030": "基础化工(申万)",
    "801040": "钢铁(申万)",
    "801050": "有色金属(申万)",
    "801080": "电子(申万)",
    "801110": "家用电器(申万)",
    "801120": "食品饮料(申万)",
    "801130": "纺织服饰(申万)",
    "801140": "轻工制造(申万)",
    "801150": "医药生物(申万)",
    "801160": "公用事业(申万)",
    "801170": "交通运输(申万)",
    "801180": "房地产(申万)",
    "801200": "商贸零售(申万)",
    "801210": "社会服务(申万)",
    "801730": "电力设备(申万)",
    "801740": "国防军工(申万)",
    "801750": "计算机(申万)",
    "801760": "传媒(申万)",
    "801770": "通信(申万)",
    "801780": "银行(申万)",
    "801790": "非银金融(申万)",
    "801880": "汽车(申万)",
    "801890": "机械设备(申万)",
    "801950": "煤炭(申万)",
    "801960": "石油石化(申万)",
    "801970": "环保(申万)",
    "801980": "美容护理(申万)",
}

ALL_INDEX_NAMES: dict[str, str] = {**INDEX_PRESET, **INDUSTRY_INDEX_PRESET}

# T+0 / T+1 的公开常见分类。**涨跌幅限制一律留空**，因为它必须按交易所当期规则核实，
# 与其写个猜的数字，不如让页面显示"待核实"。
T_PLUS_BY_CLASS: dict[str, int] = {
    "broad": 1,
    "industry": 1,
    "bond": 0,
    "gold": 0,
    "cross_border": 0,
}

# 资产类别的中文名。界面上**必须**显示它，而不是只给一个代码——
# 没人记得住 510300 与 510500 哪个是沪深300、哪个是中证500。
ASSET_CLASS_LABELS: dict[str, str] = {
    "broad": "宽基",
    "industry": "行业",
    "bond": "债券",
    "convertible": "可转债",
    "gold": "商品（黄金）",
    "cross_border": "跨境",
    "cash": "现金",
}


def asset_class_label(asset_class: str | None) -> str:
    """资产类别的中文名；未知类别原样返回，不猜。"""
    text = str(asset_class or "").strip()
    if not text:
        return "未分类"
    return ASSET_CLASS_LABELS.get(text, text)


def symbol_label(symbol: str, name: str | None = None, asset_class: str | None = None) -> str:
    """标的的**人读标签**：``名称（代码 · 板块）``，缺失的字段自动省略。

    全项目只在这里决定"怎么称呼一个标的"。散落各处手写会让静态站与应用
    对同一个标的叫法不同——那正是"两条路线互相矛盾"的一种。
    """
    parts = [str(symbol)]
    label = asset_class_label(asset_class) if asset_class else ""
    if label and label != "未分类":
        parts.append(label)
    detail = " · ".join(parts)
    clean_name = str(name or "").strip()
    if not clean_name or clean_name == str(symbol):
        # 没有名称时，只有"板块"这类额外信息才值得加括号；
        # 否则会输出「（510300）」这种既冗余又难看的标签，不如直接给代码。
        return f"（{detail}）" if len(parts) > 1 else str(symbol)
    return f"{clean_name}（{detail}）"


def label_maps(meta: Any) -> tuple[dict[str, str], dict[str, str]]:
    """从 ``etf_meta`` 造出两张映射：``(完整标签, 简短名称)``。

    * 完整标签用于表格、图例说明、自定义页——``沪深300ETF华泰柏瑞（510300 · 宽基）``
    * 简短名称用于图表坐标轴与图例——``沪深300ETF华泰柏瑞``（长标签会把画布挤爆）

    缺元数据时退回代码本身，而不是留空或猜名字。
    """
    labels: dict[str, str] = {}
    names: dict[str, str] = {}
    if meta is None or getattr(meta, "empty", True):
        return labels, names
    columns = set(getattr(meta, "columns", []))
    if "symbol" not in columns:
        return labels, names
    has_name = "name" in columns
    has_class = "asset_class" in columns
    for row in meta.itertuples(index=False):
        symbol = str(getattr(row, "symbol"))
        name = str(getattr(row, "name")) if has_name and getattr(row, "name") is not None else None
        asset_class = str(getattr(row, "asset_class")) if has_class and getattr(row, "asset_class") is not None else None
        labels[symbol] = symbol_label(symbol, name, asset_class)
        names[symbol] = (name or symbol)
    return labels, names
