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
