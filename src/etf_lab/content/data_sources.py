"""数据集的采集配置；不是每条历史记录的来源证明。"""

DATASETS = (
    ("etf_price", "ETF 行情", "close_adj", "腾讯主源 / 东财备用；搜狐校验"),
    ("index_price", "指数行情", "close", "腾讯主源 / 东财备用"),
    ("fund_nav", "基金净值", "nav", "天天基金 / 新浪备用"),
    ("bond_yield", "国债收益率", "yield", "中债 / 新浪备用"),
    ("fx_rate", "汇率", "close", "新浪中行牌价中的央行中间价"),
    ("future_daily", "期货行情", "close", "尚未接入采集"),
    ("option_daily", "期权行情", "close", "尚未接入采集；现有期权模块为理论实验"),
)
