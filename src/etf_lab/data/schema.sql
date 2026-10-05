-- ETF Portfolio Lab · 本地数据仓 schema（DuckDB）
--
-- 口径约定（全项目统一，写入 data_version 后不再变更）：
--   * etf_price.close      = 未复权收盘价（原始行情）
--   * etf_price.close_adj  = 前复权收盘价（本项目**唯一**用于收益计算的价）
--   * etf_price.adj_factor = close_adj / close，便于随时还原口径
-- 指数不可直接交易，仅用于补充 ETF 上市时间过短导致的样本不足。

CREATE TABLE IF NOT EXISTS data_version (
    version     VARCHAR PRIMARY KEY,
    updated_at  TIMESTAMP,
    source      VARCHAR,
    notes       VARCHAR
);

CREATE TABLE IF NOT EXISTS etf_meta (
    symbol            VARCHAR PRIMARY KEY,
    name              VARCHAR,
    exchange          VARCHAR,
    asset_class       VARCHAR,      -- broad / industry / bond / gold / cross_border
    underlying_index  VARCHAR,
    list_date         DATE,
    mgmt_fee          DOUBLE,       -- 年费率，None 表示未知（不得用 0 代替）
    custodian_fee     DOUBLE,
    currency          VARCHAR,
    is_cross_border   BOOLEAN,
    t_plus            INTEGER,      -- 0 = T+0，1 = T+1
    price_limit       DOUBLE,       -- 涨跌幅限制，None 表示待核实
    notes             VARCHAR
);

CREATE TABLE IF NOT EXISTS etf_price (
    symbol      VARCHAR,
    date        DATE,
    open        DOUBLE,
    high        DOUBLE,
    low         DOUBLE,
    close       DOUBLE,
    volume      DOUBLE,
    amount      DOUBLE,
    adj_factor  DOUBLE,
    close_adj   DOUBLE,
    PRIMARY KEY (symbol, date)
);

CREATE TABLE IF NOT EXISTS index_price (
    index_code  VARCHAR,
    name        VARCHAR,
    date        DATE,
    close       DOUBLE,
    PRIMARY KEY (index_code, date)
);

-- 基金公司口径的净值（东财 lsjz 主源 / 新浪兜底）。
-- 用途：折溢价率 = 未复权市场价 / nav − 1；以及观察分红发生的时间。
-- 注意：中国的"累计净值"通常是 单位净值 + 历史累计分红（简单累加，不考虑分红再投资），
-- 因此 acc_nav 的增长**不等于**分红再投资的总收益，不能直接当作总收益序列用。
CREATE TABLE IF NOT EXISTS fund_nav (
    symbol        VARCHAR,
    date          DATE,
    nav           DOUBLE,   -- 单位净值
    acc_nav       DOUBLE,   -- 累计净值
    dividend      DOUBLE,   -- 当日分红（若数据源提供，否则 NULL）
    daily_change  DOUBLE,   -- 日增长率（%），来自数据源
    PRIMARY KEY (symbol, date)
);

CREATE TABLE IF NOT EXISTS fx_rate (
    pair   VARCHAR,     -- 例如 USDCNH / HKDCNY
    date   DATE,
    close  DOUBLE,
    PRIMARY KEY (pair, date)
);

CREATE TABLE IF NOT EXISTS bond_yield (
    date   DATE,
    code   VARCHAR,     -- 例如 CN10Y
    tenor  VARCHAR,
    yield  DOUBLE,
    PRIMARY KEY (date, code)
);

CREATE TABLE IF NOT EXISTS future_daily (
    symbol  VARCHAR,
    date    DATE,
    close   DOUBLE,
    settle  DOUBLE,
    volume  DOUBLE,
    oi      DOUBLE,
    PRIMARY KEY (symbol, date)
);

CREATE TABLE IF NOT EXISTS option_daily (
    symbol       VARCHAR,
    date         DATE,
    underlying   VARCHAR,
    expiry       DATE,
    strike       DOUBLE,
    option_type  VARCHAR,      -- C / P
    close        DOUBLE,
    volume       DOUBLE,
    oi           DOUBLE,
    PRIMARY KEY (symbol, date)
);

-- 计算结果缓存：key = sha256(规范化JSON(模块+函数+参数+data_version+code_version))
CREATE TABLE IF NOT EXISTS result_cache (
    cache_key   VARCHAR PRIMARY KEY,
    module      VARCHAR,
    payload     VARCHAR,
    created_at  TIMESTAMP,
    bytes       BIGINT
);

CREATE TABLE IF NOT EXISTS preset_portfolios (
    id               VARCHAR PRIMARY KEY,
    name             VARCHAR,
    description      VARCHAR,
    definition_json  VARCHAR
);

-- 用户与组合**不在这个库里**：行情库必须是只读制品（多个进程只读共享），
-- 而保存组合需要写。两者放同一个 DuckDB 文件会互相锁死。
-- 账号与已保存的组合见 data/users_schema.sql（独立的 data/users.duckdb）。
