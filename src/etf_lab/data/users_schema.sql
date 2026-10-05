-- 用户与组合：**独立于行情库**的第二个数据库文件（data/users.duckdb）。
--
-- 为什么必须分开
-- --------------
-- DuckDB 里只要有任一进程以读写方式打开文件，就取独占锁；而且**数据库实例在进程内
-- 常驻**——即使 con.close()，锁也会留到进程退出。本应用的进程池必须以只读打开行情库
-- 才能并发计算（见 services/jobs.py 的说明）。
--
-- 如果"保存组合"写的是同一个文件，主进程一写就把读写锁拿走，
-- 所有 worker 立刻失败，表现是"有的组合能打开、有的报 500"。
-- 把用户数据放到独立文件后，两边互不干扰：
--
--   行情库 data/lab.duckdb    只读制品，由 etl 采集产生，多个进程共享读
--   用户库 data/users.duckdb  事务型小库，只有主进程读写
--
-- 设计意图（与 schema.sql 的注释一致）：**登录仅用于保存组合，匿名用户完全不落盘**，
-- 因此这个库里只有"账号"和"用户保存的组合"两类数据，没有任何分析功能依赖它。

CREATE TABLE IF NOT EXISTS users (
    username    VARCHAR PRIMARY KEY,
    pw_hash     VARCHAR NOT NULL,
    created_at  TIMESTAMP
);

CREATE TABLE IF NOT EXISTS portfolios (
    id               VARCHAR PRIMARY KEY,
    username         VARCHAR NOT NULL,
    name             VARCHAR,
    definition_json  VARCHAR,
    updated_at       TIMESTAMP
);
