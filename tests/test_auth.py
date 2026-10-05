"""账号与组合保存的对照测试（不访问网络）。

这里的重点有两个：

1. **匿名优先**：保存功能必须完全独立于行情库——它写的是另一个文件。
   这条如果破了，进程池会立刻因为 DuckDB 的写锁整体失败（实测踩过）。
2. **数据隔离**：用户只能读到自己的组合，且"不存在"与"不属于你"返回同一个结果。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from etf_lab.data import repo
from etf_lab.services import auth


def _users_db(tmp_path: Path) -> Path:
    return tmp_path / "users.duckdb"


# --------------------------------------------------------------------------- #
# 口令
# --------------------------------------------------------------------------- #
def test_password_hash_round_trip() -> None:
    stored = auth.hash_password("correct horse battery")
    assert stored.startswith("pbkdf2_sha256$200000$")
    assert auth.verify_password("correct horse battery", stored)
    assert not auth.verify_password("correct horse batteru", stored)


def test_password_hash_is_salted() -> None:
    """同一口令两次哈希必须不同（否则等于没加盐，彩虹表直接命中）。"""
    first = auth.hash_password("same password")
    second = auth.hash_password("same password")
    assert first != second
    assert auth.verify_password("same password", first)
    assert auth.verify_password("same password", second)


def test_password_hash_records_algorithm_and_iterations() -> None:
    """哈希自描述，将来提高迭代次数不会让老口令失效。"""
    stored = auth.hash_password("another password", iterations=1000)
    assert stored.split("$")[1] == "1000"
    assert auth.verify_password("another password", stored)


def test_short_password_rejected() -> None:
    with pytest.raises(auth.AuthError, match="至少"):
        auth.hash_password("short")


def test_verify_password_rejects_malformed_hash() -> None:
    """哈希字段坏了要返回 False，而不是抛异常（异常会泄露内部细节）。"""
    for broken in ("", "not-a-hash", "pbkdf2_sha256$abc$zz$yy", "md5$1$aa$bb"):
        assert auth.verify_password("whatever", broken) is False


# --------------------------------------------------------------------------- #
# 账号
# --------------------------------------------------------------------------- #
def test_register_and_authenticate(tmp_path: Path) -> None:
    con = repo.connect_users(_users_db(tmp_path))
    try:
        assert auth.register(con, "Alice", "a good password") == "alice"
        assert auth.authenticate(con, "alice", "a good password")
        assert auth.authenticate(con, "ALICE", "a good password"), "用户名大小写不敏感"
        assert not auth.authenticate(con, "alice", "wrong password")
        assert not auth.authenticate(con, "nobody", "a good password")
    finally:
        con.close()


def test_duplicate_registration_rejected(tmp_path: Path) -> None:
    con = repo.connect_users(_users_db(tmp_path))
    try:
        auth.register(con, "bob", "a good password")
        with pytest.raises(auth.AuthError, match="已经被注册"):
            auth.register(con, "Bob", "another password")
    finally:
        con.close()


def test_plaintext_password_never_stored(tmp_path: Path) -> None:
    """库里不能出现明文口令——这是最容易犯、后果最重的错误。"""
    path = _users_db(tmp_path)
    con = repo.connect_users(path)
    try:
        auth.register(con, "carol", "super secret password")
    finally:
        con.close()
    raw = path.read_bytes()
    assert b"super secret password" not in raw


# --------------------------------------------------------------------------- #
# 组合保存
# --------------------------------------------------------------------------- #
def test_save_list_load_delete_round_trip(tmp_path: Path) -> None:
    con = repo.connect_users(_users_db(tmp_path))
    try:
        auth.register(con, "dave", "a good password")
        definition = {"weights": {"AAA": 0.6, "BBB": 0.4}, "dca": {"amount": 2000.0}}
        saved_id = auth.save_portfolio(con, "dave", "我的组合", definition)

        listed = auth.list_portfolios(con, "dave")
        assert [row["name"] for row in listed] == ["我的组合"]

        loaded = auth.load_portfolio(con, "dave", saved_id)
        assert loaded is not None
        assert loaded["definition"] == definition, "存取必须完全一致（含中文键值）"

        assert auth.delete_portfolio(con, "dave", saved_id)
        assert auth.list_portfolios(con, "dave") == []
        assert auth.load_portfolio(con, "dave", saved_id) is None
    finally:
        con.close()


def test_saving_only_stores_definitions_not_results(tmp_path: Path) -> None:
    """只存组合定义，不存计算结果。

    结果会随数据版本变化——存下来就会在下次打开时显示过期数字，
    那比不存更糟（而且它正是"两条路线结论不一致"的来源之一）。
    """
    con = repo.connect_users(_users_db(tmp_path))
    try:
        definition = {"weights": {"AAA": 1.0}}
        auth.save_portfolio(con, "erin", "只存定义", definition)
        row = con.execute("SELECT definition_json FROM portfolios").fetchone()
        payload = json.loads(row[0])
        assert set(payload) == {"weights"}
        assert "metrics" not in payload and "nav" not in payload
    finally:
        con.close()


def test_users_cannot_read_each_others_portfolios(tmp_path: Path) -> None:
    """跨用户读取必须失败，且"不存在"与"无权限"要给出同一个结果（不泄露存在性）。"""
    con = repo.connect_users(_users_db(tmp_path))
    try:
        auth.register(con, "frank", "a good password")
        auth.register(con, "grace", "a good password")
        frank_id = auth.save_portfolio(con, "frank", "frank 的组合", {"weights": {"AAA": 1.0}})

        assert auth.load_portfolio(con, "grace", frank_id) is None
        assert auth.list_portfolios(con, "grace") == []
        assert not auth.delete_portfolio(con, "grace", frank_id)
        # frank 自己的还在
        assert auth.load_portfolio(con, "frank", frank_id) is not None
    finally:
        con.close()


def test_save_rejects_empty_name(tmp_path: Path) -> None:
    con = repo.connect_users(_users_db(tmp_path))
    try:
        with pytest.raises(auth.AuthError, match="名称"):
            auth.save_portfolio(con, "heidi", "   ", {"weights": {}})
    finally:
        con.close()


# --------------------------------------------------------------------------- #
# 与行情库的隔离：这条如果破了，进程池会整体失败
# --------------------------------------------------------------------------- #
def test_user_database_is_a_separate_file(tmp_path: Path) -> None:
    """用户库必须是独立文件，且行情库的 schema 里不再有 users/portfolios。

    行情库要被多个进程**只读**共享，而保存组合需要写；放同一个文件必然互相锁死
    （实测表现：未缓存的组合页全部 500，报「另一个程序正在使用此文件」）。
    """
    assert repo.DEFAULT_USERS_DB_PATH != repo.DEFAULT_DB_PATH
    schema_text = (Path(repo.__file__).with_name("schema.sql")).read_text(encoding="utf-8")
    assert "CREATE TABLE IF NOT EXISTS users" not in schema_text
    assert "CREATE TABLE IF NOT EXISTS portfolios" not in schema_text
    # 用户库的表在独立 schema 里
    users_schema = Path(repo.__file__).with_name("users_schema.sql").read_text(encoding="utf-8")
    assert "CREATE TABLE IF NOT EXISTS users" in users_schema
    assert "CREATE TABLE IF NOT EXISTS portfolios" in users_schema


def test_saving_does_not_block_read_only_market_data(tmp_path: Path) -> None:
    """保存组合之后，行情库仍然能被只读打开（保存不碰行情库）。

    这是"登录仅保存组合"这个定位在架构上的要求：写用户数据不能影响行情数据的读。
    """
    market = tmp_path / "lab.duckdb"
    market_con = repo.connect(market)
    market_con.execute("CREATE TABLE IF NOT EXISTS etf_meta (symbol VARCHAR)")
    market_con.close()

    users_con = repo.connect_users(_users_db(tmp_path))
    try:
        auth.save_portfolio(users_con, "ivan", "组合", {"weights": {"AAA": 1.0}})
    finally:
        users_con.close()

    # 行情库此刻仍可被只读打开（若保存误写行情库，这里会因为写锁而失败）
    reader = repo.connect(market, read_only=True)
    try:
        assert reader.execute("SELECT COUNT(*) FROM etf_meta").fetchone()[0] == 0
    finally:
        reader.close()


def test_market_data_counts_no_longer_include_user_tables(tmp_path: Path) -> None:
    """行情库的"数据底座"不应再列用户表——它们不在这个库里。"""
    con = repo.connect(tmp_path / "lab.duckdb")
    try:
        counts = repo.table_counts(con)
    finally:
        con.close()
    assert "users" not in counts
    assert "portfolios" not in counts
