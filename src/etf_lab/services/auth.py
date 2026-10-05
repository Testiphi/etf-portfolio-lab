"""账号与组合保存：**唯一的写库路径**，且只写用户库，绝不碰行情库。

定位（也是本模块存在的理由）
----------------------------
本项目**匿名可用全部功能**：三条路线里的任何分析都不需要登录。
登录只做一件事——把你自己配好的组合存下来，方便下次打开。
因此这里没有权限模型、没有角色、没有任何"登录才能看"的内容。

安全上的诚实说明
----------------
* 口令用 **PBKDF2-HMAC-SHA256**（20 万次迭代、每用户 16 字节随机盐）存储，
  校验用 ``hmac.compare_digest`` 做定时安全比较；
* 哈希字符串自带算法与参数（``pbkdf2_sha256$迭代次数$盐$摘要``），
  以后调参数不会让老口令失效；
* **但这不是一套加固过的账号系统**：它面向本地/自托管的单机小站，
  没有登录限流、没有口令找回、没有二次验证。
  部署到公网**必须走 HTTPS**——否则口令在链路上是明文。
* 匿名用户**完全不落盘**：不登录时不会往用户库里写任何一行。
"""

from __future__ import annotations

import datetime as dt
import hashlib
import hmac
import json
import secrets
import uuid
from typing import Any, Mapping

import duckdb

ALGORITHM = "pbkdf2_sha256"
ITERATIONS = 200_000
SALT_BYTES = 16
MIN_PASSWORD_LENGTH = 8


class AuthError(RuntimeError):
    """账号相关的可预期错误（用户名已存在、口令太短等），消息直接给用户看。"""


def hash_password(password: str, *, salt: bytes | None = None, iterations: int = ITERATIONS) -> str:
    """生成自描述的口令哈希字符串。"""
    if len(password) < MIN_PASSWORD_LENGTH:
        raise AuthError(f"口令至少 {MIN_PASSWORD_LENGTH} 位")
    salt = salt or secrets.token_bytes(SALT_BYTES)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return f"{ALGORITHM}${iterations}${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    """校验口令；哈希格式不对时返回 False（而不是抛异常泄露细节）。"""
    try:
        algorithm, iterations_text, salt_hex, digest_hex = stored.split("$")
        if algorithm != ALGORITHM:
            return False
        digest = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), bytes.fromhex(salt_hex), int(iterations_text)
        )
    except (ValueError, AttributeError, TypeError):
        return False
    return hmac.compare_digest(digest.hex(), digest_hex)


def normalize_username(username: str) -> str:
    """用户名规范化：去空白并转小写，避免 "Alice" 与 "alice" 变成两个账号。"""
    return username.strip().lower()


def register(con: duckdb.DuckDBPyConnection, username: str, password: str) -> str:
    """注册账号；用户名已存在时抛 :class:`AuthError`。"""
    name = normalize_username(username)
    if len(name) < 3:
        raise AuthError("用户名至少 3 个字符")
    existing = con.execute("SELECT 1 FROM users WHERE username = ?", [name]).fetchone()
    if existing:
        raise AuthError("这个用户名已经被注册了")
    con.execute(
        "INSERT INTO users (username, pw_hash, created_at) VALUES (?, ?, ?)",
        [name, hash_password(password), dt.datetime.now()],
    )
    return name


def authenticate(con: duckdb.DuckDBPyConnection, username: str, password: str) -> bool:
    """校验用户名与口令。**用户名不存在与口令错误返回同一个结果**，不区分。"""
    name = normalize_username(username)
    row = con.execute("SELECT pw_hash FROM users WHERE username = ?", [name]).fetchone()
    if not row:
        # 仍然做一次哈希，避免用响应时间判断"用户是否存在"
        hash_password(password if len(password) >= MIN_PASSWORD_LENGTH else "x" * MIN_PASSWORD_LENGTH)
        return False
    return verify_password(password, str(row[0]))


def save_portfolio(
    con: duckdb.DuckDBPyConnection,
    username: str,
    name: str,
    definition: Mapping[str, Any],
    *,
    portfolio_id: str | None = None,
) -> str:
    """保存（或覆盖）一个组合定义，返回组合 id。

    ``definition`` 是纯字典（权重、定投计划等）——**不存计算结果**：
    结果会随数据版本变化，存下来就会变成过期数字，那比不存更糟。
    """
    owner = normalize_username(username)
    if not name.strip():
        raise AuthError("组合名称不能为空")
    identifier = portfolio_id or uuid.uuid4().hex
    payload = json.dumps(definition, ensure_ascii=False, sort_keys=True)
    con.execute("DELETE FROM portfolios WHERE id = ? AND username = ?", [identifier, owner])
    con.execute(
        "INSERT INTO portfolios (id, username, name, definition_json, updated_at) VALUES (?, ?, ?, ?, ?)",
        [identifier, owner, name.strip(), payload, dt.datetime.now()],
    )
    return identifier


def list_portfolios(con: duckdb.DuckDBPyConnection, username: str) -> list[dict[str, Any]]:
    """列出该用户保存的组合（只返回自己的）。"""
    owner = normalize_username(username)
    rows = con.execute(
        "SELECT id, name, updated_at FROM portfolios WHERE username = ? ORDER BY updated_at DESC",
        [owner],
    ).fetchall()
    return [
        {"id": str(row[0]), "name": str(row[1]), "updated_at": str(row[2])[:19]}
        for row in rows
    ]


def load_portfolio(con: duckdb.DuckDBPyConnection, username: str, portfolio_id: str) -> dict[str, Any] | None:
    """读取一个组合；**不属于该用户时返回 None**（而不是报"无权限"，避免泄露存在性）。"""
    owner = normalize_username(username)
    row = con.execute(
        "SELECT id, name, definition_json FROM portfolios WHERE id = ? AND username = ?",
        [portfolio_id, owner],
    ).fetchone()
    if not row:
        return None
    return {
        "id": str(row[0]),
        "name": str(row[1]),
        "definition": json.loads(row[2]),
    }


def delete_portfolio(con: duckdb.DuckDBPyConnection, username: str, portfolio_id: str) -> bool:
    """删除一个组合；返回是否真的删掉了（同样只能删自己的）。"""
    owner = normalize_username(username)
    before = con.execute(
        "SELECT 1 FROM portfolios WHERE id = ? AND username = ?", [portfolio_id, owner]
    ).fetchone()
    if not before:
        return False
    con.execute("DELETE FROM portfolios WHERE id = ? AND username = ?", [portfolio_id, owner])
    return True
