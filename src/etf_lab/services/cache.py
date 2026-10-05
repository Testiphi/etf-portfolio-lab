"""计算结果缓存：让"同一个参数组合只算一次"。

缓存键必须同时绑住三样东西，缺一个都会出现"看起来对但其实过期"的结果：

1. **参数**——规范化后的 JSON，浮点统一舍入到 1e-10，避免 0.30000000000000004 这类
   浮点噪声把同一个请求变成两个缓存条目；
2. **数据版本**——``data_version`` 变了说明行情更新了，旧结果必须失效；
3. **代码版本**——公式改了，旧结果同样必须失效。

这也是小规格服务器上最有效的性能手段：示例组合被不同访客反复打开时全是命中缓存。
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

from etf_lab import __version__
from etf_lab.data import repo

FLOAT_PRECISION = 10


def canonicalize(value: Any) -> Any:
    """把任意嵌套结构规范化成可稳定哈希的形式。"""
    if isinstance(value, float):
        if value != value:  # NaN
            return "NaN"
        if value in (float("inf"), float("-inf")):
            return "Inf" if value > 0 else "-Inf"
        return round(value, FLOAT_PRECISION)
    if isinstance(value, Mapping):
        return {str(k): canonicalize(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [canonicalize(v) for v in value]
    if isinstance(value, set):
        return sorted(canonicalize(v) for v in value)
    return value


def cache_key(module: str, function: str, params: Mapping[str, Any], data_version: str) -> str:
    """生成缓存键。"""
    payload = {
        "module": module,
        "function": function,
        "params": canonicalize(dict(params)),
        "data_version": data_version,
        "code_version": __version__,
    }
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def get_or_compute(con, module: str, function: str, params: Mapping[str, Any], compute) -> Any:  # noqa: ANN001, ANN201
    """命中则返回缓存，否则计算并写回。

    ``compute`` 必须是无副作用、可独立测试的可调用对象（通常是 ``core/`` 里的纯函数包装）。
    """
    version = repo.latest_data_version(con)
    key = cache_key(module, function, params, version)
    cached = repo.cache_get(con, key)
    if cached is not None:
        return json.loads(cached)

    result = compute()
    repo.cache_put(con, key, module, json.dumps(result, ensure_ascii=False, default=str))
    return result


def cache_stats(con) -> dict[str, Any]:  # noqa: ANN001
    """缓存概况——页面上要让人看得见"这次是算出来的还是读出来的"。"""
    row = con.execute("SELECT COUNT(*), COALESCE(SUM(bytes), 0) FROM result_cache").fetchone()
    return {"entries": int(row[0]), "bytes": int(row[1])}


def clear(con) -> int:  # noqa: ANN001
    count = int(con.execute("SELECT COUNT(*) FROM result_cache").fetchone()[0])
    con.execute("DELETE FROM result_cache")
    return count
