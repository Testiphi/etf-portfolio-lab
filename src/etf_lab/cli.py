"""命令行入口：``python -m etf_lab.cli <命令>``。

命令
----
``probe``      探测各数据源接口是否可用（M0 证伪步骤）
``fetch``      采集数据到本地 DuckDB
``info``       打印数据底座概况
``build-site`` 生成静态站到 docs/（路线 A）
``app``        启动 NiceGUI 实算应用（路线 C）
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from etf_lab import __version__


def _print_json(payload: object) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))


def cmd_probe(args: argparse.Namespace) -> int:
    from etf_lab.etl import fetch

    results = fetch.probe_sources()
    ok = sum(1 for r in results if r.get("ok"))
    _print_json(results)
    print(f"\n可用 {ok}/{len(results)} 个接口")
    return 0 if ok else 1


def cmd_fetch(args: argparse.Namespace) -> int:
    from etf_lab.data import repo
    from etf_lab.etl import fetch

    con = repo.connect(args.db)
    reports = []
    if args.preset in ("core", "all"):
        reports += fetch.fetch_etf_prices(con, start=args.start, verify=not args.no_verify)
        reports += fetch.fetch_index_prices(con, start=args.start)
    if args.preset in ("nav", "all"):
        # 净值只用于折溢价，不参与收益计算；服务端每页仅 20 条，窗口不宜过大
        reports += fetch.fetch_fund_nav(con, start=args.nav_start)
    if args.preset in ("macro", "all"):
        reports += fetch.fetch_bond_yields(con, start=args.start)
        for pending in fetch.PENDING_SOURCES:
            print(f"[待接入] {pending}")

    failed = [r for r in reports if not r.ok]
    warned = [r for r in reports if r.quality and r.quality.get("warning")]
    _print_json(fetch.report_to_dicts(reports))
    counts = repo.table_counts(con)
    print("\n数据底座：", json.dumps(counts, ensure_ascii=False))
    version = repo.log_data_version(con, source="tencent+sohu(verify)", notes=f"preset={args.preset}, start={args.start}")
    print(f"data_version = {version}")
    if warned:
        print(f"\n有 {len(warned)} 项未通过交叉校验（见 quality.warning）", file=sys.stderr)
    if failed:
        print(f"\n有 {len(failed)} 项失败（已如实记录，未做填充）", file=sys.stderr)
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    from etf_lab.data import repo

    con = repo.connect(args.db, read_only=False)
    counts = repo.table_counts(con)
    _print_json({"db": str(args.db or repo.DEFAULT_DB_PATH), "data_version": repo.latest_data_version(con), "counts": counts})
    return 0


def cmd_build_site(args: argparse.Namespace) -> int:
    from etf_lab.reports import static_site

    out = static_site.build(out_dir=Path(args.out), db_path=args.db)
    print(f"静态站已生成：{out}")
    return 0


def cmd_app(args: argparse.Namespace) -> int:
    from etf_lab.app import main as app_main

    app_main.run(host=args.host, port=args.port, db_path=args.db)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="etf-lab", description="ETF 组合数值实验室（教育用途，非投资建议）")
    parser.add_argument("--version", action="version", version=f"etf-portfolio-lab {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_probe = sub.add_parser("probe", help="探测数据源接口可用性")
    p_probe.set_defaults(func=cmd_probe)

    p_fetch = sub.add_parser("fetch", help="采集数据到本地 DuckDB")
    p_fetch.add_argument("--preset", choices=["core", "nav", "macro", "all"], default="core")
    p_fetch.add_argument("--start", default="2012-01-01")
    p_fetch.add_argument("--nav-start", default="2022-01-01", help="净值抓取起点（净值仅用于折溢价）")
    p_fetch.add_argument("--db", default=None)
    p_fetch.add_argument("--no-verify", action="store_true", help="跳过与校验源的收盘价交叉校验（不建议）")
    p_fetch.set_defaults(func=cmd_fetch)

    p_info = sub.add_parser("info", help="打印数据底座概况")
    p_info.add_argument("--db", default=None)
    p_info.set_defaults(func=cmd_info)

    p_site = sub.add_parser("build-site", help="生成静态站（路线 A）")
    p_site.add_argument("--out", default="docs")
    p_site.add_argument("--db", default=None)
    p_site.set_defaults(func=cmd_build_site)

    p_app = sub.add_parser("app", help="启动 NiceGUI 应用（路线 C）")
    p_app.add_argument("--host", default="127.0.0.1")
    p_app.add_argument("--port", type=int, default=8080)
    p_app.add_argument("--db", default=None)
    p_app.set_defaults(func=cmd_app)

    p_user = sub.add_parser("user", help="管理账号（账号只用于保存组合）")
    user_sub = p_user.add_subparsers(dest="action", required=True)
    p_user_add = user_sub.add_parser("add", help="新建账号（口令交互输入，避免留在命令历史里）")
    p_user_add.add_argument("username")
    p_user_add.add_argument("--password", default=None, help="不推荐：会留在命令历史里")
    p_user_list = user_sub.add_parser("list", help="列出已有账号")
    p_user_list.set_defaults(func=cmd_user_list)
    p_user_add.set_defaults(func=cmd_user_add)

    return parser


def cmd_user_add(args) -> int:
    """新建账号。口令默认交互输入——写在命令行参数里会留在 shell 历史中。"""
    import getpass

    from etf_lab.data import repo
    from etf_lab.services import auth

    password = getattr(args, "password", None) or getpass.getpass("设置口令：")
    con = repo.connect_users()
    try:
        name = auth.register(con, args.username, password)
    except auth.AuthError as exc:
        print(f"失败：{exc}")
        return 1
    finally:
        con.close()
    print(f"已创建账号 {name}（存在 data/users.duckdb，与行情库分开）")
    print("它只用于保存组合；所有分析功能匿名即可使用。")
    return 0


def cmd_user_list(args) -> int:
    from etf_lab.data import repo

    if not repo.DEFAULT_USERS_DB_PATH.exists():
        print("还没有用户库，也没有任何账号。用 python -m etf_lab.cli user add <用户名> 创建。")
        return 0
    con = repo.connect_users(read_only=True)
    try:
        rows = con.execute("SELECT username, created_at FROM users ORDER BY username").fetchall()
    finally:
        con.close()
    if not rows:
        print("还没有任何账号。用 python -m etf_lab.cli user add <用户名> 创建。")
        return 0
    for username, created_at in rows:
        print(f"  {username}  创建于 {str(created_at)[:19]}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
