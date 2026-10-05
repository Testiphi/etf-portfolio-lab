"""深色密集仪表盘的样式与展示基元。

设计原则（对应产品定位）
------------------------
1. **默认视图零解释文字**：页面上只出现数字、表格、图；公式与「何时会骗人」
   一律藏在 ``<details>`` 里，点击就地展开，不跳页。
2. **密集**：12 栅格 + 指标键盘，信息密度优先于留白。
3. **深色**：低亮度面板 + 高对比数字，长时间看不累。

用 ``<details>/<summary>`` 而不是 JS 是有意的：静态站（路线 A）无需任何脚本即可交互，
离线打开也照常工作。
"""

from __future__ import annotations

import html
from typing import Any, Mapping

import plotly.graph_objects as go

from etf_lab.content import teaching

# 深色配色。低饱和背景 + 高对比文字，避免长时间阅读疲劳。
PALETTE = {
    "bg": "#0e1116",
    "panel": "#151a22",
    "panel_alt": "#1b2130",
    "line": "#252c3a",
    "fg": "#d8dee9",
    "muted": "#8b95a7",
    "accent": "#4da3ff",
    "warn": "#ff7a59",
    "ok": "#3ddc97",
    "gold": "#e8c07d",
    "purple": "#b98cff",
}

SERIES_COLORS = [PALETTE["accent"], PALETTE["ok"], PALETTE["gold"], PALETTE["purple"], PALETTE["warn"], "#5ec8d8"]

STYLE = """
:root {
  --bg:#0e1116; --panel:#151a22; --panel-alt:#1b2130; --line:#252c3a;
  --fg:#d8dee9; --muted:#8b95a7; --accent:#4da3ff; --warn:#ff7a59; --ok:#3ddc97; --gold:#e8c07d;
  --mono: ui-monospace, "Cascadia Mono", Consolas, monospace;
}
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:var(--fg); line-height:1.55;
  font-family: -apple-system, "Segoe UI", "Microsoft YaHei", sans-serif; font-size:14px; }
a { color:var(--accent); text-decoration:none; }
a:hover { text-decoration:underline; }
code { font-family:var(--mono); background:var(--panel-alt); padding:1px 5px; border-radius:4px; font-size:12px; }

.site-header { display:flex; align-items:center; justify-content:space-between;
  padding:10px 18px; background:var(--panel); border-bottom:1px solid var(--line);
  position:sticky; top:0; z-index:20; }
.brand { font-weight:700; color:var(--fg); letter-spacing:.3px; }
.site-header nav a { margin-left:14px; color:var(--muted); font-size:13px; }
.site-header nav a:hover { color:var(--accent); }

main { max-width:1500px; margin:0 auto; padding:16px 18px 60px; }
h1 { font-size:20px; margin:0 0 4px; }
h2 { font-size:14px; margin:0; font-weight:600; color:var(--fg); }
.sub { color:var(--muted); font-size:12px; }

.titlebar { display:flex; flex-wrap:wrap; gap:10px; align-items:baseline; justify-content:space-between;
  padding:12px 14px; background:var(--panel); border:1px solid var(--line); border-radius:8px; margin-bottom:12px; }
.titlebar .q { color:var(--muted); font-size:13px; }

.grid { display:grid; grid-template-columns:repeat(12, minmax(0,1fr)); gap:12px; }
.span-3 { grid-column: span 3; } .span-4 { grid-column: span 4; }
.span-6 { grid-column: span 6; } .span-8 { grid-column: span 8; }
.span-12 { grid-column: span 12; }
@media (max-width: 1000px) { .span-3,.span-4,.span-6,.span-8 { grid-column: span 12; } }

.panel { background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:10px 12px; min-width:0; }
.panel > header { display:flex; align-items:baseline; justify-content:space-between; gap:8px; margin-bottom:8px; }

/* 指标键盘：每格是可以点开的 <details>，展开后是公式与陷阱。
   注意选择器写成 .tile .label 而不是 details.tile .label——
   没有关联知识卡片的那几格是普通 <div>，早期版本用 details 前缀选择器导致
   「偏度/超额峰度/样本交易日」三格的标签与数值变成行内元素、样式全丢。 */
.tiles { display:grid; grid-template-columns:repeat(auto-fit, minmax(146px,1fr)); gap:8px; }
.tile { background:var(--panel-alt); border:1px solid var(--line); border-radius:7px; min-width:0; }
.tile.static { padding:8px 10px; }
details.tile > summary { list-style:none; cursor:pointer; padding:8px 10px; }
details.tile > summary::-webkit-details-marker { display:none; }
.tile .label { display:block; color:var(--muted); font-size:11.5px; }
.tile .val { display:block; font-family:var(--mono); font-size:18px; font-weight:600; }
details.tile[open] { border-color:var(--accent); }
details.tile .body { padding:0 10px 10px; color:var(--muted); font-size:12px; border-top:1px dashed var(--line); }
.hintmark { color:var(--accent); font-size:11px; }
.val.warn-val { color:var(--warn); }

/* 洞察条：数据触发时出现，点开才是知识 */
.insights { display:grid; gap:8px; }
details.insight { background:var(--panel-alt); border-left:3px solid var(--accent); border-radius:6px; }
details.insight > summary { cursor:pointer; padding:8px 12px; font-size:13px; }
details.insight.warn { border-left-color:var(--warn); }
details.insight .body { padding:0 12px 10px; color:var(--muted); font-size:12.5px; }

/* 解锁清单 */
.unlocks { display:grid; grid-template-columns:repeat(auto-fit,minmax(230px,1fr)); gap:8px; }
.unlock { background:var(--panel-alt); border:1px solid var(--line); border-radius:7px; padding:8px 10px; font-size:12.5px; }
.unlock.on { border-color:var(--ok); }
.unlock .t { font-weight:600; color:var(--fg); }
.unlock .s { color:var(--muted); }
.badge { display:inline-block; font-size:11px; padding:1px 6px; border-radius:10px; border:1px solid var(--line); color:var(--muted); }
.badge.on { color:var(--ok); border-color:var(--ok); }
.badge.pending { color:var(--gold); border-color:var(--gold); }

/* 表格 */
table { width:100%; border-collapse:collapse; font-family:var(--mono); font-size:12.5px; }
th, td { padding:5px 8px; text-align:right; border-bottom:1px solid var(--line); white-space:nowrap; }
th:first-child, td:first-child { text-align:left; font-family:inherit; }
thead th { color:var(--muted); font-weight:500; font-size:11.5px; text-transform:uppercase; letter-spacing:.3px; }
tbody tr:hover { background:var(--panel-alt); }
td.warn, .warn-text { color:var(--warn); }
td.ok { color:var(--ok); }

/* 卡片（点开后展开的知识） */
.card { background:var(--panel-alt); border:1px solid var(--line); border-radius:6px; padding:10px; margin-top:8px; }
.card h3 { margin:0 0 6px; font-size:13px; }
.card .formula { font-family:var(--mono); background:var(--bg); border:1px solid var(--line);
  padding:5px 7px; border-radius:5px; font-size:12px; color:var(--gold); overflow-x:auto; }
.card p { margin:6px 0 0; }
.card .mislead { color:var(--warn); }

ul.presets { list-style:none; padding:0; margin:0; display:grid; gap:8px; }
ul.presets li { background:var(--panel-alt); border:1px solid var(--line); border-radius:7px; padding:10px 12px; display:flex; justify-content:space-between; gap:12px; align-items:baseline; }
ul.presets .metrics { font-family:var(--mono); color:var(--muted); font-size:12.5px; white-space:nowrap; }

footer { border-top:1px solid var(--line); background:var(--panel); color:var(--muted);
  padding:14px 18px; font-size:12px; }
footer .disclaimer { max-width:1500px; margin:0 auto 4px; }
footer .meta { max-width:1500px; margin:0 auto; }
.note { color:var(--muted); font-size:12px; margin-top:8px; }

/* 档位切换：纯 CSS（radio + 兄弟选择器），无 JS。
   目的是"一进页面数据就摆在面前"，切组合不刷新、不跳页。 */
.gear-input { position:absolute; opacity:0; pointer-events:none; }
.gear-labels { display:flex; gap:6px; flex-wrap:wrap; align-items:center; margin-bottom:12px; }
.gear-labels label { padding:5px 14px; border:1px solid var(--line); border-radius:14px;
  cursor:pointer; color:var(--muted); font-size:13px; user-select:none; }
.gear-labels label:hover { border-color:var(--accent); color:var(--fg); }
.gear-custom { margin-left:auto; }
.gear-custom summary { list-style:none; cursor:pointer; padding:5px 14px; border:1px dashed var(--line);
  border-radius:14px; color:var(--muted); font-size:13px; }
.gear-custom summary::-webkit-details-marker { display:none; }
.gear-custom[open] summary { border-color:var(--accent); color:var(--accent); }
.gear-pane { display:none; }
"""


def esc(text: Any) -> str:
    return html.escape(str(text))


def pct(value: Any, digits: int = 2) -> str:
    if value is None:
        return "—"
    try:
        return f"{float(value) * 100:.{digits}f}%"
    except (TypeError, ValueError):
        return "—"


def num(value: Any, digits: int = 2) -> str:
    if value is None:
        return "—"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "—"


def bp(value: Any) -> str:
    """基点显示：收益归因用 bp 比 % 更能看出差别。"""
    if value is None:
        return "—"
    try:
        return f"{float(value) * 10000:,.0f} bp"
    except (TypeError, ValueError):
        return "—"


def dark(fig: go.Figure, height: int | None = None) -> go.Figure:
    """给图表套上统一深色样式。"""
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": PALETTE["fg"], "size": 12},
        margin={"l": 54, "r": 20, "t": 40, "b": 34},
        legend={"orientation": "h", "y": 1.08, "x": 0, "bgcolor": "rgba(0,0,0,0)"},
        hoverlabel={"bgcolor": PALETTE["panel_alt"], "font": {"color": PALETTE["fg"]}},
        title={"font": {"size": 13, "color": PALETTE["muted"]}},
    )
    fig.update_xaxes(gridcolor=PALETTE["line"], zerolinecolor=PALETTE["line"])
    fig.update_yaxes(gridcolor=PALETTE["line"], zerolinecolor=PALETTE["line"])
    if height:
        fig.update_layout(height=height)
    return fig


def card_html(key: str) -> str:
    """一张知识卡片（展开后显示）。"""
    card = teaching.card(key)
    return (
        f'<div class="card">'
        f'<h3>{esc(card["title"])}</h3>'
        f'<div class="formula">{esc(card["formula"])}</div>'
        f'<p><strong>说明什么：</strong>{esc(card["means"])}</p>'
        f'<p class="mislead"><strong>什么时候会骗人：</strong>{esc(card["misleads"])}</p>'
        f"</div>"
    )


def metric_tile(label: str, value: str, card: str | None = None, note: str | None = None, warn: bool = False) -> str:
    """指标键盘里的一格：默认只有标签与数值，点开才看到公式与陷阱。

    没有关联知识卡片时渲染成普通 ``<div class="tile static">``——
    结构与有卡片的一致，样式因此不会丢（本项目踩过这个坑）。
    """
    mark = '<span class="hintmark"> ◂</span>' if card else ""
    value_class = "val warn-val" if warn else "val"
    head = f'<span class="label">{esc(label)}{mark}</span><span class="{value_class}">{esc(value)}</span>'
    if not card:
        return f'<div class="tile static">{head}</div>'
    body = card_html(card)
    if note:
        body += f'<p class="note">{esc(note)}</p>'
    return f'<details class="tile"><summary>{head}</summary><div class="body">{body}</div></details>'


def insight_html(insight: Mapping[str, Any], cards_lookup: Any = None) -> str:
    """一条洞察条：标题是结论，展开是知识与证据数字。"""
    level = str(insight.get("level", "info"))
    evidence = insight.get("evidence") or {}
    evidence_html = ""
    if evidence:
        rows = "".join(
            f"<tr><td>{esc(k)}</td><td>{esc(_format_evidence(v))}</td></tr>" for k, v in evidence.items()
        )
        evidence_html = f'<table><tbody>{rows}</tbody></table>'
    card_key = str(insight.get("card", ""))
    card_block = card_html(card_key) if card_key else ""
    return (
        f'<details class="insight {esc(level)}"><summary>{esc(insight.get("title", ""))}</summary>'
        f'<div class="body">{card_block}{evidence_html}</div></details>'
    )


def _format_evidence(value: Any) -> str:
    if isinstance(value, float):
        if abs(value) < 1.5 and value != 0:
            return f"{value:.4f}"
        return f"{value:,.2f}"
    return str(value)


def panel(title: str, body: str, *, span: int = 6, subtitle: str | None = None) -> str:
    """一个仪表盘面板。"""
    sub = f'<span class="sub">{esc(subtitle)}</span>' if subtitle else ""
    return (
        f'<section class="panel span-{span}"><header><h2>{esc(title)}</h2>{sub}</header>'
        f"<div>{body}</div></section>"
    )


def figure_html(fig: go.Figure, div_id: str) -> str:
    return fig.to_html(full_html=False, include_plotlyjs=False, div_id=div_id, config={"displaylogo": False, "responsive": True})
