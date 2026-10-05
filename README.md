# ETF Portfolio Lab · 组合数值实验室

> **教育用途声明**：本项目是一个**教学与工具演示**项目，目的是帮助使用者理解收益、风险、对冲等
> 数字**是怎么算出来的、代表什么、在什么情况下会误导人**。
> 本站**不提供投资建议、不推荐任何标的、不构成任何收益承诺**。
> 所有回测与模拟结果都是**特定假设下的历史统计**，不代表未来表现。
> 项目维护者不是持牌投资顾问，使用本项目产生的任何决策及其后果由使用者自行承担。

---

## 在线访问

**<https://testiphi.github.io/etf-portfolio-lab/>** —— 路线 A 的静态站，零服务器、零备案。

也可以直接双击本仓库的 `docs/index.html` **离线打开**：图表数据外置成相对路径的
**脚本文件**（而不是用 `fetch()` 读 JSON），因此 `file://` 协议下同样能正常渲染。
首屏只装载当前档位的数据，切换组合时再按需加载。

---

面向 A 股指数 ETF 的组合诊断与数值实验工具：输入你持有的 ETF、仓位与定投计划，
得到历史上的年化收益区间、风险敞口、回撤特征、对冲方案成本、可加入的新板块以及调仓方向。

**匿名优先。** 三条路线里的**全部分析功能都不需要登录**；登录只做一件事——
把你自己配好的组合存下来（`data/users.duckdb`，与行情库分开）。匿名状态下不落盘任何一行。
账号口令用 PBKDF2-HMAC-SHA256 加盐存储，但这**不是一套加固过的账号系统**：
没有登录限流与口令找回，公网部署必须走 HTTPS。

---

**设计主张：默认视图只有数字。** 页面是深色密集仪表盘——指标键盘、净值与水下曲线、
收益归因（对数贡献）、权重 vs 风险贡献、回撤最深五段、定投三种收益率口径、滚动夏普、
各标的单独持有、RBSA 因子敞口矩阵、利率环境与无风险利率、久期与利率冲击、
历史波动率期限结构、保护成本与 Greeks、蒙特卡洛四模型对比、历史情节重放。

公式与「什么时候会骗人」藏在数字里的 `<details>` 后面，**点开才出现**；
而知识点的浮现由**你的组合算出来的数据**触发（例如「510300 占 50% 权重却贡献 94% 风险」
会自动冒出来，并给出成分 VaR 的解释）。配出特定结构还会解锁对应模块——
**含债券才解锁久期与利率冲击**，含跨境才解锁汇率贡献。
已实现模块与锁定模块在页面上都写明状态，**不假装覆盖**。

---

## 三条并行路线（本仓库同时实现）

| 路线 | 形态 | 运行方式 | 适用场景 |
|---|---|---|---|
| **A** | 静态交互报告站 | Python 引擎离线算出结果 → 生成自包含 Plotly HTML → GitHub Pages | 零服务器、零备案、发给别人看 |
| **B** | 浏览器内计算 | 同一份 `core/` 用 Pyodide 在浏览器里跑 | 永久零服务器但可交互重算（评估中） |
| **C** | 实算应用 | NiceGUI 单进程 + 计算/界面分离 + 进程池 + DuckDB | 真正改参数即时重算，部署在国内 ECS |

路线 A 已上线：**<https://testiphi.github.io/etf-portfolio-lab/>**（源在 `docs/`，推送 `main` 后自动部署）。

三条路线**共用同一个 `core/`**：这是本仓库最重要的设计约束。详见下方「架构契约」。

---

## 快速开始

```bash
# 1) 环境：用 uv 建虚拟环境并装依赖（国内直连 PyPI 会超时，用镜像）
uv venv .venv --python 3.12
uv pip install --python .venv\Scripts\python.exe numpy pandas scipy statsmodels duckdb plotly pyyaml requests pytest nicegui `
  --index-url https://mirrors.aliyun.com/pypi/simple/

#    未做 editable 安装也能直接跑（下面统一依赖 PYTHONPATH）：
#    PowerShell:  $env:PYTHONPATH='src'

# 2) 跑数值校验测试（这一步必须全绿，共 223 项）
.venv\Scripts\python.exe -m pytest -q

# 3) 采集数据到本地 DuckDB（生成 data/lab.duckdb，已 gitignore）
#    腾讯接口有限流，脚本内置 1 秒/次全局限速，一次完整采集需要几分钟
.venv\Scripts\python.exe -m etf_lab.cli fetch --preset core
#    国债收益率曲线（无风险利率与久期的来源），中债主源 + 新浪兜底
.venv\Scripts\python.exe -m etf_lab.cli fetch --preset macro --start 2015-01-01

#    可选：建一个账号用于保存组合（不建也能用全部功能，匿名不落盘）
.venv\Scripts\python.exe -m etf_lab.cli user add 你的用户名

# 4) 生成静态站（A 路线）→ docs/，直接用浏览器打开 docs/index.html
.venv\Scripts\python.exe -m etf_lab.cli build-site

# 5) 启动实算应用（C 路线）→ http://127.0.0.1:8080
.venv\Scripts\python.exe -m etf_lab.cli app

# 数据源可用性探测 / 入库数据交叉校验
.venv\Scripts\python.exe -m etf_lab.cli probe
.venv\Scripts\python.exe scripts\verify_stored_data.py
```

> **akshare 不是必需依赖**。主数据路径走 `etl/tencent.py` 的纯 HTTP 实现——akshare 依赖
> `py_mini_racer`（内含 V8 二进制），杀毒软件会误报甚至直接杀进程，本项目就被卡巴斯基
> 中断过一次。akshare 保留为可选的备用源（`pip install -e ".[data]"`）。

---

## 架构契约（改代码前必读）

多智能体/多人并行开发时，以下规则**不可违反**，否则模块会互相覆盖或结论不可复现：

1. **`core/` 是纯函数区**：不读数据库、不写文件、不 import 任何 UI、随机过程必须显式传 `seed`。
   取数由 `data/repo.py` 完成后把 DataFrame 传进来。
2. **`app/` 不做计算**：只收集输入、展示输出。任何超过约 1 秒的计算走 `services/` 的进程池。
3. **缓存键 = `sha256(规范化JSON(模块+函数+参数+data_version+code_version))`**，浮点统一舍入到 `1e-10`。
4. **不静默填充**：缺失值保留 `NaN` 并显式报告，禁止 `ffill`/`fillna(0)` 隐藏数据问题。
5. **数值改动必须附对照测试**：见下方「数值验收标准」。
6. **复权口径全项目固定一种**，跨数据源拼接复权价必错。

```
src/etf_lab/
├── core/          # 纯计算（本仓库的价值核心，可独立测试）
│   ├── returns.py     # 复权收益、净值曲线、费率
│   ├── metrics.py     # 年化/波动/回撤/Sharpe/Sortino/Calmar/VaR/CVaR
│   ├── dca.py         # 定投与 XIRR
│   ├── correlation.py # 相关性、成分 VaR、加入新板块的边际影响
│   ├── exposure.py    # RBSA 收益法敞口（约束回归）
│   ├── simulate.py    # 蒙特卡洛（四模型 + 收敛诊断）
│   ├── derivatives.py # 期权定价、Greeks、隐含波动率反解
│   ├── rates.py       # 无风险利率、久期回归、利率冲击情景
│   ├── episodes.py    # 历史情节重放（区间统计与覆盖度）
│   ├── optimize.py    # 组合优化（规划中）
│   └── hedge.py       # Delta-Gamma 复制与再平衡频率（Boyle–Emanuel 标度律）
├── data/          # DuckDB schema 与读写层（唯一允许碰数据库的地方）
├── etl/           # 采集：tencent（主）/ sohu（校验）/ fund_nav / bond_yield / eastmoney（备用）
├── reports/       # A 路线：静态站生成（计算与渲染分离）
├── app/           # C 路线：NiceGUI 界面
├── services/      # 缓存、进程池封装、可 pickle 的作业函数
├── content/       # 教学卡片文案（怎么算/说明什么/何时会误导）
├── tests/         # 数值对照测试（223 项，含采集解析与防呆）
└── cli.py         # 统一命令入口
```

---

## 数据来源与口径

| 数据 | 来源 | 口径说明 |
|---|---|---|
| ETF 行情 | 腾讯公开接口（主源） | **前复权**用于一切收益计算；同时保留未复权价与复权因子 |
| ETF 行情校验 | 搜狐公开接口（未复权） | 逐日交叉比对，中位差异为 0 |
| 指数行情 | 腾讯公开接口 | 指数不可直接交易，用于补 ETF 上市时间短的样本不足 |
| 基金净值 | 天天基金（`f10/lsjz`）+ 新浪兜底 | 折溢价 = 未复权收盘价 / 单位净值 − 1；QDII 净值有披露滞后 |
| 国债收益率曲线 | **中债官方（主源）** + 新浪（兜底） | 8 个期限、2015 年至今 23,496 行。无风险利率取 1 年期（实测 1.2197%），**页面显示所用值**；久期由收益对收益率变动的回归反推 |
| 汇率 | 待接入 | 跨境 ETF 的汇率贡献必须单列 |
| 股指期货 | 待接入 | 基差/贴水是对冲成本的核心 |
| ETF 期权 | 待接入 | 拿不到完整期权链时退化为「历史波动率下的理论定价 + Greeks 演示」 |

数据源实测细节（谁可用、谁被限流、复权口径如何验证）见 [ARCHITECTURE.md](ARCHITECTURE.md) §2。

**指数成分股不做穿透**：免费源缺少可靠的历史时点成分名单，本项目改用
**收益法风格分析（RBSA，Sharpe 1992）**——用宽基与行业指数的收益做约束回归来估计敞口，
并如实标注这是近似。

---

## 数值验收标准（CI 必须通过）

| 模块 | 门槛 |
|---|---|
| 期权定价 | BS 对齐标准算例；Greeks 与有限差分误差 < 1e-4；IV 反解残差 < 1e-6 |
| 蒙特卡洛 | 欧式看涨 MC 价格与 BS 闭式解一致；标准误按 1/√N 收敛 |
| 回测引擎 | 已知构造序列（恒定收益、单次分红）偏差为 0 |
| 定投 | XIRR 用已知现金流算例校验 |
| 优化 | 权重和为 1、约束生效、极端输入（单资产/全相关/缺失）不崩 |

---

## 路线图

- [x] M0-1 仓库骨架、契约冻结、`core/` 最小集（收益/指标/定投/相关性）
- [x] M0-2 多源采集 → DuckDB：7 只 ETF 23,494 行 + 指数 35,157 行 + 基金净值 11,529 行 + 国债收益率 23,496 行
- [x] A-1 静态站生成器 + 三个示例组合 + 概念页 + 口径页（`docs/`），已上线 GitHub Pages
- [x] C-1 NiceGUI 应用 + 进程池 + 缓存骨架（实测 `cpu_bound` 生效）
- [x] B-1 Pyodide 可行性评估（结论见 [ARCHITECTURE.md](ARCHITECTURE.md) §5）
- [x] M3 蒙特卡洛：四个模型并排 + 收敛诊断 + 1% 分位
- [x] M4 期权 Greeks + 保护成本 + 历史波动率期限结构
- [x] M5 RBSA 敞口矩阵 + 加入板块的边际影响
- [x] M6 登录保存组合（**匿名可用全部功能，登录只用于保存**；账号与组合存在独立的
      `data/users.duckdb`，与只读的行情库分开——原因见 ARCHITECTURE §10）
- [ ] M7 压力测试（把历史情节推广到自定义冲击）
- [ ] M7 组合优化（`core/optimize.py`，尚未实现）
- [x] M8 Delta-Gamma 复制与再平衡频率（`core/hedge.py`）：误差 ∝ √Δt、成本 ∝ 1/Δt，
      模拟与 Boyle–Emanuel 解析预期的比值 1.06~1.15，最优频率随成本水平移动
- [ ] M9 数据补全：股指期货基差、ETF 期权行情（把 Greeks 与认沽定价从理论值换成隐含波动率）、汇率
- [x] 数据补全：国债收益率曲线（无风险利率不再靠假设）

**决策依据、数据源实测结论与已修复缺陷的留档见 [ARCHITECTURE.md](ARCHITECTURE.md)。**

---

## 许可证

MIT，见 [LICENSE](LICENSE)。**行情数据不随仓库分发**，请用采集脚本自行获取并遵守数据源条款。
