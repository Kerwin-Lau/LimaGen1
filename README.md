# Lima_Gen1

Lima_Gen1 是一个面向个人量化研究的 Windows 本地项目，用于构建从行情数据获取、技术指标计算、交易信号扫描、策略回测到策略因子权重优化的一体化工作流。

项目当前覆盖 A 股和港股，后续计划扩展美股支持。美股模块目前处于开发中，具体目录和实现路径 TBD。

> 本项目主要用于个人量化研究、数据分析和历史回测，不构成任何投资建议。

## 主要功能

项目当前包含以下功能：

1. 从 AKShare 获取 A 股和港股的最新行情数据。
2. 保存逐股票日线历史数据，并支持增量更新和周期性重构。
3. 计算常见技术指标以及个人定制的 Z 家战法指标。
4. 按交易日扫描 A 股和港股交易信号，生成 Excel 股票清单供人工查看。
5. 扫描指定时间区间内 A 股股票的涨幅榜单。
6. 对 A 股交易策略进行历史回测并生成回测报告。
7. 通过“强化学习优化”模块对策略因子权重进行优化；当前底层实现采用 CMA-ES 黑盒优化方法。

## 项目工作流

项目的主要数据流如下：

```text
股票清单
   │
   ▼
AKShare 行情接口
   │
   ▼
01_Database：原始日线/周线数据
   │
   ▼
02_DataProcess：数据更新、指标计算、周线股票池和区间涨幅扫描
   │
   ├──────────────► 03_SignalScan：每日交易信号扫描与 Excel 报告
   │                              │
   │                              ▼
   │                       人工查看候选股票
   │
   └──────────────► 05_BackTest：历史回测
                                  │
                                  ▼
                         06_RL：策略因子权重优化
```

其中，`04_Strategy` 抽取并封装了信号扫描中的交易信号生成逻辑，使策略逻辑可以被信号扫描和回测模块复用。

## 目录结构

```text
Lima_Gen1/
├── 01_Database/
│   ├── 01_Ashares/
│   │   ├── ASharesList.xlsx
│   │   └── 01_RawData-Daily/       # A 股逐股票日线 CSV
│   └── 02_Hshares/
│       ├── HSharesList.xlsx
│       └── 01_RawData-Daily/       # 港股逐股票日线 CSV
│
├── 02_DataProcess/
│   ├── 01_Ashares/
│   │   ├── 01_DaliyUpdate/         # A 股日线数据更新与指标计算
│   │   ├── 02_WeeklyUpdate/        # A 股周线更新与周线股票池筛选
│   │   └── 03_SpecifiedPeriodScan/ # 指定区间涨幅榜单扫描
│   └── 02_Hshares/
│       ├── 01_DaliyUpdate/         # 港股日线数据更新
│       └── 02_WeeklyUpdate/        # 港股周线更新与股票池筛选
│
├── 03_SignalScan/
│   ├── 01_Ashares/01_Z-Strategy/   # A 股 Z 策略信号扫描与报告
│   └── 02_Hshares/01_Z-Strategy/   # 港股 Z 策略信号扫描与报告
│
├── 04_Strategy/
│   └── 01_Ashares/                 # 策略接口、信号逻辑和综合评分
│
├── 05_BackTest/
│   ├── 00_History/                 # 历史脚本和回测结果
│   └── 01_Ashares/20260615/        # 当前 A 股回测入口
│
└── 06_RL/
    └── 01_Ashares/01_Alpha/        # 策略权重优化和训练产物
```

## 数据源

项目目前只使用 AKShare 作为数据接口，A 股和港股行情均从 AKShare 获取。

AKShare 官方数据接口文档：

<https://akshare.akfamily.xyz/data/index.html>

AKShare 是外部网络数据源，脚本运行依赖网络连接和远程接口的可用性。接口返回字段、数据更新时间、访问限制和网络状况都可能影响数据更新结果，因此运行后应检查日志和生成的数据文件。

## 数据文件与更新策略

### 股票清单

A 股和港股分别使用股票清单文件：

```text
01_Database/01_Ashares/ASharesList.xlsx
01_Database/02_Hshares/HSharesList.xlsx
```

股票代码通常位于 Excel 首列，数据更新脚本会对代码进行清洗和补齐。

### 日线数据

日线数据以每只股票一个 CSV 文件保存，基础字段通常包括：

```text
date, open, high, low, close, volume
```

在数据处理过程中，会继续追加技术指标和策略特征字段。

### 每周数据重构流程

虽然日线更新脚本支持增量更新，但项目每周会主动重构日线历史数据，原因包括：

- 配股、分红等公司行为可能导致前复权历史行情发生变化；
- 市场可能有新股发行；
- 股票可能被标记为 ST 或发生其他股票状态变化；
- 需要根据最新股票清单和市场状态重新建立数据基础。

推荐的每周流程是：

1. 清空以下两个目录中的原始日线数据：

   ```text
   01_Database/01_Ashares/01_RawData-Daily
   01_Database/02_Hshares/01_RawData-Daily
   ```

2. 更新以下股票清单文件：

   ```text
   01_Database/01_Ashares/ASharesList.xlsx
   01_Database/02_Hshares/HSharesList.xlsx
   ```

3. 运行日线数据更新脚本：

   ```text
   02_DataProcess/01_Ashares/01_DaliyUpdate/DataUpdate_ASharesDaily.py
   02_DataProcess/02_Hshares/01_DaliyUpdate/DataUpdate_HSharesDaily.py
   ```

4. 获取从 2020 年 1 月至本周最后一个交易日的个股日线历史数据。

5. 运行周线更新脚本，生成处于多头趋势的股票清单：

   ```text
   02_DataProcess/01_Ashares/02_WeeklyUpdate/DataUpdate_ASharesWeekly.py
   02_DataProcess/02_Hshares/02_WeeklyUpdate/DataUpdate_HSharesWeekly.py
   ```

   后续一周的信号扫描只使用本周周线扫描得到的股票池。

## 技术指标与策略特征

数据处理模块会计算常见技术指标，包括 KDJ、BBI、MACD 等，也会计算项目定制的策略特征。

| 字段 | 含义 |
|---|---|
| `K`、`D`、`J` | KDJ 指标，其中 J 值用于 Z 策略的超买超卖和反转信号 |
| `BBI`、`BBI_DIF` | 多周期均线综合指标及其变化，用于判断趋势 |
| `DIF`、`DEA`、`MACD` | MACD 指标 |
| `Short_LS` | Z 家战法黄线；由多组短中期均线综合得到 |
| `Short_Trend` | Z 家战法白线；由短期 EMA 平滑得到 |
| `Brick_High` / `Brick_Low` | Z 家战法砖形图指标，用于识别砖形图方向变化 |
| `L2` / `L1` / `M` / `H1` / `H2` | PE Band 估值区间，用于识别股价所处的估值位置 |
| `short_term_fund` | Z 家战法补票战法的短期资金指标 |
| `long_term_fund` | Z 家战法补票战法的长期资金指标 |

### KDJ

项目实现了与通达信类公式一致的 KDJ 计算逻辑。默认使用 `N=9`、`M1=3`、`M2=3`，先计算 N 日 RSV，再使用通达信 SMA 平滑逻辑递推 K、D，最后计算：

```text
J = 3 × K - 2 × D
```

### BBI

BBI 使用 3、6、12、24 日简单移动平均线的均值：

```text
BBI = mean(MA3, MA6, MA12, MA24)
BBI_DIF = BBI.diff()
```

### Short_LS 与 Short_Trend

`Short_LS` 使用 14、28、57、114 日均线的平均值，作为 Z 家战法中的黄线。`Short_Trend` 基于 10 日 EMA 的二次平滑结果，作为 Z 家战法中的白线。两条线还用于识别短线金叉相关信号。

### 砖形图指标

项目根据高低价滚动区间和通达信 SMA 平滑逻辑计算砖形图数值，并将当日值与前一交易日值分别保存为 `Brick_High` 和 `Brick_Low`。策略通过比较两者的相对位置识别红砖等信号。

### PE Band

当数据中存在 `pe_ttm` 时，项目基于 PE 数据的最小值、中位数和分段间隔构造多个估值区间：`L2`、`L1`、`M`、`H1`、`H2`。策略使用这些区间识别股价处于低估、合理或高估位置的状态。

### 补票资金指标

`short_term_fund` 基于短期滚动最高价和最低价计算价格在短期区间中的位置；`long_term_fund` 使用更长周期的滚动区间进行相同类型的归一化计算。两者用于识别补票 P1/P2 等策略信号。

## 每个交易日的信号扫描

每个交易日收盘后，分别运行：

```text
03_SignalScan/01_Ashares/01_Z-Strategy/SignalScan_Ashares_Z-Strategy.py
03_SignalScan/02_Hshares/01_Z-Strategy/SignalScan_Hshares_Z-Strategy.py
```

扫描结果分别输出到：

```text
03_SignalScan/01_Ashares/01_Z-Strategy/01_Report
03_SignalScan/02_Hshares/01_Z-Strategy/01_Report
```

报告通常包含股票基本信息、行情和指标数据、策略信号状态以及综合评分等信息，当前主要用于人工查看，不会被后续脚本自动读取。

信号扫描前，脚本会优先使用周线更新阶段产生的最新股票池，因此每周股票池重构会影响下一周的日常扫描范围。

## 指定时间区间涨幅扫描

脚本：

```text
02_DataProcess/01_Ashares/03_SpecifiedPeriodScan/SpecifiedPeriodScan.py
```

该工具用于扫描指定时间区间内的 A 股股票，计算区间涨幅并生成涨幅榜单，适合进行阶段性行情分析和股票表现筛选。它与每日交易信号扫描是两个独立功能：前者关注指定区间的涨幅排名，后者关注当前交易日是否产生策略信号。

## 策略模块

策略核心位于：

```text
04_Strategy/01_Ashares/Z-Strategy.py
```

该模块将信号生成和综合评分逻辑从信号扫描脚本中抽取出来，使策略逻辑可以在信号扫描和回测中复用。核心设计包括：

- `StrategyBase`：策略基类，定义 `generate_signals` 和 `compute_score` 接口；
- `ZStrategy`：当前 Z 家战法的具体实现；
- `ZWeights`：集中管理策略因子权重，并提供权重向量注入和字典转换能力。

当前策略涉及的信号类型包括 J 值负值与反转、补票 P1/P2、BBI 趋势、L1/L2 价位、突破确认、短线下跌未破位、砖形图红砖、短线金叉、创新高以及成交量异动等。

策略综合评分由多个因子按权重加权得到。`ZWeights` 的权重可以由 `06_RL` 模块优化后重新注入策略。

## A 股回测

目前回测仅实现 A 股，主要入口为：

```text
05_BackTest/01_Ashares/20260615/AShares_BackTest_20260615.py
```

回测模块会加载历史日线数据和股票池，复用策略信号与评分逻辑，并使用向量化回测框架执行历史模拟。回测报告中的具体指标和输出内容以该脚本的报告生成逻辑为准，通常包括资产净值、交易统计、收益表现和风险指标等。

历史回测只反映特定历史区间、股票池、参数和交易假设下的模拟结果，不代表未来表现。

## 强化学习优化模块

`06_RL` 用于强化学习优化策略因子权重。当前实现采用 CMA-ES 作为底层黑盒优化方法：

1. 从默认策略权重开始；
2. 在约束范围内生成多组候选权重；
3. 将每组权重注入 Z 策略并运行回测；
4. 使用回测最终资产净值作为评价目标；
5. 通过训练集、验证集和测试集比较权重表现；
6. 保存最优权重、训练历史、检查点和日志。

主要脚本：

```text
06_RL/01_Ashares/01_Alpha/RL_Alpha_AShares.py
```

该模块会调用回测引擎进行批量评估，并支持多进程并行以提高训练效率。重新训练后，应检查生成的权重文件是否与策略模块当前的字段顺序和版本保持一致，并在正式使用前进行独立验证。

## 安装与环境

项目目前没有统一的 `requirements.txt`、`environment.yml` 或其他环境配置文件。建议使用 Windows 本地 Python 环境，并根据项目代码实际导入的库逐项安装和验证。

代码中涉及的主要依赖包括：

```text
akshare
pandas
numpy
TA-Lib
tqdm
python-dateutil
urllib3
psutil
openpyxl
vectorbt
cma
```

不同脚本可能还会间接依赖这些库的底层依赖。首次运行时，如果出现 `ModuleNotFoundError`，需要根据报错安装对应模块。

建议按以下顺序验证环境：

1. 验证 Python 和 pip 可用；
2. 安装代码实际使用的依赖；
3. 验证 `akshare` 可以访问接口；
4. 准备 A 股和港股股票清单文件；
5. 先运行数据更新脚本，再运行周线更新和信号扫描脚本；
6. 最后按需运行回测或权重优化模块。

## 运行顺序总结

### 每周运行

```text
1. 更新 ASharesList.xlsx 和 HSharesList.xlsx
2. 清空 A 股、港股日线原始数据目录
3. DataUpdate_ASharesDaily.py
4. DataUpdate_HSharesDaily.py
5. DataUpdate_ASharesWeekly.py
6. DataUpdate_HSharesWeekly.py
```

### 每个交易日收盘后运行

```text
1. SignalScan_Ashares_Z-Strategy.py
2. SignalScan_Hshares_Z-Strategy.py
3. 查看 A 股和港股 01_Report 目录下生成的 Excel 报告
```

### 按需运行

```text
指定区间涨幅扫描：SpecifiedPeriodScan.py
A 股历史回测：AShares_BackTest_20260615.py
策略权重优化：RL_Alpha_AShares.py
```

## 已知限制

- 当前依赖 AKShare 外部网络接口，接口不可用或字段变化时可能导致数据更新失败。
- 项目主要按照 Windows 本机环境编写，代码中存在绝对路径，迁移到其他电脑前需要统一调整路径配置。
- 当前没有统一的依赖清单和环境锁定文件。
- 每周需要重构前复权日线历史数据，以处理公司行为和股票池变化。
- 信号扫描结果当前主要供人工查看，尚未形成自动化交易执行链路。
- 回测目前只实现 A 股，港股回测尚未完成。
- 美股支持处于开发中，具体路径 TBD。
- 不同历史版本脚本和策略权重可能存在差异，运行时应确认调用的是当前版本。
- 数据更新、指标计算、信号扫描、回测和权重优化之间的字段版本需要保持一致。

## 后续计划

项目后续计划包括：

- 开发美股数据和策略支持，具体目录和实现路径 TBD；
- 完善 A 股与港股回测能力；
- 统一配置文件和依赖管理；
- 减少硬编码绝对路径，提高项目迁移能力；
- 完善数据更新、指标计算、信号扫描和回测之间的模块化接口；
- 增加更系统的 walk-forward 验证和策略稳定性评估；
- 在保持人工审核环节的前提下，逐步完善候选股票结果的自动化整理。

## 免责声明

本项目仅用于个人量化研究、数据分析和历史回测，不构成任何投资建议。

行情数据依赖 AKShare 等外部接口，可能存在数据缺失、数据延迟、接口变更、网络中断以及前复权历史数据调整等问题。项目生成的交易信号、涨幅榜单和回测结果仅代表特定数据、参数和历史区间下的分析结果，不代表未来收益。

任何使用者都应独立核验数据和策略逻辑，并自行承担由投资决策、程序运行或数据误差造成的风险。
