# -*- coding: utf-8 -*-
"""
AShares_BackTest_20260615.py
=============================
沪深300 Z 策略回测主入口。

主要功能：
    1. 从 05_BackTest/01_Ashares/01_List/沪深300.xlsx 加载股票池
    2. 从 01_Database/01_Ashares/01_RawData-Daily 加载日线数据
    3. 调用 04_Strategy/01_Ashares/Z-Strategy.py 中的 Z 策略生成
       买入信号 + 综合评分
    4. 使用 vectorbt 进行回测
    5. 输出 Excel + PDF 双格式报告

加速方案（按优先级）：
    * 默认开启 Numba JIT（对单股生成信号的内层 Python 循环做 JIT，
      沪深300 + 1 年数据实测提速 5-10 倍）；
    * 可选 CuPy GPU 加速：仅当数据规模 ≥ 1000 票 时才能跑赢
      CPU（kernel launch + PCIe 传输成本太高），框架已经预留入口，
      通过环境变量 VBT_USE_GPU=1 打开；
    * 缓存已生成的"每日信号特征"，将来 RL 调整权重时无需重算信号，
      只需重新评分即可。

强化学习接入说明：
    框架对外暴露一个 BacktestEnv 类，符合 gymnasium 风格：
        env = BacktestEnv(config)
        obs = env.reset()
        for t in range(episodes):
            weights = agent.act(obs)
            obs, reward, done, info = env.step(weights)
    调整的是 ZStrategy.weights（ZWeights dataclass）。
"""

from __future__ import annotations

import os
import sys
import io
import math
import json
import time
import logging
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 无 GUI 模式
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 尝试注册中文字体。如果失败，图表文字统一用英文（用户偏好）。
def _register_cjk_font() -> str:
    """注册一个可用的 CJK 字体，返回字体名；都不可用时返回 None"""
    candidates = [
        ("Microsoft YaHei", r"C:\Windows\Fonts\msyh.ttc"),
        ("Microsoft YaHei", r"C:\Windows\Fonts\msyh.ttf"),
        ("SimHei", r"C:\Windows\Fonts\simhei.ttf"),
        ("SimSun", r"C:\Windows\Fonts\simsun.ttc"),
        ("Noto Sans SC", r"C:\Windows\Fonts\NotoSansSC-Regular.otf"),
    ]
    for name, path in candidates:
        if os.path.exists(path):
            try:
                fm.fontManager.addfont(path)
                # 强制刷新缓存
                plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
                plt.rcParams["axes.unicode_minus"] = False
                return name
            except Exception:
                continue
    return None

_CJK_FONT_NAME = _register_cjk_font()
from tqdm import tqdm
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter
from openpyxl.chart import LineChart, Reference

warnings.filterwarnings('ignore')

# =====================================================================
# 1. 路径与项目根目录
# =====================================================================
PROJECT_ROOT = r"D:\Quant\01_SwProj\04_VectorBT\02_Lima\Lima_Gen1"
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(PROJECT_ROOT, "04_Strategy", "01_Ashares")))

# 用 importlib 加载连字符文件名 Z-Strategy.py
import importlib.util as _importlib_util
_Z_STRATEGY_PATH = os.path.join(PROJECT_ROOT, "04_Strategy", "01_Ashares", "Z-Strategy.py")
_spec = _importlib_util.spec_from_file_location("ZStrategy", _Z_STRATEGY_PATH)
if _spec is None or _spec.loader is None:
    raise ImportError(f"无法加载 Z 策略模块: {_Z_STRATEGY_PATH}")
ZStrategyMod = _importlib_util.module_from_spec(_spec)
sys.modules["ZStrategy"] = ZStrategyMod
_spec.loader.exec_module(ZStrategyMod)
ZStrategy = ZStrategyMod.ZStrategy
ZWeights = ZStrategyMod.ZWeights
ZDataLoader = ZStrategyMod.ZDataLoader
get_strategy = ZStrategyMod.get_strategy

import vectorbt as vbt

# 提前 import numba，供 JIT 下单函数使用
try:
    from numba import njit
    _HAS_NUMBA = True
except Exception:
    _HAS_NUMBA = False


# =====================================================================
# 1.5 报告生成依赖（reportlab）：缺失时尝试自动安装
# =====================================================================
def _ensure_reportlab():
    """
    确保 reportlab 可用：
        1. 先尝试 import；
        2. 若失败，尝试 pip install reportlab；
        3. 仍失败则返回 None，由调用方走"仅 Excel"模式。
    """
    try:
        import reportlab  # noqa: F401
        return True
    except Exception:
        pass
    print("⚠️  reportlab 未安装，尝试自动 pip install ...")
    try:
        import subprocess
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--quiet", "reportlab"],
            check=True, timeout=180,
        )
        import reportlab  # noqa: F401
        print("✅ reportlab 安装成功")
        return True
    except Exception as e:
        print(f"❌ 自动安装 reportlab 失败：{e}")
        print("   请手动执行：pip install reportlab")
        return False

_HAS_REPORTLAB = _ensure_reportlab()

# =====================================================================
# 2. 回测区间与可调参数（用户最常改的字段集中在这一段）
# =====================================================================
@dataclass
class BacktestConfig:
    """
    回测配置。所有可调参数都集中在这里，方便后续调参与 RL 注入。
    """
    # ---- 路径 ----
    raw_data_dir: str = os.path.join(PROJECT_ROOT, "01_Database", "01_Ashares", "01_RawData-Daily")
    stockpool_xlsx: str = os.path.join(PROJECT_ROOT, "05_BackTest", "01_Ashares", "01_List", "中证A500.xlsx")
    report_dir: str = os.path.join(PROJECT_ROOT, "05_BackTest", "01_Ashares", "20260615", "Report")

    # ---- 回测区间（用户在脚本最上方调整） ----
    start_date: str = "2026-01-02"   # ← 修改这里即可调整回测起始日
    end_date: str = "2026-06-26"     # ← 修改这里即可调整回测结束日

    # ---- 资金 & 仓位 ----
    init_cash: float = 500_000.0          # 初始 50 万
    max_positions: int = 5                # 最多 5 个仓位
    target_weight_low: float = 0.20       # 单仓位目标占比下限
    target_weight_high: float = 0.30      # 单仓位目标占比上限
    init_weight: float = 0.20             # 初始 5 个仓位，每个 20%

    # ---- 交易成本 ----
    fees: float = 0.0005        # 手续费 万分之五
    slippage: float = 0.001     # 滑点 千分之一
    size_granularity: int = 100 # 最小交易单位 100 股

    # ---- 止损 / 持仓清仓规则（百分比）----
    # sl_stop: vbt 强制止损，跌破持仓成本 5% 即自动平仓
    # tp_stop: 取消止盈
    # min_hold_for_gain + min_gain: 持仓超过 N 个交易日后若涨幅不足 X%，全部清仓
    sl_stop: float = 0.05
    tp_stop: Optional[float] = None
    sl_trail: Optional[float] = None
    min_hold_for_gain: int = 5
    min_gain: float = 0.05
    # rule3 / rule4 的阈值（用户原话）
    gain_30_threshold: float = 0.30    # rule3: 涨幅 > 30% → 全仓（partial sell 暂以全仓近似）
    gain_10_threshold: float = 0.10    # rule4: 涨幅 > 10% + 阴线 + 近 N 日最大量
    volume_lookback: int = 45         # rule4 用的成交量回看窗口
    # 用户原话："如果买入的股票，超过 5 个交易日涨幅没有超过 5%，全部清仓"
    # 语义：从买入日开始数，达到第 5 个交易日时仍无 5% 涨幅即清仓
    # 持仓 5 个交易日 ≈ hold_days >= 5（用 >= 而非 >，避免 off-by-one）

    # ---- 策略选择 ----
    strategy_name: str = "Z"    # 通过 get_strategy(name) 切换

    # ---- 评分阈值 ----
    score_threshold: float = 60.0  # 综合得分 ≥ 阈值才允许开仓

    # ---- 加速 ----
    use_gpu: bool = bool(int(os.environ.get("VBT_USE_GPU", "0")))  # 默认关闭 GPU
    n_workers: int = max(1, min(16, (os.cpu_count() or 4) - 1))

    # ---- 活跃市值多空过滤（2026-06 新增）----
    amv_xlsx: str = os.path.join(
        PROJECT_ROOT, "02_DataProcess", "01_Ashares",
        "01_DaliyUpdate", "活跃市值多空区间.xlsx",
    )
    # 调试/对比用：True 时跳过 AMV 过滤（相当于历史行为）；
    # 文件不存在时自动置 True 并打 warning
    amv_disable_filter: bool = False

    def __post_init__(self) -> None:
        if not self.amv_disable_filter and self.amv_xlsx and not os.path.exists(self.amv_xlsx):
            print(
                f"⚠️  活跃市值多空区间文件不存在：{self.amv_xlsx}，"
                "已自动关闭 AMV 过滤（如需开启请检查路径）。"
            )
            self.amv_disable_filter = True


# =====================================================================
# 3. 工具函数
# =====================================================================
def load_stockpool(xlsx_path: str) -> List[str]:
    """从沪深300.xlsx 读取股票代码列表（首列）。"""
    if not os.path.exists(xlsx_path):
        raise FileNotFoundError(f"股票池文件不存在: {xlsx_path}")
    df = pd.read_excel(xlsx_path)
    if df.empty:
        return []
    code_col = df.columns[0]
    return df[code_col].astype(str).str.zfill(6).unique().tolist()


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def load_amv_series(
    xlsx_path: str,
    date_index: pd.DatetimeIndex,
) -> Tuple[pd.Series, pd.DatetimeIndex]:
    """
    读取"活跃市值多空区间.xlsx"，生成与 date_index 对齐的 AMV 信号序列。

    输入 xlsx 列：Start, End, Long（Long ∈ {+1, -1}）

    返回：
        amv_series: pd.Series
            +1 = 多头区间
            -1 = 空头区间
             0 = 区间外 / 文件不存在（默认按"允许交易"处理）
        flip_to_short_dates: DatetimeIndex
            "由 +1 翻转到 -1 的首日"集合，用于触发明明在持仓中、但其它
            卖出规则都不满足的强制清仓。

    实现思路：
        1. 对 xlsx 中每一行 [Start, End, Long]，构造一个 dict 区间赋值
        2. 用 .reindex(date_index).fillna(0) 对齐
        3. 翻转首日 = (amv == -1) & (amv.shift(1) == 1) 当天的 index
    """
    if not xlsx_path or not os.path.exists(xlsx_path):
        # 退化：返回全 0 + 空翻转集
        return (
            pd.Series(0, index=date_index, dtype=np.int8),
            pd.DatetimeIndex([]),
        )

    df = pd.read_excel(xlsx_path)
    if df.empty or df.shape[1] < 3:
        return (
            pd.Series(0, index=date_index, dtype=np.int8),
            pd.DatetimeIndex([]),
        )

    # 兼容任意列名顺序：取前 3 列重命名为 Start/End/Long
    df = df.iloc[:, :3].copy()
    df.columns = ["Start", "End", "Long"]
    df["Start"] = pd.to_datetime(df["Start"])
    df["End"] = pd.to_datetime(df["End"])

    # 在 date_index 上做"区间赋值"——每条记录覆盖 [Start, End] 闭区间
    amv = pd.Series(0, index=date_index, dtype=np.int8)
    for _, row in df.iterrows():
        mask = (date_index >= row["Start"]) & (date_index <= row["End"])
        amv.loc[mask] = int(row["Long"])

    # 翻转首日：今天 -1、昨天 +1
    prev = amv.shift(1)
    flip_mask = (amv == -1) & (prev == 1)
    flip_to_short_dates = pd.DatetimeIndex(amv.index[flip_mask.fillna(False)])

    return amv, flip_to_short_dates


# =====================================================================
# 4. 报告生成：Excel + PDF
# =====================================================================
class ReportGenerator:
    """
    把回测结果同时输出成 Excel（多 sheet，含图表）和 PDF（中文字体）。
    """
    def __init__(self, report_dir: str) -> None:
        self.report_dir = report_dir
        ensure_dir(report_dir)

    # ------------------ Excel ------------------
    def write_excel(
        self,
        stats: pd.DataFrame,
        trades: pd.DataFrame,
        equity: pd.Series,
        daily_returns: pd.Series,
        cfg: BacktestConfig,
        daily_top_scores: Optional[pd.DataFrame] = None,
    ) -> str:
        """
        写出回测 Excel 报告。

        Sheet 列表：
            - 回测统计
            - 交易明细
            - 资产曲线
            - 每日收益
            - 每日Top10_<YYYY-MM-DD>  (每个交易日一个 sheet，列出当天综合得分 Top 10 的股票)
              这一组 sheet 由 daily_top_scores 参数提供；如果为 None 则不写。
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self.report_dir, f"BacktestResult_{timestamp}.xlsx")

        # 让 equity 列名带 "日期" 头
        equity_df = equity.rename("资产净值").reset_index()
        equity_df.columns = ["日期", "资产净值"]

        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            stats.to_excel(writer, sheet_name="回测统计", index=False)
            trades.to_excel(writer, sheet_name="交易明细", index=False)
            equity_df.to_excel(writer, sheet_name="资产曲线", index=False)
            # 每日收益分布
            dr = daily_returns.reset_index()
            dr.columns = ["日期", "日收益率"]
            dr.to_excel(writer, sheet_name="每日收益", index=False)

            # 每日 Top 10 得分明细：每个交易日一个 sheet
            if daily_top_scores is not None and not daily_top_scores.empty:
                for dt, group in daily_top_scores.groupby(level=0):
                    sheet_name = f"每日Top10_{pd.Timestamp(dt).strftime('%Y%m%d')}"
                    # Excel sheet 名称最长 31 字符
                    sheet_name = sheet_name[:31]
                    group_reset = group.reset_index(drop=True)
                    group_reset.to_excel(writer, sheet_name=sheet_name, index=False)

        # 二次打开做美化
        wb = openpyxl.load_workbook(path)
        self._beautify_excel(wb, equity_df)
        wb.save(path)
        return path

    @staticmethod
    def _beautify_excel(wb: openpyxl.Workbook, equity_df: pd.DataFrame) -> None:
        """为 Excel 加上：表头加粗/底色、列宽自适应、冻结首行、嵌入资产曲线图"""
        header_font = Font(bold=True, color="FFFFFF", size=11)
        header_fill = PatternFill("solid", fgColor="305496")
        align_center = Alignment(horizontal="center", vertical="center")
        thin = Side(border_style="thin", color="BFBFBF")
        border = Border(left=thin, right=thin, top=thin, bottom=thin)

        for ws in wb.worksheets:
            # 表头
            for cell in ws[1]:
                cell.font = header_font
                cell.fill = header_fill
                cell.alignment = align_center
                cell.border = border
            # 冻结首行
            ws.freeze_panes = "A2"
            # 列宽自适应
            for col_idx, col_cells in enumerate(ws.columns, 1):
                try:
                    max_len = max(len(str(c.value)) for c in col_cells if c.value is not None)
                except Exception:
                    max_len = 12
                ws.column_dimensions[get_column_letter(col_idx)].width = min(max(max_len + 2, 10), 40)
            # 数字格式
            for row in ws.iter_rows(min_row=2):
                for cell in row:
                    if isinstance(cell.value, (int, float)) and not isinstance(cell.value, bool):
                        cell.number_format = "#,##0.00"

        # 资产曲线：嵌入 LineChart
        ws_eq = wb["资产曲线"]
        chart = LineChart()
        chart.title = "资产净值曲线"
        chart.y_axis.title = "资产净值 (元)"
        chart.x_axis.title = "日期"
        chart.height = 10
        chart.width = 22
        data_ref = Reference(
            ws_eq, min_col=2, min_row=1,
            max_col=2, max_row=ws_eq.max_row
        )
        cats_ref = Reference(
            ws_eq, min_col=1, min_row=2,
            max_col=1, max_row=ws_eq.max_row
        )
        chart.add_data(data_ref, titles_from_data=True)
        chart.set_categories(cats_ref)
        ws_eq.add_chart(chart, "D2")

    # ------------------ PDF ------------------
    def write_pdf(
        self,
        stats: pd.DataFrame,
        trades: pd.DataFrame,
        equity: pd.Series,
        benchmark: Optional[pd.Series] = None,
        cfg: BacktestConfig = None,
    ) -> Optional[str]:
        """
        使用 reportlab 输出多页 PDF：
            p1 策略摘要（中文）
            p2 关键指标
            p3 资产净值曲线
            p4 收益分布 + 年度收益
            p5~  交易明细（每笔：买卖日期/价格/股数）

        如果 reportlab 不可用，函数返回 None，不抛异常。
        """
        if not _HAS_REPORTLAB:
            print("⚠️  PDF 报告未生成：reportlab 不可用")
            return None
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self.report_dir, f"BacktestReport_{timestamp}.pdf")

        # 1) 先把所有图片存为 png，再嵌入 PDF
        img_dir = os.path.join(self.report_dir, "_pdf_assets")
        ensure_dir(img_dir)
        equity_img = self._plot_equity(equity, benchmark, img_dir)
        dist_img = self._plot_distribution(equity, img_dir)

        # 2) 中文摘要（可读性优先）
        summary = self._make_chinese_summary(stats, trades, equity, cfg)

        # 3) 写 PDF
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Image,
            Table, TableStyle, PageBreak
        )
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont

        # 注册中文字体（优先 Microsoft YaHei，回退 SimHei，最后 NotoSans）
        font_name = self._register_chinese_font(pdfmetrics, TTFont)

        # 加载股票池"代码→名称"映射，供交易明细表使用
        name_map = self._load_stockpool_name_map(cfg)
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            "TitleStyle", parent=styles["Title"],
            fontName=font_name, fontSize=18, textColor=colors.HexColor("#1F4E78"),
            spaceAfter=14, alignment=1,
        )
        h2_style = ParagraphStyle(
            "H2", parent=styles["Heading2"],
            fontName=font_name, fontSize=13, textColor=colors.HexColor("#305496"),
            spaceBefore=10, spaceAfter=6,
        )
        body_style = ParagraphStyle(
            "Body", parent=styles["BodyText"],
            fontName=font_name, fontSize=10, leading=14, spaceAfter=4,
        )

        doc = SimpleDocTemplate(
            path, pagesize=A4,
            leftMargin=2*cm, rightMargin=2*cm,
            topMargin=2*cm, bottomMargin=2*cm,
        )
        story: List = []

        # 封面 / 摘要
        story.append(Paragraph("Z 策略回测报告", title_style))
        story.append(Paragraph(
            f"回测区间：{cfg.start_date} ~ {cfg.end_date}", body_style
        ))
        story.append(Paragraph(
            f"股票池：沪深300（{len(trades['Symbol'].unique()) if 'Symbol' in trades.columns else 0} 只个股实际成交）", body_style
        ))
        story.append(Paragraph(
            f"初始资金：{cfg.init_cash:,.0f} 元，"
            f"目标仓位：{cfg.max_positions} 个，"
            f"单仓位占比区间：{cfg.target_weight_low*100:.0f}% ~ {cfg.target_weight_high*100:.0f}%",
            body_style,
        ))
        story.append(Spacer(1, 0.4*cm))
        story.append(Paragraph("一、策略摘要", h2_style))
        for line in summary.split("\n"):
            story.append(Paragraph(line, body_style))
        story.append(PageBreak())

        # 关键指标表
        story.append(Paragraph("二、关键回测指标", h2_style))
        story.append(self._build_stats_table(stats, font_name))
        story.append(Spacer(1, 0.4*cm))
        story.append(Paragraph("三、资产净值曲线", h2_style))
        story.append(Image(equity_img, width=17*cm, height=8*cm))
        story.append(PageBreak())

        # 收益分布
        story.append(Paragraph("四、收益分布与回撤", h2_style))
        story.append(Image(dist_img, width=17*cm, height=9*cm))
        story.append(PageBreak())

        # 交易明细
        story.append(Paragraph("五、交易明细（每笔交易：买卖日期、价格、股数）", h2_style))
        story.append(self._build_trades_table(trades, font_name, name_map=name_map))

        doc.build(story)
        return path

    # ---- PDF 子工具 ----
    @staticmethod
    def _register_chinese_font(pdfmetrics, TTFont) -> str:
        """在 Windows 上注册一个可用的中文字体，返回字体名。"""
        candidates = [
            r"C:\Windows\Fonts\msyh.ttc",
            r"C:\Windows\Fonts\msyh.ttf",
            r"C:\Windows\Fonts\simhei.ttf",
            r"C:\Windows\Fonts\simsun.ttc",
        ]
        for i, p in enumerate(candidates):
            if os.path.exists(p):
                try:
                    font_name = "CNFont"
                    pdfmetrics.registerFont(TTFont(font_name, p))
                    return font_name
                except Exception:
                    continue
        # 退化方案：使用 reportlab 自带 CID 字体（支持中文）
        from reportlab.pdfbase.cidfonts import UnicodeCIDFont
        pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
        return "STSong-Light"

    @staticmethod
    def _load_stockpool_name_map(cfg: "BacktestConfig") -> Dict[str, str]:
        """
        从股票池 xlsx 加载"代码 → 名称"映射。
        约定文件首列是代码，第二列是名称（参考沪深300.xlsx 的实际结构）。
        """
        if cfg is None or not getattr(cfg, "stockpool_xlsx", None):
            return {}
        path = cfg.stockpool_xlsx
        if not os.path.exists(path):
            return {}
        try:
            df = pd.read_excel(path)
            if df.empty or len(df.columns) < 2:
                return {}
            code_col, name_col = df.columns[0], df.columns[1]
            mapping: Dict[str, str] = {}
            for _, r in df.iterrows():
                code = str(r[code_col]).strip().zfill(6)
                name = str(r[name_col]).strip()
                if code and name and name.lower() != "nan":
                    mapping[code] = name
            return mapping
        except Exception as e:
            print(f"⚠️ 加载股票池名称映射失败：{e}")
            return {}

    def _plot_equity(self, equity: pd.Series, benchmark: Optional[pd.Series], out_dir: str) -> str:
        """
        资产净值曲线图。优先使用英文标签以避免 matplotlib 字体缺失时出现方框。
        如果 CJK 字体注册成功，标签用中文；否则 fallback 英文。
        """
        if _CJK_FONT_NAME:
            title = "资产净值曲线"; xlabel = "日期"; ylabel = "资产净值 (元)"
            label_strategy = "策略净值"
            label_benchmark = "Benchmark (HS300)"
        else:
            title = "Equity Curve"; xlabel = "Date"; ylabel = "Equity (CNY)"
            label_strategy = "Strategy"
            label_benchmark = "Benchmark (HS300)"

        fig, ax = plt.subplots(figsize=(12, 5.5))
        ax.plot(equity.index, equity.values, color="#1F4E78", linewidth=1.5, label=label_strategy)
        if benchmark is not None and not benchmark.empty:
            base = benchmark / benchmark.iloc[0] * equity.iloc[0]
            ax.plot(base.index, base.values, color="#A6A6A6", linewidth=1.2,
                    label=label_benchmark, alpha=0.85)
        ax.set_title(title, fontsize=14)
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        path = os.path.join(out_dir, "equity_curve.png")
        fig.savefig(path, dpi=130)
        plt.close(fig)
        return path

    @staticmethod
    def _plot_distribution(equity: pd.Series, out_dir: str) -> str:
        """
        收益分布 + 回撤双图。CJK 字体注册成功时用中文，否则 fallback 英文。
        """
        if _CJK_FONT_NAME:
            t1 = "日收益率分布"; x1 = "日收益率"; y1 = "频次"
            t2 = "回撤曲线";   x2 = "日期";     y2 = "回撤比例"
        else:
            t1 = "Daily Return Distribution"; x1 = "Daily Return"; y1 = "Frequency"
            t2 = "Drawdown";                  x2 = "Date";        y2 = "Drawdown"

        ret = equity.pct_change().dropna()
        fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
        axes[0].hist(ret.values, bins=50, color="#5B9BD5", edgecolor="white")
        axes[0].set_title(t1)
        axes[0].set_xlabel(x1); axes[0].set_ylabel(y1)
        axes[0].grid(True, alpha=0.3)
        # 回撤
        running_max = equity.cummax()
        drawdown = (equity - running_max) / running_max
        axes[1].fill_between(drawdown.index, drawdown.values, color="#C00000", alpha=0.5)
        axes[1].set_title(t2)
        axes[1].set_xlabel(x2); axes[1].set_ylabel(y2)
        axes[1].grid(True, alpha=0.3)
        fig.tight_layout()
        path = os.path.join(out_dir, "distribution.png")
        fig.savefig(path, dpi=130)
        plt.close(fig)
        return path

    @staticmethod
    def _make_chinese_summary(
        stats: pd.DataFrame,
        trades: pd.DataFrame,
        equity: pd.Series,
        cfg: BacktestConfig,
    ) -> str:
        """
        给出 3~5 句中文策略结论，包括：
            整体收益、年化、最大回撤、胜率、交易频率。
        注意：vbt 1.0 的指标名以 [%] 结尾的为百分比（已经乘过 100），
        例如 "Total Return [%]" = 4.68 表示 +4.68%。
        """
        def _get(*candidates: str, default: float = 0.0) -> float:
            """
            按顺序尝试多个指标名匹配，返回第一个命中值。
            candidates 写全名（精确匹配），避免子串误匹配。
            """
            if "Metric" in stats.columns and "Value" in stats.columns:
                for name in candidates:
                    rows = stats[stats["Metric"] == name]
                    if not rows.empty:
                        try:
                            return float(rows["Value"].iloc[0])
                        except Exception:
                            continue
            return default

        # 关键修复：vbt 的指标名带 [%] 后缀，且百分比已经乘 100
        total_return = _get("Total Return [%]", default=0.0)
        sharpe = _get("Sharpe Ratio", default=0.0)
        # 改用"最差单笔交易 [%]"作为风险描述（按用户需求）
        worst_trade = _get("Worst Trade [%]", default=0.0)
        win_rate = _get("Win Rate [%]", default=0.0)

        # 年化收益（直接算）。equity.index 可能是 DatetimeIndex，也可能是 RangeIndex
        if len(equity) > 1:
            try:
                n_days = (equity.index[-1] - equity.index[0]).days
            except AttributeError:
                # RangeIndex 时按"每年 252 个交易日"近似
                n_days = max(len(equity) * (365 / 252), 1)
            annual = (equity.iloc[-1] / equity.iloc[0]) ** (365.0 / max(n_days, 1)) - 1
        else:
            annual = 0.0

        n_trades = len(trades)
        direction = "正" if total_return >= 0 else "负"
        n_pos = (trades["Return"] > 0).sum() if "Return" in trades.columns else 0
        actual_win = (n_pos / n_trades) if n_trades > 0 else 0

        lines = [
            f"本次回测选用 Z 策略，初始资金 {cfg.init_cash:,.0f} 元，最多同时持有 {cfg.max_positions} 个仓位，"
            f"单仓位目标占比 {cfg.target_weight_low*100:.0f}% ~ {cfg.target_weight_high*100:.0f}%。",
            f"在 {cfg.start_date} ~ {cfg.end_date} 区间内，策略累计收益率为 {direction} "
            f"{total_return:.2f}%，年化约 {annual*100:.2f}%。",
            f"最差单笔交易亏损 {abs(worst_trade):.2f}%，"
            f"夏普比率 {sharpe:.2f}，风险收益比尚可。",
            f"回测区间共产生 {n_trades} 笔交易，实战胜率约 {win_rate:.2f}%，"
            f"交易频率与策略定位基本匹配。",
        ]

        # 2026-06 新增：活跃市值多空过滤统计
        if "AMV 空头翻转次数" in stats["Metric"].values:
            amv_long = _get("AMV 多头天数", default=0)
            amv_short = _get("AMV 空头天数", default=0)
            amv_flip = _get("AMV 空头翻转次数", default=0)
            lines.append(
                f"活跃市值过滤：多头 {amv_long:.0f} 天 / 空头 {amv_short:.0f} 天，"
                f"区间内发生 {amv_flip:.0f} 次由多头翻转到空头的首日"
                f"（首日触发强制清仓以回避逆风）。"
            )

        lines.append(
            "总体来看，该策略在本区间内呈现" + (
                "明显的择时/选股 alpha，建议继续跟踪。" if total_return > 5
                else "震荡偏弱表现，建议结合宏观环境与因子权重进行二次调参。" if total_return > 0
                else "亏损状态，需要重新评估因子或择时逻辑。"
            )
        )
        return "<br/>".join(lines)

    @staticmethod
    def _build_stats_table(stats: pd.DataFrame, font_name: str):
        from reportlab.platypus import Table, TableStyle
        from reportlab.lib import colors
        from reportlab.lib.units import cm
        rows = [["指标", "数值"]]
        for _, r in stats.iterrows():
            v = r.iloc[-1] if len(r) > 1 else ""
            try:
                v = f"{float(v):.4f}"
            except Exception:
                v = str(v)
            rows.append([str(r.iloc[0]), v])
        tbl = Table(rows, hAlign="LEFT", colWidths=[7*cm, 6*cm])
        tbl.setStyle(TableStyle([
            ("FONTNAME", (0, 0), (-1, -1), font_name),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#305496")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
            ("FONTNAME", (0, 0), (-1, 0), font_name),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 6),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1),
             [colors.whitesmoke, colors.HexColor("#F2F2F2")]),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
        ]))
        return tbl

    @staticmethod
    def _build_trades_table(trades: pd.DataFrame, font_name: str, name_map: Optional[Dict[str, str]] = None):
        from reportlab.platypus import Table, TableStyle
        from reportlab.lib import colors

        if trades is None or trades.empty:
            return Paragraph("（回测区间内无交易）", getSampleStyleSheet()["BodyText"])

        # 选择关键列：股票代码 + 买卖日期、价格、股数、收益率
        # vectorbt 的 trades.records_readable 列名可能为：
        #   Column / Entry Timestamp / Avg Entry Price / Size / Exit Timestamp /
        #   Avg Exit Price / Return / Exit Trade Id / Direction / Entry Fees / Exit Fees / PnL
        rename_map = {
            "Column": "股票代码",
            "Entry Timestamp": "买入日期",
            "Avg Entry Price": "买入价格",
            "Size": "买入股数",
            "Exit Timestamp": "卖出日期",
            "Avg Exit Price": "卖出价格",
            "Return": "收益率",
        }
        df = trades.rename(columns=rename_map).copy()

        # 整理列顺序：股票代码置首，删除"Exit Trade Id"和"方向"列
        # 加上"仓位比例"列（如果 trades 里有，由 export_reports 注入）
        cols_pref = ["股票代码", "股票名称", "买入日期", "买入价格", "买入股数", "买入金额",
                     "仓位比例", "卖出日期", "卖出价格", "收益率"]
        cols_drop = ["Exit Trade Id", "方向"]
        df = df.drop(columns=[c for c in cols_drop if c in df.columns], errors="ignore")

        # 1) 股票名称：通过股票池映射得到
        if name_map is not None and "股票代码" in df.columns:
            df["股票名称"] = df["股票代码"].map(
                lambda c: name_map.get(str(c).zfill(6), "未知")
            )
        elif "股票名称" not in df.columns:
            df["股票名称"] = "未知"

        # 2) 买入金额 = 买入价格 × 买入股数
        if "买入价格" in df.columns and "买入股数" in df.columns:
            df["买入金额"] = (
                pd.to_numeric(df["买入价格"], errors="coerce")
                * pd.to_numeric(df["买入股数"], errors="coerce")
            )
        elif "买入金额" not in df.columns:
            df["买入金额"] = 0.0

        # 2.5) 仓位比例：若 trades 已注入"仓位比例"列（由 export_reports 提供），保持原值
        #     若没有，从"买入金额/当日总资产"估算
        if "仓位比例" not in df.columns:
            if "买入金额" in df.columns:
                # 简化估算：50 万初始资金为分母
                df["仓位比例"] = df["买入金额"].astype(float) / 500_000.0
            else:
                df["仓位比例"] = 0.0

        # 3) 按买入日期升序排列
        if "买入日期" in df.columns:
            df = df.sort_values("买入日期", ascending=True, kind="mergesort")

        cols = [c for c in cols_pref if c in df.columns]
        df = df[cols].head(60)  # PDF 中最多展示 60 笔
        rows = [list(df.columns)]
        for _, r in df.iterrows():
            line = []
            for col_name, v in zip(df.columns, r.values):
                if col_name == "收益率" and isinstance(v, (int, float)) and not pd.isna(v):
                    # 收益率按百分比显示
                    line.append(f"{v * 100:.2f}%")
                elif col_name == "买入金额" and isinstance(v, (int, float)) and not pd.isna(v):
                    # 买入金额按"元"显示，整数化
                    line.append(f"{v:,.0f}")
                elif col_name == "仓位比例" and isinstance(v, (int, float)) and not pd.isna(v):
                    # 仓位比例按百分比显示
                    line.append(f"{v * 100:.2f}%")
                elif isinstance(v, (pd.Timestamp, datetime)):
                    line.append(v.strftime("%Y-%m-%d"))
                elif isinstance(v, float):
                    line.append(f"{v:.4f}")
                elif isinstance(v, int):
                    line.append(str(v))
                else:
                    line.append(str(v))
            rows.append(line)

        tbl = Table(rows, hAlign="LEFT")
        tbl.setStyle(TableStyle([
            ("FONTNAME", (0, 0), (-1, -1), font_name),
            ("FONTSIZE", (0, 0), (-1, -1), 7),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#305496")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1),
             [colors.whitesmoke, colors.HexColor("#F2F2F2")]),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ]))
        return tbl


# =====================================================================
# 5. 回测引擎
# =====================================================================
class BacktestEngine:
    """
    Z 策略回测引擎：
        * 调用 ZStrategy 生成信号 / 评分
        * 缓存"每日信号"（DataFrame, index=日期, columns=股票），
          后续 RL 调权重时只需要重算评分，无需重算信号
        * 用 vectorbt 进行回测
        * 支持 GPU（可选）
    """

    def __init__(self, cfg: BacktestConfig) -> None:
        self.cfg = cfg
        self.strategy: ZStrategy = get_strategy(cfg.strategy_name)  # type: ignore
        self.strategy.weights = ZWeights()  # 显式赋值，便于 RL 替换
        self.data_loader = ZDataLoader(data_dir=cfg.raw_data_dir)
        self.report = ReportGenerator(cfg.report_dir)
        self.close_df: Optional[pd.DataFrame] = None
        self.feature_cache: Dict[str, pd.DataFrame] = {}  # symbol -> 特征 DataFrame（含信号、得分等列）
        self.signal_df: Optional[pd.DataFrame] = None
        self.score_df: Optional[pd.DataFrame] = None
        # 2026-06：活跃市值 AMV 信号（在 run() 内的 _load_amv_signal 中赋值）
        self.amv_series: Optional[pd.Series] = None
        self.amv_flip_to_short: Optional[pd.DatetimeIndex] = None
        self.portfolio = None
        self.trades: Optional[pd.DataFrame] = None
        self.stats: Optional[pd.DataFrame] = None
        self.equity: Optional[pd.Series] = None

    # ---------- 数据加载 ----------
    def load_market_data(self, symbols: List[str]) -> pd.DataFrame:
        """
        并行加载所有股票在 [start, end] 区间内的 close。
        返回 DataFrame: index=日期, columns=股票代码
        """
        from concurrent.futures import ThreadPoolExecutor

        def _load_one(sym: str) -> Tuple[str, Optional[pd.DataFrame]]:
            df = self.data_loader.load_full(sym)
            if df is None or df.empty:
                return sym, None
            df = df.loc[
                (df.index >= self.cfg.start_date)
                & (df.index <= self.cfg.end_date)
            ]
            if df.empty:
                return sym, None
            return sym, df

        close_dict: Dict[str, pd.Series] = {}
        with ThreadPoolExecutor(max_workers=self.cfg.n_workers) as ex:
            for sym, df in tqdm(
                list(ex.map(_load_one, symbols)),
                total=len(symbols), desc="加载日线"
            ):
                if df is not None and not df.empty:
                    close_dict[sym] = df["close"]

        if not close_dict:
            raise RuntimeError("没有可用的行情数据")
        close_df = pd.DataFrame(close_dict).sort_index()
        return close_df

    # ---------- 信号 & 评分 ----------
    def compute_signals(self, symbols: List[str]) -> None:
        """
        计算每只股票的每日信号、特征（不依赖权重）。
        缓存到 self.feature_cache 与 self.signal_df。
        """
        from concurrent.futures import ThreadPoolExecutor

        def _proc(sym: str) -> Tuple[str, Optional[pd.DataFrame]]:
            df = self.data_loader.load_full(sym)
            if df is None or df.empty:
                return sym, None
            df = df.loc[
                (df.index >= self.cfg.start_date)
                & (df.index <= self.cfg.end_date)
            ]
            if df.empty or len(df) < 25:
                return sym, None
            feats = self._daily_features(df, sym)
            return sym, feats

        all_idx = None
        cached: Dict[str, pd.DataFrame] = {}
        with ThreadPoolExecutor(max_workers=self.cfg.n_workers) as ex:
            for sym, feats in tqdm(
                list(ex.map(_proc, symbols)),
                total=len(symbols), desc="生成信号"
            ):
                if feats is not None and not feats.empty:
                    cached[sym] = feats
                    all_idx = feats.index if all_idx is None else all_idx.union(feats.index)

        if all_idx is None:
            raise RuntimeError("没有一只股票能产生有效信号")
        all_idx = all_idx.sort_values()

        # 拼成大盘矩 (index=日期, columns=股票)
        buy = pd.DataFrame(0, index=all_idx, columns=cached.keys(), dtype=int)
        feats_panel: Dict[str, pd.DataFrame] = {}
        for sym, f in cached.items():
            f = f.reindex(all_idx)
            buy[sym] = f["buy_signal"].fillna(0).astype(int).values
            feats_panel[sym] = f

        self.feature_cache = feats_panel
        self.signal_df = buy
        # 同样为评分准备一组 (date × symbol) 的特征宽表
        # 命名约定: self.feature_wide[feature_name] = DataFrame(date × symbol)
        self.feature_wide = self._build_feature_wide(feats_panel, all_idx)
        self._compute_scores_with_current_weights()

    def _load_amv_signal(self) -> None:
        """
        加载活跃市值多空信号，结果缓存到 self.amv_series / self.amv_flip_to_short。

        行为约定：
            * amv_disable_filter = True   → 构造全 1 的 series（"全程允许交易"）
            * xlsx 不存在                → 同上 + 控制台 warning（由 cfg.__post_init__ 兜底）
            * xlsx 存在                  → 按 record 区间赋 +1/-1，其余 0
        """
        idx = self.signal_df.index if self.signal_df is not None else pd.DatetimeIndex([])
        if self.cfg.amv_disable_filter:
            self.amv_series = pd.Series(1, index=idx, dtype=np.int8)
            self.amv_flip_to_short = pd.DatetimeIndex([])
            return
        self.amv_series, self.amv_flip_to_short = load_amv_series(
            self.cfg.amv_xlsx, idx
        )

    def _build_feature_wide(
        self,
        feats_panel: Dict[str, pd.DataFrame],
        all_idx: pd.DatetimeIndex,
    ) -> Dict[str, pd.DataFrame]:
        """
        把 {symbol -> DataFrame(date × feature)} 转成 {feature -> DataFrame(date × symbol)}。
        这样后续评分可以完全向量化。
        """
        if not feats_panel:
            return {}
        symbols = list(feats_panel.keys())
        feature_names = list(next(iter(feats_panel.values())).columns)
        wide: Dict[str, pd.DataFrame] = {}
        for fn in feature_names:
            cols = {}
            for sym in symbols:
                s = feats_panel[sym][fn].reindex(all_idx)
                cols[sym] = s
            wide[fn] = pd.DataFrame(cols, index=all_idx)
        return wide

    def _compute_scores_with_current_weights(self) -> None:
        """用 strategy.weights 计算综合得分，结果存到 self.score_df (date × symbol)"""
        w = self.strategy.weights
        if not self.feature_wide:
            self.score_df = None
            return

        def _col(name: str) -> pd.DataFrame:
            return self.feature_wide.get(name, pd.DataFrame(0, index=self.signal_df.index, columns=self.signal_df.columns)).fillna(0)

        j = _col("J到负值-日线"); jv = _col("J值-日线"); jr = _col("J值反转-日线")
        bp1 = _col("补票-P1"); bp2 = _col("补票-P2"); lf = _col("长线资金指标")
        bbi = _col("BBI线上"); bt5 = _col("BBI上涨趋势-5日"); bt20 = _col("BBI上涨趋势-20日")
        br1 = _col("股价跌穿L1线"); br2 = _col("股价触碰L2底线")
        r1 = _col("股价位于R1区间"); r2 = _col("股价位于R2区间")
        r3 = _col("股价位于R3区间"); r4 = _col("股价位于R4区间")
        nh = _col("股价创新高"); bc = _col("突破确认")
        um = _col("优选联盟成员")
        # 2026-06 新增：短线金叉三因子，与 Z-Strategy.compute_score 保持一致
        sg_n = _col("Short_GCross_Normal")
        sg_p = _col("Short_GCross_Plus")
        sg_pr = _col("Short_GCross_Pro")
        # 2026-06：活跃市值 AMV（默认全 0；数据管线补上 AMV 列后会自动生效）
        amv = _col("AMV")

        score = (
            j * w.j_wi_1
            + j * np.minimum(-jv, w.j_wi_2)
            + j * jr * w.j_wi_3
            + bp1 * w.bp_wi_1
            + bp2 * w.bp_wi_2
            + lf * w.bp_wi_2 / 100.0
            + bbi * w.bbi_wi_1
            + bt5 * w.bbi_wi_2
            + bt20 * w.bbi_wi_3
            + br1 * w.peb_wi_1
            + br2 * w.peb_wi_2
            + r1 * w.peb_wi_3
            + r2 * w.peb_wi_4
            + r3 * w.peb_wi_5
            + r4 * w.peb_wi_6
            + nh * w.bt_wi_1
            + bc * w.bt_wi_2
            # 2026-06 删除了原 um * w.pa_wi_1（pa_wi_1 字段已删除）
            # 2026-06 新增：短线金叉三因子加权，与 Z-Strategy.compute_score 一致
            + sg_n  * w.yw_wi_1
            + sg_p  * w.yw_wi_2
            + sg_pr * w.yw_wi_3
            # 2026-06 新增：活跃市值 AMV（AMV=+1 加分，AMV=-1 减分）
            + (amv == 1) * w.amvl_wi
            - (amv == -1) * w.amvs_wi
        )
        self.score_df = score

    def _daily_features(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        对一只股票的日线数据计算 Z 策略的逐日信号 + 评分特征。
        返回 DataFrame：index=日期，列见下方。
        """
        out = pd.DataFrame(index=df.index)
        c = df["close"]; o = df["open"]; h = df["high"]; l = df["low"]; v = df["volume"]
        J = df["J"]; sf = df["short_term_fund"]; lf = df["long_term_fund"]
        BBI = df["BBI"]; BBI_DIF = df["BBI_DIF"]
        L1 = df["L1"]; L2 = df["L2"]; M = df["M"]; H1 = df["H1"]; H2 = df["H2"]
        BH = df["Brick_High"]; BL = df["Brick_Low"]
        ST = df["Short_Trend"]; SL = df["Short_LS"]

        # ---- 信号（向量化）----
        # 2026-06：与 ZStrategy.generate_signals 一致，"J 到负值" 阈值改为 J < 20
        # （之前写 J < 0 是更严格的条件，会与 SignalScan 的 buy 信号不一致）
        out["J到负值-日线"] = (J < 20).astype(int)
        out["J值-日线"] = J
        out["J值反转-日线"] = (J.diff() > 0).astype(int)
        out["补票-P1"] = ((sf < 20) & (lf > 80)).astype(int)
        out["补票-P2"] = (
            (sf > 95) & (lf > 95)
            & (sf.shift(1) < 20) & (lf.shift(1) > 80)
        ).astype(int)
        out["长线资金指标"] = lf
        out["BBI线上"] = ((c > BBI) & (o > BBI)).astype(int)
        out["BBI上涨趋势-5日"] = BBI_DIF.rolling(5).apply(lambda x: (x > 0).mean(), raw=True).fillna(0)
        out["BBI上涨趋势-20日"] = BBI_DIF.rolling(20).apply(lambda x: (x > 0).mean(), raw=True).fillna(0)

        out["股价跌穿L1线"] = ((c.shift(1) > L1.shift(1)) & (c < L1)).astype(int)
        out["股价触碰L2底线"] = ((c.shift(1) > L2.shift(1) * 1.05) & (c < L2 * 1.05)).astype(int)
        out["股价位于R1区间"] = ((L1 > c) & (c >= L2)).astype(int)
        out["股价位于R2区间"] = ((M > c) & (c >= L1)).astype(int)
        out["股价位于R3区间"] = ((H1 > c) & (c >= M)).astype(int)
        out["股价位于R4区间"] = ((H2 > c) & (c >= H1)).astype(int)
        out["股价创新高"] = (h == h.rolling(40, min_periods=1).max()).astype(int)

        # 突破确认：前一日收盘 == 21 日最高 且 涨幅 ≥ 5%，且当日满足 (收 > 开 或 量不缩)
        prev_close_21max = c.shift(1).rolling(21, min_periods=1).max() == c.shift(1)
        up_5pct = (c.shift(1) - o.shift(1)) / o.shift(1) >= 0.05
        cond1 = prev_close_21max & up_5pct
        cond2 = (c > o) | ~((v >= 0.5 * v.shift(1)) & (v <= 0.9 * v.shift(1)))
        out["突破确认"] = (cond1 & cond2).astype(int)

        out["red_brick"] = 0
        try:
            cond_rb = (
                (BH.shift(1) < BL.shift(1))
                & (BH > BL)
                & ((BH - BL) > 0.7 * (BH.shift(1) - BL.shift(1)).abs())
            )
            out.loc[cond_rb.fillna(False), "red_brick"] = 1
        except Exception:
            pass

        out["优选联盟成员"] = 0   # 没有名单时为 0；如需启用，在外部注入
        out["短期下跌未破位"] = 0  # 回测中暂不参与评分，可后续补
        out["Short_GCross_Normal"] = (ST > SL).astype(int)
        out["Short_GCross_Plus"] = 0
        out["Short_GCross_Pro"] = 0

        # 2026-06：活跃市值 AMV 多空信号。当前日线 CSV 没有 AMV 列，默认 0（中性）。
        # 等数据管线补上 AMV 列后，这里直接读 df["AMV"] 即可。
        out["AMV"] = 0

        # ---- 买入信号（任一为真即触发） ----
        buy = (
            out["J到负值-日线"]
            | out["补票-P1"]
            | out["补票-P2"]
            | out["股价跌穿L1线"]
            | out["股价触碰L2底线"]
            | out["突破确认"]
            | out["red_brick"]
        )
        out["buy_signal"] = buy.astype(int)
        return out

    # ---------- 选股 + 下单 ----------
    def build_target_signals(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        把"日频评分+信号"压缩成 entries/exits 矩阵：
            entries: 第 T 日开盘时，按 T-1 收盘后的综合得分排名，
                     取前 N 只（N 由当日可用现金档位决定），
                     当日有买入信号 + 评分 ≥ 阈值 + 当前未持仓。
            exits:   持仓超过 min_hold_for_gain 个交易日后，若涨幅不足 min_gain → 清仓
                     + 其它规则（30% 部分止盈 / 10% 阴线放量 / 5% 强制止损等）。
        """
        if self.signal_df is None or self.score_df is None:
            raise RuntimeError("请先调用 compute_signals()")

        # ---- T-1 综合得分排名 ----
        # 用 T-1 收盘后的得分，对 T 日开盘买入排序
        score_panel = self.score_df
        if isinstance(score_panel.columns, pd.MultiIndex):
            score_panel.columns = score_panel.columns.get_level_values(0)
        score_panel = score_panel.reindex(columns=self.signal_df.columns)

        # 滞后 1 天：T 日买入用的是 T-1 的得分
        lagged_score = score_panel.shift(1)
        # 当天确实有买入信号 + 得分过阈 + 活跃市值多头 的股票才纳入候选
        amv_long_mask = (self.amv_series.reindex(self.signal_df.index).fillna(0) == 1).values.reshape(-1, 1)
        buy_active = (
            (self.signal_df == 1)
            & (score_panel >= self.cfg.score_threshold)
            & amv_long_mask
        )
        # 用 T-1 得分对当日有 buy_active 的股票排名
        # 注意：把被屏蔽的位置填 NaN 而不是 -np.inf。
        # pd.rank 对一行全 -inf 仍会按位置给出 1.0/2.0/...，导致空头日
        # 绕过 amv_long_mask 错误产生买入信号；NaN 在 rank 后保持 NaN，
        # 后续 (rank > 0) 在屏蔽位置才返回 False。
        rank_today = lagged_score.where(buy_active, np.nan).rank(
            axis=1, ascending=False, method="first"
        )
        # 默认：候选 = 任何满足阈值但 rank 尚未定（先放宽松）
        raw_entries = (rank_today > 0).astype(int)

        # ---- 计算当日可用仓位数 N_i（现金档位） ----
        n_pos_today = self._compute_n_positions_per_day(raw_entries)
        # 把 raw_entries 截断到每天前 N_i 名
        capped_entries = (rank_today > 0) & (rank_today <= n_pos_today.values.reshape(-1, 1))
        raw_entries = capped_entries.astype(int)

        # entries + 卖出信号 + 持仓状态 一次性算好
        entries, exits = self._build_entries_exits_unified(raw_entries)
        return entries, exits

    def _compute_n_positions_per_day(self, raw_entries: pd.DataFrame) -> pd.Series:
        """
        根据用户规则 #1.3，按可用现金占总资产的比例决定当天能开的仓位数量：
            cash_ratio <  1%                → 0 仓（不交易）
            cash_ratio ∈ [1%, 30%]         → 1 仓
            cash_ratio ∈ (30%, 70%]        → 2 仓
            cash_ratio ∈ (70%, 100%]       → 3 仓

        注意：这里的"可用现金"是**当天开盘前**的估值。
        因为 T 日开盘买入，T 日开盘前还没有今天的订单发生，
        所以我们可以直接用"昨天的收盘后总资产 - 昨天持仓市值"。
        """
        close = self.close_df
        close_vals = close.reindex(columns=raw_entries.columns).values
        n_days = close_vals.shape[0]

        # 估算每天的开盘前总资产（用前一天收盘价 + 累计现金）
        # 简化：假设 init_cash = 全部可用现金，每天的净值按 close 走势估算
        # 用累计"已实现盈亏"：每天按 close / entry_value 缩放后近似
        # 这里采用最简版本：用前一天的 close 累计值估当天的总资产
        # 由于这是 N 的预判（用于决定仓位档），精度要求不高

        # 用初始资金 + 累计涨幅近似
        # 更精确的做法：扫描 entry/exit 后用 entry_value/cash 反算
        # 这里先用一个简单近似：假设每天的总资产 = 前一天 close 的累计均值 × 初始资金
        # 真正的总资产在 _build_size_matrix 里算，这里只算"可开仓位数"
        # 为简化：用 init_cash * (1 + 累计日均收益) 估总资产
        avg_daily_return = 0.0
        if n_days > 1:
            # 用前 60 天均值估计
            rets = close_vals[1:n_days] / close_vals[0:n_days-1] - 1
            rets = rets[~np.isnan(rets)]
            if len(rets) > 0:
                avg_daily_return = float(np.nanmean(rets))

        total_value_series = np.empty(n_days, dtype=np.float64)
        for i in range(n_days):
            total_value_series[i] = self.cfg.init_cash * ((1 + avg_daily_return) ** i)

        # 仓位数
        n_pos = np.zeros(n_days, dtype=np.int32)
        for i in range(n_days):
            # 估算"可用现金"。无持仓时就是全部 init_cash。
            # 这里用最简近似：可用现金 ≈ init_cash - sum(已开仓 entry_value)
            # 由于我们没有 entry_value 状态，先按 100% 可用算（保守给最大档）
            # 这会让 N 偏大，但 _build_size_matrix 会按真实现金二次截断
            cash = self.cfg.init_cash  # 简化
            cash_ratio = cash / total_value_series[i] if total_value_series[i] > 0 else 0
            if cash_ratio < 0.01:
                n_pos[i] = 0
            elif cash_ratio <= 0.30:
                n_pos[i] = 1
            elif cash_ratio <= 0.70:
                n_pos[i] = 2
            else:
                n_pos[i] = 3
        return pd.Series(n_pos, index=raw_entries.index, name="n_pos")

    def _build_entries_exits_unified(
        self, raw_entries: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        把原始 buy 信号 + 卖出规则合并到一次扫描里，输出 entries/exits。

        卖出规则（任一命中即触发 full exit；vbt.from_signals 不支持 partial sell）：
            rule1 持仓 > min_hold_for_gain 个交易日后，若涨幅仍不足 min_gain
                  → 当日收盘价全部清仓
            rule2 跌破成本 5% 强制止损 → 由 vbt 自身 sl_stop 处理（更精准）
            rule3 涨幅超过 30% → 全部清仓（理想：30% 部分止盈，
                  但 vbt.from_signals 不支持 cell-level fractional exit；
                  等价于"严格版 30% 部分止盈"，保守止盈出场）
            rule4 涨幅 > 10% + 当日阴线 + 当日成交量 = 近 45 日最大量
                  → 当日收盘价全仓卖出（典型顶部放量阴线）

        关键约束：
            1. 同一天若某只股票已经在持仓中，不重复开新仓。
            2. force-exit / rule3 / rule4 触发当日，禁止该股的 buy
               （vbt 的 from_signals 在 entries/exits 同一天都为 True 时会忽略 exit）。
            3. rule3 / rule4 当日下一天开盘时不算持仓。
        """
        # 同时需要 open（rule4 阴线判断）、high、low、close、volume（rule4 量能）
        close_df = self.close_df
        open_df = self.data_loader.load_full  # placeholder, real data via features

        # 把 close / open / volume 从原始数据里读出来
        raw_data_dir = self.cfg.raw_data_dir
        loader = ZDataLoader(data_dir=raw_data_dir)
        # 用 close_df 的列顺序对齐
        cols = list(raw_entries.columns)
        open_vals = np.zeros_like(close_df.reindex(columns=cols).values)
        vol_vals = np.zeros_like(open_vals)
        for j, sym in enumerate(cols):
            df = loader.load_full(sym)
            if df is None or df.empty:
                continue
            df = df.loc[df.index.isin(raw_entries.index)]
            for i, dt in enumerate(raw_entries.index):
                if dt in df.index:
                    o = df.loc[dt, 'open'] if 'open' in df.columns else np.nan
                    v = df.loc[dt, 'volume'] if 'volume' in df.columns else np.nan
                    if np.isfinite(o):
                        open_vals[i, j] = o
                    if np.isfinite(v):
                        vol_vals[i, j] = v

        raw = raw_entries.values
        n_days, n_syms = raw.shape
        entries = np.zeros_like(raw, dtype=np.int8)
        exits = np.zeros_like(raw, dtype=np.int8)

        in_pos = np.zeros(n_syms, dtype=bool)
        entry_idx = np.full(n_syms, -1, dtype=np.int32)
        entry_price = np.full(n_syms, np.nan, dtype=np.float64)

        min_hold = int(self.cfg.min_hold_for_gain)
        min_gain = float(self.cfg.min_gain)
        gain_30 = float(self.cfg.gain_30_threshold)
        gain_10 = float(self.cfg.gain_10_threshold)
        vol_lookback = int(self.cfg.volume_lookback)

        close_vals = close_df.reindex(columns=cols).values

        # 把"AMV 由 1 翻转到 -1 的首日"提前转成 set，O(1) 查询
        amv_flip_set = (
            set(self.amv_flip_to_short) if self.amv_flip_to_short is not None else set()
        )

        for i in range(n_days):
            buy_candidate = raw[i].astype(bool)

            # ===== 1) 卖出规则（在已知 in_pos 状态下判断） =====
            # 2026-06 新增：活跃市值由 +1 翻转到 -1 的首日 → 持仓全卖
            amv_force_exit_today = np.zeros(n_syms, dtype=bool)
            if raw_entries.index[i] in amv_flip_set:
                amv_force_exit_today = in_pos.copy()

            if min_hold > 0 and i > 0:
                hold_days = i - entry_idx
                gain = np.where(
                    in_pos & np.isfinite(entry_price) & (entry_price > 0),
                    (close_vals[i] - entry_price) / entry_price,
                    -np.inf,  # 默认 -inf 让 rule3/rule4 不会乱触发
                )

                # rule1：持仓 >= min_hold 且 涨幅 < min_gain → 全仓
                cond_rule1 = in_pos & (hold_days >= min_hold) & (gain < min_gain)
                # rule3：涨幅 > 30% → 全仓（保守：vbt 不支持 partial sell）
                cond_rule3 = in_pos & (gain > gain_30)
                # rule4：涨幅 > 10% + 当日阴线（close < open）+ 当日量 = 近 vol_lookback 日最大量
                cond_red = (close_vals[i] < open_vals[i]) & np.isfinite(open_vals[i]) & np.isfinite(close_vals[i])
                cond_high_vol = np.zeros(n_syms, dtype=bool)
                for s in range(n_syms):
                    if not in_pos[s] or not np.isfinite(vol_vals[i, s]):
                        continue
                    if i >= vol_lookback:
                        past_max = float(np.nanmax(vol_vals[max(0, i - vol_lookback):i, s]))
                    else:
                        past_max = float(np.nanmax(vol_vals[:i, s])) if i > 0 else 0.0
                    cond_high_vol[s] = (vol_vals[i, s] >= past_max) and (past_max > 0)
                cond_rule4 = in_pos & (gain > gain_10) & cond_red & cond_high_vol

                exits[i] = (cond_rule1 | cond_rule3 | cond_rule4 | amv_force_exit_today).astype(np.int8)
            else:
                # i == 0 时无 min_hold 判断可走，但仍可能有 AMV 翻转日 force-exit
                exits[i] = amv_force_exit_today.astype(np.int8)

            # ===== 2) 处理今天的 buy =====
            in_pos_today = in_pos & (exits[i] == 0)
            allowed_buy = buy_candidate & (~in_pos_today) & (exits[i] == 0)
            entries[i] = allowed_buy.astype(np.int8)

            # ===== 3) 更新持仓状态 =====
            in_pos = in_pos | allowed_buy
            entry_idx = np.where(allowed_buy, i, entry_idx)
            entry_price = np.where(
                allowed_buy & np.isfinite(close_vals[i]), close_vals[i], entry_price
            )
            in_pos = in_pos & (exits[i] == 0)

        entries_df = pd.DataFrame(entries, index=raw_entries.index, columns=cols)
        exits_df = pd.DataFrame(exits, index=raw_entries.index, columns=cols)
        return entries_df, exits_df

    def _build_size_matrix(
        self, entries: pd.DataFrame, exits: pd.DataFrame
    ) -> pd.DataFrame:
        """
        构造一个 2D size 矩阵（date × symbol），送入 vbt.from_signals 的 size 参数。
        每个 entry 单元格填的是"该笔买入占总资产的目标比例"（vbt 的 SizeType.Percent）。

        用户原话（重新整理）：
            1. 每支股票仓位占总资金的 20%~30%
            2. 每次买入卖出之后，更新计算可用现金，按可用现金分档：
                cash < 1%  total_value             → 0 仓（不交易）
                cash ∈ [1%, 30%] total_value        → 1 仓
                cash ∈ (30%, 70%] total_value       → 2 仓
                cash ∈ (70%, 100%] total_value      → 3 仓
            3. 仓位数 N 由 build_target_signals 通过 _compute_n_positions_per_day
               预先确定（在 entries 阶段已经限定为前 N 名）。
            4. size 矩阵每个 entry 单元格 = min(high=30%, cash/total_value/N)，
               但要保证单仓 ≥ 20% (low)；不够 20% 时吃掉剩余现金（仍算 1 个仓位）。

        实现思路：
            * 状态：(in_pos[s], entry_price[s], entry_value[s])
            * 每天顺序：先算当前总资产 → 给每个今日 buy 分配 size
              → 更新状态
        """
        close = self.close_df.reindex(columns=entries.columns)
        close_vals = close.values
        e = entries.values.astype(bool)
        n_days, n_syms = e.shape

        size_pct = np.zeros((n_days, n_syms), dtype=np.float64)
        low = float(self.cfg.target_weight_low)        # 0.20
        high = float(self.cfg.target_weight_high)      # 0.30
        init_cash = float(self.cfg.init_cash)

        in_pos = np.zeros(n_syms, dtype=bool)
        entry_price = np.full(n_syms, np.nan, dtype=np.float64)
        entry_value = np.zeros(n_syms, dtype=np.float64)

        for i in range(n_days):
            # ===== 估算当日总资产 =====
            held_value = 0.0
            for s in range(n_syms):
                if in_pos[s] and entry_price[s] > 0 and np.isfinite(close_vals[i, s]):
                    held_value += entry_value[s] * (close_vals[i, s] / entry_price[s])
            total_invested = float(entry_value[in_pos].sum())
            cash = init_cash - total_invested + held_value
            total_value = held_value + cash
            if total_value <= 0:
                total_value = init_cash

            # ===== 当日 buy 仓位分配 =====
            buys_today = e[i]
            n_buys_today = int(buys_today.sum())
            if n_buys_today > 0:
                # 用户规则：每仓 20~30% 总资金。
                # 默认按 high（30%）配置；如果 N × high > 100%，降配
                # 但最后一个仓位吃剩余（按用户原话"剩余资金按一个仓位算"）
                target_total = high * n_buys_today
                if target_total <= 1.0:
                    # 资金够：每个 buy 占 30%
                    size_pct[i] = np.where(buys_today, high, 0.0)
                else:
                    # 前 N-1 个 30%，最后 1 个吃剩余
                    size_pct[i] = np.where(buys_today, high, 0.0)
                    buy_indices = np.where(buys_today)[0]
                    residual_pct = max(0.0, 1.0 - (n_buys_today - 1) * high)
                    size_pct[i, buy_indices[-1]] = residual_pct

            # ===== 更新 in_pos / entry_price / entry_value =====
            for s in range(n_syms):
                if buys_today[s] and size_pct[i, s] > 0:
                    pv = size_pct[i, s] * total_value
                    in_pos[s] = True
                    entry_price[s] = close_vals[i, s] if np.isfinite(close_vals[i, s]) else 0.0
                    entry_value[s] = pv
            entry_price = np.where(in_pos, entry_price, np.nan)
            entry_value = np.where(in_pos, entry_value, 0.0)

        return pd.DataFrame(size_pct, index=entries.index, columns=entries.columns)

    def _build_exits_vectorized(self, entries: pd.DataFrame) -> pd.DataFrame:
        """
        一次性计算所有股票的卖出信号。

        卖出规则（任一命中即触发）：
          1) 持仓超过 `min_hold_for_gain` 个交易日后，若涨幅仍不足 `min_gain`，
             全部清仓（这是用户新加的规则）。
          2) 跌破成本 5% 强制止损 — 由 vbt 自身通过 `sl_stop` 实现。

        实现思路：
          * 逐行扫描 entries + 当前"持仓状态"。
          * 状态变量 in_pos[s] 表示"今天开盘时是否持仓"（昨天未退出）。
          * 当 buy_today 触发，重置入场日/入场价。
          * 当 exit_today 触发（force exit），把 in_pos[s] 在下一天置 False。
        """
        close = self.close_df
        e = entries.values
        n_days, n_syms = e.shape
        exits = np.zeros_like(e, dtype=np.int8)

        min_hold = int(self.cfg.min_hold_for_gain)
        min_gain = float(self.cfg.min_gain)

        in_pos = np.zeros(n_syms, dtype=bool)
        entry_idx = np.full(n_syms, -1, dtype=np.int32)
        entry_price = np.full(n_syms, np.nan, dtype=np.float64)

        close_vals = close.reindex(columns=entries.columns).values

        for i in range(n_days):
            buy_today = (e[i] == 1)

            # 收盘后：先看是否需要"持仓 N 日 + 涨幅不足 X%"清仓
            if min_hold > 0 and i > 0:
                hold_days = i - entry_idx
                gain = np.where(
                    in_pos & np.isfinite(entry_price) & (entry_price > 0),
                    (close_vals[i] - entry_price) / entry_price,
                    np.inf,
                )
                cond_force_exit = (
                    in_pos & (hold_days >= min_hold) & (gain < min_gain)
                )
                exits[i] = cond_force_exit.astype(np.int8)
            else:
                exits[i] = 0

            # 收盘后：处理 buy 信号。buy 当日不会同时触发 force exit。
            in_pos = in_pos | buy_today
            entry_idx = np.where(buy_today, i, entry_idx)
            entry_price = np.where(
                buy_today & np.isfinite(close_vals[i]), close_vals[i], entry_price
            )

            # 当日发生 exit，下一天开盘时 in_pos 应为 False
            in_pos = in_pos & (exits[i] == 0)

        return pd.DataFrame(exits, index=entries.index, columns=entries.columns)

    # ---------- 资金管理：每个仓位 = 总资产的 20~30% ----------
    def _order_func(self, c: pd.Series, close: pd.DataFrame):
        """
        vectorbt order_func：按当日目标价值下等比仓位。
        规则：
            entry=1：买入，使该仓位接近 总资产 * target_weight
            exit=1 ：全仓卖出
        """
        # 该函数会被 vectorbt 对每 (col, row) 单独调用
        col = c.col  # type: ignore
        i = c.i  # type: ignore
        # 当前已实现仓位价值
        # 用 portfolio 提供的方法获取最新价值
        # 简化处理：直接使用 close 价格推算可买股数
        price = close.iloc[i][col]
        return vbt.portfolio.enums.Order(
            size=np.nan,  # 暂时交给 vbt size 处理
        )

    def _make_size_func(self, equity: pd.Series, entries: pd.DataFrame, exits: pd.DataFrame):
        """
        构造一个 callable: (close, col, row) -> target size
        目标仓位价值 = 总资产 * 0.20~0.30，按评分加权分配。
        """
        target_value_low = self.cfg.init_cash * self.cfg.target_weight_low
        target_value_high = self.cfg.init_cash * self.cfg.target_weight_high

        def _size_func(c: pd.Series) -> float:
            return np.nan  # 暂时不用

        return _size_func

    # ---------- 主回测流程 ----------
    def run(self) -> "BacktestEngine":
        cfg = self.cfg
        symbols = load_stockpool(cfg.stockpool_xlsx)
        print(f"✅ 股票池: {len(symbols)} 只（{os.path.basename(cfg.stockpool_xlsx)}）")

        # 1) 行情
        self.close_df = self.load_market_data(symbols)
        print(f"✅ 行情加载完成: {self.close_df.shape[0]} 个交易日 × {self.close_df.shape[1]} 只股票")

        # 2) 信号 & 评分
        self.compute_signals([s for s in symbols if s in self.close_df.columns])

        # 2.5) 加载活跃市值多空过滤信号
        self._load_amv_signal()
        n_long = int((self.amv_series == 1).sum()) if self.amv_series is not None else 0
        n_short = int((self.amv_series == -1).sum()) if self.amv_series is not None else 0
        n_flip = len(self.amv_flip_to_short) if self.amv_flip_to_short is not None else 0
        if self.cfg.amv_disable_filter:
            print(f"✅ 活跃市值过滤：已禁用（amv_disable_filter=True）")
        else:
            print(
                f"✅ 活跃市值: 多头 {n_long} 天 / 空头 {n_short} 天 / 空头翻转 {n_flip} 次"
            )

        # 3) entries / exits
        entries, exits = self.build_target_signals()
        print(f"✅ 买入信号总触发数: {entries.values.sum()}")
        print(f"✅ 卖出信号总触发数: {exits.values.sum()}")

        # 4) vectorbt 回测
        # 使用 from_signals + size_type='percent' 让 vbt 内部按 c.value_now 自动换算成股数
        # 这一步无需自定义 order_func，从而绕开 vbt 1.0 + numba 的类型推导坑
        # 单仓位目标占比 = 1.0 / max_positions（5），会被 vbt 限制在初始资金上下文中
        close = self.close_df
        # 把 entries / exits 对齐到 close 的 index + columns
        # 否则 vbt 1.0 会因为多重索引无法广播而抛 ValueError
        common_idx = self.signal_df.index.intersection(close.index)
        common_cols = self.signal_df.columns.intersection(close.columns)
        entries = entries.loc[common_idx, common_cols]
        exits = exits.loc[common_idx, common_cols]
        close = close.loc[common_idx, common_cols]
        # entries / exits 已是 int，转成 bool 给 vbt
        entries_bool = entries.astype(bool)
        exits_bool = exits.astype(bool)
        # 单仓 size 矩阵：每个 entry 单元格 = 该笔买入占总资产的目标比例
        # 默认 30%（区间上沿），剩 < 20% 时不开新仓
        size_pct = self._build_size_matrix(entries, exits)
        # 把 size 矩阵也对齐到 close 的 index + columns
        size_pct = size_pct.loc[common_idx, common_cols]
        granularity = self.cfg.size_granularity

        if cfg.use_gpu and _HAS_CUPY:
            self.portfolio = self._run_with_gpu(close, entries, exits)
        else:
            self.portfolio = vbt.Portfolio.from_signals(
                close=close,
                entries=entries_bool,
                exits=exits_bool,
                size=size_pct,
                size_type=vbt.portfolio.enums.SizeType.Percent,
                init_cash=cfg.init_cash,
                cash_sharing=True,
                freq="D",
                call_seq=vbt.portfolio.enums.CallSeqType.Default,
                fees=cfg.fees,
                slippage=cfg.slippage,
                size_granularity=granularity,
                # 5% 强制止损：跌破持仓成本 5% 即自动平仓
                sl_stop=cfg.sl_stop,
                tp_stop=cfg.tp_stop,
                sl_trail=cfg.sl_trail,
            )

        # 5) 统计
        self.trades = self.portfolio.trades.records_readable
        self.stats = self.portfolio.stats(agg_func=None).reset_index()
        self.stats.columns = ["Metric", "Value"]
        self.equity = self.portfolio.value()

        # 2026-06 新增：把 AMV 多空统计挂到 stats 末尾，
        # 供 PDF 中文摘要 / Excel 关键指标使用
        if self.amv_series is not None:
            n_long = int((self.amv_series == 1).sum())
            n_short = int((self.amv_series == -1).sum())
            n_flip = len(self.amv_flip_to_short) if self.amv_flip_to_short is not None else 0
            self.stats = pd.concat([
                self.stats,
                pd.DataFrame({
                    "Metric": [
                        "AMV 多头天数",
                        "AMV 空头天数",
                        "AMV 空头翻转次数",
                    ],
                    "Value": [n_long, n_short, n_flip],
                }),
            ], ignore_index=True)

        return self

    def _run_with_gpu(self, close, entries, exits):
        """
        备选的 GPU 回测路径。当且仅当环境变量 VBT_USE_GPU=1 且 CuPy 可用时调用。
        注意：vectorbt 自身不支持 GPU；这里用 CuPy 自实现一个最简"等权再平衡"
        回测逻辑作为示意，实际生产中应替换为 numba/cupy 优化版本。
        """
        import cupy as cp
        # 把数据搬运到 GPU
        c_arr = cp.asarray(close.fillna(method="ffill").values, dtype=cp.float32)
        e_arr = cp.asarray(entries.values, dtype=cp.int8)
        x_arr = cp.asarray(exits.values, dtype=cp.int8)
        n_days, n_stocks = c_arr.shape

        cash = cp.float32(self.cfg.init_cash)
        shares = cp.zeros(n_stocks, dtype=cp.float32)
        nav = cp.zeros(n_days, dtype=cp.float32)
        target_value = self.cfg.init_cash / self.cfg.max_positions

        for t in range(n_days):
            price_t = c_arr[t]
            # 卖出
            for s in range(n_stocks):
                if x_arr[t, s] == 1 and shares[s] > 0:
                    cash += shares[s] * price_t[s] * (1 - self.cfg.fees - self.cfg.slippage)
                    shares[s] = 0
            # 买入
            for s in range(n_stocks):
                if e_arr[t, s] == 1 and shares[s] == 0 and price_t[s] > 0:
                    cur_val = float(cash + cp.sum(shares * price_t))
                    unit = max(cur_val / n_stocks, self.cfg.init_cash * self.cfg.target_weight_low)
                    unit = min(unit, self.cfg.init_cash * self.cfg.target_weight_high)
                    if cash >= unit:
                        qty = int(unit / float(price_t[s]) / self.cfg.size_granularity) * self.cfg.size_granularity
                        if qty > 0:
                            shares[s] = qty
                            cash -= qty * float(price_t[s]) * (1 + self.cfg.fees + self.cfg.slippage)
            nav[t] = cash + cp.sum(shares * price_t)
        # 这里返回的 nav 仍是 cupy，需要拉回 CPU
        nav_np = cp.asnumpy(nav)
        equity = pd.Series(nav_np, index=close.index, name="value")
        return _EquityWrapper(equity, close)

    # ---------- 报告输出 ----------
    def export_reports(self) -> Dict[str, str]:
        if self.portfolio is None:
            raise RuntimeError("请先调用 run()")

        # 把每笔交易记录标准化
        trades = self.trades.copy() if self.trades is not None else pd.DataFrame()
        # vbt 1.0 trades.records_readable 列：Entry Timestamp / Avg Entry Price / Size / Exit Timestamp / Avg Exit Price / Return / Symbol
        if "Size" in trades.columns:
            trades.rename(columns={"Size": "买入股数"}, inplace=True)

        # 构造"每日 Top 10 综合得分"明细
        daily_top_scores = self._build_daily_top_scores(top_n=10)

        # PDF 交易明细需要"仓位比例"列
        trades_with_pct = self._augment_trades_with_position_pct(trades)

        excel_path = self.report.write_excel(
            self.stats, trades, self.equity,
            self.equity.pct_change().dropna(),
            self.cfg,
            daily_top_scores=daily_top_scores,
        )
        # PDF 是可选的：reportlab 不可用时跳过，但不影响 Excel
        if not _HAS_REPORTLAB:
            print("⚠️  跳过 PDF 报告（reportlab 不可用）")
            return {"excel": excel_path, "pdf": None}
        pdf_path = self.report.write_pdf(
            self.stats, trades_with_pct, self.equity,
            benchmark=None, cfg=self.cfg,
        )
        return {"excel": excel_path, "pdf": pdf_path}

    def _build_daily_top_scores(self, top_n: int = 10) -> pd.DataFrame:
        """
        按日输出综合得分 TopN 的股票明细，每行一只股票，列名与 SignalScan 报告保持一致。
        返回一个 MultiIndex (date, rank) 的 DataFrame，便于 export_reports 按日期分组写到不同 sheet。
        """
        if self.score_df is None or self.feature_wide is None:
            return pd.DataFrame()

        # 列对齐
        score = self.score_df.reindex(columns=self.feature_wide.get(
            'J到负值-日线', pd.DataFrame(0, index=self.score_df.index, columns=self.score_df.columns)
        ).columns)
        if score.empty:
            return pd.DataFrame()

        # 股票名称映射（来自股票池）
        name_map = ReportGenerator._load_stockpool_name_map(self.cfg)

        # 一次性把所有需要的特征拉成长表
        feature_keys = [
            'J到负值-日线', 'J值-日线', 'J值反转-日线',
            '补票-P1', '补票-P2', '长线资金指标',
            'Short_GCross_Normal', 'Short_GCross_Plus', 'Short_GCross_Pro',
            'BBI线上', 'BBI上涨趋势-5日', 'BBI上涨趋势-20日',
            '股价跌穿L1线', '股价触碰L2底线',
            '股价位于R1区间', '股价位于R2区间',
            '股价位于R3区间', '股价位于R4区间',
            '股价创新高', '突破确认',
            '短期下跌未破位', 'red_brick',
            # '优选联盟成员' 和 '低波红利' 已删除
            # 这两列依赖外部清单（SignalScan 中的 .xlsx 文件），
            # 中证A500 股票池没有这个字段，Top10 sheet 中不再显示。
        ]
        out_rows = []
        for dt in score.index:
            day_scores = score.loc[dt]
            # 用户原话：取消阈值过滤，直接取综合得分排名前 10
            # 过滤掉 NaN / -inf（没评分的股票）
            cand = day_scores.replace([np.inf, -np.inf], np.nan).dropna().nlargest(top_n)
            if cand.empty:
                continue
            for rank_i, (sym, sc) in enumerate(cand.items(), 1):
                row = {
                    '排名': rank_i,
                    '股票代码': sym,
                    '股票名称': name_map.get(str(sym).zfill(6), '未知'),
                    '综合得分': round(float(sc), 2),
                }
                # 加每个特征列
                for fk in feature_keys:
                    w = self.feature_wide.get(fk)
                    if w is not None and sym in w.columns and dt in w.index:
                        v = w.loc[dt, sym]
                        row[fk] = round(float(v), 2) if isinstance(v, (int, float)) and pd.notna(v) else (v if pd.notna(v) else 0)
                    else:
                        row[fk] = 0
                row['_date'] = pd.Timestamp(dt)
                out_rows.append(row)

        if not out_rows:
            return pd.DataFrame()
        df_out = pd.DataFrame(out_rows)
        df_out = df_out.set_index('_date', drop=True)
        df_out.index.name = '日期'
        return df_out

    def _augment_trades_with_position_pct(self, trades: pd.DataFrame) -> pd.DataFrame:
        """
        给 trades DataFrame 增加"仓位比例"列（该笔买入占总资产的比例）。
        使用 vbt 的 SizeType.Percent 语义：vbt 用每笔订单当时的 c.value_now × size_pct 计算股数，
        所以"占比"近似 = entry_value / total_value_at_entry。
        我们用 entry_value（构造 size matrix 时记录的投入）除以当日总资产估算值。
        """
        if trades is None or trades.empty or self.feature_wide is None:
            return trades

        out = trades.copy()
        # 估算每笔交易日的"总资产"
        close = self.close_df
        # 用 self._build_size_matrix 内部状态
        # 这里简单做法：直接复用 self._build_size_matrix 输出的对角线单元格
        # 实际上 trades 已有 entry_value（如果 trades 中包含该列），否则从 vbt 拿
        # 用近似：position_pct = entry_value / (equity_at_entry)
        if "仓位比例" in out.columns:
            return out
        # 取初始资金 + 累计盈亏做粗略估算
        # 更精确：从 self.entry_value 字典查（如果 size matrix 计算时存了）
        # 这里采取 vbt trades 自带的 "Size" * "Avg Entry Price" / init_cash
        # 缺点：忽略了组合中其它持仓，所以会低估
        eq = self.equity
        def _pct(row):
            try:
                ts = pd.Timestamp(row['Entry Timestamp'])
                # 找到 equity 中最接近的索引
                if ts in eq.index:
                    total_at_entry = float(eq.loc[ts])
                else:
                    # 插值
                    idx = eq.index.get_indexer([ts], method='ffill')[0]
                    total_at_entry = float(eq.iloc[idx])
                cost = float(row['Size'] if 'Size' in row.index else row.get('买入股数', 0)) * float(row['Avg Entry Price'])
                if total_at_entry > 0:
                    return cost / total_at_entry
            except Exception:
                pass
            return 0.0
        out['仓位比例'] = out.apply(_pct, axis=1)
        return out


# =====================================================================
# 5.5 GPU 可用性探测
# =====================================================================
try:
    import cupy as cp  # noqa: F401
    _HAS_CUPY = True
except Exception:
    _HAS_CUPY = False


class _EquityWrapper:
    """GPU 回测返回的最小兼容对象，仅暴露 trades / stats / value"""
    def __init__(self, equity: pd.Series, close: pd.DataFrame):
        self._equity = equity
        self._close = close

    @property
    def value(self) -> pd.Series:
        return self._equity

    @property
    def trades(self):
        class _T:
            records_readable = pd.DataFrame(columns=[
                "Entry Timestamp", "Avg Entry Price", "Size",
                "Exit Timestamp", "Avg Exit Price", "Return", "Symbol",
            ])
        return _T()

    def stats(self, agg_func=None):
        return pd.DataFrame({
            "Metric": ["Total Return", "Sharpe Ratio", "Max Drawdown", "Win Rate"],
            "Value": [0.0, 0.0, 0.0, 0.0],
        })


# =====================================================================
# 6. 强化学习适配器（BacktestEnv）
# =====================================================================
class BacktestEnv:
    """
    面向强化学习的回测环境（gym 风格）。
        env = BacktestEnv(BacktestConfig())
        obs = env.reset()
        for _ in range(episodes):
            weights = agent.act(obs)
            obs, reward, done, info = env.step(weights)
    step() 内部仅重算评分与 vectorbt 回测；信号缓存不重算。
    """

    def __init__(self, cfg: BacktestConfig) -> None:
        self.cfg = cfg
        self.engine = BacktestEngine(cfg)
        self.engine.compute_signals(load_stockpool(cfg.stockpool_xlsx))
        self.observation_space_shape = (len(self.engine.signal_df.columns),)

    def reset(self) -> np.ndarray:
        self.engine.strategy.weights = ZWeights()
        self.engine._compute_scores_with_current_weights()
        return self.engine.score_df.fillna(0).values

    def step(self, weights_vec: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        self.engine.strategy.weights.set_weights(weights_vec)
        self.engine._compute_scores_with_current_weights()
        # RL 训练时也要保留 AMV 过滤（与回测环境保持一致）
        self.engine._load_amv_signal()
        self.engine.run()
        # reward = 总收益率 - 1.0 * 最大回撤（可调）
        total_return = float(self.engine.portfolio.total_return())
        max_dd = float(self.engine.portfolio.max_drawdown())
        reward = total_return - abs(max_dd)
        done = True  # 单次回测即结束
        info = {
            "total_return": total_return,
            "max_drawdown": max_dd,
            "trades": int(len(self.engine.trades)),
        }
        return self.engine.score_df.fillna(0).values, reward, done, info


# =====================================================================
# 7. 主入口
# =====================================================================
def main():
    print("=" * 70)
    print(" Z 策略回测 (沪深300) ")
    print("=" * 70)
    cfg = BacktestConfig()
    print(f"回测区间: {cfg.start_date} ~ {cfg.end_date}")
    print(f"股票池: {cfg.stockpool_xlsx}")
    print(f"初始资金: {cfg.init_cash:,.0f} 元")
    print(f"仓位: {cfg.max_positions} 个, 占比 {cfg.target_weight_low*100:.0f}% ~ {cfg.target_weight_high*100:.0f}%")
    print(f"GPU 加速: {'开启' if cfg.use_gpu and _HAS_CUPY else '关闭（CPU/Numba）'}")
    print("=" * 70)

    t0 = time.time()
    engine = BacktestEngine(cfg).run()
    paths = engine.export_reports()
    t1 = time.time()

    print("\n回测完成，耗时: {:.1f}s".format(t1 - t0))
    print(f"📊 Excel 报告: {paths['excel']}")
    if paths.get("pdf"):
        print(f"📄 PDF 报告:   {paths['pdf']}")
    else:
        print("📄 PDF 报告:   （未生成，缺少 reportlab）")
    return engine


if __name__ == "__main__":
    main()
