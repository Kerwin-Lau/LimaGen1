# -*- coding: utf-8 -*-
"""
Z-Strategy.py
=============
Z 策略的核心实现模块：负责生成买入/卖出信号、计算股票综合评分。
本文件被以下两个调用方使用：
    1. SignalScan_Ashares_Z-Strategy - Temp.py  （盘后信号扫描）
    2. AShares_BackTest_20260615.py              （回测框架）

设计原则：
    * 把"信号生成"和"评分计算"封装成可替换的策略类 StrategyBase，
      未来新增策略时只需继承本类并实现 generate_signals / compute_score
      接口即可达到"灵活切换"的目的；
    * 把所有的可调权重（WEIGHTS）集中到一个 dataclass 中，
      以便后续接入强化学习（RL）来动态优化这些权重。
      强化学习侧只需做：
          new_weights = rl_agent.act(state)
          strategy.set_weights(new_weights)
      即可立刻生效，无需修改任何业务代码。

数据结构约定：
    输入 data 需至少包含以下列：
        open / high / low / close / volume
        J / short_term_fund / long_term_fund
        BBI / BBI_DIF
        L1 / L2 / M / H1 / H2
        Brick_High / Brick_Low
        Short_LS / Short_Trend
    这些字段全部由 02_DataProcess 模块预处理后存入
    D:\\Quant\\01_SwProj\\04_VectorBT\\02_Lima\\Lima_Gen1\\01_Database\\01_Ashares\\01_RawData-Daily
    中的每日 CSV 中。
"""

from __future__ import annotations

import os
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional, List, Iterable

import numpy as np
import pandas as pd


# ====================================================================
# 1. 路径常量（被调用方共享）
# ====================================================================
PROJECT_ROOT = r"D:\Quant\01_SwProj\04_VectorBT\02_Lima\Lima_Gen1"

# 原始日线数据根目录
DEFAULT_RAW_DATA_DIR = os.path.join(
    PROJECT_ROOT, "01_Database", "01_Ashares", "01_RawData-Daily"
)

# 股票池文件目录（沪深300 等）
DEFAULT_STOCKPOOL_DIR = os.path.join(
    PROJECT_ROOT, "05_BackTest", "01_Ashares", "01_List"
)

# 信号扫描报告输出目录
DEFAULT_SCAN_RESULT_DIR = os.path.join(
    PROJECT_ROOT, "03_SignalScan", "01_Ashares", "01_Z-Strategy", "01_Report"
)

# 数据更新脚本
DAILY_UPDATE_SCRIPT = os.path.join(
    PROJECT_ROOT,
    "02_DataProcess", "01_Ashares", "01_DaliyUpdate",
    "DataUpdate_ASharesDaily.py"
)


# ====================================================================
# 2. 权重定义（dataclass 形式，便于 RL 动态注入）
# ====================================================================
@dataclass
class ZWeights:
    """
    Z 策略的评分权重。
    使用 dataclass 的好处：
        1. 强化学习 agent 可以直接调用 `set_weights(vector)`；
        2. to_dict() / from_dict() 方便持久化到文件/数据库；
        3. 字段顺序固定，避免 dict 顺序带来的隐患。

    ----------------------------------------------------------------------
    ⚠️ 权重来源说明（重要）
    ----------------------------------------------------------------------
    以下默认权重不再是手工调参得到的初始值，而是经过 3 轮 CMA-ES
    （协方差矩阵自适应进化策略）在中证 A500 股票池上跑 30 代 × 16 个体
    优化得到的最优权重组合（**22 维**）。

    训练配置：
        算法          : CMA-ES (cma 4.4.4)
        训练区间      : 2025-06-25 ~ 2025-09-03（100 只抽样股票）
        验证区间      : 2025-12-09 ~ 2026-01-28（500 只全量）
        测试区间      : 2026-04-08 ~ 2026-05-25（500 只全量）
        权重搜索范围  : init_value × [0.01, 5.0]   ← 用户 2026-06 放宽
        训练脚本      : 06_RL/01_Ashares/01_Alpha/RL_Alpha_AShares.py
        训练产物      : 06_RL/01_Ashares/01_Alpha/outputs/best_weights.json

    训练结果（End Value，初始资金 50 万）：
        v1 (18 维, init×[0.5, 2.0]) : 训练 603,932 / 验证 644,659 / 测试 604,499
        v2 (21 维, init×[0.5, 2.0]) : 训练 613,393 / 验证 629,862 / 测试 561,591
        v3 (21 维, init×[0.01, 5.0]): 训练 616,192 / 验证 641,226 / 测试 661,844  ← 21 维基线
        v4 (22 维)                : 准备中（本轮新增 amvl_wi/amvs_wi，未重训）
        总耗时 : 39.2 分钟（8 worker 并行）

    维度变化历史：
        v1 : 18 维（j/bp/bbi/peb/bt/pa）
        v2 : 21 维（+yw_wi_1/2/3 短线金叉）
        v4 : 22 维（-pa_wi_1, +amvl_wi/amvs_wi 活跃市值多空信号）
    ----------------------------------------------------------------------
    ⚠️ 关于第 4 步（active market value, AMV）的不一致提醒
    ----------------------------------------------------------------------
    AMV 来自日线 CSV 的 'AMV' 列（+1 多头 / -1 空头 / 0 或缺失 中性）。
    本次只改 ZStrategy（数据管线 + 回测 + RL 训练器都未同步改）。
    实际效果：
        * 当前 CSV 没有 AMV 列 → amv_val 永远 = 0 → 此因子不贡献分数
        * 后续数据管线补上 AMV 列后，会自动生效
        * 部署 SignalScan 报告和回测结果会再次出现不一致（与之前 pa_wi_1 情形类似）
    ----------------------------------------------------------------------
    """

    # ----- J 值相关 -----
    j_wi_1: float = 45.00   # J 到负值（init=13.81 → 3.26x，放大近 3.3 倍）
    j_wi_2: float = 23.90   # J 值（init=24.70 → 0.97x，几乎不变）
    j_wi_3: float = 25.28   # J 反转（init=5.06 → 5.00x，**顶到约束上限**）

    # ----- 资金补票 -----
    bp_wi_1: float = 17.49  # 补票 P1（init=7.90 → 2.21x，明显放大）
    bp_wi_2: float = 60.27  # 补票 P2（init=14.27 → 4.22x，大幅放大）
    bp_wi_3: float = 3.13   # 长线资金（init=13.54 → 0.23x，大幅压低）

    # ----- BBI 趋势 -----
    bbi_wi_1: float = 30.75  # BBI 线上（init=11.10 → 2.77x，明显放大）
    bbi_wi_2: float = 35.20  # BBI 5 日趋势（init=7.24 → 4.86x，**接近约束上限**）
    bbi_wi_3: float = 11.77  # BBI 20 日趋势（init=6.38 → 1.85x，放大近 2 倍）

    # ----- 价位区间 -----
    peb_wi_1: float = 53.62  # 跌穿 L1（init=22.07 → 2.43x，明显放大）
    peb_wi_2: float = 30.65  # 触碰 L2（init=26.72 → 1.15x，几乎不变）
    peb_wi_3: float = 8.50   # 位于 R1（init=12.65 → 0.67x，砍 1/3）
    peb_wi_4: float = 8.48   # 位于 R2（init=7.84 → 1.08x，几乎不变）
    peb_wi_5: float = -12.32 # 位于 R3（init=-8.96 → 1.37x 绝对值，加大高位减分力度）
    peb_wi_6: float = -30.48 # 位于 R4（init=-19.89 → 1.53x 绝对值，进一步强化高位减分）

    # ----- 突破 -----
    bt_wi_1: float = 11.74  # 创新高（init=9.63 → 1.22x，略升）
    bt_wi_2: float = 88.57  # 突破确认（init=39.83 → 2.22x，放大 2.2 倍）

    # 注意：原"优选联盟" pa_wi_1 已删除（2026-06 用户需求）
    #       CMA-ES v3 训练结果认为该因子贡献为 0，应直接关掉
    # ----- 短线金叉（2026-06 用户新增需求，对应 SignalScan 报告的 J/K/L 列）-----
    # 三项默认 = CMA-ES v3 最优值，加权后总分贡献最大
    yw_wi_1: float = 36.30  # Short_GCross_Normal（init=10.00 → 3.63x，J 列）
    yw_wi_2: float = 49.85  # Short_GCross_Plus（init=10.00 → 4.98x，**顶到约束上限**，K 列）
    yw_wi_3: float = 35.25  # Short_GCross_Pro（init=10.00 → 3.53x，L 列）

    # ----- 活跃市值 AMV（2026-06 用户新增需求，对应日线 CSV 的 AMV 列）-----
    # AMV = +1 表示活跃市值处于多头区间；AMV = -1 表示空头区间；0/缺失表示中性
    # 公式（在 compute_score 中实现）：
    #     amv_val = +amvl_wi * (AMV == 1) - amvs_wi * (AMV == -1)
    # 即：多头时加分（+amvl_wi），空头时减分（-amvs_wi），中性/缺失时 0
    # 默认值 = 20（用户指定）；后续可用 RL 进一步优化
    amvl_wi: float = 20.0   # AMV 多头加分权重
    amvs_wi: float = 20.0   # AMV 空头减分权重

    def set_weights(self, vec: Iterable[float]) -> None:
        # 强化学习 agent 注入新权重的入口，顺序需与 to_array 一致
        keys = list(asdict(self).keys())
        vals = list(vec)
        if len(vals) != len(keys):
            raise ValueError(
                f"权重向量长度不匹配：期望 {len(keys)}，实际 {len(vals)}"
            )
        for k, v in zip(keys, vals):
            setattr(self, k, float(v))

    @classmethod
    def from_dict(cls, d: Dict[str, float]) -> "ZWeights":
        return cls(**{k: float(v) for k, v in d.items() if k in cls.__dataclass_fields__})


# ====================================================================
# 3. 策略基类
# ====================================================================
class StrategyBase(ABC):
    """
    所有策略的基类。后续可扩展出 "X 策略"、"Y 策略" 等，
    业务调用方只通过这个名字完成切换。
    """

    def __init__(self, weights: Optional[ZWeights] = None, **kwargs: Any) -> None:
        self.weights: ZWeights = weights or ZWeights()
        self.name: str = self.__class__.__name__
        self.params: Dict[str, Any] = kwargs

    # ---- 需要子类实现的接口 ----
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        对单只股票的 OHLCV+指标数据计算信号。
        返回 dict（键名约定见下文 ZStrategy.generate_signals）。
        """
        ...

    @abstractmethod
    def compute_score(self, row: pd.Series) -> float:
        """
        对单只股票某一行（已展开成 dict 的特征行）计算综合得分。
        """
        ...

    # ---- 通用辅助 ----
    def set_weights(self, weights: ZWeights) -> None:
        """强化学习 / 网格搜索更新权重时调用"""
        self.weights = weights


# ====================================================================
# 4. Z 策略实现
# ====================================================================
class ZStrategy(StrategyBase):
    """
    Z 策略：
        * 买入信号：J 负值 / 补票 P1P2 / 突破 L1 / 触底 L2 / 突破确认 / red_brick
        * 卖出信号：持仓超过 20 日 / J 由 >100 跌到 <100 / 量价异动 / 假突破
        * 综合评分：按 ZWeights 中的权重加权求和
    """

    # ---- 买入信号 ----
    def generate_signals(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        计算单只股票的买入/特征信号。
        入参 data 已经按时间升序排列，至少 20 行。
        """
        # 默认返回值（数据不足时回退）
        empty: Dict[str, Any] = {
            'j_negative': 0,
            'j_value': 0.0,
            'j_reversal': 0,
            'p1_signal': 0,
            'p2_signal': 0,
            'bbi_trend_5d': 0.0,
            'bbi_trend_20d': 0.0,
            'bbi_online': 0,
            'break_L1': 0,
            'touch_L2': 0,
            'breakthrough_confirm': 0,
            'short_term_down_no_break': 0,
            'red_brick': 0,
            'in_R1': 0, 'in_R2': 0, 'in_R3': 0, 'in_R4': 0,
            'is_new_high': 0,
            'short_gcross_normal': 0,
            'short_gcross_plus': 0,
            'short_gcross_pro': 0,
            'long_term_fund': 0.0,
        }
        if data is None or len(data) < 20:
            return empty

        # -------- 基础值 --------
        latest_j = float(data['J'].iloc[-1])
        prev_j = float(data['J'].iloc[-2])

        latest_short_fund = float(data['short_term_fund'].iloc[-1])
        latest_long_fund = float(data['long_term_fund'].iloc[-1])
        prev_short_fund = float(data['short_term_fund'].iloc[-2])
        prev_long_fund = float(data['long_term_fund'].iloc[-2])

        # BBI 趋势
        bbi_dif = data['BBI_DIF'].iloc[-20:]
        bbi_trend_5d = float((bbi_dif.iloc[-5:] > 0).mean())
        bbi_trend_20d = float((bbi_dif > 0).mean())

        # 简单信号
        j_negative = 1 if latest_j < 20 else 0
        j_reversal = 1 if latest_j > prev_j else 0
        p1_signal = 1 if (latest_short_fund < 20 and latest_long_fund > 80) else 0
        p2_signal = 1 if (
            latest_short_fund > 95 and latest_long_fund > 95
            and prev_short_fund < 20 and prev_long_fund > 80
        ) else 0

        # -------- BBI 线上 --------
        try:
            latest_close = float(data['close'].iloc[-1])
            latest_open = float(data['open'].iloc[-1])
            latest_BBI = float(data['BBI'].iloc[-1])
            bbi_online = 1 if (latest_close > latest_BBI and latest_open > latest_BBI) else 0
        except Exception:
            bbi_online = 0
            latest_close = float(data['close'].iloc[-1]) if 'close' in data.columns else 0.0
            latest_open = float(data['open'].iloc[-1]) if 'open' in data.columns else 0.0

        # -------- L1 / L2 突破 / 触底 --------
        try:
            prev_close = float(data['close'].iloc[-2])
            prev_L1 = float(data['L1'].iloc[-2])
            latest_L1 = float(data['L1'].iloc[-1])
            prev_L2 = float(data['L2'].iloc[-2])
            latest_L2 = float(data['L2'].iloc[-1])
            break_L1 = 1 if (prev_close > prev_L1 and latest_close < latest_L1) else 0
            touch_L2 = 1 if (prev_close > prev_L2 * 1.05 and latest_close < latest_L2 * 1.05) else 0
        except Exception:
            break_L1 = 0
            touch_L2 = 0

        # -------- 突破确认 / 短期下跌未破位 --------
        try:
            prev_open = float(data['open'].iloc[-2])
            latest_open = float(data['open'].iloc[-1])
            prev_vol = float(data['volume'].iloc[-2])
            latest_vol = float(data['volume'].iloc[-1])

            prev_21_1_close = data['close'].iloc[-22:-1]
            cond1 = (
                len(prev_21_1_close) > 0
                and prev_close == float(prev_21_1_close.max())
                and (prev_close - prev_open) / prev_open >= 0.05
            )
            if latest_close > latest_open:
                cond2 = True
            else:
                cond2 = not (0.5 * prev_vol <= latest_vol <= 0.9 * prev_vol)
            breakthrough_confirm = 1 if (cond1 and cond2) else 0

            # 短期下跌未破位
            short_term_down_no_break = 0
            if len(data) >= 4:
                day4 = data.iloc[-4]
                cond_1_1 = (
                    day4['close'] > day4['open']
                    and (day4['close'] - day4['open']) / day4['open'] > 0.05
                )
                last3 = data.iloc[-3:]
                cond_1_2 = bool((last3['close'] < last3['open']).all())
                cond_1_3 = data['close'].iloc[-1] > day4['open']
                if cond_1_1 and cond_1_2 and cond_1_3:
                    short_term_down_no_break = 1
        except Exception:
            breakthrough_confirm = 0
            short_term_down_no_break = 0

        # -------- red_brick 红砖 --------
        red_brick = 0
        try:
            prev_brick_high = float(data['Brick_High'].iloc[-2])
            prev_brick_low = float(data['Brick_Low'].iloc[-2])
            latest_brick_high = float(data['Brick_High'].iloc[-1])
            latest_brick_low = float(data['Brick_Low'].iloc[-1])

            cond_rb_1 = prev_brick_high < prev_brick_low
            cond_rb_2 = latest_brick_high > latest_brick_low
            prev_diff = abs(prev_brick_high - prev_brick_low)
            latest_diff = abs(latest_brick_high - latest_brick_low)
            cond_rb_3 = latest_diff > 0.7 * prev_diff
            if cond_rb_1 and cond_rb_2 and cond_rb_3:
                red_brick = 1
        except Exception:
            red_brick = 0

        # -------- 短线金叉 --------
        try:
            latest_short_trend = float(data['Short_Trend'].iloc[-1])
            latest_short_ls = float(data['Short_LS'].iloc[-1])
            short_gcross_normal = 1 if latest_short_trend > latest_short_ls else 0
            within_trend_range = (
                latest_short_trend != 0
                and abs(latest_close - latest_short_trend) / abs(latest_short_trend) <= 0.02
            )
            within_ls_range = (
                latest_short_ls != 0
                and abs(latest_close - latest_short_ls) / abs(latest_short_ls) <= 0.02
            )
            short_gcross_plus = 1 if (short_gcross_normal == 1 and (within_trend_range or within_ls_range)) else 0

            # short_gcross_pro
            try:
                latest_high = float(data['high'].iloc[-1])
                latest_low = float(data['low'].iloc[-1])
                amplitude_ratio = (latest_high - latest_low) / latest_open if latest_open else 0
                change_ratio = (latest_close - latest_open) / latest_open if latest_open else 0
                condition1 = amplitude_ratio <= 0.07
                condition2 = -0.018 <= change_ratio <= 0.02
                condition3 = short_gcross_plus == 1
                if len(data) >= 10:
                    past_10 = data['volume'].iloc[-10:]
                    condition4 = latest_vol == float(past_10.min())
                else:
                    condition4 = False
                short_gcross_pro = 1 if (condition1 and condition2 and condition3 and condition4) else 0
            except Exception:
                short_gcross_pro = 0
        except Exception:
            short_gcross_normal = 0
            short_gcross_plus = 0
            short_gcross_pro = 0

        # -------- 区间 R1~R4 --------
        try:
            latest_L1 = float(data['L1'].iloc[-1])
            latest_L2 = float(data['L2'].iloc[-1])
            latest_M = float(data['M'].iloc[-1])
            latest_H1 = float(data['H1'].iloc[-1])
            latest_H2 = float(data['H2'].iloc[-1])
            in_R1 = 1 if (latest_L1 > latest_close >= latest_L2) else 0
            in_R2 = 1 if (latest_M > latest_close >= latest_L1) else 0
            in_R3 = 1 if (latest_H1 > latest_close >= latest_M) else 0
            in_R4 = 1 if (latest_H2 > latest_close >= latest_H1) else 0
        except Exception:
            in_R1 = in_R2 = in_R3 = in_R4 = 0

        # -------- 创新高 --------
        try:
            latest_highs = data['high'].iloc[-3:]
            max_high_40 = data['high'].iloc[-40:].max()
            is_new_high = 1 if (latest_highs == max_high_40).any() else 0
        except Exception:
            is_new_high = 0

        return {
            'j_negative': j_negative,
            'j_value': round(latest_j, 2),
            'j_reversal': j_reversal,
            'p1_signal': p1_signal,
            'p2_signal': p2_signal,
            'bbi_trend_5d': round(bbi_trend_5d, 2),
            'bbi_trend_20d': round(bbi_trend_20d, 2),
            'bbi_online': bbi_online,
            'break_L1': break_L1,
            'touch_L2': touch_L2,
            'breakthrough_confirm': breakthrough_confirm,
            'short_term_down_no_break': short_term_down_no_break,
            'red_brick': red_brick,
            'in_R1': in_R1, 'in_R2': in_R2, 'in_R3': in_R3, 'in_R4': in_R4,
            'is_new_high': is_new_high,
            'short_gcross_normal': short_gcross_normal,
            'short_gcross_plus': short_gcross_plus,
            'short_gcross_pro': short_gcross_pro,
            'long_term_fund': round(latest_long_fund, 2),
        }

    # ---- 评分 ----
    def compute_score(self, row: pd.Series) -> float:
        """
        对单行特征（dict-like）计算 Z 策略综合得分。
        字段命名与 SignalScan 报告保持一致，便于调用方直接使用。
        """
        w = self.weights
        j_val = (
            row['J到负值-日线'] * w.j_wi_1
            + row['J到负值-日线'] * min(-1 * row['J值-日线'], w.j_wi_2)
            + row['J到负值-日线'] * row['J值反转-日线'] * w.j_wi_3
        )
        bp_val = (
            row['补票-P1'] * w.bp_wi_1
            + row['补票-P2'] * w.bp_wi_2
            + row['长线资金指标'] * w.bp_wi_2 / 100.0
        )
        bbi_val = (
            row['BBI线上'] * w.bbi_wi_1
            + row['BBI上涨趋势-5日'] * w.bbi_wi_2
            + row['BBI上涨趋势-20日'] * w.bbi_wi_3
        )
        peb_val = (
            row['股价跌穿L1线'] * w.peb_wi_1
            + row['股价触碰L2底线'] * w.peb_wi_2
            + row['股价位于R1区间'] * w.peb_wi_3
            + row['股价位于R2区间'] * w.peb_wi_4
            + row['股价位于R3区间'] * w.peb_wi_5
            + row['股价位于R4区间'] * w.peb_wi_6
        )
        bt_val = (
            row['股价创新高'] * w.bt_wi_1
            + row['突破确认'] * w.bt_wi_2
        )
        # 注意：原"优选联盟 pa_wi_1"已删除（CMA-ES v3 训练结果认为应关闭）
        # 2026-06 新增：短线金叉三因子加权（与 SignalScan 报告 J/K/L 列对应）
        yw_val = (
            row['Short_GCross_Normal'] * w.yw_wi_1
            + row['Short_GCross_Plus'] * w.yw_wi_2
            + row['Short_GCross_Pro'] * w.yw_wi_3
        )
        # 2026-06 新增：活跃市值 AMV 多空信号
        #   AMV == +1  → amv_val = +amvl_wi（多头加分）
        #   AMV == -1  → amv_val = -amvs_wi（空头减分）
        #   其他/缺失  → amv_val = 0（中性，无贡献）
        # row.get('AMV', 0) 防御性写法：日线 CSV 当前没 AMV 列时自动当 0
        amv = row.get('AMV', 0)
        amv_val = w.amvl_wi * int(amv == 1) - w.amvs_wi * int(amv == -1)
        return float(j_val + bp_val + bbi_val + peb_val + bt_val + yw_val + amv_val)

    # ---- 卖出信号（仅在回测中使用） ----
    def generate_sell_signals(
        self,
        buy_signal: pd.Series,
        data: pd.DataFrame,
        max_hold: int = 20,
    ) -> pd.Series:
        """
        给定一只股票的买入信号序列与原始数据，生成对应的卖出信号序列。
        卖出规则：
            rule1 持仓超过 max_hold 日；
            rule2 J 由 >100 跌到 <100（短期超买回吐）；
            rule3 当日为阴线 + 当日量为近 20 日最大 + 较前一日放量 1.15 倍
                  + 上影线长 + 下影线长（典型顶部形态）；
            rule4 前一日为 21 日新高 + 当日阴线 + 成交量在 0.5~0.9 倍前一日（缩量假突破）。
        """
        sell = pd.Series(0, index=buy_signal.index, dtype=int)
        if data is None or data.empty:
            return sell

        hold_count = 0
        prev_date = None
        # 取 20 日量、21 日收盘用于规则 3/4
        vol_20_max = data['volume'].rolling(20, min_periods=1).max()
        close_21_max = data['close'].rolling(21, min_periods=1).max()

        for i, date in enumerate(buy_signal.index):
            buy_today = int(buy_signal.iloc[i]) == 1
            if buy_today:
                hold_count = 1
            elif hold_count > 0:
                hold_count += 1
            else:
                hold_count = 0

            rule1 = hold_count > max_hold

            rule2 = False
            if prev_date is not None and prev_date in data.index and date in data.index:
                prev_j = float(data.loc[prev_date, 'J'])
                curr_j = float(data.loc[date, 'J'])
                rule2 = (prev_j > 100) and (curr_j < 100)

            rule3 = False
            if date in data.index:
                row = data.loc[date]
                if {'close', 'open', 'high', 'low', 'volume'}.issubset(data.columns):
                    curr_close = float(row['close']); curr_open = float(row['open'])
                    curr_high = float(row['high']); curr_low = float(row['low'])
                    curr_vol = float(row['volume'])
                    if prev_date is not None and prev_date in data.index:
                        prev_vol = float(data.loc[prev_date, 'volume'])
                        max_vol_20 = float(vol_20_max.loc[date])
                        rule3 = (
                            (curr_close < curr_open)
                            and (curr_vol == max_vol_20)
                            and (curr_vol > prev_vol * 1.15)
                            and (curr_high > curr_open * 1.02)
                            and (curr_low < curr_close * 0.98)
                        )

            rule4 = False
            if prev_date is not None and date in data.index and prev_date in data.index:
                prev_close = float(data.loc[prev_date, 'close'])
                prev_open = float(data.loc[prev_date, 'open'])
                prev_vol = float(data.loc[prev_date, 'volume'])
                curr_close = float(data.loc[date, 'close'])
                curr_open = float(data.loc[date, 'open'])
                curr_vol = float(data.loc[date, 'volume'])
                max_close_21 = float(close_21_max.loc[prev_date])
                rule4 = (
                    (prev_close == max_close_21)
                    and (curr_close < curr_open)
                    and (curr_vol > prev_vol * 0.5)
                    and (curr_vol < prev_vol * 0.9)
                )

            if rule1 or rule2 or rule3 or rule4:
                sell.iloc[i] = 1
                hold_count = 0

            prev_date = date
        return sell


# ====================================================================
# 5. 数据加载工具
# ====================================================================
class ZDataLoader:
    """
    负责把 01_RawData-Daily 中的 CSV 读成 DataFrame。
    供 SignalScan 与 回测 共用。
    """

    def __init__(self, data_dir: str = DEFAULT_RAW_DATA_DIR) -> None:
        self.data_dir = data_dir

    def load(self, symbol: str, n_bars: int = 100) -> Optional[pd.DataFrame]:
        """加载单只股票的最近 n_bars 根 K 线"""
        path = os.path.join(self.data_dir, f"{symbol}.csv")
        if not os.path.exists(path):
            return None
        try:
            data = pd.read_csv(
                path,
                parse_dates=['date'],
                index_col='date',
                date_format='%Y-%m-%d'
            )
            return data.ffill().bfill().tail(n_bars)
        except Exception:
            return None

    def load_full(self, symbol: str) -> Optional[pd.DataFrame]:
        """加载单只股票全部历史 K 线（用于回测）"""
        path = os.path.join(self.data_dir, f"{symbol}.csv")
        if not os.path.exists(path):
            return None
        try:
            data = pd.read_csv(
                path,
                parse_dates=['date'],
                index_col='date',
                date_format='%Y-%m-%d'
            )
            return data.ffill().bfill()
        except Exception:
            return None


# ====================================================================
# 6. 工厂方法：业务调用方用这个函数得到一个策略实例
# ====================================================================
_STRATEGY_REGISTRY: Dict[str, type] = {
    'Z': ZStrategy,
}


def get_strategy(name: str = 'Z', **kwargs: Any) -> StrategyBase:
    """
    业务调用方入口：将来新增其他策略时，只需要在这里注册即可。
    """
    cls = _STRATEGY_REGISTRY.get(name)
    if cls is None:
        raise ValueError(
            f"未知策略: {name}，目前已注册: {list(_STRATEGY_REGISTRY.keys())}"
        )
    return cls(**kwargs)


def register_strategy(name: str, cls: type) -> None:
    """注册新策略的辅助方法，供未来 X 策略 / Y 策略使用"""
    if not issubclass(cls, StrategyBase):
        raise TypeError("策略类必须继承自 StrategyBase")
    _STRATEGY_REGISTRY[name] = cls


# ====================================================================
# 7. 自测入口
# ====================================================================
if __name__ == "__main__":
    print("Z-Strategy 自测：")
    loader = ZDataLoader()
    df = loader.load("000001", n_bars=100)
    if df is not None:
        strategy = get_strategy("Z")
        signals = strategy.generate_signals(df)
        print("买入信号：", {k: v for k, v in signals.items() if k.startswith(('j_', 'p', 'break', 'touch', 'red', 'in_', 'is_', 'short_'))})

        # 构造一行特征数据，模拟 compute_score
        feature_row = {
            'J到负值-日线': signals['j_negative'],
            'J值-日线': signals['j_value'],
            'J值反转-日线': signals['j_reversal'],
            '补票-P1': signals['p1_signal'],
            '补票-P2': signals['p2_signal'],
            '长线资金指标': signals['long_term_fund'],
            'BBI线上': signals['bbi_online'],
            'BBI上涨趋势-5日': signals['bbi_trend_5d'],
            'BBI上涨趋势-20日': signals['bbi_trend_20d'],
            '股价跌穿L1线': signals['break_L1'],
            '股价触碰L2底线': signals['touch_L2'],
            '股价位于R1区间': signals['in_R1'],
            '股价位于R2区间': signals['in_R2'],
            '股价位于R3区间': signals['in_R3'],
            '股价位于R4区间': signals['in_R4'],
            '股价创新高': signals['is_new_high'],
            '突破确认': signals['breakthrough_confirm'],
            '优选联盟成员': 0,
        }
        row = pd.Series(feature_row)
        print(f"综合得分: {strategy.compute_score(row):.2f}")
    else:
        print("未找到 000001.csv，跳过自测")
