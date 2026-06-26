# -*- coding: utf-8 -*-
"""
RL_Alpha_AShares.py
====================
基于 CMA-ES 的强化学习 / 黑盒优化框架，用于优化 Z 策略的 19 个权重因子。

架构（自顶向下）：

    ┌─────────────────────────────────────────┐
    │  main()                                  │  ← 入口（CLI / 编程式）
    └──────────┬──────────────────────────────┘
               │
    ┌──────────▼──────────────────────────────┐
    │  CMATrainer.train()                      │  ← CMA-ES 主循环
    │  - 多进程并行评估                        │
    │  - 权重约束 (init × [0.5, 2.0])          │
    │  - checkpoint + best 记录                │
    │  - 日志 (train/val 曲线)                 │
    └──────────┬──────────────────────────────┘
               │
    ┌──────────▼──────────────────────────────┐
    │  WeightEnv.evaluate(weights_vec)        │  ← 包装 AShares_BackTest
    │  - 训练: 中证A500 抽 100 只              │
    │  - 验证: 中证A500 全 500 只               │
    │  - reward: 终值 End Value                  │
    └─────────────────────────────────────────┘

调用关系：
    RL_Alpha_AShares.py
        ↓ import
    AShares_BackTest_20260615.py (BacktestConfig, BacktestEngine)
        ↓ importlib
    Z-Strategy.py (ZWeights)

训练 / 验证 / 测试时间窗（用户指定）：
    train: 2025/06/25 ~ 2025/09/03
    val:   2025/12/09 ~ 2026/01/28
    test:  2026/04/08 ~ 2026/05/25

股票池：05_BackTest/01_Ashares/01_List/中证A500.xlsx
"""

from __future__ import annotations

import os
import sys
import json
import time
import random
import logging
import argparse
import multiprocessing as mp
import warnings
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

# 抑制 vbt 的一些警告
warnings.filterwarnings("ignore")

# ============================================================================
# 路径常量
# ============================================================================
PROJECT_ROOT = r"D:\Quant\01_SwProj\04_VectorBT\02_Lima\Lima_Gen1"
BACKTEST_DIR = os.path.join(PROJECT_ROOT, "05_BackTest", "01_Ashares", "20260615")
Z_STRATEGY_DIR = os.path.join(PROJECT_ROOT, "04_Strategy", "01_Ashares")
STOCKPOOL_XLSX = os.path.join(
    PROJECT_ROOT, "05_BackTest", "01_Ashares", "01_List", "中证A500.xlsx"
)

# 输出目录
OUTPUT_DIR = os.path.join(
    PROJECT_ROOT, "06_RL", "01_Ashares", "01_Alpha", "outputs"
)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 把 backtest 目录加进 sys.path，让 import 能找到 AShares_BackTest_20260615
if BACKTEST_DIR not in sys.path:
    sys.path.insert(0, BACKTEST_DIR)

# 日志（同时写文件 + 输出到终端）
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(OUTPUT_DIR, "train_v5.log"), encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("RL_Alpha")


# ============================================================================
# 1. 训练配置
# ============================================================================
@dataclass
class TrainConfig:
    """训练配置。所有可调参数集中在这里，方便 RL 调优。"""

    # 时间窗（用户指定）
    # v5 训练窗口（用户 2026-06 指定）
    train_start: str = "2025-03-01"
    train_end:   str = "2025-06-30"
    val_start:   str = "2025-07-01"
    val_end:     str = "2025-10-30"
    test_start:  str = "2026-02-01"
    test_end:    str = "2026-06-30"

    # 股票池与抽样
    stockpool_xlsx: str = STOCKPOOL_XLSX
    train_n_stocks: int = 100    # 训练时从中证A500 抽 100 只
    val_n_stocks:   int = 500    # 验证用全部 500 只
    train_seed: int = 42         # 抽样随机种子（保证可复现）

    # CMA-ES 超参
    cma_sigma:    float = 1.0     # 初始搜索步长（归一化空间 [0,1] 上的标准差；v5 搜索空间放宽后调到 1.0）
    cma_popsize:  int   = 16      # 每代个体数
    cma_max_gen:  int   = 30      # 最大迭代代数
    cma_seed:     int   = 0

    # 权重约束：init_value * [0.01, 10.0]  ← v5 放宽（用户 2026-06）
    weight_bound_low:  float = 0.01
    weight_bound_high: float = 10.0

    # 资源
    n_workers: int = 8            # 多进程并行评估 worker 数

    # v5 产物路径（加 _v5 后缀，保留 v3 best_weights.json 作为对比基线）
    output_dir: str = OUTPUT_DIR
    checkpoint_path:    str = os.path.join(OUTPUT_DIR, "cma_checkpoint_v5.npz")
    best_weights_path:  str = os.path.join(OUTPUT_DIR, "best_weights_v5.json")
    history_path:       str = os.path.join(OUTPUT_DIR, "history_v5.json")


# ============================================================================
# 2. 工具函数
# ============================================================================
def load_stockpool_codes(xlsx_path: str) -> List[str]:
    """读股票池首列（代码），补零到 6 位。"""
    df = pd.read_excel(xlsx_path)
    if df.empty:
        return []
    code_col = df.columns[0]
    codes = df[code_col].astype(str).str.zfill(6).unique().tolist()
    return codes


def sample_train_codes(all_codes: List[str], n: int, seed: int) -> List[str]:
    """从中证 A500 抽 n 只股票作为训练子集。固定 seed 保证可复现。"""
    if n >= len(all_codes):
        return list(all_codes)
    rng = random.Random(seed)
    return sorted(rng.sample(all_codes, n))


# ============================================================================
# 3. 单次回测评估（worker 函数）
# ============================================================================
def evaluate_single_backtest(
    weights_vec: np.ndarray,
    cfg_dict: Dict[str, Any],
    period_key: str,        # "train" / "val" / "test"
    init_cash: float,
    max_positions: int = 5,
) -> float:
    """
    用一组权重跑一次完整回测，返回 End Value（最终资产净值）。

    为什么不用 BacktestEnv.step()：
        - BacktestEnv 每次都新建 BacktestEngine + 重跑 vectorbt，速度慢
        - 我们这里要的是"快速、单次、只关心最终收益"的评估
        - 干脆直接调 BacktestEngine，效率更高
    """
    from AShares_BackTest_20260615 import BacktestConfig, BacktestEngine
    import vectorbt as vbt

    period = cfg_dict[period_key]
    codes  = cfg_dict[f"{period_key}_codes"]

    # 构造回测配置（worker 不需要写正式报告，省掉 Excel/PDF 开销）
    bt_cfg = BacktestConfig(
        start_date=period["start"],
        end_date=period["end"],
        raw_data_dir=os.path.join(
            PROJECT_ROOT, "01_Database", "01_Ashares", "01_RawData-Daily"
        ),
        stockpool_xlsx=cfg_dict["stockpool_xlsx"],
        report_dir=os.path.join(OUTPUT_DIR, "_worker_reports"),
        init_cash=init_cash,
        max_positions=max_positions,
    )

    engine = BacktestEngine(bt_cfg)
    # 19 维 numpy 向量 → ZWeights dataclass
    engine.strategy.weights.set_weights(weights_vec)

    try:
        engine.close_df = engine.load_market_data(codes)
        engine.compute_signals(codes)
        # 注入权重后必须重算评分（compute_signals 用的是默认权重）
        engine._compute_scores_with_current_weights()
        entries, exits = engine.build_target_signals()
        size_pct = engine._build_size_matrix(entries, exits)

        # 把 entries / exits / size 矩阵对齐到 close 的 index + columns
        common_idx = engine.signal_df.index.intersection(engine.close_df.index)
        common_cols = engine.signal_df.columns.intersection(engine.close_df.columns)
        entries = entries.loc[common_idx, common_cols]
        exits = exits.loc[common_idx, common_cols]
        close = engine.close_df.loc[common_idx, common_cols]
        size_pct = size_pct.loc[common_idx, common_cols]

        engine.portfolio = vbt.Portfolio.from_signals(
            close=close,
            entries=entries.astype(bool),
            exits=exits.astype(bool),
            size=size_pct,
            size_type=vbt.portfolio.enums.SizeType.Percent,
            init_cash=bt_cfg.init_cash,
            cash_sharing=True,
            freq="D",
            call_seq=vbt.portfolio.enums.CallSeqType.Default,
            fees=bt_cfg.fees,
            slippage=bt_cfg.slippage,
            size_granularity=bt_cfg.size_granularity,
            sl_stop=bt_cfg.sl_stop,
            tp_stop=bt_cfg.tp_stop,
            sl_trail=bt_cfg.sl_trail,
        )
        end_value = float(engine.portfolio.value().iloc[-1])
        return end_value
    except Exception as e:
        log.warning(f"[{period_key}] evaluate failed: {e}")
        return 0.0


def _evaluate_with_global(args):
    """多进程 worker 的入口。"""
    weights_vec, cfg_dict, period_key, init_cash, max_pos = args
    return evaluate_single_backtest(weights_vec, cfg_dict, period_key, init_cash, max_pos)


# ============================================================================
# 4. WeightEnv：CMA-ES 用的环境
# ============================================================================
class WeightEnv:
    """
    连续 19 维权重优化环境。

    状态 = 上一代评估得到的 reward
    动作 = 19 维连续权重（在 [bound_low, bound_high] * init_value 范围内）
    奖励 = 训练集 End Value（用户要求）

    为什么不用标准 RL 的 per-step episode？
        - 这个场景的 reward 只在 episode 末尾出一次（final End Value）
        - 用 PPO 之类的 per-step RL 反而难收敛
        - CMA-ES（进化策略）天然适合"一次性评估"场景
    """

    # ZWeights 字段顺序，必须与 ZStrategy.ZWeights 定义一致
    WEIGHT_FIELDS = [
        "j_wi_1", "j_wi_2", "j_wi_3",
        "bp_wi_1", "bp_wi_2", "bp_wi_3",
        "bbi_wi_1", "bbi_wi_2", "bbi_wi_3",
        "peb_wi_1", "peb_wi_2", "peb_wi_3", "peb_wi_4", "peb_wi_5", "peb_wi_6",
        "bt_wi_1", "bt_wi_2",
        # 2026-06 删除 pa_wi_1（优选联盟清单对中证 A500 几乎都是 0，CMA-ES v3 训练结果认为应关闭）
        # 2026-06 新增：短线金叉三因子，对应 SignalScan 报告的 J/K/L 列
        "yw_wi_1", "yw_wi_2", "yw_wi_3",
        # 2026-06 新增：活跃市值 AMV 多空信号权重
        # 公式：amv_val = +amvl_wi * (AMV==1) - amvs_wi * (AMV==-1)
        "amvl_wi", "amvs_wi",
    ]
    N_WEIGHTS = len(WEIGHT_FIELDS)   # 22

    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        self._init_weights = self._load_default_weights()
        self.lows  = self._init_weights * cfg.weight_bound_low
        self.highs = self._init_weights * cfg.weight_bound_high
        self._cfg_dict = self._build_cfg_dict()

    def _load_default_weights(self) -> np.ndarray:
        """加载 ZStrategy.ZWeights 默认值"""
        from AShares_BackTest_20260615 import ZWeights
        w = ZWeights()
        return np.array([getattr(w, f) for f in self.WEIGHT_FIELDS], dtype=np.float64)

    def _build_cfg_dict(self) -> Dict[str, Any]:
        """构造多进程可序列化的 cfg 字典"""
        all_codes = load_stockpool_codes(self.cfg.stockpool_xlsx)
        train_codes = sample_train_codes(
            all_codes, self.cfg.train_n_stocks, self.cfg.train_seed
        )
        val_codes = all_codes[:self.cfg.val_n_stocks]
        test_codes = all_codes  # 测试集用全部
        return {
            "stockpool_xlsx": self.cfg.stockpool_xlsx,
            "train": {"start": self.cfg.train_start, "end": self.cfg.train_end},
            "val":   {"start": self.cfg.val_start,   "end": self.cfg.val_end},
            "test":  {"start": self.cfg.test_start,  "end": self.cfg.test_end},
            "train_codes": train_codes,
            "val_codes":   val_codes,
            "test_codes":  test_codes,
        }

    def normalized_to_weights(self, x: np.ndarray) -> np.ndarray:
        """把 CMA-ES 工作在的 [0, 1]^N 空间映射回真实权重"""
        return self.lows + (self.highs - self.lows) * x

    def weights_to_normalized(self, w: np.ndarray) -> np.ndarray:
        """真实权重 → 归一化空间"""
        return (w - self.lows) / (self.highs - self.lows + 1e-12)

    def evaluate(
        self,
        weights_vec: np.ndarray,
        period: str = "train",
        n_workers: int = 1,
    ) -> float:
        """评估一组权重，返回 End Value。"""
        from AShares_BackTest_20260615 import BacktestConfig
        return evaluate_single_backtest(
            weights_vec, self._cfg_dict, period,
            init_cash=BacktestConfig().init_cash,
            max_positions=BacktestConfig().max_positions,
        )


# ============================================================================
# 5. CMA-ES 训练器
# ============================================================================
class CMATrainer:
    """
    用 CMA-ES 优化 ZWeights。

    工作流程：
        1. 在归一化空间 [0,1]^19 上初始化种群
        2. 每代：
           a. CMA-ES 采样 popsize 个个体
           b. 映射回真实权重空间
           c. 并行跑 N 个回测（多进程）
           d. 用 End Value 作为 fitness，更新 CMA-ES 状态
           e. 记录 best_so_far，在验证集上评估（防过拟合）
        3. 训练结束：保存最优权重、训练历史
    """

    def __init__(self, env: WeightEnv, cfg: TrainConfig):
        self.env = env
        self.cfg = cfg
        self.history: List[Dict[str, Any]] = []
        self.best_train_end_value: float = -np.inf
        self.best_val_end_value:   float = -np.inf
        self.best_weights:         Optional[np.ndarray] = None

    def _init_cma(self):
        """初始化 CMA-ES。在归一化空间 [0,1]^N 上工作。"""
        import cma
        x0 = np.full(self.env.N_WEIGHTS, 0.5)  # 初始点取中位（对应真实权重的初始值）
        sigma0 = self.cfg.cma_sigma
        es = cma.CMAEvolutionStrategy(
            x0.tolist(),
            sigma0,
            {
                "popsize": self.cfg.cma_popsize,
                "seed":    self.cfg.cma_seed,
                "bounds":  [0.0, 1.0],   # 强制归一化空间
                "verbose": -9,           # 关闭 cma 自己的输出
            },
        )
        return es

    def _evaluate_population(
        self,
        population_norm: np.ndarray,
        period: str,
    ) -> np.ndarray:
        """
        并行评估种群。返回每个个体的 End Value。

        population_norm: (popsize, N) 矩阵
        返回: (popsize,) End Value 数组
        """
        # 映射到真实权重
        population_real = np.array(
            [self.env.normalized_to_weights(x) for x in population_norm]
        )

        from AShares_BackTest_20260615 import BacktestConfig
        init_cash = BacktestConfig().init_cash
        max_pos   = BacktestConfig().max_positions

        args_list = [
            (w, self.env._cfg_dict, period, init_cash, max_pos)
            for w in population_real
        ]

        results: List[float] = []
        n_workers = min(self.cfg.n_workers, len(args_list))
        if n_workers <= 1:
            for a in args_list:
                results.append(_evaluate_with_global(a))
        else:
            ctx = mp.get_context("spawn")
            with ctx.Pool(processes=n_workers) as pool:
                for r in pool.imap_unordered(_evaluate_with_global, args_list):
                    results.append(r)
        return np.array(results, dtype=np.float64)

    def train(self) -> Dict[str, Any]:
        """主训练循环。"""
        import cma
        log.info("=" * 70)
        log.info("CMA-ES 训练开始")
        log.info("=" * 70)
        log.info(f"训练区间: {self.cfg.train_start} ~ {self.cfg.train_end}")
        log.info(f"验证区间: {self.cfg.val_start} ~ {self.cfg.val_end}")
        log.info(f"测试区间: {self.cfg.test_start} ~ {self.cfg.test_end}")
        log.info(f"股票池: {self.cfg.stockpool_xlsx}")
        log.info(
            f"训练用 {self.cfg.train_n_stocks} 只抽样股票，"
            f"验证用全部 {self.cfg.val_n_stocks} 只"
        )
        log.info(
            f"种群大小: {self.cfg.cma_popsize},  "
            f"最大代数: {self.cfg.cma_max_gen}"
        )
        log.info(f"并行 worker: {self.cfg.n_workers}")
        log.info(
            f"权重维度: {self.env.N_WEIGHTS},  "
            f"范围: init × [{self.cfg.weight_bound_low}, {self.cfg.weight_bound_high}]"
        )
        log.info(
            f"初始权重 (前 5 个): {self.env._init_weights[:5].round(2).tolist()}"
        )
        log.info("=" * 70)

        es = self._init_cma()
        gen = 0
        t_start = time.time()

        while not es.stop() and gen < self.cfg.cma_max_gen:
            gen += 1
            t_gen = time.time()

            # 1) 采样种群（归一化空间）
            solutions = es.ask()
            population_norm = np.array(solutions)

            # 2) 训练集评估
            train_rewards = self._evaluate_population(population_norm, period="train")
            # CMA-ES 最小化 → 取负号
            es.tell(solutions, (-train_rewards).tolist())

            # 3) 找本代 best
            best_idx = int(np.argmax(train_rewards))
            best_x_norm = population_norm[best_idx]
            best_x_real = self.env.normalized_to_weights(best_x_norm)
            best_train = float(train_rewards[best_idx])

            # 4) 验证集评估（只对 best 跑 1 次）
            val_reward = self.env.evaluate(best_x_real, period="val", n_workers=1)

            # 5) 更新全局 best
            if val_reward > self.best_val_end_value:
                self.best_val_end_value = val_reward
                self.best_train_end_value = best_train
                self.best_weights = best_x_real
                log.info(
                    f"  ★ 新最优! train={best_train:.0f}  "
                    f"val={val_reward:.0f}  "
                    f"weights[:5]={best_x_real[:5].round(2).tolist()}"
                )

            # 6) 记录历史
            self.history.append({
                "gen": gen,
                "best_train": best_train,
                "best_val":   val_reward,
                "mean_train": float(np.mean(train_rewards)),
                "min_train":  float(np.min(train_rewards)),
                "max_train":  float(np.max(train_rewards)),
                "global_best_train": self.best_train_end_value,
                "global_best_val":   self.best_val_end_value,
                "elapsed_sec": time.time() - t_gen,
                "total_elapsed_sec": time.time() - t_start,
            })

            log.info(
                f"gen={gen:3d}/{self.cfg.cma_max_gen}  "
                f"train_best={best_train:>10.0f}  "
                f"val_best  ={val_reward:>10.0f}  "
                f"mean={float(np.mean(train_rewards)):>10.0f}  "
                f"elapsed={time.time()-t_gen:>6.1f}s  "
                f"total={(time.time()-t_start)/60:>5.1f}min"
            )

        self._save_results(gen, time.time() - t_start)
        return {
            "best_weights": self.best_weights,
            "best_train":   self.best_train_end_value,
            "best_val":     self.best_val_end_value,
            "history":      self.history,
        }

    def _save_results(self, n_gen: int, total_sec: float):
        """保存 checkpoint / best weights / history"""
        if self.best_weights is not None:
            best_dict = {
                field: float(w)
                for field, w in zip(self.env.WEIGHT_FIELDS, self.best_weights)
            }
            with open(self.cfg.best_weights_path, "w", encoding="utf-8") as f:
                json.dump({
                    "weights": best_dict,
                    "train_end_value": self.best_train_end_value,
                    "val_end_value":   self.best_val_end_value,
                    "n_gen": n_gen,
                    "total_sec": total_sec,
                    "config": {
                        "train_window":  [self.cfg.train_start, self.cfg.train_end],
                        "val_window":    [self.cfg.val_start, self.cfg.val_end],
                        "test_window":   [self.cfg.test_start, self.cfg.test_end],
                        "train_n_stocks": self.cfg.train_n_stocks,
                        "val_n_stocks":   self.cfg.val_n_stocks,
                    },
                }, f, ensure_ascii=False, indent=2)
            log.info(f"💾 Best weights saved → {self.cfg.best_weights_path}")

        with open(self.cfg.history_path, "w", encoding="utf-8") as f:
            json.dump(self.history, f, ensure_ascii=False, indent=2)
        log.info(f"💾 History saved → {self.cfg.history_path}")

        log.info("=" * 70)
        log.info("训练结束")
        log.info("=" * 70)
        log.info(f"总代数: {n_gen},  总耗时: {total_sec/60:.1f} min")
        log.info(f"训练集最优 End Value: {self.best_train_end_value:,.0f}")
        log.info(f"验证集最优 End Value: {self.best_val_end_value:,.0f}")
        if self.best_weights is not None:
            log.info("最优权重 (vs 初始):")
            for field, w_init, w_opt in zip(
                self.env.WEIGHT_FIELDS,
                self.env._init_weights,
                self.best_weights
            ):
                ratio = w_opt / w_init if w_init != 0 else float("inf")
                log.info(
                    f"  {field:>10s}  init={w_init:>7.2f}  "
                    f"opt={w_opt:>7.2f}  ratio={ratio:>5.2f}x"
                )


# ============================================================================
# 6. 测试集评估
# ============================================================================
def evaluate_test(best_weights: np.ndarray, cfg: TrainConfig) -> float:
    """用最优权重在测试集上跑最终评估。"""
    log.info("=" * 70)
    log.info("测试集最终评估")
    log.info("=" * 70)
    env = WeightEnv(cfg)
    end_value = env.evaluate(best_weights, period="test", n_workers=1)
    log.info(f"测试集 End Value: {end_value:,.0f}")
    return end_value


# ============================================================================
# 7. CLI 入口
# ============================================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Z 策略权重 CMA-ES 优化")
    parser.add_argument("--gens", type=int, default=30, help="CMA-ES 最大代数")
    parser.add_argument("--popsize", type=int, default=16, help="每代个体数")
    parser.add_argument("--sigma", type=float, default=0.5, help="初始搜索步长")
    parser.add_argument("--workers", type=int, default=8, help="并行 worker 数")
    parser.add_argument("--train_n", type=int, default=100, help="训练用股票数")
    parser.add_argument("--seed", type=int, default=42, help="抽样随机种子")
    parser.add_argument(
        "--smoke", action="store_true",
        help="Smoke test（1 代 4 个体 + 30 只股票，约 2 分钟）"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = TrainConfig(
        cma_max_gen    = 1 if args.smoke else args.gens,
        cma_popsize    = 4 if args.smoke else args.popsize,
        cma_sigma      = args.sigma,
        n_workers      = min(args.workers, 4) if args.smoke else args.workers,
        train_n_stocks = 30 if args.smoke else args.train_n,
        train_seed     = args.seed,
    )

    env = WeightEnv(cfg)
    trainer = CMATrainer(env, cfg)
    result = trainer.train()

    if result["best_weights"] is not None and not args.smoke:
        test_end_value = evaluate_test(result["best_weights"], cfg)
        # 把测试集结果追加到 best_weights.json
        bp = cfg.best_weights_path
        if os.path.exists(bp):
            with open(bp, "r", encoding="utf-8") as f:
                d = json.load(f)
            d["test_end_value"] = test_end_value
            with open(bp, "w", encoding="utf-8") as f:
                json.dump(d, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
