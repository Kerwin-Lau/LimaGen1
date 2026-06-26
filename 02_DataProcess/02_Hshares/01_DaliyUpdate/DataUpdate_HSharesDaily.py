"""
H 股（港股）日线数据更新脚本

功能概述：
    1) 从 HSharesList.xlsx 读取港股代码池
    2) 调用 AKShare 接口拉取每只股票的日线行情（前复权）与估值快照
    3) 计算与 A 股版一致的技术指标（KDJ / BBI / MACD / 资金 / 多空线 / 趋势 / 砖形图）
    4) 输出到 01_Database/02_Hshares/01_RawData-Daily/{5位码}.csv

设计要点（与 A 股版 DataUpdate_ASharesDaily.py 保持结构一致）：
    - 起止日期固定 HISTORY_START_DATE = "20200102"，latest_trade_date 由抽样探测得到
    - 增量更新：若 csv 已存在，按 csv A 列 (date) 的 max + 1 作为增量起点；否则全量拉取
    - 行情首选 stock_hk_hist（东方财富，服务端日期过滤，更快）；
      备用 stock_hk_daily（新浪，客户端日期过滤）
    - 估值快照接口 stock_hk_financial_indicator_em 返回 单行快照（无时序），
      所有日期共享同一份估值列（H 股 PE 通道因此无法做时序分位，已跳过）
    - 技术指标使用 通达信 SMA 自定义实现（与 A 股版完全一致）

作者：Lima Gen1
"""

import akshare as ak
import pandas as pd
import numpy as np
import os
import re
import talib
from datetime import datetime
from dateutil.relativedelta import relativedelta
from tqdm import tqdm
import concurrent.futures
from functools import partial
import time
import urllib3
import random
import threading
import psutil

# 禁用 SSL 警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


# ============================ 路径配置（可被外部覆盖）============================
# H 股行情数据的统一起始日期（全量拉取与对齐窗口的下界）
# - 与 A 股版保持一致：2020-01-02 是因为 AKShare 历史稳定性较好，且对绝大多数
#   港股能完整覆盖近 6+ 年的交易日。
# - 该常量同时作用于：
#     1) 全量拉取时 ak.stock_hk_hist(start_date=...)
#     2) 增量分支里 aligned_history_df 的下界过滤（避免历史窗口被截断）
# - 增量更新分支不会因为该常量而触发历史重拉：只要 csv 已存在，仍然按
#   csv A 列(date) 的 max + 1 作为增量起点；该常量只是防止合并时把窗口外
#   的历史数据丢掉。
HISTORY_START_DATE = "20200102"

# 默认股票清单：全量 H 股名单
DEFAULT_STOCK_LIST_PATH = (
    r"D:\Quant\01_SwProj\04_VectorBT\02_Lima\Lima_Gen1"
    r"\01_Database\02_Hshares\HSharesList.xlsx"
)
# 备用股票清单目录：从周线选股结果目录中取日期最新的 xlsx（保留接口，本期未启用）
WEEKLY_REPORT_DIR = (
    r"D:\Quant\01_SwProj\04_VectorBT\02_Lima\Lima_Gen1"
    r"\02_DataProcess\02_Hshares\02_WeeklyUpdate\01_Report"
)
# 默认数据保存目录
DEFAULT_SAVE_DIR = (
    r"D:\Quant\01_SwProj\04_VectorBT\02_Lima\Lima_Gen1"
    r"\01_Database\02_Hshares\01_RawData-Daily"
)


# ============================ 模块级工具函数 ============================
def resolve_latest_weekly_report(report_dir: str = WEEKLY_REPORT_DIR) -> str:
    """
    在周线选股报告目录中，按文件修改时间找出日期最新的 xlsx 文件路径。

    Args:
        report_dir: 周线报告所在目录

    Returns:
        最新 xlsx 文件的完整绝对路径

    Raises:
        FileNotFoundError: 当目录下没有 xlsx 文件时
    """
    if not os.path.isdir(report_dir):
        raise FileNotFoundError(f"周线报告目录不存在: {report_dir}")

    xlsx_files = [f for f in os.listdir(report_dir) if f.lower().endswith(".xlsx")]
    if not xlsx_files:
        raise FileNotFoundError(f"周线报告目录中没有 xlsx 文件: {report_dir}")

    # 按修改时间倒序，取最新
    xlsx_files.sort(key=lambda f: os.path.getmtime(os.path.join(report_dir, f)), reverse=True)
    return os.path.join(report_dir, xlsx_files[0])


def load_stock_codes_from_xlsx(
    xlsx_path: str,
    stock_list_type: str = "auto",
) -> tuple:
    """
    从给定的 xlsx 股票清单中读取股票代码与股票名称。

    H 股版列约定（HSharesList.xlsx）：
        A 列 = 5 位港股代码（如 00700）
        B 列 = 中文名称
        C 列 = 细分行业（可选，不使用）

    Args:
        xlsx_path: xlsx 路径
        stock_list_type: 预留参数（"auto"/"full"/"weekly"），逻辑一致

    Returns:
        (codes: list[str], names: dict[str, str])
        codes 已做 5 位补 0、去重、保持顺序
    """
    if not os.path.exists(xlsx_path):
        raise FileNotFoundError(f"股票清单文件不存在: {xlsx_path}")

    df = pd.read_excel(xlsx_path, sheet_name=0, dtype=str)
    if df.empty or df.shape[1] < 1:
        raise ValueError(f"股票清单文件格式异常: {xlsx_path}")

    # 第一列：股票代码；第二列（若存在）：股票名称
    code_col = df.columns[0]
    name_col = df.columns[1] if df.shape[1] > 1 else None

    codes_raw = df[code_col].dropna().astype(str).str.strip().tolist()
    # 5 位补 0，过滤空字符串（只保留数字字符）
    codes = []
    for c in codes_raw:
        c_clean = "".join(ch for ch in c if ch.isdigit())
        if not c_clean:
            continue
        c_padded = c_clean.zfill(5)
        codes.append(c_padded)

    # 去重保持顺序
    seen = set()
    unique_codes = []
    for c in codes:
        if c not in seen:
            seen.add(c)
            unique_codes.append(c)

    # 解析股票名称
    names = {}
    if name_col is not None and name_col in df.columns:
        for _, row in df.iterrows():
            code_raw = row[code_col]
            if pd.isna(code_raw):
                continue
            code_clean = "".join(ch for ch in str(code_raw).strip() if ch.isdigit()).zfill(5)
            name_val = row[name_col]
            if code_clean and pd.notna(name_val):
                names[code_clean] = str(name_val).strip()
            elif code_clean:
                names.setdefault(code_clean, "未知")

    return unique_codes, names


def pad_stock_code(code) -> str:
    """补全港股代码至 5 位（导出供其他文件复用）"""
    code_str = str(code).strip()
    return "".join(ch for ch in code_str if ch.isdigit()).zfill(5)


# ============================ StockDataUpdater 类 ============================
class StockDataUpdater:
    """H 股数据更新器（结构与 A 股版保持一致）"""

    def __init__(self, max_workers=None, retry_attempts=3, base_delay=0.5):
        """
        初始化数据更新器

        Args:
            max_workers: 最大线程数，None 表示自动检测
            retry_attempts: 网络重试次数
            base_delay: 基础延时（秒）
        """
        self.max_workers = max_workers or min(32, self.get_physical_cpu_cores() * 2)
        self.retry_attempts = retry_attempts
        self.base_delay = base_delay
        self.lock = threading.Lock()
        self.success_count = 0
        self.error_count = 0
        self.skip_count = 0
        self.failed_stocks = []  # 记录失败的股票代码和原因
        self.stock_names = {}  # 缓存股票名称

    # --------------------------- 通用工具 ---------------------------
    def get_stock_name(self, stock_code):
        """获取股票名称"""
        return self.stock_names.get(stock_code, "未知")

    def format_failure_reason(self, reason):
        """格式化失败原因，使其更友好"""
        reason_lower = reason.lower()

        if "无增量数据" in reason:
            return "无增量数据（停牌）"
        elif "no value to decode" in reason_lower:
            return "数据解码失败"
        elif "merge keys are not unique" in reason_lower:
            return "数据合并失败"
        elif "连接" in reason or "connection" in reason_lower:
            return "网络连接失败"
        elif "超时" in reason or "timeout" in reason_lower:
            return "请求超时"
        elif "无效代码" in reason:
            return "股票代码无效"
        else:
            # 截取前 30 个字符，避免显示过长
            return reason[:30] + "..." if len(reason) > 30 else reason

    def get_physical_cpu_cores(self):
        """获取物理 CPU 核心数（而非逻辑处理器数）"""
        try:
            if hasattr(psutil, 'cpu_count'):
                physical_cores = psutil.cpu_count(logical=False)
                if physical_cores:
                    return physical_cores
            logical_cores = os.cpu_count() or 1
            return max(1, logical_cores // 2)
        except Exception:
            return 4

    def get_optimal_thread_count(self, total_stocks):
        """动态计算最优线程数：基于股票总数与物理核心数"""
        physical_cores = self.get_physical_cpu_cores()
        logical_cores = os.cpu_count() or 1

        if total_stocks < 50:
            return min(4, physical_cores)
        elif total_stocks < 200:
            return min(8, physical_cores * 2)
        else:
            return min(16, physical_cores * 3)  # 最多使用物理核心数的 3 倍

    def random_sleep(self, min_seconds=None, max_seconds=None):
        """智能随机延时（防止接口触发限流）"""
        if min_seconds is None:
            min_seconds = self.base_delay
        if max_seconds is None:
            max_seconds = self.base_delay * 2

        sleep_time = random.uniform(min_seconds, max_seconds)
        time.sleep(sleep_time)

    def get_date_range(self, years=3):
        """生成动态时间范围（备用函数，主流程不直接使用）"""
        end_date = datetime.now().strftime("%Y%m%d")
        start_date = (datetime.now() - relativedelta(years=years)).strftime("%Y%m%d")
        return start_date, end_date

    def filter_by_date(self, df, start_date, end_date):
        """通用日期过滤函数"""
        date_col = next((col for col in ['trade_date', 'date', '日期'] if col in df.columns), None)
        if not date_col:
            raise ValueError(f"未找到日期字段，可用列名: {df.columns.tolist()}")

        df = df.copy()
        df['date'] = pd.to_datetime(df[date_col])
        start_dt = datetime.strptime(start_date, "%Y%m%d")
        end_dt = datetime.strptime(end_date, "%Y%m%d")
        return df[(df['date'] >= start_dt) & (df['date'] <= end_dt)]

    def detect_latest_trade_date(self, codes, min_samples=5, max_samples=10, lookback_days=30):
        """
        从给定股票代码中随机抽样，检测最新交易日。

        实现思路：
            - 取 5~10 只随机港股，回看最近 30 个日历日
            - 用 stock_hk_hist 拉取每只股票的日线，收集每只 max(date)
            - 取出现次数最多的日期作为 latest_trade_date，避免个别停牌股干扰

        Args:
            codes: 可迭代的港股代码列表
            min_samples: 最少抽样数量（不超过股票总数）
            max_samples: 最多抽样数量
            lookback_days: 回看天数窗口
        """
        unique_codes = list({self.pad_stock_code(c) for c in codes if pd.notna(c)})
        if not unique_codes:
            return datetime.now().strftime("%Y%m%d")

        sample_count = min(len(unique_codes), max_samples)
        sample_count = max(sample_count, min(len(unique_codes), min_samples))
        if sample_count <= 0:
            return datetime.now().strftime("%Y%m%d")

        sample_codes = random.sample(unique_codes, sample_count)

        end_candidate = datetime.now().strftime("%Y%m%d")
        start_candidate = (datetime.now() - relativedelta(days=lookback_days)).strftime("%Y%m%d")

        latest_dates = []

        for code in sample_codes:
            try:
                df = self.fetch_data_with_retry(
                    ak.stock_hk_hist,
                    symbol=code,
                    period="daily",
                    start_date=start_candidate,
                    end_date=end_candidate,
                    adjust="qfq",
                )
                if df is None or df.empty:
                    continue
                if 'date' not in df.columns and '日期' in df.columns:
                    df = df.rename(columns={'日期': 'date'})
                if 'date' not in df.columns:
                    continue
                df['date'] = pd.to_datetime(df['date'])
                latest_dates.append(df['date'].max())
            except Exception:
                continue

        if not latest_dates:
            return datetime.now().strftime("%Y%m%d")

        date_series = pd.to_datetime(pd.Series(latest_dates)).dt.normalize()
        latest_trade_date = date_series.value_counts().idxmax()
        return latest_trade_date.strftime("%Y%m%d")

    # --------------------------- 技术指标：通达信 SMA / KDJ ---------------------------
    def tdx_sma(self, series, n, m=1):
        """
        通达信 SMA 平滑算法实现：
            SMA(X, N, M) = (M * X + (N - M) * Y_prev) / N

        与 A 股版完全一致。遇 NaN 继承上一交易日值（与通达信行为一致）。
        """
        s = pd.Series(series).astype(float)
        values = s.values
        result = np.full_like(values, np.nan, dtype=float)

        prev = np.nan
        for i, x in enumerate(values):
            if np.isnan(x):
                result[i] = prev
                continue
            if np.isnan(prev):
                result[i] = x
            else:
                result[i] = (m * x + (n - m) * prev) / n
            prev = result[i]

        return pd.Series(result, index=s.index)

    def calculate_kdj(self, df, n=9, m1=3, m2=3):
        """
        计算 KDJ 指标，与通达信/同花顺股票软件保持一致。

        通达信公式（默认 N=9, M1=3, M2=3）：
            RSV := (CLOSE - LLV(LOW, N)) / (HHV(HIGH, N) - LLV(LOW, N)) * 100
            K   := SMA(RSV, M1, 1)
            D   := SMA(K,   M2, 1)
            J   := 3 * K - 2 * D

        Args:
            df: 包含 high, low, close 列的 DataFrame，必须按时间正序排列
            n: RSV 计算周期，默认 9
            m1: K 值平滑周期，默认 3
            m2: D 值平滑周期，默认 3
        """
        df_work = df.copy()
        if 'date' in df_work.columns:
            df_work = df_work.sort_values('date').reset_index(drop=True)
        elif 'trade_date' in df_work.columns:
            df_work = df_work.sort_values('trade_date').reset_index(drop=True)

        high = df_work['high'].values
        low = df_work['low'].values
        close = df_work['close'].values
        length = len(df_work)

        # 1) 计算 RSV：N 日窗口 HHV/LLV
        rsv = np.full(length, np.nan)
        for i in range(n - 1, length):
            period_high = high[i - n + 1:i + 1]
            period_low = low[i - n + 1:i + 1]
            highest = np.max(period_high)
            lowest = np.min(period_low)

            if highest != lowest:
                rsv[i] = 100 * (close[i] - lowest) / (highest - lowest)
            else:
                rsv[i] = 50  # 最高等于最低时 RSV 取 50

        # 2) K / D：使用通达信 SMA 平滑（首个有效值即作为初值）
        k = self.tdx_sma(pd.Series(rsv), n=m1, m=1).values
        d = self.tdx_sma(pd.Series(k), n=m2, m=1).values
        # 3) J = 3K - 2D
        j = 3 * k - 2 * d

        return k, d, j

    # --------------------------- 技术指标：集成 ---------------------------
    def calculate_ta_indicators(self, df):
        """
        计算技术指标（与 A 股版保持一致）。

        包含指标：
            KDJ (K/D/J)        - 通达信算法
            BBI / BBI_DIF      - MA3/6/12/24 均值及其差分
            MACD (DIF/DEA/MACD) - 12/26/9
            short_term_fund    - 3 周期资金指标
            long_term_fund     - 21 周期资金指标
            Short_LS           - SMA14/28/57/114 均值
            Short_Trend        - 双 EMA(10)
            Brick_High/Low     - 通达信砖形图

        注：H 股版跳过 PE 通道（L2/L1/M/H1/H2）与 investment_income，
           因为 stock_hk_financial_indicator_em 返回的是当前一期快照，
           没有时序历史，无法做按日期分位的 PE 通道计算。
        """
        # KDJ：使用与 A 股版完全一致的通达信实现
        df['K'], df['D'], df['J'] = self.calculate_kdj(df, n=9, m1=3, m2=3)

        # BBI
        periods = [3, 6, 12, 24]
        for p in periods:
            df[f'MA{p}'] = talib.SMA(df['close'], timeperiod=p)
        df['BBI'] = df[[f'MA{p}' for p in periods]].mean(axis=1)
        df['BBI_DIF'] = df['BBI'].diff().fillna(0)

        # MACD
        df['DIF'], df['DEA'], df['MACD'] = talib.MACD(
            df['close'],
            fastperiod=12,
            slowperiod=26,
            signalperiod=9,
        )

        # 短期资金指标（3 周期）
        df['short_term_low'] = df['low'].rolling(window=3).min()
        df['short_term_high'] = df['close'].rolling(window=3).max()
        df['short_term_fund'] = 100 * (df['close'] - df['short_term_low']) / (
                df['short_term_high'] - df['short_term_low'])

        # 长期资金指标（21 周期）
        df['long_term_low'] = df['low'].rolling(window=21).min()
        df['long_term_high'] = df['close'].rolling(window=21).max()
        df['long_term_fund'] = 100 * (df['close'] - df['long_term_low']) / (
                df['long_term_high'] - df['long_term_low'])

        # 短期多空线 Short_LS（SMA14/28/57/114 均值）
        sma14 = talib.SMA(df['close'], timeperiod=14)
        sma28 = talib.SMA(df['close'], timeperiod=28)
        sma57 = talib.SMA(df['close'], timeperiod=57)
        sma114 = talib.SMA(df['close'], timeperiod=114)
        df['Short_LS'] = np.round((sma14 + sma28 + sma57 + sma114) / 4.0, 2)

        # 短期趋势 Short_Trend（双 EMA10）
        ema10_first = talib.EMA(df['close'], timeperiod=10)
        df['Short_Trend'] = np.round(talib.EMA(ema10_first, timeperiod=10), 2)

        # 砖形图指标 Brick_High / Brick_Low（基于通达信公式）
        hhv_4 = df['high'].rolling(window=4, min_periods=4).max()
        llv_4 = df['low'].rolling(window=4, min_periods=4).min()
        denom = (hhv_4 - llv_4).replace(0, np.nan)

        var1a = (hhv_4 - df['close']) / denom * 100 - 90
        var2a = self.tdx_sma(var1a, n=4, m=1) + 100
        var3a = (df['close'] - llv_4) / denom * 100
        var4a = self.tdx_sma(var3a, n=6, m=1)
        var5a = self.tdx_sma(var4a, n=6, m=1) + 100
        var6a = var5a - var2a

        brick_value = np.where(var6a > 4, var6a - 4, 0.0)
        brick_series = pd.Series(brick_value, index=df.index)
        prev_brick = brick_series.shift(1)

        df['Brick_High'] = brick_series
        df['Brick_Low'] = prev_brick

        # 删除临时列
        df = df.drop(columns=[f'MA{p}' for p in periods] +
                             ['short_term_low', 'short_term_high',
                              'long_term_low', 'long_term_high'])

        return df

    # --------------------------- 合并 & 保存 ---------------------------
    def merge_and_save(self, price_df, indicator_df, save_path, symbol, end_date):
        """
        合并行情 + 估值快照 + 技术指标，写入 CSV。

        Args:
            price_df: 行情 DataFrame（含 open/high/low/close/volume；date 列）
            indicator_df: 估值快照 DataFrame（来自 stock_hk_financial_indicator_em，
                          含 market_cap / float_market_cap / pe_ttm / pb）
            save_path: 输出 CSV 路径
            symbol: 港股代码（5 位字符串）
            end_date: 价格窗口的截止日期（YYYYMMDD），用于把单行估值快照锚到末日

        H 股版与 A 股版的差异：
            - 估值列只有 4 列（market_cap / float_market_cap / pe_ttm / pb）；
              pe_static / peg / pcf / ps 暂不写入。
            - 估值快照是单行（当前一期），通过 end_date 对齐 + ffill/bfill 填充到所有日期。
        """
        try:
            # 计算技术指标
            price_df = self.calculate_ta_indicators(price_df.copy())

            # 添加股票代码标识
            price_df = price_df.assign(symbol=symbol)
            indicator_df = indicator_df.assign(symbol=symbol)

            # 日期处理
            price_df['date'] = pd.to_datetime(
                price_df['trade_date'] if 'trade_date' in price_df else price_df['date']
            )
            indicator_df['date'] = pd.to_datetime(indicator_df['trade_date'])

            # 估值快照是单行，把 trade_date 强制对齐到价格窗口末日 end_date，
            # 保证 left join 能命中价格表最后一行；后续 ffill/bfill 把这一行扩散到全历史日
            indicator_df['date'] = pd.to_datetime(end_date)

            # 左连接：估值快照按 date 对齐（实际只有 1 行，所有日期共享）
            merged_df = pd.merge(
                price_df,
                indicator_df,
                on=['date', 'symbol'],
                how='left',
                suffixes=('', '_indicator'),
            ).sort_values('date').reset_index(drop=True)

            # 估值列 ffill/bfill（单行快照 → 整列同值）
            indicator_cols = ['market_cap', 'float_market_cap', 'pe_ttm', 'pb']
            for col in indicator_cols:
                if col in merged_df.columns:
                    merged_df[col] = merged_df[col].ffill().bfill()

            # 去重
            merged_df = merged_df.drop_duplicates(subset=['date'], keep='last')

            # 数值列清洗（按 symbol 分组 ffill/bfill）
            numeric_cols = merged_df.select_dtypes(include=np.number).columns.difference(['symbol']).tolist()
            numeric_cols = [col for col in numeric_cols if col not in ['date', 'symbol']]
            if numeric_cols:
                merged_df[numeric_cols] = merged_df.groupby('symbol', group_keys=False)[numeric_cols].apply(
                    lambda x: x.ffill().bfill()
                )

            # 输出列（H 股版：22 列；不含 PE 通道 / investment_income / pe_static/peg/pcf/ps）
            base_cols = ['date', 'symbol', 'open', 'high', 'low', 'close', 'volume']
            ta_cols = ['K', 'D', 'J', 'BBI', 'BBI_DIF', 'DIF', 'DEA', 'MACD',
                       'short_term_fund', 'long_term_fund',
                       'Short_LS', 'Short_Trend', 'Brick_High', 'Brick_Low']
            value_cols = ['market_cap', 'float_market_cap', 'pe_ttm', 'pb']

            all_columns = base_cols + ta_cols + value_cols
            output_columns = [col for col in all_columns if col in merged_df.columns]

            # 保存：用 Python 原生文本写入（utf-8-sig），避免 pandas to_csv 的 mmap 句柄问题
            import gc as _gc
            _gc.collect()
            with open(save_path, "w", encoding="utf-8-sig", newline="") as fh:
                merged_df[output_columns].to_csv(fh, index=False)

        except Exception as e:
            raise Exception(f"合并保存失败: {str(e)[:100]}")

    # --------------------------- 网络层：重试 + 超时 ---------------------------
    def fetch_data_with_retry(self, func, *args, **kwargs):
        """
        带重试机制和超时控制的数据获取。

        每次调用：
            - 随机延时 0.2~0.5 秒防限流
            - 用 ThreadPoolExecutor(max_workers=1) + future.result(timeout=15) 强制 15s 超时
        重试策略：
            - 最多 retry_attempts 次
            - TimeoutError: 退避 (attempt+1)*3 秒
            - 其他异常: 退避 (attempt+1)*2 秒
        """
        timeout_seconds = 15
        for attempt in range(self.retry_attempts):
            try:
                self.random_sleep(0.2, 0.5)
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(func, *args, **kwargs)
                    try:
                        result = future.result(timeout=timeout_seconds)
                    except concurrent.futures.TimeoutError:
                        raise TimeoutError(f"请求超时 ({timeout_seconds}秒)")
                if result is not None and not result.empty:
                    return result
            except TimeoutError as e:
                if attempt < self.retry_attempts - 1:
                    delay = (attempt + 1) * 3
                    time.sleep(delay)
                    continue
                raise e
            except Exception as e:
                if attempt < self.retry_attempts - 1:
                    delay = (attempt + 1) * 2
                    time.sleep(delay)
                    continue
                raise e
        return None

    # --------------------------- 工具：代码补 0 ---------------------------
    @staticmethod
    def pad_stock_code(code):
        """补全港股代码至 5 位（实例方法版）"""
        code_str = str(code).strip()
        return "".join(ch for ch in code_str if ch.isdigit()).zfill(5)

    # --------------------------- 工具：读历史 CSV ---------------------------
    def _read_existing_history(self, save_path: str, start_date: str):
        """
        读取已存在的 csv 历史数据，并做规范化处理。

        Returns:
            (history_df, latest_date_str, aligned_history_df)
            - history_df:          保留全量日期的历史数据
            - latest_date_str:     csv A 列(date) 的最大日期，格式 YYYYMMDD
            - aligned_history_df:  仅保留 date >= start_date 的窗口内数据（用于拼接）
        """
        history_df = pd.read_csv(save_path, encoding='utf-8-sig')
        # 兼容 BOM / 旧版本列名
        history_df = history_df.rename(columns={
            '﻿date': 'date',
            'trade_date': 'date',
            '日期': 'date',
        })

        if 'date' not in history_df.columns:
            raise ValueError(f"日期列缺失，实际列名: {history_df.columns.tolist()}")

        history_df['date'] = pd.to_datetime(history_df['date'])
        latest_date = history_df['date'].max()
        latest_date_str = latest_date.strftime("%Y%m%d")

        aligned_history_df = history_df[history_df['date'] >= pd.to_datetime(start_date)].copy()
        return history_df, latest_date_str, aligned_history_df

    # --------------------------- H 股特有：行情获取 ---------------------------
    def fetch_hk_price(self, raw_code, start_date, end_date):
        """
        获取港股日线行情（前复权）。

        首选 ak.stock_hk_hist（东方财富，服务端日期过滤，更快）；
        失败则回退 ak.stock_hk_daily（新浪，客户端日期过滤）。

        Returns:
            DataFrame 列：date / open / high / low / close / volume
        """
        # ----- 1) 首选：stock_hk_hist（东方财富）-----
        try:
            df = self.fetch_data_with_retry(
                ak.stock_hk_hist,
                symbol=raw_code,
                period="daily",
                start_date=start_date,
                end_date=end_date,
                adjust="qfq",
            )
            if df is not None and not df.empty:
                # 列名归一化（中文列名 → 英文）
                if '日期' in df.columns:
                    df = df.rename(columns={
                        '日期': 'date',
                        '开盘': 'open',
                        '最高': 'high',
                        '最低': 'low',
                        '收盘': 'close',
                        '成交量': 'volume',
                    })
                required = {'date', 'open', 'high', 'low', 'close', 'volume'}
                if required.issubset(df.columns):
                    df = df[list(required)].copy()
                    df['date'] = pd.to_datetime(df['date'])
                    return df.reset_index(drop=True)
        except Exception:
            pass

        # ----- 2) 备用：stock_hk_daily（新浪）-----
        try:
            df = self.fetch_data_with_retry(
                ak.stock_hk_daily,
                symbol=raw_code,
                adjust="qfq",
            )
            if df is None or df.empty:
                return pd.DataFrame()
            df['date'] = pd.to_datetime(df['date'])
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            df = df[(df['date'] >= start_dt) & (df['date'] <= end_dt)].copy()
            return df.reset_index(drop=True)
        except Exception:
            return pd.DataFrame()

    # --------------------------- H 股特有：估值快照 ---------------------------
    def fetch_hk_indicator(self, raw_code):
        """
        获取港股最新一期估值快照（东方财富）。

        接口：ak.stock_hk_financial_indicator_em(symbol)
        返回单行 DataFrame，含 总市值(港元) / 港股市值(港元) / 市盈率 / 市净率 等 21 列。

        字段映射：
            总市值(港元)   → market_cap
            港股市值(港元) → float_market_cap
            市盈率        → pe_ttm
            市净率        → pb

        注：H 股无 PE(静)/PEG/PCF/PS 的稳定接口，本期不取这 4 列。
        """
        try:
            df = self.fetch_data_with_retry(
                ak.stock_hk_financial_indicator_em,
                symbol=raw_code,
            )
            if df is None or df.empty:
                return pd.DataFrame()

            # 列名映射
            df = df.rename(columns={
                '总市值(港元)':   'market_cap',
                '港股市值(港元)': 'float_market_cap',
                '市盈率':        'pe_ttm',
                '市净率':        'pb',
            })
            cols = ['market_cap', 'float_market_cap', 'pe_ttm', 'pb']
            df = df[[c for c in cols if c in df.columns]].copy()

            # 估值快照没有 trade_date 列，用"今天"作为所有日期的共同锚点
            df['trade_date'] = pd.Timestamp(datetime.now().strftime("%Y-%m-%d"))
            return df
        except Exception:
            return pd.DataFrame()

    # --------------------------- H 股特有：单股票处理 ---------------------------
    def process_hk_single_stock(self, raw_code, start_date, end_date, save_dir):
        """
        处理单只港股：全量/增量分支 + 3 次外层重试。

        Args:
            raw_code: 5 位港股代码字符串
            start_date: 全量拉取起始日期（YYYYMMDD），固定 HISTORY_START_DATE
            end_date: 拉取截止日期（YYYYMMDD），来自最新交易日探测
            save_dir: CSV 保存目录
        """
        max_retries = 3
        retry_delay = 5  # 重试延迟秒数

        for attempt in range(max_retries):
            try:
                # 校验代码
                if not raw_code or len(raw_code) != 5 or not raw_code.isdigit():
                    return f"❌ 无效代码: {raw_code}"

                save_path = os.path.join(save_dir, f"{raw_code}.csv")
                # 是否增量更新（csv 已存在即走增量分支）
                is_incremental = os.path.exists(save_path)

                if is_incremental:
                    # ============ 已有文件：基于 csv A 列日期做增量判断 ============
                    try:
                        # 1) 读取 csv 最大日期
                        history_df, latest_date_str, aligned_history_df = self._read_existing_history(
                            save_path, start_date
                        )

                        # 2) 与本次 end_date 对比：一致则直接跳过
                        if latest_date_str == end_date:
                            return f"⏩ 已是最新数据: {raw_code}"

                        # 3) 从 latest_date + 1 到 end_date 拉增量
                        new_start = (
                            pd.to_datetime(latest_date_str) + pd.Timedelta(days=1)
                        ).strftime("%Y%m%d")

                        temp_price_df = self.fetch_hk_price(raw_code, new_start, end_date)

                        if temp_price_df is None or temp_price_df.empty:
                            return f"⚠️ 无增量数据: {raw_code}"

                        # 把增量 date 列重命名为 trade_date，与全量分支统一
                        if 'date' in temp_price_df.columns and 'trade_date' not in temp_price_df.columns:
                            temp_price_df = temp_price_df.rename(columns={'date': 'trade_date'})
                        if 'date' in aligned_history_df.columns and 'trade_date' not in aligned_history_df.columns:
                            aligned_history_df = aligned_history_df.rename(columns={'date': 'trade_date'})

                        # 估值快照（无时间窗口概念，单行即可）
                        temp_indicator_df = self.fetch_hk_indicator(raw_code)
                        if temp_indicator_df is None or temp_indicator_df.empty:
                            return f"⚠️ 无估值数据: {raw_code}"

                        # 4) 拼接：aligned_history_df（窗口内旧数据） + temp_price_df（增量）
                        combined_price = pd.concat(
                            [aligned_history_df, temp_price_df], ignore_index=True
                        )
                        if 'trade_date' in combined_price.columns:
                            combined_price['date'] = pd.to_datetime(combined_price['trade_date'])
                            combined_price = combined_price.drop(columns=['trade_date'])
                        else:
                            combined_price['date'] = pd.to_datetime(combined_price['date'])
                        combined_price = combined_price.drop_duplicates(
                            subset=['date'], keep='last'
                        ).sort_values('date').reset_index(drop=True)

                        indicator_df = temp_indicator_df

                    except Exception as e:
                        if attempt < max_retries - 1:
                            print(f"重试 {raw_code} - 原因: {str(e)[:100]}")
                            time.sleep(retry_delay)
                            continue
                        return f"❌ 处理历史数据失败: {str(e)[:100]} - {raw_code}"

                else:
                    # ============ 文件不存在：全量拉取 ============
                    try:
                        full_price_df = self.fetch_hk_price(raw_code, start_date, end_date)

                        if full_price_df is None or full_price_df.empty:
                            return f"⚠️ 无价格数据: {raw_code}"

                        full_price_df = full_price_df.rename(columns={'date': 'trade_date'})

                        full_indicator_df = self.fetch_hk_indicator(raw_code)
                        if full_indicator_df is None or full_indicator_df.empty:
                            return f"⚠️ 无估值数据: {raw_code}"

                        combined_price = full_price_df
                        indicator_df = full_indicator_df

                    except Exception as e:
                        if attempt < max_retries - 1:
                            print(f"重试 {raw_code} - 原因: {str(e)[:100]}")
                            time.sleep(retry_delay)
                            continue
                        return f"❌ 获取全量数据失败: {str(e)[:100]} - {raw_code}"

                # ============ 统一存储：调用 merge_and_save 重算技术指标 ============
                try:
                    self.merge_and_save(
                        price_df=combined_price,
                        indicator_df=indicator_df,
                        save_path=save_path,
                        symbol=raw_code,
                        end_date=end_date,
                    )
                    tag = "🔄 增量" if is_incremental else "🆕 全量"
                    return f"✅ {tag} {raw_code}"

                except Exception as e:
                    if attempt < max_retries - 1:
                        print(f"重试 {raw_code} - 原因: {str(e)[:100]}")
                        time.sleep(retry_delay)
                        continue
                    return f"❌ 保存数据失败: {str(e)[:100]} - {raw_code}"

            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"重试 {raw_code} - 原因: {str(e)[:100]}")
                    time.sleep(retry_delay)
                    continue
                return f"❌ 处理失败: {str(e)[:100]} - {raw_code}"

    # --------------------------- 线程安全统计 ---------------------------
    def update_statistics(self, result):
        """更新统计信息（线程安全）"""
        with self.lock:
            if result.startswith("✅"):
                self.success_count += 1
            elif result.startswith("⏩"):
                self.skip_count += 1
            else:
                self.error_count += 1
                if "❌" in result or "⚠️" in result:
                    stock_code = "未知"
                    reason = "未知"

                    # 提取股票代码（5 位数字）
                    code_match = re.search(r'\b\d{5}\b', result)
                    if code_match:
                        stock_code = code_match.group()

                    # 提取失败原因
                    if ": " in result:
                        parts = result.split(": ", 1)
                        if len(parts) == 2:
                            if parts[1].strip().isdigit() and len(parts[1].strip()) == 5:
                                reason = parts[0].replace("⚠️ ", "").replace("❌ ", "").strip()
                            else:
                                error_msg = parts[1]
                                error_msg = re.sub(r'\s*-\s*\d{5}$', '', error_msg)
                                reason = error_msg.strip()

                    self.failed_stocks.append((stock_code, reason))

    # --------------------------- 主流程 ---------------------------
    def main(
        self,
        stock_list_path: str = None,
        save_dir: str = None,
        max_stocks: int = None,
    ):
        """
        主函数（支持被其他脚本调用时指定股票清单与保存目录）

        Args:
            stock_list_path: 港股清单 xlsx 路径
                - 传具体 xlsx 路径时按该文件处理
                - 传 None 时使用默认 HSharesList.xlsx 路径
            save_dir: 数据保存目录；None 时使用默认 DEFAULT_SAVE_DIR
            max_stocks: 若不为 None，则只处理前 max_stocks 只股票（用于快速验证）
        """
        # 路径配置：未指定时使用默认值
        stock_list_path = stock_list_path or DEFAULT_STOCK_LIST_PATH
        save_dir = save_dir or DEFAULT_SAVE_DIR
        os.makedirs(save_dir, exist_ok=True)

        # 读取股票清单
        try:
            codes, names = load_stock_codes_from_xlsx(stock_list_path)
            self.stock_names.update(names)
            stock_codes = list(codes)
            total = len(stock_codes)

            if max_stocks is not None and total > max_stocks:
                stock_codes = stock_codes[:max_stocks]
                total = len(stock_codes)

            print(f"📄 股票清单: {stock_list_path}")
            print(f"📊 总股票数: {total}")

        except Exception as e:
            print(f"读取股票清单失败 [{stock_list_path}]: {e}")
            return

        # 动态计算线程数
        optimal_threads = self.get_optimal_thread_count(total)
        physical_cores = self.get_physical_cpu_cores()
        logical_cores = os.cpu_count() or 1
        print(f"🧵 使用线程数: {optimal_threads} (基于{physical_cores}核计算)")
        print(f"💻 CPU核心数: {physical_cores}核{logical_cores}线程")
        print(f"💾 可用内存: {psutil.virtual_memory().available // (1024 ** 3)} GB")

        # 探测最新交易日 + 固定起始日期
        latest_trade_date = self.detect_latest_trade_date(
            stock_codes, min_samples=5, max_samples=10, lookback_days=30
        )
        start_date = HISTORY_START_DATE
        end_date = latest_trade_date
        print(f"📅 最新交易日: {latest_trade_date}")
        print(f"📅 数据日期范围: {start_date} - {end_date}")

        start_time = time.time()

        # 多线程主循环
        with tqdm(total=total, desc="🚀 多线程数据更新",
                  bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]') as pbar:
            with concurrent.futures.ThreadPoolExecutor(max_workers=optimal_threads) as executor:
                # 偏函数绑定公共参数
                process_func = partial(
                    self.process_hk_single_stock,
                    start_date=start_date,
                    end_date=end_date,
                    save_dir=save_dir,
                )

                # 提交所有任务
                future_to_stock = {
                    executor.submit(process_func, code): code
                    for code in stock_codes
                }

                # 处理完成的任务
                for future in concurrent.futures.as_completed(future_to_stock):
                    code = future_to_stock[future]
                    try:
                        result = future.result()
                        self.update_statistics(result)

                        pbar.set_postfix({
                            '成功': self.success_count,
                            '跳过': self.skip_count,
                            '失败': self.error_count,
                            '当前': result[:20] + "..." if len(result) > 20 else result,
                        })

                    except Exception as e:
                        self.update_statistics(f"❌ 异常: {code}")
                        pbar.set_postfix_str(f"❌ 异常: {code}")
                    finally:
                        pbar.update(1)

        end_time = time.time()
        duration = end_time - start_time

        # 最终统计
        print(f"\n🎉 数据更新完成!")
        print(f"⏱️  总耗时: {duration:.2f} 秒")
        print(f"📈 成功: {self.success_count} 只")
        print(f"⏩ 跳过: {self.skip_count} 只")
        print(f"❌ 失败: {self.error_count} 只")
        if duration > 0:
            print(f"⚡ 平均速度: {total / duration:.2f} 只/秒")
        print(f"💾 数据保存目录: {save_dir}")

        if self.error_count > 0:
            print(f"⚠️  失败率: {self.error_count / total * 100:.1f}%")
            print(f"\n📋 失败股票详情:")
            for stock_code, reason in self.failed_stocks:
                stock_name = self.get_stock_name(stock_code)
                formatted_reason = self.format_failure_reason(reason)
                print(f"   {stock_code}   {stock_name}  {formatted_reason}")


# ============================ 外部调用入口 ============================
def run_Dailyupdate(
    stock_list_source: str = "default",
    save_dir: str = None,
    max_stocks: int = None,
) -> None:
    """
    供其他脚本调用的统一入口。

    Args:
        stock_list_source: 股票清单来源标识
            - "default"（默认）: 使用 HSharesList.xlsx 全量名单
            - "weekly"           : 自动从 WEEKLY_REPORT_DIR 中挑选日期最新的 xlsx 文件
            - 具体 xlsx 路径      : 按给定路径直接处理
        save_dir: 数据保存目录，None 时使用默认 DEFAULT_SAVE_DIR
        max_stocks: 若不为 None，则只处理前 max_stocks 只股票（用于快速验证）
    """
    # 解析股票清单路径
    if stock_list_source == "default":
        resolved_path = DEFAULT_STOCK_LIST_PATH
    elif stock_list_source == "weekly":
        resolved_path = resolve_latest_weekly_report()
        print(f"📌 使用最新周线选股结果: {resolved_path}")
    elif isinstance(stock_list_source, str) and stock_list_source.lower().endswith(".xlsx"):
        resolved_path = stock_list_source
    else:
        raise ValueError(
            f"不支持的 stock_list_source: {stock_list_source}。"
            f"请传入 'default'、'weekly' 或 xlsx 文件绝对路径。"
        )

    updater = StockDataUpdater(
        max_workers=None,  # 自动检测
        retry_attempts=3,
        base_delay=0.3,
    )
    updater.main(
        stock_list_path=resolved_path,
        save_dir=save_dir,
        max_stocks=max_stocks,
    )


# ============================ AkShare 接口选型说明 ============================
# 1) 行情数据：使用 ak.stock_hk_hist(symbol, period, start_date, end_date, adjust)
#    - 前复权 adjust='qfq'，可改为 'hfq'（后复权）或 ''（不复权）
#    - 服务端日期过滤（东方财富），速度优于 stock_hk_daily
#    - 备选 ak.stock_hk_daily(symbol, adjust) 返回全量历史，
#      本脚本在 hist 失败时作为 fallback（客户端日期过滤）
# 2) 估值快照：使用 ak.stock_hk_financial_indicator_em(symbol)
#    - 返回东方财富港股核心指标（含 总市值(港元)/港股市值(港元)/市盈率/市净率 等 21 列）
#    - 该接口只返回当前一期快照，无历史时序；
#      H 股版因此跳过 PE 通道（L2/L1/M/H1/H2），不输出 pe_static/peg/pcf/ps
# 3) 若 akshare 后续版本调整字段名导致合并报错，仅需调整本文件中对应的
#    rename(columns={...}) 映射与 filter_by_date 日期列名即可，模块对外接口不变。
# ===========================================================================


def main():
    """脚本直接运行时的入口：默认从 HSharesList.xlsx 全量更新"""
    run_Dailyupdate(stock_list_source="default")


if __name__ == "__main__":
    main()