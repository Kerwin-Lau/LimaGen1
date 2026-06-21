import pandas as pd
import numpy as np
import os
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
import subprocess
import sys
import psutil

# 禁用 SSL 警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


# ============================ 路径配置 ============================
# 股票清单（用于过滤与周线计算）
ASHARES_LIST_PATH = (
    r"D:\Quant\01_SwProj\04_VectorBT\02_Lima\Lima_Gen1"
    r"\01_Database\01_Ashares\ASharesList.xlsx"
)
# 日线 csv 数据目录
DAILY_DATA_DIR = (
    r"D:\Quant\01_SwProj\04_VectorBT\02_Lima\Lima_Gen1"
    r"\01_Database\01_Ashares\01_RawData-Daily"
)
# 周线 csv 数据保存目录
WEEKLY_SAVE_DIR = (
    r"D:\Quant\01_SwProj\04_VectorBT\02_Lima\Lima_Gen1"
    r"\01_Database\01_Ashares\02_RawData-Weekly"
)
# 日线更新脚本
DAILY_UPDATE_SCRIPT = (
    r"D:\Quant\01_SwProj\04_VectorBT\02_Lima\Lima_Gen1"
    r"\02_DataProcess\01_Ashares\01_DaliyUpdate\DataUpdate_ASharesDaily.py"
)
# 周线筛选结果报告输出目录
WEEKLY_REPORT_DIR = (
    r"D:\Quant\01_SwProj\04_VectorBT\02_Lima\Lima_Gen1"
    r"\02_DataProcess\01_Ashares\02_WeeklyUpdate\01_Report"
)
# ================================================================


class StockDataUpdater:
    """周线数据更新器（基于本地日线 csv）"""

    def __init__(self, max_workers=None, retry_attempts=3, base_delay=0.5):
        self.max_workers = max_workers or min(32, self.get_physical_cpu_cores() * 2)
        self.retry_attempts = retry_attempts
        self.base_delay = base_delay
        self.lock = threading.Lock()
        self.success_count = 0
        self.error_count = 0
        self.skip_count = 0
        self.failed_stocks = []
        self.stock_names = {}
        # 缓存 ASharesList 全量信息（含行业、地区），方便报告输出时复制
        self.stock_meta_df = None

    def get_stock_name(self, stock_code):
        return self.stock_names.get(stock_code, "未知")

    def get_physical_cpu_cores(self):
        try:
            if hasattr(psutil, 'cpu_count'):
                physical_cores = psutil.cpu_count(logical=False)
                if physical_cores:
                    return physical_cores
            logical_cores = os.cpu_count() or 1
            return max(1, logical_cores // 2)
        except:
            return 4

    def get_optimal_thread_count(self, total_stocks):
        physical_cores = self.get_physical_cpu_cores()
        if total_stocks < 50:
            return min(4, physical_cores)
        elif total_stocks < 200:
            return min(8, physical_cores * 2)
        else:
            return min(16, physical_cores * 3)

    def pad_stock_code(self, code):
        code_str = str(code).strip()
        return "".join(ch for ch in code_str if ch.isdigit()).zfill(6)

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

    def convert_price_to_weekly(self, df):
        """
        将日线价格数据转换为周线数据（周一开盘、周五收盘、本周最高、本周最低）
        """
        if df is None or df.empty:
            return df

        price_df = df.copy()
        if 'trade_date' in price_df.columns:
            price_df['date'] = pd.to_datetime(price_df['trade_date'])
            # 丢弃原 trade_date 列，避免 set_index 后 reset_index 产生重复列
            price_df = price_df.drop(columns=['trade_date'])
        elif 'date' in price_df.columns:
            price_df['date'] = pd.to_datetime(price_df['date'])
        else:
            raise ValueError("价格数据中未找到日期列")

        price_df = price_df.sort_values('date').set_index('date')

        agg_map = {}
        if 'open' in price_df.columns:
            agg_map['open'] = 'first'
        if 'high' in price_df.columns:
            agg_map['high'] = 'max'
        if 'low' in price_df.columns:
            agg_map['low'] = 'min'
        if 'close' in price_df.columns:
            agg_map['close'] = 'last'
        if 'volume' in price_df.columns:
            agg_map['volume'] = 'sum'
        if 'amount' in price_df.columns:
            agg_map['amount'] = 'sum'
        if 'outstanding_share' in price_df.columns:
            agg_map['outstanding_share'] = 'last'
        if 'turnover' in price_df.columns:
            agg_map['turnover'] = 'sum'

        weekly = price_df.resample('W-FRI').agg(agg_map)

        # 去除完全没有价格数据的周
        price_cols = [c for c in ['open', 'high', 'low', 'close'] if c in weekly.columns]
        if price_cols:
            weekly = weekly.dropna(subset=price_cols, how='all')

        weekly = weekly.reset_index().rename(columns={'date': 'trade_date'})
        # 防御：若因列名冲突出现重复 trade_date，保留第一个
        if weekly.columns.duplicated().any():
            weekly = weekly.loc[:, ~weekly.columns.duplicated()]
        return weekly

    def convert_indicator_to_weekly(self, df):
        """
        将日线指标/估值数据转换为周线数据（取每周最后一个交易日的数值）
        """
        if df is None or df.empty:
            return df

        ind_df = df.copy()
        if 'trade_date' in ind_df.columns:
            ind_df['date'] = pd.to_datetime(ind_df['trade_date'])
            # 丢弃原 trade_date 列，避免 set_index 后 reset_index 产生重复列
            ind_df = ind_df.drop(columns=['trade_date'])
        elif 'date' in ind_df.columns:
            ind_df['date'] = pd.to_datetime(ind_df['date'])
        else:
            raise ValueError("指标数据中未找到日期列")

        ind_df = ind_df.sort_values('date').set_index('date')
        weekly = ind_df.resample('W-FRI').last()
        weekly = weekly.reset_index().rename(columns={'date': 'trade_date'})
        # 防御：去重可能的重复列名
        if weekly.columns.duplicated().any():
            weekly = weekly.loc[:, ~weekly.columns.duplicated()]
        return weekly

    def tdx_sma(self, series, n, m=1):
        """
        通达信 SMA 平滑算法实现
        SMA(X, N, M) = (M * X + (N - M) * Y_前一日) / N

        通达信原版行为：
          - 序列首个非 NaN 值直接作为初值
          - 后续逐日按公式递推
          - 遇 NaN 继承上一交易日值
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
        计算KDJ指标，与通达信/同花顺等股票软件保持一致。

        通达信公式编辑器公式（默认参数 N=9, M1=3, M2=3）：
            RSV:=(CLOSE-LLV(LOW,N))/(HHV(HIGH,N)-LLV(LOW,N))*100;
            K:=SMA(RSV,M1,1);     // 通达信SMA(X,N,M)=(M*X+(N-M)*Y')/N
            D:=SMA(K,M2,1);       // 这里 m=1，n=M1/M2，所以 alpha=1/M1、1/M2
            J:=3*K-2*D;

        当 M1=M2=3 时，alpha = 1/3，等价于常见的 "K = (2/3)*K' + (1/3)*RSV" 形式，
        但要求使用通达信原生的 SMA 平滑逻辑（遇 NaN 继承上一交易日的值），
        不能用普通 EMA（普通 EMA 在 RSV 序列起始的 NaN 段会污染初始递推）。

        关键点（与通达信一致）：
            - 第一个有效 RSV 出现的位置 i0：K[i0] = RSV[i0]，D[i0] = RSV[i0]
            - i0 之前无 K/D 值（保持 NaN）
            - 之后每日 K = SMA(RSV, M1, 1)，D = SMA(K, M2, 1)

        Args:
            df: 包含 high, low, close 列的 DataFrame，必须按时间正序排列。
            n: RSV 计算周期，默认 9。
            m1: K 值平滑周期（SMA 周期），默认 3。
            m2: D 值平滑周期（SMA 周期），默认 3。
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

        # 1) 计算 RSV（未成熟随机值）：使用 N 周窗口的 HHV/LLV
        rsv = np.full(length, np.nan)
        for i in range(n - 1, length):
            period_high = high[i - n + 1:i + 1]
            period_low = low[i - n + 1:i + 1]
            highest = np.max(period_high)
            lowest = np.min(period_low)

            if highest != lowest:
                rsv[i] = 100 * (close[i] - lowest) / (highest - lowest)
            else:
                rsv[i] = 50  # 最高等于最低时，RSV 设为 50

        # 2) 计算 K 和 D：使用通达信原生 SMA 平滑
        #    K = SMA(RSV, M1, 1)：首个有效 RSV 即作为 K 初值
        #    D = SMA(K,   M2, 1)：首个有效 K   即作为 D 初值
        #    tdx_sma 内部遇 NaN 继承上一交易日值，与通达信行为一致
        k = self.tdx_sma(pd.Series(rsv), n=m1, m=1).values
        d = self.tdx_sma(pd.Series(k), n=m2, m=1).values

        # 3) 计算 J：J = 3 * K - 2 * D
        j = 3 * k - 2 * d

        return k, d, j

    def calculate_ta_indicators(self, df):
        """计算技术指标（与原版保持一致）"""
        df['K'], df['D'], df['J'] = self.calculate_kdj(df, n=9, m1=3, m2=3)

        periods = [3, 6, 12, 24]
        for p in periods:
            df[f'MA{p}'] = talib.SMA(df['close'], timeperiod=p)
        df['BBI'] = df[[f'MA{p}' for p in periods]].mean(axis=1)
        df['BBI_DIF'] = df['BBI'].diff().fillna(0)

        df['DIF'], df['DEA'], df['MACD'] = talib.MACD(
            df['close'],
            fastperiod=12,
            slowperiod=26,
            signalperiod=9
        )

        df['short_term_low'] = df['low'].rolling(window=3).min()
        df['short_term_high'] = df['close'].rolling(window=3).max()
        df['short_term_fund'] = 100 * (df['close'] - df['short_term_low']) / (
                df['short_term_high'] - df['short_term_low'])

        df['long_term_low'] = df['low'].rolling(window=21).min()
        df['long_term_high'] = df['close'].rolling(window=21).max()
        df['long_term_fund'] = 100 * (df['close'] - df['long_term_low']) / (
                df['long_term_high'] - df['long_term_low'])

        sma14 = talib.SMA(df['close'], timeperiod=14)
        sma28 = talib.SMA(df['close'], timeperiod=28)
        sma57 = talib.SMA(df['close'], timeperiod=57)
        sma114 = talib.SMA(df['close'], timeperiod=114)
        df['Short_LS'] = np.round((sma14 + sma28 + sma57 + sma114) / 4.0, 2)

        ema10_first = talib.EMA(df['close'], timeperiod=10)
        df['Short_Trend'] = np.round(talib.EMA(ema10_first, timeperiod=10), 2)

        df = df.drop(columns=[f'MA{p}' for p in periods] +
                             ['short_term_low', 'short_term_high', 'long_term_low', 'long_term_high'])

        return df

    def merge_and_save(self, price_df, indicator_df, save_path, symbol, end_date=None):
        """重构版数据合并保存函数（含技术指标）"""
        try:
            price_df = self.calculate_ta_indicators(price_df.copy())

            price_df = price_df.assign(symbol=symbol)
            price_df['date'] = pd.to_datetime(price_df['trade_date'] if 'trade_date' in price_df else price_df['date'])
            price_df = price_df.sort_values('date').drop_duplicates(subset=['date', 'symbol'], keep='last')

            merged_df = price_df.copy()

            if indicator_df is not None and not indicator_df.empty:
                try:
                    indicator_df = indicator_df.assign(symbol=symbol)
                    indicator_df['date'] = pd.to_datetime(indicator_df['trade_date'])
                    indicator_df = indicator_df.sort_values('date').drop_duplicates(subset=['date', 'symbol'],
                                                                                     keep='last')

                    merged_df = pd.merge(
                        price_df,
                        indicator_df,
                        on=['date', 'symbol'],
                        how='left',
                        suffixes=('_price', '_indicator')
                    ).sort_values('date').reset_index(drop=True)
                except Exception as e:
                    print(f"估值数据合并失败，代码 {symbol}: {str(e)[:80]}")
                    merged_df = price_df.copy()

            merged_df = merged_df.sort_values('date').drop_duplicates(subset=['date', 'symbol'], keep='last')

            numeric_cols = merged_df.select_dtypes(include=np.number).columns.difference(['symbol']).tolist()
            numeric_cols = [col for col in numeric_cols if col not in ['date', 'symbol']]

            if numeric_cols:
                merged_df[numeric_cols] = merged_df.groupby('symbol', group_keys=False)[numeric_cols].apply(
                    lambda x: x.ffill().bfill()
                )

            if end_date is not None:
                merged_df['last_trade_date'] = pd.to_datetime(end_date)

            base_cols = ['date', 'symbol', 'open', 'high', 'low', 'close', 'volume', 'amount', 'outstanding_share',
                         'turnover', 'last_trade_date']
            ta_cols = ['K', 'D', 'J', 'BBI', 'BBI_DIF', 'DIF', 'DEA', 'MACD', 'short_term_fund', 'long_term_fund',
                       'Short_LS', 'Short_Trend']
            value_cols = ['market_cap', 'float_market_cap', 'pe_ttm', 'pe_static', 'pb', 'peg', 'pcf', 'ps']

            all_columns = base_cols + ta_cols + value_cols
            output_columns = [col for col in all_columns if col in merged_df.columns]

            merged_df[output_columns].to_csv(save_path, index=False, encoding='utf_8_sig')

        except Exception as e:
            raise Exception(f"合并保存失败: {str(e)[:100]}")

    # ------------------------------------------------------------
    # Daily 校验与触发更新
    # ------------------------------------------------------------
    def _sample_daily_csv_dates(self, sample_count=10):
        """
        从 DAILY_DATA_DIR 随机抽取 sample_count 个 csv，读取 A 列 (date) 的最小/最大日期。

        Returns:
            (sample_codes, sample_results) 其中 sample_results 是
            list of dict: {code, min_date, max_date}，
            若目录下没有 csv，则返回 ([], [])
        """
        if not os.path.isdir(DAILY_DATA_DIR):
            return [], []

        all_csv = [f for f in os.listdir(DAILY_DATA_DIR) if f.lower().endswith(".csv")]
        if not all_csv:
            return [], []

        actual_count = min(sample_count, len(all_csv))
        sampled = random.sample(all_csv, actual_count)

        results = []
        codes = []
        for fname in sampled:
            code = os.path.splitext(fname)[0]
            file_path = os.path.join(DAILY_DATA_DIR, fname)
            try:
                df = pd.read_csv(file_path, encoding='utf_8_sig', usecols=[0])
                df.columns = ['date']
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
                df = df.dropna(subset=['date'])
                if df.empty:
                    continue
                results.append({
                    'code': code,
                    'min_date': df['date'].min(),
                    'max_date': df['date'].max(),
                })
                codes.append(code)
            except Exception:
                continue

        return codes, results

    def _check_daily_range_match(self, start_date, end_date, sample_count=10):
        """
        随机抽取 sample_count 个 daily csv，仅校验 A 列 (date) 的最新日期是否 == end_date。
        不再校验 start_date。

        Returns:
            (bool, list) (是否全部 end_date 匹配, 详细信息)
        """
        _, results = self._sample_daily_csv_dates(sample_count=sample_count)
        if not results:
            return False, []

        end_dt = datetime.strptime(end_date, "%Y%m%d")

        details = []
        all_match = True
        for r in results:
            max_ok = pd.notna(r['max_date']) and r['max_date'] == end_dt
            if not max_ok:
                all_match = False
            details.append({
                'code': r['code'],
                'min_date': r['min_date'],
                'max_date': r['max_date'],
                'end_match': max_ok,
            })

        return all_match, details

    def _trigger_daily_update(self):
        """
        调用 DAILY_UPDATE_SCRIPT 进行一次日线数据更新。
        """
        print(f"\n📥 日线数据不匹配，开始调用 DataUpdate_ASharesDaily.py ...")
        if not os.path.exists(DAILY_UPDATE_SCRIPT):
            raise FileNotFoundError(f"日线更新脚本不存在: {DAILY_UPDATE_SCRIPT}")

        # 使用当前 Python 解释器执行
        result = subprocess.run(
            [sys.executable, DAILY_UPDATE_SCRIPT],
            capture_output=True,
            text=True
        )
        # 输出末尾若干行（防止刷屏）
        out_tail = "\n".join(result.stdout.splitlines()[-20:]) if result.stdout else ""
        err_tail = "\n".join(result.stderr.splitlines()[-20:]) if result.stderr else ""
        print(out_tail)
        if result.returncode != 0:
            print(f"⚠️  日线更新脚本退出码: {result.returncode}")
            if err_tail:
                print(err_tail)
        else:
            print("✅ 日线数据更新完成")
        return result.returncode

    # ------------------------------------------------------------
    # ASharesList 读取与过滤
    # ------------------------------------------------------------
    def load_filtered_stock_list(self, xlsx_path=ASHARES_LIST_PATH):
        """
        读取 ASharesList.xlsx 并按要求过滤：
          - B列 股票名称 不包含 "ST"
          - C列 一二级行业 不包含 "房地产"
        返回过滤后的 DataFrame（含全部 5 列）。
        """
        if not os.path.exists(xlsx_path):
            raise FileNotFoundError(f"股票清单文件不存在: {xlsx_path}")

        df = pd.read_excel(xlsx_path, sheet_name=0, dtype=str)
        # 去掉列名前后的不可见字符
        df.columns = [str(c).strip().strip('﻿') for c in df.columns]

        # 列名约定：股票代码, 股票名称, 一二级行业, 细分行业, 地区
        code_col = df.columns[0]
        name_col = df.columns[1] if len(df.columns) > 1 else None
        industry_col = df.columns[2] if len(df.columns) > 2 else None

        df[code_col] = df[code_col].apply(self.pad_stock_code)
        df = df[df[code_col].str.len() == 6]

        if name_col is not None and name_col in df.columns:
            df = df[~df[name_col].fillna("").str.contains("ST", na=False)]
        if industry_col is not None and industry_col in df.columns:
            df = df[~df[industry_col].fillna("").str.contains("房地产", na=False)]

        # 缓存全量 meta（包含所有列），方便报告输出时按代码查找 5 列数据
        self.stock_meta_df = df.copy()
        return df

    # ------------------------------------------------------------
    # 单股票周线处理（从本地日线 csv 读取）
    # ------------------------------------------------------------
    def process_single_stock(self, raw_code, start_date, end_date, save_dir):
        """从本地日线 csv 读取数据并转换为周线"""
        max_retries = 3
        retry_delay = 5

        for attempt in range(max_retries):
            try:
                save_path = os.path.join(save_dir, f"{raw_code}.csv")

                # 文件存在性判断：如果最新日期已是 end_date，则直接跳过
                if os.path.exists(save_path):
                    try:
                        history_df = pd.read_csv(save_path, encoding='utf_8_sig')
                        history_df = history_df.rename(columns={
                            '﻿date': 'date',
                            'trade_date': 'date',
                            '日期': 'date'
                        })

                        latest_date = None
                        if 'last_trade_date' in history_df.columns:
                            history_df['last_trade_date'] = pd.to_datetime(
                                history_df['last_trade_date'], errors='coerce'
                            )
                            if history_df['last_trade_date'].notna().any():
                                latest_date = history_df['last_trade_date'].max()

                        if latest_date is None and 'date' in history_df.columns:
                            history_df['date'] = pd.to_datetime(history_df['date'], errors='coerce')
                            if history_df['date'].notna().any():
                                latest_date = history_df['date'].max()

                        if latest_date is not None:
                            latest_date_str = latest_date.strftime("%Y%m%d")
                            if latest_date_str == end_date:
                                return f"⏩ 已是最新数据: {raw_code}"
                    except Exception as e:
                        if attempt < max_retries - 1:
                            print(f"重试 {raw_code} - 原因: {str(e)[:100]}")
                            time.sleep(retry_delay)
                            continue
                        return f"❌ 处理历史数据失败: {str(e)[:100]} - {raw_code}"

                # 从本地日线 csv 读取数据
                daily_csv = os.path.join(DAILY_DATA_DIR, f"{raw_code}.csv")
                if not os.path.exists(daily_csv):
                    return f"⚠️ 缺失日线数据: {raw_code}"

                try:
                    try:
                        full_price_df = pd.read_csv(daily_csv, encoding='utf_8_sig')
                    except pd.errors.EmptyDataError:
                        # 日线 csv 损坏/为空：清理目标周线文件（避免下次误判跳过），直接返回失败
                        if os.path.exists(save_path):
                            try:
                                os.remove(save_path)
                            except OSError:
                                pass
                        return f"❌ 日线 csv 损坏/为空: {raw_code}"

                    if full_price_df.empty:
                        # 日线 csv 存在但解析后为空，删除目标 csv（如果存在），避免下次误判跳过
                        if os.path.exists(save_path):
                            try:
                                os.remove(save_path)
                            except OSError:
                                pass
                        return f"⚠️ 日线数据为空: {raw_code}"

                    # 处理 date/trade_date 列名
                    if 'trade_date' in full_price_df.columns and 'date' not in full_price_df.columns:
                        full_price_df = full_price_df.rename(columns={'trade_date': 'date'})

                    full_price_df = self.filter_by_date(full_price_df, start_date, end_date)
                    if full_price_df is None or full_price_df.empty:
                        return f"⚠️ 时间区间内无日线数据: {raw_code}"

                    full_price_df = full_price_df.rename(columns={'date': 'trade_date'})

                    # 日线数据转换为周线数据
                    weekly_price = self.convert_price_to_weekly(full_price_df)

                    # 估值/市值指标已合并在日线 csv 中：直接复用同源数据
                    weekly_indicator = self.convert_indicator_to_weekly(full_price_df)

                    # 估值列只保留与价格/技术指标不重叠的字段，
                    # 避免与 weekly_price 在 merge 时产生 _price/_indicator 后缀冲突
                    _price_ta_cols = {
                        'open', 'high', 'low', 'close', 'volume', 'amount',
                        'outstanding_share', 'turnover',
                        'K', 'D', 'J', 'BBI', 'BBI_DIF', 'DIF', 'DEA', 'MACD',
                        'short_term_fund', 'long_term_fund', 'Short_LS', 'Short_Trend',
                        'Brick_High', 'Brick_Low', 'L2', 'L1', 'M', 'H1', 'H2',
                        'investment_income',
                    }
                    _value_cols = [
                        'market_cap', 'float_market_cap', 'pe_ttm', 'pe_static',
                        'pb', 'peg', 'pcf', 'ps',
                    ]
                    keep_cols = ['trade_date'] + _value_cols
                    weekly_indicator = weekly_indicator[
                        [c for c in keep_cols if c in weekly_indicator.columns]
                    ]

                    if weekly_price is None or weekly_price.empty:
                        return f"⚠️ 周线数据为空: {raw_code}"

                    self.merge_and_save(
                        price_df=weekly_price,
                        indicator_df=weekly_indicator,
                        save_path=save_path,
                        symbol=raw_code,
                        end_date=end_date
                    )
                    return f"✅ {raw_code}"

                except Exception as e:
                    if attempt < max_retries - 1:
                        print(f"重试 {raw_code} - 原因: {str(e)[:100]}")
                        time.sleep(retry_delay)
                        continue
                    return f"❌ 处理失败: {str(e)[:100]} - {raw_code}"

            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"重试 {raw_code} - 原因: {str(e)[:100]}")
                    time.sleep(retry_delay)
                    continue
                return f"❌ 处理失败: {str(e)[:100]} - {raw_code}"

    def update_statistics(self, result):
        """更新统计信息"""
        with self.lock:
            if result.startswith("✅"):
                self.success_count += 1
            elif result.startswith("⏩"):
                self.skip_count += 1
            else:
                self.error_count += 1
                if "❌" in result or "⚠️" in result:
                    import re
                    stock_code = "未知"
                    reason = "未知"
                    code_match = re.search(r'\b\d{6}\b', result)
                    if code_match:
                        stock_code = code_match.group()
                    if ": " in result:
                        parts = result.split(": ", 1)
                        if len(parts) == 2:
                            if parts[1].strip().isdigit() and len(parts[1].strip()) == 6:
                                reason = parts[0].replace("⚠️ ", "").replace("❌ ", "").strip()
                            else:
                                error_msg = parts[1]
                                error_msg = re.sub(r'\s*-\s*\d{6}$', '', error_msg)
                                reason = error_msg.strip()
                    self.failed_stocks.append((stock_code, reason))

    def scan_and_export_signals(self, save_dir, output_path):
        """
        对所有股票周线数据进行筛选，输出符合条件的股票列表到 Excel
        """
        records = []
        latest_date = None

        # 用 ASharesList 的 5 列信息填充报告
        meta_df = self.stock_meta_df
        if meta_df is None or meta_df.empty:
            # 兜底：重新加载一次
            self.load_filtered_stock_list()
            meta_df = self.stock_meta_df

        # 列名约定：股票代码, 股票名称, 一二级行业, 细分行业, 地区
        if meta_df is not None and not meta_df.empty:
            meta_df = meta_df.copy()
            meta_df.columns = [str(c).strip().strip('﻿') for c in meta_df.columns]
            meta_df['__code'] = meta_df.iloc[:, 0].apply(self.pad_stock_code)
            meta_df_indexed = meta_df.set_index('__code')
        else:
            meta_df_indexed = pd.DataFrame()

        def lookup_meta(code):
            # 如果该股票不在过滤后的清单中（曾被标为 ST 或行业为房地产），
            # 返回 None，由调用方过滤掉，不进入 reports
            if meta_df_indexed.empty or code not in meta_df_indexed.index:
                return None
            row = meta_df_indexed.loc[code]
            return {
                '股票代码': code,
                '股票名称': row.iloc[1] if len(row) > 1 else '未知',
                '一二级行业': row.iloc[2] if len(row) > 2 else '未知',
                '细分行业': row.iloc[3] if len(row) > 3 else '未知',
                '地区': row.iloc[4] if len(row) > 4 else '未知',
            }

        # 收集所有需要扫描的 csv 文件
        all_files = [f for f in os.listdir(save_dir) if f.lower().endswith(".csv")]
        skipped_not_in_filter = 0
        for filename in tqdm(all_files, desc="🔍 筛选周线信号",
                             bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'):
            stock_code = os.path.splitext(filename)[0]
            file_path = os.path.join(save_dir, filename)

            try:
                df = pd.read_csv(file_path, encoding='utf_8_sig')
            except Exception:
                continue

            if df.empty:
                continue

            required_cols = {'date', 'Short_Trend', 'Short_LS', 'close'}
            if not required_cols.issubset(set(df.columns)):
                try:
                    df = df.sort_values('date')
                    last = df.iloc[-1]
                    cur_date = last['date']
                    if pd.notna(cur_date):
                        if latest_date is None or cur_date > latest_date:
                            latest_date = cur_date
                except Exception:
                    pass
                continue

            df = df.sort_values('date')
            last = df.iloc[-1]

            cur_date = last['date']
            if pd.notna(cur_date):
                if latest_date is None or cur_date > latest_date:
                    latest_date = cur_date

            if pd.isna(last['Short_Trend']) or pd.isna(last['Short_LS']) or pd.isna(last['close']):
                continue

            cond1 = last['Short_Trend'] > last['Short_LS']
            cond2 = last['close'] >= 0.98 * last['Short_LS']

            if cond1 and cond2:
                meta = lookup_meta(stock_code)
                if meta is None:
                    # 周线 csv 中残留的 ST / 房地产股票，不进入报告
                    skipped_not_in_filter += 1
                    continue
                records.append(meta)

        if skipped_not_in_filter > 0:
            print(f"   ⏭️  跳过 {skipped_not_in_filter} 只 ST/房地产残留股票（不在过滤清单中）")

        if records:
            result_df = pd.DataFrame(records, columns=['股票代码', '股票名称', '一二级行业', '细分行业', '地区'])
        else:
            result_df = pd.DataFrame(columns=['股票代码', '股票名称', '一二级行业', '细分行业', '地区'])

        # 加入日期后缀
        final_path = output_path
        if latest_date is not None:
            try:
                date_str = pd.to_datetime(latest_date).strftime("%Y%m%d")
                base, ext = os.path.splitext(output_path)
                final_path = f"{base}_{date_str}{ext}"
            except Exception:
                final_path = output_path

        result_df.to_excel(final_path, index=False)
        return final_path

    def main(self, max_stocks=None):
        """主函数"""
        # 周线数据保存目录
        save_dir = WEEKLY_SAVE_DIR
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(WEEKLY_REPORT_DIR, exist_ok=True)

        # 读取 ASharesList.xlsx，并按要求过滤（不含 ST、不含房地产）
        try:
            df = self.load_filtered_stock_list(ASHARES_LIST_PATH)
            code_col = df.columns[0]
            df[code_col] = df[code_col].apply(self.pad_stock_code)
            indicator_codes = df[code_col].dropna().unique().tolist()
            price_codes = list(indicator_codes)
            total = len(indicator_codes)

            # 预加载股票名称
            if len(df.columns) > 1:
                for _, row in df.iterrows():
                    code = row.iloc[0]
                    if pd.notna(row.iloc[1]):
                        self.stock_names[code] = str(row.iloc[1])
                    else:
                        self.stock_names[code] = "未知"

            # 若指定最大股票数（仅用于快速验证）
            if max_stocks is not None and total > max_stocks:
                indicator_codes = indicator_codes[:max_stocks]
                price_codes = price_codes[:max_stocks]
                total = len(indicator_codes)

        except Exception as e:
            print(f"读取Excel文件失败: {e}")
            return

        # 动态计算线程数
        optimal_threads = self.get_optimal_thread_count(total)
        physical_cores = self.get_physical_cpu_cores()
        logical_cores = os.cpu_count() or 1
        print(f"📊 过滤后股票数: {total}")
        print(f"🧵 使用线程数: {optimal_threads} (基于{physical_cores}核计算)")
        print(f"💻 CPU核心数: {physical_cores}核{logical_cores}线程")
        print(f"💾 可用内存: {psutil.virtual_memory().available // (1024 ** 3)} GB")

        # 基于过滤后的股票，随机抽样检测最新交易日
        end_date = datetime.now().strftime("%Y%m%d")
        lookback_days = 30
        start_candidate = (datetime.now() - relativedelta(days=lookback_days)).strftime("%Y%m%d")

        latest_dates = []
        sample_codes = random.sample(indicator_codes, min(10, len(indicator_codes)))
        for code in sample_codes:
            csv_path = os.path.join(DAILY_DATA_DIR, f"{code}.csv")
            if not os.path.exists(csv_path):
                continue
            try:
                tmp = pd.read_csv(csv_path, encoding='utf_8_sig', usecols=[0])
                tmp.columns = ['date']
                tmp['date'] = pd.to_datetime(tmp['date'], errors='coerce')
                tmp = tmp.dropna(subset=['date'])
                if not tmp.empty:
                    latest_dates.append(tmp['date'].max())
            except Exception:
                continue

        if latest_dates:
            date_series = pd.to_datetime(pd.Series(latest_dates)).dt.normalize()
            end_date = date_series.value_counts().idxmax().strftime("%Y%m%d")
        else:
            end_date = datetime.now().strftime("%Y%m%d")

        start_date = (datetime.strptime(end_date, "%Y%m%d") - relativedelta(years=3)).strftime("%Y%m%d")
        print(f"📅 最新交易日: {end_date}")
        print(f"📅 数据日期范围: {start_date} - {end_date}")

        # ============ 步骤 2: 校验 Daliy 数据最新日期 ============
        try:
            all_match, details = self._check_daily_range_match(start_date, end_date, sample_count=10)
            if not all_match:
                print("\n⚠️ 抽样检查 Daliy 数据最新日期与 end_date 不匹配，详细信息：")
                for d in details:
                    if not d['end_match']:
                        min_str = d['min_date'].strftime('%Y%m%d') if pd.notna(d['min_date']) else 'N/A'
                        max_str = d['max_date'].strftime('%Y%m%d') if pd.notna(d['max_date']) else 'N/A'
                        print(f"   {d['code']}: min={min_str}, max={max_str}, end_match={d['end_match']}")
                self._trigger_daily_update()
            else:
                print("✅ 抽样 10 只股票 Daliy 数据最新日期与 end_date 匹配，跳过 Daliy 更新")
        except Exception as e:
            print(f"⚠️ 抽样校验 Daliy 失败: {str(e)[:200]}")

        # ============ 步骤 3~5: 多线程计算周线 ============
        start_time = time.time()

        with tqdm(total=total, desc="🚀 多线程周线更新",
                  bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]') as pbar:
            with concurrent.futures.ThreadPoolExecutor(max_workers=optimal_threads) as executor:
                process_func = partial(
                    self.process_single_stock,
                    start_date=start_date,
                    end_date=end_date,
                    save_dir=save_dir
                )

                future_to_stock = {
                    executor.submit(process_func, raw_code): raw_code
                    for raw_code in indicator_codes
                }

                for future in concurrent.futures.as_completed(future_to_stock):
                    raw_code = future_to_stock[future]
                    try:
                        result = future.result()
                        self.update_statistics(result)
                        pbar.set_postfix({
                            '成功': self.success_count,
                            '跳过': self.skip_count,
                            '失败': self.error_count,
                            '当前': result[:20] + "..." if len(result) > 20 else result
                        })
                    except Exception as e:
                        self.update_statistics(f"❌ 异常: {raw_code}")
                        pbar.set_postfix_str(f"❌ 异常: {raw_code}")
                    finally:
                        pbar.update(1)

        end_time = time.time()
        duration = end_time - start_time

        print(f"\n🎉 周线数据更新完成!")
        print(f"⏱️  总耗时: {duration:.2f} 秒")
        print(f"📈 成功: {self.success_count} 只")
        print(f"⏩ 跳过: {self.skip_count} 只")
        print(f"❌ 失败: {self.error_count} 只")
        print(f"⚡ 平均速度: {total / duration:.2f} 只/秒")

        if self.error_count > 0:
            print(f"⚠️  失败率: {self.error_count / total * 100:.1f}%")
            print(f"\n📋 失败股票详情:")
            for stock_code, reason in self.failed_stocks:
                stock_name = self.get_stock_name(stock_code)
                print(f"   {stock_code}   {stock_name}  {reason}")

        # ============ 步骤 6: 输出报告到 01_Report ============
        try:
            base_path = os.path.join(WEEKLY_REPORT_DIR, "ASharesWeek_FilterResult.xlsx")
            result_path = self.scan_and_export_signals(save_dir, base_path)
            print(f"\n✅ 周线筛选结果已保存到: {result_path}")
        except Exception as e:
            print(f"\n❌ 周线筛选结果导出失败: {str(e)[:100]}")


def main():
    updater = StockDataUpdater(
        max_workers=None,
        retry_attempts=3,
        base_delay=0.3
    )
    updater.main()


if __name__ == "__main__":
    main()
