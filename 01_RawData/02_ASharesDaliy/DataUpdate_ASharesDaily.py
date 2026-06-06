import akshare as ak
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
from queue import Queue
import psutil

# 禁用 SSL 警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class StockDataUpdater:
    """优化的股票数据更新器"""

    def __init__(self, max_workers=None, retry_attempts=3, base_delay=0.5):
        """
        初始化数据更新器

        Args:
            max_workers: 最大线程数，None表示自动检测
            retry_attempts: 重试次数
            base_delay: 基础延迟时间
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
            # 截取前30个字符，避免显示过长
            return reason[:30] + "..." if len(reason) > 30 else reason

    def get_physical_cpu_cores(self):
        """获取物理CPU核心数（而不是逻辑处理器数）"""
        try:
            # 尝试获取物理核心数
            if hasattr(psutil, 'cpu_count'):
                physical_cores = psutil.cpu_count(logical=False)
                if physical_cores:
                    return physical_cores
            # 如果无法获取物理核心数，使用逻辑处理器数的一半（通常是超线程）
            logical_cores = os.cpu_count() or 1
            return max(1, logical_cores // 2)
        except:
            # 如果都失败，返回一个合理的默认值
            return 4

    def get_optimal_thread_count(self, total_stocks):
        """动态计算最优线程数"""
        physical_cores = self.get_physical_cpu_cores()
        logical_cores = os.cpu_count() or 1

        # 基于股票数量调整线程数，使用物理核心数作为基础
        if total_stocks < 50:
            return min(4, physical_cores)
        elif total_stocks < 200:
            return min(8, physical_cores * 2)
        else:
            return min(16, physical_cores * 3)  # 最多使用物理核心数的3倍

    def random_sleep(self, min_seconds=None, max_seconds=None):
        """智能随机延时"""
        if min_seconds is None:
            min_seconds = self.base_delay
        if max_seconds is None:
            max_seconds = self.base_delay * 2

        sleep_time = random.uniform(min_seconds, max_seconds)
        time.sleep(sleep_time)

    def process_price_code(self, code):
        """处理价格接口的股票代码格式"""
        code_str = str(code).strip()
        num_part = ''.join(filter(str.isdigit, code_str))
        if not num_part:
            return None

        first_digit = num_part[0]

        if first_digit == '6':
            return f'sh{num_part}'
        elif first_digit in ('0', '3'):
            return f'sz{num_part}'
        elif first_digit in ('8', '9'):
            return f'bj{num_part}'
        else:
            return None

    def get_date_range(self, years=3):
        """生成动态时间范围"""
        end_date = datetime.now().strftime("%Y%m%d")
        start_date = (datetime.now() - relativedelta(years=years)).strftime("%Y%m%d")
        return start_date, end_date

    def pad_stock_code(self, code):
        """补全股票代码至6位"""
        code_str = str(code).strip()
        return code_str.zfill(6)

    def filter_by_date(self, df, start_date, end_date):
        """通用日期过滤函数"""
        date_col = next((col for col in ['trade_date', 'date', '日期'] if col in df.columns), None)
        if not date_col:
            raise ValueError(f"未找到日期字段，可用列名: {df.columns.tolist()}")

        df['date'] = pd.to_datetime(df[date_col])
        start_dt = datetime.strptime(start_date, "%Y%m%d")
        end_dt = datetime.strptime(end_date, "%Y%m%d")
        return df[(df['date'] >= start_dt) & (df['date'] <= end_dt)]

    def detect_latest_trade_date(self, codes, min_samples=5, max_samples=10, lookback_days=30):
        """
        从给定股票代码中随机抽样，检测最新交易日（参考周线脚本实现）

        Args:
            codes: 可迭代的股票代码列表
            min_samples: 最少抽样数量（不超过股票总数）
            max_samples: 最多抽样数量
            lookback_days: 回看天数窗口
        """
        unique_codes = list({self.pad_stock_code(c) for c in codes if pd.notna(c)})
        if not unique_codes:
            return datetime.now().strftime("%Y%m%d")

        # 决定抽样数量：介于 min_samples 和 max_samples 之间，且不超过股票总数
        sample_count = min(len(unique_codes), max_samples)
        sample_count = max(sample_count, min(len(unique_codes), min_samples))

        if sample_count <= 0:
            return datetime.now().strftime("%Y%m%d")

        sample_codes = random.sample(unique_codes, sample_count)

        end_candidate = datetime.now().strftime("%Y%m%d")
        start_candidate = (datetime.now() - relativedelta(days=lookback_days)).strftime("%Y%m%d")

        latest_dates = []

        for code in sample_codes:
            price_symbol = self.process_price_code(code)
            if not price_symbol:
                continue

            try:
                df = self.fetch_data_with_retry(
                    ak.stock_zh_a_daily,
                    symbol=price_symbol,
                    adjust="qfq",
                    start_date=start_candidate,
                    end_date=end_candidate
                )
                if df is None or df.empty:
                    continue

                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                elif 'trade_date' in df.columns:
                    df['date'] = pd.to_datetime(df['trade_date'])
                else:
                    continue

                latest_dates.append(df['date'].max())
            except Exception:
                continue

        if not latest_dates:
            # 如果所有抽样股票都失败，则退回当前日期
            return datetime.now().strftime("%Y%m%d")

        # 使用出现次数最多的日期作为最新交易日，避免个别异常值影响结果
        date_series = pd.to_datetime(pd.Series(latest_dates)).dt.normalize()
        latest_trade_date = date_series.value_counts().idxmax()
        return latest_trade_date.strftime("%Y%m%d")

    def tdx_sma(self, series, n, m=1):
        """
        通达信SMA平滑算法实现
        SMA(X, N, M) = (M * X + (N - M) * Y_前一日) / N
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
        计算KDJ指标，与股票软件保持一致
        使用标准公式：K = (2/3) * K_前一日 + (1/3) * RSV
                     D = (2/3) * D_前一日 + (1/3) * K
                     J = 3 * K - 2 * D
        
        Args:
            df: 包含high, low, close列的DataFrame，必须按时间正序排列
            n: RSV计算周期，默认9
            m1: K值平滑周期，默认3（实际使用固定1/3平滑系数）
            m2: D值平滑周期，默认3（实际使用固定1/3平滑系数）
        """
        # 确保数据按时间正序排列并重置索引
        df_work = df.copy()
        if 'date' in df_work.columns:
            df_work = df_work.sort_values('date').reset_index(drop=True)
        elif 'trade_date' in df_work.columns:
            df_work = df_work.sort_values('trade_date').reset_index(drop=True)
        
        high = df_work['high'].values
        low = df_work['low'].values
        close = df_work['close'].values
        length = len(df_work)
        
        # 计算RSV（未成熟随机值）
        rsv = np.full(length, np.nan)
        for i in range(n - 1, length):
            period_high = high[i - n + 1:i + 1]
            period_low = low[i - n + 1:i + 1]
            highest = np.max(period_high)
            lowest = np.min(period_low)
            
            if highest != lowest:
                rsv[i] = 100 * (close[i] - lowest) / (highest - lowest)
            else:
                rsv[i] = 50  # 如果最高价等于最低价，RSV设为50
        
        # 计算K值和D值（使用EMA平滑，初始值为50）
        k = np.full(length, np.nan)
        d = np.full(length, np.nan)
        
        # 找到第一个有效的RSV值
        first_valid_idx = None
        for i in range(length):
            if not np.isnan(rsv[i]):
                first_valid_idx = i
                break
        
        if first_valid_idx is not None:
            # 初始化K和D值
            # 有些软件使用第一个RSV值作为初始K值，有些使用50
            # 这里使用第一个RSV值，更符合大多数股票软件的实现
            k[first_valid_idx] = rsv[first_valid_idx]
            d[first_valid_idx] = rsv[first_valid_idx]  # D的初始值也使用第一个RSV
            
            # 计算K值：K = (2/3) * K_前一日 + (1/3) * RSV
            # 标准KDJ公式，平滑系数固定为1/3
            alpha_k = 1.0 / 3.0
            for i in range(first_valid_idx + 1, length):
                if not np.isnan(rsv[i]):
                    k[i] = (1 - alpha_k) * k[i - 1] + alpha_k * rsv[i]
                else:
                    k[i] = k[i - 1]
            
            # 计算D值：D = (2/3) * D_前一日 + (1/3) * K
            # 标准KDJ公式，平滑系数固定为1/3
            alpha_d = 1.0 / 3.0
            for i in range(first_valid_idx + 1, length):
                if not np.isnan(k[i]):
                    d[i] = (1 - alpha_d) * d[i - 1] + alpha_d * k[i]
                else:
                    d[i] = d[i - 1]
        
        # 计算J值：J = 3 * K - 2 * D
        j = 3 * k - 2 * d
        
        return k, d, j

    def calculate_ta_indicators(self, df):
        """计算技术指标（与原版保持一致，并增加砖形图指标）"""
        # KDJ指标 - 使用自定义函数确保与股票软件一致
        df['K'], df['D'], df['J'] = self.calculate_kdj(df, n=9, m1=3, m2=3)

        # BBI指标
        periods = [3, 6, 12, 24]
        for p in periods:
            df[f'MA{p}'] = talib.SMA(df['close'], timeperiod=p)
        df['BBI'] = df[[f'MA{p}' for p in periods]].mean(axis=1)
        df['BBI_DIF'] = df['BBI'].diff().fillna(0)

        # MACD指标
        df['DIF'], df['DEA'], df['MACD'] = talib.MACD(
            df['close'],
            fastperiod=12,
            slowperiod=26,
            signalperiod=9
        )

        # 计算短期资金指标
        df['short_term_low'] = df['low'].rolling(window=3).min()
        df['short_term_high'] = df['close'].rolling(window=3).max()
        df['short_term_fund'] = 100 * (df['close'] - df['short_term_low']) / (
                    df['short_term_high'] - df['short_term_low'])

        # 计算长期资金指标
        df['long_term_low'] = df['low'].rolling(window=21).min()
        df['long_term_high'] = df['close'].rolling(window=21).max()
        df['long_term_fund'] = 100 * (df['close'] - df['long_term_low']) / (df['long_term_high'] - df['long_term_low'])

        # 短期多空线
        sma14 = talib.SMA(df['close'], timeperiod=14)
        sma28 = talib.SMA(df['close'], timeperiod=28)
        sma57 = talib.SMA(df['close'], timeperiod=57)
        sma114 = talib.SMA(df['close'], timeperiod=114)
        df['Short_LS'] = np.round((sma14 + sma28 + sma57 + sma114) / 4.0, 2)

        # 短期趋势
        ema10_first = talib.EMA(df['close'], timeperiod=10)
        df['Short_Trend'] = np.round(talib.EMA(ema10_first, timeperiod=10), 2)

        # 砖形图指标（基于通达信公式）
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

        # 这里直接用“当日砖形图值”和“昨日砖形图值”来编码颜色：
        # - 若 Brick_High > Brick_Low，则当日砖形图上穿昨日，为红柱
        # - 若 Brick_High < Brick_Low，则当日砖形图下穿昨日，为绿柱
        df['Brick_High'] = brick_series
        df['Brick_Low'] = prev_brick

        # PE通道计算
        if 'pe_ttm' in df.columns:
            df['pe_ttm'] = df['pe_ttm'].replace([np.inf, -np.inf], np.nan)
            df['pe_ttm'] = df['pe_ttm'].ffill().bfill()

            min_pe = df['pe_ttm'].min()
            median_pe = df['pe_ttm'].median()
            step = (median_pe - min_pe) / 2 if median_pe > min_pe else 0
            L2 = min_pe
            M = median_pe
            L1 = L2 + step
            H1 = M + step
            H2 = H1 + step

            df['investment_income'] = df['close'] / df['pe_ttm'].replace(0, np.nan).fillna(1)
            df['L2'] = L2
            df['L1'] = L1
            df['M'] = M
            df['H1'] = H1
            df['H2'] = H2

        # 删除临时列
        df = df.drop(columns=[f'MA{p}' for p in periods] +
                             ['short_term_low', 'short_term_high', 'long_term_low', 'long_term_high'])

        return df

    def merge_and_save(self, price_df, indicator_df, save_path, symbol):
        """重构版数据合并保存函数（含技术指标）"""
        try:
            # 计算技术指标
            price_df = self.calculate_ta_indicators(price_df.copy())

            # 添加股票代码标识
            price_df = price_df.assign(symbol=symbol)
            indicator_df = indicator_df.assign(symbol=symbol)

            # 日期处理
            price_df['date'] = pd.to_datetime(price_df['trade_date'] if 'trade_date' in price_df else price_df['date'])
            indicator_df['date'] = pd.to_datetime(indicator_df['trade_date'])

            # 合并数据
            merged_df = pd.merge(
                price_df,
                indicator_df,
                on=['date', 'symbol'],
                how='inner',
                suffixes=('_price', '_indicator'),
                validate='one_to_one'
            ).sort_values('date').reset_index(drop=True)

            # 去重操作
            merged_df = merged_df.drop_duplicates(subset=['date'], keep='last')

            # 数据清洗
            numeric_cols = merged_df.select_dtypes(include=np.number).columns.difference(['symbol']).tolist()
            numeric_cols = [col for col in numeric_cols if col not in ['date', 'symbol']]

            if numeric_cols:
                merged_df[numeric_cols] = merged_df.groupby('symbol', group_keys=False)[numeric_cols].apply(
                    lambda x: x.ffill().bfill()
                )

            # 存储阶段
            base_cols = ['date', 'symbol', 'open', 'high', 'low', 'close', 'volume', 'amount', 'outstanding_share',
                         'turnover']
            ta_cols = ['K', 'D', 'J', 'BBI', 'BBI_DIF', 'DIF', 'DEA', 'MACD',
                       'short_term_fund', 'long_term_fund',
                       'Short_LS', 'Short_Trend', 'Brick_High', 'Brick_Low']
            pe_cols = ['L2', 'L1', 'M', 'H1', 'H2', 'investment_income']
            value_cols = ['market_cap', 'float_market_cap', 'pe_ttm', 'pe_static', 'pb', 'peg', 'pcf', 'ps']

            all_columns = base_cols + ta_cols + pe_cols + value_cols
            output_columns = [col for col in all_columns if col in merged_df.columns]

            # 验证PE通道列
            missing_pe_cols = [col for col in pe_cols if col not in output_columns]
            if missing_pe_cols:
                merged_df['pe_ttm'] = merged_df['pe_ttm'].replace([np.inf, -np.inf, 0], np.nan).ffill().bfill()
                if 'investment_income' not in merged_df.columns:
                    merged_df['investment_income'] = merged_df['close'] / merged_df['pe_ttm'].replace(0, np.nan).fillna(
                        1)
                min_pe = merged_df['pe_ttm'].min()
                median_pe = merged_df['pe_ttm'].median()
                step = (median_pe - min_pe) / 2 if median_pe > min_pe else 0
                L2 = min_pe
                M = median_pe
                L1 = L2 + step
                H1 = M + step
                H2 = H1 + step
                merged_df['L2'] = L2 * merged_df['investment_income']
                merged_df['L1'] = L1 * merged_df['investment_income']
                merged_df['M'] = M * merged_df['investment_income']
                merged_df['H1'] = H1 * merged_df['investment_income']
                merged_df['H2'] = H2 * merged_df['investment_income']
                output_columns = [col for col in all_columns if col in merged_df.columns]

            # 保存数据
            merged_df[output_columns].to_csv(save_path, index=False, encoding='utf_8_sig')

        except Exception as e:
            raise Exception(f"合并保存失败: {str(e)[:100]}")

    def fetch_data_with_retry(self, func, *args, **kwargs):
        """带重试机制和超时控制的数据获取"""
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

    def process_single_stock(self, raw_code, price_code, start_date, end_date, save_dir):
        """优化的单股票处理函数"""
        max_retries = 3
        retry_delay = 5  # 重试延迟秒数

        for attempt in range(max_retries):
            try:
                # 股票代码转换
                price_symbol = self.process_price_code(price_code)
                if not price_symbol:
                    return f"❌ 无效代码: {price_code}"

                save_path = os.path.join(save_dir, f"{raw_code}.csv")

                # 文件存在性判断
                if os.path.exists(save_path):
                    try:
                        # 读取历史数据
                        history_df = pd.read_csv(save_path, encoding='utf_8_sig')
                        history_df = history_df.rename(columns={
                            '\ufeffdate': 'date',
                            'trade_date': 'date',
                            '日期': 'date'
                        })

                        if 'date' not in history_df.columns:
                            raise ValueError(f"日期列缺失，实际列名: {history_df.columns.tolist()}")

                        history_df['date'] = pd.to_datetime(history_df['date'])
                        latest_date = history_df['date'].max()
                        latest_date_str = latest_date.strftime("%Y%m%d")

                        if latest_date_str == end_date:
                            return f"⏩ 已是最新数据: {raw_code}"

                        # 删除早于start_date的数据
                        history_df = history_df[history_df['date'] >= pd.to_datetime(start_date)]
                        new_start = (latest_date + pd.Timedelta(days=1)).strftime("%Y%m%d")

                        # 获取增量数据
                        temp_price_df = self.fetch_data_with_retry(
                            ak.stock_zh_a_daily,
                            symbol=price_symbol,
                            adjust="qfq",
                            start_date=new_start,
                            end_date=end_date
                        )

                        if temp_price_df is None or temp_price_df.empty:
                            return f"⚠️ 无增量数据: {raw_code}"

                        temp_price_df = temp_price_df.rename(columns={'date': 'trade_date'})

                        temp_indicator_df = self.fetch_data_with_retry(
                            ak.stock_value_em,
                            symbol=raw_code
                        )

                        if temp_indicator_df is None or temp_indicator_df.empty:
                            return f"⚠️ 无增量指标: {raw_code}"

                        # 重命名指标数据列
                        temp_indicator_df = temp_indicator_df.rename(columns={
                            '数据日期': 'trade_date',
                            '总市值': 'market_cap',
                            '流通市值': 'float_market_cap',
                            'PE(TTM)': 'pe_ttm',
                            'PE(静)': 'pe_static',
                            '市净率': 'pb',
                            'PEG值': 'peg',
                            '市现率': 'pcf',
                            '市销率': 'ps'
                        })
                        temp_indicator_df = self.filter_by_date(temp_indicator_df, new_start, end_date)

                        # 合并历史数据和增量数据
                        combined_price = pd.concat([history_df, temp_price_df], ignore_index=True)
                        combined_price['date'] = pd.to_datetime(combined_price['date'])
                        combined_price = combined_price.drop_duplicates(subset=['date'], keep='last').sort_values(
                            'date')

                    except Exception as e:
                        if attempt < max_retries - 1:
                            print(f"重试 {raw_code} - 原因: {str(e)[:100]}")
                            time.sleep(retry_delay)
                            continue
                        return f"❌ 处理历史数据失败: {str(e)[:100]} - {raw_code}"

                else:
                    # 全量数据获取
                    try:
                        full_price_df = self.fetch_data_with_retry(
                            ak.stock_zh_a_daily,
                            symbol=price_symbol,
                            adjust="qfq",
                            start_date=start_date,
                            end_date=end_date
                        )

                        if full_price_df is None or full_price_df.empty:
                            return f"⚠️ 无价格数据: {raw_code}"

                        full_price_df = full_price_df.rename(columns={'date': 'trade_date'})

                        full_indicator_df = self.fetch_data_with_retry(
                            ak.stock_value_em,
                            symbol=raw_code
                        )

                        if full_indicator_df is None or full_indicator_df.empty:
                            return f"⚠️ 无指标数据: {raw_code}"

                        # 重命名指标数据列
                        full_indicator_df = full_indicator_df.rename(columns={
                            '数据日期': 'trade_date',
                            '总市值': 'market_cap',
                            '流通市值': 'float_market_cap',
                            'PE(TTM)': 'pe_ttm',
                            'PE(静)': 'pe_static',
                            '市净率': 'pb',
                            'PEG值': 'peg',
                            '市现率': 'pcf',
                            '市销率': 'ps'
                        })
                        full_indicator_df = self.filter_by_date(full_indicator_df, start_date, end_date)

                        combined_price = full_price_df

                    except Exception as e:
                        if attempt < max_retries - 1:
                            print(f"重试 {raw_code} - 原因: {str(e)[:100]}")
                            time.sleep(retry_delay)
                            continue
                        return f"❌ 获取全量数据失败: {str(e)[:100]} - {raw_code}"

                # 统一存储处理
                try:
                    indicator_df = temp_indicator_df if os.path.exists(save_path) else full_indicator_df
                    self.merge_and_save(
                        price_df=combined_price,
                        indicator_df=indicator_df,
                        save_path=save_path,
                        symbol=raw_code
                    )
                    return f"✅ {raw_code}"

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

    def update_statistics(self, result):
        """更新统计信息"""
        with self.lock:
            if result.startswith("✅"):
                self.success_count += 1
            elif result.startswith("⏩"):
                self.skip_count += 1
            else:
                self.error_count += 1
                # 记录失败的股票代码和原因
                if "❌" in result or "⚠️" in result:
                    stock_code = "未知"
                    reason = "未知"

                    # 使用正则表达式提取股票代码和失败原因
                    import re

                    # 提取股票代码（6位数字）
                    code_match = re.search(r'\b\d{6}\b', result)
                    if code_match:
                        stock_code = code_match.group()

                    # 提取失败原因
                    if ": " in result:
                        parts = result.split(": ", 1)
                        if len(parts) == 2:
                            # 如果第二部分是股票代码，那么第一部分是原因
                            if parts[1].strip().isdigit() and len(parts[1].strip()) == 6:
                                reason = parts[0].replace("⚠️ ", "").replace("❌ ", "").strip()
                            else:
                                # 如果第二部分包含股票代码，提取错误信息
                                error_msg = parts[1]
                                # 移除股票代码部分
                                error_msg = re.sub(r'\s*-\s*\d{6}$', '', error_msg)
                                reason = error_msg.strip()

                    self.failed_stocks.append((stock_code, reason))

    def main(self, max_stocks=None):
        """主函数

        Args:
            max_stocks: 若不为 None，则只处理前 max_stocks 只股票
        """
        # 路径配置
        excel_path = r"D:\Quant\01_SwProj\04_VectorBT\02_Lima\Lima_Gen1\01_RawData\02_ASharesDaliy\ASharesPro.xlsx"
        save_dir = r"D:\Quant\01_SwProj\04_VectorBT\02_Lima\Lima_Gen1\01_RawData\02_ASharesDaliy\StocksData"
        os.makedirs(save_dir, exist_ok=True)

        # 读取Excel文件
        try:
            df = pd.read_excel(excel_path, dtype={'股票代码': str})
            df.iloc[:, 0] = df.iloc[:, 0].apply(self.pad_stock_code)
            indicator_codes = df.iloc[:, 0].dropna().unique().tolist()
            price_codes = df.iloc[:, 0].dropna().unique().tolist()
            total = len(indicator_codes)

            # 若指定了最大股票数量，则只处理前 max_stocks 只股票（用于快速验证）
            if max_stocks is not None and total > max_stocks:
                indicator_codes = indicator_codes[:max_stocks]
                price_codes = price_codes[:max_stocks]
                total = len(indicator_codes)

            # 预加载股票名称到缓存
            for _, row in df.iterrows():
                stock_code = row.iloc[0]
                if len(row) > 1 and pd.notna(row.iloc[1]):
                    self.stock_names[stock_code] = str(row.iloc[1])
                else:
                    self.stock_names[stock_code] = "未知"

        except Exception as e:
            print(f"读取Excel文件失败: {e}")
            return

        # 动态计算线程数
        optimal_threads = self.get_optimal_thread_count(total)
        physical_cores = self.get_physical_cpu_cores()
        logical_cores = os.cpu_count() or 1
        print(f"📊 总股票数: {total}")
        print(f"🧵 使用线程数: {optimal_threads} (基于{physical_cores}核计算)")
        print(f"💻 CPU核心数: {physical_cores}核{logical_cores}线程")
        print(f"💾 可用内存: {psutil.virtual_memory().available // (1024 ** 3)} GB")

        # 基于随机多只股票检测最新交易日，并以此回溯 3 年
        latest_trade_date = self.detect_latest_trade_date(indicator_codes, min_samples=5, max_samples=10,
                                                          lookback_days=30)
        start_date = (datetime.strptime(latest_trade_date, "%Y%m%d") - relativedelta(years=3)).strftime("%Y%m%d")
        end_date = latest_trade_date
        print(f"📅 最新交易日: {latest_trade_date}")
        print(f"📅 数据日期范围: {start_date} - {end_date}")

        start_time = time.time()

        # 创建进度条
        with tqdm(total=total, desc="🚀 多线程数据更新",
                  bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]') as pbar:

            # 使用优化的线程池
            with concurrent.futures.ThreadPoolExecutor(max_workers=optimal_threads) as executor:
                # 创建偏函数
                process_func = partial(
                    self.process_single_stock,
                    start_date=start_date,
                    end_date=end_date,
                    save_dir=save_dir
                )

                # 提交所有任务
                future_to_stock = {
                    executor.submit(process_func, raw_code, price_code): (raw_code, price_code)
                    for raw_code, price_code in zip(indicator_codes, price_codes)
                }

                # 处理完成的任务
                for future in concurrent.futures.as_completed(future_to_stock):
                    raw_code, price_code = future_to_stock[future]
                    try:
                        result = future.result()
                        self.update_statistics(result)

                        # 更新进度条显示
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

        # 输出最终统计
        print(f"\n🎉 数据更新完成!")
        print(f"⏱️  总耗时: {duration:.2f} 秒")
        print(f"📈 成功: {self.success_count} 只")
        print(f"⏩ 跳过: {self.skip_count} 只")
        print(f"❌ 失败: {self.error_count} 只")
        print(f"⚡ 平均速度: {total / duration:.2f} 只/秒")

        if self.error_count > 0:
            print(f"⚠️  失败率: {self.error_count / total * 100:.1f}%")
            print(f"\n📋 失败股票详情:")

            # 直接显示每只失败股票，一行一个
            for stock_code, reason in self.failed_stocks:
                stock_name = self.get_stock_name(stock_code)
                # 格式化失败原因，添加更友好的描述
                formatted_reason = self.format_failure_reason(reason)
                print(f"   {stock_code}   {stock_name}  {formatted_reason}")


def main():
    """主入口函数"""
    # 创建优化版本的数据更新器
    updater = StockDataUpdater(
        max_workers=None,  # 自动检测
        retry_attempts=3,
        base_delay=0.3
    )

    # 处理 ASharesPro.xlsx 中的全部股票
    updater.main()


if __name__ == "__main__":
    main()
