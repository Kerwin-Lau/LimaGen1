import akshare as ak
import pandas as pd
import numpy as np  # 关键修复点[3](@ref)
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

# 禁用 SSL 警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def random_sleep(min_seconds=0.3, max_seconds=0.8):
    """随机延时函数"""
    sleep_time = random.uniform(min_seconds, max_seconds)
    time.sleep(sleep_time)


def process_price_code(code):
    """处理价格接口的股票代码格式"""
    code_str = str(code).strip()
    num_part = ''.join(filter(str.isdigit, code_str))
    if not num_part:
        print(f"❌ 无效代码格式: {code}")
        return None
    
    first_digit = num_part[0]
    
    # 按照新规则处理股票代码前缀
    if first_digit == '6':
        return f'sh{num_part}'
    elif first_digit in ('0', '3'):
        return f'sz{num_part}'
    elif first_digit in ('8', '9'):
        return f'bj{num_part}'
    else:
        print(f"❌ 不支持的股票代码开头: {code} (开头数字: {first_digit})")
        return None


def get_date_range(years=5):
    """生成动态时间范围"""
    end_date = datetime.now().strftime("%Y%m%d")
    start_date = (datetime.now() - relativedelta(years=years)).strftime("%Y%m%d")
    return start_date, end_date


def pad_stock_code(code):
    """补全股票代码至6位"""
    code_str = str(code).strip()
    return code_str.zfill(6)


def filter_by_date(df, start_date, end_date):
    """通用日期过滤函数（兼容不同接口）"""
    date_col = next((col for col in ['trade_date', 'date', '日期'] if col in df.columns), None)
    if not date_col:
        raise ValueError(f"未找到日期字段，可用列名: {df.columns.tolist()}")

    df['date'] = pd.to_datetime(df[date_col])
    start_dt = datetime.strptime(start_date, "%Y%m%d")
    end_dt = datetime.strptime(end_date, "%Y%m%d")
    return df[(df['date'] >= start_dt) & (df['date'] <= end_dt)]


def calculate_ta_indicators(df):
    """计算技术指标（KDJ+BBI+MACD+资金指标+PE通道）"""
    # KDJ指标
    df['K'], df['D'] = talib.STOCH(
        df['high'].values, df['low'].values, df['close'].values,
        fastk_period=9, slowk_period=3, slowk_matype=0,
        slowd_period=3, slowd_matype=0
    )
    df['J'] = 3 * df['K'] - 2 * df['D']

    # BBI指标
    periods = [3, 6, 12, 24]
    for p in periods:
        df[f'MA{p}'] = talib.SMA(df['close'], timeperiod=p)
    df['BBI'] = df[[f'MA{p}' for p in periods]].mean(axis=1)
    df['BBI_DIF'] = df['BBI'].diff().fillna(0)

    # MACD指标（参数与国内软件对齐）
    df['DIF'], df['DEA'], df['MACD'] = talib.MACD(
        df['close'],
        fastperiod=12,
        slowperiod=26,
        signalperiod=9
    )

    # 计算短期资金指标
    df['short_term_low'] = df['low'].rolling(window=3).min()
    df['short_term_high'] = df['close'].rolling(window=3).max()
    df['short_term_fund'] = 100 * (df['close'] - df['short_term_low']) / (df['short_term_high'] - df['short_term_low'])

    # 计算长期资金指标
    df['long_term_low'] = df['low'].rolling(window=21).min()
    df['long_term_high'] = df['close'].rolling(window=21).max()
    df['long_term_fund'] = 100 * (df['close'] - df['long_term_low']) / (df['long_term_high'] - df['long_term_low'])

    # 短期多空线 Short_LS = (SMA(C,14)+SMA(C,28)+SMA(C,57)+SMA(C,114))/4，保留两位小数
    sma14 = talib.SMA(df['close'], timeperiod=14)
    sma28 = talib.SMA(df['close'], timeperiod=28)
    sma57 = talib.SMA(df['close'], timeperiod=57)
    sma114 = talib.SMA(df['close'], timeperiod=114)
    df['Short_LS'] = np.round((sma14 + sma28 + sma57 + sma114) / 4.0, 2)

    # 短期趋势 Short_Trend = EMA(EMA(C,10),10)，保留两位小数
    ema10_first = talib.EMA(df['close'], timeperiod=10)
    df['Short_Trend'] = np.round(talib.EMA(ema10_first, timeperiod=10), 2)

    # PE通道计算
    if 'pe_ttm' in df.columns:
        # 确保pe_ttm列没有无效值
        df['pe_ttm'] = df['pe_ttm'].replace([np.inf, -np.inf], np.nan)
        df['pe_ttm'] = df['pe_ttm'].ffill().bfill()

        # 新PE通道计算方法
        min_pe = df['pe_ttm'].min()  # L2
        median_pe = df['pe_ttm'].median()  # M
        step = (median_pe - min_pe) / 2 if median_pe > min_pe else 0
        L2 = min_pe
        M = median_pe
        L1 = L2 + step
        H1 = M + step
        H2 = H1 + step

        # 计算投资收益率
        df['investment_income'] = df['close'] / df['pe_ttm'].replace(0, np.nan).fillna(1)

        # 赋值PE通道
        df['L2'] = L2
        df['L1'] = L1
        df['M'] = M
        df['H1'] = H1
        df['H2'] = H2

    # 删除临时列
    df = df.drop(columns=[f'MA{p}' for p in periods] +
                         ['short_term_low', 'short_term_high', 'long_term_low', 'long_term_high'])

    return df


def merge_and_save(price_df, indicator_df, save_path, symbol):
    """重构版数据合并保存函数（含技术指标）"""
    try:
        # ========== 预处理阶段 ==========
        # 计算技术指标
        price_df = calculate_ta_indicators(price_df.copy())

        # 添加股票代码标识
        price_df = price_df.assign(symbol=symbol)
        indicator_df = indicator_df.assign(symbol=symbol)

        # ========== 日期处理优化 ==========
        price_df['date'] = pd.to_datetime(price_df['trade_date'] if 'trade_date' in price_df else price_df['date'])
        indicator_df['date'] = pd.to_datetime(indicator_df['trade_date'])

        # ========== 合并阶段 ==========
        merged_df = pd.merge(
            price_df,
            indicator_df,
            on=['date', 'symbol'],
            how='inner',
            suffixes=('_price', '_indicator'),
            validate='one_to_one'
        ).sort_values('date').reset_index(drop=True)

        # 新增去重操作
        merged_df = merged_df.drop_duplicates(subset=['date'], keep='last')

        # ========== 数据清洗 ==========
        numeric_cols = merged_df.select_dtypes(include=np.number).columns.difference(['symbol']).tolist()
        numeric_cols = [col for col in numeric_cols if col not in ['date', 'symbol']]

        if numeric_cols:
            merged_df[numeric_cols] = merged_df.groupby('symbol', group_keys=False)[numeric_cols].apply(
                lambda x: x.ffill().bfill()
            )

        # ========== 存储阶段 ==========
        # 基础列
        base_cols = ['date', 'symbol', 'open', 'high', 'low', 'close', 'volume', 'amount', 'outstanding_share',
                     'turnover']

        # 技术指标列（按要求在 long_term_fund 后加入 Short_LS 和 Short_Trend）
        ta_cols = ['K', 'D', 'J', 'BBI', 'BBI_DIF', 'DIF', 'DEA', 'MACD', 'short_term_fund', 'long_term_fund', 'Short_LS', 'Short_Trend']

        # PE通道列
        pe_cols = ['L2', 'L1', 'M', 'H1', 'H2', 'investment_income']

        # 估值指标列
        value_cols = ['market_cap', 'float_market_cap', 'pe_ttm', 'pe_static', 'pb', 'peg', 'pcf', 'ps']

        # 确保所有列都存在
        all_columns = base_cols + ta_cols + pe_cols + value_cols
        output_columns = [col for col in all_columns if col in merged_df.columns]

        # 验证PE通道列是否存在
        missing_pe_cols = [col for col in pe_cols if col not in output_columns]
        if missing_pe_cols:
            # 重新计算PE通道，采用与calculate_ta_indicators一致的新方法
            merged_df['pe_ttm'] = merged_df['pe_ttm'].replace([np.inf, -np.inf, 0], np.nan).ffill().bfill()
            # 若investment_income不存在，先补充计算
            if 'investment_income' not in merged_df.columns:
                merged_df['investment_income'] = merged_df['close'] / merged_df['pe_ttm'].replace(0, np.nan).fillna(1)
            min_pe = merged_df['pe_ttm'].min()  # L2
            median_pe = merged_df['pe_ttm'].median()  # M
            step = (median_pe - min_pe) / 2 if median_pe > min_pe else 0
            L2 = min_pe
            M = median_pe
            L1 = L2 + step
            H1 = M + step
            H2 = H1 + step
            # 赋值为常数乘以对应行的investment_income
            merged_df['L2'] = L2 * merged_df['investment_income']
            merged_df['L1'] = L1 * merged_df['investment_income']
            merged_df['M'] = M * merged_df['investment_income']
            merged_df['H1'] = H1 * merged_df['investment_income']
            merged_df['H2'] = H2 * merged_df['investment_income']
            # 更新输出列
            output_columns = [col for col in all_columns if col in merged_df.columns]

        # 保存数据
        merged_df[output_columns].to_csv(save_path, index=False, encoding='utf_8_sig')

    except Exception as e:
        print(f"❌ {symbol} 合并失败: {str(e)[:100]}...")
        raise


def process_single_stock(raw_code, price_code, start_date, end_date, save_dir):
    """处理单个股票的数据获取和保存"""
    max_retries = 3
    retry_delay = 5  # 重试延迟秒数

    for attempt in range(max_retries):
        try:
            # 股票代码转换
            price_symbol = process_price_code(price_code)
            if not price_symbol:
                return f"❌ 无效代码: {price_code}"

            save_path = os.path.join(save_dir, f"{raw_code}.csv")

            # ================== 文件存在性判断 ==================
            if os.path.exists(save_path):
                try:
                    # 读取历史数据
                    history_df = pd.read_csv(save_path, encoding='utf_8_sig')

                    # 统一列名处理
                    history_df = history_df.rename(columns={
                        '\ufeffdate': 'date',
                        'trade_date': 'date',
                        '日期': 'date'
                    })

                    # 确保日期列存在并转换为datetime
                    if 'date' not in history_df.columns:
                        raise ValueError(f"日期列缺失，实际列名: {history_df.columns.tolist()}")

                    history_df['date'] = pd.to_datetime(history_df['date'])

                    # 获取最新日期
                    latest_date = history_df['date'].max()
                    latest_date_str = latest_date.strftime("%Y%m%d")

                    # 判断是否已是最新数据
                    if latest_date_str == end_date:
                        return f"⏩ 已是最新数据: {raw_code}"

                    # 删除早于start_date的数据
                    history_df = history_df[history_df['date'] >= pd.to_datetime(start_date)]

                    # 计算增量起始日期
                    new_start = (latest_date + pd.Timedelta(days=1)).strftime("%Y%m%d")

                    # 获取增量价格数据
                    temp_price_df = None
                    for _ in range(3):  # 最多重试3次
                        try:
                            random_sleep(1, 2)  # 添加随机延时
                            temp_price_df = ak.stock_zh_a_daily(
                                symbol=price_symbol,
                                adjust="qfq",
                                start_date=new_start,
                                end_date=end_date
                            )
                            if temp_price_df is not None and not temp_price_df.empty:
                                break
                        except Exception as e:
                            print(f"获取价格数据失败 {raw_code}: {str(e)[:100]}")
                            random_sleep(2, 3)  # 失败后增加延时

                    if temp_price_df is None or temp_price_df.empty:
                        return f"⚠️ 无增量数据: {raw_code}"

                    # 统一价格数据列名
                    temp_price_df = temp_price_df.rename(columns={'date': 'trade_date'})

                    # 获取增量指标数据
                    temp_indicator_df = None
                    for _ in range(3):  # 最多重试3次
                        try:
                            random_sleep(2, 4)  # 指标数据获取增加更长的延时
                            temp_indicator_df = ak.stock_value_em(symbol=raw_code)
                            if temp_indicator_df is not None and not temp_indicator_df.empty:
                                # 重命名列以匹配之前的格式
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
                                temp_indicator_df = filter_by_date(temp_indicator_df, new_start, end_date)
                                break
                        except Exception as e:
                            print(f"获取指标数据失败 {raw_code}: {str(e)[:100]}")
                            random_sleep(3, 5)  # 失败后增加更长的延时

                    if temp_indicator_df is None or temp_indicator_df.empty:
                        return f"⚠️ 无增量指标: {raw_code}"

                    # 合并历史数据和增量数据
                    combined_price = pd.concat([
                        history_df,
                        temp_price_df
                    ], ignore_index=True)

                    # 日期去重和排序
                    combined_price['date'] = pd.to_datetime(combined_price['date'])
                    combined_price = combined_price.drop_duplicates(subset=['date'], keep='last').sort_values('date')

                except Exception as e:
                    if attempt < max_retries - 1:
                        print(f"重试 {raw_code} - 原因: {str(e)[:100]}")
                        time.sleep(retry_delay)
                        continue
                    return f"❌ 处理历史数据失败: {str(e)[:100]}..."

            else:
                # ================== 全量数据获取 ==================
                try:
                    # 获取完整价格数据
                    full_price_df = None
                    for _ in range(3):  # 最多重试3次
                        try:
                            random_sleep(0.3, 0.8)  # 添加随机延时
                            full_price_df = ak.stock_zh_a_daily(
                                symbol=price_symbol,
                                adjust="qfq",
                                start_date=start_date,
                                end_date=end_date
                            )
                            if full_price_df is not None and not full_price_df.empty:
                                break
                        except Exception as e:
                            print(f"获取价格数据失败 {raw_code}: {str(e)[:100]}")
                            random_sleep(0.3, 0.8)  # 失败后增加延时

                    if full_price_df is None or full_price_df.empty:
                        return f"⚠️ 无价格数据: {raw_code}"

                    # 统一价格数据列名
                    full_price_df = full_price_df.rename(columns={'date': 'trade_date'})

                    # 获取完整指标数据
                    full_indicator_df = None
                    for _ in range(3):  # 最多重试3次
                        try:
                            random_sleep(0.3, 0.8)  # 指标数据获取增加更长的延时
                            full_indicator_df = ak.stock_value_em(symbol=raw_code)
                            if full_indicator_df is not None and not full_indicator_df.empty:
                                # 重命名列以匹配之前的格式
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
                                full_indicator_df = filter_by_date(full_indicator_df, start_date, end_date)
                                break
                        except Exception as e:
                            print(f"获取指标数据失败 {raw_code}: {str(e)[:100]}")
                            random_sleep(3, 5)  # 失败后增加更长的延时

                    if full_indicator_df is None or full_indicator_df.empty:
                        return f"⚠️ 无指标数据: {raw_code}"

                    combined_price = full_price_df

                except Exception as e:
                    if attempt < max_retries - 1:
                        print(f"重试 {raw_code} - 原因: {str(e)[:100]}")
                        time.sleep(retry_delay)
                        continue
                    return f"❌ 获取全量数据失败: {str(e)[:100]}..."

            # ================== 统一存储处理 ==================
            try:
                # 最终数据合并保存
                merge_and_save(
                    price_df=combined_price,
                    indicator_df=temp_indicator_df if os.path.exists(save_path) else full_indicator_df,
                    save_path=save_path,
                    symbol=raw_code
                )
                return f"✅ {raw_code}"
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"重试 {raw_code} - 原因: {str(e)[:100]}")
                    time.sleep(retry_delay)
                    continue
                return f"❌ 保存数据失败: {str(e)[:100]}..."

        except Exception as e:
            if attempt < max_retries - 1:
                print(f"重试 {raw_code} - 原因: {str(e)[:100]}")
                time.sleep(retry_delay)
                continue
            return f"❌ 处理失败: {str(e)[:100]}..."


def main():
    # 路径配置
    excel_path = r"D:\Quant\01_SwProj\04_VectorBT\02_Lima\Lima_Gen1\01_RawData\02_ASharesPro\ASharesPro.xlsx"
    save_dir = r"D:\Quant\01_SwProj\04_VectorBT\02_Lima\Lima_Gen1\01_RawData\02_ASharesPro\StocksData"
    os.makedirs(save_dir, exist_ok=True)

    # 读取Excel文件
    try:
        # 读取Excel文件，跳过表头
        df = pd.read_excel(excel_path, dtype={'股票代码': str})  # 明确指定股票代码列为字符串类型
        # 处理股票代码
        df.iloc[:, 0] = df.iloc[:, 0].apply(pad_stock_code)
        indicator_codes = df.iloc[:, 0].dropna().unique().tolist()
        # 使用第一列作为价格代码（因为第二列现在是股票名称）
        price_codes = df.iloc[:, 0].dropna().unique().tolist()
        total = len(indicator_codes)
    except Exception as e:
        print(f"读取Excel文件失败: {e}")
        return

    # 获取日期范围
    start_date, end_date = get_date_range()

    # 创建进度条
    with tqdm(total=total, desc="🔄 数据合并进度") as pbar:
        # 使用线程池进行并行处理，减少并发数以避免服务器限制
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            # 创建偏函数，固定部分参数
            process_func = partial(
                process_single_stock,
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
                    pbar.set_postfix_str(result)
                except Exception as e:
                    pbar.set_postfix_str(f"❌ 失败: {raw_code}")
                finally:
                    pbar.update(1)


if __name__ == "__main__":
    main()