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

# 禁用 SSL 警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def random_sleep(min_seconds=0.3, max_seconds=0.8):
    sleep_time = random.uniform(min_seconds, max_seconds)
    time.sleep(sleep_time)

def process_price_code(code):
    code_str = str(code).strip()
    num_part = ''.join(filter(str.isdigit, code_str))
    if not num_part:
        return None
    first_digit = num_part[0]
    if first_digit in ('6', '9'):
        return f'sh{num_part}'
    elif first_digit in ('0', '3', '2'):
        return f'sz{num_part}'
    else:
        return None

def pad_stock_code(code):
    code_str = str(code).strip()
    return code_str.zfill(6)

def filter_by_date(df, start_date, end_date):
    date_col = next((col for col in ['trade_date', 'date', '日期'] if col in df.columns), None)
    if not date_col:
        raise ValueError(f"未找到日期字段，可用列名: {df.columns.tolist()}")
    df['date'] = pd.to_datetime(df[date_col])
    start_dt = datetime.strptime(start_date, "%Y%m%d")
    end_dt = datetime.strptime(end_date, "%Y%m%d")
    return df[(df['date'] >= start_dt) & (df['date'] <= end_dt)]

def calculate_ta_indicators(df):
    df['K'], df['D'] = talib.STOCH(
        df['high'].values, df['low'].values, df['close'].values,
        fastk_period=9, slowk_period=3, slowk_matype=0,
        slowd_period=3, slowd_matype=0
    )
    df['J'] = 3 * df['K'] - 2 * df['D']
    periods = [3, 6, 12, 24]
    for p in periods:
        df[f'MA{p}'] = talib.SMA(df['close'], timeperiod=p)
    df['BBI'] = df[[f'MA{p}' for p in periods]].mean(axis=1)
    df['BBI_DIF'] = df['BBI'].diff().fillna(0)
    df['DIF'], df['DEA'], df['MACD'] = talib.MACD(
        df['close'], fastperiod=12, slowperiod=26, signalperiod=9
    )
    df['short_term_low'] = df['low'].rolling(window=3).min()
    df['short_term_high'] = df['close'].rolling(window=3).max()
    df['short_term_fund'] = 100 * (df['close'] - df['short_term_low']) / (df['short_term_high'] - df['short_term_low'])
    df['long_term_low'] = df['low'].rolling(window=21).min()
    df['long_term_high'] = df['close'].rolling(window=21).max()
    df['long_term_fund'] = 100 * (df['close'] - df['long_term_low']) / (df['long_term_high'] - df['long_term_low'])
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
    df = df.drop(columns=[f'MA{p}' for p in periods] + ['short_term_low', 'short_term_high', 'long_term_low', 'long_term_high'])
    return df

def merge_and_save(price_df, indicator_df, save_path, symbol):
    try:
        price_df = calculate_ta_indicators(price_df.copy())
        price_df = price_df.assign(symbol=symbol)
        indicator_df = indicator_df.assign(symbol=symbol)
        price_df['date'] = pd.to_datetime(price_df['trade_date'] if 'trade_date' in price_df else price_df['date'])
        indicator_df['date'] = pd.to_datetime(indicator_df['trade_date'])
        merged_df = pd.merge(
            price_df,
            indicator_df,
            on=['date', 'symbol'],
            how='inner',
            suffixes=('_price', '_indicator'),
            validate='one_to_one'
        ).sort_values('date').reset_index(drop=True)
        merged_df = merged_df.drop_duplicates(subset=['date'], keep='last')
        numeric_cols = merged_df.select_dtypes(include=np.number).columns.difference(['symbol']).tolist()
        numeric_cols = [col for col in numeric_cols if col not in ['date', 'symbol']]
        if numeric_cols:
            merged_df[numeric_cols] = merged_df.groupby('symbol', group_keys=False)[numeric_cols].apply(
                lambda x: x.ffill().bfill()
            )
        base_cols = ['date', 'symbol', 'open', 'high', 'low', 'close', 'volume', 'amount', 'outstanding_share',
                     'turnover']
        ta_cols = ['K', 'D', 'J', 'BBI', 'BBI_DIF', 'DIF', 'DEA', 'MACD', 'short_term_fund', 'long_term_fund']
        pe_cols = ['L2', 'L1', 'M', 'H1', 'H2', 'investment_income']
        value_cols = ['market_cap', 'float_market_cap', 'pe_ttm', 'pe_static', 'pb', 'peg', 'pcf', 'ps']
        all_columns = base_cols + ta_cols + pe_cols + value_cols
        output_columns = [col for col in all_columns if col in merged_df.columns]
        missing_pe_cols = [col for col in pe_cols if col not in output_columns]
        if missing_pe_cols:
            merged_df['pe_ttm'] = merged_df['pe_ttm'].replace([np.inf, -np.inf, 0], np.nan).ffill().bfill()
            if 'investment_income' not in merged_df.columns:
                merged_df['investment_income'] = merged_df['close'] / merged_df['pe_ttm'].replace(0, np.nan).fillna(1)
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
        merged_df[output_columns].to_csv(save_path, index=False, encoding='utf_8_sig')
    except Exception as e:
        print(f"❌ {symbol} 合并失败: {str(e)[:100]}...")
        raise

def get_custom_date_range():
    # 获取所有交易日
    trade_days = ak.tool_trade_date_hist_sina()
    trade_days = pd.to_datetime(trade_days['trade_date'])
    
    # 使用当前运行代码的日期作为结束日期
    end_date = datetime.now()
    
    # 计算5年前的日期
    five_years_ago = end_date - relativedelta(years=5)
    
    # 找到5年前最近的交易日索引
    five_years_ago_idx = trade_days[trade_days >= five_years_ago].index[0]
    
    # 从5年前再往前数40个交易日（注意：往前数是索引减小）
    start_idx = five_years_ago_idx - 40
    
    # 确保索引不越界
    if start_idx < 0:
        start_idx = 0
    
    start_date = trade_days.iloc[start_idx].strftime('%Y%m%d')
    end_date = end_date.strftime('%Y%m%d')
    
    print(f"📅 数据日期范围: {start_date} 至 {end_date}")
    print(f"📊 预计交易日数: {len(trade_days) - start_idx}")
    print(f"🕐 5年前日期: {five_years_ago.strftime('%Y-%m-%d')}")
    print(f"🎯 5年前第40个交易日: {trade_days.iloc[start_idx].strftime('%Y-%m-%d')}")
    
    return start_date, end_date

def process_single_stock(raw_code, price_code, start_date, end_date, save_dir):
    max_retries = 3
    retry_delay = 5
    for attempt in range(max_retries):
        try:
            price_symbol = process_price_code(price_code)
            if not price_symbol:
                return f"❌ 无效代码: {price_code}"
            save_path = os.path.join(save_dir, f"{raw_code}.csv")
            if os.path.exists(save_path):
                try:
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
                    history_df = history_df[history_df['date'] >= pd.to_datetime(start_date)]
                    new_start = (latest_date + pd.Timedelta(days=1)).strftime("%Y%m%d")
                    temp_price_df = None
                    for _ in range(3):
                        try:
                            random_sleep(1, 2)
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
                            random_sleep(2, 3)
                    if temp_price_df is None or temp_price_df.empty:
                        return f"⚠️ 无增量数据: {raw_code}"
                    temp_price_df = temp_price_df.rename(columns={'date': 'trade_date'})
                    temp_indicator_df = None
                    for _ in range(3):
                        try:
                            random_sleep(2, 4)
                            temp_indicator_df = ak.stock_value_em(symbol=raw_code)
                            if temp_indicator_df is not None and not temp_indicator_df.empty:
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
                            random_sleep(3, 5)
                    if temp_indicator_df is None or temp_indicator_df.empty:
                        return f"⚠️ 无增量指标: {raw_code}"
                    combined_price = pd.concat([
                        history_df,
                        temp_price_df
                    ], ignore_index=True)
                    combined_price['date'] = pd.to_datetime(combined_price['date'])
                    combined_price = combined_price.drop_duplicates(subset=['date'], keep='last').sort_values('date')
                except Exception as e:
                    if attempt < max_retries - 1:
                        print(f"重试 {raw_code} - 原因: {str(e)[:100]}")
                        time.sleep(retry_delay)
                        continue
                    return f"❌ 处理历史数据失败: {str(e)[:100]}..."
            else:
                try:
                    full_price_df = None
                    for _ in range(3):
                        try:
                            random_sleep(0.3, 0.8)
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
                            random_sleep(0.3, 0.8)
                    if full_price_df is None or full_price_df.empty:
                        return f"⚠️ 无价格数据: {raw_code}"
                    full_price_df = full_price_df.rename(columns={'date': 'trade_date'})
                    full_indicator_df = None
                    for _ in range(3):
                        try:
                            random_sleep(0.3, 0.8)
                            full_indicator_df = ak.stock_value_em(symbol=raw_code)
                            if full_indicator_df is not None and not full_indicator_df.empty:
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
                            random_sleep(3, 5)
                    if full_indicator_df is None or full_indicator_df.empty:
                        return f"⚠️ 无指标数据: {raw_code}"
                    combined_price = full_price_df
                except Exception as e:
                    if attempt < max_retries - 1:
                        print(f"重试 {raw_code} - 原因: {str(e)[:100]}")
                        time.sleep(retry_delay)
                        continue
                    return f"❌ 获取全量数据失败: {str(e)[:100]}..."
            try:
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
    excel_path = r"D:\Quant\01_SwProj\04_VectorBT\02_Lima\Lima_Gen1\01_RawData\01_ASharesLite\AsharesLite.xlsx"
    save_dir = r"D:\Quant\01_SwProj\04_VectorBT\02_Lima\Lima_Gen1\01_RawData\01_ASharesLite\StocksData"
    os.makedirs(save_dir, exist_ok=True)
    try:
        df = pd.read_excel(excel_path, dtype={'代码': str})
        df['代码'] = df['代码'].apply(pad_stock_code)
        indicator_codes = df['代码'].dropna().unique().tolist()
        price_codes = df['代码'].dropna().unique().tolist()
        total = len(indicator_codes)
    except Exception as e:
        print(f"读取Excel文件失败: {e}")
        return
    start_date, end_date = get_custom_date_range()
    with tqdm(total=total, desc="🔄 数据合并进度") as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            process_func = partial(
                process_single_stock,
                start_date=start_date,
                end_date=end_date,
                save_dir=save_dir
            )
            future_to_stock = {
                executor.submit(process_func, raw_code, price_code): (raw_code, price_code)
                for raw_code, price_code in zip(indicator_codes, price_codes)
            }
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