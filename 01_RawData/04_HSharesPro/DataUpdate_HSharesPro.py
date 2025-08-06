import akshare as ak
import pandas as pd
import numpy as np
import os
import talib
import time
import traceback
import random
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from tqdm import tqdm
import concurrent.futures
from functools import partial
import urllib3

# 禁用 SSL 警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


# 改进1：增强代码有效性验证
def validate_hk_code(code):
    """验证港股代码有效性"""
    return len(code) == 5 and code.isdigit()


# 改进2：安全获取数据函数（含重试机制和接口切换）
def safe_get_hk_data(symbol, start_date, end_date, spot_data=None, max_retries=3):
    """带重试机制和接口切换的数据获取"""
    
    def try_stock_hk_hist():
        try:
            time.sleep(random.uniform(1, 3))
            df = ak.stock_hk_hist(
                symbol=symbol,
                period="daily",
                start_date=start_date,
                end_date=end_date,
                adjust="qfq"
            )
            if df.empty:
                return pd.DataFrame()
            required_cols = {'日期', '开盘', '最高', '最低', '收盘', '成交量'}
            if not required_cols.issubset(df.columns):
                return pd.DataFrame()
            df = df[['日期', '开盘', '最高', '最低', '收盘', '成交量']].copy()
            return df.rename(columns={
                '日期': 'date',
                '开盘': 'open',
                '最高': 'high',
                '最低': 'low',
                '收盘': 'close',
                '成交量': 'volume'
            })
        except Exception as e:
            print(f"⚠️ stock_hk_hist接口失败: {symbol}, 错误: {str(e)[:50]}")
            return pd.DataFrame()

    def try_stock_hk_daily():
        try:
            time.sleep(random.uniform(1, 3))
            df = ak.stock_hk_daily(
                symbol=symbol,
                adjust="qfq"
            )
            if df.empty:
                return pd.DataFrame()
            required_cols = {'date', 'open', 'high', 'low', 'close', 'volume'}
            if not required_cols.issubset(df.columns):
                return pd.DataFrame()
            df = df[['date', 'open', 'high', 'low', 'close', 'volume']].copy()
            # 日期筛选
            df['date'] = pd.to_datetime(df['date'])
            start = pd.to_datetime(start_date)
            end = pd.to_datetime(end_date)
            df = df[(df['date'] >= start) & (df['date'] <= end)]
            # 检查是否需要补充今日数据
            if not df.empty and spot_data is not None:
                max_daily_date = df['date'].max()
                if (end - max_daily_date).days == 1:
                    # 从全局spot_data中查找对应股票数据
                    try:
                        symbol_str = str(symbol).zfill(5)
                        spot_row = spot_data[spot_data['股票代码'] == symbol_str]
                        if not spot_row.empty:
                            spot_row = spot_row.iloc[0]
                            spot_dict = {
                                'date': pd.to_datetime(spot_row['日期时间']).strftime('%Y-%m-%d'),
                                'open': spot_row['今开'],
                                'high': spot_row['最高'],
                                'low': spot_row['最低'],
                                'close': spot_row['最新价'],
                                'volume': spot_row['成交量']
                            }
                            # 只在spot日期等于end_date时才补充
                            if spot_dict['date'] == end.strftime('%Y-%m-%d'):
                                spot_df = pd.DataFrame([spot_dict])
                                df = pd.concat([df, spot_df], ignore_index=True)
                    except Exception as e:
                        print(f"⚠️ 从spot_data查找数据失败: {symbol}, 错误: {str(e)[:50]}")
            # 恢复date为字符串格式，保持和hist一致
            df['date'] = df['date'].dt.strftime('%Y-%m-%d')
            return df.reset_index(drop=True)
        except Exception as e:
            print(f"⚠️ stock_hk_daily接口失败: {symbol}, 错误: {str(e)[:50]}")
            return pd.DataFrame()

    # 接口切换逻辑
    df = try_stock_hk_hist()
    if not df.empty:
        return df
    df = try_stock_hk_daily()
    if not df.empty:
        return df
    df = try_stock_hk_hist()
    if not df.empty:
        return df
    df = try_stock_hk_daily()
    if not df.empty:
        return df
    print(f"❌ 所有接口都无法获取数据: {symbol}")
    return pd.DataFrame()


def pad_stock_code(code):
    """补全港股代码至5位"""
    code_str = str(code).strip().zfill(5)
    return code_str


def get_date_range(years=5):
    """生成动态时间范围，包含周末日期判断"""
    today = datetime.now()
    
    # 周末日期判断逻辑
    if today.weekday() == 5:  # 周六
        end_date = (today - timedelta(days=1)).strftime("%Y%m%d")
    elif today.weekday() == 6:  # 周日
        end_date = (today - timedelta(days=2)).strftime("%Y%m%d")
    else:
        end_date = today.strftime("%Y%m%d")
    
    start_date = (today - relativedelta(years=years)).strftime("%Y%m%d")
    return start_date, end_date


def get_spot_data():
    """获取全市场实时行情数据"""
    try:
        print("📊 正在获取全市场实时行情数据...")
        spot_data = ak.stock_hk_spot()
        print(f"✅ 成功获取 {len(spot_data)} 只股票的实时行情数据")
        return spot_data
    except Exception as e:
        print(f"❌ 获取实时行情数据失败: {str(e)[:100]}")
        return None


def calculate_ta_indicators(df):
    """计算技术指标（KDJ+BBI+MACD+资金指标）"""
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
    df['short_term_fund'] = 100 * (df['close'] - df['short_term_low']) / (df['short_term_high'] - df['short_term_low'])

    # 计算长期资金指标
    df['long_term_low'] = df['low'].rolling(window=21).min()
    df['long_term_high'] = df['close'].rolling(window=21).max()
    df['long_term_fund'] = 100 * (df['close'] - df['long_term_low']) / (df['long_term_high'] - df['long_term_low'])

    # 删除临时列
    df = df.drop(columns=[f'MA{p}' for p in periods] + 
                 ['short_term_low', 'short_term_high', 'long_term_low', 'long_term_high'])

    return df


def process_and_save(price_df, save_path, symbol):
    """处理并保存数据"""
    try:
        # 改进4：增强空数据校验
        if not isinstance(price_df, pd.DataFrame) or price_df.empty:
            raise ValueError("空或无效数据")

        # 字段类型转换
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        price_df[numeric_cols] = price_df[numeric_cols].apply(pd.to_numeric, errors='coerce')

        # 计算技术指标
        price_df = calculate_ta_indicators(price_df.copy())
        price_df['symbol'] = symbol

        # 处理日期格式
        price_df['date'] = pd.to_datetime(price_df['date'])
        price_df = price_df.sort_values('date').reset_index(drop=True)

        # 去重处理
        price_df = price_df.drop_duplicates(subset=['date'], keep='last')

        # 数据清洗
        numeric_cols = price_df.select_dtypes(include=np.number).columns.difference(['symbol']).tolist()
        numeric_cols = [col for col in numeric_cols if col not in ['date', 'symbol']]

        if numeric_cols:
            price_df[numeric_cols] = price_df.groupby('symbol', group_keys=False)[numeric_cols].apply(
                lambda x: x.ffill().bfill()
            )

        # 定义输出列
        base_cols = ['date', 'symbol', 'open', 'high', 'low', 'close', 'volume']
        ta_cols = ['K', 'D', 'J', 'BBI', 'BBI_DIF', 'DIF', 'DEA', 'MACD', 
                  'short_term_fund', 'long_term_fund']
        output_columns = [col for col in (base_cols + ta_cols) if col in price_df.columns]

        price_df[output_columns].to_csv(save_path, index=False, encoding='utf_8_sig')

    except Exception as e:
        print(f"❌ {symbol} 处理失败: {traceback.format_exc()[:200]}")
        raise


def process_single_stock(raw_code, start_date, end_date, save_dir, spot_data=None):
    """处理单个股票的数据获取和保存"""
    max_retries = 3
    retry_delay = 5  # 重试延迟秒数
    
    for attempt in range(max_retries):
        try:
            save_path = os.path.join(save_dir, f"{raw_code}.csv")

            # 文件存在性判断
            if os.path.exists(save_path):
                try:
                    # 读取历史数据
                    history_df = pd.read_csv(save_path, encoding='utf_8_sig')
                    history_df['date'] = pd.to_datetime(history_df['date'])
                    
                    # 获取最新日期
                    latest_date = history_df['date'].max()
                    if latest_date >= pd.to_datetime(end_date):
                        return f"⏩ 已是最新数据: {raw_code}"
                    
                    # 计算增量起始日期
                    new_start = (latest_date + pd.Timedelta(days=1)).strftime("%Y%m%d")
                    temp_price_df = safe_get_hk_data(raw_code, new_start, end_date, spot_data)
                    
                    if not temp_price_df.empty:
                        combined_price = pd.concat([history_df, temp_price_df])
                    else:
                        combined_price = history_df
                except Exception as e:
                    if attempt < max_retries - 1:
                        print(f"重试 {raw_code} - 原因: {str(e)[:100]}")
                        time.sleep(retry_delay)
                        continue
                    return f"❌ 处理历史数据失败: {str(e)[:100]}..."
            else:
                # 全量数据获取
                combined_price = safe_get_hk_data(raw_code, start_date, end_date, spot_data)
                if combined_price.empty:
                    return f"⚠️ 无数据: {raw_code}"

            # 数据处理和保存
            if not combined_price.empty:
                process_and_save(combined_price, save_path, raw_code)
                return f"✅ {raw_code}"
            else:
                return f"⚠️ 无效数据: {raw_code}"

        except Exception as e:
            if attempt < max_retries - 1:
                print(f"重试 {raw_code} - 原因: {str(e)[:100]}")
                time.sleep(retry_delay)
                continue
            return f"❌ 处理失败: {str(e)[:100]}..."


def main():
    # 路径配置
    excel_path = r"D:\Quant\01_SwProj\04_VectorBT\02_Lima\Lima_Gen1\01_RawData\04_HSharesPro\HSharesPro.xlsx"
    save_dir = r"D:\Quant\01_SwProj\04_VectorBT\02_Lima\Lima_Gen1\01_RawData\04_HSharesPro\StocksData"
    os.makedirs(save_dir, exist_ok=True)

    try:
        # 读取Excel文件，跳过表头
        df = pd.read_excel(excel_path, dtype={'股票代码': str})
        stock_codes = df.iloc[:, 0].apply(pad_stock_code).dropna().unique().tolist()
        total = len(stock_codes)
    except Exception as e:
        print(f"读取Excel文件失败: {e}")
        return

    # 获取日期范围
    start_date, end_date = get_date_range()
    print(f"📅 数据获取时间范围: {start_date} 至 {end_date}")

    # 获取全市场实时行情数据（只调用一次）
    spot_data = get_spot_data()

    # 创建进度条
    with tqdm(total=total, desc="🔄 数据处理进度") as pbar:
        # 使用线程池进行并行处理
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            # 创建偏函数，固定部分参数
            process_func = partial(
                process_single_stock,
                start_date=start_date,
                end_date=end_date,
                save_dir=save_dir,
                spot_data=spot_data
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
                    pbar.set_postfix_str(result)
                except Exception as e:
                    pbar.set_postfix_str(f"❌ 失败: {code}")
                finally:
                    pbar.update(1)


if __name__ == "__main__":
    main()