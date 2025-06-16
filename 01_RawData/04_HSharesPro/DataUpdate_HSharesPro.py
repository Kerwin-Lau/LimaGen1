import akshare as ak
import pandas as pd
import numpy as np
import os
import talib
import time
import traceback
from datetime import datetime
from dateutil.relativedelta import relativedelta
from tqdm import tqdm


# 改进1：增强代码有效性验证
def validate_hk_code(code):
    """验证港股代码有效性"""
    return len(code) == 5 and code.isdigit()


# 改进2：安全获取数据函数（含重试机制）
def safe_get_hk_data(symbol, start_date, end_date, max_retries=3):
    """带重试机制的数据获取"""
    for _ in range(max_retries):
        try:
            df = ak.stock_hk_hist(
                symbol=symbol,
                period="daily",
                start_date=start_date,
                end_date=end_date,
                adjust="qfq"
            )
            # 改进3：动态字段映射校验
            required_cols = {'日期', '开盘', '最高', '最低', '收盘', '成交量','成交额','振幅','涨跌幅','涨跌额','换手率'}
            if df.empty or not required_cols.issubset(df.columns):
                return pd.DataFrame()

            return df.rename(columns={
                '日期': 'date',
                '开盘': 'open',
                '最高': 'high',
                '最低': 'low',
                '收盘': 'close',
                '成交量': 'volume',
                '成交额': 'amount',
                '振幅': 'amplitude',
                '涨跌幅': 'change_pct',
                '涨跌额': 'change_amount',
                '换手率': 'turnover_rate'
            })
        except Exception as e:
            print(f"⚠️ 获取数据失败: {symbol}, 重试 {_ + 1}/{max_retries}")
            time.sleep(2)
    return pd.DataFrame()


def pad_stock_code(code):
    """补全港股代码至5位"""
    code_str = str(code).strip().zfill(5)
    return code_str


def get_date_range(years=5):
    end_date = datetime.now().strftime("%Y%m%d")
    start_date = (datetime.now() - relativedelta(years=years)).strftime("%Y%m%d")
    return start_date, end_date


def calculate_ta_indicators(df):
    """计算技术指标（包含BBI_DIF）"""
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
    # 新增BBI_DIF
    df['BBI_DIF'] = df['BBI'].diff().fillna(0)

    # MACD指标
    df['DIF'], df['DEA'], df['MACD'] = talib.MACD(
        df['close'],
        fastperiod=12,
        slowperiod=26,
        signalperiod=9
    )

    return df.drop(columns=[f'MA{p}' for p in periods])


def process_and_save(price_df, save_path, symbol):
    """处理并保存数据"""
    try:
        # 改进4：增强空数据校验
        if not isinstance(price_df, pd.DataFrame) or price_df.empty:
            raise ValueError("空或无效数据")

        # 字段类型转换
        numeric_cols = ['open','high','low','close','volume','amount',
                       'amplitude','change_pct','change_amount','turnover_rate']
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
        base_cols = ['date', 'symbol', 'open', 'high', 'low', 'close', 'volume', 'amount', 'amplitude', 'change_pct', 'change_amount', 'turnover_rate']
        ta_cols = ['K', 'D', 'J', 'BBI', 'BBI_DIF', 'DIF', 'DEA', 'MACD']
        output_columns = [col for col in (base_cols + ta_cols) if col in price_df.columns]

        price_df[output_columns].to_csv(save_path, index=False, encoding='utf_8_sig')

    except Exception as e:
        print(f"❌ {symbol} 处理失败: {traceback.format_exc()[:200]}")
        raise


def main():
    # 路径配置
    csv_path = r"D:\Quant\01_SwProj\04_VectorBT\02_Lima\Lima_Gen1\01_RawData\04_HSharesPro\HSharesPro.csv"
    save_dir = r"D:\Quant\01_SwProj\04_VectorBT\02_Lima\Lima_Gen1\01_RawData\04_HSharesPro\StocksData"
    os.makedirs(save_dir, exist_ok=True)

    # 改进5：添加黑名单机制
    BAD_CODES = set()

    try:
        df = pd.read_csv(csv_path, header=None, dtype=str)
        stock_codes = [pad_stock_code(c) for c in df[0].unique()]
        stock_codes = [c for c in stock_codes if validate_hk_code(c) and c not in BAD_CODES]
        total = len(stock_codes)
    except Exception as e:
        print(f"读取CSV文件失败: {e}")
        return

    with tqdm(total=total, desc="🔄 数据处理进度") as pbar:
        start_date, end_date = get_date_range()

        for raw_code in stock_codes:
            time.sleep(1)  # 请求间隔
            try:
                symbol = f"{raw_code}"  # 直接使用5位数字代码
                save_path = os.path.join(save_dir, f"{raw_code}.csv")

                # 历史数据处理逻辑
                if os.path.exists(save_path):
                    history_df = pd.read_csv(save_path, encoding='utf_8_sig')
                    # 处理列名兼容性
                    history_df = history_df.rename(columns={c: c.replace('\ufeff', '') for c in history_df.columns})
                    history_df['date'] = pd.to_datetime(history_df['date'])

                    latest_date = history_df['date'].max()
                    if latest_date >= pd.to_datetime(end_date):
                        pbar.update(1)
                        continue

                    new_start = (latest_date + pd.Timedelta(days=1)).strftime("%Y%m%d")
                    temp_price_df = safe_get_hk_data(symbol, new_start, end_date)

                    # 处理增量数据
                    if not temp_price_df.empty:
                        common_cols = list(set(history_df.columns) & set(temp_price_df.columns))
                        combined_price = pd.concat([history_df[common_cols], temp_price_df[common_cols]])
                    else:
                        combined_price = history_df
                else:
                    # 全量数据获取
                    full_price_df = safe_get_hk_data(symbol, start_date, end_date)
                    combined_price = full_price_df

                # 空数据跳过保存
                if not combined_price.empty:
                    process_and_save(combined_price, save_path, raw_code)
                    pbar.set_postfix_str(f"✅ {raw_code}")
                else:
                    BAD_CODES.add(raw_code)
                    print(f"⏭ 无效代码加入黑名单: {raw_code}")

            except Exception as e:
                print(f"❌ 处理失败: {raw_code} - {str(e)[:100]}")
                BAD_CODES.add(raw_code)
            finally:
                pbar.update(1)


if __name__ == "__main__":
    main()