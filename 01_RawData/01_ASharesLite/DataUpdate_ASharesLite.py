import akshare as ak
import pandas as pd
import os
import sys
from datetime import datetime
from dateutil.relativedelta import relativedelta
from tqdm import tqdm  # 新增进度条库[9,10](@ref)


def process_price_code(code):
    """处理价格接口的股票代码格式"""
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
    """按时间范围过滤数据"""
    df['date'] = pd.to_datetime(df['trade_date'] if 'trade_date' in df else df['日期'])
    start_dt = datetime.strptime(start_date, "%Y%m%d")
    end_dt = datetime.strptime(end_date, "%Y%m%d")
    return df[(df['date'] >= start_dt) & (df['date'] <= end_dt)]

# 技术指标计算函数
def calculate_technical_indicators(df):
    """计算多种技术指标"""
    # KDJ指标计算[6,8](@ref)
    df['K'], df['D'] = talib.STOCH(
        df['high'].values,
        df['low'].values,
        df['close'].values,
        fastk_period=9,
        slowk_period=3,
        slowk_matype=0,
        slowd_period=3,
        slowd_matype=0
    )
    df['J'] = 3 * df['K'] - 2 * df['D']

    # BBI多空指标计算[9,11](@ref)
    periods = [3, 6, 12, 24]
    for p in periods:
        df[f'MA{p}'] = talib.SMA(df['close'], timeperiod=p)
    df['BBI'] = df[[f'MA{p}' for p in periods]].mean(axis=1)
    df['BBI_DIF'] = df['BBI'].diff()  # BBI差值指标

    # MACD指标计算[12,14](@ref)
    df['DIF'], df['DEA'], df['MACD'] = talib.MACD(
        df['close'],
        fastperiod=12,
        slowperiod=26,
        signalperiod=9
    )

    # 清理中间列
    return df.drop(columns=[f'MA{p}' for p in periods])


def merge_and_save(price_df, indicator_df, save_path):
    """合并数据并保存[1,2,3](@ref)"""
    # 动态获取日期列名
    price_date_col = 'trade_date' if 'trade_date' in price_df else 'date'
    indicator_date_col = '日期' if '日期' in indicator_df else 'date'
    # 统一日期格式
    price_df = price_df.rename(columns={price_date_col: 'date'})
    indicator_df = indicator_df.rename(columns={indicator_date_col: 'date'})
    price_df['date'] = pd.to_datetime(price_df['date'])
    indicator_df['date'] = pd.to_datetime(indicator_df['date'])

    # 按日期合并（保留交集）
    merged_df = pd.merge(
        price_df,
        indicator_df,
        on='date',
        how='inner',  # 优先保证数据完整性
        suffixes=('_price', '_indicator')
    ).sort_values('date')

    # 有限度的填充（仅填充同一股票的数据）
    merged_df = merged_df.ffill().bfill()

    # 保存时保留索引日期
    merged_df.to_csv(save_path, index=False)


def main():
    # 路径配置
    csv_path = r"D:\Quant\01_SwProj\04_VectorBT\02_Lima\Lima_Gen1\01_RawData\01_ASharesLite\ASharesLite.csv"
    save_dir = r"D:\Quant\01_SwProj\04_VectorBT\02_Lima\Lima_Gen1\01_RawData\01_ASharesLite\StocksData"
    os.makedirs(save_dir, exist_ok=True)

    # 读取CSV文件
    try:
        df = pd.read_csv(csv_path, header=None, dtype=str)
        df[0] = df[0].apply(pad_stock_code)
        indicator_codes = df[0].dropna().unique().tolist()
        price_codes = df[1].dropna().unique().tolist()
        total = len(indicator_codes)  # 总任务量
    except Exception as e:
        print(f"读取CSV文件失败: {e}")
        return

    # 初始化进度条[9,10](@ref)
    with tqdm(total=total, desc="🔄 数据合并进度",
              bar_format="{l_bar}{bar:50}{r_bar}{bar:-50b}") as pbar:
        start_date, end_date = get_date_range()

        for raw_code, price_code in zip(indicator_codes, price_codes):
            # 处理价格代码
            price_symbol = process_price_code(price_code)
            if not price_symbol:
                pbar.set_postfix_str(f"❌ 无效代码: {price_code}")
                pbar.update(1)
                continue

            # 获取数据
            try:
                # 获取价格数据
                price_df = ak.stock_zh_a_daily(
                    symbol=price_symbol,
                    adjust="qfq",
                    start_date=start_date,
                    end_date=end_date
                )
                # 获取指标数据
                indicator_df = ak.stock_a_indicator_lg(symbol=raw_code)
                indicator_df = filter_by_date(indicator_df, start_date, end_date)

                # 合并与保存
                if not price_df.empty and not indicator_df.empty:
                    save_path = os.path.join(save_dir, f"{raw_code}.csv")
                    merge_and_save(price_df, indicator_df, save_path)
                    pbar.set_postfix_str(f"✅ {raw_code}")
                else:
                    pbar.set_postfix_str(f"⚠️ 空数据: {raw_code}")

            except Exception as e:
                pbar.set_postfix_str(f"❌ 失败: {raw_code}")

            finally:
                pbar.update(1)


if __name__ == "__main__":
    main()