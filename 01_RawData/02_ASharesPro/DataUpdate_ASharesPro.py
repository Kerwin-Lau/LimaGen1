import akshare as ak
import pandas as pd
import numpy as np  # 关键修复点[3](@ref)
import os
import talib
from datetime import datetime
from dateutil.relativedelta import relativedelta
from tqdm import tqdm


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
    """通用日期过滤函数（兼容不同接口）"""
    date_col = next((col for col in ['trade_date', 'date', '日期'] if col in df.columns), None)
    if not date_col:
        raise ValueError(f"未找到日期字段，可用列名: {df.columns.tolist()}")

    df['date'] = pd.to_datetime(df[date_col])
    start_dt = datetime.strptime(start_date, "%Y%m%d")
    end_dt = datetime.strptime(end_date, "%Y%m%d")
    return df[(df['date'] >= start_dt) & (df['date'] <= end_dt)]


def calculate_ta_indicators(df):
    """计算技术指标（KDJ+BBI+MACD）[6,8](@ref)"""
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

    return df.drop(columns=[f'MA{p}' for p in periods])


def merge_and_save(price_df, indicator_df, save_path, symbol):
    """重构版数据合并保存函数（含技术指标）"""
    try:
        # ========== 预处理阶段 ==========
        # 计算技术指标[6,8](@ref)
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

        # 新增去重操作[7](@ref)
        merged_df = merged_df.drop_duplicates(subset=['date'], keep='last')

        # ========== 数据清洗 ==========
        numeric_cols = merged_df.select_dtypes(include=np.number).columns.difference(['symbol']).tolist()
        numeric_cols = [col for col in numeric_cols if col not in ['date', 'symbol']]

        if numeric_cols:
            merged_df[numeric_cols] = merged_df.groupby('symbol', group_keys=False)[numeric_cols].apply(
                lambda x: x.ffill().bfill()
            )

        # ========== 存储阶段 ==========
        base_cols = ['date', 'symbol', 'open', 'high', 'low', 'close', 'volume', 'amount','outstanding_share', 'turnover']
        ta_cols = ['K', 'D', 'J', 'BBI','BBI_DIF']
        # 新增MACD指标字段[6](@ref)
        ta_cols += ['DIF', 'DEA', 'MACD']
        indicator_cols = [col for col in indicator_df.columns if col not in ['date', 'symbol', 'trade_date']]

        output_columns = [col for col in (base_cols + ta_cols + indicator_cols) if col in merged_df.columns]
        merged_df[output_columns].to_csv(save_path, index=False, encoding='utf_8_sig')

    except Exception as e:
        print(f"❌ {symbol} 合并失败: {str(e)[:100]}...")
        raise


def main():
    # 路径配置
    csv_path = r"D:\Quant\01_SwProj\04_VectorBT\02_Lima\Lima_Gen1\01_RawData\02_ASharesPro\ASharesPro.csv"
    save_dir = r"D:\Quant\01_SwProj\04_VectorBT\02_Lima\Lima_Gen1\01_RawData\02_ASharesPro\StocksData"
    os.makedirs(save_dir, exist_ok=True)

    # 读取CSV文件
    try:
        df = pd.read_csv(csv_path, header=None, dtype=str)
        df[0] = df[0].apply(pad_stock_code)
        indicator_codes = df[0].dropna().unique().tolist()
        price_codes = df[1].dropna().unique().tolist()
        total = len(indicator_codes)
    except Exception as e:
        print(f"读取CSV文件失败: {e}")
        return

    # 初始化进度条
    with tqdm(total=total, desc="🔄 数据合并进度") as pbar:
        start_date, end_date = get_date_range()

        for raw_code, price_code in zip(indicator_codes, price_codes):
            try:
                # 股票代码转换
                price_symbol = process_price_code(price_code)
                if not price_symbol:
                    pbar.set_postfix_str(f"❌ 无效代码: {price_code}")
                    pbar.update(1)
                    continue

                save_path = os.path.join(save_dir, f"{raw_code}.csv")

                # ================== 文件存在性判断 ==================
                if os.path.exists(save_path):
                    #添加encoding参数处理BOM头
                    history_df = pd.read_csv(save_path, encoding='utf_8_sig')
                    #处理带BOM的列名
                    history_df = history_df.rename(columns={'\ufeffdate': 'date'})
                    history_df = history_df.rename(columns={'trade_date': 'date'})
                    #确保日期列转为datetime
                    if 'date' not in history_df.columns:
                        raise ValueError(f"日期列缺失，实际列名: {history_df.columns.tolist()}")
                    history_df['date'] = pd.to_datetime(history_df['date'])
                    #统一列名处理逻辑
                    required_columns = ['date', 'open', 'high', 'low', 'close', 'volume',
                                        'amount', 'outstanding_share', 'turnover']

                    # ========== 数据完整性验证 ==========
                    latest_date = history_df['date'].max().strftime("%Y%m%d")

                    # 判断是否已是最新数据(流程图菱形判断)
                    if latest_date == end_date:
                        pbar.set_postfix_str(f"⏩ 已是最新数据")
                        pbar.update(1)
                        continue

                    # ========== 历史数据过滤 ==========
                    # 删除早于start_date的数据(流程图方框)
                    history_df = history_df[history_df['date'] >= pd.to_datetime(start_date)]

                    # ========== 增量数据获取 ==========
                    # 计算增量起始日(最后日期+1天)
                    last_date = history_df['date'].max()
                    new_start = (last_date + pd.Timedelta(days=1)).strftime("%Y%m%d")

                    # 获取增量价格数据(流程图接口调用)
                    temp_price_df = ak.stock_zh_a_daily(
                        symbol=price_symbol, adjust="qfq",
                        start_date=new_start, end_date=end_date
                    ).rename(columns={'date': 'trade_date'})

                    # 获取增量指标数据
                    temp_indicator_df = ak.stock_a_indicator_lg(raw_code)
                    temp_indicator_df = filter_by_date(temp_indicator_df, start_date, end_date)

                    # ========== 数据合并处理 ==========
                    # 列对齐处理(确保历史数据与增量数据字段一致)
                    aligned_columns = ['date', 'open', 'high', 'low', 'close', 'volume',
                                       'amount', 'outstanding_share', 'turnover']

                    # 合并历史数据与增量数据(流程图合并存储)
                    combined_price = pd.concat([
                        history_df[aligned_columns],
                        temp_price_df[aligned_columns]
                    ], ignore_index=True)

                    # 日期去重和排序
                    # 统一日期格式（处理可能的字符串格式）
                    combined_price['date'] = pd.to_datetime(combined_price['date'])
                    combined_price = combined_price.drop_duplicates('date').sort_values('date')
                    combined_price = combined_price.rename(columns={'trade_date': 'date'})

                else:
                    # ================== 全量数据获取 ==================
                    # 获取完整价格数据
                    full_price_df = ak.stock_zh_a_daily(
                        symbol=price_symbol, adjust="qfq",
                        start_date=start_date, end_date=end_date
                    ).reset_index().rename(columns={'date': 'trade_date'})

                    # 获取完整指标数据
                    full_indicator_df = ak.stock_a_indicator_lg(raw_code)
                    full_indicator_df = filter_by_date(full_indicator_df, start_date, end_date)

                    # 字段对齐
                    aligned_columns = ['trade_date', 'open', 'high', 'low', 'close', 'volume',
                                       'amount', 'outstanding_share', 'turnover']
                    combined_price = full_price_df[aligned_columns]
                    combined_price['date'] = pd.to_datetime(combined_price['trade_date'])

                # ================== 统一存储处理 ==================
                # 最终数据合并保存
                merge_and_save(
                    price_df=combined_price,
                    indicator_df=temp_indicator_df if os.path.exists(save_path) else full_indicator_df,
                    save_path=save_path,
                    symbol=raw_code
                )
                pbar.set_postfix_str(f"✅ {raw_code}")

            except Exception as e:
                print(f"❌ 处理失败: {str(e)[:100]}...")
                pbar.set_postfix_str(f"❌ 失败: {raw_code}")
            finally:
                pbar.update(1)


if __name__ == "__main__":
    main()