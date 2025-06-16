import os
import numpy as np
import pandas as pd
from datetime import datetime

# 配置参数
CONFIG = {
    "data_root": r"D:\Quant\01_SwProj\04_VectorBT\02_Lima\Lima_Gen1\01_RawData",
    "stockpool_file": "StockPoolPro.csv",
    "data_folder": "StockPoolPro"
}


def load_stock_codes():
    """加载股票代码列表"""
    try:
        df = pd.read_csv(os.path.join(CONFIG['data_root'], CONFIG['stockpool_file']))
        # 假设股票代码列是第一列，列名可能为'代码'或未命名
        code_col = df.columns[0]  # 获取第一列的列名
        # 转换为字符串，并补足前导零到6位
        codes = df[code_col].astype(str).str.zfill(6).unique()
        return list(codes)
    except Exception as e:
        print(f"加载股票代码失败: {str(e)}")
        return []


def load_and_preprocess_data(symbol):
    """数据加载与特征工程（增强版）"""
    try:
        # 构造完整路径
        data_dir = os.path.join(CONFIG['data_root'], CONFIG['data_folder'])
        price_path = os.path.join(data_dir, f"{symbol}_price.csv")
        indicator_path = os.path.join(data_dir, f"{symbol}_indicator.csv")

        # 验证文件存在性
        if not all(os.path.exists(p) for p in [price_path, indicator_path]):
            return None

        # 加载数据（显式指定日期格式）
        price_data = pd.read_csv(
            price_path,
            usecols=['date', 'open', 'high', 'low', 'close', 'volume'],
            parse_dates=['date'],
            index_col='date',
            date_format='%Y-%m-%d'
        )
        indicator_data = pd.read_csv(
            indicator_path,
            parse_dates=['trade_date'],
            index_col='trade_date',
            date_format='%Y-%m-%d'
        )

        # 合并数据集（处理时区问题）
        merged_data = pd.merge_asof(
            price_data.sort_index(),
            indicator_data.sort_index(),
            left_index=True,
            right_index=True,
            direction='nearest'
        ).ffill()

        # PE通道计算（增强鲁棒性）
        merged_data['pe_ttm'] = merged_data['pe_ttm'].replace(
            [np.inf, -np.inf, 0], np.nan
        ).ffill().bfill()  # 关键修改点

        min_pe = merged_data['pe_ttm'].min()
        max_pe = merged_data['pe_ttm'].max()
        steps = (max_pe - min_pe) / 4 if max_pe != min_pe else 0

        pe_levels = {
            'L2': min_pe + steps,
            'L1': min_pe + steps * 2,
            'M': min_pe + steps * 3,
            'H1': min_pe + steps * 4,
            'H2': min_pe + steps * 5
        }

        for name, value in pe_levels.items():
            merged_data[name] = value * merged_data['close'] / merged_data['pe_ttm']

        # 周线KDJ计算（修复版本）
        def calculate_kdj(df, fastk_period=9):
            low_min = df['low'].rolling(fastk_period).min()
            high_max = df['high'].rolling(fastk_period).max()
            rsv = 100 * (df['close'] - low_min) / (high_max - low_min)
            rsv = rsv.replace([np.inf, -np.inf], np.nan).ffill()
            k = rsv.ewm(com=2).mean()
            d = k.ewm(com=2).mean()
            j = 3 * k - 2 * d
            return pd.DataFrame({'weekly_K': k, 'weekly_D': d, 'weekly_J': j})

        weekly_data = merged_data.resample('W-FRI').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last'
        })
        weekly_kdj = calculate_kdj(weekly_data)

        # 合并周线数据
        merged_data = merged_data.join(weekly_kdj, how='left').ffill()

        return merged_data[-100:]  # 取最近100个交易日

    except Exception as e:
        print(f"数据处理失败[{symbol}]: {str(e)}")
        return None


def generate_buy_signals(merged_data):
    """生成带明细的买入信号"""
    if merged_data is None or len(merged_data) < 4:
        return []

    close = merged_data['close']
    l1 = merged_data['L1']
    l2 = merged_data.get('L2', l1 * 0.95)
    weekly_j = merged_data['weekly_J']

    # 各条件计算
    conditions = [
        {  # 条件1：L1支撑位反弹
            'name': 'L1支撑位反弹',
            'logic': (close.shift(2) > l1.shift(2)) &
                     (close.shift(1) < l1.shift(1)) &
                     (close > close.shift(1))
        },
        {  # 条件2：三日反转(L1区间)
            'name': '三日反转(L1区间)',
            'logic': close.shift(3).between(l1.shift(3) * 0.97, l1.shift(3) * 1.03) &
                     (close > close.shift(3) * 1.1)
        },
        {  # 条件3：三日反转(L2区间)
            'name': '三日反转(L2区间)',
            'logic': close.shift(3).between(l2.shift(3) * 0.97, l2.shift(3) * 1.03) &
                     (close > close.shift(3) * 1.05)
        },
        {  # 条件4：J指标负值回升
            'name': '周线J值负回升',
            'logic': (weekly_j < 0) &
                     (weekly_j > weekly_j.shift(1))
        }
    ]

    # 获取最新信号状态
    active_conditions = []
    for cond in conditions:
        try:
            if cond['logic'].iloc[-1]:
                active_conditions.append(cond['name'])
        except IndexError:
            continue

    return active_conditions


def main():
    stock_codes = load_stock_codes()
    results = []

    for i, symbol in enumerate(stock_codes):
        print(f"\r处理进度: {i + 1}/{len(stock_codes)}", end='')

        # 获取股票名称（从指标文件）
        try:
            indicator_path = os.path.join(CONFIG['data_root'], CONFIG['data_folder'],
                                          f"{symbol}_indicator.csv")
            stock_name = pd.read_csv(indicator_path)['name'].iloc[0]
        except:
            stock_name = "未知"

        # 数据处理与信号生成
        data = load_and_preprocess_data(symbol)
        conditions = generate_buy_signals(data)

        if conditions:
            results.append({
                '代码': symbol,
                '名称': stock_name,
                '条件': ' | '.join(conditions)
            })

    # 结果输出
    print("\n\n符合买入条件的股票：")
    print("{:<10} {:<10} {}".format('代码', '名称', '触发条件'))
    for item in results:
        print("{:<10} {:<10} {}".format(item['代码'], item['名称'], item['条件']))


if __name__ == "__main__":
    main()