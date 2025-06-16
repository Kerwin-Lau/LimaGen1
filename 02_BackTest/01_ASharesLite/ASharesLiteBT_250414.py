import numpy as np
import cupy as cp
import vectorbt as vbt
import pandas as pd
import matplotlib
import os
from datetime import datetime
from dateutil.relativedelta import relativedelta

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.widgets import Cursor
import matplotlib.dates as mdates


def load_stock_data(symbol):
    """加载股票价格数据（增加异常处理）"""
    file_path = f"D:/Quant/01_SwProj/04_VectorBT/02_Lima/Lima_Gen1/01_RawData/StockPoolLite/{symbol}_price.csv"
    try:
        df = pd.read_csv(file_path, parse_dates=['date'], index_col='date').sort_index()
        df = df[['open', 'high', 'low', 'close', 'volume']]  # 确保列名统一
        end_date = datetime.now()
        return df.loc[end_date - relativedelta(years=5):end_date]
    except Exception as e:
        print(f"加载价格数据失败: {e}")
        return pd.DataFrame()


def load_pe_data(symbol):
    """加载PE数据（增强健壮性）"""
    file_path = f"D:/Quant/01_SwProj/04_VectorBT/02_Lima/Lima_Gen1/01_RawData/StockPoolLite/{symbol}_indicator.csv"
    try:
        df = pd.read_csv(file_path, parse_dates=['trade_date'], index_col='trade_date')
        df.index.name = 'date'  # 统一索引名称
        return df[['pe_ttm']].sort_index()
    except Exception as e:
        print(f"加载指标数据失败: {e}")
        return pd.DataFrame()


def calculate_pe_bands(merged_data):
    """修复后的PE通道计算"""
    merged_data['pe_ttm'] = merged_data['pe_ttm'].replace([np.inf, -np.inf], 0).fillna(0)

    min_pe = merged_data['pe_ttm'].min()
    max_pe = merged_data['pe_ttm'].max()
    step = (max_pe - min_pe) / 4

    pe_levels = [
        (min_pe, 'L2'),
        (min_pe + step, 'L1'),
        (min_pe + 2 * step, 'M'),
        (min_pe + 3 * step, 'H1'),
        (max_pe, 'H2')
    ]

    merged_data['investment_income'] = merged_data['close'] / merged_data['pe_ttm'].replace(0, 1)

    for value, col_name in pe_levels:
        merged_data[col_name] = value * merged_data['investment_income']

    return merged_data


def generate_buy_signals(merged_data):
    """独立生成买入信号[7](@ref)
    包含3个主要入场逻辑：
    1. L1支撑位反弹策略
    2. 三日反转策略(L1区间)
    3. 三日反转策略(L2区间)"""
    close = merged_data['close']
    l1 = merged_data['L1']
    l2 = merged_data.get('L2', l1 * 0.95)  # 兼容无L2的情况

    # 条件1：L1支撑位反弹
    cond1 = (
            (close.shift(2) > l1.shift(2)) &  # 前两日高于L1
            (close.shift(1) < l1.shift(1)) &  # 前一日跌破L1
            (close > close.shift(1))  # 当日回升
    )

    # 条件2：三日反转(L1区间)
    cond2 = (
            close.shift(3).between(l1.shift(3) * 0.97, l1.shift(3) * 1.03) &  # 三日前在L1±3%区间
            (close > close.shift(3) * 1.05)  # 累计涨幅超5%
    )

    # 条件3：三日反转(L2区间)
    cond3 = (
            close.shift(3).between(l2.shift(3) * 0.97, l2.shift(3) * 1.03) &  # 三日前在L2±3%区间
            (close > close.shift(3) * 1.05)
    )

    return (cond1 | cond2 | cond3).fillna(False)


def generate_sell_signals(merged_data, buy_signals):
    """整合优化版卖出信号生成函数
    新增条件：
    1. 近5日有H1突破且现价跌破三日支撑
    2. 三日连续低于L1且现价弱势
    保留原有动态止损和趋势突破逻辑"""

    close = merged_data['close']
    h1 = merged_data['H1']
    l1 = merged_data['L1']
    m = merged_data['M']
    prev_3_close = close.shift(3)  # 三日前收盘价(复用计算)

    # 条件1：近5日有H1突破且现价跌破三日支撑[7](@ref)
    cond1 = (
            (close.rolling(5).max() > h1) &  # 5日内最高价突破H1
            (close < prev_3_close)  # 现价低于三日前收盘价
    )

    # 条件2：三日连续低于L1且现价弱势[3](@ref)
    cond2 = (
            (close < l1).rolling(3).sum() == 3 &  # 连续三日低于L1
            (close < prev_3_close)  # 现价低于三日前收盘价
    )

    # 保留原有条件
    cond3 = (close > m).astype(int).diff() == 1  # 突破M线后回调[8](@ref)

    # 优化动态止损机制[7](@ref)
    buy_prices = close.where(buy_signals).ffill()
    cond4 = (close - buy_prices) / buy_prices < -0.2  # 浮亏超20%

    # 复合条件(包含原有H1突破逻辑)
    return (
            cond1 |cond2 |cond3 |cond4
    ).fillna(False)


def backtest(entries, exits, close_prices):
    """执行回测（增加空值保护）"""
    pd.set_option('future.no_silent_downcasting', True)
    return vbt.Portfolio.from_signals(
        close_prices,
        entries.astype(bool),
        exits.astype(bool),
        fees=0.001,
        freq='1D',
        init_cash=1e5
    )


def main(symbol="601336"):
    # 数据加载
    price_data = load_stock_data(symbol)
    pe_data = load_pe_data(symbol)

    if price_data.empty or pe_data.empty:
        print("数据加载失败，请检查文件路径和格式")
        return

    # 合并数据
    merged_data = pd.merge(price_data, pe_data, left_index=True, right_index=True, how='left')
    merged_data = calculate_pe_bands(merged_data)

    # 验证列存在性[1](@ref)
    print("可用列:", merged_data.columns.tolist())

    # 信号生成
    entries = generate_buy_signals(merged_data)
    exits = generate_sell_signals(merged_data, entries)

    # 执行回测
    portfolio = backtest(entries, exits, merged_data['close'])

    # 结果展示
    print(portfolio.stats())
    portfolio.plot().show()

    # 可视化通道线
    fig, ax = plt.subplots(figsize=(12, 6))
    merged_data[['close', 'L1', 'L2', 'M', 'H1', 'H2']].plot(ax=ax)
    ax.set_title(f"PE Band Strategy - {symbol}")

    # 光标交互
    cursor = Cursor(ax, useblit=True, color='red')
    annot = ax.annotate("", xy=(0, 0), xytext=(-20, 20), textcoords="offset points",
                        bbox=dict(boxstyle="round", fc="w"))

    def update_annot(event):
        if event.inaxes == ax:
            x = mdates.num2date(event.xdata).date()
            y = event.ydata
            annot.xy = (event.xdata, y)
            annot.set_text(f"Date: {x}\nPrice: {y:.2f}")
            annot.set_visible(True)
            fig.canvas.draw()

    fig.canvas.mpl_connect("motion_notify_event", update_annot)

    plt.show()


if __name__ == "__main__":
    main()