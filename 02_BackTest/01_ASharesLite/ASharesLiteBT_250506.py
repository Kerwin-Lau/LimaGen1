import numpy as np
import vectorbt as vbt
import pandas as pd
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
pd.set_option('future.no_silent_downcasting', True)  # 关闭自动下转型
from vectorbt.base.indexing import IndexingError  # 新增导入

# 全局配置
CONFIG = {
    "data_path": "D:/Quant/01_SwProj/04_VectorBT/02_Lima/Lima_Gen1/01_RawData/01_ASharesLite",
    "stockpool_folder": "StocksData",
    "result_path": "D:/Quant/01_SwProj/04_VectorBT/02_Lima/Lima_Gen1/02_BackTest/01_ASharesLite",
    "initial_cash": 2e5,
    "n_symbols": 436,
    "fees": 0.001,
    "position_size": 0.1,
    "size_granularity": 100  # 100股为最小交易单位
}


def build_data_paths():
    """构建标准化文件路径"""
    return {
        "pool_csv": os.path.join(CONFIG['data_path'], "ASharesLite.csv"),
        "stock_data_dir": os.path.join(CONFIG['data_path'], CONFIG['stockpool_folder'])
    }


def load_symbols():
    """加载股票池"""
    pool_file = os.path.join(CONFIG['data_path'], 'ASharesLite.csv')
    df = pd.read_csv(pool_file, header=None, names=['raw_code'])
    df['code'] = df['raw_code'].str.extract(r'(\d{6})', expand=False)
    if df['code'].isnull().any():
        invalid_codes = df[df['code'].isnull()]['raw_code'].tolist()
        raise ValueError(f"股票池包含无效代码: {invalid_codes}")
    return df['code'].astype(str).tolist()  # 移除切片操作

# 数据加载与特征计算
def load_and_preprocess_data(symbol):
    """数据加载与特征工程（处理合并后的CSV文件）"""
    try:
        # 补全为6位数字代码
        symbol = f"{int(symbol):06d}"

        data_dir = os.path.join(CONFIG['data_path'], CONFIG['stockpool_folder'])
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"数据目录不存在: {data_dir}")

        # 构造新的文件路径
        symbol_path = os.path.join(data_dir, f"{symbol}.csv")
        if not os.path.isfile(symbol_path):
            print(f"文件缺失: {symbol}")
            return None

        # 读取合并后的数据文件
        data = pd.read_csv(
            symbol_path,
            parse_dates=['trade_date'],  # 假设日期列名为trade_date
            index_col='trade_date',
            usecols=['trade_date', 'open', 'high', 'low', 'close', 'volume', 'pe_ttm']
        )

        # 处理PE数据
        data['pe_ttm'] = data['pe_ttm'].replace([np.inf, -np.inf, 0], 1e-6).ffill()
        min_pe = data['pe_ttm'].min()
        max_pe = data['pe_ttm'].max()
        steps = (max_pe - min_pe) / 4
        data['investment_income'] = data['close'] / data['pe_ttm']
        pe_levels = [min_pe + i * steps for i in range(5)]
        level_names = ['L2', 'L1', 'M', 'H1', 'H2']
        for value, name in zip(pe_levels, level_names):
            data[name] = value * data['investment_income']

        # 计算周线KDJ
        def calculate_kdj(df, period=9):
            low_min = df['low'].rolling(period).min()
            high_max = df['high'].rolling(period).max()
            rsv = (df['close'] - low_min) / (high_max - low_min) * 100
            k = rsv.ewm(com=2).mean()
            d = k.ewm(com=2).mean()
            j = 3 * k - 2 * d
            return pd.DataFrame({'K': k, 'D': d, 'J': j}, index=df.index)

        weekly_data = data.resample('W-FRI').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last'
        })
        weekly_kdj = calculate_kdj(weekly_data)
        weekly_kdj.columns = ['weekly_K', 'weekly_D', 'weekly_J']

        # 合并周线KDJ到日线数据
        data = pd.merge_asof(
            data.sort_index(),
            weekly_kdj.sort_index(),
            left_index=True,
            right_index=True,
            direction='backward'
        )

        # 新增BBI指标计算（3/6/12/24日均价平均）
        for period in [3, 6, 12, 24]:
            data[f'MA{period}'] = data['close'].rolling(period).mean()
        data['BBI'] = data[['MA3', 'MA6', 'MA12', 'MA24']].mean(axis=1)

        # 清理中间列
        data.drop(columns=['MA3', 'MA6', 'MA12', 'MA24'], inplace=True)

        return data

    except Exception as e:
        print(f"数据加载失败[{symbol}]: {str(e)}")
        return None

# ================== 新增核心函数 ==================
def build_multi_symbols_data(symbols):
    """构建多股票信号矩阵（基于预对齐数据）"""
    close_list, entries_list, exits_list = [], [], []

    for symbol in tqdm(symbols, desc="数据整合"):
        merged_data = load_and_preprocess_data(symbol)
        if merged_data is None or merged_data.empty:
            continue  # 网页3建议跳过空数据

        # 生成信号（保持原有逻辑）
        buy_signals = generate_buy_signals(merged_data)
        sell_signals = generate_sell_signals(merged_data, buy_signals)

        # 直接使用预对齐的日期索引（网页6数据对齐方法）
        aligned_index = merged_data.index

        # 构建序列（网页7推荐方式）
        close_series = merged_data['close'].astype(float)
        buy_series = buy_signals.astype('boolean').astype(bool)
        sell_series = sell_signals.astype('boolean').astype(bool)
        for series in [buy_series, sell_series]:
            if series.dtype != bool:
                series = series.astype(bool)

        # 添加至列表（网页1/5合并方法）
        close_list.append(close_series.rename(symbol))
        entries_list.append(buy_series.rename(symbol))
        exits_list.append(sell_series.rename(symbol))

    # 构建多维DataFrame（网页2/4合并策略）
    return (
        pd.concat(close_list, axis=1).sort_index(),
        pd.concat(entries_list, axis=1).astype(bool).sort_index(),  # 网页2方法
        pd.concat(exits_list, axis=1).astype(bool).sort_index()
    )

def generate_buy_signals(merged_data):
    """生成买入信号（新增KDJ条件）"""
    close = merged_data['close']
    l1 = merged_data['L1']
    l2 = merged_data.get('L2', l1 * 0.95)
    weekly_j = merged_data['weekly_J']

    # 条件1：L1支撑位反弹
    cond1 = (close.shift(2) > l1.shift(2)) & (close.shift(1) < l1.shift(1)) & (close > close.shift(1))
    # 条件2：三日反转(L1区间)
    cond2 = close.shift(3).between(l1.shift(3) * 0.97, l1.shift(3) * 1.03) & (close > close.shift(3) * 1.1)
    # 条件3：三日反转(L2区间)
    cond3 = close.shift(3).between(l2.shift(3) * 0.97, l2.shift(3) * 1.03) & (close > close.shift(3) * 1.05)
    # 条件4：周线的KDJ指标中的J指标，数值为负值，且当周J值比上一周J值大
    cond4 = (weekly_j < 0) & (weekly_j > weekly_j.shift(1))

    return (cond1 | cond2 | cond3 | cond4).astype('boolean').fillna(False).astype(bool)

def generate_sell_signals(merged_data, buy_signals):

    close = merged_data['close']
    h1 = merged_data['H1']
    l1 = merged_data['L1']
    m = merged_data['M']
    prev_3_close = close.shift(3)
    weekly_j = merged_data['weekly_J']
    weekly_k = merged_data['weekly_K']
    weekly_d = merged_data['weekly_D']

    # 条件1:近5日有H1突破且现价跌破三日支撑
    cond1 = (close.rolling(5).max() > h1) & (close < prev_3_close)
    # 条件2：三日连续低于L1且现价弱势
    cond2 = (close < l1).rolling(3).sum() == 3 & (close < prev_3_close)
    # 条件3：突破M线后回调
    cond3 = (close > m).astype(int).diff() == 1
    # 条件4：周线的KDJ指标中的J指标，数值大于50，且J值小于K值和D值，且J值连续两周下降
    j_cond1 = weekly_j > 50
    j_cond2 = (weekly_j < weekly_k) & (weekly_j < weekly_d)
    j_cond3 = weekly_j < weekly_j.shift(1)  # 连续两周下降
    cond4 = j_cond1 & j_cond2 & j_cond3

    # 新增步骤：定义卖出条件组合
    sell_condition = ( cond1 | cond2 | cond3 | cond4 ).fillna(False)

    return sell_condition.astype('boolean').fillna(False).astype(bool)

# 持仓状态检测（兼容最新版本）
def get_holding_status(portfolio):
    """动态适配不同版本持仓接口"""
    try:
        # 新版检测逻辑
        open_positions = portfolio.positions[portfolio.positions.is_open]
        if len(open_positions) > 0:
            last_position = open_positions[-1]
            if last_position.size > 0:
                return "Hold"
    except AttributeError:
        # 旧版兼容逻辑
        if hasattr(portfolio, 'open_positions') and len(portfolio.open_positions) > 0:
            return "Hold"
    return "None"


def run_group_backtest(close_df, entries_df, exits_df):
    """执行组合回测（网页1、网页4方法改进）"""
    portfolio = vbt.Portfolio.from_signals(
        close=close_df,
        entries=entries_df,
        exits=exits_df,
        size=20000,
        size_type=vbt.portfolio.SizeType.Value,
        size_granularity=CONFIG['size_granularity'],
        cash_sharing=True,  # 启用资金共享
        group_by=True,  # 按组合计算绩效
        init_cash=CONFIG['initial_cash'],
        fees=CONFIG['fees'],
        slippage=0.001,
        freq='D',  # 确保当日卖出资金可用
        call_seq='auto', # 优化订单执行顺序（网页9建议）
        sl_stop = 0.18
    )

    # 解析交易记录（新版索引方式）
    trades_list = []
    try:
        all_trades = portfolio.trades.records_readable
        for symbol in close_df.columns:
            # 使用 VectorBT 默认的 'Column' 字段筛选股票
            symbol_mask = all_trades['Column'] == symbol
            if symbol_mask.any():
                symbol_trades = all_trades[symbol_mask].copy()
                # 将 VectorBT 的 'Column' 重命名为 'symbol'
                symbol_trades['symbol'] = symbol  # 或直接使用 symbol_trades['Column']
                trades_list.append(symbol_trades)
    except Exception as e:
        print(f"交易记录处理异常: {str(e)}")


    trades_df = pd.concat(trades_list) if trades_list else pd.DataFrame()
    stats = portfolio.stats(agg_func=None).reset_index()
    return portfolio, trades_df, stats


# ================== 新增统计函数 ==================
def print_group_stats(portfolio, stats_df):
    """组合级统计输出（网页4改进版）"""
    print("\n" + "=" * 40)
    print("组合整体表现")
    print("=" * 40)
    print(f"初始资金: {CONFIG['initial_cash']:,.2f}元")
    print(f"最终净值: {portfolio.total_profit():,.2f}元")



def save_series_to_csv(close_df, entries_df, exits_df):
    """将时间序列数据存储为CSV文件"""
    output_path = CONFIG['result_path']

    # 修复买入信号转换（网页5/网页7方法）
    entries_df = entries_df.astype('Int64').fillna(0).astype(int)  # 处理NaN
    exits_df = exits_df.astype('Int64').fillna(0).astype(int)     # 处理NaN

    # 存储收盘价序列
    close_df.to_csv(
        os.path.join(output_path, "CloseSeries.csv"),
        index_label="date",
        encoding='gbk',  # 支持中文路径[1](@ref)
        chunksize=1000  # 分块写入大文件[3](@ref)
    )

    # 存储买入信号（布尔值转为1/0）
    entries_df.to_csv(
        os.path.join(output_path, "BuySignals.csv"),
        index_label="date",
        header=[f"{col}_BUY" for col in entries_df.columns]  # 添加类型标记[7](@ref)
    )

    # 存储卖出信号（添加交易备注）
    exits_df.to_csv(
        os.path.join(output_path, "SellSignals.csv"),
        index_label="date",
        header=[f"{col}_SELL" for col in exits_df.columns]
    )

def main():
    symbols = load_symbols()

    # 构建统一数据（网页1关键步骤）
    close_df, entries_df, exits_df = build_multi_symbols_data(symbols)

    print(entries_df.dtypes)  # 应全部显示bool
    print(close_df.dtypes)

    # 执行组合回测（网页4方法）
    portfolio, trades_df, stats_df = run_group_backtest(close_df, entries_df, exits_df)

    # 新增：保存原始序列数据
    save_series_to_csv(close_df, entries_df, exits_df)

    # 结果保存（新增分组保存）
    os.makedirs(CONFIG['result_path'], exist_ok=True)
    trades_df.to_csv(os.path.join(CONFIG['result_path'], 'GroupTrades.csv'), index=False)
    stats_df.to_csv(os.path.join(CONFIG['result_path'], 'GroupStats.csv'), index=False)

    # 打印组合统计（增强版）
    print_group_stats(portfolio, stats_df)


if __name__ == "__main__":
    main()