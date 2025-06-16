import numpy as np
import vectorbt as vbt
import pandas as pd
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

# 全局配置
CONFIG = {
    "data_path": "D:/Quant/01_SwProj/04_VectorBT/02_Lima/Lima_Gen1/01_RawData",
    "stockpool_folder": "StockPoolLite",
    "result_path": "D:/Quant/01_SwProj/04_VectorBT/02_Lima/Lima_Gen1/02_StrategyTraining",
    "initial_cash": 2e5,
    "n_symbols": 437,
    "fees": 0.001,
    "position_size": 0.2,
    "target_percent": 20,  # 总资产20%用于买入
    "size_granularity": 100  # 100股为最小交易单位
}


def build_data_paths():
    """构建标准化文件路径"""
    return {
        "pool_csv": os.path.join(CONFIG['data_path'], "StockPoolLite.csv"),
        "stock_data_dir": os.path.join(CONFIG['data_path'], CONFIG['stockpool_folder'])
    }


def load_symbols():
    """加载股票池"""
    pool_file = os.path.join(CONFIG['data_path'], 'StockPoolLite.csv')
    df = pd.read_csv(pool_file, header=None, names=['raw_code'])
    df['code'] = df['raw_code'].str.extract(r'(\d{6})', expand=False)
    if df['code'].isnull().any():
        invalid_codes = df[df['code'].isnull()]['raw_code'].tolist()
        raise ValueError(f"股票池包含无效代码: {invalid_codes}")
    return df['code'].astype(str).tolist()  # 移除切片操作

# 数据加载与特征计算
def load_and_preprocess_data(symbol):
    """数据加载与特征工程（含路径修复）"""
    try:
        # ==================== 新增路径验证逻辑 ====================
        # 强制补全为6位数字代码（处理前导零问题）
        symbol = f"{int(symbol):06d}"

        # 构造标准化路径（使用CONFIG配置参数）
        data_dir = os.path.join(CONFIG['data_path'], CONFIG['stockpool_folder'])
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"数据目录不存在: {data_dir}")

        # 构造完整文件路径（网页8推荐方式）
        price_path = os.path.join(data_dir, f"{symbol}_price.csv")
        indicator_path = os.path.join(data_dir, f"{symbol}_indicator.csv")

        # 添加文件存在性检查（网页6建议）
        if not all([os.path.isfile(price_path), os.path.isfile(indicator_path)]):
            print(f"文件缺失: {symbol}")
            return None
        # ========================================================

        # 数据加载（保持原始列名）
        price_data = pd.read_csv(
            price_path,  # 使用标准化路径
            usecols=['date', 'open', 'high', 'low', 'close', 'volume'],
            parse_dates=['date'],
            index_col='date'
        )
        pe_data = pd.read_csv(
            indicator_path,  # 使用标准化路径
            parse_dates=['trade_date'],
            index_col='trade_date'
        )

        # 合并数据（处理列名冲突）
        merged_data = pd.merge(
            price_data,
            pe_data,
            left_index=True,
            right_index=True,
            how='left',
            suffixes=('_price', '_indicator')
        )

        # 关键修复：重命名PE数据中的close列（网页7路径拼接经验）
        merged_data.rename(columns={
            'close_indicator': 'pe_close',
            'close_price': 'close'  # 确保主close列存在
        }, inplace=True)

        # PE通道计算
        merged_data['pe_ttm'] = merged_data['pe_ttm'].replace([np.inf, -np.inf, 0], 1e-6).ffill()
        min_pe = merged_data['pe_ttm'].min()
        max_pe = merged_data['pe_ttm'].max()
        steps = (max_pe - min_pe) / 4
        merged_data['investment_income'] = merged_data['close'] / merged_data['pe_ttm']
        pe_levels = [min_pe + i * steps for i in range(5)]
        level_names = ['L2', 'L1', 'M', 'H1', 'H2']
        for value, name in zip(pe_levels, level_names):
            merged_data[name] = value * merged_data['investment_income']

        # 计算周线KDJ（新增）
        def calculate_kdj(df, period=9):
            """计算KDJ指标"""
            low_min = df['low'].rolling(period).min()
            high_max = df['high'].rolling(period).max()
            rsv = (df['close'] - low_min) / (high_max - low_min) * 100
            k = rsv.ewm(com=2).mean()
            d = k.ewm(com=2).mean()
            j = 3 * k - 2 * d
            return pd.DataFrame({'K': k, 'D': d, 'J': j}, index=df.index)

        # 生成周线数据（使用正确的close列）
        weekly_data = merged_data.resample('W-FRI').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last'  # 现在close列已存在
        })
        weekly_kdj = calculate_kdj(weekly_data)
        weekly_kdj.columns = ['weekly_K', 'weekly_D', 'weekly_J']

        # 合并回日线数据
        merged_data = pd.merge_asof(
            merged_data.sort_index(),
            weekly_kdj.sort_index(),
            left_index=True,
            right_index=True,
            direction='backward'
        )

        return merged_data

    except Exception as e:
        print(f"数据加载失败[{symbol}]: {str(e)}")
        return None

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

    return ( cond1 | cond2 | cond3 | cond4).fillna(False)

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

    return sell_condition

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

def backtest_strategy(symbol):
    """执行回测"""
    try:
        # 数据加载与特征工程
        merged_data = load_and_preprocess_data(symbol)
        if merged_data is None:
            return None, pd.DataFrame(), pd.DataFrame()

        # 信号生成（新调用方式）
        buy_signals = generate_buy_signals(merged_data)
        sell_signals = generate_sell_signals(merged_data, buy_signals)

        # 创建投资组合
        portfolio = vbt.Portfolio.from_signals(
            close=merged_data['close'],
            entries=buy_signals,
            exits=sell_signals,
            size=0.1,
            size_type=vbt.portfolio.SizeType.Percent,
            size_granularity=100,
            accumulate=False,
            allow_partial=False,
            cash_sharing=True,
            group_by=True,
            init_cash=CONFIG['initial_cash'],
            fees=CONFIG['fees'],
            slippage=0.001,
            freq='D',
            sl_stop=0.18
        )

        # 持仓状态检测（基于positions对象）
        holding_status = get_holding_status(portfolio)

        # 交易记录统计（优化空值处理）
        closed_trades = portfolio.trades.closed
        holding_bars = np.array([], dtype=np.float64)
        trades_df = pd.DataFrame()
        date_index = merged_data.index  # 获取时间序列索引

        if len(closed_trades) > 0:
            holding_bars = closed_trades.records['exit_idx'] - closed_trades.records['entry_idx']
            closed_df = pd.DataFrame(closed_trades.records)

            # 添加关键字段
            closed_df['symbol'] = symbol
            closed_df['entry_date'] = date_index[closed_df['entry_idx']].strftime('%Y-%m-%d')
            closed_df['exit_date'] = date_index[closed_df['exit_idx']].strftime('%Y-%m-%d')

            # 字段重命名映射
            column_map = {
                'id': 'trade_id',
                'pnl': 'realized_pnl',
                'return': 'return_rate'
            }
            closed_df = closed_df.rename(columns=column_map)

            # 方法一：动态调整列顺序（自动包含其他字段）[6](@ref)
            base_columns = ['symbol', 'entry_date', 'exit_date']
            other_columns = [c for c in closed_df if c not in base_columns]
            closed_df = closed_df[base_columns + other_columns]
            trades_df = closed_df

        stats = {
            "symbol": symbol,
            "total_trades": len(closed_trades),
            "total_pnl": closed_trades.pnl.sum(),
            "total_return": closed_trades.returns.sum(),
            "win_rate": closed_trades.winning.count() / len(closed_trades) if len(closed_trades) > 0 else 0,
            "total_holding_bars": holding_bars.sum(),
            "avg_holding_bars": holding_bars.mean() if len(closed_trades) > 0 else 0,
            "holding_status": holding_status  # 新增字段
        }


        return portfolio, trades_df, pd.DataFrame([stats])

    except Exception as e:
        print(f"回测{symbol}失败: {str(e)}")
        return None, pd.DataFrame(), pd.DataFrame([{
            "symbol": symbol,
            "holding_status": "Error",
            "total_trades": 0,
            "total_pnl": 0,
            "total_return": 0,
            "win_rate": 0,
            "total_holding_bars": 0,
            "avg_holding_bars": 0
        }])


def print_strategy_stats(stats_df, trades_df):
    """打印策略统计结果"""
    print("\n" + "=" * 40)
    print("策略整体表现汇总")
    print("=" * 40)

    # 基本统计
    initial_cash = CONFIG['initial_cash']
    total_return = stats_df['total_pnl'].sum()
    total_return_rate = total_return/initial_cash
    avg_win_rate = stats_df['win_rate'].mean() * 100
    total_trades = stats_df['total_trades'].sum()
    annual_return_rate = (1+total_return_rate)**(0.2)-1

    print(f"\n▶ 基础统计")
    print(f"Start Value: {initial_cash:,.2f}元")
    print(f"End Value: {initial_cash + total_return:,.2f}元")
    print(f"Total Return: {total_return:,.2f}元")
    print(f"Total Return Rate [%]: {total_return_rate*100:,.2f}%")
    print(f"Annual Return Rate [%]: {annual_return_rate*100:,.2f}%")
    print(f"Avg Win Rate: {avg_win_rate:.1f}%")
    print(f"Total Trades: {total_trades}次")

    # ============== 新增持仓股票清单输出 ==============
    print("\n" + "-" * 40)
    print("持仓股票清单")
    print("-" * 40)

    # 提取持仓记录
    holdings = stats_df[stats_df['holding_status'] == 'Hold']
    if not holdings.empty:
        # 格式化表格输出
        print(f"{'股票代码':<10}{'持仓股数':<12}{'持仓市值':<15}{'买入时间':<12}")
        for _, row in holdings.iterrows():
            # 从交易记录中提取持仓详情
            symbol_trades = trades_df[trades_df['symbol'] == row['symbol']]
            if not symbol_trades.empty:
                # 获取最后一次买入记录
                last_buy = symbol_trades[symbol_trades['direction'] == 'Long'].iloc[-1]
                print(f"{row['symbol']:<12}"
                      f"{last_buy['size']:<12.0f}"
                      f"{last_buy['size'] * last_buy['entry_price']:<15.2f}"
                      f"{last_buy['entry_date']:<12}")
    else:
        print("⚠️ 无持仓股票")

    # 持仓时间统计 (添加空值检查)
    if not trades_df.empty and 'exit_idx' in trades_df and 'entry_idx' in trades_df:
        holding_bars = trades_df['exit_idx'] - trades_df['entry_idx']
        avg_holding = holding_bars.mean()
        std_holding = holding_bars.std()
        print(f"\n▶ 持仓时间分析")
        print(f"平均持仓周期: {avg_holding:.1f}个交易日")
        print(f"持仓周期标准差: {std_holding:.1f}天")
    else:
        print("\n▶ 无有效持仓记录")

    # 收益率分布 (添加列存在性检查)
    if not trades_df.empty and 'return' in trades_df.columns:
        positive_returns = trades_df[trades_df['return'] > 0]
        negative_returns = trades_df[trades_df['return'] <= 0]

        print(f"\n▶ 收益率分布")
        print(f"平均单笔收益率: {trades_df['return'].mean() * 100:.1f}%")
        print(f"最大单笔盈利: {trades_df['return'].max() * 100:.1f}%")
        print(f"最大单笔亏损: {trades_df['return'].min() * 100:.1f}%")
        if not positive_returns.empty:
            print(f"盈利交易平均收益: {positive_returns['return'].mean() * 100:.1f}%")
        if not negative_returns.empty:
            print(f"亏损交易平均损失: {negative_returns['return'].mean() * 100:.1f}%")
    else:
            print("\n▶ 无收益率数据")



def main():
    symbols = load_symbols()
    all_trades = []
    all_stats = []

    for symbol in tqdm(symbols, desc="回测进度"):
        portfolio, trades, stats = backtest_strategy(symbol)
        if portfolio is not None and not trades.empty:
            all_trades.append(trades)
            all_stats.append(stats)

    # 合并结果
    all_trades_df = pd.concat(all_trades).reset_index(drop=True) if all_trades else pd.DataFrame()
    stats_df = pd.concat(all_stats).reset_index(drop=True) if all_stats else pd.DataFrame()

    # 保存结果
    os.makedirs(CONFIG['result_path'], exist_ok=True)
    if not all_trades_df.empty:
        all_trades_df.to_csv(os.path.join(CONFIG['result_path'], 'TradingDetails.csv'), index=False)
    if not stats_df.empty:
        stats_df.to_csv(os.path.join(CONFIG['result_path'], 'TradingSummary.csv'), index=False)

    # 打印统计结果
    if not stats_df.empty:
        print_strategy_stats(stats_df, all_trades_df)
    else:
        print("\n⚠️ 警告: 所有股票回测均未产生交易信号")


if __name__ == "__main__":
    main()