import numpy as np
import vectorbt as vbt
import pandas as pd
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import openpyxl
pd.set_option('future.no_silent_downcasting', True)  # 关闭自动下转型
from vectorbt.base.indexing import IndexingError  # 新增导入

# ================== 配置与参数集中管理 ==================
CONFIG = {
    "data_path": r"D:/Quant/01_SwProj/04_VectorBT/02_Lima/Lima_Gen1/01_RawData/01_ASharesLite",
    "stockpool_file": "AsharesLite.xlsx",  # 股票池文件（Excel）
    "stockpool_folder": "StocksData",
    "result_path": r"D:/Quant/01_SwProj/04_VectorBT/02_Lima/Lima_Gen1/02_BackTest/01_ASharesLite",
    "initial_cash": 2e5,
    "n_symbols": 436,
    "fees": 0.001,
    "position_size": 0.1,
    "size_granularity": 100,  # 100股为最小交易单位
    "score_threshold": 80,     # 买入信号分数阈值，便于后续强化学习优化
    "weights": {               # 买入信号权重参数，便于后续强化学习优化
        'j_wi_1': 25,
        'j_wi_2': 15,
        'j_wi_3': 10,
        'bp_wi_1': 10,
        'bp_wi_2': 15,
        'bp_wi_3': 25,
        'bbi_wi_1': 10,
        'bbi_wi_2': 5,
        'bbi_wi_3': 5,
        'peb_wi_1': 20,
        'peb_wi_2': 15,
        'peb_wi_3': 25,
        'peb_wi_4': 15,
        'peb_wi_5': -5,
        'peb_wi_6': -10,
        'bt_wi_1': 10,
        'bt_wi_2': 20,
        'pa_wi_1': 10
    }
}

# ================== 股票池读取（Excel） ==================
def load_symbols_from_excel():
    """从Excel读取股票池代码，自动补零对齐"""
    pool_file = os.path.join(CONFIG['data_path'], CONFIG['stockpool_file'])
    df = pd.read_excel(pool_file, engine='openpyxl')
    code_col = df.columns[0]
    codes = df[code_col].astype(str).str.zfill(6).unique().tolist()
    return codes

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

# ================== 回测区间自动限定 ==================
def get_backtest_date_range():
    """获取回测区间：今天起往前5年"""
    end_date = datetime.today()
    start_date = end_date - timedelta(days=365*5)
    return start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')

# ================== 数据加载与特征计算（带区间筛选） ==================
def load_and_preprocess_data(symbol, start_date=None, end_date=None):
    """数据加载与特征工程，自动筛选回测区间"""
    try:
        symbol = f"{int(symbol):06d}"
        data_dir = os.path.join(CONFIG['data_path'], CONFIG['stockpool_folder'])
        symbol_path = os.path.join(data_dir, f"{symbol}.csv")
        if not os.path.isfile(symbol_path):
            print(f"文件缺失: {symbol}")
            return None
        data = pd.read_csv(
            symbol_path,
            parse_dates=['date'],
            index_col='date',
            encoding='utf-8'
        )
        # 按回测区间筛选
        if start_date and end_date:
            data = data.loc[(data.index >= start_date) & (data.index <= end_date)]
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

# ================== 买入信号打分体系 ==================
def calculate_total_score(row, w):
    # j_val
    j_val = (
        row['J到负值-日线'] * w['j_wi_1']
        + row['J到负值-日线'] * min(-1 * row['J值-日线'], w['j_wi_2'])
        + row['J到负值-日线'] * row['J值反转-日线'] * w['j_wi_3']
    )
    # bp_val
    bp_val = (
        row['补票-P1'] * w['bp_wi_1']
        + row['补票-P2'] * w['bp_wi_2']
        + row['长线资金指标'] * w['bp_wi_2'] / 100
    )
    # bbi_val
    bbi_val = (
        row['BBI线上'] * w['bbi_wi_1']
        + row['BBI上涨趋势-5日'] * w['bbi_wi_2']
        + row['BBI上涨趋势-20日'] * w['bbi_wi_3']
    )
    # peb_val
    peb_val = (
        row['股价跌穿L1线'] * w['peb_wi_1']
        + row['股价触碰L2底线'] * w['peb_wi_2']
        + row['股价位于R1区间'] * w['peb_wi_3']
        + row['股价位于R2区间'] * w['peb_wi_4']
        + row['股价位于R3区间'] * w['peb_wi_5']
        + row['股价位于R4区间'] * w['peb_wi_6']
    )
    # bt_val
    bt_val = row['股价创新高'] * w['bt_wi_1'] + row['突破确认'] * w['bt_wi_2']
    # pa_val
    pa_val = row['优选联盟成员'] * w['pa_wi_1']
    return j_val + bp_val + bbi_val + peb_val + bt_val + pa_val


def generate_buy_signals_for_all(symbols, start_date, end_date, weights, score_threshold):
    """每日对所有股票打分，分数最高且大于阈值的股票生成买入信号"""
    all_dates = None
    stock_scores = {}
    stock_features = {}
    for symbol in tqdm(symbols, desc="特征与打分计算"):
        data = load_and_preprocess_data(symbol, start_date, end_date)
        if data is None or data.empty:
            continue
        # 计算特征（此处需与打分体系字段一一对应，示例仅部分字段，后续可补充完善）
        features = pd.DataFrame(index=data.index)
        # 保留原始数据字段，用于卖出信号计算
        features['close'] = data['close']
        features['open'] = data['open']
        features['high'] = data['high']
        features['low'] = data['low']
        features['volume'] = data['volume']
        # 买入信号特征
        features['J到负值-日线'] = (data['J'] < 0).astype(int)
        features['J值-日线'] = data['J']
        features['J值反转-日线'] = (data['J'].diff() > 0).astype(int)
        features['补票-P1'] = ((data['short_term_fund'] < 20) & (data['long_term_fund'] > 80)).astype(int)
        features['补票-P2'] = ((data['short_term_fund'] > 95) & (data['long_term_fund'] > 95) & (data['short_term_fund'].shift(1) < 20) & (data['long_term_fund'].shift(1) > 80)).astype(int)
        features['长线资金指标'] = data['long_term_fund']
        features['BBI线上'] = ((data['close'] > data['BBI']) & (data['open'] > data['BBI'])).astype(int)
        features['BBI上涨趋势-5日'] = (data['BBI_DIF'].rolling(5).apply(lambda x: (x > 0).mean(), raw=True)).fillna(0)
        features['BBI上涨趋势-20日'] = (data['BBI_DIF'].rolling(20).apply(lambda x: (x > 0).mean(), raw=True)).fillna(0)
        features['股价跌穿L1线'] = ((data['close'].shift(1) > data['L1'].shift(1)) & (data['close'] < data['L1'])).astype(int)
        features['股价触碰L2底线'] = ((data['close'].shift(1) > data['L2'].shift(1) * 1.05) & (data['close'] < data['L2'] * 1.05)).astype(int)
        features['股价位于R1区间'] = ((data['L1'] > data['close']) & (data['close'] >= data['L2'])).astype(int)
        features['股价位于R2区间'] = ((data['M'] > data['close']) & (data['close'] >= data['L1'])).astype(int)
        features['股价位于R3区间'] = ((data['H1'] > data['close']) & (data['close'] >= data['M'])).astype(int)
        features['股价位于R4区间'] = ((data['H2'] > data['close']) & (data['close'] >= data['H1'])).astype(int)
        features['股价创新高'] = (data['high'] == data['high'].rolling(40, min_periods=1).max()).astype(int)
        features['突破确认'] = 0  # 可根据实际规则补充
        features['优选联盟成员'] = 0  # 可根据实际联盟成员名单补充
        # 计算综合得分
        features['综合得分'] = features.apply(lambda row: calculate_total_score(row, weights), axis=1)
        stock_scores[symbol] = features['综合得分']
        stock_features[symbol] = features
        if all_dates is None:
            all_dates = features.index
        else:
            all_dates = all_dates.union(features.index)
    # 构建每日买入信号矩阵
    buy_signal_df = pd.DataFrame(index=all_dates, columns=symbols, dtype=int).fillna(0)
    for date in all_dates:
        daily_scores = {symbol: stock_scores[symbol].get(date, 0) for symbol in symbols if symbol in stock_scores}
        if not daily_scores:
            continue
        max_symbol = max(daily_scores, key=daily_scores.get)
        max_score = daily_scores[max_symbol]
        if max_score >= score_threshold:
            buy_signal_df.at[date, max_symbol] = 1
    return buy_signal_df, stock_features

# ================== 卖出信号生成体系 ==================
def generate_sell_signals_for_all(buy_signal_df, stock_features, symbols, start_date, end_date):
    """生成卖出信号，五条规则，任一满足即卖出"""
    all_dates = buy_signal_df.index
    sell_signal_df = pd.DataFrame(index=all_dates, columns=symbols, dtype=int).fillna(0)
    holding_days = pd.DataFrame(0, index=all_dates, columns=symbols)
    for symbol in symbols:
        features = stock_features.get(symbol)
        if features is None or features.empty:
            continue
        buy_signal = buy_signal_df[symbol].fillna(0)
        sell_signal = pd.Series(0, index=all_dates)
        hold_count = 0
        for i, date in enumerate(all_dates):
            # 1. 持有天数超过20天
            if buy_signal.iloc[i] == 1:
                hold_count = 1
            elif hold_count > 0:
                hold_count += 1
            else:
                hold_count = 0
            holding_days.iloc[i, holding_days.columns.get_loc(symbol)] = hold_count
            rule1 = hold_count > 20
            # 2. J列前一日>100且当日<100
            rule2 = False
            if i > 0 and date in features.index and all_dates[i-1] in features.index:
                prev_j = features.loc[all_dates[i-1], 'J值-日线']
                curr_j = features.loc[date, 'J值-日线']
                rule2 = (prev_j > 100) and (curr_j < 100)
            # 3. 顶部大风车
            rule3 = False
            if date in features.index:
                row = features.loc[date]
                if i >= 19:
                    curr_close = row.get('close', None)
                    curr_open = row.get('open', None)
                    curr_high = row.get('high', None)
                    curr_low = row.get('low', None)
                    curr_vol = row.get('volume', None)
                    prev_vol = features.loc[all_dates[i-1], 'volume'] if i > 0 and all_dates[i-1] in features.index else None
                    if all(x is not None for x in [curr_close, curr_open, curr_high, curr_low, curr_vol, prev_vol]):
                        max_vol_20 = features['volume'].iloc[max(0, i-19):i+1].max()
                        rule3 = (
                            (curr_close < curr_open)
                            and (curr_vol == max_vol_20)
                            and (curr_vol > prev_vol * 1.15)
                            and (curr_high > curr_open * 1.02)
                            and (curr_low < curr_close * 0.98)
                        )
            # 4. 3/4阴量线
            rule4 = False
            if i > 0 and i >= 20 and date in features.index and all_dates[i-1] in features.index:
                prev_close = features.loc[all_dates[i-1], 'close']
                prev_open = features.loc[all_dates[i-1], 'open']
                prev_vol = features.loc[all_dates[i-1], 'volume']
                curr_close = features.loc[date, 'close']
                curr_open = features.loc[date, 'open']
                curr_vol = features.loc[date, 'volume']
                max_close_21 = features['close'].iloc[i-21:i].max() if i >= 21 else None
                if all(x is not None for x in [prev_close, prev_open, prev_vol, curr_close, curr_open, curr_vol, max_close_21]):
                    rule4 = (
                        (prev_close == max_close_21)
                        and (curr_close < curr_open)
                        and (curr_vol > prev_vol * 0.5)
                        and (curr_vol < prev_vol * 0.9)
                    )
            # 5. 强制止损5%（由回测框架sl_stop实现，这里不单独生成信号）
            # 卖出信号：任一规则满足
            if rule1 or rule2 or rule3 or rule4:
                sell_signal.iloc[i] = 1
                hold_count = 0  # 卖出后持仓归零
        sell_signal_df[symbol] = sell_signal
    return sell_signal_df

def main():
    # 1. 获取股票池和回测区间
    symbols = load_symbols_from_excel()
    start_date, end_date = get_backtest_date_range()
    weights = CONFIG['weights']
    score_threshold = CONFIG['score_threshold']

    # 2. 生成买入信号（打分体系）
    buy_signal_df, stock_features = generate_buy_signals_for_all(symbols, start_date, end_date, weights, score_threshold)

    # 3. 生成卖出信号（五条规则）
    sell_signal_df = generate_sell_signals_for_all(buy_signal_df, stock_features, symbols, start_date, end_date)

    # 4. 构建收盘价矩阵
    close_df = pd.DataFrame(index=buy_signal_df.index, columns=symbols, dtype=float)
    for symbol in symbols:
        data = load_and_preprocess_data(symbol, start_date, end_date)
        if data is not None and not data.empty:
            close_df[symbol] = data['close']

    # 5. 执行组合回测
    portfolio = vbt.Portfolio.from_signals(
        close=close_df,
        entries=buy_signal_df.astype(bool),
        exits=sell_signal_df.astype(bool),
        size=20000,
        size_type=vbt.portfolio.SizeType.Value,
        size_granularity=CONFIG['size_granularity'],
        cash_sharing=True,
        group_by=True,
        init_cash=CONFIG['initial_cash'],
        fees=CONFIG['fees'],
        slippage=0.001,
        freq='D',
        call_seq='auto',
        sl_stop=0.05  # 强制止损5%
    )

    # 6. 结果统计与输出
    stats = portfolio.stats(agg_func=None).reset_index()
    trades = portfolio.trades.records_readable
    # 年度收益率
    returns = portfolio.returns()
    yearly_returns = returns.groupby(returns.index.year).apply(lambda x: (1 + x).prod() - 1)
    # 平均年化收益率
    years = (close_df.index[-1] - close_df.index[0]).days / 365
    total_return = portfolio.total_return()
    annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

    # 7. 多sheet Excel输出
    output_path = os.path.join(CONFIG['result_path'], f"BacktestResult_{datetime.now().strftime('%Y%m%d')}.xlsx")
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        close_df.to_excel(writer, sheet_name='收盘价')
        buy_signal_df.to_excel(writer, sheet_name='买入信号')
        sell_signal_df.to_excel(writer, sheet_name='卖出信号')
        trades.to_excel(writer, sheet_name='交易明细', index=False)
        stats.to_excel(writer, sheet_name='回测统计', index=False)
        yearly_returns.to_frame('年度收益率').to_excel(writer, sheet_name='年度收益率')
        pd.DataFrame({'平均年化收益率': [annualized_return]}).to_excel(writer, sheet_name='年化收益率')
    print(f"全部结果已输出至: {output_path}")

if __name__ == "__main__":
    main()