import os
import pandas as pd
from datetime import datetime

# 配置参数
CONFIG = {
    "data_root": r"D:\Quant\01_SwProj\04_VectorBT\02_Lima\Lima_Gen1\01_RawData\02_ASharesPro",
    "stockpool_file": "ASharesPro.xlsx",
    "data_folder": "StocksData",
    "result_folder": r"D:\Quant\01_SwProj\04_VectorBT\02_Lima\Lima_Gen1\03_BuySignalScan\02_ASharesPro",
    "low_volatility_file": "低波红利清单.xlsx"
}

# 权重系数集中管理，便于后续维护
WEIGHTS = {
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

def get_latest_trade_date():
    """获取数据目录中最新的交易日期"""
    max_date = datetime.min
    data_dir = os.path.join(CONFIG['data_root'], CONFIG['data_folder'])

    for file in os.listdir(data_dir):
        if file.endswith('.csv'):
            try:
                df = pd.read_csv(
                    os.path.join(data_dir, file),
                    usecols=['date'],
                    parse_dates=['date']
                )
                file_max = df['date'].max()
                if pd.notnull(file_max) and file_max > max_date:
                    max_date = file_max
            except Exception as e:
                continue

    return max_date.strftime('%Y%m%d') if max_date != datetime.min else datetime.now().strftime('%Y%m%d')

def load_stock_codes():
    """加载股票代码、名称和细分行业信息"""
    try:
        df = pd.read_excel(os.path.join(CONFIG['data_root'], CONFIG['stockpool_file']))
        # 第1列为代码，第2列为名称，第3列为细分行业
        code_col = df.columns[0]
        name_col = df.columns[1]
        industry_col = df.columns[2]

        # 处理股票代码格式
        df[code_col] = df[code_col].astype(str).str.zfill(6)
        codes = df[code_col].unique().tolist()

        # 创建代码到名称和行业的映射
        code_to_name = df.set_index(code_col)[name_col].to_dict()
        code_to_industry = df.set_index(code_col)[industry_col].to_dict()

        return codes, code_to_name, code_to_industry
    except Exception as e:
        print(f"加载股票代码失败: {str(e)}")
        return [], {}, {}


def load_and_preprocess_data(symbol):
    """简化版数据加载"""
    try:
        data_dir = os.path.join(CONFIG['data_root'], CONFIG['data_folder'])
        data_path = os.path.join(data_dir, f"{symbol}.csv")

        if not os.path.exists(data_path):
            return None

        # 加载数据并预处理
        data = pd.read_csv(
            data_path,
            parse_dates=['date'],
            index_col='date',
            date_format='%Y-%m-%d'
        )

        # 填充缺失值并取最后100个交易日
        return data.ffill().bfill()[-100:]
    except Exception as e:
        print(f"数据处理失败[{symbol}]: {str(e)}")
        return None


def generate_buy_signals(data):
    """生成买入信号
    参数:
        data (pd.DataFrame): 股票数据，包含J、short_term_fund、long_term_fund、L1、L2、close等列
    返回:
        dict: 包含各种买入信号标志的字典
    """
    if data is None or len(data) < 20:
        return {
            'j_negative': 0,
            'j_value': 0.0,
            'j_reversal': 0,
            'p1_signal': 0,
            'p2_signal': 0,
            'bbi_trend_5d': 0.0,
            'bbi_trend_20d': 0.0,
            'break_L1': 0,
            'touch_L2': 0,
            'breakthrough_confirm': 0
        }
    # 获取最新和前一天的J值
    latest_j = data['J'].iloc[-1]
    prev_j = data['J'].iloc[-2]
    # 获取最新和前一天的短期和长期资金指标
    latest_short_fund = data['short_term_fund'].iloc[-1]
    latest_long_fund = data['long_term_fund'].iloc[-1]
    prev_short_fund = data['short_term_fund'].iloc[-2]
    prev_long_fund = data['long_term_fund'].iloc[-2]
    # 计算BBI趋势
    bbi_dif = data['BBI_DIF'].iloc[-20:]
    bbi_trend_5d = (bbi_dif.iloc[-5:] > 0).mean()
    bbi_trend_20d = (bbi_dif > 0).mean()
    j_negative = 1 if latest_j < 0 else 0
    p1_signal = 1 if (latest_short_fund < 20 and latest_long_fund > 80) else 0
    p2_signal = 1 if (latest_short_fund > 95 and latest_long_fund > 95 and prev_short_fund < 20 and prev_long_fund > 80) else 0
    try:
        prev_close = data['close'].iloc[-2]
        latest_close = data['close'].iloc[-1]
        prev_L1 = data['L1'].iloc[-2]
        latest_L1 = data['L1'].iloc[-1]
        prev_L2 = data['L2'].iloc[-2]
        latest_L2 = data['L2'].iloc[-1]
        # 突破确认信号
        prev_21_1_close = data['close'].iloc[-22:-1]
        prev_open = data['open'].iloc[-2]
        latest_open = data['open'].iloc[-1]
        prev_vol = data['volume'].iloc[-2]
        latest_vol = data['volume'].iloc[-1]
        cond1 = (prev_close == prev_21_1_close.max()) and ((prev_close - prev_open) / prev_open >= 0.05)
        if latest_close > latest_open:
            cond2 = True
        else:
            cond2 = not (0.5 * prev_vol <= latest_vol <= 0.9 * prev_vol)
        breakthrough_confirm = 1 if (cond1 and cond2) else 0
    except Exception as e:
        return {
            'j_negative': j_negative,
            'j_value': latest_j,
            'j_reversal': 1 if latest_j > prev_j else 0,
            'p1_signal': p1_signal,
            'p2_signal': p2_signal,
            'bbi_trend_5d': bbi_trend_5d,
            'bbi_trend_20d': bbi_trend_20d,
            'break_L1': 0,
            'touch_L2': 0,
            'breakthrough_confirm': 0
        }
    break_L1 = 1 if (prev_close > prev_L1 and latest_close < latest_L1) else 0
    touch_L2 = 1 if (prev_close > prev_L2 * 1.05 and latest_close < latest_L2 * 1.05) else 0
    return {
        'j_negative': j_negative,
        'j_value': latest_j,
        'j_reversal': 1 if latest_j > prev_j else 0,
        'p1_signal': p1_signal,
        'p2_signal': p2_signal,
        'bbi_trend_5d': bbi_trend_5d,
        'bbi_trend_20d': bbi_trend_20d,
        'break_L1': break_L1,
        'touch_L2': touch_L2,
        'breakthrough_confirm': breakthrough_confirm
    }

def load_low_volatility_stocks():
    """加载低波红利股票清单"""
    try:
        file_path = os.path.join(CONFIG['result_folder'], CONFIG['low_volatility_file'])
        df = pd.read_excel(file_path)
        return set(df.iloc[:, 0].astype(str).str.zfill(6).tolist())
    except Exception as e:
        print(f"加载低波红利清单失败: {str(e)}")
        return set()

def load_selected_union_members():
    """加载优选联盟成员清单"""
    try:
        file_path = os.path.join(CONFIG['result_folder'], '优选联盟成员清单.xlsx')
        df = pd.read_excel(file_path)
        # 代码列名为'代码'，需补零
        return set(df.iloc[:, 0].astype(str).str.zfill(6).tolist())
    except Exception as e:
        print(f"加载优选联盟成员清单失败: {str(e)}")
        return set()

def calculate_total_score(row, w=WEIGHTS):
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

def main():
    """主函数：扫描股票并生成买入信号"""
    # 加载股票代码和低波红利清单
    stock_codes, code_to_name, code_to_industry = load_stock_codes()
    low_volatility_stocks = load_low_volatility_stocks()
    selected_union_members = load_selected_union_members()
    latest_date = get_latest_trade_date()
    
    # 存储结果
    results = []
    
    # 处理每只股票
    total_stocks = len(stock_codes)
    for i, symbol in enumerate(stock_codes, 1):
        print(f"\r处理进度: {i}/{total_stocks}", end='')
        data = load_and_preprocess_data(symbol)
        if data is None:
            continue
        signals = generate_buy_signals(data)
        # 只输出激活买入信号的股票，包括突破确认
        if any([
            signals['j_negative'],
            signals['p1_signal'],
            signals['p2_signal'],
            signals['break_L1'],
            signals['touch_L2'],
            signals['breakthrough_confirm']
        ]):
            try:
                latest_close = data['close'].iloc[-1]
                latest_open = data['open'].iloc[-1]
                latest_BBI = data['BBI'].iloc[-1]
                bbi_online = 1 if (latest_close > latest_BBI and latest_open > latest_BBI) else 0
            except Exception as e:
                bbi_online = 0
            try:
                long_term_fund = data['long_term_fund'].iloc[-1]
            except Exception as e:
                long_term_fund = 0
            # 区间判断
            try:
                latest_L1 = data['L1'].iloc[-1]
                latest_L2 = data['L2'].iloc[-1]
                latest_M = data['M'].iloc[-1]
                latest_H1 = data['H1'].iloc[-1]
                latest_H2 = data['H2'].iloc[-1]
                in_R1 = 1 if (latest_L1 > latest_close >= latest_L2) else 0
                in_R2 = 1 if (latest_M > latest_close >= latest_L1) else 0
                in_R3 = 1 if (latest_H1 > latest_close >= latest_M) else 0
                in_R4 = 1 if (latest_H2 > latest_close >= latest_H1) else 0
            except Exception as e:
                in_R1 = in_R2 = in_R3 = in_R4 = 0
            try:
                latest_highs = data['high'].iloc[-3:]
                max_high_40 = data['high'].iloc[-40:].max()
                is_new_high = 1 if (latest_highs == max_high_40).any() else 0
            except Exception as e:
                is_new_high = 0
            is_union_member = 1 if symbol in selected_union_members else 0
            results.append({
                '股票代码': symbol,
                '股票名称': code_to_name.get(symbol, "未知"),
                '细分行业': code_to_industry.get(symbol, "未知"),
                'J到负值-日线': signals['j_negative'],
                'J值-日线': round(signals['j_value'], 2),
                'J值反转-日线': signals['j_reversal'],
                '补票-P1': signals['p1_signal'],
                '补票-P2': signals['p2_signal'],
                '长线资金指标': long_term_fund,
                'BBI线上': bbi_online,
                'BBI上涨趋势-5日': round(signals['bbi_trend_5d'], 2),
                'BBI上涨趋势-20日': round(signals['bbi_trend_20d'], 2),
                '股价跌穿L1线': signals['break_L1'],
                '股价触碰L2底线': signals['touch_L2'],
                '股价位于R1区间': in_R1,
                '股价位于R2区间': in_R2,
                '股价位于R3区间': in_R3,
                '股价位于R4区间': in_R4,
                '股价创新高': is_new_high,
                '突破确认': signals['breakthrough_confirm'],
                '优选联盟成员': is_union_member,
                '低波红利': 1 if symbol in low_volatility_stocks else 0
            })
    
    # 创建结果DataFrame
    if results:
        output_df = pd.DataFrame(results)
        # 计算综合得分
        output_df['综合得分'] = output_df.apply(lambda row: calculate_total_score(row, WEIGHTS), axis=1)
        
        # 设置输出文件名
        filename = f"ASharesPro_ScanResult_{latest_date}.xlsx"
        output_path = os.path.join(CONFIG['result_folder'], filename)
        
        # 保存为Excel文件
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            output_df.to_excel(writer, index=False, sheet_name='买入信号')
            
            # 设置列宽
            worksheet = writer.sheets['买入信号']
            for idx, col in enumerate(output_df.columns):
                max_length = max(
                    output_df[col].astype(str).apply(len).max(),
                    len(col)
                )
                worksheet.column_dimensions[chr(65 + idx)].width = max_length + 2
        
        print(f"\n\n结果已保存至：{output_path}")
        print(f"\n共发现 {len(results)} 只股票符合买入条件")
    else:
        print("\n\n未发现符合买入条件的股票")


if __name__ == "__main__":
    main()