import os
import pandas as pd
from datetime import datetime

# 配置参数
CONFIG = {
    "data_root": r"D:\Quant\01_SwProj\04_VectorBT\02_Lima\Lima_Gen1\01_RawData\04_HSharesPro",
    "stockpool_file": "HSharesPro.xlsx",
    "data_folder": "StocksData",
    "result_folder": r"D:\Quant\01_SwProj\04_VectorBT\02_Lima\Lima_Gen1\03_BuySignalScan\04_HSharesPro"
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
        # 读取Excel文件
        file_path = os.path.join(CONFIG['data_root'], CONFIG['stockpool_file'])
        print(f"正在读取文件: {file_path}")
        
        df = pd.read_excel(file_path)
        print("Excel文件列名:", df.columns.tolist())
        print("\n前几行数据:")
        print(df.head())
        
        # 处理股票代码格式
        df['代码'] = df['代码'].astype(str).str.zfill(5)
        codes = df['代码'].unique().tolist()

        # 创建代码到名称和行业的映射
        code_to_name = df.set_index('代码')['名称'].to_dict()
        code_to_industry = df.set_index('代码')['细分行业'].to_dict()

        return codes, code_to_name, code_to_industry
    except Exception as e:
        print(f"加载股票代码失败: {str(e)}")
        print("错误详情:", e.__class__.__name__)
        import traceback
        print(traceback.format_exc())
        return [], {}, {}


def load_and_preprocess_data(symbol):
    """优化后的数据加载函数"""
    try:
        data_dir = os.path.join(CONFIG['data_root'], CONFIG['data_folder'])
        data_path = os.path.join(data_dir, f"{symbol}.csv")

        if not os.path.exists(data_path):
            return None

        # 加载数据并预处理（处理BOM头）
        data = pd.read_csv(
            data_path,
            encoding='utf-8-sig',
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
        data (pd.DataFrame): 股票数据，包含J、short_term_fund、long_term_fund等列
        
    返回:
        dict: 包含各种买入信号标志的字典
    """
    if data is None or len(data) < 22:  # 确保至少有22天数据用于突破确认
        return {
            'j_negative': 0,
            'j_value': 0.0,
            'j_reversal': 0,
            'p1_signal': 0,
            'p2_signal': 0,
            'bbi_trend_5d': 0.0,
            'bbi_trend_20d': 0.0,
            'bbi_above': 0,
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
    bbi_dif = data['BBI_DIF'].iloc[-20:]  # 获取最近20天的BBI_DIF数据
    bbi_trend_5d = (bbi_dif.iloc[-5:] > 0).mean()  # 计算最近5天为正数的占比
    bbi_trend_20d = (bbi_dif > 0).mean()  # 计算最近20天为正数的占比

    # 检查BBI线上
    latest_close = data['close'].iloc[-1]
    latest_open = data['open'].iloc[-1]
    latest_bbi = data['BBI'].iloc[-1]
    bbi_above = 1 if (latest_close > latest_bbi and latest_open > latest_bbi) else 0

    # 条件1: J值为负
    j_negative = 1 if latest_j < 0 else 0
    
    # 条件2: 补票-P1
    p1_signal = 1 if (latest_short_fund < 20 and latest_long_fund > 80) else 0
    
    # 条件3: 补票-P2
    p2_signal = 1 if (latest_short_fund > 95 and latest_long_fund > 95 and 
                     prev_short_fund < 20 and prev_long_fund > 80) else 0

    # 新增：突破确认信号
    try:
        prev_close = data['close'].iloc[-2]
        prev_open = data['open'].iloc[-2]
        latest_close = data['close'].iloc[-1]
        latest_open = data['open'].iloc[-1]
        prev_vol = data['volume'].iloc[-2]
        latest_vol = data['volume'].iloc[-1]
        prev_21_1_close = data['close'].iloc[-22:-1]
        cond1 = (prev_close == prev_21_1_close.max()) and ((prev_close - prev_open) / prev_open >= 0.05)
        if latest_close > latest_open:
            cond2 = True
        else:
            cond2 = not (0.5 * prev_vol <= latest_vol <= 0.9 * prev_vol)
        breakthrough_confirm = 1 if (cond1 and cond2) else 0
    except Exception as e:
        breakthrough_confirm = 0

    return {
        'j_negative': j_negative,
        'j_value': latest_j,
        'j_reversal': 1 if latest_j > prev_j else 0,
        'p1_signal': p1_signal,
        'p2_signal': p2_signal,
        'bbi_trend_5d': bbi_trend_5d,
        'bbi_trend_20d': bbi_trend_20d,
        'bbi_above': bbi_above,
        'breakthrough_confirm': breakthrough_confirm
    }


def main():
    """主函数：扫描股票并生成买入信号"""
    # 加载股票代码
    stock_codes, code_to_name, code_to_industry = load_stock_codes()
    latest_date = get_latest_trade_date()
    
    # 存储结果
    results = []
    
    # 处理每只股票
    total_stocks = len(stock_codes)
    for i, symbol in enumerate(stock_codes, 1):
        print(f"\r处理进度: {i}/{total_stocks}", end='')
        
        # 加载并处理数据
        data = load_and_preprocess_data(symbol)
        if data is None:
            continue
            
        # 生成买入信号
        signals = generate_buy_signals(data)
        
        # 检查是否有任何买入信号
        if any([signals['j_negative'], signals['p1_signal'], signals['p2_signal'], signals['breakthrough_confirm']]):
            results.append({
                '股票代码': symbol,
                '股票名称': code_to_name.get(symbol, "未知"),
                '细分行业': code_to_industry.get(symbol, "未知"),
                'J到负值-日线': signals['j_negative'],
                'J值-日线': round(signals['j_value'], 2),
                'J值反转-日线': signals['j_reversal'],
                '补票-P1': signals['p1_signal'],
                '补票-P2': signals['p2_signal'],
                'BBI上涨趋势-5日': round(signals['bbi_trend_5d'] * 100, 2),  # 转换为百分比
                'BBI上涨趋势-20日': round(signals['bbi_trend_20d'] * 100, 2),  # 转换为百分比
                'BBI线上': signals['bbi_above'],
                '突破确认': signals['breakthrough_confirm']
            })
    
    # 创建结果DataFrame
    if results:
        output_df = pd.DataFrame(results)
        
        # 设置输出文件名
        filename = f"HSharesPro_ScanResult_{latest_date}.xlsx"
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