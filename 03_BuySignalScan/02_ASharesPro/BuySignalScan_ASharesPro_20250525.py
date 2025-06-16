import os
import pandas as pd
from datetime import datetime

# 配置参数
CONFIG = {
    "data_root": r"D:\Quant\01_SwProj\04_VectorBT\02_Lima\Lima_Gen1\01_RawData\02_ASharesPro",
    "stockpool_file": "ASharesPro.csv",
    "data_folder": "StocksData",
    "result_folder": r"D:\Quant\01_SwProj\04_VectorBT\02_Lima\Lima_Gen1\03_BuySignalScan\02_ASharesPro",
}


def get_latest_trade_date():
    """获取数据目录中最新的交易日期[6,8](@ref)"""
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
    """加载股票代码及名称"""
    try:
        df = pd.read_csv(os.path.join(CONFIG['data_root'], CONFIG['stockpool_file']))
        # 第1列为代码，第3列为名称
        code_col = df.columns[0]
        name_col = df.columns[2]

        # 处理股票代码格式
        df[code_col] = df[code_col].astype(str).str.zfill(6)
        codes = df[code_col].unique().tolist()

        # 创建代码到名称的映射
        code_to_name = df.set_index(code_col)[name_col].to_dict()

        return codes, code_to_name
    except Exception as e:
        print(f"加载股票代码失败: {str(e)}")
        return [], {}


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
    """新版买入信号生成"""
    if data is None or len(data) == 0:
        return []

    # 检查J列是否存在
    if 'J' not in data.columns:
        return []

    # 获取最新J值
    latest_j = data['J'].iloc[-1]

    return ['J指标负值'] if latest_j < 0 else []


def main():
    stock_codes, code_to_name = load_stock_codes()
    results = []
    latest_date = get_latest_trade_date()

    for i, symbol in enumerate(stock_codes):
        print(f"\r处理进度: {i + 1}/{len(stock_codes)}", end='')

        data = load_and_preprocess_data(symbol)
        conditions = generate_buy_signals(data)

        if conditions:
            # 新增J值提取逻辑
            latest_j = data['J'].iloc[-1] if (data is not None and 'J' in data.columns) else 0.0

            # 计算BBI差值指标
            bbi_diff_10 = data['BBI_DIF'].tail(10).sum() if (data is not None and 'BBI_DIF' in data.columns) else 0.0

            results.append({
                '代码': symbol,
                '名称': code_to_name.get(symbol, "未知"),
                '触发条件': ' | '.join(conditions),
                '近10日BBI差值': round(bbi_diff_10, 2),
                '最新J值': round(latest_j, 2)  # 新增字段
            })

    # 结果输出和保存
    output_df = pd.DataFrame(results)
    filename = f"ASharesPro_ScanResult_{latest_date}.csv"
    output_path = os.path.join(CONFIG['result_folder'], filename)

    print("\n\n符合买入条件的股票：")
    # 调整打印格式
    print("{:<10} {:<10} {:<20} {:<15} {:<10}".format(
        '代码', '名称', '触发条件', '近10日BBI差值', '最新J值'))

    for item in results:
        print("{:<10} {:<10} {:<20} {:<15.2f} {:<10.2f}".format(
            item['代码'],
            item['名称'],
            item['触发条件'],
            item['近10日BBI差值'],
            item['最新J值']
        ))

    # 保存CSV文件
    output_df.to_csv(output_path, index=False, encoding='utf_8_sig')
    print(f"\n\n结果已保存至：{output_path}")


if __name__ == "__main__":
    main()