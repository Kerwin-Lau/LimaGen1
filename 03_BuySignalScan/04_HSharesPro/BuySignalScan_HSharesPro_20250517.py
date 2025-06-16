import os
import pandas as pd

# 配置参数
CONFIG = {
    "data_root": r"D:\Quant\01_SwProj\04_VectorBT\02_Lima\Lima_Gen1\01_RawData\04_HSharesPro",
    "stockpool_file": "HSharesPro.csv",
    "data_folder": "StocksData"
}


def load_stock_codes():
    """加载股票代码及名称"""
    try:
        # 读取无表头CSV文件
        df = pd.read_csv(
            os.path.join(CONFIG['data_root'], CONFIG['stockpool_file']),
            header=None,
            names=['code', 'name']  # 添加自定义列名
        )

        # 处理股票代码格式（根据实际需求设置zfill位数）
        df['code'] = df['code'].astype(str).str.zfill(5)
        codes = df['code'].unique().tolist()

        # 创建代码到名称的映射
        code_to_name = df.set_index('code')['name'].to_dict()

        return codes, code_to_name
    except Exception as e:
        print(f"加载股票代码失败: {str(e)}")
        return [], {}


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
    """新版买入信号生成"""
    if data is None or len(data) == 0:
        return []

    if 'J' not in data.columns:
        return []

    latest_j = data['J'].iloc[-1]
    return ['J指标负值'] if latest_j < 0 else []


def main():
    stock_codes, code_to_name = load_stock_codes()
    results = []

    for i, symbol in enumerate(stock_codes):
        print(f"\r处理进度: {i + 1}/{len(stock_codes)}", end='')

        # 获取股票名称
        stock_name = code_to_name.get(symbol, "未知")

        # 数据处理与信号生成
        stock_name = code_to_name.get(symbol, "未知")
        data = load_and_preprocess_data(symbol)
        conditions = generate_buy_signals(data)

        if conditions:
        # 计算近30个交易日BBI_DIF的和
            bbi_diff = 0.0
            if data is not None and not data.empty and 'BBI_DIF' in data.columns:
                bbi_diff = data['BBI_DIF'].tail(30).sum()

            results.append({
                '代码': symbol,
                '名称': stock_name,
                '条件': ' | '.join(conditions),
                'BBI差值30日': round(bbi_diff, 2)
            })

    # 按BBI差值降序排序
    results_sorted = sorted(results, key=lambda x: x['BBI差值30日'], reverse=True)

    # 格式化输出
    print("\n\n符合买入条件的股票（按BBI差值降序排列）：")
    print("{:<10} {:<10} {:<20} {:<15}".format('代码', '名称', '触发条件', '近30日BBI差值'))
    for item in results_sorted:
        print("{:<10} {:<10} {:<20} {:<15.2f}".format(
            item['代码'],
            item['名称'],
            item['条件'],
            item['BBI差值30日']
        ))

if __name__ == "__main__":
    main()