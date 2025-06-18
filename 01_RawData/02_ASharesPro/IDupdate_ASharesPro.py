import pandas as pd
import os


def process_stock_code(code):
    """
    处理股票代码的规则实现
    [2,4,11](@ref)
    """
    code_str = str(code).strip().zfill(6)

    # 条件2.1：6位且首数字是6
    if len(code_str) == 6 and code_str[0] == '6':
        return f'sh{code_str}'
    # 条件2.2 & 2.3：其他情况统一处理
    else:
        return f'sz{code_str.zfill(6)}'


def main():
    # 文件路径配置
    dir_path = r'D:\Quant\01_SwProj\04_VectorBT\02_Lima\Lima_Gen1\01_RawData\02_ASharesPro'
    excel_path = os.path.join(dir_path, 'ASharesPro.excel')  # 假设文件名是ASharesPro.csv

    try:
        # 读取CSV文件[6,10](@ref)
        df = pd.read_excel(excel_path, header=None, dtype=str)

        # 处理第二列[9,11](@ref)
        df[1] = df[0].apply(process_stock_code)

        # 保持第三列不变
        if df.shape[1] > 2:
            df[2] = df[2]  # 显式保留第三列
        else:
            df[2] = None  # 如果原文件只有两列，新增空第三列

        # 覆盖保存文件[3,5](@ref)
        df.to_csv(excel_path, index=False, header=False, encoding='utf-8-sig')

        print(f"成功处理并保存文件：{excel_path}")

    except Exception as e:
        print(f"处理失败：{str(e)}")
        print("可能原因：1.文件路径错误 2.文件被占用 3.数据格式异常")


if __name__ == "__main__":
    main()