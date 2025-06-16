import akshare as ak
import pandas as pd
import os
import sys
from datetime import datetime
from dateutil.relativedelta import relativedelta
from tqdm import tqdm  # 新增进度条库[9,10](@ref)

# 获取价格数据（需验证格式）
price_df = ak.stock_zh_a_daily(symbol=f"sh{"601816"}", adjust="qfq")  # 注意添加交易所前缀[2](@ref)
print("价格数据样例:\n", price_df.head(2))

# 获取指标数据（验证接口有效性）
try:
    indicator_df = ak.stock_a_indicator_lg(symbol="601816")  # 改用新接口[5](@ref)
    print("指标数据样例:\n", indicator_df.head(2))
except Exception as e:
    print("指标接口异常:", str(e))