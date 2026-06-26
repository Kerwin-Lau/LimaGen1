import os
import sys
import pandas as pd
from datetime import datetime

# 配置参数
CONFIG = {
    "data_root": r"D:\Quant\01_SwProj\04_VectorBT\02_Lima\Lima_Gen1\01_Database\01_Ashares",
    "stockpool_dir": r"D:\Quant\01_SwProj\04_VectorBT\02_Lima\Lima_Gen1\02_DataProcess\01_Ashares\02_WeeklyUpdate\01_Report",
    "data_folder": "01_RawData-Daily",
    "result_folder": r"D:\Quant\01_SwProj\04_VectorBT\02_Lima\Lima_Gen1\03_SignalScan\01_Ashares\01_Z-Strategy\01_Report",
    "lists_folder": r"D:\Quant\01_SwProj\04_VectorBT\02_Lima\Lima_Gen1\03_SignalScan\01_Ashares\01_Z-Strategy",
    "low_volatility_file": "低波红利清单.xlsx",
    "union_members_file": "联盟成员清单.xlsx",
    "selected_union_members_file": "优选联盟成员清单.xlsx",
    "daily_update_script": r"D:\Quant\01_SwProj\04_VectorBT\02_Lima\Lima_Gen1\02_DataProcess\01_Ashares\01_DaliyUpdate\DataUpdate_ASharesDaily.py"
}


def get_latest_stockpool_file():
    """获取股票池目录中最新的 xlsx 文件路径"""
    stockpool_dir = CONFIG['stockpool_dir']
    if not os.path.isdir(stockpool_dir):
        raise FileNotFoundError(f"股票池目录不存在: {stockpool_dir}")
    xlsx_files = [f for f in os.listdir(stockpool_dir) if f.lower().endswith('.xlsx')]
    if not xlsx_files:
        raise FileNotFoundError(f"股票池目录中没有 xlsx 文件: {stockpool_dir}")
    xlsx_files.sort(key=lambda f: os.path.getmtime(os.path.join(stockpool_dir, f)), reverse=True)
    return os.path.join(stockpool_dir, xlsx_files[0])

# 权重系数集中管理，便于后续维护
#
# ----------------------------------------------------------------------
# ⚠️ 权重来源说明（重要）
# ----------------------------------------------------------------------
# 以下 21 维权重不再是手工调参的初始值，而是经过 3 轮 CMA-ES
# （协方差矩阵自适应进化策略）在中证 A500 股票池上跑 30 代 × 16 个体
# 优化得到的最优权重组合（v3，最终采用）。
#
# 训练配置：
#     算法          : CMA-ES (cma 4.4.4)
#     训练区间      : 2025-06-25 ~ 2025-09-03（100 只抽样股票）
#     验证区间      : 2025-12-09 ~ 2026-01-28（500 只全量）
#     测试区间      : 2026-04-08 ~ 2026-05-25（500 只全量）
#     权重搜索范围  : init_value × [0.01, 5.0]   ← 用户 2026-06 放宽
#     训练脚本      : 06_RL/01_Ashares/01_Alpha/RL_Alpha_AShares.py
#     训练产物      : 06_RL/01_Ashares/01_Alpha/outputs/best_weights.json
#
# 训练结果（End Value，初始资金 50 万）：
#     v1 (18 维, init×[0.5, 2.0]) : 训练 603,932 / 验证 644,659 / 测试 604,499
#     v2 (21 维, init×[0.5, 2.0]) : 训练 613,393 / 验证 629,862 / 测试 561,591
#     v3 (21 维, init×[0.01, 5.0]): 训练 616,192 / 验证 641,226 / 测试 661,844  ← 当前采用
#     总耗时 : 39.2 分钟（8 worker 并行）
#
# 部署说明：
#     * 直接使用本 WEIGHTS 即享受训练后的优化效果
#     * 重新训练时调 RL_Alpha_AShares.py，会写回 best_weights.json
#     * 每个字段后面带 (init=原始值 → ratio=x) 注释，标明 CMA-ES 调整方向
# ----------------------------------------------------------------------
WEIGHTS = {
    # ----- J 值相关（init 默认 25/15/10） -----
    'j_wi_1': 45.00,    # init=25.00 → 3.26x，放大近 3.3 倍
    'j_wi_2': 23.90,    # init=15.00 → 1.59x（v1）/ 0.97x（当前），几乎不变
    'j_wi_3': 25.28,    # init=10.00 → 5.00x，**顶到约束上限**，CMA 强烈放大 J 反转信号

    # ----- 资金补票（init 默认 10/15/25） -----
    'bp_wi_1': 17.49,   # init=10.00 → 1.75x，放大近 2 倍
    'bp_wi_2': 60.27,   # init=15.00 → 4.02x，大幅放大
    'bp_wi_3': 3.13,    # init=25.00 → 0.13x，大幅压低

    # ----- BBI 趋势（init 默认 10/5/5） -----
    'bbi_wi_1': 30.75,  # init=10.00 → 3.08x，明显放大
    'bbi_wi_2': 35.20,  # init=5.00 → 7.04x，**接近约束上限**，BBI 5 日趋势权重最大
    'bbi_wi_3': 11.77,  # init=5.00 → 2.35x，放大 2 倍多

    # ----- 价位区间（init 默认 20/15/25/15/-5/-10） -----
    'peb_wi_1': 53.62,  # init=20.00 → 2.68x，明显放大
    'peb_wi_2': 30.65,  # init=15.00 → 2.04x，大幅放大，"抄底 L2"信号被 CMA 强化
    'peb_wi_3': 8.50,   # init=25.00 → 0.34x，砍到 1/3
    'peb_wi_4': 8.48,   # init=15.00 → 0.57x，砍半
    'peb_wi_5': -12.32, # init=-5.00 → 2.46x 绝对值，加大高位 R3 减分力度
    'peb_wi_6': -30.48, # init=-10.00 → 3.05x 绝对值，进一步强化高位 R4 减分

    # ----- 突破（init 默认 10/20） -----
    'bt_wi_1': 11.74,   # init=10.00 → 1.17x，略升
    'bt_wi_2': 88.57,   # init=20.00 → 4.43x，**接近约束上限**，突破确认信号极重要

    # ----- 优选联盟（init 默认 10） -----
    'pa_wi_1': 0.17,    # init=10.00 → 0.02x，**顶到约束下限**，CMA 强烈认为应关闭

    # ----- 短线金叉（2026-06 新增，对应报告 J/K/L 列） -----
    'yw_wi_1': 36.30,   # init=10.00 → 3.63x，Short_GCross_Normal（J 列）
    'yw_wi_2': 49.85,   # init=10.00 → 4.98x，**顶到约束上限**，Short_GCross_Plus（K 列）
    'yw_wi_3': 35.25,   # init=10.00 → 3.53x，Short_GCross_Pro（L 列）
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


def detect_remote_latest_trade_date(codes, min_samples=5, max_samples=10, lookback_days=30):
    """
    从给定股票代码中随机抽样，使用 akshare 检测远程最新交易日。
    不依赖 DataUpdate_ASharesDaily 脚本，直接调用 akshare 接口。
    """
    try:
        import akshare as ak
        import random as _random
        import concurrent.futures
        from dateutil.relativedelta import relativedelta
    except Exception as e:
        print(f"⚠️ 检测远程交易日失败，缺少依赖: {str(e)}")
        return None

    unique_codes = list({str(c).strip().zfill(6) for c in codes if c})
    if not unique_codes:
        return None

    sample_count = min(len(unique_codes), max_samples)
    sample_count = max(sample_count, min(len(unique_codes), min_samples))
    if sample_count <= 0:
        return None

    sample_codes = _random.sample(unique_codes, sample_count)

    end_candidate = datetime.now().strftime("%Y%m%d")
    start_candidate = (datetime.now() - relativedelta(days=lookback_days)).strftime("%Y%m%d")

    def _process_price_code(code):
        num_part = ''.join(ch for ch in str(code) if ch.isdigit())
        if not num_part:
            return None
        first_digit = num_part[0]
        if first_digit == '6':
            return f'sh{num_part}'
        elif first_digit in ('0', '3'):
            return f'sz{num_part}'
        elif first_digit in ('8', '9'):
            return f'bj{num_part}'
        return None

    latest_dates = []
    for code in sample_codes:
        price_symbol = _process_price_code(code)
        if not price_symbol:
            continue
        try:
            df = ak.stock_zh_a_daily(
                symbol=price_symbol,
                adjust="qfq",
                start_date=start_candidate,
                end_date=end_candidate
            )
            if df is None or df.empty:
                continue
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
            elif 'trade_date' in df.columns:
                df['date'] = pd.to_datetime(df['trade_date'])
            else:
                continue
            latest_dates.append(df['date'].max())
        except Exception:
            continue

    if not latest_dates:
        return None

    date_series = pd.to_datetime(pd.Series(latest_dates)).dt.normalize()
    # 用 max() 而非 value_counts().idxmax()：要的是"最新"，不是"最常见"；
    # 否则若多数抽样股票仍停在 20260618、少数更新到 20260622，会被众数带偏误判
    return date_series.max().strftime('%Y%m%d')


def check_and_update_daily_data(stock_codes, local_latest_date):
    """
    随机抽样比对远程最新交易日与本地数据日期：
      - 若远程日期 == 本地日期：无需更新
      - 若不一致：调用 DataUpdate_ASharesDaily.py，仅更新股票池中股票代码
    返回 True 表示数据已就绪（无需更新或更新成功），False 表示更新失败。
    """
    if not stock_codes:
        print("⚠️ 股票池为空，跳过数据更新检查")
        return True

    print("\n🔍 正在检测远程最新交易日...")
    remote_latest = detect_remote_latest_trade_date(stock_codes)
    if not remote_latest:
        print("⚠️ 无法获取远程交易日，跳过数据更新检查，继续使用本地数据")
        return True

    print(f"📅 本地数据最新日期: {local_latest_date}")
    print(f"📅 远程检测最新日期: {remote_latest}")

    if remote_latest <= local_latest_date:
        print("✅ 本地数据已是最新，无需更新")
        return True

    # 数据不一致：调用更新脚本，仅更新股票池中股票代码
    print(f"🔄 本地数据落后于远程交易日，开始更新股票池中 {len(stock_codes)} 只股票...")
    update_script = CONFIG['daily_update_script']
    if not os.path.exists(update_script):
        print(f"❌ 更新脚本不存在: {update_script}")
        return False

    # 在 01_Report 目录下生成临时股票清单 xlsx，仅包含股票池中的代码
    try:
        os.makedirs(CONFIG['result_folder'], exist_ok=True)
        tmp_xlsx_name = f"_tmp_stockpool_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        tmp_xlsx_path = os.path.join(CONFIG['result_folder'], tmp_xlsx_name)

        # 第一列：股票代码（6位），第二列：股票名称
        tmp_df = pd.DataFrame({
            '代码': [str(c).strip().zfill(6) for c in stock_codes],
            '名称': ['未知'] * len(stock_codes)
        })
        tmp_df.to_excel(tmp_xlsx_path, index=False)
        print(f"📄 已生成临时股票清单: {tmp_xlsx_path} (共 {len(stock_codes)} 只)")
    except Exception as e:
        print(f"❌ 生成临时股票清单失败: {str(e)}")
        return False

    try:
        # 将更新脚本所在目录加入 sys.path，便于 import DataUpdate_ASharesDaily
        script_dir = os.path.dirname(update_script)
        if script_dir not in sys.path:
            sys.path.insert(0, script_dir)

        # 动态加载 DataUpdate_ASharesDaily 模块
        import importlib.util
        spec = importlib.util.spec_from_file_location("DataUpdate_ASharesDaily", update_script)
        if spec is None or spec.loader is None:
            print(f"❌ 无法加载更新脚本: {update_script}")
            return False
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        if not hasattr(module, "run_Dailyupdate"):
            print("❌ 更新脚本缺少 run_Dailyupdate 入口")
            return False

        # 仅更新股票池中股票代码
        module.run_Dailyupdate(
            stock_list_source=tmp_xlsx_path,
            save_dir=os.path.join(CONFIG['data_root'], CONFIG['data_folder']),
        )
        print("✅ 数据更新完成")
        return True
    except Exception as e:
        print(f"❌ 调用数据更新脚本失败: {str(e)}")
        return False
    finally:
        # 清理临时清单
        try:
            if os.path.exists(tmp_xlsx_path):
                os.remove(tmp_xlsx_path)
        except Exception:
            pass

def load_stock_codes():
    """加载股票代码、名称和细分行业信息（使用最新的 xlsx 股票池文件）"""
    try:
        latest_stockpool = get_latest_stockpool_file()
        print(f"📌 使用最新股票池文件: {latest_stockpool}")
        df = pd.read_excel(latest_stockpool)
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
            'breakthrough_confirm': 0,
            'short_term_down_no_break': 0,
            'red_brick': 0
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
    j_negative = 1 if latest_j < 20 else 0
    p1_signal = 1 if (latest_short_fund < 20 and latest_long_fund > 80) else 0
    p2_signal = 1 if (latest_short_fund > 95 and latest_long_fund > 95 and prev_short_fund < 20 and prev_long_fund > 80) else 0

    # 计算 red_brick 买入信号
    red_brick = 0
    try:
        prev_brick_high = data['Brick_High'].iloc[-2]
        prev_brick_low = data['Brick_Low'].iloc[-2]
        latest_brick_high = data['Brick_High'].iloc[-1]
        latest_brick_low = data['Brick_Low'].iloc[-1]

        cond_rb_1 = prev_brick_high < prev_brick_low
        cond_rb_2 = latest_brick_high > latest_brick_low
        prev_diff = abs(prev_brick_high - prev_brick_low)
        latest_diff = abs(latest_brick_high - latest_brick_low)
        cond_rb_3 = latest_diff > 0.7 * prev_diff

        if cond_rb_1 and cond_rb_2 and cond_rb_3:
            red_brick = 1
    except Exception:
        red_brick = 0

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
        # 短期下跌未破位信号
        short_term_down_no_break = 0
        if len(data) >= 4:
            day4 = data.iloc[-4]  # 从最近一个交易日往前数的第4天
            cond_1_1 = (day4['close'] > day4['open']) and ((day4['close'] - day4['open']) / day4['open'] > 0.05)
            last3 = data.iloc[-3:]
            cond_1_2 = all(last3['close'] < last3['open'])
            cond_1_3 = data['close'].iloc[-1] > day4['open']
            if cond_1_1 and cond_1_2 and cond_1_3:
                short_term_down_no_break = 1
    except Exception as e:
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
            'breakthrough_confirm': breakthrough_confirm,
            'short_term_down_no_break': short_term_down_no_break,
            'red_brick': red_brick
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
        'breakthrough_confirm': breakthrough_confirm,
        'short_term_down_no_break': short_term_down_no_break,
        'red_brick': red_brick
    }

def load_low_volatility_stocks():
    """加载低波红利股票清单"""
    try:
        file_path = os.path.join(CONFIG['lists_folder'], CONFIG['low_volatility_file'])
        df = pd.read_excel(file_path)
        return set(df.iloc[:, 0].astype(str).str.zfill(6).tolist())
    except Exception as e:
        print(f"加载低波红利清单失败: {str(e)}")
        return set()

def load_union_members():
    """加载联盟成员清单"""
    try:
        file_path = os.path.join(CONFIG['lists_folder'], CONFIG['union_members_file'])
        df = pd.read_excel(file_path)
        return set(df.iloc[:, 0].astype(str).str.zfill(6).tolist())
    except Exception as e:
        print(f"加载联盟成员清单失败: {str(e)}")
        return set()

def load_selected_union_members():
    """加载优选联盟成员清单"""
    try:
        file_path = os.path.join(CONFIG['lists_folder'], CONFIG['selected_union_members_file'])
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
    # 2026-06 新增：短线金叉三因子加权（对应报告 J/K/L 列）
    yw_val = (
        row['Short_GCross_Normal'] * w['yw_wi_1']
        + row['Short_GCross_Plus'] * w['yw_wi_2']
        + row['Short_GCross_Pro'] * w['yw_wi_3']
    )
    return j_val + bp_val + bbi_val + peb_val + bt_val + pa_val + yw_val

def main():
    """主函数：扫描股票并生成买入信号"""
    # 加载股票代码和低波红利清单
    stock_codes, code_to_name, code_to_industry = load_stock_codes()
    low_volatility_stocks = load_low_volatility_stocks()
    selected_union_members = load_selected_union_members()
    latest_date = get_latest_trade_date()

    if stock_codes:
        # 触发数据更新检查；无论成功/失败/无需更新，都重新读取本地 CSV
        # 取得最新日期，避免使用更新前缓存的旧 latest_date 写入文件名
        check_and_update_daily_data(stock_codes, latest_date)
        new_latest_date = get_latest_trade_date()
        if new_latest_date > latest_date:
            print(f"📅 本地最新交易日已刷新: {latest_date} → {new_latest_date}")
        elif new_latest_date == latest_date:
            print(f"⚠️ 本地数据未更新到最新交易日，沿用: {latest_date}")
        latest_date = new_latest_date

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
            signals['breakthrough_confirm'],
            signals['red_brick']
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
            # 计算短线金叉信号
            try:
                latest_short_trend = data['Short_Trend'].iloc[-1]
                latest_short_ls = data['Short_LS'].iloc[-1]
                short_gcross_normal = 1 if latest_short_trend > latest_short_ls else 0
                within_trend_range = False
                within_ls_range = False
                if latest_short_trend != 0:
                    within_trend_range = abs(latest_close - latest_short_trend) / abs(latest_short_trend) <= 0.02
                if latest_short_ls != 0:
                    within_ls_range = abs(latest_close - latest_short_ls) / abs(latest_short_ls) <= 0.02
                short_gcross_plus = 1 if (short_gcross_normal == 1 and (within_trend_range or within_ls_range)) else 0
                
                # 计算short_gcross_pro参数
                try:
                    latest_high = data['high'].iloc[-1]
                    latest_low = data['low'].iloc[-1]
                    latest_open = data['open'].iloc[-1]
                    latest_close = data['close'].iloc[-1]
                    latest_volume = data['volume'].iloc[-1]
                    
                    # 条件1：当日股价振幅在7%以内
                    amplitude_ratio = (latest_high - latest_low) / latest_open
                    condition1 = amplitude_ratio <= 0.07
                    
                    # 条件2：当日股价涨跌幅在-1.8%至2%以内
                    change_ratio = (latest_close - latest_open) / latest_open
                    condition2 = -0.018 <= change_ratio <= 0.02
                    
                    # 条件3：对应股票的short_gcross_plus=1
                    condition3 = short_gcross_plus == 1
                    
                    # 条件4：最新交易日volume是过去10个交易日的最小值
                    if len(data) >= 10:
                        past_10_volumes = data['volume'].iloc[-10:]
                        condition4 = latest_volume == past_10_volumes.min()
                    else:
                        condition4 = False
                    
                    # 四个条件同时满足，short_gcross_pro=1
                    short_gcross_pro = 1 if (condition1 and condition2 and condition3 and condition4) else 0
                except Exception:
                    short_gcross_pro = 0
            except Exception:
                short_gcross_normal = 0
                short_gcross_plus = 0
                short_gcross_pro = 0
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
                'Short_GCross_Normal': short_gcross_normal,
                'Short_GCross_Plus': short_gcross_plus,
                'Short_GCross_Pro': short_gcross_pro,
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
                '短期下跌未破位': signals['short_term_down_no_break'],
            'red_brick': signals['red_brick'],
                '优选联盟成员': is_union_member,
                '低波红利': 1 if symbol in low_volatility_stocks else 0
            })
    
    # 创建结果DataFrame
    if results:
        output_df = pd.DataFrame(results)
        # 计算综合得分
        output_df['综合得分'] = output_df.apply(lambda row: calculate_total_score(row, WEIGHTS), axis=1)
        # 长线资金指标保留两位小数
        if '长线资金指标' in output_df.columns:
            output_df['长线资金指标'] = output_df['长线资金指标'].round(2)
        
        # 设置输出文件名
        filename = f"ASharesPro_ScanResult_{latest_date}.xlsx"
        output_path = os.path.join(CONFIG['result_folder'], filename)
        
        # 保存为Excel文件（分表：主板、创业科创、北交）
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # 分类掩码
            main_mask = output_df['股票代码'].str.startswith(('00', '60'))
            cxkc_mask = output_df['股票代码'].str.startswith(('30', '688'))
            bj_mask = output_df['股票代码'].str.startswith(('82', '83', '87', '88', '920'))
            # 各子表
            df_main = output_df[main_mask].copy()
            df_cxkc = output_df[cxkc_mask].copy()
            df_bj = output_df[bj_mask].copy()
            # 写入各sheet
            df_main.to_excel(writer, index=False, sheet_name='主板')
            df_cxkc.to_excel(writer, index=False, sheet_name='创业科创')
            df_bj.to_excel(writer, index=False, sheet_name='北交')
            # 设置列宽函数
            def _set_col_width(ws, df_ref):
                from openpyxl.utils import get_column_letter
                for idx, col in enumerate(df_ref.columns):
                    try:
                        max_length = max(df_ref[col].astype(str).apply(len).max(), len(col))
                    except Exception:
                        max_length = len(col)
                    # 使用openpyxl的get_column_letter函数来正确获取列字母
                    col_letter = get_column_letter(idx + 1)
                    ws.column_dimensions[col_letter].width = max_length + 2
            # 应用列宽
            _set_col_width(writer.sheets['主板'], df_main if not df_main.empty else output_df)
            _set_col_width(writer.sheets['创业科创'], df_cxkc if not df_cxkc.empty else output_df)
            _set_col_width(writer.sheets['北交'], df_bj if not df_bj.empty else output_df)
        
        print(f"\n\n结果已保存至：{output_path}")
        print(f"\n共发现 {len(results)} 只股票符合买入条件")
    else:
        print("\n\n未发现符合买入条件的股票")


if __name__ == "__main__":
    main()