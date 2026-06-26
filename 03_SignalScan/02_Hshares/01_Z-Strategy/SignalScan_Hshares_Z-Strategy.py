"""
H 股 Z-Strategy 信号扫描脚本

功能概述：
    1) 从 H 股周线报告目录取最新 xlsx 作为股票池
    2) 根据当前时间（18:00 为分界）和本地数据日期决定是否调用
       DataUpdate_HSharesDaily.py 拉取/补齐日线数据
    3) 对每只股票读取日线 csv，计算 KDJ / BBI / 资金 / Brick / 突破确认等买入信号
    4) 输出到 03_SignalScan/02_Hshares/01_Z-Strategy/01_Report/HSharesPro_ScanResult_{YYYYMMDD}.xlsx

与 A 股版（SignalScan_Ashares_Z-Strategy.py）的关键差异：
    - 数据路径：H 股 `01_Database/02_Hshares/01_RawData-Daily`，代码 5 位
    - 18:00 前后两段式数据更新策略（详见 check_and_update_daily_data）
    - 股票池来源：周线报告目录最新 xlsx
    - H 股 csv 无 L1/L2/M/H1/H2 列（PE 通道时序被日线脚本跳过），
      故不输出 PE 通道相关列与优选联盟/低波红利列
    - 信号判断用 akshare stock_hk_hist 探测远程最新交易日（替代 A 股 stock_zh_a_daily）

作者：Lima Gen1
"""

import os
import sys
import time
import pandas as pd
from datetime import datetime

# ============================ 路径配置 ============================
CONFIG = {
    # H 股日线 csv 根目录（与 DataUpdate_HSharesDaily.DEFAULT_SAVE_DIR 一致）
    "data_root": r"D:\Quant\01_SwProj\04_VectorBT\02_Lima\Lima_Gen1\01_Database\02_Hshares",
    # 周线选股报告目录（按 mtime 取最新 xlsx 作为股票池）
    "stockpool_dir": r"D:\Quant\01_SwProj\04_VectorBT\02_Lima\Lima_Gen1\02_DataProcess\02_Hshares\02_WeeklyUpdate\01_Report",
    # 日线 csv 子目录名
    "data_folder": "01_RawData-Daily",
    # 信号报告输出目录
    "result_folder": r"D:\Quant\01_SwProj\04_VectorBT\02_Lima\Lima_Gen1\03_SignalScan\02_Hshares\01_Z-Strategy\01_Report",
    # 辅助清单目录（H 股版暂无 3 个清单 xlsx，保留字段以便未来接入）
    "lists_folder": r"D:\Quant\01_SwProj\04_VectorBT\02_Lima\Lima_Gen1\03_SignalScan\02_Hshares\01_Z-Strategy",
    "low_volatility_file": "低波红利清单.xlsx",
    "union_members_file": "联盟成员清单.xlsx",
    "selected_union_members_file": "优选联盟成员清单.xlsx",
    # 日线更新脚本
    "daily_update_script": r"D:\Quant\01_SwProj\04_VectorBT\02_Lima\Lima_Gen1\02_DataProcess\02_Hshares\01_DaliyUpdate\DataUpdate_HSharesDaily.py",
}


# ============================ 权重（综合得分）============================
# 综合得分 = j_val + bp_val + bbi_val + bt_val
# 移除 A 股版的 peb_* 6 个权重（PE 通道列不输出）和 pa_wi_1（优选联盟不输出）
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
    'bt_wi_1': 10,
    'bt_wi_2': 20,
}


# ============================ 模块级函数 ============================
def get_latest_stockpool_file():
    """获取股票池目录中最新的 xlsx 文件路径（按 mtime 倒序）"""
    stockpool_dir = CONFIG['stockpool_dir']
    if not os.path.isdir(stockpool_dir):
        raise FileNotFoundError(f"股票池目录不存在: {stockpool_dir}")
    xlsx_files = [f for f in os.listdir(stockpool_dir) if f.lower().endswith('.xlsx')]
    if not xlsx_files:
        raise FileNotFoundError(f"股票池目录中没有 xlsx 文件: {stockpool_dir}")
    xlsx_files.sort(key=lambda f: os.path.getmtime(os.path.join(stockpool_dir, f)), reverse=True)
    return os.path.join(stockpool_dir, xlsx_files[0])


def get_latest_trade_date():
    """获取数据目录中最新的交易日期（YYYYMMDD）"""
    max_date = datetime.min
    data_dir = os.path.join(CONFIG['data_root'], CONFIG['data_folder'])

    if not os.path.isdir(data_dir):
        return datetime.now().strftime("%Y%m%d")

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
            except Exception:
                continue

    return max_date.strftime('%Y%m%d') if max_date != datetime.min else datetime.now().strftime('%Y%m%d')


def is_workday(dt):
    """
    判断 dt 是否为"工作日"（周一-周五），仅基于 weekday()。
    法定节假日未做判断（避免引入 jsl-calendar 等外部依赖）。
    实际场景：用户在节假日手动跑脚本时，本判断会"误判为工作日"，
    但不会导致数据错误——仅会导致走"18:00-08:00 区间"分支。
    """
    return dt.weekday() < 5  # 0=周一 ... 4=周五


def _normalize_hk_columns(df):
    """
    将 akshare 港股接口返回的中文列名归一化为英文列名。
    stock_hk_daily 返回的列：日期/开盘/最高/最低/收盘/成交量
    stock_hk_hist 已是英文列名，本函数对其为 no-op。
    """
    if df is None or df.empty:
        return df
    rename_map = {
        '日期': 'date', '开盘': 'open', '最高': 'high',
        '最低': 'low', '收盘': 'close', '成交量': 'volume',
    }
    for src, dst in rename_map.items():
        if src in df.columns:
            df = df.rename(columns={src: dst})
    return df


def _fetch_hk_with_retry(symbol, start_date, end_date, max_retries=3):
    """
    单只港股的远程探测：首选 ak.stock_hk_hist，失败回退 ak.stock_hk_daily。
    内部 3 次指数退避（1s/2s/4s），主备接口都失败时抛出最后一次的异常。
    返回: pandas.DataFrame（含 date 列）或 None（取不到时）
    """
    import akshare as ak

    last_exc = None
    for attempt in range(max_retries):
        # 1) 首选：stock_hk_hist（东方财富，服务端日期过滤）
        try:
            df = ak.stock_hk_hist(
                symbol=symbol,
                period="daily",
                start_date=start_date,
                end_date=end_date,
                adjust="qfq",
            )
            if df is not None and not df.empty:
                df = _normalize_hk_columns(df)
                if 'date' in df.columns:
                    return df
        except Exception as e:
            last_exc = e

        # 2) 备用：stock_hk_daily（新浪，客户端日期过滤）
        try:
            df_full = ak.stock_hk_daily(symbol=symbol, adjust="qfq")
            if df_full is not None and not df_full.empty:
                df_full = _normalize_hk_columns(df_full)
                if 'date' in df_full.columns:
                    df_full['date'] = pd.to_datetime(df_full['date'])
                    start_dt = pd.to_datetime(start_date)
                    end_dt = pd.to_datetime(end_date)
                    df = df_full[(df_full['date'] >= start_dt) & (df_full['date'] <= end_dt)].copy()
                    if not df.empty:
                        return df
        except Exception as e:
            last_exc = e

        # 主备都失败：指数退避（1s/2s/4s）
        if attempt < max_retries - 1:
            time.sleep(2 ** attempt)

    if last_exc is not None:
        raise last_exc
    return None


def detect_remote_latest_trade_date(codes, min_samples=5, max_samples=10, lookback_days=30):
    """
    从给定港股代码中随机抽样，使用 akshare 检测远程最新交易日。
    内部对每只股票走主备接口 + 3 次指数退避，单只失败不影响其他抽样。
    返回 YYYYMMDD 字符串；所有抽样都失败时返回 None。
    """
    try:
        import random as _random
        from dateutil.relativedelta import relativedelta
    except Exception as e:
        print(f"⚠️ 检测远程交易日失败，缺少依赖: {str(e)}")
        return None

    unique_codes = list({str(c).strip().zfill(5) for c in codes if c})
    if not unique_codes:
        return None

    sample_count = min(len(unique_codes), max_samples)
    sample_count = max(sample_count, min(len(unique_codes), min_samples))
    if sample_count <= 0:
        return None

    sample_codes = _random.sample(unique_codes, sample_count)

    end_candidate = datetime.now().strftime("%Y%m%d")
    start_candidate = (datetime.now() - relativedelta(days=lookback_days)).strftime("%Y%m%d")

    latest_dates = []
    for idx, code in enumerate(sample_codes, 1):
        try:
            df = _fetch_hk_with_retry(code, start_candidate, end_candidate)
            if df is None or df.empty:
                print(f"⚠️ 第 {idx} 只股票 {code} 无可用数据")
                continue
            if 'date' not in df.columns:
                print(f"⚠️ 第 {idx} 只股票 {code} 返回数据缺少 date 列")
                continue
            df['date'] = pd.to_datetime(df['date'])
            latest_dates.append(df['date'].max())
        except Exception as e:
            print(f"⚠️ 第 {idx} 只股票 {code} 获取失败: {str(e)[:50]}")
            continue

    if not latest_dates:
        return None

    date_series = pd.to_datetime(pd.Series(latest_dates)).dt.normalize()
    # 用 max() 而非 value_counts().idxmax()：要的是"最新"，不是"最常见"；
    # 否则若多数抽样股票仍停在 20260618、少数更新到 20260622，会被众数带偏误判
    return date_series.max().strftime('%Y%m%d')


def check_and_update_daily_data(stock_codes, local_latest_date):
    """
    H 股版日线数据更新检查（满足需求 2 + 3）：
      - 18:00 前：
          - 本地 == 远程最新交易日 → 跳过
          - 本地 != 远程最新交易日 → 触发更新
      - 18:00 后：
          - 本地 == 远程最新交易日 → 跳过
          - 本地 != 远程最新交易日 → 触发更新

    触发更新时：仅更新 stock_codes 中的股票（日线脚本内部对已是最新者会快速跳过）。
    返回 True 表示数据已就绪，False 表示更新失败。
    """
    if not stock_codes:
        print("⚠️ 股票池为空，跳过数据更新检查")
        return True

    now = datetime.now()
    workday = is_workday(now)
    # 时间窗策略：
    #   - 工作日（周一-周五）：仅在 18:00-次日 08:00 允许更新（覆盖盘后/盘前数据回灌窗口）
    #   - 周末/节假日：全天允许更新
    if workday:
        today_08 = now.replace(hour=8, minute=0, second=0, microsecond=0)
        today_18 = now.replace(hour=18, minute=0, second=0, microsecond=0)
        in_allowed_window = (now >= today_18) or (now < today_08)
        window_desc = "工作日 18:00-08:00 窗口"
    else:
        in_allowed_window = True
        window_desc = "非工作日（周末/节假日）全天"

    print(f"🕒 当前时间: {now.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📅 当日类型: {window_desc}，是否允许更新: {in_allowed_window}")

    if not in_allowed_window:
        print("⏸️ 当前时间在禁止更新时间窗口内，跳过数据更新检查，沿用本地数据")
        return True

    print("\n🔍 正在探测远程最新交易日...")
    remote_latest = detect_remote_latest_trade_date(stock_codes)
    if not remote_latest:
        print("⚠️ 无法获取远程交易日，跳过数据更新检查，继续使用本地数据")
        return True

    print(f"📅 本地数据最新日期: {local_latest_date}")
    print(f"📅 远程检测最新日期: {remote_latest}")

    local_matches_remote = (local_latest_date == remote_latest)

    if local_matches_remote:
        print(f"✅ 本地 == 远程最新交易日（{remote_latest}），跳过数据更新")
        return True

    print(f"🔄 本地 ({local_latest_date}) 落后于远程 ({remote_latest})，触发股票池更新")

    # 调用更新脚本
    update_script = CONFIG['daily_update_script']
    if not os.path.exists(update_script):
        print(f"❌ 更新脚本不存在: {update_script}")
        return False

    try:
        os.makedirs(CONFIG['result_folder'], exist_ok=True)
        tmp_xlsx_name = f"_tmp_hk_stockpool_{now.strftime('%Y%m%d_%H%M%S')}.xlsx"
        tmp_xlsx_path = os.path.join(CONFIG['result_folder'], tmp_xlsx_name)

        # H 股代码 5 位补 0；'名称' 列填"未知"以兼容 DataUpdate 脚本读取
        tmp_df = pd.DataFrame({
            '代码': [str(c).strip().zfill(5) for c in stock_codes],
            '名称': ['未知'] * len(stock_codes),
        })
        tmp_df.to_excel(tmp_xlsx_path, index=False)
        print(f"📄 已生成临时股票清单: {tmp_xlsx_path} (共 {len(stock_codes)} 只)")
    except Exception as e:
        print(f"❌ 生成临时股票清单失败: {str(e)}")
        return False

    try:
        script_dir = os.path.dirname(update_script)
        if script_dir not in sys.path:
            sys.path.insert(0, script_dir)

        import importlib.util
        spec = importlib.util.spec_from_file_location("DataUpdate_HSharesDaily", update_script)
        if spec is None or spec.loader is None:
            print(f"❌ 无法加载更新脚本: {update_script}")
            return False
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        if not hasattr(module, "run_Dailyupdate"):
            print("❌ 更新脚本缺少 run_Dailyupdate 入口")
            return False

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
        try:
            if os.path.exists(tmp_xlsx_path):
                os.remove(tmp_xlsx_path)
        except Exception:
            pass


def load_stock_codes():
    """
    加载股票代码、名称和细分行业信息（使用周线报告目录最新 xlsx）。
    H 股代码 5 位补 0。
    """
    try:
        latest_stockpool = get_latest_stockpool_file()
        print(f"📌 使用最新股票池文件: {latest_stockpool}")
        df = pd.read_excel(latest_stockpool)
        code_col = df.columns[0]
        name_col = df.columns[1]
        industry_col = df.columns[2]

        # H 股代码 5 位补 0
        df[code_col] = df[code_col].astype(str).str.zfill(5)
        codes = df[code_col].unique().tolist()

        code_to_name = df.set_index(code_col)[name_col].to_dict()
        code_to_industry = df.set_index(code_col)[industry_col].to_dict()

        return codes, code_to_name, code_to_industry
    except Exception as e:
        print(f"加载股票代码失败: {str(e)}")
        return [], {}, {}


def load_and_preprocess_data(symbol):
    """加载单只股票日线 csv，预处理后取最后 100 个交易日"""
    try:
        data_dir = os.path.join(CONFIG['data_root'], CONFIG['data_folder'])
        data_path = os.path.join(data_dir, f"{symbol}.csv")

        if not os.path.exists(data_path):
            return None

        data = pd.read_csv(
            data_path,
            encoding='utf-8-sig',
            parse_dates=['date'],
            index_col='date',
            date_format='%Y-%m-%d'
        )

        return data.ffill().bfill()[-100:]
    except Exception as e:
        print(f"数据处理失败[{symbol}]: {str(e)}")
        return None


def generate_buy_signals(data):
    """
    H 股版生成买入信号（与 A 股版算法保持同步修改）。

    包含信号：
        j_negative, j_value, j_reversal, p1_signal, p2_signal,
        bbi_trend_5d, bbi_trend_20d, bbi_above,
        break_L1, touch_L2,            # H 股 csv 无 L1/L2，恒 0
        breakthrough_confirm, short_term_down_no_break, red_brick
    """
    if data is None or len(data) < 22:
        return {
            'j_negative': 0,
            'j_value': 0.0,
            'j_reversal': 0,
            'p1_signal': 0,
            'p2_signal': 0,
            'bbi_trend_5d': 0.0,
            'bbi_trend_20d': 0.0,
            'bbi_above': 0,
            'break_L1': 0,
            'touch_L2': 0,
            'breakthrough_confirm': 0,
            'short_term_down_no_break': 0,
            'red_brick': 0,
        }

    # J / 资金
    latest_j = data['J'].iloc[-1]
    prev_j = data['J'].iloc[-2]
    latest_short_fund = data['short_term_fund'].iloc[-1]
    latest_long_fund = data['long_term_fund'].iloc[-1]
    prev_short_fund = data['short_term_fund'].iloc[-2]
    prev_long_fund = data['long_term_fund'].iloc[-2]

    # BBI 趋势
    bbi_dif = data['BBI_DIF'].iloc[-20:]
    bbi_trend_5d = (bbi_dif.iloc[-5:] > 0).mean()
    bbi_trend_20d = (bbi_dif > 0).mean()

    # BBI 线上
    latest_close = data['close'].iloc[-1]
    latest_open = data['open'].iloc[-1]
    latest_bbi = data['BBI'].iloc[-1]
    bbi_above = 1 if (latest_close > latest_bbi and latest_open > latest_bbi) else 0

    # 基础 5 个
    j_negative = 1 if latest_j < 20 else 0
    p1_signal = 1 if (latest_short_fund < 20 and latest_long_fund > 80) else 0
    p2_signal = 1 if (
        latest_short_fund > 95 and latest_long_fund > 95
        and prev_short_fund < 20 and prev_long_fund > 80
    ) else 0

    # red_brick（A 股版算法直接搬）
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

    # 突破确认 + 短期下跌未破位（A 股版算法直接搬）
    try:
        prev_close = data['close'].iloc[-2]
        prev_open = data['open'].iloc[-2]
        prev_vol = data['volume'].iloc[-2]
        latest_vol = data['volume'].iloc[-1]
        prev_21_1_close = data['close'].iloc[-22:-1]
        cond1 = (prev_close == prev_21_1_close.max()) and ((prev_close - prev_open) / prev_open >= 0.05)
        if latest_close > latest_open:
            cond2 = True
        else:
            cond2 = not (0.5 * prev_vol <= latest_vol <= 0.9 * prev_vol)
        breakthrough_confirm = 1 if (cond1 and cond2) else 0

        short_term_down_no_break = 0
        if len(data) >= 4:
            day4 = data.iloc[-4]
            cond_1_1 = (day4['close'] > day4['open']) and ((day4['close'] - day4['open']) / day4['open'] > 0.05)
            last3 = data.iloc[-3:]
            cond_1_2 = all(last3['close'] < last3['open'])
            cond_1_3 = data['close'].iloc[-1] > day4['open']
            if cond_1_1 and cond_1_2 and cond_1_3:
                short_term_down_no_break = 1
    except Exception:
        breakthrough_confirm = 0
        short_term_down_no_break = 0

    return {
        'j_negative': j_negative,
        'j_value': latest_j,
        'j_reversal': 1 if latest_j > prev_j else 0,
        'p1_signal': p1_signal,
        'p2_signal': p2_signal,
        'bbi_trend_5d': bbi_trend_5d,
        'bbi_trend_20d': bbi_trend_20d,
        'bbi_above': bbi_above,
        'break_L1': 0,             # H 股 csv 无 L1/L2
        'touch_L2': 0,             # H 股 csv 无 L1/L2
        'breakthrough_confirm': breakthrough_confirm,
        'short_term_down_no_break': short_term_down_no_break,
        'red_brick': red_brick,
    }


def calculate_total_score(row, w=WEIGHTS):
    """综合得分 = j_val + bp_val + bbi_val + bt_val（H 股版精简版，无 PE 段/优选联盟段）"""
    j_val = (
        row['J到负值-日线'] * w['j_wi_1']
        + row['J到负值-日线'] * min(-1 * row['J值-日线'], w['j_wi_2'])
        + row['J到负值-日线'] * row['J值反转-日线'] * w['j_wi_3']
    )
    bp_val = (
        row['补票-P1'] * w['bp_wi_1']
        + row['补票-P2'] * w['bp_wi_2']
        + row['长线资金指标'] * w['bp_wi_2'] / 100
    )
    bbi_val = (
        row['BBI线上'] * w['bbi_wi_1']
        + row['BBI上涨趋势-5日'] * w['bbi_wi_2']
        + row['BBI上涨趋势-20日'] * w['bbi_wi_3']
    )
    bt_val = row['股价创新高'] * w['bt_wi_1'] + row['突破确认'] * w['bt_wi_2']
    return j_val + bp_val + bbi_val + bt_val


def main():
    """主函数：扫描股票并生成买入信号"""
    # 加载股票池（从周线报告目录最新 xlsx）
    stock_codes, code_to_name, code_to_industry = load_stock_codes()
    if not stock_codes:
        print("⚠️ 股票池为空，退出")
        return

    latest_date = get_latest_trade_date()

    # 18:00 前后两段式数据更新检查
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

    results = []
    total_stocks = len(stock_codes)

    for i, symbol in enumerate(stock_codes, 1):
        print(f"\r处理进度: {i}/{total_stocks}", end='')
        data = load_and_preprocess_data(symbol)
        if data is None:
            continue
        signals = generate_buy_signals(data)

        # 筛选通过条件：A 股版 7 条件 - L1/L2(2 个) = 5 条件
        # H 股版：j_negative / p1 / p2 / breakthrough_confirm / red_brick
        if any([
            signals['j_negative'],
            signals['p1_signal'],
            signals['p2_signal'],
            signals['breakthrough_confirm'],
            signals['red_brick'],
        ]):
            try:
                latest_close = data['close'].iloc[-1]
                latest_open = data['open'].iloc[-1]
                latest_BBI = data['BBI'].iloc[-1]
                bbi_online = 1 if (latest_close > latest_BBI and latest_open > latest_BBI) else 0
            except Exception:
                bbi_online = 0

            try:
                long_term_fund = data['long_term_fund'].iloc[-1]
            except Exception:
                long_term_fund = 0

            # 短线金叉信号（A 股版算法直接搬）
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

                # short_gcross_pro
                try:
                    latest_high = data['high'].iloc[-1]
                    latest_low = data['low'].iloc[-1]
                    latest_open_px = data['open'].iloc[-1]
                    latest_close = data['close'].iloc[-1]
                    latest_volume = data['volume'].iloc[-1]
                    amplitude_ratio = (latest_high - latest_low) / latest_open_px
                    condition1 = amplitude_ratio <= 0.07
                    change_ratio = (latest_close - latest_open_px) / latest_open_px
                    condition2 = -0.018 <= change_ratio <= 0.02
                    condition3 = short_gcross_plus == 1
                    if len(data) >= 10:
                        past_10_volumes = data['volume'].iloc[-10:]
                        condition4 = latest_volume == past_10_volumes.min()
                    else:
                        condition4 = False
                    short_gcross_pro = 1 if (condition1 and condition2 and condition3 and condition4) else 0
                except Exception:
                    short_gcross_pro = 0
            except Exception:
                short_gcross_normal = 0
                short_gcross_plus = 0
                short_gcross_pro = 0

            # 股价创新高（40 日新高）
            try:
                latest_highs = data['high'].iloc[-3:]
                max_high_40 = data['high'].iloc[-40:].max()
                is_new_high = 1 if (latest_highs == max_high_40).any() else 0
            except Exception:
                is_new_high = 0

            results.append({
                '股票代码': symbol,
                '股票名称': code_to_name.get(symbol, "未知"),
                '细分行业': code_to_industry.get(symbol, "未知"),
                'J到负值-日线': signals['j_negative'],
                'J值-日线': round(signals['j_value'], 2),
                'J值反转-日线': signals['j_reversal'],
                '补票-P1': signals['p1_signal'],
                '补票-P2': signals['p2_signal'],
                '长线资金指标': round(long_term_fund, 2),
                'Short_GCross_Normal': short_gcross_normal,
                'Short_GCross_Plus': short_gcross_plus,
                'Short_GCross_Pro': short_gcross_pro,
                'BBI线上': bbi_online,
                'BBI上涨趋势-5日': round(signals['bbi_trend_5d'], 2),
                'BBI上涨趋势-20日': round(signals['bbi_trend_20d'], 2),
                '股价创新高': is_new_high,
                '突破确认': signals['breakthrough_confirm'],
                '短期下跌未破位': signals['short_term_down_no_break'],
                'red_brick': signals['red_brick'],
            })

    # 输出
    if results:
        output_df = pd.DataFrame(results)
        output_df['综合得分'] = output_df.apply(lambda row: calculate_total_score(row, WEIGHTS), axis=1)

        filename = f"HSharesPro_ScanResult_{latest_date}.xlsx"
        output_path = os.path.join(CONFIG['result_folder'], filename)

        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            output_df.to_excel(writer, index=False, sheet_name='买入信号')
            from openpyxl.utils import get_column_letter
            ws = writer.sheets['买入信号']
            for idx, col in enumerate(output_df.columns):
                try:
                    max_length = max(output_df[col].astype(str).apply(len).max(), len(col))
                except Exception:
                    max_length = len(col)
                ws.column_dimensions[get_column_letter(idx + 1)].width = max_length + 2

        print(f"\n\n结果已保存至：{output_path}")
        print(f"\n共发现 {len(results)} 只股票符合买入条件")
    else:
        print("\n\n未发现符合买入条件的股票")


if __name__ == "__main__":
    main()