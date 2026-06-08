# -*- coding: utf-8 -*-
"""
SpecifiedPeriodScan.py
======================

对指定时间段内 A 股全量日线数据进行区间扫描，输出每只股票区间内的最低价/最高价
及对应日期，并计算区间最大涨幅。

使用方式：
    1. 直接运行本脚本，使用模块顶部默认时间段（20260408-20260525）；
    2. 修改模块顶部 ``START_DATE`` / ``END_DATE`` 重新执行即可生成新的报告。

Author: Lima_Gen1
"""

import os
import glob
import logging
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from tqdm import tqdm
from openpyxl import load_workbook

# ============================ 时间段配置（可被外部覆盖）============================
# 默认扫描区间：2026年4月8日至2026年5月25日
START_DATE = "20250104"
END_DATE = "20250228"


# ============================ 路径配置 ============================
# 股票清单
ASHARES_LIST_PATH = (
    r"D:\Quant\01_SwProj\04_VectorBT\02_Lima\Lima_Gen1"
    r"\01_Database\01_Ashares\ASharesList.xlsx"
)
# 日线 CSV 所在目录
DAILY_DATA_DIR = (
    r"D:\Quant\01_SwProj\04_VectorBT\02_Lima\Lima_Gen1"
    r"\01_Database\01_Ashares\01_RawData-Daily"
)
# 报告输出目录
REPORT_DIR = (
    r"D:\Quant\01_SwProj\04_VectorBT\02_Lima\Lima_Gen1"
    r"\02_DataProcess\01_Ashares\03_SpecifiedPeriodScan\01_Report"
)


# ============================ 日志配置 ============================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ============================ 核心逻辑 ============================
def _scan_single_stock(csv_path: str, start_dt: pd.Timestamp, end_dt: pd.Timestamp) -> dict | None:
    """
    扫描单只股票的日线数据，提取区间内最低/最高价及对应日期，并计算最大涨幅。

    Args:
        csv_path:  日线 CSV 文件绝对路径。
        start_dt: 区间起始日期（含）。
        end_dt:   区间结束日期（含）。

    Returns:
        dict 包含 ``股票代码`` / ``区间最低价`` / ``最低价日期`` /
        ``区间最高价`` / ``最高价日期`` / ``区间最大涨幅`` 字段；
        若区间内无数据、或 ``最高价日期`` 早于 ``最低价日期``，则返回 ``None``。
    """
    try:
        # 仅读取需要的列，减少 IO
        df = pd.read_csv(
            csv_path,
            usecols=["date", "low", "high"],
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("读取文件失败: %s, 错误: %s", csv_path, exc)
        return None

    if df.empty:
        return None

    # 日期列转换为 datetime
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df = df[(df["date"] >= start_dt) & (df["date"] <= end_dt)]

    if df.empty:
        return None

    # 最低价 / 最高价
    low_min = df["low"].min()
    low_idx = df["low"].idxmin()
    low_date = df.loc[low_idx, "date"]

    high_max = df["high"].max()
    high_idx = df["high"].idxmax()
    high_date = df.loc[high_idx, "date"]

    # 区间最大涨幅：(区间最高 - 区间最低) / 区间最低
    if pd.isna(low_min) or low_min == 0:
        max_rise = float("nan")
    else:
        max_rise = (high_max - low_min) / low_min

    # 最高价日期早于最低价日期，视为无效区间（在最低点之前就已见顶）
    if pd.notna(low_date) and pd.notna(high_date) and high_date < low_date:
        return None

    # 股票代码：取 CSV 文件名（不含扩展名）
    stock_code = os.path.splitext(os.path.basename(csv_path))[0]

    return {
        "股票代码": stock_code,
        "区间最低价": round(float(low_min), 4),
        "最低价日期": low_date.strftime("%Y-%m-%d"),
        "区间最高价": round(float(high_max), 4),
        "最高价日期": high_date.strftime("%Y-%m-%d"),
        "区间最大涨幅": round(float(max_rise), 6) if pd.notna(max_rise) else None,
    }


def scan_specified_period(
    start_date: str = START_DATE,
    end_date: str = END_DATE,
    daily_data_dir: str = DAILY_DATA_DIR,
    ashares_list_path: str = ASHARES_LIST_PATH,
    report_dir: str = REPORT_DIR,
    max_workers: int = 8,
) -> str:
    """
    扫描指定时间段内所有 A 股的最低/最高价及最大涨幅，输出 xlsx 报告。

    Args:
        start_date:        起始日期字符串，格式 ``YYYYMMDD``。
        end_date:          结束日期字符串，格式 ``YYYYMMDD``。
        daily_data_dir:    日线 CSV 目录。
        ashares_list_path: 股票清单 xlsx 路径。
        report_dir:        报告输出目录。
        max_workers:       并发线程数。

    Returns:
        生成的 xlsx 文件绝对路径。
    """
    # ---- 1. 解析时间段 ----
    try:
        start_dt = pd.to_datetime(start_date, format="%Y%m%d")
        end_dt = pd.to_datetime(end_date, format="%Y%m%d")
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"日期格式错误，应为 YYYYMMDD：{exc}") from exc

    if start_dt > end_dt:
        raise ValueError(f"起始日期 {start_date} 晚于结束日期 {end_date}")

    logger.info("扫描区间: %s ~ %s", start_dt.date(), end_dt.date())

    # ---- 2. 加载股票清单 ----
    if not os.path.isfile(ashares_list_path):
        raise FileNotFoundError(f"股票清单不存在: {ashares_list_path}")

    ashares_df = pd.read_excel(ashares_list_path, dtype={"股票代码": str})
    logger.info("股票清单加载完成，共 %d 条记录", len(ashares_df))

    # ---- 3. 遍历日线 CSV ----
    if not os.path.isdir(daily_data_dir):
        raise FileNotFoundError(f"日线数据目录不存在: {daily_data_dir}")

    csv_files = sorted(glob.glob(os.path.join(daily_data_dir, "*.csv")))
    logger.info("待扫描的日线文件数: %d", len(csv_files))

    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(_scan_single_stock, p, start_dt, end_dt): p
            for p in csv_files
        }
        for future in tqdm(
            as_completed(future_map),
            total=len(future_map),
            desc="扫描日线",
            unit="只",
            mininterval=0.5,
        ):
            res = future.result()
            if res is not None:
                results.append(res)

    if not results:
        raise RuntimeError("指定区间内未扫描到任何有效数据，请检查时间段或数据完整性")

    scan_df = pd.DataFrame(results)
    logger.info("扫描结果数: %d", len(scan_df))

    # ---- 4. 与股票清单合并 ----
    final_df = ashares_df.merge(scan_df, on="股票代码", how="left")

    # 按区间最大涨幅降序排列（无效值排到最后）
    final_df = final_df.sort_values(
        by="区间最大涨幅",
        ascending=False,
        na_position="last",
    ).reset_index(drop=True)

    # ---- 5. 输出 xlsx ----
    os.makedirs(report_dir, exist_ok=True)
    out_file = os.path.join(report_dir, f"SpecifiedPeriodScan_{start_date}-{end_date}.xlsx")

    final_df.to_excel(out_file, index=False)

    # ---- 6. 对「区间最大涨幅」列应用百分比格式（保留 1 位小数）----
    # 说明：to_excel 写入的是 float 数值（如 0.4254），
    # 这里用 openpyxl 后处理把单元格显示为百分比（42.5%），底层数值不变。
    # 格式代码 "0.0%" 表示：百分比形式 + 小数点后保留 1 位。
    _apply_percent_format(out_file, column_name="区间最大涨幅", fmt_code="0.0%")

    logger.info("报告输出完成: %s", out_file)

    return out_file


def _apply_percent_format(xlsx_path: str, column_name: str, fmt_code: str) -> None:
    """
    对 xlsx 文件中指定列应用百分比显示格式（不修改底层数值）。

    Args:
        xlsx_path:  xlsx 文件绝对路径。
        column_name: 要设置格式的列名。
        fmt_code:   openpyxl 数字格式代码（如 ``"0.0%"``）。
    """
    wb = load_workbook(xlsx_path)
    ws = wb.active

    # 找到目标列号（1-based）
    header_cells = next(ws.iter_rows(min_row=1, max_row=1, values_only=False))
    col_idx = None
    for cell in header_cells:
        if cell.value == column_name:
            col_idx = cell.column
            break

    if col_idx is None:
        wb.close()
        logger.warning("未找到列 %s，跳过百分比格式化", column_name)
        return

    # 跳过表头，对数据行应用百分比格式
    for row in ws.iter_rows(min_row=2, min_col=col_idx, max_col=col_idx):
        for cell in row:
            if cell.value is not None:
                cell.number_format = fmt_code

    wb.save(xlsx_path)
    wb.close()


# ============================ 入口 ============================
if __name__ == "__main__":
    scan_specified_period()
