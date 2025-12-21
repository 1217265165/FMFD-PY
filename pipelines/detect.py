import os
import json
import glob
import numpy as np
import pandas as pd
from FMFD.baseline.baseline import align_to_frequency
from FMFD.baseline.config import (
    BASELINE_ARTIFACTS, BASELINE_META, BAND_RANGES,
    OUTPUT_DIR, DETECTION_RESULTS
)
from FMFD.features.extract import extract_system_features
from FMFD.BRB.system_brb import system_level_infer
from FMFD.BRB.module_brb import module_level_infer

def load_thresholds(path="thresholds.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def apply_thresholds(features, thresholds):
    """
    返回告警标志 dict：warn / alarm / ok，双阈值策略。
    """
    flags = {}
    for k, v in thresholds.items():
        val = features.get(k, None)
        if val is None:
            continue
        low = v.get("warn", None)
        high = v.get("alarm", None)
        if high is not None and abs(val) >= high:
            flags[k] = "alarm"
        elif low is not None and abs(val) >= low:
            flags[k] = "warn"
        else:
            flags[k] = "ok"
    return flags

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1) 加载基线
    art = np.load(BASELINE_ARTIFACTS)
    frequency = art["frequency"]
    rrs = art["rrs"]
    bounds = (art["upper"], art["lower"])
    with open(BASELINE_META, "r", encoding="utf-8") as f:
        meta = json.load(f)
    band_ranges = meta.get("band_ranges", BAND_RANGES)

    # 2) 加载阈值
    thresholds = load_thresholds("../thresholds.json")

    # 3) 读取待检数据（to_detect 目录下的 csv）
    files = glob.glob("./to_detect/*.csv")
    if not files:
        raise FileNotFoundError("to_detect 目录下未找到待检 CSV")

    rows = []
    for fpath in files:
        df = pd.read_csv(fpath)
        if df.shape[1] < 2:
            continue
        freq_raw = df.iloc[:, 0].values
        amp_raw = df.iloc[:, 1].values
        amp = align_to_frequency(frequency, freq_raw, amp_raw)

        # 特征
        feats = extract_system_features(frequency, rrs, bounds, band_ranges, amp)
        # BRB
        sys_probs = system_level_infer(feats)
        mod_probs = module_level_infer(feats, sys_probs)
        # 阈值判别
        flags = apply_thresholds(feats, thresholds)

        row = {
            "file": fpath,
            **feats,
            **{f"sys_{k}": v for k, v in sys_probs.items()},
            **{f"mod_{k}": v for k, v in mod_probs.items()},
            **{f"flag_{k}": v for k, v in flags.items()},
        }
        rows.append(row)

    out_df = pd.DataFrame(rows)
    out_df.to_csv(DETECTION_RESULTS, index=False)
    print(f"检测结果已保存: {DETECTION_RESULTS}")

if __name__ == "__main__":
    main()