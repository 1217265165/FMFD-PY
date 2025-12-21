import os
import json
import numpy as np
import pandas as pd
from FMFD.baseline.baseline import load_and_align, compute_rrs_bounds, detect_switch_steps
from FMFD.baseline.config import (
    BAND_RANGES, K_LIST, SWITCH_TOL,
    BASELINE_ARTIFACTS, BASELINE_META,
    NORMAL_FEATURE_STATS, SWITCH_CSV, SWITCH_JSON, PLOT_PATH,
    OUTPUT_DIR
)
from FMFD.baseline.viz import plot_rrs_envelope_switch
from FMFD.features.extract import extract_system_features

def main():
    # 确保输出目录存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1) 加载并对齐正常数据
    folder_path = "../normal_response_data"  # 按你的要求改为上一层目录
    frequency, traces, names = load_and_align(folder_path)

    # 2) 计算 RRS 与分段包络
    rrs, bounds = compute_rrs_bounds(frequency, traces, BAND_RANGES, K_LIST)

    # 3) 切换点步进
    switch_feats = detect_switch_steps(frequency, traces, BAND_RANGES, tol=SWITCH_TOL)

    # 4) 可视化
    plot_rrs_envelope_switch(frequency, traces, rrs, bounds, switch_feats, PLOT_PATH)

    # 5) 保存基线产物
    np.savez(BASELINE_ARTIFACTS, frequency=frequency, rrs=rrs, upper=bounds[0], lower=bounds[1])
    with open(BASELINE_META, "w", encoding="utf-8") as f:
        json.dump({"band_ranges": BAND_RANGES, "k_list": K_LIST}, f, ensure_ascii=False, indent=2)

    # 6) 保存切换点特性
    pd.DataFrame(switch_feats).to_csv(SWITCH_CSV, index=False)
    with open(SWITCH_JSON, "w", encoding="utf-8") as f:
        json.dump(switch_feats, f, indent=4, ensure_ascii=False)

    # 7) 正常特征统计（用于阈值初设）
    feats_list = []
    for i in range(traces.shape[0]):
        amp = traces[i]
        feats = extract_system_features(frequency, rrs, bounds, BAND_RANGES, amp)
        feats_list.append(feats)
    stats_df = pd.DataFrame(feats_list)
    stats_df.describe(percentiles=[0.5, 0.9, 0.95, 0.99]).to_csv(NORMAL_FEATURE_STATS)

    print("基线包络与RRS已保存:", BASELINE_ARTIFACTS, BASELINE_META)
    print("切换点特性已保存:", SWITCH_CSV, SWITCH_JSON)
    print("正常特征统计已保存:", NORMAL_FEATURE_STATS)

if __name__ == "__main__":
    main()