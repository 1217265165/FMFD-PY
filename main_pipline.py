"""
main_pipeline.py

端到端主脚本（使用新模块列表和系统级异常：幅度失准、频率失准、参考电平失准）
流程：
1) 采集（或读取现有 RAW_OUTPUT_CSV）
2) 特征提取（compute_feature_matrix）
3) 初始 BRB 推理（使用 brb_rules.yaml）
4) 可选：CMA-ES 规则优化（optimize_brb）

用法示例:
python -m FMFD.main_pipeline --collect   # 若需采集
python -m FMFD.main_pipeline             # 直接用已有 RAW_OUTPUT_CSV
"""
import argparse
import pandas as pd
from data_acquisition import acquire_sequence
from feature_extraction import compute_feature_matrix
from brb_engine import load_rules, inference
from optimize_brb import optimize_rules
from instruments_config import RAW_OUTPUT_CSV, FEATURE_OUTPUT_PREFIX, RULES_YAML
import os

def main(collect=False, optimize=False, supervised=False, label_col=None):
    # 1) 采集或读取已有原始数据
    if collect:
        raw = acquire_sequence()
    else:
        raw = pd.read_csv(RAW_OUTPUT_CSV, encoding="utf-8")

    # 2) 特征提取
    feat_df, module_meta = compute_feature_matrix(raw)
    feat_csv = f"{FEATURE_OUTPUT_PREFIX}_features.csv"
    module_csv = f"{FEATURE_OUTPUT_PREFIX}_module_meta.csv"
    feat_df.to_csv(feat_csv, index=False, encoding="utf-8-sig")
    module_meta.to_csv(module_csv, index=False, encoding="utf-8-sig")
    print("Feature matrix saved:", feat_csv)
    print("Module meta saved:", module_csv)

    # 3) 初始 BRB 推理（只用 RULES_YAML，避免与旧规则/自动规则冲突）
    rules_doc = load_rules(RULES_YAML)
    indicator_map = {}
    for r in rules_doc["rules"].keys():
        col = f"{r}__activation"
        if col in feat_df.columns:
            indicator_map[r] = col
        else:
            # fallback：名称包含匹配
            fallback = None
            for c in feat_df.columns:
                if r in c:
                    fallback = c; break
            if fallback:
                indicator_map[r] = fallback
    print("Indicator mapping:", indicator_map)

    out_init = inference(feat_df, rules_doc, indicator_map)
    out_init.to_csv("brb_initial_outputs.csv", index=False, encoding="utf-8-sig")
    print("Saved brb_initial_outputs.csv")

    # 4) 可选优化（CMA-ES）
    if optimize:
        out_path, params_opt, best_obj = optimize_rules(RULES_YAML, feat_df, indicator_map, label_col=label_col, supervised=supervised, maxiter=80, popsize=16)
        rules_opt = load_rules(out_path)
        out_opt = inference(feat_df, rules_opt, indicator_map)
        out_opt.to_csv("brb_optimized_outputs.csv", index=False, encoding="utf-8-sig")
        print("Saved brb_optimized_outputs.csv")
        print("Optimized params:", params_opt)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--collect", action="store_true", help="是否执行采集")
    p.add_argument("--optimize", action="store_true", help="是否执行优化")
    p.add_argument("--supervised", action="store_true", help="是否有监督优化（需 label_col）")
    p.add_argument("--label_col", default=None, help="若 supervised=True，指定标签列名")
    args = p.parse_args()
    main(collect=args.collect, optimize=args.optimize, supervised=args.supervised, label_col=args.label_col)