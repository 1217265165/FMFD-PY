import pandas as pd
from FMFD.baseline.baseline import load_and_align, compute_rrs_bounds
from FMFD.baseline.config import BAND_RANGES, K_LIST
from FMFD.simulation.dataset import simulate_fault_dataset
from FMFD.BRB.system_brb import system_level_infer
from FMFD.BRB.module_brb import module_level_infer
from FMFD.features.extract import extract_system_features

def main():
    # 1) 基线
    freq, traces, _ = load_and_align("./normal_response_data")
    rrs, bounds = compute_rrs_bounds(freq, traces, BAND_RANGES, K_LIST)
    baseline = {
        "frequency": freq,
        "traces": traces,
        "rrs": rrs,
        "bounds": bounds,
        "band_ranges": BAND_RANGES,
    }

    # 2) 仿真故障数据集（含系统特征）
    df = simulate_fault_dataset(baseline, n_samples=150, seed=2025)

    # 3) BRB 两层推理
    sys_probs_list = []
    mod_probs_list = []
    for _, row in df.iterrows():
        feats = {k: row[k] for k in ["gain", "bias", "comp", "df", "viol_rate", "step_score", "res_slope", "ripple_var"]}
        sys_p = system_level_infer(feats)
        mod_p = module_level_infer(feats, sys_p)
        sys_probs_list.append(sys_p)
        mod_probs_list.append(mod_p)

    # 合并结果
    sys_df = pd.DataFrame(sys_probs_list).add_prefix("sys_")
    mod_df = pd.DataFrame(mod_probs_list).add_prefix("mod_")
    out_df = pd.concat([df, sys_df, mod_df], axis=1)

    out_csv = "sim_fault_dataset.csv"
    out_df.to_csv(out_csv, index=False)
    print(f"仿真+BRB结果已保存: {out_csv}")

    # 可选：打印均值便于观察
    print("系统级异常概率均值：")
    print(sys_df.mean())
    print("模块级故障概率均值：")
    print(mod_df.mean())

if __name__ == "__main__":
    main()