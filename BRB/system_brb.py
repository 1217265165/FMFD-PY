from .utils import SimpleBRB, ERBRB, BRBRule, normalize_feature
import math


def system_level_infer_simple(features):
    """
    原始简化版系统级 BRB 推理（保留用于兼容）。
    """
    md_amp = max(
        normalize_feature(abs(features["bias"]), 0.1, 1.0),
        normalize_feature(abs(features["gain"] - 1.0), 0.02, 0.2),
        normalize_feature(abs(features["comp"]), 0.01, 0.1),
    )
    md_freq = normalize_feature(abs(features["df"]), 1e6, 5e7)
    md_rl = max(
        normalize_feature(features["step_score"], 0.2, 1.5),
        normalize_feature(features["viol_rate"], 0.02, 0.2),
    )

    labels = ["幅度失准", "频率失准", "参考电平失准"]
    rules = [
        BRBRule(weight=1.0, belief={"幅度失准": 0.8, "频率失准": 0.1, "参考电平失准": 0.1}),
        BRBRule(weight=1.0, belief={"幅度失准": 0.1, "频率失准": 0.8, "参考电平失准": 0.1}),
        BRBRule(weight=1.0, belief={"幅度失准": 0.1, "频率失准": 0.1, "参考电平失准": 0.8}),
    ]
    brb = SimpleBRB(labels, rules)
    return brb.infer([md_amp, md_freq, md_rl])


def system_level_infer_er(features):
    """
    暂时用 softmax 直接根据连续指标输出三类异常概率，
    目的是先验证 amp_raw/freq_raw/ref_raw 是否有区分度。
    """
    # 1) 连续特征 → [0,1] 指标
    amp_raw = max(
        normalize_feature(abs(features.get("bias", 0.0)), 0.1, 1.0),
        normalize_feature(abs(features.get("gain", 1.0) - 1.0), 0.02, 0.2),
        normalize_feature(abs(features.get("comp", 0.0)), 0.01, 0.1),
    )
    freq_raw = normalize_feature(abs(features.get("df", 0.0)), 1e6, 5e7)
    ref_raw = max(
        normalize_feature(features.get("step_score", 0.0), 0.2, 1.5),
        normalize_feature(features.get("viol_rate", 0.0), 0.02, 0.2),
    )

    print("[DEBUG-ER] amp_raw=", amp_raw, "freq_raw=", freq_raw, "ref_raw=", ref_raw)

    # 2) softmax 放大差异
    alpha = 5.0  # 温度系数，越大越偏向最大者
    xs = [amp_raw, freq_raw, ref_raw]
    es = [math.exp(alpha * x) for x in xs]
    s = sum(es) + 1e-12
    probs = [e / s for e in es]

    result = {
        "幅度失准": probs[0],
        "频率失准": probs[1],
        "参考电平失准": probs[2],
    }
    print("[DEBUG-ER] softmax_probs=", result)
    return result


def system_level_infer(features, mode: str = "er"):
    """
    对外暴露的统一接口。
    mode: "simple" 使用原 SimpleBRB，"er" 使用当前 softmax 版。
    """
    if mode == "simple":
        return system_level_infer_simple(features)
    elif mode == "er":
        return system_level_infer_er(features)
    else:
        raise ValueError(f"Unknown mode for system_level_infer: {mode}")