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
    改进的系统级BRB推理，增加正常状态检测。
    
    改进点：
    1. 添加正常状态判别逻辑
    2. 调整特征归一化阈值以更好适应数据分布
    3. 降低softmax温度系数，提高不确定性表达能力
    4. 使用综合阈值判断是否为正常状态
    """
    # 1) 连续特征 → [0,1] 指标，调整阈值范围
    # 幅度相关指标
    bias_score = normalize_feature(abs(features.get("bias", 0.0)), 0.05, 0.5)
    gain_score = normalize_feature(abs(features.get("gain", 1.0) - 1.0), 0.01, 0.15)
    comp_score = normalize_feature(abs(features.get("comp", 0.0)), 0.005, 0.08)
    amp_raw = max(bias_score, gain_score, comp_score)
    
    # 频率相关指标
    freq_raw = normalize_feature(abs(features.get("df", 0.0)), 5e5, 3e7)
    
    # 参考电平相关指标
    step_score = normalize_feature(features.get("step_score", 0.0), 0.1, 1.0)
    viol_score = normalize_feature(features.get("viol_rate", 0.0), 0.01, 0.15)
    ref_raw = max(step_score, viol_score)

    # 2) 正常状态检测：所有指标都较低时判为正常
    # 使用加权平均而非最大值，更能反映整体状态
    overall_score = 0.4 * amp_raw + 0.3 * freq_raw + 0.3 * ref_raw
    normal_threshold = 0.15  # 综合得分低于此阈值判为正常
    
    if overall_score < normal_threshold:
        # 正常状态：返回低置信度的均匀分布
        result = {
            "幅度失准": overall_score / 3,
            "频率失准": overall_score / 3,
            "参考电平失准": overall_score / 3,
        }
        return result
    
    # 3) 故障状态：使用softmax分配到具体故障类型
    # 降低温度系数，允许表达更多不确定性
    alpha = 2.0  # 原来是5.0，降低后不会过度偏向单一类别
    xs = [amp_raw, freq_raw, ref_raw]
    es = [math.exp(alpha * x) for x in xs]
    s = sum(es) + 1e-12
    probs = [e / s for e in es]

    result = {
        "幅度失准": probs[0],
        "频率失准": probs[1],
        "参考电平失准": probs[2],
    }
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