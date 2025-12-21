from .utils import SimpleBRB, BRBRule, normalize_feature

def system_level_infer(features):
    """
    第一层 BRB：输出系统级三类异常概率。
    features 需包含: gain, bias, comp, df, step_score, viol_rate
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