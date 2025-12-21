from .utils import SimpleBRB, BRBRule, normalize_feature

def system_level_infer(features):
    # 加强区分：amp 起点更高；freq 起点更低且权重更大；ref 起点更低且稍高权重
    md_amp = max(
        normalize_feature(abs(features["gain"] - 1.0), 0.10, 0.40),
        normalize_feature(features.get("ripple_var", 0.0), 0.10, 0.30),
        normalize_feature(abs(features.get("res_slope", 0.0)), 1e-11, 5e-10),
    )
    md_freq = max(
        normalize_feature(abs(features["df"]), 50.0, 5e4),           # 更低起点
        normalize_feature(features.get("viol_rate", 0.0), 0.003, 0.20),
    )
    md_ref = max(
        normalize_feature(abs(features["bias"]), 0.003, 0.15),
        normalize_feature(features.get("step_score", 0.0), 0.005, 0.25),
    )

    labels = ["幅度失准", "频率失准", "参考电平失准"]
    rules = [
        BRBRule(weight=0.6, belief={"幅度失准": 0.8, "频率失准": 0.1, "参考电平失准": 0.1}),
        BRBRule(weight=1.6, belief={"幅度失准": 0.1, "频率失准": 0.8, "参考电平失准": 0.1}),
        BRBRule(weight=1.3, belief={"幅度失准": 0.1, "频率失准": 0.1, "参考电平失准": 0.8}),
    ]
    brb = SimpleBRB(labels, rules)
    return brb.infer([md_amp, md_freq, md_ref])