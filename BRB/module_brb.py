import numpy as np
from .utils import SimpleBRB, BRBRule, normalize_feature

def module_level_infer(features, sys_probs):
    """
    第二层 BRB：输出 21 个模块的故障概率（与 brb_rules.yaml 的 modules_order 一致）。
    依赖特征: step_score, res_slope, ripple_var, df, viol_rate, gain, bias,
             switch_step_err_max, nonswitch_step_max (新增)
    """
    md_step_raw = max(
        features["step_score"],
        features.get("switch_step_err_max", 0.0),
        features.get("nonswitch_step_max", 0.0),
    )
    md_step = normalize_feature(md_step_raw, 0.2, 1.5)
    md_slope = normalize_feature(abs(features["res_slope"]), 1e-12, 1e-10)
    md_ripple = normalize_feature(features["ripple_var"], 0.001, 0.02)
    md_df = normalize_feature(abs(features["df"]), 1e6, 5e7)
    md_viol = normalize_feature(features["viol_rate"], 0.02, 0.2)
    md_gain_bias = max(
        normalize_feature(abs(features["bias"]), 0.1, 1.0),
        normalize_feature(abs(features["gain"] - 1.0), 0.02, 0.2),
    )
    md = np.mean([md_step, md_slope, md_ripple, md_df, md_viol, md_gain_bias])

    labels = [
        "衰减器",
        "前置放大器",
        "低频段前置低通滤波器",
        "低频段第一混频器",
        "高频段YTF滤波器",
        "高频段混频器",
        "时钟振荡器",
        "时钟合成与同步网络",
        "本振源（谐波发生器）",
        "本振混频组件",
        "校准源",
        "存储器",
        "校准信号开关",
        "中频放大器",
        "ADC",
        "数字RBW",
        "数字放大器",
        "数字检波器",
        "VBW滤波器",
        "电源模块",
        "未定义/其他",
    ]

    rules = [
        BRBRule(weight=0.8 * sys_probs.get("参考电平失准", 0.3),
                belief={"衰减器": 0.60, "校准源": 0.08, "存储器": 0.06, "校准信号开关": 0.16, "未定义/其他": 0.10}),
        BRBRule(weight=0.6 * sys_probs.get("幅度失准", 0.3),
                belief={"前置放大器": 0.40, "中频放大器": 0.25, "数字放大器": 0.20, "衰减器": 0.10, "ADC": 0.05}),
        BRBRule(weight=0.7 * sys_probs.get("频率失准", 0.3),
                belief={"时钟振荡器": 0.35, "时钟合成与同步网络": 0.35, "本振源（谐波发生器）": 0.15, "本振混频组件": 0.15}),
        BRBRule(weight=0.5, belief={"高频段YTF滤波器": 0.60, "高频段混频器": 0.40}),
        BRBRule(weight=0.5, belief={"低频段前置低通滤波器": 0.60, "低频段第一混频器": 0.40}),
        BRBRule(weight=0.4, belief={"数字RBW": 0.30, "数字检波器": 0.35, "VBW滤波器": 0.25, "ADC": 0.10}),
        BRBRule(weight=0.3, belief={"电源模块": 0.80, "未定义/其他": 0.20}),
        BRBRule(weight=0.2, belief={"未定义/其他": 1.0}),
    ]

    brb = SimpleBRB(labels, rules)
    return brb.infer([md])