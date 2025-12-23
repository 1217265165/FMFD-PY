import numpy as np
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class BRBRule:
    weight: float
    belief: Dict[str, float]  # label -> degree


class SimpleBRB:
    """
    现有的简化版 BRB 组合器：
    - 不区分不同规则的匹配度，仅用一个 matching_degrees 列表
    - 通过 (规则权重 × 匹配度乘积) 作为激活度，再对结论做加权平均

    保留这个类，兼容原有 system_brb / module_brb 的用法。
    """
    def __init__(self, labels: List[str], rules: List[BRBRule]):
        self.labels = labels
        self.rules = rules

    def infer(self, matching_degrees: List[float]) -> Dict[str, float]:
        activations = []
        for r in self.rules:
            act = r.weight * np.prod(matching_degrees)
            activations.append((act, r.belief))

        total = sum(a for a, _ in activations) + 1e-9
        out = {lab: 0.0 for lab in self.labels}

        for a, bel in activations:
            for lab in self.labels:
                out[lab] += (a / total) * bel.get(lab, 0.0)

        s = sum(out.values()) + 1e-9
        for lab in self.labels:
            out[lab] = out[lab] / s
        return out


class ERBRB:
    """
    更接近论文 2.2 节“基于证据推理的递推合成”形式的 BRB 组合器。

    特点：
    - 每条规则有自己的匹配度向量 alpha_k^i（i 为属性）
    - 计算规则激活权重 w_k（由属性匹配度 + 规则权重决定）
    - 显式引入“无知项” u_k = 1 - sum_j beta_{k,j}
    - 按规则逐条递推合成，并计算简化版冲突系数 K

    使用方式（后续我们会在 system_brb.py / module_brb.py 里逐步替换 SimpleBRB）：
        brb = ERBRB(labels, rules)
        # matching_degrees_per_rule: 长度 = 规则数，每个元素是该规则各属性的匹配度列表
        result = brb.infer(matching_degrees_per_rule)
    """

    def __init__(self, labels: List[str], rules: List[BRBRule]):
        self.labels = labels
        self.rules = rules

    def infer(self, matching_degrees_per_rule: List[List[float]]) -> Dict[str, float]:
        """
        Parameters
        ----------
        matching_degrees_per_rule : List[List[float]]
            形如:
                [
                    [alpha_1^1, alpha_1^2, ..., alpha_1^M],  # 第1条规则的属性匹配度
                    [alpha_2^1, alpha_2^2, ..., alpha_2^M],  # 第2条规则
                    ...
                ]
            - 长度 = 规则条数 K
            - 每个子列表长度 = 属性个数 M

        Returns
        -------
        Dict[str, float]
            对各结论标签的最终置信度分布，和为 1。
        """
        assert len(matching_degrees_per_rule) == len(self.rules), \
            "matching_degrees_per_rule 的长度必须等于规则数"

        K_rules = len(self.rules)
        J_labels = len(self.labels)

        # -------- 1) 规则激活权重 w_k 计算 --------
        act_list = []
        for r, alphas in zip(self.rules, matching_degrees_per_rule):
            # 这里采用属性匹配度的乘积作为综合匹配度 m_k
            # 若后续需要引入属性权重，可改为加权几何平均
            m_k = float(np.prod(alphas))
            act_list.append(r.weight * m_k)

        act_arr = np.array(act_list, dtype=float)
        if act_arr.sum() <= 1e-12:
            # 若全部为 0，退化为各规则均匀激活
            act_arr[:] = 1.0 / K_rules
        else:
            act_arr = act_arr / act_arr.sum()  # 归一化为规则激活权重

        # -------- 2) 每条规则的证据分配 beta_kj 与无知项 u_k --------
        beta_list = []  # shape: (K, J)
        u_list = []     # shape: (K,)
        for k, r in enumerate(self.rules):
            beta_k = np.array(
                [r.belief.get(lab, 0.0) for lab in self.labels],
                dtype=float
            )
            s = beta_k.sum()
            if s > 1.0:
                # 若专家给出的置信度和超过 1，则整体压缩到 1
                beta_k = beta_k / s
                s = 1.0
            u_k = max(0.0, 1.0 - s)  # 无知项
            beta_list.append(beta_k)
            u_list.append(u_k)

        beta_arr = np.vstack(beta_list)        # (K, J)
        u_arr = np.array(u_list, dtype=float)  # (K,)

        # -------- 3) 递推合成规则 --------
        # 初始化：先用第 1 条规则
        m = act_arr[0] * beta_arr[0]   # 对每个 label 的支持度分配
        u = act_arr[0] * u_arr[0]      # 无知项

        for k in range(1, K_rules):
            m_k = act_arr[k] * beta_arr[k]
            u_k = act_arr[k] * u_arr[k]

            # 冲突系数 K（简化版，只考虑结论冲突）
            conflict = float(np.sum(m * m_k))
            K = 1.0 - conflict

            # 合成（简化 ER 公式）
            new_m = (m * (m_k + u_k) + m_k * u) / (K + 1e-9)
            new_u = (u * u_k) / (K + 1e-9)

            m, u = new_m, new_u

        # -------- 4) 归一化得到最终置信度 --------
        s_total = float(m.sum() + u)
        if s_total <= 1e-12:
            # 若完全退化，则返回均匀分布
            return {lab: 1.0 / J_labels for lab in self.labels}

        m = m / s_total
        return {lab: float(v) for lab, v in zip(self.labels, m)}


def normalize_feature(x, low, high):
    """线性归一化到 [0, 1]，低于 low→0，高于 high→1。"""
    if x <= low:
        return 0.0
    if x >= high:
        return 1.0
    return (x - low) / (high - low)