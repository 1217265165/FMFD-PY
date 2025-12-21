import numpy as np
from dataclasses import dataclass
from typing import Dict, List

@dataclass
class BRBRule:
    weight: float
    belief: Dict[str, float]  # label -> degree

def normalize_feature(x, low, high, floor=1e-3):
    """线性归一到[0,1]，低于low→floor，高于high→1，留一点最小激活以避免全 0。"""
    if x <= low:
        return floor
    if x >= high:
        return 1.0
    return max(floor, (x - low) / (high - low))

class SimpleBRB:
    """
    简化的ER组合：
    - 若 len(matching_degrees)==len(rules)，每条规则用对应的匹配度（常见一前件一规则）
    - 否则用乘积，乘积为 0 时用均值兜底，避免全 0。
    """
    def __init__(self, labels: List[str], rules: List[BRBRule]):
        self.labels = labels
        self.rules = rules

    def infer(self, matching_degrees: List[float]) -> Dict[str, float]:
        activations = []
        use_one_to_one = len(matching_degrees) == len(self.rules)
        for i, r in enumerate(self.rules):
            if use_one_to_one:
                act_raw = matching_degrees[i]
            else:
                prod = np.prod(matching_degrees)
                act_raw = prod if prod > 1e-6 else float(np.mean(matching_degrees))
            act = r.weight * act_raw
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