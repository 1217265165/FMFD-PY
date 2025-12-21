import numpy as np
from dataclasses import dataclass
from typing import Dict, List

@dataclass
class BRBRule:
    weight: float
    belief: Dict[str, float]  # label -> degree

class SimpleBRB:
    """
    简化的ER组合：对所有规则的激活度加权归一，再加权合成后归一。
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

def normalize_feature(x, low, high):
    """线性归一化到[0,1]，低于low→0，高于high→1。"""
    if x <= low:
        return 0.0
    if x >= high:
        return 1.0
    return (x - low) / (high - low)