"""
Comparison methods for hierarchical BRB fault diagnosis.

This module implements baseline methods from the literature for comparison:
- HCF (Zhang et al., 2022): Hierarchical Cognitive Framework
- BRB-P (Ming et al., 2023): BRB with Probability constraint
- ER-c (Zhang et al., 2024): Enhanced Reasoning with credibility assessment
"""

from .hcf import HCFMethod
from .brb_p import BRBPMethod
from .er_c import ERCMethod

__all__ = ['HCFMethod', 'BRBPMethod', 'ERCMethod']
