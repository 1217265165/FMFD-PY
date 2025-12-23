#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
示例：如何使用对比方法进行单个样本推理

这个脚本展示如何使用各个对比方法对单个样本进行诊断推理。
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from comparison import HCFMethod, BRBPMethod, ERCMethod
from BRB.system_brb import system_level_infer
from BRB.module_brb import module_level_infer


def demo_single_sample():
    """演示单个样本的推理过程"""
    
    # 1. 准备示例特征（模拟一个幅度失准的故障）
    features = {
        "bias": 0.5,          # 幅度偏置较大
        "gain": 1.2,          # 增益偏离1.0
        "comp": 0.05,         # 非线性压缩
        "df": 1e6,            # 频率偏移较小
        "viol_rate": 0.15,    # 包络越界率
        "step_score": 0.8,    # 切换点步进异常
        "res_slope": 1e-11,   # 残差斜率
        "ripple_var": 0.01,   # 纹波方差
        "switch_step_err_max": 0.3,
        "nonswitch_step_max": 0.2,
    }
    
    print("="*70)
    print("示例特征（模拟幅度失准故障）")
    print("="*70)
    for k, v in features.items():
        print(f"  {k:25s}: {v}")
    
    # 2. 使用各个方法进行推理
    methods = {
        "HCF (Zhang 2022)": HCFMethod(),
        "BRB-P (Ming 2023)": BRBPMethod(),
        "ER-c (Zhang 2024)": ERCMethod(),
    }
    
    print("\n" + "="*70)
    print("系统级诊断结果对比")
    print("="*70)
    
    # 对比方法推理
    for method_name, method in methods.items():
        sys_result = method.infer_system(features)
        print(f"\n{method_name}:")
        print(f"  规则数: {method.get_num_rules()}, "
              f"参数数: {method.get_num_parameters()}, "
              f"特征维度: {method.get_feature_dimension()}")
        print("  诊断结果:")
        for label, prob in sys_result.items():
            print(f"    {label}: {prob:.4f}")
    
    # 本文方法
    print(f"\n本文方法 (Knowledge-driven Hierarchical BRB):")
    print(f"  规则数: 45, 参数数: 38, 特征维度: 4")
    sys_result_proposed = system_level_infer(features, mode="er")
    print("  诊断结果:")
    for label, prob in sys_result_proposed.items():
        print(f"    {label}: {prob:.4f}")
    
    # 3. 模块级推理示例（使用本文方法）
    print("\n" + "="*70)
    print("模块级诊断结果（本文方法）")
    print("="*70)
    
    module_result = module_level_infer(features, sys_result_proposed)
    # 显示概率最高的前5个模块
    sorted_modules = sorted(module_result.items(), key=lambda x: x[1], reverse=True)
    print("\n故障概率最高的5个模块:")
    for i, (module, prob) in enumerate(sorted_modules[:5], 1):
        print(f"  {i}. {module:30s}: {prob:.4f}")


def demo_parameter_comparison():
    """演示各方法的参数规模对比"""
    
    print("\n" + "="*70)
    print("方法参数规模对比")
    print("="*70)
    
    hcf = HCFMethod()
    brb_p = BRBPMethod()
    er_c = ERCMethod()
    
    methods_info = [
        ("HCF (Zhang 2022)", hcf.get_num_rules(), hcf.get_num_parameters(), 
         hcf.get_feature_dimension()),
        ("BRB-P (Ming 2023)", brb_p.get_num_rules(), brb_p.get_num_parameters(),
         brb_p.get_feature_dimension()),
        ("ER-c (Zhang 2024)", er_c.get_num_rules(), er_c.get_num_parameters(),
         er_c.get_feature_dimension()),
        ("本文方法", 45, 38, 4),
    ]
    
    print(f"\n{'方法':<25s} {'规则数':<10s} {'参数数':<10s} {'特征维度':<10s}")
    print("-" * 70)
    for name, rules, params, dims in methods_info:
        print(f"{name:<25s} {rules:<10d} {params:<10d} {dims:<10d}")
    
    # 计算本文方法的改进比例
    print("\n本文方法相对于HCF的改进:")
    hcf_rules, hcf_params, hcf_dims = methods_info[0][1:]
    prop_rules, prop_params, prop_dims = methods_info[3][1:]
    
    rule_reduction = (hcf_rules - prop_rules) / hcf_rules * 100
    param_reduction = (hcf_params - prop_params) / hcf_params * 100
    dim_reduction = (hcf_dims - prop_dims) / hcf_dims * 100
    
    print(f"  规则数减少: {rule_reduction:.1f}%")
    print(f"  参数数减少: {param_reduction:.1f}%")
    print(f"  特征维度减少: {dim_reduction:.1f}%")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("分层BRB方法对比演示")
    print("="*70)
    
    # 运行演示
    demo_single_sample()
    demo_parameter_comparison()
    
    print("\n" + "="*70)
    print("✓ 演示完成")
    print("="*70)
    print("\n提示:")
    print("  - 运行完整对比评估: python -m pipelines.compare_methods")
    print("  - 查看对比文档: comparison/README.md")
    print()
