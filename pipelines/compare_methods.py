#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compare_methods.py

Comparison evaluation of hierarchical BRB methods:
- HCF (Zhang et al., 2022)
- BRB-P (Ming et al., 2023)
- ER-c (Zhang et al., 2024)
- Proposed hierarchical method (Current implementation)

This script:
1. Loads simulated fault data
2. Runs all comparison methods
3. Calculates performance metrics
4. Generates comparison tables and visualizations
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix

# Import comparison methods
from FMFD.comparison.hcf import HCFMethod
from FMFD.comparison.brb_p import BRBPMethod
from FMFD.comparison.er_c import ERCMethod
from FMFD.BRB.system_brb import system_level_infer
from FMFD.BRB.module_brb import module_level_infer

# Chinese font support
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial"]
plt.rcParams["axes.unicode_minus"] = False


class MethodComparison:
    """Compare different hierarchical BRB methods."""
    
    def __init__(self, data_dir: Path = None, output_dir: Path = None):
        if data_dir is None:
            repo_root = Path(__file__).resolve().parents[1]
            self.data_dir = repo_root / "Output" / "sim_spectrum"
        else:
            self.data_dir = Path(data_dir)
        
        # Separate output directory for comparison results
        if output_dir is None:
            repo_root = Path(__file__).resolve().parents[1]
            self.output_dir = repo_root / "Output" / "comparison_results"
        else:
            self.output_dir = Path(output_dir)
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.methods = {
            "HCF (Zhang 2022)": HCFMethod(),
            "BRB-P (Ming 2023)": BRBPMethod(),
            "ER-c (Zhang 2024)": ERCMethod(),
            "本文方法": None  # Use existing implementation
        }
        
        self.results = {}
        
    def load_data(self) -> Tuple[pd.DataFrame, Dict]:
        """Load features and labels from simulation results."""
        feats_path = self.data_dir / "features_brb.csv"
        labels_path = self.data_dir / "labels.json"
        
        if not feats_path.exists() or not labels_path.exists():
            raise FileNotFoundError(
                f"Data files not found in {self.data_dir}. "
                "Please run simulation first: python -m FMFD.pipelines.simulate.run_sinulation_brb"
            )
        
        df = pd.read_csv(feats_path, encoding="utf-8-sig")
        with open(labels_path, "r", encoding="utf-8") as f:
            labels = json.load(f)
        
        return df, labels
    
    def extract_features(self, row: pd.Series) -> Dict[str, float]:
        """Extract features from data row."""
        return {
            "bias": row.get("bias", 0.0),
            "gain": row.get("gain", 1.0),
            "comp": row.get("comp", 0.0),
            "df": row.get("df", 0.0),
            "viol_rate": row.get("viol_rate", 0.0),
            "step_score": row.get("step_score", 0.0),
            "res_slope": row.get("res_slope", 0.0),
            "ripple_var": row.get("ripple_var", 0.0),
            "switch_step_err_max": row.get("switch_step_err_max", 0.0),
            "nonswitch_step_max": row.get("nonswitch_step_max", 0.0),
        }
    
    def map_system_label(self, sample_id: str, labels_json: Dict) -> str:
        """Map sample to system-level label."""
        info = labels_json.get(sample_id, {})
        if not isinstance(info, dict):
            return "未知"
        
        if info.get("type") == "normal":
            return "正常"
        
        sys_fault = info.get("system_fault_class", "")
        if sys_fault == "amp_error":
            return "幅度失准"
        if sys_fault == "freq_error":
            return "频率失准"
        if sys_fault == "ref_error":
            return "参考电平失准"
        
        return "未知"
    
    def evaluate_method(self, method_name: str, df: pd.DataFrame,
                       labels_json: Dict) -> Dict:
        """Evaluate a single method."""
        print(f"\n{'='*60}")
        print(f"评估方法: {method_name}")
        print(f"{'='*60}")
        
        method = self.methods[method_name]
        predictions = []
        true_labels = []
        inference_times = []
        
        for idx, row in df.iterrows():
            sample_id = str(row["sample_id"])
            true_label = self.map_system_label(sample_id, labels_json)
            
            if true_label == "未知":
                continue
            
            features = self.extract_features(row)
            
            # Measure inference time
            start_time = time.perf_counter()
            
            if method_name == "本文方法":
                # Use existing implementation
                sys_probs = system_level_infer(features, mode="er")
            else:
                # Use comparison method
                sys_probs = method.infer_system(features)
            
            end_time = time.perf_counter()
            inference_times.append((end_time - start_time) * 1000)  # ms
            
            # Get prediction
            if true_label == "正常":
                # For normal samples, use improved threshold
                max_prob = max(sys_probs.values())
                # Lower threshold to 0.3 for better normal detection
                pred_label = "正常" if max_prob < 0.3 else max(sys_probs, key=sys_probs.get)
            else:
                # For fault samples, predict the fault type with highest probability
                max_prob = max(sys_probs.values())
                # Only predict fault if confidence is above threshold
                if max_prob < 0.3:
                    pred_label = "正常"
                else:
                    pred_label = max(sys_probs, key=sys_probs.get)
            
            predictions.append(pred_label)
            true_labels.append(true_label)
        
        # Calculate metrics
        accuracy = accuracy_score(true_labels, predictions)
        
        # Calculate per-class accuracy
        label_order = ["正常", "幅度失准", "频率失准", "参考电平失准"]
        cm = confusion_matrix(true_labels, predictions, labels=label_order)
        
        per_class_acc = {}
        for i, label in enumerate(label_order):
            if cm[i].sum() > 0:
                per_class_acc[label] = cm[i, i] / cm[i].sum()
            else:
                per_class_acc[label] = 0.0
        
        # Get method statistics
        if method_name == "本文方法":
            num_rules = 45  # As per paper
            num_params = 38
            feature_dim = 4
        else:
            num_rules = method.get_num_rules()
            num_params = method.get_num_parameters()
            feature_dim = method.get_feature_dimension()
        
        avg_inference_time = np.mean(inference_times)
        
        results = {
            "method_name": method_name,
            "total_samples": len(true_labels),
            "accuracy": accuracy,
            "per_class_accuracy": per_class_acc,
            "num_rules": num_rules,
            "num_parameters": num_params,
            "feature_dimension": feature_dim,
            "avg_inference_time_ms": avg_inference_time,
            "confusion_matrix": cm,
            "label_order": label_order
        }
        
        print(f"总样本数: {len(true_labels)}")
        print(f"准确率: {accuracy:.4f}")
        print(f"规则数: {num_rules}")
        print(f"参数数: {num_params}")
        print(f"特征维度: {feature_dim}")
        print(f"平均推理时间: {avg_inference_time:.2f} ms")
        
        return results
    
    def run_comparison(self):
        """Run comparison for all methods."""
        df, labels_json = self.load_data()
        
        for method_name in self.methods.keys():
            try:
                result = self.evaluate_method(method_name, df, labels_json)
                self.results[method_name] = result
            except Exception as e:
                print(f"错误: 评估 {method_name} 时出错: {e}")
                import traceback
                traceback.print_exc()
    
    def generate_comparison_table(self, output_path: Path = None):
        """Generate comparison table (Table 3-2 style)."""
        if output_path is None:
            output_path = self.output_dir / "comparison_table.csv"
        
        # Prepare table data
        table_data = []
        
        for method_name, result in self.results.items():
            # Calculate reduction percentages for proposed method
            if method_name == "本文方法":
                # Find HCF as baseline
                hcf_result = self.results.get("HCF (Zhang 2022)", {})
                if hcf_result:
                    rule_reduction = (
                        (hcf_result["num_rules"] - result["num_rules"]) /
                        hcf_result["num_rules"] * 100
                    )
                    param_reduction = (
                        (hcf_result["num_parameters"] - result["num_parameters"]) /
                        hcf_result["num_parameters"] * 100
                    )
                    feature_reduction = (
                        (hcf_result["feature_dimension"] - result["feature_dimension"]) /
                        hcf_result["feature_dimension"] * 100
                    )
                else:
                    rule_reduction = param_reduction = feature_reduction = 0
                
                rules_str = f"{result['num_rules']}↓{rule_reduction:.0f}%"
                params_str = f"{result['num_parameters']}↓{param_reduction:.0f}%"
                features_str = f"{result['feature_dimension']}↓{feature_reduction:.0f}%"
            else:
                rules_str = str(result['num_rules'])
                params_str = str(result['num_parameters'])
                features_str = str(result['feature_dimension'])
            
            table_data.append({
                "方法": method_name,
                "总规则数": rules_str,
                "参数总数": params_str,
                "系统级特征维度": features_str,
                "诊断准确率": f"{result['accuracy']:.4f}",
                "平均推理时间(ms)": f"{result['avg_inference_time_ms']:.2f}",
                "样本数": result['total_samples']
            })
        
        df_table = pd.DataFrame(table_data)
        df_table.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"\n比较表已保存到: {output_path}")
        
        return df_table
    
    def generate_performance_table(self, output_path: Path = None):
        """Generate performance breakdown table (Table 3-3 style)."""
        if output_path is None:
            output_path = self.output_dir / "performance_table.csv"
        
        # Prepare per-class performance data
        table_data = []
        
        for method_name, result in self.results.items():
            for label, acc in result['per_class_accuracy'].items():
                table_data.append({
                    "方法": method_name,
                    "异常类型": label,
                    "准确率": f"{acc:.4f}"
                })
        
        df_perf = pd.DataFrame(table_data)
        df_perf.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"性能表已保存到: {output_path}")
        
        return df_perf
    
    def plot_comparison(self, output_path: Path = None):
        """Plot accuracy vs rules comparison (Figure 3-4 style)."""
        if output_path is None:
            output_path = self.output_dir / "comparison_plot.png"
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Extract data for plotting
        methods = []
        accuracies = []
        num_rules = []
        
        for method_name, result in self.results.items():
            methods.append(method_name)
            accuracies.append(result['accuracy'] * 100)  # Convert to percentage
            num_rules.append(result['num_rules'])
        
        # Create scatter plot
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        for i, (method, acc, rules) in enumerate(zip(methods, accuracies, num_rules)):
            ax.scatter(rules, acc, s=200, c=colors[i], alpha=0.7,
                      edgecolors='black', linewidth=1.5, label=method)
        
        ax.set_xlabel('规则数量', fontsize=12)
        ax.set_ylabel('诊断准确率 (%)', fontsize=12)
        ax.set_title('准确率-规则数权衡关系', fontsize=14, fontweight='bold')
        ax.legend(loc='lower left', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"比较图已保存到: {output_path}")
    
    def plot_confusion_matrices(self, output_path: Path = None):
        """Plot confusion matrices for all methods."""
        if output_path is None:
            output_path = self.output_dir / "confusion_matrices.png"
        
        n_methods = len(self.results)
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        axes = axes.flatten()
        
        for idx, (method_name, result) in enumerate(self.results.items()):
            ax = axes[idx]
            cm = result['confusion_matrix']
            labels = result['label_order']
            
            # Plot confusion matrix
            im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
            ax.set_title(f'{method_name}\n准确率: {result["accuracy"]:.4f}', 
                        fontsize=11, fontweight='bold')
            
            # Add colorbar
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            
            # Set ticks
            tick_marks = np.arange(len(labels))
            ax.set_xticks(tick_marks)
            ax.set_yticks(tick_marks)
            ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
            ax.set_yticklabels(labels, fontsize=9)
            
            # Add text annotations
            thresh = cm.max() / 2.
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, format(cm[i, j], 'd'),
                           ha="center", va="center",
                           color="white" if cm[i, j] > thresh else "black",
                           fontsize=10)
            
            ax.set_ylabel('真实标签', fontsize=10)
            ax.set_xlabel('预测标签', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"混淆矩阵图已保存到: {output_path}")
    
    def generate_summary_report(self, output_path: Path = None):
        """Generate comprehensive summary report."""
        if output_path is None:
            output_path = self.output_dir / "comparison_summary.txt"
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("="*80 + "\n")
            f.write("分层BRB方法对比评估报告\n")
            f.write("="*80 + "\n\n")
            
            for method_name, result in self.results.items():
                f.write(f"\n{'='*60}\n")
                f.write(f"{method_name}\n")
                f.write(f"{'='*60}\n")
                f.write(f"总规则数: {result['num_rules']}\n")
                f.write(f"参数总数: {result['num_parameters']}\n")
                f.write(f"特征维度: {result['feature_dimension']}\n")
                f.write(f"总体准确率: {result['accuracy']:.4f}\n")
                f.write(f"平均推理时间: {result['avg_inference_time_ms']:.2f} ms\n")
                f.write(f"\n各类准确率:\n")
                for label, acc in result['per_class_accuracy'].items():
                    f.write(f"  {label}: {acc:.4f}\n")
            
            # Add summary comparison
            f.write(f"\n\n{'='*80}\n")
            f.write("关键指标对比\n")
            f.write(f"{'='*80}\n\n")
            
            # Find best in each category
            best_accuracy = max(self.results.items(), 
                              key=lambda x: x[1]['accuracy'])
            least_rules = min(self.results.items(),
                            key=lambda x: x[1]['num_rules'])
            least_params = min(self.results.items(),
                             key=lambda x: x[1]['num_parameters'])
            fastest = min(self.results.items(),
                        key=lambda x: x[1]['avg_inference_time_ms'])
            
            f.write(f"最高准确率: {best_accuracy[0]} ({best_accuracy[1]['accuracy']:.4f})\n")
            f.write(f"最少规则数: {least_rules[0]} ({least_rules[1]['num_rules']})\n")
            f.write(f"最少参数数: {least_params[0]} ({least_params[1]['num_parameters']})\n")
            f.write(f"最快推理: {fastest[0]} ({fastest[1]['avg_inference_time_ms']:.2f} ms)\n")
        
        print(f"总结报告已保存到: {output_path}")


def main():
    """Main function to run comparison."""
    print("开始分层BRB方法对比评估...")
    
    comparison = MethodComparison()
    
    # Run comparison
    comparison.run_comparison()
    
    # Generate outputs
    print("\n生成对比结果...")
    comparison.generate_comparison_table()
    comparison.generate_performance_table()
    comparison.plot_comparison()
    comparison.plot_confusion_matrices()
    comparison.generate_summary_report()
    
    print("\n✓ 对比评估完成！")


if __name__ == "__main__":
    main()
