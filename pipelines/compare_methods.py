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
from sklearn.metrics import (
    accuracy_score, confusion_matrix, precision_score,
    recall_score, f1_score, classification_report
)
import warnings
warnings.filterwarnings('ignore')

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
        
        # Calculate precision, recall, F1 for each class
        label_order = ["正常", "幅度失准", "频率失准", "参考电平失准"]
        
        # Micro and macro averages
        precision_macro = precision_score(true_labels, predictions, labels=label_order, average='macro', zero_division=0)
        recall_macro = recall_score(true_labels, predictions, labels=label_order, average='macro', zero_division=0)
        f1_macro = f1_score(true_labels, predictions, labels=label_order, average='macro', zero_division=0)
        
        precision_weighted = precision_score(true_labels, predictions, labels=label_order, average='weighted', zero_division=0)
        recall_weighted = recall_score(true_labels, predictions, labels=label_order, average='weighted', zero_division=0)
        f1_weighted = f1_score(true_labels, predictions, labels=label_order, average='weighted', zero_division=0)
        
        # Per-class metrics
        precision_per_class = precision_score(true_labels, predictions, labels=label_order, average=None, zero_division=0)
        recall_per_class = recall_score(true_labels, predictions, labels=label_order, average=None, zero_division=0)
        f1_per_class = f1_score(true_labels, predictions, labels=label_order, average=None, zero_division=0)
        
        # Calculate per-class accuracy
        cm = confusion_matrix(true_labels, predictions, labels=label_order)
        
        per_class_acc = {}
        per_class_precision = {}
        per_class_recall = {}
        per_class_f1 = {}
        
        for i, label in enumerate(label_order):
            if cm[i].sum() > 0:
                per_class_acc[label] = cm[i, i] / cm[i].sum()
            else:
                per_class_acc[label] = 0.0
            per_class_precision[label] = precision_per_class[i]
            per_class_recall[label] = recall_per_class[i]
            per_class_f1[label] = f1_per_class[i]
        
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
        std_inference_time = np.std(inference_times)
        
        results = {
            "method_name": method_name,
            "total_samples": len(true_labels),
            "accuracy": accuracy,
            "precision_macro": precision_macro,
            "recall_macro": recall_macro,
            "f1_macro": f1_macro,
            "precision_weighted": precision_weighted,
            "recall_weighted": recall_weighted,
            "f1_weighted": f1_weighted,
            "per_class_accuracy": per_class_acc,
            "per_class_precision": per_class_precision,
            "per_class_recall": per_class_recall,
            "per_class_f1": per_class_f1,
            "num_rules": num_rules,
            "num_parameters": num_params,
            "feature_dimension": feature_dim,
            "avg_inference_time_ms": avg_inference_time,
            "std_inference_time_ms": std_inference_time,
            "confusion_matrix": cm,
            "label_order": label_order
        }
        
        print(f"总样本数: {len(true_labels)}")
        print(f"准确率: {accuracy:.4f}")
        print(f"精确率(宏平均): {precision_macro:.4f}")
        print(f"召回率(宏平均): {recall_macro:.4f}")
        print(f"F1分数(宏平均): {f1_macro:.4f}")
        print(f"规则数: {num_rules}")
        print(f"参数数: {num_params}")
        print(f"特征维度: {feature_dim}")
        print(f"平均推理时间: {avg_inference_time:.2f}±{std_inference_time:.2f} ms")
        
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
                f.write(f"精确率(宏平均): {result['precision_macro']:.4f}\n")
                f.write(f"召回率(宏平均): {result['recall_macro']:.4f}\n")
                f.write(f"F1分数(宏平均): {result['f1_macro']:.4f}\n")
                f.write(f"精确率(加权平均): {result['precision_weighted']:.4f}\n")
                f.write(f"召回率(加权平均): {result['recall_weighted']:.4f}\n")
                f.write(f"F1分数(加权平均): {result['f1_weighted']:.4f}\n")
                f.write(f"平均推理时间: {result['avg_inference_time_ms']:.2f}±{result['std_inference_time_ms']:.2f} ms\n")
                f.write(f"\n各类准确率:\n")
                for label, acc in result['per_class_accuracy'].items():
                    f.write(f"  {label}: {acc:.4f}\n")
                f.write(f"\n各类精确率:\n")
                for label, prec in result['per_class_precision'].items():
                    f.write(f"  {label}: {prec:.4f}\n")
                f.write(f"\n各类召回率:\n")
                for label, rec in result['per_class_recall'].items():
                    f.write(f"  {label}: {rec:.4f}\n")
                f.write(f"\n各类F1分数:\n")
                for label, f1 in result['per_class_f1'].items():
                    f.write(f"  {label}: {f1:.4f}\n")
            
            # Add summary comparison
            f.write(f"\n\n{'='*80}\n")
            f.write("关键指标对比\n")
            f.write(f"{'='*80}\n\n")
            
            # Find best in each category
            best_accuracy = max(self.results.items(), 
                              key=lambda x: x[1]['accuracy'])
            best_precision = max(self.results.items(),
                               key=lambda x: x[1]['precision_macro'])
            best_recall = max(self.results.items(),
                            key=lambda x: x[1]['recall_macro'])
            best_f1 = max(self.results.items(),
                        key=lambda x: x[1]['f1_macro'])
            least_rules = min(self.results.items(),
                            key=lambda x: x[1]['num_rules'])
            least_params = min(self.results.items(),
                             key=lambda x: x[1]['num_parameters'])
            fastest = min(self.results.items(),
                        key=lambda x: x[1]['avg_inference_time_ms'])
            
            f.write(f"最高准确率: {best_accuracy[0]} ({best_accuracy[1]['accuracy']:.4f})\n")
            f.write(f"最高精确率: {best_precision[0]} ({best_precision[1]['precision_macro']:.4f})\n")
            f.write(f"最高召回率: {best_recall[0]} ({best_recall[1]['recall_macro']:.4f})\n")
            f.write(f"最高F1分数: {best_f1[0]} ({best_f1[1]['f1_macro']:.4f})\n")
            f.write(f"最少规则数: {least_rules[0]} ({least_rules[1]['num_rules']})\n")
            f.write(f"最少参数数: {least_params[0]} ({least_params[1]['num_parameters']})\n")
            f.write(f"最快推理: {fastest[0]} ({fastest[1]['avg_inference_time_ms']:.2f} ms)\n")
            
            # Calculate comprehensive score
            f.write(f"\n\n{'='*80}\n")
            f.write("综合评分 (归一化加权平均)\n")
            f.write(f"{'='*80}\n\n")
            f.write("评分公式: 0.3×准确率 + 0.2×F1分数 + 0.2×效率 + 0.15×规则压缩 + 0.15×参数压缩\n\n")
            
            # Normalize and calculate scores
            max_acc = max(r['accuracy'] for r in self.results.values())
            max_f1 = max(r['f1_macro'] for r in self.results.values())
            max_time = max(r['avg_inference_time_ms'] for r in self.results.values())
            max_rules = max(r['num_rules'] for r in self.results.values())
            max_params = max(r['num_parameters'] for r in self.results.values())
            
            for method_name, result in self.results.items():
                score = (
                    0.3 * (result['accuracy'] / max_acc) +
                    0.2 * (result['f1_macro'] / max_f1) +
                    0.2 * (1 - result['avg_inference_time_ms'] / max_time) +
                    0.15 * (1 - result['num_rules'] / max_rules) +
                    0.15 * (1 - result['num_parameters'] / max_params)
                )
                f.write(f"{method_name}: {score:.4f}\n")
        
        print(f"总结报告已保存到: {output_path}")
    
    def plot_radar_chart(self, output_path: Path = None):
        """Plot radar chart comparing multiple metrics."""
        if output_path is None:
            output_path = self.output_dir / "radar_comparison.png"
        
        # Metrics to compare (normalized to 0-1)
        metrics = ['准确率', '精确率', '召回率', 'F1分数', '效率', '复杂度']
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # Close the plot
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        for idx, (method_name, result) in enumerate(self.results.items()):
            # Normalize metrics to 0-1 scale
            accuracy_norm = result['accuracy']
            precision_norm = result['precision_macro']
            recall_norm = result['recall_macro']
            f1_norm = result['f1_macro']
            
            # Efficiency: inverse of inference time (normalized)
            max_time = max(r['avg_inference_time_ms'] for r in self.results.values())
            efficiency_norm = 1 - (result['avg_inference_time_ms'] / max_time)
            
            # Complexity: inverse of rules+params (normalized)
            max_complexity = max(r['num_rules'] + r['num_parameters'] for r in self.results.values())
            complexity_norm = 1 - ((result['num_rules'] + result['num_parameters']) / max_complexity)
            
            values = [accuracy_norm, precision_norm, recall_norm, f1_norm, efficiency_norm, complexity_norm]
            values += values[:1]  # Close the plot
            
            ax.plot(angles, values, 'o-', linewidth=2, label=method_name, 
                   color=colors[idx], markersize=8)
            ax.fill(angles, values, alpha=0.15, color=colors[idx])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics, fontsize=11)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=9)
        ax.set_title('方法综合性能对比雷达图', fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"雷达图已保存到: {output_path}")
    
    def plot_metrics_bar_chart(self, output_path: Path = None):
        """Plot bar chart for detailed metrics comparison."""
        if output_path is None:
            output_path = self.output_dir / "metrics_bar_chart.png"
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        methods = list(self.results.keys())
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        # 1. Accuracy metrics
        ax1 = axes[0, 0]
        accuracy_data = [self.results[m]['accuracy'] * 100 for m in methods]
        precision_data = [self.results[m]['precision_macro'] * 100 for m in methods]
        recall_data = [self.results[m]['recall_macro'] * 100 for m in methods]
        f1_data = [self.results[m]['f1_macro'] * 100 for m in methods]
        
        x = np.arange(len(methods))
        width = 0.2
        
        ax1.bar(x - 1.5*width, accuracy_data, width, label='准确率', color='#3498db')
        ax1.bar(x - 0.5*width, precision_data, width, label='精确率', color='#2ecc71')
        ax1.bar(x + 0.5*width, recall_data, width, label='召回率', color='#f39c12')
        ax1.bar(x + 1.5*width, f1_data, width, label='F1分数', color='#e74c3c')
        
        ax1.set_ylabel('百分比 (%)', fontsize=11)
        ax1.set_title('分类性能指标对比', fontsize=12, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels([m.split('(')[0].strip() for m in methods], fontsize=9, rotation=15, ha='right')
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.set_ylim(0, 105)
        
        # 2. Model complexity
        ax2 = axes[0, 1]
        rules_data = [self.results[m]['num_rules'] for m in methods]
        params_data = [self.results[m]['num_parameters'] for m in methods]
        
        ax2_twin = ax2.twinx()
        
        bars1 = ax2.bar(x - width/2, rules_data, width, label='规则数', color='#9b59b6')
        bars2 = ax2_twin.bar(x + width/2, params_data, width, label='参数数', color='#e67e22')
        
        ax2.set_ylabel('规则数', fontsize=11, color='#9b59b6')
        ax2_twin.set_ylabel('参数数', fontsize=11, color='#e67e22')
        ax2.set_title('模型复杂度对比', fontsize=12, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels([m.split('(')[0].strip() for m in methods], fontsize=9, rotation=15, ha='right')
        ax2.tick_params(axis='y', labelcolor='#9b59b6')
        ax2_twin.tick_params(axis='y', labelcolor='#e67e22')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Combine legends
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2_twin.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=9)
        
        # 3. Inference time
        ax3 = axes[1, 0]
        time_data = [self.results[m]['avg_inference_time_ms'] for m in methods]
        time_std = [self.results[m]['std_inference_time_ms'] for m in methods]
        
        bars = ax3.bar(x, time_data, color=colors, alpha=0.7, edgecolor='black', linewidth=1)
        ax3.errorbar(x, time_data, yerr=time_std, fmt='none', ecolor='black', capsize=5, linewidth=1.5)
        
        ax3.set_ylabel('推理时间 (ms)', fontsize=11)
        ax3.set_title('推理效率对比', fontsize=12, fontweight='bold')
        ax3.set_xticks(x)
        ax3.set_xticklabels([m.split('(')[0].strip() for m in methods], fontsize=9, rotation=15, ha='right')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for i, (bar, val, std) in enumerate(zip(bars, time_data, time_std)):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + std + 0.05,
                    f'{val:.2f}±{std:.2f}',
                    ha='center', va='bottom', fontsize=8)
        
        # 4. Per-class F1 scores
        ax4 = axes[1, 1]
        labels = ["正常", "幅度失准", "频率失准", "参考电平失准"]
        
        x_classes = np.arange(len(labels))
        width_class = 0.2
        
        for i, method in enumerate(methods):
            f1_scores = [self.results[method]['per_class_f1'][label] * 100 for label in labels]
            offset = (i - len(methods)/2 + 0.5) * width_class
            ax4.bar(x_classes + offset, f1_scores, width_class, 
                   label=method.split('(')[0].strip(), color=colors[i], alpha=0.7)
        
        ax4.set_ylabel('F1分数 (%)', fontsize=11)
        ax4.set_title('各类别F1分数对比', fontsize=12, fontweight='bold')
        ax4.set_xticks(x_classes)
        ax4.set_xticklabels(labels, fontsize=9, rotation=15, ha='right')
        ax4.legend(fontsize=8, loc='lower left')
        ax4.grid(True, alpha=0.3, axis='y')
        ax4.set_ylim(0, 105)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"详细指标柱状图已保存到: {output_path}")
    
    def generate_detailed_metrics_table(self, output_path: Path = None):
        """Generate detailed metrics comparison table."""
        if output_path is None:
            output_path = self.output_dir / "detailed_metrics.csv"
        
        table_data = []
        
        for method_name, result in self.results.items():
            row = {
                "方法": method_name,
                "准确率": f"{result['accuracy']:.4f}",
                "精确率(宏)": f"{result['precision_macro']:.4f}",
                "召回率(宏)": f"{result['recall_macro']:.4f}",
                "F1分数(宏)": f"{result['f1_macro']:.4f}",
                "精确率(加权)": f"{result['precision_weighted']:.4f}",
                "召回率(加权)": f"{result['recall_weighted']:.4f}",
                "F1分数(加权)": f"{result['f1_weighted']:.4f}",
                "推理时间(ms)": f"{result['avg_inference_time_ms']:.2f}±{result['std_inference_time_ms']:.2f}",
            }
            table_data.append(row)
        
        df = pd.DataFrame(table_data)
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        
        print(f"详细指标表已保存到: {output_path}")
        print("\n详细指标对比:")
        print(df.to_string(index=False))


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
    comparison.generate_detailed_metrics_table()
    
    # Generate visualizations
    print("\n生成可视化图表...")
    comparison.plot_comparison()
    comparison.plot_confusion_matrices()
    comparison.plot_radar_chart()
    comparison.plot_metrics_bar_chart()
    
    # Generate summary
    comparison.generate_summary_report()
    
    print("\n✓ 对比评估完成！所有结果已保存到 Output/comparison_results/")
    print("\n生成的文件:")
    print("  - comparison_table.csv        # 方法对比表")
    print("  - performance_table.csv       # 性能细分表")
    print("  - detailed_metrics.csv        # 详细指标表")
    print("  - comparison_plot.png         # 准确率-规则数权衡图")
    print("  - confusion_matrices.png      # 混淆矩阵对比")
    print("  - radar_comparison.png        # 雷达图综合对比")
    print("  - metrics_bar_chart.png       # 详细指标柱状图")
    print("  - comparison_summary.txt      # 文本总结报告")


if __name__ == "__main__":
    main()
