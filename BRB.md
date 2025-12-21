# BRB 两层方案（21 模块）使用说明

## 模块列表（与 `brb_rules.yaml` / `BRB/module_brb.py` / `features/feature_extraction.py` 一致）
1. 衰减器  
2. 前置放大器  
3. 低频段前置低通滤波器  
4. 低频段第一混频器  
5. 高频段 YTF 滤波器  
6. 高频段混频器  
7. 时钟振荡器  
8. 时钟合成与同步网络  
9. 本振源（谐波发生器）  
10. 本振混频组件  
11. 校准源  
12. 存储器  
13. 校准信号开关  
14. 中频放大器  
15. ADC  
16. 数字 RBW  
17. 数字放大器  
18. 数字检波器  
19. VBW 滤波器  
20. 电源模块  
21. 未定义/其他  

系统级异常：幅度失准、频率失准、参考电平失准。

## 目录与关键文件
FMFD/  
├─ baseline/  
│  ├─ baseline.py          # 读取/对齐正常 CSV，计算 RRS/包络、切换点特性  
│  ├─ config.py            # 频段、输出路径、字体等配置（Output/ 下的文件名）  
│  ├─ viz.py               # RRS/包络/切换点可视化  
├─ BRB/  
│  ├─ system_brb.py        # 第一层 BRB（幅度失准/频率失准/参考电平失准）  
│  ├─ module_brb.py        # 第二层 BRB（21 模块）  
│  ├─ utils.py             # BRB 简化推理工具  
├─ features/  
│  ├─ extract.py           # 系统特征提取（含切换点/非切换台阶异常）  
│  ├─ feature_extraction.py# 采集数据特征工程 + module_meta(21 维)，与 brb_rules.yaml 对齐  
├─ pipelines/  
│  ├─ run_babeline.py      # 基线构建：对齐正常数据，算 RRS/包络，输出 Output/ 下 npz/json/png/csv  
│  ├─ run_simulation_brb.py# 仿真故障→特征→两层 BRB，输出 sim_fault_dataset.csv  
│  ├─ detect.py            # 检测待检 CSV（to_detect/），输出 Output/detection_results.csv  
│  └─ Output/              # 运行后生成的结果目录（baseline_artifacts.npz 等）  
├─ simulation/  
│  ├─ faults.py            # 故障/畸变注入（基于当前曲线 σ 自适应幅度）  
│  ├─ dataset.py           # 若存在：组合基线 + faults 生成仿真数据的封装  
├─ normal_response_data/   # 放正常频响 CSV（frequency, amplitude_dB 两列）  
├─ brb_rules.yaml          # 21 模块的规则/先验（与 feature_extraction/module_brb 对齐）  
├─ thresholds.json         # 检测阈值（detect.py 使用）  
├─ brb_chains_generated.yaml# 可选：自动链路规则，main_pipeline 可合并  
├─ main_pipline.py         # 端到端入口（若更名 main_pipeline.py，命令同步）  
└─ BRB.md                  # 本说明  

## 使用步骤

### 0. 环境
安装依赖：numpy, pandas, scipy, scikit-learn, matplotlib, pyyaml, cma（若做优化）。

### 1. 基线构建（正常数据）
1) 将 ≥30 条正常频响 CSV 放到 `FMFD/normal_response_data/`（两列：frequency, amplitude_dB）。  
2) 运行：
```bash
python -m FMFD.pipelines.run_babeline
```
输出（在 Output/）：baseline_artifacts.npz（frequency, rrs, upper, lower）、baseline_meta.json（band_ranges, k_list）、switching_features.csv/json、normal_feature_stats.csv、baseline_switching.png。

### 2. 故障仿真 + BRB（可选）
```bash
python -m FMFD.pipelines.run_simulation_brb
```
生成 sim_fault_dataset.csv（含系统特征与两层 BRB 概率）。

### 3. 检测待检数据
1) 待检 CSV 放 `FMFD/to_detect/`（frequency, amplitude 两列）。  
2) 确认 thresholds.json 路径（detect.py 默认相对 `../thresholds.json`；若在根运行，改成根目录的 thresholds.json 或保持文件放上一级）。  
3) 运行：
```bash
python -m FMFD.pipelines.detect
```
输出：`Output/detection_results.csv`（特征、系统/模块概率、warn/alarm/ok 标志）。

### 4. 端到端主流程（采集→特征→BRB→可选优化）
入口：`FMFD/main_pipline.py`  
示例（使用已有 RAW_OUTPUT_CSV）：
```bash
python -m FMFD.main_pipline
```
参数：  
- `--collect` 启动采集（data_acquisition.acquire_sequence）  
- `--optimize` 启用 CMA-ES 规则优化  
- `--supervised --label_col <列名>` 有监督优化时指定标签列  
输出：`<FEATURE_OUTPUT_PREFIX>_features.csv`、`<FEATURE_OUTPUT_PREFIX>_module_meta.csv`、`brb_initial_outputs.csv`，可选 `brb_optimized_outputs.csv`。

## 一致性说明
- `brb_rules.yaml`、`BRB/module_brb.py`、`features/feature_extraction.py` 的模块顺序已统一为 21 模块。  
- system_brb.py 输出三类系统级异常；module_brb.py 第二层输出 21 模块概率。  
- feature_extraction.py 生成的 module_meta_* 列即 21 维，与 brb_rules.yaml 对齐。  
- detect.py、run_simulation_brb.py 已使用两层 BRB（system_brb + module_brb）。  
- 如需单层 BRB（brb_engine + brb_rules.yaml），需确保 modules_order 与数据列一致后自行调用 brb_engine.inference。  
- 若重命名 main_pipline.py，请同步更新运行命令。