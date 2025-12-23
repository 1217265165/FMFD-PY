# 分层BRB方法对比模块

本模块实现了与其他分层诊断方法的对比实验，包括：

## 对比方法

### 1. HCF (Zhang et al., 2022)
**Hierarchical Cognitive Framework - 基于领域知识与数据融合的分层认知框架**

- **特点**：
  - 依赖专家预先定义的模块关联度矩阵
  - 无动态规则激活机制
  - 全量特征处理
  
- **技术参数**：
  - 总规则数：~130条
  - 参数总数：~200+
  - 系统级特征维度：8-15维
  - 故障模式数：21种

- **优势**：专家知识充分融入，在已知故障模式下准确率高
- **劣势**：规则库大，无法有效控制规则爆炸问题


### 2. BRB-P (Ming et al., 2023)
**BRB with Probability constraint - 引入概率表约束优化的BRB**

- **特点**：
  - 在BRB基础上引入概率约束优化
  - 改进规则学习过程
  - 未从源头削减规则数量
  
- **技术参数**：
  - 总规则数：81条
  - 参数总数：571个
  - 系统级特征维度：15维
  - 适用场景：继电器故障诊断（6种模式）

- **优势**：概率约束提升了推理一致性
- **劣势**：参数多，小样本场景下容易过拟合


### 3. ER-c (Zhang et al., 2024)
**Enhanced Reasoning with credibility - 强化可信度评估的证据推理**

- **特点**：
  - 强化推理过程中的结论可信度评估
  - 引入规则可靠性度量
  - 规则库规模控制仍依赖后优化
  
- **技术参数**：
  - 总规则数：~60条
  - 参数总数：~150个
  - 系统级特征维度：~10维
  - 适用场景：焊接机故障诊断（3种模式）

- **优势**：可信度评估提升诊断可靠性
- **劣势**：未解决规则数量根本问题


### 4. 本文方法
**Knowledge-driven Hierarchical BRB - 知识驱动的分层BRB**

- **特点**：
  - 基于物理机理的特征-模块映射
  - 系统级-模块级分层推理架构
  - 条件激活机制控制规则爆炸
  
- **技术参数**：
  - 总规则数：45条（↓59%）
  - 参数总数：38个（↓70-93%）
  - 系统级特征维度：4维（↓60%）
  - 故障场景：7模块×3等级=21种组合

- **优势**：规则少、参数少、推理快、小样本适应性强
- **劣势**：需要领域知识引导特征选择


## 使用方法

### 1. 运行基线数据生成
```bash
python -m FMFD.pipelines.run_babeline
```

### 2. 运行故障仿真（生成测试数据）
```bash
python -m FMFD.pipelines.simulate.run_sinulation_brb
```

### 3. 运行方法对比
```bash
python -m FMFD.pipelines.compare_methods
```

### 4. 查看对比结果

对比结果保存在 `Output/sim_spectrum/` 目录下：

- `comparison_table.csv` - 方法对比表（Table 3-2）
- `performance_table.csv` - 性能细分表（Table 3-3）
- `comparison_plot.png` - 准确率-规则数权衡图（Figure 3-4）
- `confusion_matrices.png` - 各方法混淆矩阵
- `comparison_summary.txt` - 详细对比报告


## 对比指标

### 1. 规模指标
- **总规则数**：反映规则爆炸控制程度
- **参数总数**：反映优化复杂度
- **特征维度**：反映输入空间维度

### 2. 性能指标
- **诊断准确率**：整体判别能力
- **各类准确率**：细分类别性能
- **推理时间**：计算效率

### 3. 小样本适应性
- **参数-样本比**：参数充足度
- **推荐样本需求**：最小样本量估计


## 关键发现

根据表3-2和图3-4的对比结果：

1. **规则库压缩**：本文方法规则数从130条→45条，削减59%
2. **参数简化**：参数总数从571个→38个，削减93%
3. **特征降维**：系统级特征从15维→4维，降低73%
4. **准确率保持**：94.18%的准确率，在复杂场景下优于BRB-P和ER-c
5. **推理加速**：推理时间加速3.08倍
6. **小样本需求**：推荐样本从62-100条→19条，降低70%


## 实现说明

各对比方法的实现位于 `comparison/` 目录：

```
comparison/
├── __init__.py       # 模块导入
├── hcf.py           # HCF方法实现
├── brb_p.py         # BRB-P方法实现
└── er_c.py          # ER-c方法实现
```

对比评估脚本：
```
pipelines/
└── compare_methods.py  # 对比评估主脚本
```


## 参考文献

1. Zhang et al. (2022). "Hierarchical Cognitive Framework for Fault Diagnosis"
2. Ming et al. (2023). "BRB with Probability Constraint for Relay Fault Diagnosis"
3. Zhang et al. (2024). "Enhanced Reasoning with Credibility Assessment"


## 注意事项

1. **数据要求**：需要先运行仿真生成测试数据
2. **中文字体**：图表中文显示需要系统安装SimHei或Microsoft YaHei字体
3. **计算时间**：完整对比需要几分钟，取决于样本数量
4. **内存占用**：处理275个样本约需500MB内存


## 扩展说明

如需添加新的对比方法：

1. 在 `comparison/` 目录下创建新的方法实现
2. 实现 `infer_system()` 和 `infer_module()` 接口
3. 实现 `get_num_rules()`, `get_num_parameters()`, `get_feature_dimension()` 方法
4. 在 `compare_methods.py` 中注册新方法
