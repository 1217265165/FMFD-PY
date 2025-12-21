
import pandas as pd
df = pd.read_csv(r"D:/PycharmProjects/FMFD/FMFD/Output/sim_spectrum/features_brb.csv")
print("sys_pred 分布:", df[[c for c in df if c.startswith("sys_")]].idxmax(axis=1).str.replace("sys_","").value_counts(), "\n")
print("按真值分组的 sys_* 均值：")
print(df.groupby("label_system_fault_class")[[c for c in df if c.startswith("sys_")]].mean())