"""
Enhanced feature_extraction.py

Purpose:
 - From raw acquisition CSV (acquired_measurements.csv) generate an enriched features set for:
     - symptom detection (amplitude error, ref error, freq error, phase degradation)
     - per-condition / per-attenuator response characterization
     - spectral/trace based metrics (noise floor, spurs, SNR) if trace data available
     - time-series features (rolling mean/std, trend slope) if timestamp present
     - anomaly scores (IsolationForest), clustering labels (DBSCAN/KMeans)
 - Map symptom activations to module-level meta using RULES_SIMPLE (belief vectors)
 - Produce outputs:
     * {prefix}_features_enhanced.csv
     * {prefix}_module_meta.csv
     * {prefix}_feature_summary.csv
     * {prefix}_feature_importances.csv (RF-based, if labels provided)

Usage:
    python feature_extraction.py --input acquired_measurements.csv --prefix run_enh
Dependencies:
    pandas, numpy, scipy, scikit-learn, pywt (optional, for wavelet), statsmodels (optional)
Notes:
 - If you have raw trace data in CSV (e.g., TRACE1 array), the script will attempt to compute spur/noise features.
 - Some features require multiple measurements per condition (repeats) or different atten settings; TEST_SEQUENCE must have produced such samples.
"""
import pandas as _pd
from typing import Tuple as _Tuple

import argparse
import math
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
import warnings
warnings.simplefilter("ignore")

# ---------- CONFIG ----------
# 你可以根据实验调整这些参数
ATTEN_GROUP_COLS = ["freq_Hz_set", "power_dBm_set"]  # 分组键，用于观察不同衰减/条件下的响应
MIN_REPEATS_FOR_STATS = 2
DBSCAN_EPS = 0.9
DBSCAN_MIN_SAMPLES = 5
ISO_CONTAMINATION = 0.05

# ---------- BRB 映射（症状 -> belief_vector），请保持与 brb_rules.yaml 的 modules_order 一致 ----------
RULES_SIMPLE = {
  "幅度测量准确度": [0.12,0.12,0.06,0.06,0.06,0.06,0.00,0.00,0.03,0.04,0.05,0.02,0.03,0.10,0.10,0.03,0.06,0.04,0.02,0.00,0.00],
  "频率读数准确度": [0.00,0.00,0.00,0.05,0.02,0.05,0.25,0.25,0.20,0.08,0.00,0.00,0.00,0.00,0.02,0.02,0.00,0.00,0.00,0.00,0.06],
  "参考电平准确度": [0.20,0.12,0.03,0.03,0.02,0.02,0.00,0.00,0.00,0.00,0.08,0.05,0.08,0.10,0.10,0.03,0.05,0.05,0.02,0.02,0.00]
}

MODULES_ORDER = [
    "衰减器", "前置放大器", "低频段前置低通滤波器", "低频段第一混频器",
    "高频段YTF滤波器", "高频段混频器",
    "时钟振荡器", "时钟合成与同步网络", "本振源（谐波发生器）", "本振混频组件",
    "校准源", "存储器", "校准信号开关",
    "中频放大器", "ADC", "数字RBW", "数字放大器", "数字检波器", "VBW滤波器",
    "电源模块", "未定义/其他"
]

# ---------- HELPERS ----------
def safe_div(a, b, eps=1e-12):
    try:
        return a / (b + eps)
    except:
        return np.nan

def normalize_belief(vec):
    arr = np.array(vec, dtype=float)
    s = arr.sum()
    if s <= 0:
        return np.ones_like(arr) / len(arr)
    return arr / s

def robust_stats(arr):
    """Return mean, std, mad, skew, kurtosis, min, max, count of finite entries"""
    a = np.array(arr)
    finite = a[np.isfinite(a)]
    if finite.size == 0:
        return dict(mean=np.nan, std=np.nan, mad=np.nan, skew=np.nan, kurt=np.nan, min=np.nan, max=np.nan, count=0)
    return dict(mean=float(np.mean(finite)),
                std=float(np.std(finite, ddof=1) if finite.size>1 else 0.0),
                mad=float(np.median(np.abs(finite - np.median(finite)))),
                skew=float(skew(finite)) if finite.size>2 else np.nan,
                kurt=float(kurtosis(finite)) if finite.size>3 else np.nan,
                min=float(np.min(finite)),
                max=float(np.max(finite)),
                count=int(finite.size))

def linear_trend(x, y):
    """Return slope, intercept of linear regression y ~ x"""
    try:
        x = np.array(x).reshape(-1,1)
        y = np.array(y)
        if len(y) < 2:
            return np.nan, np.nan
        reg = LinearRegression().fit(x, y)
        return float(reg.coef_[0]), float(reg.intercept_)
    except:
        return np.nan, np.nan

# ---------- FEATURE ENGINEERING ----------
def extract_enhanced_features(df_raw: pd.DataFrame, prefix="run_enh"):
    df = df_raw.copy()

    # 基础症状特征
    if "peak_dBm" in df.columns and "power_dBm_set" in df.columns:
        df["expected_dBm"] = df["power_dBm_set"] - df.get("atten_dB", 0)
        df["amplitude_error_dB"] = df["peak_dBm"] - df["expected_dBm"]
    else:
        df["amplitude_error_dB"] = np.nan
        df["expected_dBm"] = np.nan

    if "peak_freq_Hz" in df.columns and "freq_Hz_set" in df.columns:
        df["frequency_error_Hz"] = df["peak_freq_Hz"] - df["freq_Hz_set"]
        df["frequency_error_ppm"] = df["frequency_error_Hz"] / (df["freq_Hz_set"] + 1e-12) * 1e6
    else:
        df["frequency_error_Hz"] = np.nan
        df["frequency_error_ppm"] = np.nan

    if "phase_noise_dBc_perHz" in df.columns:
        df["phase_noise_raw"] = df["phase_noise_dBc_perHz"]
    else:
        df["phase_noise_raw"] = np.nan

    if "ref_level_dBm" in df.columns:
        df["ref_level_raw"] = df["ref_level_dBm"]
    else:
        df["ref_level_raw"] = np.nan

    # 分组统计（按 freq/power 观察衰减器/路径切换影响）
    group_cols = [c for c in ATTEN_GROUP_COLS if c in df.columns]
    if len(group_cols) == 0:
        group_cols = ["seq_index"] if "seq_index" in df.columns else []
    if group_cols:
        grouped = df.groupby(group_cols)
        agg = grouped.agg({
            "amplitude_error_dB": ["mean", "std"],
            "frequency_error_Hz": ["mean", "std"],
            "phase_noise_raw": ["mean", "std"],
            "ref_level_raw": ["mean", "std"]
        })
        agg.columns = ["_".join(x).strip() for x in agg.columns.values]
        agg = agg.reset_index()
        df = df.merge(agg, on=group_cols, how="left")
    else:
        df["amplitude_error_dB_mean"] = df["amplitude_error_dB"].mean()
        df["amplitude_error_dB_std"] = df["amplitude_error_dB"].std()
        df["frequency_error_Hz_mean"] = df["frequency_error_Hz"].mean()
        df["frequency_error_Hz_std"] = df["frequency_error_Hz"].std()
        df["phase_noise_raw_mean"] = df["phase_noise_raw"].mean()
        df["phase_noise_raw_std"] = df["phase_noise_raw"].std()
        df["ref_level_raw_mean"] = df["ref_level_raw"].mean()
        df["ref_level_raw_std"] = df["ref_level_raw"].std()

    # 归一化/衍生
    for col in ["amplitude_error_dB", "frequency_error_Hz", "phase_noise_raw", "ref_level_raw"]:
        if col in df.columns:
            global_mean = df[col].mean()
            global_std = df[col].std() if not math.isnan(df[col].std()) else 1.0
            df[f"{col}_z"] = (df[col] - global_mean) / (global_std + 1e-12)
            df[f"{col}_abs"] = np.abs(df[col])
            df[f"{col}_rel_pct"] = (df[col] - global_mean) / (global_mean + 1e-12) * 100.0
        else:
            df[f"{col}_z"] = np.nan
            df[f"{col}_abs"] = np.nan
            df[f"{col}_rel_pct"] = np.nan

    # 衰减-幅度斜率（若存在衰减设置）
    if "atten_dB" in df.columns and group_cols:
        slopes = []
        keys = []
        for name, g in df.groupby(group_cols):
            if g["atten_dB"].nunique() >= 2:
                slope, intercept = linear_trend(g["atten_dB"].values, g["amplitude_error_dB"].fillna(0.0).values)
            else:
                slope, intercept = np.nan, np.nan
            keys.append(name)
            slopes.append((slope, intercept))
        slope_map = {k: v[0] for k, v in zip(keys, slopes)}
        def get_slope(row):
            key = tuple(row[c] for c in group_cols) if len(group_cols)>1 else row[group_cols[0]]
            return slope_map.get(key, np.nan)
        df["atten_amp_slope"] = df.apply(get_slope, axis=1)

    # trace-based 特征（可选，若 CSV 中包含 trace 列）
    trace_cols = [c for c in df.columns if "trace" in c.lower() or "TRACE" in c]
    if trace_cols:
        for tc in trace_cols:
            def parse_trace_cell(cell):
                try:
                    if isinstance(cell, str):
                        arr = np.fromstring(cell.strip("[]"), sep=",")
                        if arr.size > 0:
                            return arr
                    if isinstance(cell, (list, np.ndarray)):
                        return np.array(cell, dtype=float)
                except:
                    pass
                return None
            spurs = []
            noise_floors = []
            max_spurs = []
            snrs = []
            for val in df[tc].values:
                arr = parse_trace_cell(val)
                if arr is None or arr.size == 0:
                    spurs.append(0)
                    noise_floors.append(np.nan)
                    max_spurs.append(np.nan)
                    snrs.append(np.nan)
                    continue
                nf = np.percentile(arr, 20)
                noise_floors.append(float(nf))
                spur_thresh = nf + 6.0
                peaks = arr[arr > spur_thresh]
                spurs.append(int(peaks.size))
                max_spurs.append(float(np.max(peaks) if peaks.size>0 else np.nan))
                primary = np.max(arr)
                snrs.append(float(primary - nf))
            df[f"{tc}_spur_count"] = spurs
            df[f"{tc}_noise_floor"] = noise_floors
            df[f"{tc}_max_spur"] = max_spurs
            df[f"{tc}_snr"] = snrs

    # 异常检测（IsolationForest）
    indicators = ["amplitude_error_dB", "frequency_error_Hz", "phase_noise_raw", "ref_level_raw"]
    avail_inds = [c for c in indicators if c in df.columns]
    if len(avail_inds) >= 1:
        X = df[avail_inds].fillna(0.0).values
        try:
            iso = IsolationForest(n_estimators=200, contamination=ISO_CONTAMINATION, random_state=42)
            iso.fit(X)
            scores = iso.decision_function(X)
            smin, smax = scores.min(), scores.max()
            norm = (scores - smin) / (smax - smin + 1e-12)
            df["if_anom_score"] = 1.0 - norm
            df["if_is_outlier"] = (df["if_anom_score"] > 0.9).astype(int)
        except Exception:
            df["if_anom_score"] = np.nan
            df["if_is_outlier"] = 0
    else:
        df["if_anom_score"] = np.nan
        df["if_is_outlier"] = 0

    # 聚类（DBSCAN + KMeans，可选）
    cluster_feats = []
    for f in ["amplitude_error_dB_z", "frequency_error_Hz_z", "phase_noise_raw_z", "ref_level_raw_z", "if_anom_score"]:
        if f in df.columns:
            cluster_feats.append(f)
    if cluster_feats:
        Xc = df[cluster_feats].fillna(0.0).values
        try:
            scaler = StandardScaler()
            Xc_s = scaler.fit_transform(Xc)
            db = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES).fit(Xc_s)
            df["cluster_dbscan"] = db.labels_
            k = min(7, max(2, int(math.sqrt(len(df)))))
            km = KMeans(n_clusters=k, random_state=42).fit_predict(Xc_s)
            df["cluster_kmeans"] = km
        except Exception:
            df["cluster_dbscan"] = -1
            df["cluster_kmeans"] = -1
    else:
        df["cluster_dbscan"] = -1
        df["cluster_kmeans"] = -1

    # 症状 activation（0..1），映射到模块 belief
    activations = {}
    amp_ref = 0.5
    activations["幅度测量准确度"] = np.clip(np.abs(df["amplitude_error_dB"].fillna(0.0)) / amp_ref, 0.0, 1.0)
    freq_ref = 5.0
    activations["频率读数准确度"] = np.clip(np.abs(df["frequency_error_Hz"].fillna(0.0)) / freq_ref, 0.0, 1.0)
    if "phase_noise_raw" in df.columns:
        med = np.nanmedian(df["phase_noise_raw"].dropna()) if df["phase_noise_raw"].dropna().size>0 else 0.0
        deg = df["phase_noise_raw"].fillna(med) - med
        phase_ref = 3.0
        activations["相位噪声"] = np.clip(deg / phase_ref, 0.0, 1.0)
    else:
        activations["相位噪声"] = np.zeros(len(df))
    ref_ref = 1.0
    activations["参考电平准确度"] = np.clip(np.abs(df["ref_level_raw"].fillna(0.0)) / ref_ref, 0.0, 1.0)

    for sym, acts in activations.items():
        df[f"{sym}__activation"] = acts

    # 模块级 meta（按 RULES_SIMPLE 权重加权）
    bv_list = [normalize_belief(RULES_SIMPLE[sym]) for sym in RULES_SIMPLE.keys()]
    sym_order = list(RULES_SIMPLE.keys())
    module_meta = np.zeros((len(df), len(bv_list[0])), dtype=float)
    for i, sym in enumerate(sym_order):
        acts = df[f"{sym}__activation"].fillna(0.0).values
        bv = bv_list[i]
        module_meta += np.outer(acts, bv)
    row_sums = module_meta.sum(axis=1, keepdims=True)
    row_sums[row_sums==0] = 1.0
    module_meta = module_meta / row_sums
    module_cols = [f"module_{m}" for m in MODULES_ORDER]
    module_meta_df = pd.DataFrame(module_meta, columns=module_cols)

    # 特征汇总
    feat_summary = []
    for c in df.columns:
        s = df[c]
        feat_summary.append({
            "features": c,
            "n_nonnull": int(s.notnull().sum()),
            "pct_nonnull": float(s.notnull().mean()),
            "mean": float(s.mean()) if pd.api.types.is_numeric_dtype(s) else np.nan,
            "std": float(s.std()) if pd.api.types.is_numeric_dtype(s) else np.nan,
            "min": float(s.min()) if pd.api.types.is_numeric_dtype(s) else np.nan,
            "max": float(s.max()) if pd.api.types.is_numeric_dtype(s) else np.nan
        })
    feat_summary_df = pd.DataFrame(feat_summary)

    # 有监督特征重要性（如果存在标签列）
    feature_importances = None
    label_col = None
    for possible in ["true_module", "真实模块", "label", "ground_truth"]:
        if possible in df.columns:
            label_col = possible
            break
    if label_col is not None:
        try:
            Xcols = [c for c in df.columns if any(k in c for k in ["amplitude_error", "frequency_error", "phase_noise", "ref_level", "__activation", "_anom", "_z", "_snr", "_noise_floor", "atten_amp_slope"])]
            Xcols = [c for c in Xcols if pd.api.types.is_numeric_dtype(df[c])]
            X = df[Xcols].fillna(0.0).values
            y = df[label_col].values
            if y.dtype.kind in {"U", "S", "O"}:
                uniq = list(pd.factorize(y)[1])
                mapping = {v:i for i,v in enumerate(uniq)}
                y_idx = np.array([mapping.get(v, -1) for v in y])
                valid_mask = (y_idx >= 0)
                y_train = y_idx[valid_mask]
                X_train = X[valid_mask]
            else:
                y_train = y
                X_train = X
            rf = RandomForestClassifier(n_estimators=200, random_state=42)
            rf.fit(X_train, y_train)
            imps = rf.feature_importances_
            feature_importances = pd.DataFrame({"features": Xcols, "importance": imps}).sort_values("importance", ascending=False)
        except Exception:
            feature_importances = None

    # 保存输出
    prefix = prefix if prefix else "run_enh"
    feat_out_path = f"{prefix}_features_enhanced.csv"
    module_meta_path = f"{prefix}_module_meta.csv"
    feat_summary_path = f"{prefix}_feature_summary.csv"
    df_out = pd.concat([df.reset_index(drop=True), module_meta_df.reset_index(drop=True)], axis=1)
    df_out.to_csv(feat_out_path, index=False, encoding="utf-8-sig")
    module_meta_df.to_csv(module_meta_path, index=False, encoding="utf-8-sig")
    feat_summary_df.to_csv(feat_summary_path, index=False, encoding="utf-8-sig")
    if feature_importances is not None:
        feature_importances.to_csv(f"{prefix}_feature_importances.csv", index=False, encoding="utf-8-sig")
    return df_out, module_meta_df, feat_summary_df, feature_importances

# 兼容接口，供 main_pipeline 调用
def compute_feature_matrix(raw_input, prefix="run_enh"):
    if isinstance(raw_input, str):
        raw_df = _pd.read_csv(raw_input, encoding="utf-8")
    elif isinstance(raw_input, _pd.DataFrame):
        raw_df = raw_input.copy()
    else:
        raise ValueError("raw_input must be a pandas.DataFrame or path to CSV")
    feat_df, module_meta_df, feat_summary_df, feature_importances = extract_enhanced_features(raw_df, prefix=prefix)
    return feat_df, module_meta_df

# CLI
if __name__ == "__main__":
    input_csv = r"D:\PycharmProjects\FMFD\V2\acquired_measurements.csv"
    prefix = "run_test"
    raw = pd.read_csv(input_csv, encoding="utf-8")
    extract_enhanced_features(raw, prefix=prefix)
    print(f"[INFO] features extraction done. input={input_csv}, prefix={prefix}")