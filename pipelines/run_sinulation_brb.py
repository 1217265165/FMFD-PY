#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
信号级仿真 + BRB 推理一体脚本（修正版：自动兜底频率/幅度列名）
------------------------------------------------------------
流程：
1) 读取 Excel 模板频响
2) 仿真 N_NORMAL 条正常样本
3) 统计 μ/σ
4) 仿真 N_FAULT 条故障样本（幅度/频率/参考电平故障）
5) 写出原始曲线、labels、statistics
6) 基于仿真曲线提取系统特征 -> BRB 两层推理 -> 输出 features_brb.csv

依赖：
- pandas, numpy, scipy, openpyxl
- FMFD.baseline.baseline: compute_rrs_bounds
- FMFD.baseline.config: BAND_RANGES, K_LIST
- FMFD.features.extract: extract_system_features
- FMFD.BRB.system_brb: system_level_infer
- FMFD.BRB.module_brb: module_level_infer
"""

from pathlib import Path
import json
import re
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from FMFD.baseline.baseline import compute_rrs_bounds
from FMFD.baseline.config import BAND_RANGES, K_LIST
from FMFD.features.extract import extract_system_features
from FMFD.BRB.system_brb import system_level_infer
from FMFD.BRB.module_brb import module_level_infer

# ================= 配置 =================
FREQ_XLSX      = r"E:\Spectrum AnalyzerData\端口1_频响.xlsx"  # 模板频响
FREQ_SHEET     = 0
FREQ_CHANNEL   = "OFF-AC-扫频"  # 若不存在，退回第二列为幅度

OUT_DIR        = Path("./sim_spectrum")
N_NORMAL       = 50
N_FAULT        = 225
RANDOM_SEED    = 20251204
rng = np.random.RandomState(RANDOM_SEED)
# ======================================


def parse_freq_value(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x,(int,float,np.integer,np.floating)):
        return float(x)
    s = str(x).strip().replace(',','')
    m = re.match(r'^([+-]?\d+(\.\d+)?)(?:[eE]([+-]?\d+))?\s*([kKmMgG]?)(?:h?z)?$', s)
    if m:
        base = float(m.group(1))
        expo = int(m.group(3)) if m.group(3) else 0
        suf = m.group(4).lower() if m.group(4) else ''
        val = base * (10 ** expo)
        if suf == 'k': val *= 1e3
        if suf == 'm': val *= 1e6
        if suf == 'g': val *= 1e9
        return float(val)
    m2 = re.search(r'([+-]?\d+(\.\d+)?)', s)
    if m2:
        num = float(m2.group(1))
        if re.search(r'khz', s, re.I) or re.search(r'k$', s, re.I): return num*1e3
        if re.search(r'mhz', s, re.I) or re.search(r'm$', s, re.I): return num*1e6
        if re.search(r'ghz', s, re.I) or re.search(r'g$', s, re.I): return num*1e9
        return num
    try:
        return float(s)
    except:
        return np.nan


def read_freq_baseline(xlsx_path, sheet=0, channel=FREQ_CHANNEL):
    df = pd.read_excel(xlsx_path, sheet_name=sheet, engine="openpyxl")
    if channel not in df.columns:
        if df.shape[1] >= 2:
            amp_col = df.columns[1]
            print(f"警告：找不到列名'{channel}'，退回第二列 '{amp_col}' 作为幅度列")
        else:
            raise RuntimeError("Excel没有足够的列用于频率和幅度")
    else:
        amp_col = channel
    freq_col = df.columns[0]

    baseline = df[[freq_col, amp_col]].copy()
    baseline.iloc[:,0] = baseline.iloc[:,0].apply(parse_freq_value)
    baseline = baseline.dropna(subset=[baseline.columns[0]])
    baseline = baseline.sort_values(by=baseline.columns[0]).reset_index(drop=True)

    freqs = baseline.iloc[:,0].values.astype(float)
    amps_db = baseline.iloc[:,1].astype(float).values
    return freqs, amps_db


def estimate_sigma_from_template(amp, window_frac=0.05):
    x = np.asarray(amp, dtype=float)
    n = len(x)
    w = max(11, int(round(n * window_frac)))
    if w % 2 == 0:
        w += 1
    half = w // 2
    pad = np.pad(x, (half, half), mode="edge")
    sig = np.zeros(n)
    for i in range(n):
        seg = pad[i:i+w]
        med = np.median(seg)
        mad = np.median(np.abs(seg - med))
        sig[i] = max(1e-6, 1.4826 * mad)
    return sig


def generate_normal_from_template(freqs, amp_template, n_normal=50):
    sigma0 = estimate_sigma_from_template(amp_template)
    normals = []
    for _ in range(n_normal):
        alpha_slow = max(0.0, rng.normal(0.3, 0.1))
        logf = np.log10(np.maximum(freqs, 1e-12))
        coefs = np.polyfit(logf, rng.normal(0, 1, len(freqs)), deg=2)
        drift_raw = np.polyval(coefs, logf)
        drift_raw = (drift_raw - drift_raw.mean()) / (drift_raw.std() + 1e-6)
        slow = alpha_slow * sigma0 * drift_raw

        alpha_fine = max(0.0, rng.normal(0.8, 0.2))
        fine = alpha_fine * sigma0 * rng.normal(0, 1, len(freqs))

        amp = amp_template + slow + fine
        normals.append(amp)
    normals = np.stack(normals, axis=0)
    return freqs, normals


def compute_mu_sigma(amps_normal):
    mu = np.mean(amps_normal, axis=0)
    sigma = np.std(amps_normal, axis=0, ddof=1)
    sigma = np.maximum(sigma, 1e-6)
    return mu, sigma


def apply_amplitude_fault(freqs, baseline):
    amp = baseline.copy()
    faults = []

    if rng.rand() < 0.7:
        severity = rng.choice(["light", "medium", "heavy"])
        delta = {"light": rng.uniform(-1.5,-0.5),
                 "medium": rng.uniform(-3.0,-1.5),
                 "heavy": rng.uniform(-4.0,-3.0)}[severity]
        amp += delta
        faults.append({"module": "Attenuator/Gain", "type": "amplitude_offset",
                       "severity": severity, "delta_db": float(delta)})

    if rng.rand() < 0.6:
        severity = rng.choice(["light", "medium", "heavy"])
        pk2pk = {"light": rng.uniform(0.1,0.25),
                 "medium": rng.uniform(0.25,0.4),
                 "heavy": rng.uniform(0.4,0.7)}[severity]
        fmin, fmax = freqs[0], freqs[-1]
        bw = fmax - fmin
        f_start = rng.uniform(fmin, fmax - 0.2*bw)
        f_end   = f_start + rng.uniform(0.1*bw, 0.3*bw)
        seg = (freqs >= f_start) & (freqs <= f_end)
        period = rng.uniform((f_end-f_start)/5, (f_end-f_start)/2)
        phase = rng.uniform(0, 2*np.pi)
        ripple = 0.5 * pk2pk * np.sin(2*np.pi*(freqs-f_start)/period + phase)
        amp[seg] += ripple[seg]
        faults.append({"module": "PreAmp/IFAmp", "type": "ripple",
                       "severity": severity, "pk2pk_db": float(pk2pk),
                       "f_start_Hz": float(f_start), "f_end_Hz": float(f_end)})

    if rng.rand() < 0.5:
        n_spikes = rng.randint(1, 4)
        for _ in range(n_spikes):
            idx = rng.randint(0, len(freqs))
            f0 = freqs[idx]
            severity = rng.choice(["light", "medium", "heavy"])
            d = {"light": rng.uniform(1.0,2.0),
                 "medium": rng.uniform(2.0,3.0),
                 "heavy": rng.uniform(3.0,4.0)}[severity]
            width = (freqs[-1] - freqs[0]) * 1e-3
            gauss = d * np.exp(-0.5 * ((freqs - f0) / (width/2.355))**2)
            amp += gauss
            faults.append({"module": "Connector/Match", "type": "point_spike",
                           "severity": severity, "center_Hz": float(f0),
                           "delta_db": float(d)})
    return amp, faults


def apply_frequency_fault(freqs, baseline):
    severity = rng.choice(["light", "medium", "heavy"])
    eps = {"light": rng.uniform(-50e-6, 50e-6),
           "medium": rng.uniform(-100e-6, 100e-6),
           "heavy": rng.uniform(-150e-6, 150e-6)}[severity]
    f_true = freqs * (1.0 + eps)
    interp = interp1d(f_true, baseline, bounds_error=False,
                      fill_value=(baseline[0], baseline[-1]))
    amp_fault = interp(freqs)
    faults = [{"module": "LO/Clock", "type": "freq_scale",
               "severity": severity, "eps": float(eps)}]
    return amp_fault, faults


def apply_reference_fault(freqs, baseline):
    severity = rng.choice(["light", "medium", "heavy"])
    delta = {"light": rng.uniform(0.5,1.0),
             "medium": rng.uniform(1.0,1.5),
             "heavy": rng.uniform(1.5,2.5)}[severity]
    amp_fault = baseline + delta
    faults = [{"module": "CalSource/RefAmp", "type": "ref_level",
               "severity": severity, "delta_db": float(delta)}]
    return amp_fault, faults


def quick_features(freqs, amp, mu):
    feats = {}
    feats["amp_offset_db"] = float(np.median(amp - mu))
    mask = (freqs > 0) & np.isfinite(amp)
    if mask.sum() > 3:
        logf = np.log10(freqs[mask])
        coefs = np.polyfit(logf, amp[mask], 1)
        slope = coefs[0]
        trend = np.polyval(coefs, logf)
        ripple = amp[mask] - trend
        feats["slope_db_per_decade"] = float(slope)
        feats["ripple_rms_db"] = float(np.std(ripple))
    else:
        feats["slope_db_per_decade"] = 0.0
        feats["ripple_rms_db"] = 0.0
    feats["flatness_db"] = float(np.std(amp - np.median(amp)))
    return feats


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    freqs, amp_template = read_freq_baseline(FREQ_XLSX, sheet=FREQ_SHEET, channel=FREQ_CHANNEL)
    print(f"模板频点数={len(freqs)}，范围：{freqs[0]/1e6:.3f} MHz ~ {freqs[-1]/1e9:.3f} GHz")

    freqs, normal_amps = generate_normal_from_template(freqs, amp_template, n_normal=N_NORMAL)
    print(f"已生成正常样本: {N_NORMAL}")

    mu, sigma = compute_mu_sigma(normal_amps)
    print("已计算 μ(f) / σ(f)")

    labels = {}
    feats_quick = []

    for i in range(N_NORMAL):
        sid = f"normal_{i:03d}"
        amp = normal_amps[i]
        pd.DataFrame({"freq_Hz": freqs, "amplitude_dB": amp}).to_csv(OUT_DIR / f"{sid}.csv", index=False, encoding="utf-8")
        labels[sid] = {"type": "normal", "faults": []}
        fts = quick_features(freqs, amp, mu); fts["sample_id"] = sid
        feats_quick.append(fts)

    for k in range(N_FAULT):
        sid = f"fault_{k:03d}"
        baseline = normal_amps[rng.randint(0, N_NORMAL)].copy()
        sys_fault_class = rng.choice(["amp_error", "freq_error", "ref_error"], p=[0.45, 0.30, 0.25])
        if sys_fault_class == "amp_error":
            amp_fault, flist = apply_amplitude_fault(freqs, baseline)
        elif sys_fault_class == "freq_error":
            amp_fault, flist = apply_frequency_fault(freqs, baseline)
        else:
            amp_fault, flist = apply_reference_fault(freqs, baseline)

        pd.DataFrame({"freq_Hz": freqs, "amplitude_dB": amp_fault}).to_csv(OUT_DIR / f"{sid}.csv", index=False, encoding="utf-8")
        labels[sid] = {"type": "fault", "system_fault_class": sys_fault_class, "faults": flist}
        fts = quick_features(freqs, amp_fault, mu); fts["sample_id"] = sid
        feats_quick.append(fts)
        if (k+1) % 50 == 0:
            print(f"已生成故障样本 {k+1}/{N_FAULT}")

    with open(OUT_DIR / "labels.json", "w", encoding="utf-8") as f:
        json.dump(labels, f, indent=2, ensure_ascii=False)
    pd.DataFrame(feats_quick).to_csv(OUT_DIR / "features_quick.csv", index=False, encoding="utf-8")
    stats = {"freq_Hz": freqs.tolist(), "mu_dB": mu.tolist(), "sigma_dB": sigma.tolist(),
             "n_normal": int(N_NORMAL), "n_fault": int(N_FAULT)}
    with open(OUT_DIR / "statistics.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print("曲线/标签/统计已写出")

    # --- BRB 特征 + 推理 ---
    rrs, bounds = compute_rrs_bounds(freqs, normal_amps, BAND_RANGES, K_LIST)

    rows = []
    for csv_path in sorted(OUT_DIR.glob("*.csv")):
        df = pd.read_csv(csv_path)
        # 自动兜底列名
        freq_col = "freq_Hz" if "freq_Hz" in df.columns else df.columns[0]
        if "amplitude_dB" in df.columns:
            amp_col = "amplitude_dB"
        else:
            amp_col = df.columns[1] if df.shape[1] >= 2 else df.columns[0]

        freq = df[freq_col].to_numpy(dtype=float)
        amp = df[amp_col].to_numpy(dtype=float)

        # 如果频点与模板不一致，必要时可插值到 freqs：
        # if not np.allclose(freq, freqs):
        #     amp = np.interp(freqs, freq, amp)
        #     freq = freqs

        feats = extract_system_features(freqs, rrs, bounds, BAND_RANGES, amp)
        sys_p = system_level_infer(feats)
        mod_p = module_level_infer(feats, sys_p)

        row = {"sample_id": csv_path.stem, "source_csv": str(csv_path)}
        row.update(feats)
        row.update({f"sys_{k}": v for k, v in sys_p.items()})
        row.update({f"mod_{k}": v for k, v in mod_p.items()})
        if row["sample_id"] in labels:
            lab = labels[row["sample_id"]]
            if isinstance(lab, dict):
                for k, v in lab.items():
                    row[f"label_{k}"] = v
            else:
                row["label"] = lab
        rows.append(row)

    out_brb = OUT_DIR / "features_brb.csv"
    pd.DataFrame(rows).to_csv(out_brb, index=False, encoding="utf-8")
    print(f"BRB 特征+概率已写出: {out_brb} (samples={len(rows)})")


if __name__ == "__main__":
    main()