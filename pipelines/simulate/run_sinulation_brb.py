#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
信号级仿真 + BRB 推理一体脚本（基于 run_babeline 产物，可回退 Excel 模板）
--------------------------------------------------------------------
更新点：
- 默认基线路径：./Output/baseline_artifacts.npz（相对仓库根）
- 默认输出目录：./Output/sim_spectrum
- repo_root = __file__.resolve().parents[2]（适配当前目录结构）
- 频点长度不一致：优先检查长度，不一致则插值对齐再推理
- 仅处理 normal_*.csv / fault_*.csv，跳过聚合类 CSV，避免字符串频率导致报错
- 调试打印：输出前 5 条 system_brb 原始概率，统计全 0 次数
- 防“sys_* 全 0”启发式兜底：用特征(gain/df/bias等)打分，再归一化
"""

import argparse
import json
import re
from pathlib import Path
from typing import Union, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from FMFD.baseline.baseline import compute_rrs_bounds
from FMFD.baseline.config import BAND_RANGES, K_LIST
from FMFD.features.extract import extract_system_features
from FMFD.BRB.system_brb import system_level_infer
from FMFD.BRB.module_brb import module_level_infer


def build_argparser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline_npz", default="./Output/baseline_artifacts.npz")
    ap.add_argument("--baseline_meta", default="./Output/baseline_meta.json")
    ap.add_argument("--baseline_excel", default=None)
    ap.add_argument("--freq_sheet", type=int, default=0)
    ap.add_argument("--freq_channel", default="OFF-AC-扫频")
    ap.add_argument("--out_dir", default="./Output/sim_spectrum")
    ap.add_argument("--n_normal", type=int, default=50)
    ap.add_argument("--n_fault", type=int, default=225)
    ap.add_argument("--seed", type=int, default=20251204)
    return ap


def resolve_path(repo_root: Path, p: Union[str, Path]) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (repo_root / p).resolve()


def parse_freq_value(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    s = str(x).strip().replace(",", "")
    m = re.match(r"^([+-]?\d+(\.\d+)?)(?:[eE]([+-]?\d+))?\s*([kKmMgG]?)(?:h?z)?$", s)
    if m:
        base = float(m.group(1))
        expo = int(m.group(3)) if m.group(3) else 0
        suf = m.group(4).lower() if m.group(4) else ""
        val = base * (10 ** expo)
        if suf == "k":
            val *= 1e3
        if suf == "m":
            val *= 1e6
        if suf == "g":
            val *= 1e9
        return float(val)
    m2 = re.search(r"([+-]?\d+(\.\d+)?)", s)
    if m2:
        num = float(m2.group(1))
        if re.search(r"khz", s, re.I) or re.search(r"k$", s, re.I):
            return num * 1e3
        if re.search(r"mhz", s, re.I) or re.search(r"m$", s, re.I):
            return num * 1e6
        if re.search(r"ghz", s, re.I) or re.search(r"g$", s, re.I):
            return num * 1e9
        return num
    try:
        return float(s)
    except Exception:
        return np.nan


def load_baseline_from_npz(path_npz: Path):
    if not path_npz.exists():
        return None
    data = np.load(path_npz, allow_pickle=True)
    freq = data["frequency"] if "frequency" in data else data["freq"]
    if "traces" not in data:
        raise RuntimeError("npz 文件缺少 traces，请先在 run_babeline.py 中保存 traces 后重跑")
    traces = data["traces"]
    return freq, traces


def read_freq_baseline(xlsx_path, sheet=0, channel="OFF-AC-扫频"):
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
    baseline.iloc[:, 0] = baseline.iloc[:, 0].apply(parse_freq_value)
    baseline = baseline.dropna(subset=[baseline.columns[0]])
    baseline = baseline.sort_values(by=baseline.columns[0]).reset_index(drop=True)

    freqs = baseline.iloc[:, 0].values.astype(float)
    amps_db = baseline.iloc[:, 1].astype(float).values
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
        seg = pad[i: i + w]
        med = np.median(seg)
        mad = np.median(np.abs(seg - med))
        sig[i] = max(1e-6, 1.4826 * mad)
    return sig


def augment_normals(freqs, base_traces, n_extra, rng):
    normals = list(base_traces)
    sigma0 = np.median([estimate_sigma_from_template(t) for t in base_traces], axis=0)
    for _ in range(n_extra):
        idx = rng.randint(0, len(base_traces))
        amp0 = base_traces[idx]
        alpha_slow = max(0.0, rng.normal(0.3, 0.1))
        logf = np.log10(np.maximum(freqs, 1e-12))
        coefs = np.polyfit(logf, rng.normal(0, 1, len(freqs)), deg=2)
        drift_raw = np.polyval(coefs, logf)
        drift_raw = (drift_raw - drift_raw.mean()) / (drift_raw.std() + 1e-6)
        slow = alpha_slow * sigma0 * drift_raw

        alpha_fine = max(0.0, rng.normal(0.8, 0.2))
        fine = alpha_fine * sigma0 * rng.normal(0, 1, len(freqs))

        normals.append(amp0 + slow + fine)
    return np.stack(normals, axis=0)


def compute_mu_sigma(amps_normal):
    mu = np.mean(amps_normal, axis=0)
    sigma = np.std(amps_normal, axis=0, ddof=1)
    sigma = np.maximum(sigma, 1e-6)
    return mu, sigma


def apply_amplitude_fault(freqs, baseline, rng):
    amp = baseline.copy()
    faults = []
    if rng.rand() < 0.7:
        severity = rng.choice(["light", "medium", "heavy"])
        delta = {"light": rng.uniform(-1.5, -0.5), "medium": rng.uniform(-3.0, -1.5), "heavy": rng.uniform(-4.0, -3.0)}[severity]
        amp += delta
        faults.append({"module": "Attenuator/Gain", "type": "amplitude_offset", "severity": severity, "delta_db": float(delta)})
    if rng.rand() < 0.6:
        severity = rng.choice(["light", "medium", "heavy"])
        pk2pk = {"light": rng.uniform(0.1, 0.25), "medium": rng.uniform(0.25, 0.4), "heavy": rng.uniform(0.4, 0.7)}[severity]
        fmin, fmax = freqs[0], freqs[-1]
        bw = fmax - fmin
        f_start = rng.uniform(fmin, fmax - 0.2 * bw)
        f_end = f_start + rng.uniform(0.1 * bw, 0.3 * bw)
        seg = (freqs >= f_start) & (freqs <= f_end)
        period = rng.uniform((f_end - f_start) / 5, (f_end - f_start) / 2)
        phase = rng.uniform(0, 2 * np.pi)
        ripple = 0.5 * pk2pk * np.sin(2 * np.pi * (freqs - f_start) / period + phase)
        amp[seg] += ripple[seg]
        faults.append(
            {
                "module": "PreAmp/IFAmp",
                "type": "ripple",
                "severity": severity,
                "pk2pk_db": float(pk2pk),
                "f_start_Hz": float(f_start),
                "f_end_Hz": float(f_end),
            }
        )
    if rng.rand() < 0.5:
        n_spikes = rng.randint(1, 4)
        for _ in range(n_spikes):
            idx = rng.randint(0, len(freqs))
            f0 = freqs[idx]
            severity = rng.choice(["light", "medium", "heavy"])
            d = {"light": rng.uniform(1.0, 2.0), "medium": rng.uniform(2.0, 3.0), "heavy": rng.uniform(3.0, 4.0)}[severity]
            width = (freqs[-1] - freqs[0]) * 1e-3
            gauss = d * np.exp(-0.5 * ((freqs - f0) / (width / 2.355)) ** 2)
            amp += gauss
            faults.append(
                {"module": "Connector/Match", "type": "point_spike", "severity": severity, "center_Hz": float(f0), "delta_db": float(d)}
            )
    return amp, faults


def apply_frequency_fault(freqs, baseline, rng):
    severity = rng.choice(["light", "medium", "heavy"])
    eps = {"light": rng.uniform(-50e-6, 50e-6), "medium": rng.uniform(-100e-6, 100e-6), "heavy": rng.uniform(-150e-6, 150e-6)}[
        severity
    ]
    f_true = freqs * (1.0 + eps)
    interp = interp1d(f_true, baseline, bounds_error=False, fill_value=(baseline[0], baseline[-1]))
    amp_fault = interp(freqs)
    faults = [{"module": "LO/Clock", "type": "freq_scale", "severity": severity, "eps": float(eps)}]
    return amp_fault, faults


def apply_reference_fault(freqs, baseline, rng):
    severity = rng.choice(["light", "medium", "heavy"])
    delta = {"light": rng.uniform(0.5, 1.0), "medium": rng.uniform(1.0, 1.5), "heavy": rng.uniform(1.5, 2.5)}[severity]
    amp_fault = baseline + delta
    faults = [{"module": "CalSource/RefAmp", "type": "ref_level", "severity": severity, "delta_db": float(delta)}]
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
        feats["ripple_rms_db"] = float(np.std(ripple))
        feats["slope_db_per_decade"] = float(slope)
    else:
        feats["ripple_rms_db"] = 0.0
        feats["slope_db_per_decade"] = 0.0
    feats["flatness_db"] = float(np.std(amp - np.median(amp)))
    return feats


def is_signal_csv(p: Path) -> bool:
    stem = p.stem.lower()
    return stem.startswith("normal_") or stem.startswith("fault_")


def heuristic_sys_from_feats(feats: Dict[str, float]) -> Dict[str, float]:
    # 简单启发式：用特征构造非均匀分数
    gain = abs(feats.get("gain", 0.0))
    df = abs(feats.get("df", 0.0))
    bias = abs(feats.get("bias", 0.0))
    ripple_var = feats.get("ripple_var", 0.0)
    viol = feats.get("viol_rate", 0.0)
    slope = abs(feats.get("res_slope", 0.0))
    amp_score = gain + 0.5 * ripple_var
    freq_score = df + viol + 0.2 * slope
    ref_score = bias
    scores = {"幅度失准": amp_score, "频率失准": freq_score, "参考电平失准": ref_score}
    s = sum(scores.values())
    if s <= 1e-9:
        return {k: 1.0 / len(scores) for k in scores}
    return {k: v / s for k, v in scores.items()}


def main():
    args = build_argparser().parse_args()

    repo_root = Path(__file__).resolve().parents[2]

    baseline_npz = resolve_path(repo_root, args.baseline_npz)
    baseline_meta = resolve_path(repo_root, args.baseline_meta)
    out_dir = resolve_path(repo_root, args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.RandomState(args.seed)

    baseline = load_baseline_from_npz(baseline_npz)
    if baseline:
        freqs, base_traces = baseline
        print(f"[INFO] 载入基线 npz: {baseline_npz}, 正常样本数={len(base_traces)}, 频点数={len(freqs)}")
    elif args.baseline_excel:
        freqs, amp_template = read_freq_baseline(args.baseline_excel, sheet=args.freq_sheet, channel=args.freq_channel)
        print(f"[INFO] 使用 Excel 模板: {args.baseline_excel}, 频点数={len(freqs)}")
        freqs, base_traces = freqs, np.expand_dims(amp_template, axis=0)
    else:
        raise RuntimeError(f"找不到 baseline_artifacts.npz（{baseline_npz}），且未提供 --baseline_excel")

    normals = augment_normals(freqs, base_traces, args.n_normal, rng)
    print(f"[INFO] 正常样本总数（含基线）：{len(normals)}")

    mu, sigma = compute_mu_sigma(normals)

    labels: Dict[str, Dict] = {}
    feats_quick: List[Dict] = []

    for i, amp in enumerate(normals):
        sid = f"normal_{i:03d}"
        pd.DataFrame({"freq_Hz": freqs, "amplitude_dB": amp}).to_csv(out_dir / f"{sid}.csv", index=False, encoding="utf-8")
        labels[sid] = {"type": "normal", "faults": []}
        fts = quick_features(freqs, amp, mu)
        fts["sample_id"] = sid
        feats_quick.append(fts)

    for k in range(args.n_fault):
        sid = f"fault_{k:03d}"
        baseline_amp = normals[rng.randint(0, len(normals))].copy()
        sys_fault_class = rng.choice(["amp_error", "freq_error", "ref_error"], p=[0.45, 0.30, 0.25])
        if sys_fault_class == "amp_error":
            amp_fault, flist = apply_amplitude_fault(freqs, baseline_amp, rng)
        elif sys_fault_class == "freq_error":
            amp_fault, flist = apply_frequency_fault(freqs, baseline_amp, rng)
        else:
            amp_fault, flist = apply_reference_fault(freqs, baseline_amp, rng)

        pd.DataFrame({"freq_Hz": freqs, "amplitude_dB": amp_fault}).to_csv(out_dir / f"{sid}.csv", index=False, encoding="utf-8")
        labels[sid] = {"type": "fault", "system_fault_class": sys_fault_class, "faults": flist}
        fts = quick_features(freqs, amp_fault, mu)
        fts["sample_id"] = sid
        feats_quick.append(fts)
        if (k + 1) % 50 == 0:
            print(f"[INFO] 已生成故障样本 {k+1}/{args.n_fault}")

    with open(out_dir / "labels.json", "w", encoding="utf-8") as f:
        json.dump(labels, f, indent=2, ensure_ascii=False)
    pd.DataFrame(feats_quick).to_csv(out_dir / "features_quick.csv", index=False, encoding="utf-8")
    stats = {
        "freq_Hz": freqs.tolist(),
        "mu_dB": mu.tolist(),
        "sigma_dB": sigma.tolist(),
        "n_normal": int(len(normals)),
        "n_fault": int(args.n_fault),
    }
    with open(out_dir / "statistics.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print("[INFO] 曲线/标签/统计已写出")

    rrs, bounds = compute_rrs_bounds(freqs, normals, BAND_RANGES, K_LIST)

    rows = []
    zero_cnt = 0
    for csv_path in sorted(out_dir.glob("*.csv")):
        if not is_signal_csv(csv_path):
            continue

        df = pd.read_csv(csv_path)
        if df.shape[1] < 2:
            continue

        freq_col = "freq_Hz" if "freq_Hz" in df.columns else df.columns[0]
        amp_col = "amplitude_dB" if "amplitude_dB" in df.columns else df.columns[1]

        freq_arr = pd.to_numeric(df[freq_col], errors="coerce").to_numpy()
        amp_arr = pd.to_numeric(df[amp_col], errors="coerce").to_numpy()

        if not np.isfinite(freq_arr).any() or not np.isfinite(amp_arr).any():
            continue

        if len(freq_arr) != len(freqs) or not np.allclose(freq_arr, freqs):
            amp_arr = np.interp(freqs, freq_arr, amp_arr)
            freq_arr = freqs

        feats = extract_system_features(freqs, rrs, bounds, BAND_RANGES, amp_arr)
        raw_sys_p = system_level_infer(feats)

        if len(rows) < 5:
            print("[DEBUG] raw_sys_p:", raw_sys_p)

        s_raw = sum(raw_sys_p.values())
        if s_raw <= 1e-9:
            zero_cnt += 1
            sys_p = heuristic_sys_from_feats(feats)
        else:
            sys_p = {k: v / s_raw for k, v in raw_sys_p.items()}

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

    out_brb = out_dir / "features_brb.csv"
    pd.DataFrame(rows).to_csv(out_brb, index=False, encoding="utf-8-sig")
    print(f"[INFO] BRB 特征+概率已写出: {out_brb} (samples={len(rows)}), sys_all_zero={zero_cnt}")
    if len(rows) > 0:
        print(f"[INFO] sys_all_zero_ratio = {zero_cnt/len(rows):.3f}")


if __name__ == "__main__":
    main()