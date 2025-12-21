"""
optimize_brb.py

使用 CMA-ES 优化 BRB 参数（rule_weights, attr_weights, alpha_uncertainty）。
支持监督优化（需要 label_col 指定模块标签）或无监督（熵 + top1 置信度）。

注意：优化在大数据或大量迭代时耗时较长。建议先用小数据和较小的 maxiter 调试。
"""

import numpy as np
import pandas as pd
import yaml
import time
from brb_engine import load_rules, pack_params, unpack_params, inference
import cma
from sklearn.metrics import log_loss, accuracy_score

def sample_entropy(probs: np.ndarray):
    eps = 1e-12
    ent = -np.sum(probs * np.log(np.clip(probs, eps, 1.0)), axis=1)
    return ent

def supervised_objective(x, meta, rules_doc, df, indicator_map, label_col):
    params = unpack_params(x, meta)
    # map params to rules_doc? inference uses params as override alpha only; we will pass params via params dict to inference
    out = inference(df, rules_doc, indicator_map, params={"alpha_uncertainty": params["alpha_uncertainty"]})
    modules = rules_doc["modules_order"]
    # get probs
    prob_cols = [f"belief__{m}" for m in modules]
    probs = out[prob_cols].values
    y = df[label_col].values
    if y.dtype.kind in {"U", "S", "O"}:
        idx_map = {m:i for i,m in enumerate(modules)}
        y_idx = np.array([idx_map.get(v, -1) for v in y])
        mask = (y_idx >= 0)
        if not mask.any():
            return 1e6
    else:
        y_idx = y.astype(int); mask = np.ones_like(y_idx, dtype=bool)
    probs_valid = probs[mask]
    y_valid = y_idx[mask]
    try:
        loss = log_loss(y_valid, probs_valid, labels=list(range(len(modules))))
    except Exception:
        loss = 1.0 - accuracy_score(y_valid, np.argmax(probs_valid, axis=1))
    return float(loss)

def unsupervised_objective(x, meta, rules_doc, df, indicator_map, w_entropy=0.6, w_conf=0.4):
    params = unpack_params(x, meta)
    out = inference(df, rules_doc, indicator_map, params={"alpha_uncertainty": params["alpha_uncertainty"]})
    modules = rules_doc["modules_order"]
    prob_cols = [f"belief__{m}" for m in modules]
    probs = out[prob_cols].values
    ent = sample_entropy(probs)
    mean_ent = float(np.nanmean(ent))
    mean_top1 = float(np.nanmean(np.max(probs, axis=1)))
    obj = w_entropy * mean_ent + w_conf * (1.0 - mean_top1)
    # Add regularization to prevent trivial scaling (optional)
    return float(obj)

def optimize_rules(rules_path: str, df: pd.DataFrame, indicator_map: dict, label_col: str = None, supervised: bool = False, maxiter: int = 100, popsize: int = None):
    rules_doc = load_rules(rules_path)
    x0, meta = pack_params(rules_doc)
    sigma0 = 0.2
    opts = {}
    if popsize:
        opts["popsize"] = popsize
    es = cma.CMAEvolutionStrategy(x0, sigma0, opts)
    best_x = None
    best_obj = float("inf")
    t0 = time.time()
    it = 0
    while not es.stop() and it < maxiter:
        sols = es.ask()
        objs = []
        for x in sols:
            try:
                if supervised and label_col:
                    obj = supervised_objective(x, meta, rules_doc, df, indicator_map, label_col)
                else:
                    obj = unsupervised_objective(x, meta, rules_doc, df, indicator_map)
            except Exception:
                obj = float("inf")
            objs.append(obj)
            if obj < best_obj:
                best_obj = obj
                best_x = x.copy()
        es.tell(sols, objs)
        es.disp()
        it += 1
    if best_x is None:
        best_x = es.result.xbest
    params_opt = unpack_params(best_x, meta)
    # apply parameters to rules_doc and save
    for rname in rules_doc["rules"].keys():
        if rname in params_opt.get("rule_weights", {}):
            rules_doc["rules"][rname]["rule_weight"] = float(params_opt["rule_weights"][rname])
        if rname in params_opt.get("attr_weights", {}):
            rules_doc["rules"][rname]["attr_weight"] = float(params_opt["attr_weights"][rname])
    rules_doc["alpha_uncertainty"] = float(params_opt.get("alpha_uncertainty", rules_doc.get("alpha_uncertainty", 0.85)))
    out_path = rules_path.replace(".yaml", "_optimized.yaml")
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(rules_doc, f, allow_unicode=True)
    print("[INFO] Optimization finished, best_obj=", best_obj, "saved:", out_path, "time:", time.time()-t0)
    return out_path, params_opt, best_obj