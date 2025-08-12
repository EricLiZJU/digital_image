#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import csv
import math
from pathlib import Path
from statistics import mean

ROOT = Path("logs")
OUTDIR = ROOT / "results_collect"
OUTDIR.mkdir(parents=True, exist_ok=True)

# 允许的指标字段名（兼容大小写或不同写法时可扩展）
METRIC_KEYS = {
    "oa": ["overall_accuracy", "OA", "overall", "overall-accuracy"],
    "aa": ["average_accuracy", "AA", "avg_accuracy", "average-accuracy"],
    "kappa": ["kappa", "cohen_kappa", "kappa_score"],
    "per_class": ["per_class_accuracy", "per-class-accuracy", "class_acc"],
}

def _get_first_key(d, keys):
    for k in keys:
        if k in d:
            return k
    # 容忍大小写差异
    low = {k.lower(): k for k in d.keys()}
    for k in keys:
        if k.lower() in low:
            return low[k.lower()]
    return None

def _to_pct(x):
    # 将 0~1 的小数转为百分数；若本来是百分数(>1)，保持不变
    if x is None:
        return None
    try:
        x = float(x)
    except Exception:
        return None
    if 0.0 <= x <= 1.0:
        return x * 100.0
    return x

def read_seed_metrics(seed_dir: Path):
    """读取单个 seed 的 metrics.json 与 model_profile.json"""
    mfile = seed_dir / "metrics.json"
    pfile = seed_dir / "model_profile.json"
    if not mfile.exists():
        return None

    with mfile.open("r", encoding="utf-8") as f:
        m = json.load(f)

    oa_k = _get_first_key(m, METRIC_KEYS["oa"])
    aa_k = _get_first_key(m, METRIC_KEYS["aa"])
    kp_k = _get_first_key(m, METRIC_KEYS["kappa"])
    pc_k = _get_first_key(m, METRIC_KEYS["per_class"])

    oa = _to_pct(m.get(oa_k)) if oa_k else None
    aa = _to_pct(m.get(aa_k)) if aa_k else None
    kp = float(m.get(kp_k)) if kp_k in m else None
    pc = m.get(pc_k) if pc_k else None

    flops = params = None
    if pfile.exists():
        with pfile.open("r", encoding="utf-8") as f:
            p = json.load(f)
        # 兼容几种常见写法
        for key in ["FLOPs(M)", "FLOPs", "flops(M)", "FLOPs_millions"]:
            if key in p:
                flops = float(p[key])
                break
        for key in ["Params(K)", "Params", "params(K)", "Params_thousands"]:
            if key in p:
                params = float(p[key])
                break

    return {
        "OA": oa,
        "AA": aa,
        "Kappa": kp,
        "per_class_accuracy": pc,
        "FLOPs(M)": flops,
        "Params(K)": params,
    }

def aggregate_dataset(ds_dir: Path):
    """聚合单个数据集目录下所有 seed 的结果"""
    seed_dirs = sorted([p for p in ds_dir.iterdir() if p.is_dir() and p.name.startswith("seed_")])
    if not seed_dirs:
        return None

    oa_list, aa_list, kp_list = [], [], []
    flops_list, params_list = [], []
    pc_lists = []  # list of list

    base_len = None
    used_seeds = 0

    for sd in seed_dirs:
        rec = read_seed_metrics(sd)
        if not rec:
            continue

        if rec["OA"] is not None: oa_list.append(rec["OA"])
        if rec["AA"] is not None: aa_list.append(rec["AA"])
        if rec["Kappa"] is not None: kp_list.append(rec["Kappa"])
        if rec["FLOPs(M)"] is not None: flops_list.append(rec["FLOPs(M)"])
        if rec["Params(K)"] is not None: params_list.append(rec["Params(K)"])

        pc = rec["per_class_accuracy"]
        if isinstance(pc, list) and all(isinstance(x, (int, float)) for x in pc):
            if base_len is None:
                base_len = len(pc)
                pc_lists.append(pc)
                used_seeds += 1
            else:
                if len(pc) == base_len:
                    pc_lists.append(pc)
                    used_seeds += 1
                else:
                    print(f"[WARN] {ds_dir.name}/{sd.name} 的 per_class_accuracy 长度({len(pc)})"
                          f" != {base_len}，已跳过该 seed。")
        else:
            # 没有 per-class 的也允许通过
            pass

    if not oa_list and not aa_list and not kp_list:
        return None

    # 逐类平均
    pc_mean = None
    if pc_lists:
        cols = list(zip(*pc_lists))  # [[c1_of_seed1,...],[c2_of_seed1,...],...]
        pc_mean = [mean(col) * (100.0 if (0.0 <= col[0] <= 1.0) else 1.0) for col in cols]

    row = {
        "dataset": ds_dir.name,
        "OA": round(mean(oa_list), 2) if oa_list else None,
        "AA": round(mean(aa_list), 2) if aa_list else None,
        "Kappa": round(mean(kp_list), 4) if kp_list else None,
        "per_class_accuracy": pc_mean,  # 先存数组，写出时再格式化
        "FLOPs(M)": round(mean(flops_list), 3) if flops_list else None,
        "Params(K)": round(mean(params_list), 3) if params_list else None,
    }
    return row

def format_pc_for_csv(pc):
    if pc is None:
        return ""
    # 用分号分隔，保留两位小数
    vals = []
    for x in pc:
        x = _to_pct(x) if (0.0 <= x <= 1.0) else x
        vals.append(f"{x:.2f}")
    return ";".join(vals)

def format_pc_for_md(pc):
    # 为了表格可读，依旧用分号分隔
    return format_pc_for_csv(pc)

def find_dataset_dirs(root: Path):
    # 过滤掉 results_collect 等非数据集目录
    skip = {"results_collect",}
    return [p for p in root.iterdir() if p.is_dir() and p.name not in skip]

def main():
    ds_dirs = find_dataset_dirs(ROOT)
    rows = []
    for ds in ds_dirs:
        agg = aggregate_dataset(ds)
        if agg:
            rows.append(agg)
        else:
            print(f"[INFO] 跳过空数据集：{ds.name}")

    # 排序：按数据集名
    rows.sort(key=lambda r: r["dataset"].lower())

    # 写 CSV
    csv_path = OUTDIR / "summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "OA(%)", "AA(%)", "Kappa", "per_class_accuracy(%;semicolon-separated)", "FLOPs(M)", "Params(K)"])
        for r in rows:
            writer.writerow([
                r["dataset"],
                f"{r['OA']:.2f}" if r["OA"] is not None else "",
                f"{r['AA']:.2f}" if r["AA"] is not None else "",
                f"{r['Kappa']:.4f}" if r["Kappa"] is not None else "",
                format_pc_for_csv(r["per_class_accuracy"]),
                f"{r['FLOPs(M)']:.3f}" if r["FLOPs(M)"] is not None else "",
                f"{r['Params(K)']:.3f}" if r["Params(K)"] is not None else "",
            ])

    # 写 Markdown
    md_path = OUTDIR / "summary.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write("| dataset | OA(%) | AA(%) | Kappa | per_class_accuracy (%; ; separated) | FLOPs(M) | Params(K) |\n")
        f.write("|---|---:|---:|---:|---|---:|---:|\n")
        for r in rows:
            f.write("| {dataset} | {OA} | {AA} | {Kappa} | {PC} | {FLOPs} | {Params} |\n".format(
                dataset=r["dataset"],
                OA=f"{r['OA']:.2f}" if r["OA"] is not None else "",
                AA=f"{r['AA']:.2f}" if r["AA"] is not None else "",
                Kappa=f"{r['Kappa']:.4f}" if r["Kappa"] is not None else "",
                PC=format_pc_for_md(r["per_class_accuracy"]),
                FLOPs=f"{r['FLOPs(M)']:.3f}" if r["FLOPs(M)"] is not None else "",
                Params=f"{r['Params(K)']:.3f}" if r["Params(K)"] is not None else "",
            ))

    print(f"✅ Done. CSV: {csv_path} | MD: {md_path}")

if __name__ == "__main__":
    main()