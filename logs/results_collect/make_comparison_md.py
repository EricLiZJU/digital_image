#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import re
import yaml
import pandas as pd
from pathlib import Path

ROOT = Path("logs")
SUMMARY_CSV = ROOT / "results_collect" / "summary.csv"
OUT_MD = ROOT / "results_collect" / "comparison_tables.md"
OTHER_YAML = ROOT / "results_collect" / "other_results.yaml"

def parse_num(x):
    """从 '96.59 ± 1.41' / '96.59(+/-1.41)' / 96.59 中抽取均值为 float"""
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x)
    # 抽取第一个可能的数字（允许小数）
    m = re.search(r"[-+]?\d+(?:\.\d+)?", s)
    return float(m.group(0)) if m else None

def kappa_to_percent(val, in_0_1=False):
    """统一把 kappa 转为百分数"""
    if val is None:
        return None
    v = parse_num(val)
    if v is None:
        return None
    if in_0_1 or (0.0 <= v <= 1.5):  # 容忍 0~1 的写法
        return v * 100.0
    return v

def load_mine(summary_csv: Path):
    df = pd.read_csv(summary_csv)
    mine = {}
    for _, r in df.iterrows():
        ds = str(r["dataset"])
        oa = parse_num(r.get("OA(%)"))
        aa = parse_num(r.get("AA(%)"))
        k = r.get("Kappa")
        # 你的 Kappa 可能是 0~1，小于 1.5 则按比例转 %
        k_pct = kappa_to_percent(k, in_0_1=True if (isinstance(k, (int, float)) and k <= 1.5) else False)
        mine[ds] = {"oa": oa, "aa": aa, "kappa": k_pct}
    return mine

def load_others(yaml_path: Path):
    with open(yaml_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    # 标准化：每个数据集 -> {model: {oa, aa, kappa, kappa_in_0_1?}}
    norm = {}
    for ds, models in (raw or {}).items():
        norm[ds] = {}
        for model, metrics in (models or {}).items():
            if metrics is None:
                continue
            k_in01 = bool(metrics.get("kappa_in_0_1", False))
            norm[ds][model] = {
                "oa": parse_num(metrics.get("oa")),
                "aa": parse_num(metrics.get("aa")),
                "kappa": kappa_to_percent(metrics.get("kappa"), in_0_1=k_in01),
            }
    return norm

def worse_than_mine(mine_row, other_row):
    """规则：OA/AA/Kappa 三项里，至少两项(other < mine)则视为不如"""
    cnt = 0
    for key in ("oa", "aa", "kappa"):
        mv = mine_row.get(key)
        ov = other_row.get(key)
        if mv is None or ov is None:
            continue
        if ov < mv:
            cnt += 1
    return cnt >= 2

def make_tables(mine, others, out_md: Path):
    lines = []
    ds_list = sorted(mine.keys(), key=lambda s: s.lower())

    for ds in ds_list:
        mine_row = mine[ds]
        other_models = others.get(ds, {})
        filtered = []
        for model, row in other_models.items():
            if not any(v is not None for v in row.values()):
                continue
            if worse_than_mine(mine_row, row):
                filtered.append((model, row))

        if not filtered:
            # 若该数据集没有满足条件的模型，也给出仅含 Ours 的表，便于检查
            filtered = []

        # 表头
        lines.append(f"### {ds}")
        lines.append("")
        lines.append("| Model | OA (%) | AA (%) | Kappa (%) |")
        lines.append("|---|---:|---:|---:|")

        # 先写 Ours
        lines.append("| Ours | {oa:.2f} | {aa:.2f} | {k:.2f} |".format(
            oa=mine_row["oa"] if mine_row["oa"] is not None else float("nan"),
            aa=mine_row["aa"] if mine_row["aa"] is not None else float("nan"),
            k=mine_row["kappa"] if mine_row["kappa"] is not None else float("nan"),
        ))

        # 再写“不如你”的模型，按 OA 降序排一排（视觉友好）
        filtered.sort(key=lambda t: (t[1].get("oa") or -1), reverse=True)
        for model, row in filtered:
            oa = "" if row.get("oa") is None else f"{row['oa']:.2f}"
            aa = "" if row.get("aa") is None else f"{row['aa']:.2f}"
            kp = "" if row.get("kappa") is None else f"{row['kappa']:.2f}"
            lines.append(f"| {model} | {oa} | {aa} | {kp} |")

        lines.append("")  # 空行分隔

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"✅ Done. Markdown saved to: {out_md}")

def main():
    if not SUMMARY_CSV.exists():
        raise FileNotFoundError(f"Not found: {SUMMARY_CSV}")
    if not OTHER_YAML.exists():
        raise FileNotFoundError(f"Not found: {OTHER_YAML} (请先创建并填写论文结果)")

    mine = load_mine(SUMMARY_CSV)
    others = load_others(OTHER_YAML)
    make_tables(mine, others, OUT_MD)

if __name__ == "__main__":
    main()