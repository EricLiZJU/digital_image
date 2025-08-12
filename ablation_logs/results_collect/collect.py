import os
import json
from collections import defaultdict
import pandas as pd
import numpy as np

def collect_logs(root_dir="logs"):
    all_results = {}

    for ablation_name in os.listdir(root_dir):  # baseline, wo_bn, ...
        ablation_path = os.path.join(root_dir, ablation_name)
        if not os.path.isdir(ablation_path):
            continue

        all_results[ablation_name] = {}

        for dataset_name in os.listdir(ablation_path):  # Indian_pines, etc.
            dataset_path = os.path.join(ablation_path, dataset_name)
            if not os.path.isdir(dataset_path):
                continue

            results_per_seed = []
            for seed_dir in sorted(os.listdir(dataset_path)):
                seed_path = os.path.join(dataset_path, seed_dir)
                if not os.path.isdir(seed_path):
                    continue

                try:
                    with open(os.path.join(seed_path, "metrics.json")) as f:
                        metrics = json.load(f)
                    with open(os.path.join(seed_path, "time_log.json")) as f:
                        time_log = json.load(f)
                    with open(os.path.join(seed_path, "model_profile.json")) as f:
                        profile = json.load(f)
                except Exception as e:
                    print(f"⚠️ Error in {seed_path}: {e}")
                    continue

                results_per_seed.append({
                    "seed": metrics.get("seed"),
                    "OA": metrics.get("overall_accuracy"),
                    "AA": metrics.get("average_accuracy"),
                    "Kappa": metrics.get("kappa"),
                    "Train Time (s)": time_log.get("train_time(s)"),
                    "FLOPs(M)": profile.get("FLOPs(M)"),
                    "Params(K)": profile.get("Params(K)")
                })

            # 计算均值与标准差
            if results_per_seed:
                df = pd.DataFrame(results_per_seed)
                means = df.mean(numeric_only=True).round(4).to_dict()
                stds = df.std(numeric_only=True).round(4).to_dict()

                all_results[ablation_name][dataset_name] = {
                    "average": means,
                    "std": stds,
                    "seeds": results_per_seed
                }

    return all_results


if __name__ == "__main__":
    result = collect_logs("ablation_logs")  # 修改为你的主目录名
    with open("ablation_logs/results_collect/summary_all_logs.json", "w") as f:
        json.dump(result, f, indent=2)
    print("✅ Saved summary_all_logs.json")

    # 生成表格
    rows = []
    for ablation_name, datasets in result.items():
        for dataset_name, res in datasets.items():
            mean = res["average"]
            std = res["std"]
            rows.append({
                "Ablation": ablation_name,
                "Dataset": dataset_name,
                "OA": f"{mean.get('OA', '')} ± {std.get('OA', '')}",
                "AA": f"{mean.get('AA', '')} ± {std.get('AA', '')}",
                "Kappa": f"{mean.get('Kappa', '')} ± {std.get('Kappa', '')}",
                "Train Time(s)": f"{mean.get('Train Time (s)', '')} ± {std.get('Train Time (s)', '')}",
                "FLOPs(M)": mean.get("FLOPs(M)"),
                "Params(K)": mean.get("Params(K)")
            })

    df = pd.DataFrame(rows)
    df = df.sort_values(by=["Dataset", "Ablation"])

    df.to_csv("ablation_logs/results_collect/summary_table.csv", index=False)
    print("✅ Saved CSV to summary_table.csv")

    with open("ablation_logs/results_collect/summary_table.md", "w") as f:
        f.write(df.to_markdown(index=False))
    print("✅ Saved Markdown table to summary_table.md")