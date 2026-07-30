"""Summarize 260730 innovation ablation outputs.

Reads DeepAccident summary/ranking CSVs and SUMO per_run CSVs, then writes a
single CSV that can be copied into docs/260730_四创新点消融实验备份说明.md.
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


METHODS = [
    "Innov-NONE",
    "Innov-V",
    "Innov-D",
    "Innov-M",
    "Innov-H",
    "Innov-V+D",
    "Innov-V+M",
    "Innov-V+H",
    "Innov-D+M",
    "Innov-D+H",
    "Innov-M+H",
    "Innov-V+D+M+H",
]


FLAGS = {
    "Innov-NONE": (0, 0, 0, 0),
    "Innov-V": (1, 0, 0, 0),
    "Innov-D": (0, 1, 0, 0),
    "Innov-M": (0, 0, 1, 0),
    "Innov-H": (0, 0, 0, 1),
    "Innov-V+D": (1, 1, 0, 0),
    "Innov-V+M": (1, 0, 1, 0),
    "Innov-V+H": (1, 0, 0, 1),
    "Innov-D+M": (0, 1, 1, 0),
    "Innov-D+H": (0, 1, 0, 1),
    "Innov-M+H": (0, 0, 1, 1),
    "Innov-V+D+M+H": (1, 1, 1, 1),
}


def read_deepaccident(path: Path) -> dict[str, dict[str, str]]:
    out = {m: {} for m in METHODS}
    summary = path / "summary.csv"
    ranking = path / "ranking.csv"
    if summary.exists():
        with summary.open(newline="") as f:
            for row in csv.DictReader(f):
                method = row["method"]
                if method in out:
                    out[method].update({
                        "da_wpc": row["WPC%"],
                        "da_faf": row["FA(f)"],
                        "da_det": row["Det(s)"],
                    })
    if ranking.exists():
        with ranking.open(newline="") as f:
            for row in csv.DictReader(f):
                method = row["method"]
                if method in out:
                    out[method]["da_rank"] = row["rank"]
    return out


def read_sumo(paths: list[Path]) -> dict[str, dict[str, str]]:
    agg = defaultdict(lambda: {"n": 0, "coll": 0, "overlap": 0, "sev": 0.0, "wp": 0.0, "total": 0.0})
    for path in paths:
        per_run = path / "per_run.csv"
        if not per_run.exists():
            continue
        with per_run.open(newline="") as f:
            for row in csv.DictReader(f):
                method = row["method"]
                if method not in METHODS:
                    continue
                a = agg[method]
                a["n"] += 1
                a["coll"] += int(row["collision"])
                a["overlap"] += int(row["overlap"])
                a["sev"] += float(row["severity"])
                a["wp"] += float(row["wp_coll"])
                a["total"] += float(row["wp_total"])

    out = {m: {} for m in METHODS}
    for method, a in agg.items():
        n = max(a["n"], 1)
        out[method] = {
            "sumo_runs": str(a["n"]),
            "sumo_coll": f"{a['coll'] / n:.6f}",
            "sumo_overlap": f"{a['overlap'] / n:.6f}",
            "sumo_avgsev": f"{a['sev'] / n:.6f}",
            "sumo_wpc": f"{a['wp'] / max(a['total'], 1):.6f}",
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deepaccident-dir", type=Path, default=Path("results/260730_innovation_ablation_deepaccident_full"))
    parser.add_argument("--sumo-dirs", type=Path, nargs="*", default=[
        Path("results/260730_innovation_ablation_sumo_base_smoke"),
        Path("results/260730_innovation_ablation_sumo_stress"),
        Path("results/260730_innovation_ablation_sumo_random_stress"),
    ])
    parser.add_argument("--out", type=Path, default=Path("results/260730_innovation_ablation_summary_combined.csv"))
    args = parser.parse_args()

    da = read_deepaccident(args.deepaccident_dir)
    sumo = read_sumo(args.sumo_dirs)
    rows = []
    for method in METHODS:
        v, d, m, h = FLAGS[method]
        rows.append({
            "method": method,
            "V": v,
            "D": d,
            "M": m,
            "H": h,
            "DA_WPC": da[method].get("da_wpc", ""),
            "DA_FA(f)": da[method].get("da_faf", ""),
            "DA_Det(s)": da[method].get("da_det", ""),
            "DA_Rank": da[method].get("da_rank", ""),
            "SUMO_runs": sumo[method].get("sumo_runs", ""),
            "SUMO_CollRate": sumo[method].get("sumo_coll", ""),
            "SUMO_OverlapRate": sumo[method].get("sumo_overlap", ""),
            "SUMO_AvgSev": sumo[method].get("sumo_avgsev", ""),
            "SUMO_WPC": sumo[method].get("sumo_wpc", ""),
        })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
