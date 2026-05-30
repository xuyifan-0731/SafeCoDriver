#!/usr/bin/env python3
"""Audit DeepAccident real other_vehicle label alignment into ego frame."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

SAFECO_ROOT = Path("/raid/xuyifan/jiqiuyu")
sys.path.insert(0, str(SAFECO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

torch.set_num_threads(1)

from experiments.deepaccident_loader import DeepAccidentLoader
from prototype.real_coop import load_aligned_other_vehicle_message
from prototype.trust_calib import greedy_associate


def id_residuals(ego_perception, aligned_other):
    ego_by_id = {str(agent.state.id): agent for agent in ego_perception.agents}
    residuals = []
    visible_residuals = []
    invisible_residuals = []
    for agent in aligned_other.agents:
        match = ego_by_id.get(str(agent.state.id))
        if match is None:
            continue
        residual = float(np.hypot(agent.state.x - match.state.x, agent.state.y - match.state.y))
        residuals.append(residual)
        if match.is_visible:
            visible_residuals.append(residual)
        else:
            invisible_residuals.append(residual)
    return residuals, visible_residuals, invisible_residuals


def run(args):
    loader = DeepAccidentLoader(split="all", include_invisible=True, include_coop=True)
    ckpt = torch.load(SAFECO_ROOT / "models" / "collision_net_best.pt", map_location="cpu", weights_only=False)
    val_idx = list(ckpt.get("val_scenario_idx", []))
    available_val_scenarios = len(val_idx)
    if args.max_scenarios > 0:
        val_idx = val_idx[: args.max_scenarios]

    frame_rows = []
    all_id_residuals = []
    all_visible_residuals = []
    all_invisible_residuals = []
    all_greedy_residuals = []
    frames_with_real = 0
    frames = 0
    real_invisible_sum = 0

    for si in val_idx:
        scenario = loader.scenarios[si]
        n_frames = len(scenario["frames"])
        if args.max_frames_per_scenario > 0:
            n_frames = min(n_frames, args.max_frames_per_scenario)
        for fi in range(n_frames):
            frames += 1
            frame = loader.load_frame(si, fi)
            aligned = load_aligned_other_vehicle_message(loader, si, fi)
            if aligned is None:
                continue
            frames_with_real += 1
            residuals, visible_residuals, invisible_residuals = id_residuals(frame.perception, aligned)
            all_id_residuals.extend(residuals)
            all_visible_residuals.extend(visible_residuals)
            all_invisible_residuals.extend(invisible_residuals)
            real_invisible_sum += sum(1 for agent in aligned.agents if not agent.is_visible)

            matches = greedy_associate(frame.perception.agents, aligned.agents, max_dist=args.greedy_match_dist)
            greedy_residuals = [
                float(np.hypot(ego.state.x - coop.state.x, ego.state.y - coop.state.y))
                for ego, coop, _ in matches
            ]
            all_greedy_residuals.extend(greedy_residuals)

            if args.write_frame_metrics:
                frame_rows.append(
                    {
                        "scenario_index": si,
                        "frame_index": fi,
                        "scenario_name": scenario["name"],
                        "ego_agents": len(frame.perception.agents),
                        "real_other_agents": len(aligned.agents),
                        "real_other_ego_invisible": sum(1 for agent in aligned.agents if not agent.is_visible),
                        "id_matches": len(residuals),
                        "id_residual_median": float(np.median(residuals)) if residuals else 0.0,
                        "id_residual_mean": float(np.mean(residuals)) if residuals else 0.0,
                        "greedy_matches": len(matches),
                        "greedy_residual_median": float(np.median(greedy_residuals)) if greedy_residuals else 0.0,
                        "greedy_residual_mean": float(np.mean(greedy_residuals)) if greedy_residuals else 0.0,
                    }
                )

    def stat(values, fn, default=0.0):
        return float(fn(values)) if values else default

    summary = {
        "frames": frames,
        "frames_with_real_other": frames_with_real,
        "real_other_coverage": frames_with_real / max(frames, 1),
        "actual_scenarios": len(val_idx),
        "available_val_scenarios": available_val_scenarios,
        "requested_max_scenarios": args.max_scenarios,
        "max_frames_per_scenario": args.max_frames_per_scenario,
        "id_pairs": len(all_id_residuals),
        "id_residual_mean": stat(all_id_residuals, np.mean),
        "id_residual_median": stat(all_id_residuals, np.median),
        "id_residual_p90": stat(all_id_residuals, lambda x: np.percentile(x, 90)),
        "visible_id_pairs": len(all_visible_residuals),
        "visible_id_residual_median": stat(all_visible_residuals, np.median),
        "invisible_id_pairs": len(all_invisible_residuals),
        "invisible_id_residual_median": stat(all_invisible_residuals, np.median),
        "greedy_pairs": len(all_greedy_residuals),
        "greedy_residual_median": stat(all_greedy_residuals, np.median),
        "greedy_residual_p90": stat(all_greedy_residuals, lambda x: np.percentile(x, 90)),
        "avg_real_other_ego_invisible": real_invisible_sum / max(frames_with_real, 1),
    }
    return summary, frame_rows


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-scenarios", type=int, default=22)
    parser.add_argument("--max-frames-per-scenario", type=int, default=0)
    parser.add_argument("--greedy-match-dist", type=float, default=4.0)
    parser.add_argument("--write-frame-metrics", action="store_true")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/raid/xuyifan/trusted_coop_perception/results/realcoop_alignment_audit"),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary, frame_rows = run(args)
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if frame_rows:
        with (args.out_dir / "frame_metrics.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(frame_rows[0].keys()))
            writer.writeheader()
            writer.writerows(frame_rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
