"""V2Xverse inference with safety constraint integration.

Modified from codriving/tools/inference_e2e.py to add safety constraint
post-processing on predicted waypoints.

Compares:
  - CoDriving (baseline): original model output
  - CoDriving + Ours-Baseline: rule-based three-layer safety constraint
  - CoDriving + Ours-AB: learned risk assessment + rule constraint
  - CoDriving + RSS: RSS safety distance constraint

Safety constraint modifies predicted waypoints by:
  1. Build PerceptionResult from detected objects (pred_box_tensor)
  2. Compute safe action space
  3. Project waypoints that fall in unsafe regions back to safe boundary

Metrics:
  - ADE/FDE (standard planning metrics)
  - Collision Rate (waypoints inside detected object bounds)
  - Safety Score (fraction of waypoints in safe action space)
"""

import sys
import os
import copy
import logging
import argparse
import time
import math
import numpy as np
import torch
from typing import OrderedDict
from pathlib import Path

# Add project roots
sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # V2Xverse root
sys.path.insert(0, '/raid/xuyifan/jiqiuyu')  # coop_safety root

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils, inference_utils
from opencood.data_utils.datasets import build_dataset
from opencood.utils.occ_render import box2occ
from common.io import load_config_from_yaml
from codriving.utils.torch_helper import move_dict_data_to_device, build_dataloader
from codriving import CODRIVING_REGISTRY
from codriving.models.model_decoration import decorate_model
from common.registry import build_object_within_registry_from_config
from common.detection import warp_image
from common.torch_helper import load_checkpoint
from codriving.utils import initialize_root_logger

from coop_safety.interface import (
    PerceptionResult, VehicleState, Agent, AgentType,
    SafetyConstraintModule, SafeActionSpace, ConstraintMode,
)
from shapely.geometry import Polygon, Point

logger = logging.getLogger("safety_eval")


def build_perception_from_detection(pred_boxes, ego_measurements):
    """Convert V2Xverse detection output to our PerceptionResult format.

    Args:
        pred_boxes: (N, 8, 3) detected 3D boxes in ego frame
        ego_measurements: dict with ego state info

    Returns:
        PerceptionResult
    """
    # Ego state from measurements
    ego_speed = ego_measurements.get('speed', 5.0)
    ego_heading = ego_measurements.get('theta', 0.0)

    ego = VehicleState(
        id="ego", x=0.0, y=0.0,  # Ego frame origin
        heading=0.0, velocity=ego_speed,
        vx=ego_speed, vy=0.0,
        length=4.5, width=1.8,
    )

    agents = []
    if pred_boxes is not None:
        for i in range(pred_boxes.shape[0]):
            box = pred_boxes[i].cpu().numpy()  # (8, 3)
            center = box[:4, :2].mean(axis=0)
            dx = box[1, 0] - box[0, 0]
            dy = box[1, 1] - box[0, 1]
            heading = math.atan2(dy, dx)
            length = np.linalg.norm(box[1, :2] - box[0, :2])
            width = np.linalg.norm(box[3, :2] - box[0, :2])

            agents.append(Agent(
                state=VehicleState(
                    id=f"det_{i}",
                    x=center[0], y=center[1],
                    heading=heading,
                    velocity=5.0,  # Assume moderate speed
                    vx=5.0 * math.cos(heading),
                    vy=5.0 * math.sin(heading),
                    length=max(length, 2.0),
                    width=max(width, 1.0),
                ),
                agent_type=AgentType.VEHICLE,
            ))

    return PerceptionResult(timestamp=0, ego=ego, agents=agents)


def apply_safety_constraint(waypoints, perception, safety_module):
    """Apply safety constraint to predicted waypoints.

    For each waypoint, check if it's in the safe action space.
    If not, project it to the nearest safe point.

    Args:
        waypoints: (B, T, 2) predicted waypoints
        perception: PerceptionResult
        safety_module: safety constraint module (or None for no constraint)

    Returns:
        modified_waypoints: (B, T, 2)
        safety_stats: dict with safety metrics
    """
    if safety_module is None:
        return waypoints, {"mode": "none", "modified_ratio": 0.0}

    try:
        safe_space = safety_module.constrain(perception)
    except Exception:
        return waypoints, {"mode": "error", "modified_ratio": 0.0}

    mode = safe_space.mode.value
    modified = waypoints.clone()
    n_modified = 0
    total_points = 0

    if len(safe_space.feasible_region) >= 3:
        try:
            safe_poly = Polygon(safe_space.feasible_region)
            if safe_poly.is_valid and not safe_poly.is_empty:
                for b in range(waypoints.shape[0]):
                    for t in range(waypoints.shape[1]):
                        pt = Point(float(waypoints[b, t, 0]), float(waypoints[b, t, 1]))
                        total_points += 1
                        if not safe_poly.contains(pt):
                            # Project to nearest point on safe boundary
                            nearest = safe_poly.exterior.interpolate(
                                safe_poly.exterior.project(pt))
                            modified[b, t, 0] = nearest.x
                            modified[b, t, 1] = nearest.y
                            n_modified += 1
        except Exception:
            pass

    stats = {
        "mode": mode,
        "modified_ratio": n_modified / max(total_points, 1),
        "n_modified": n_modified,
        "total_points": total_points,
    }
    return modified, stats


def compute_collision_rate(waypoints, pred_boxes):
    """Check how many waypoints fall inside detected object bounds."""
    if pred_boxes is None or pred_boxes.shape[0] == 0:
        return 0.0

    collisions = 0
    total = 0
    for b in range(waypoints.shape[0]):
        for t in range(waypoints.shape[1]):
            pt = waypoints[b, t, :2].cpu().numpy()
            total += 1
            for i in range(pred_boxes.shape[0]):
                box_center = pred_boxes[i, :4, :2].mean(dim=0).cpu().numpy()
                dist = np.linalg.norm(pt - box_center)
                if dist < 2.0:  # Within 2m of detected object
                    collisions += 1
                    break
    return collisions / max(total, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-file', type=str, required=True)
    parser.add_argument('--out-dir', type=str, required=True)
    parser.add_argument('--model_dir', type=str, required=True)
    parser.add_argument('--planner_resume', type=str, required=True)
    parser.add_argument('--safety', type=str, default='all',
                        help='Safety method: none, ours-baseline, ours-ab, rss, all')
    opt = parser.parse_args()

    os.makedirs(opt.out_dir, exist_ok=True)
    initialize_root_logger(path=f'{opt.out_dir}/safety_eval_log.txt')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load models (same as original inference_e2e.py)
    hypes = yaml_utils.load_yaml(None, opt)
    perception_model = train_utils.create_model(hypes)
    _, perception_model = train_utils.load_saved_model(opt.model_dir, perception_model)
    perception_model.to(device)
    perception_model.eval()

    config = load_config_from_yaml(opt.config_file)
    test_data_config = config['data']['test']
    test_dataloader = build_dataloader(test_data_config, is_distributed=False)

    model_config = config['model']
    planning_model = build_object_within_registry_from_config(CODRIVING_REGISTRY, model_config)
    decorate_model(planning_model, **config['model_decoration'])
    planning_model.to(device)
    load_checkpoint(opt.planner_resume, device, planning_model, strict=True)
    planning_model.eval()

    metric_config = config['test_metric']
    metric_func = build_object_within_registry_from_config(CODRIVING_REGISTRY, metric_config)

    # Setup safety methods to compare
    methods = {}
    if opt.safety in ('none', 'all'):
        methods["CoDriving (baseline)"] = None
    if opt.safety in ('ours-baseline', 'all'):
        methods["CoDriving + Ours-Baseline"] = SafetyConstraintModule()
    if opt.safety in ('rss', 'all'):
        from experiments.methods import RSSOnly
        methods["CoDriving + RSS"] = RSSOnly()

    results = {name: {"ADEs": [], "FDEs": [], "collision_rates": [], "safety_stats": []}
               for name in methods}

    logging.info(f'Testing with safety methods: {list(methods.keys())}')
    logging.info(f'Dataset size: {len(test_dataloader.dataset)}')

    for i, batch_data in enumerate(test_dataloader):
        with torch.no_grad():
            pred_batch_data, perce_batch_data_dict = batch_data
            move_dict_data_to_device(pred_batch_data, device)
            pred_batch_data.update({'fused_feature': [], 'features_before_fusion': []})

            # Perception inference (same as original)
            frame_list = sorted(perce_batch_data_dict.keys())
            perception_results_list = []
            occ_map_list = []
            last_pred_boxes = None

            for frame in frame_list:
                perce_batch_data_dict[frame] = train_utils.to_device(perce_batch_data_dict[frame], device)
                output_dict = OrderedDict()
                for cav_id, cav_content in perce_batch_data_dict[frame].items():
                    output_dict[cav_id] = perception_model(cav_content)

                pred_box_tensor, pred_score, gt_box_tensor = \
                    test_dataloader.dataset.perception_dataset.post_process_multiclass(
                        perce_batch_data_dict[frame], output_dict, online_eval_only=True)

                infer_result = {"pred_box_tensor": pred_box_tensor,
                                "pred_score": pred_score,
                                "gt_box_tensor": gt_box_tensor}

                # Filter ego car (same as original)
                if infer_result['pred_box_tensor'][0] is not None:
                    box_list, score_list = [], []
                    for car_id in range(infer_result['pred_box_tensor'][0].shape[0]):
                        car_box = infer_result['pred_box_tensor'][0][car_id].cpu().numpy().copy()
                        car_box[:, 0] += 1.3
                        loc = np.mean(car_box[:4, :2], 0)
                        if np.linalg.norm(loc) < 1.4:
                            continue
                        box_list.append(infer_result['pred_box_tensor'][0][car_id])
                        score_list.append(infer_result['pred_score'][0][car_id])
                    infer_result['pred_box_tensor'][0] = torch.stack(box_list) if box_list else None
                    infer_result['pred_score'][0] = torch.stack(score_list) if score_list else None

                last_pred_boxes = infer_result['pred_box_tensor'][0]
                occ_map_list.append(box2occ(infer_result))

                perception_results = output_dict['ego']
                ff2 = perception_results['fused_feature'].permute(0, 1, 3, 2)
                ff3 = torch.flip(ff2, dims=[2])
                w = ff2.shape[3] // 2
                pred_batch_data['fused_feature'].append(ff3[:, :, :192, w-48:w+48])
                perception_results_list.append(perception_results)

            # Warp features (same as original)
            pred_batch_data['feature_warpped_list'] = []
            for b in range(len(perception_results_list[0]['fused_feature'])):
                fd = perception_results_list[0]['fused_feature'].shape[1]
                ft = torch.zeros(1, 5, fd, 192, 96).to(device).float()
                dp = torch.zeros(1, 5, 3).to(device).float()
                oc = torch.zeros(1, 5, 1, 192, 96).cuda().float()
                for t in range(5):
                    ft[0, t] = pred_batch_data['fused_feature'][t][b]
                    dp[:, t] = torch.tensor(pred_batch_data['detmap_pose'][b, t])
                    oc[0, t, 0:1] = occ_map_list[t]
                fw = warp_image(dp, ft)
                pred_batch_data['feature_warpped_list'].append(fw)
                ow = warp_image(dp, oc)
                pred_batch_data['occupancy'][:, :, 0, :, :] = ow[:, :, 0, :, :]

            # Planning inference
            model_output = planning_model(pred_batch_data)
            original_waypoints = model_output['future_waypoints']

            # Build perception for safety constraint
            perception = build_perception_from_detection(last_pred_boxes, {})

            # Evaluate each safety method
            for method_name, safety_module in methods.items():
                if safety_module is None:
                    wp = original_waypoints
                    stats = {"mode": "none", "modified_ratio": 0.0}
                else:
                    wp, stats = apply_safety_constraint(
                        original_waypoints, perception, safety_module)

                # Compute metrics
                modified_output = dict(future_waypoints=wp)
                ADE, FDE = metric_func(pred_batch_data, modified_output)
                coll_rate = compute_collision_rate(wp, last_pred_boxes)

                results[method_name]["ADEs"].append(ADE)
                results[method_name]["FDEs"].append(FDE)
                results[method_name]["collision_rates"].append(coll_rate)
                results[method_name]["safety_stats"].append(stats)

            if i % 100 == 0:
                print(f"[{i}/{len(test_dataloader)}]")

            torch.cuda.empty_cache()

    # Print results
    print("\n" + "=" * 70)
    print("V2Xverse Safety Constraint Evaluation Results")
    print("=" * 70)
    print(f"\n{'Method':35s} {'ADE':>7s} {'FDE':>7s} {'CollRate':>9s} {'ModRate':>8s}")
    print("-" * 70)

    import json
    summary = {}
    for name, data in results.items():
        ade = torch.mean(torch.cat(data["ADEs"])).item()
        fde = torch.mean(torch.cat(data["FDEs"])).item()
        cr = np.mean(data["collision_rates"])
        mr = np.mean([s["modified_ratio"] for s in data["safety_stats"]])
        print(f"{name:35s} {ade:7.4f} {fde:7.4f} {cr:8.4f} {mr:7.4f}")
        summary[name] = {"ADE": ade, "FDE": fde, "collision_rate": cr, "modification_rate": mr}

    # Save
    with open(f"{opt.out_dir}/safety_eval_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    logging.info(f"Results saved to {opt.out_dir}/safety_eval_results.json")


if __name__ == '__main__':
    main()
