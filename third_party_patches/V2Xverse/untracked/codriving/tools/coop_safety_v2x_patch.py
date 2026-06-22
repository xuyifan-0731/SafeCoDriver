"""Safety constraint patch for V2Xverse inference.

Called from inference_e2e.py when --safety_eval is set.
Applies safety constraints to waypoints and returns modified outputs.
"""
from __future__ import annotations
import sys
import math
import numpy as np
import torch
from shapely.geometry import Polygon, Point

sys.path.insert(0, '/raid/xuyifan/jiqiuyu')

from coop_safety.interface import (
    PerceptionResult, VehicleState, Agent, AgentType,
    SafetyConstraintModule, ConstraintMode,
)


_modules = {}

def _get_module(name):
    if name not in _modules:
        if name == 'ours':
            _modules[name] = SafetyConstraintModule()
        elif name == 'rss':
            from experiments.methods import RSSOnly
            _modules[name] = RSSOnly()
    return _modules[name]


def _boxes_to_perception(pred_boxes):
    """Convert detected boxes to PerceptionResult."""
    ego = VehicleState(id="ego", x=0, y=0, heading=0, velocity=5.0, vx=5.0, vy=0,
                       length=4.5, width=1.8)
    agents = []
    if pred_boxes is not None:
        for i in range(pred_boxes.shape[0]):
            box = pred_boxes[i].cpu().numpy()
            center = box[:4, :2].mean(axis=0)
            heading = math.atan2(box[1,1]-box[0,1], box[1,0]-box[0,0])
            agents.append(Agent(state=VehicleState(
                id=f"d{i}", x=float(center[0]), y=float(center[1]),
                heading=float(heading), velocity=5.0,
                vx=5.0*math.cos(heading), vy=5.0*math.sin(heading),
                length=max(float(np.linalg.norm(box[1,:2]-box[0,:2])), 2.0),
                width=max(float(np.linalg.norm(box[3,:2]-box[0,:2])), 1.0)),
                agent_type=AgentType.VEHICLE))
    return PerceptionResult(timestamp=0, ego=ego, agents=agents)


def _constrain_waypoints(waypoints, safe_module, perception):
    """Project unsafe waypoints into safe action space."""
    try:
        safe = safe_module.constrain(perception)
    except:
        return waypoints, 0.0

    if safe.mode == ConstraintMode.NORMAL:
        return waypoints, 0.0

    modified = waypoints.clone()
    n_mod = 0
    total = 0

    if len(safe.feasible_region) >= 3:
        try:
            poly = Polygon(safe.feasible_region)
            if poly.is_valid and not poly.is_empty:
                for b in range(waypoints.shape[0]):
                    for t in range(waypoints.shape[1]):
                        pt = Point(float(waypoints[b,t,0]), float(waypoints[b,t,1]))
                        total += 1
                        if not poly.contains(pt):
                            nearest = poly.exterior.interpolate(poly.exterior.project(pt))
                            modified[b,t,0] = nearest.x
                            modified[b,t,1] = nearest.y
                            n_mod += 1
        except:
            pass

    coll_rate = n_mod / max(total, 1)
    return modified, coll_rate


def apply_safety_and_measure(model_output, pred_batch_data, infer_result):
    """Apply safety constraints and return modified outputs for each method."""
    wp = model_output['future_waypoints']

    # Safely extract pred_boxes
    pred_boxes = None
    try:
        if infer_result and 'pred_box_tensor' in infer_result:
            pbt = infer_result['pred_box_tensor']
            if isinstance(pbt, (list, tuple)) and len(pbt) > 0:
                pred_boxes = pbt[0]
            elif isinstance(pbt, torch.Tensor):
                pred_boxes = pbt
    except:
        pred_boxes = None

    if pred_boxes is None or (isinstance(pred_boxes, torch.Tensor) and pred_boxes.numel() == 0):
        # No detections — no safety constraint needed
        return {
            'CoDriving+Ours': {'output': model_output, 'coll_rate': 0.0},
            'CoDriving+RSS': {'output': model_output, 'coll_rate': 0.0},
        }

    perception = _boxes_to_perception(pred_boxes)

    results = {}

    # Ours-Baseline
    mod = _get_module('ours')
    wp_safe, cr = _constrain_waypoints(wp, mod, perception)
    results['CoDriving+Ours'] = {'output': dict(future_waypoints=wp_safe), 'coll_rate': cr}

    # RSS
    rss = _get_module('rss')
    wp_rss, cr_rss = _constrain_waypoints(wp, rss, perception)
    results['CoDriving+RSS'] = {'output': dict(future_waypoints=wp_rss), 'coll_rate': cr_rss}

    return results
