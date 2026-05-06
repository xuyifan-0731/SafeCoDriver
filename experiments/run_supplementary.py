"""Supplementary experiments for SafeCoDriver paper.

Experiments:
1. Cooperative perception ablation (ego-only vs ego+V2X)
2. Detection threshold sensitivity (V1 threshold sweep)
3. Computation efficiency (ms/frame)
4. Scenario type breakdown
5. Noise robustness (position/velocity noise)
6. Visibility-aware margin ablation
7. Multi-agent vs single-agent ablation
"""
from __future__ import annotations

import sys
import os
import time
import math
import numpy as np
from pathlib import Path
from copy import deepcopy

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.deepaccident_loader import DeepAccidentLoader, DeepAccidentFrame
from experiments.run_deepaccident_unified import (
    simulate_codriving_waypoints, check_waypoint_collision,
    evaluate_method_on_scenario, MethodResult,
)
from coop_safety.interface import (
    SafetyConstraintModule, ConstraintMode, PerceptionResult, VehicleState, Agent,
)


def exp1_cooperative_ablation(loader):
    """Exp 1: Compare ego-only vs ego+V2X perception.

    ego-only: only visible agents (is_visible=True)
    ego+V2X: all agents (visible + invisible)
    """
    print("\n" + "="*70)
    print("  EXP 1: Cooperative Perception Ablation")
    print("="*70)

    import torch
    from coop_safety.learned.collision_network import CollisionPredictionNetwork
    from coop_safety.learned.hybrid_safety import HybridSafetyConstraint

    v1_path = "/raid/xuyifan/jiqiuyu/models/collision_net_best.pt"
    v1 = CollisionPredictionNetwork()
    v1.load_state_dict(torch.load(v1_path, map_location='cpu', weights_only=False)['model'])
    v1.eval()

    configs = {
        "ego+V2X (all agents)": {"filter_invisible": False},
        "ego-only (visible only)": {"filter_invisible": True},
    }

    for config_name, cfg in configs.items():
        hybrid = HybridSafetyConstraint(
            detector_model=v1,
            base_margin_visible=2.5,
            base_margin_invisible=4.0,
            detection_threshold=0.3,
        )

        result = MethodResult(name=config_name)
        for si, s in enumerate(loader.scenarios):
            first_warning = -1
            false_alarm = 0
            wp_coll_total = 0
            modifications = 0

            for fi in range(len(s['frames'])):
                frame = loader.load_frame(si, fi)

                # Filter invisible agents if ego-only
                if cfg["filter_invisible"]:
                    frame.perception = PerceptionResult(
                        timestamp=frame.perception.timestamp,
                        ego=frame.perception.ego,
                        agents=[a for a in frame.perception.agents if a.is_visible],
                        lanes=frame.perception.lanes,
                    )

                base_wp = simulate_codriving_waypoints(frame)
                modified_wp, stats = hybrid.constrain_waypoints(base_wp, frame.perception)
                was_modified = stats.get('n_collisions_detected', 0) > 0

                if was_modified:
                    if first_warning < 0:
                        first_warning = fi
                    if not s['is_accident']:
                        false_alarm += 1
                    modifications += 1

                wp_coll_total += check_waypoint_collision(modified_wp, frame)

            result.total_frames += len(s['frames'])
            result.modified_frames += modifications
            result.waypoint_collisions += wp_coll_total
            result.total_waypoint_checks += len(s['frames']) * 10

            if s['is_accident']:
                result.n_accident_scenarios += 1
                if first_warning >= 0:
                    result.n_detected += 1
                    result.early_warning_frames.append(len(s['frames']) - first_warning)
            else:
                result.n_normal_scenarios += 1
                if false_alarm > 0:
                    result.n_false_alarm_scenarios += 1

        det_rate = result.n_detected / max(result.n_accident_scenarios, 1)
        avg_early = np.mean(result.early_warning_frames) if result.early_warning_frames else 0
        fa_rate = result.n_false_alarm_scenarios / max(result.n_normal_scenarios, 1)
        wp_coll = result.waypoint_collisions / max(result.total_waypoint_checks, 1)
        mod_rate = result.modified_frames / max(result.total_frames, 1)

        print(f"  {config_name:30s}: Det={det_rate:.1%} Early={avg_early:.1f} FA={fa_rate:.1%} WPColl={wp_coll:.1%} Mod={mod_rate:.1%}")


def exp2_threshold_sensitivity(loader):
    """Exp 2: V1 detection threshold sensitivity analysis."""
    print("\n" + "="*70)
    print("  EXP 2: Detection Threshold Sensitivity")
    print("="*70)

    import torch
    from coop_safety.learned.collision_network import CollisionPredictionNetwork
    from coop_safety.learned.hybrid_safety import HybridSafetyConstraint

    v1_path = "/raid/xuyifan/jiqiuyu/models/collision_net_best.pt"
    v1 = CollisionPredictionNetwork()
    v1.load_state_dict(torch.load(v1_path, map_location='cpu', weights_only=False)['model'])
    v1.eval()

    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    print(f"  {'Threshold':>10s} {'DetRate':>8s} {'EarlyWarn':>10s} {'FalseAlm':>9s} {'WPColl%':>8s}")
    print("  " + "-"*50)

    for thr in thresholds:
        hybrid = HybridSafetyConstraint(
            detector_model=v1,
            base_margin_visible=2.5,
            base_margin_invisible=4.0,
            detection_threshold=thr,
        )

        result = MethodResult(name=f"thr={thr}")
        for si, s in enumerate(loader.scenarios):
            first_warning = -1
            false_alarm = 0
            wp_coll_total = 0

            for fi in range(len(s['frames'])):
                frame = loader.load_frame(si, fi)
                base_wp = simulate_codriving_waypoints(frame)
                modified_wp, stats = hybrid.constrain_waypoints(base_wp, frame.perception)
                was_modified = stats.get('n_collisions_detected', 0) > 0

                if was_modified:
                    if first_warning < 0:
                        first_warning = fi
                    if not s['is_accident']:
                        false_alarm += 1

                wp_coll_total += check_waypoint_collision(modified_wp, frame)

            result.total_waypoint_checks += len(s['frames']) * 10
            result.waypoint_collisions += wp_coll_total

            if s['is_accident']:
                result.n_accident_scenarios += 1
                if first_warning >= 0:
                    result.n_detected += 1
                    result.early_warning_frames.append(len(s['frames']) - first_warning)
            else:
                result.n_normal_scenarios += 1
                if false_alarm > 0:
                    result.n_false_alarm_scenarios += 1

        det_rate = result.n_detected / max(result.n_accident_scenarios, 1)
        avg_early = np.mean(result.early_warning_frames) if result.early_warning_frames else 0
        fa_rate = result.n_false_alarm_scenarios / max(result.n_normal_scenarios, 1)
        wp_coll = result.waypoint_collisions / max(result.total_waypoint_checks, 1)

        print(f"  {thr:10.1f} {det_rate:7.1%} {avg_early:9.1f} {fa_rate:8.1%} {wp_coll:7.1%}")


def exp3_efficiency(loader):
    """Exp 3: Computation efficiency (ms/frame)."""
    print("\n" + "="*70)
    print("  EXP 3: Computation Efficiency")
    print("="*70)

    import torch
    from coop_safety.learned.collision_network import CollisionPredictionNetwork
    from coop_safety.learned.collision_network_v2 import CollisionPredictionNetV2
    from coop_safety.learned.hybrid_safety import HybridSafetyConstraint
    from experiments.methods import RSSOnly
    from experiments.methods_modern import RiskPotentialField
    from experiments.methods_new_baselines import UniE2EV2XSafety, MAPSafety, RiskMMSafety

    v1 = CollisionPredictionNetwork()
    v1.load_state_dict(torch.load("/raid/xuyifan/jiqiuyu/models/collision_net_best.pt",
                                   map_location='cpu', weights_only=False)['model'])
    v1.eval()

    methods = {
        "RSS": RSSOnly(),
        "APF": RiskPotentialField(),
        "UniE2EV2X": UniE2EV2XSafety(),
        "MAP": MAPSafety(),
        "RiskMM": RiskMMSafety(),
        "Ours-Rule": SafetyConstraintModule(),
        "Ours-Hybrid": HybridSafetyConstraint(detector_model=v1,
                                               base_margin_visible=2.5,
                                               base_margin_invisible=4.0),
    }

    # Use first 200 frames for timing
    n_frames = 200
    frames = []
    count = 0
    for si in range(len(loader.scenarios)):
        for fi in range(len(loader.scenarios[si]['frames'])):
            frames.append((si, fi))
            count += 1
            if count >= n_frames:
                break
        if count >= n_frames:
            break

    print(f"  Timing over {n_frames} frames...")
    print(f"  {'Method':20s} {'Total(ms)':>10s} {'Per-frame(ms)':>13s}")
    print("  " + "-"*45)

    for name, method in methods.items():
        t0 = time.time()
        for si, fi in frames:
            frame = loader.load_frame(si, fi)
            base_wp = simulate_codriving_waypoints(frame)
            if hasattr(method, 'constrain_waypoints'):
                method.constrain_waypoints(base_wp, frame.perception)
            else:
                method.constrain(frame.perception)
        elapsed = (time.time() - t0) * 1000
        per_frame = elapsed / n_frames
        print(f"  {name:20s} {elapsed:9.0f} {per_frame:12.1f}")


def exp4_scenario_breakdown(loader):
    """Exp 4: Performance breakdown by scenario type."""
    print("\n" + "="*70)
    print("  EXP 4: Scenario Type Breakdown")
    print("="*70)

    import torch
    from coop_safety.learned.collision_network import CollisionPredictionNetwork
    from coop_safety.learned.hybrid_safety import HybridSafetyConstraint

    v1 = CollisionPredictionNetwork()
    v1.load_state_dict(torch.load("/raid/xuyifan/jiqiuyu/models/collision_net_best.pt",
                                   map_location='cpu', weights_only=False)['model'])
    v1.eval()
    hybrid = HybridSafetyConstraint(detector_model=v1, base_margin_visible=2.5,
                                     base_margin_invisible=4.0, detection_threshold=0.3)

    # Group by scenario type
    type_results = {}
    for si, s in enumerate(loader.scenarios):
        # Extract type from name: type1_subtype1_accident/...
        parts = s['name'].split('/')
        scenario_type = parts[0].replace('_accident', '').replace('_normal', '')

        if scenario_type not in type_results:
            type_results[scenario_type] = {'det': 0, 'n_acc': 0, 'n_norm': 0, 'fa': 0, 'wp_coll': 0, 'wp_total': 0}

        first_warning = -1
        false_alarm = 0
        wp_coll = 0

        for fi in range(len(s['frames'])):
            frame = loader.load_frame(si, fi)
            base_wp = simulate_codriving_waypoints(frame)
            modified_wp, stats = hybrid.constrain_waypoints(base_wp, frame.perception)
            was_modified = stats.get('n_collisions_detected', 0) > 0

            if was_modified and first_warning < 0:
                first_warning = fi
            if was_modified and not s['is_accident']:
                false_alarm += 1

            wp_coll += check_waypoint_collision(modified_wp, frame)

        type_results[scenario_type]['wp_coll'] += wp_coll
        type_results[scenario_type]['wp_total'] += len(s['frames']) * 10

        if s['is_accident']:
            type_results[scenario_type]['n_acc'] += 1
            if first_warning >= 0:
                type_results[scenario_type]['det'] += 1
        else:
            type_results[scenario_type]['n_norm'] += 1
            if false_alarm > 0:
                type_results[scenario_type]['fa'] += 1

    print(f"  {'Type':25s} {'Scenarios':>10s} {'DetRate':>8s} {'FalseAlm':>9s} {'WPColl%':>8s}")
    print("  " + "-"*65)
    for t, r in sorted(type_results.items()):
        n_total = r['n_acc'] + r['n_norm']
        det_rate = r['det'] / max(r['n_acc'], 1)
        fa_rate = r['fa'] / max(r['n_norm'], 1)
        wp_rate = r['wp_coll'] / max(r['wp_total'], 1)
        print(f"  {t:25s} {n_total:10d} {det_rate:7.1%} {fa_rate:8.1%} {wp_rate:7.1%}")


def exp5_noise_robustness(loader):
    """Exp 5: Noise robustness (add noise to GT perception)."""
    print("\n" + "="*70)
    print("  EXP 5: Noise Robustness")
    print("="*70)

    import torch
    from coop_safety.learned.collision_network import CollisionPredictionNetwork
    from coop_safety.learned.hybrid_safety import HybridSafetyConstraint

    v1 = CollisionPredictionNetwork()
    v1.load_state_dict(torch.load("/raid/xuyifan/jiqiuyu/models/collision_net_best.pt",
                                   map_location='cpu', weights_only=False)['model'])
    v1.eval()

    noise_levels = [
        ("No noise", 0.0, 0.0),
        ("Low (0.3m, 0.5m/s)", 0.3, 0.5),
        ("Medium (0.5m, 1.0m/s)", 0.5, 1.0),
        ("High (1.0m, 2.0m/s)", 1.0, 2.0),
    ]

    print(f"  {'Noise Level':25s} {'DetRate':>8s} {'FalseAlm':>9s} {'WPColl%':>8s}")
    print("  " + "-"*55)

    for noise_name, pos_noise, vel_noise in noise_levels:
        np.random.seed(42)
        hybrid = HybridSafetyConstraint(detector_model=v1, base_margin_visible=2.5,
                                         base_margin_invisible=4.0, detection_threshold=0.3)

        result = MethodResult(name=noise_name)
        for si, s in enumerate(loader.scenarios):
            first_warning = -1
            false_alarm = 0
            wp_coll = 0

            for fi in range(len(s['frames'])):
                frame = loader.load_frame(si, fi)

                # Add noise to agent positions and velocities
                if pos_noise > 0:
                    noisy_agents = []
                    for a in frame.perception.agents:
                        noisy_state = VehicleState(
                            id=a.state.id,
                            x=a.state.x + np.random.normal(0, pos_noise),
                            y=a.state.y + np.random.normal(0, pos_noise),
                            heading=a.state.heading,
                            velocity=a.state.velocity,
                            vx=a.state.vx + np.random.normal(0, vel_noise),
                            vy=a.state.vy + np.random.normal(0, vel_noise),
                            length=a.state.length,
                            width=a.state.width,
                            vehicle_type=a.state.vehicle_type,
                        )
                        noisy_agents.append(Agent(state=noisy_state,
                                                  agent_type=a.agent_type,
                                                  is_visible=a.is_visible,
                                                  confidence=a.confidence))
                    frame.perception = PerceptionResult(
                        timestamp=frame.perception.timestamp,
                        ego=frame.perception.ego,
                        agents=noisy_agents,
                        lanes=frame.perception.lanes,
                    )

                base_wp = simulate_codriving_waypoints(frame)
                modified_wp, stats = hybrid.constrain_waypoints(base_wp, frame.perception)
                was_modified = stats.get('n_collisions_detected', 0) > 0

                if was_modified and first_warning < 0:
                    first_warning = fi
                if was_modified and not s['is_accident']:
                    false_alarm += 1

                # Check collision on ORIGINAL frame (GT), not noisy
                orig_frame = loader.load_frame(si, fi)
                wp_coll += check_waypoint_collision(modified_wp, orig_frame)

            result.total_waypoint_checks += len(s['frames']) * 10
            result.waypoint_collisions += wp_coll

            if s['is_accident']:
                result.n_accident_scenarios += 1
                if first_warning >= 0:
                    result.n_detected += 1
            else:
                result.n_normal_scenarios += 1
                if false_alarm > 0:
                    result.n_false_alarm_scenarios += 1

        det_rate = result.n_detected / max(result.n_accident_scenarios, 1)
        fa_rate = result.n_false_alarm_scenarios / max(result.n_normal_scenarios, 1)
        wp_coll = result.waypoint_collisions / max(result.total_waypoint_checks, 1)
        print(f"  {noise_name:25s} {det_rate:7.1%} {fa_rate:8.1%} {wp_coll:7.1%}")


def exp6_margin_ablation(loader):
    """Exp 6: Visibility-aware margin ablation."""
    print("\n" + "="*70)
    print("  EXP 6: Visibility-Aware Margin Ablation")
    print("="*70)

    import torch
    from coop_safety.learned.collision_network import CollisionPredictionNetwork
    from coop_safety.learned.hybrid_safety import HybridSafetyConstraint

    v1 = CollisionPredictionNetwork()
    v1.load_state_dict(torch.load("/raid/xuyifan/jiqiuyu/models/collision_net_best.pt",
                                   map_location='cpu', weights_only=False)['model'])
    v1.eval()

    configs = [
        ("Uniform 2.0m", 2.0, 2.0),
        ("Uniform 2.5m", 2.5, 2.5),
        ("Uniform 3.0m", 3.0, 3.0),
        ("Uniform 4.0m", 4.0, 4.0),
        ("Ours (2.5/4.0m)", 2.5, 4.0),
    ]

    print(f"  {'Config':25s} {'DetRate':>8s} {'FalseAlm':>9s} {'WPColl%':>8s} {'ModRate':>8s}")
    print("  " + "-"*60)

    for name, vis_m, invis_m in configs:
        hybrid = HybridSafetyConstraint(detector_model=v1,
                                         base_margin_visible=vis_m,
                                         base_margin_invisible=invis_m,
                                         detection_threshold=0.3)

        result = MethodResult(name=name)
        for si, s in enumerate(loader.scenarios):
            first_warning = -1
            false_alarm = 0
            wp_coll = 0
            modifications = 0

            for fi in range(len(s['frames'])):
                frame = loader.load_frame(si, fi)
                base_wp = simulate_codriving_waypoints(frame)
                modified_wp, stats = hybrid.constrain_waypoints(base_wp, frame.perception)
                was_modified = stats.get('n_collisions_detected', 0) > 0
                has_geo = stats.get('n_geometric_threats', 0) > 0

                if was_modified and first_warning < 0:
                    first_warning = fi
                if was_modified and not s['is_accident']:
                    false_alarm += 1
                if has_geo:
                    modifications += 1

                wp_coll += check_waypoint_collision(modified_wp, frame)

            result.total_frames += len(s['frames'])
            result.modified_frames += modifications
            result.total_waypoint_checks += len(s['frames']) * 10
            result.waypoint_collisions += wp_coll

            if s['is_accident']:
                result.n_accident_scenarios += 1
                if first_warning >= 0:
                    result.n_detected += 1
            else:
                result.n_normal_scenarios += 1
                if false_alarm > 0:
                    result.n_false_alarm_scenarios += 1

        det_rate = result.n_detected / max(result.n_accident_scenarios, 1)
        fa_rate = result.n_false_alarm_scenarios / max(result.n_normal_scenarios, 1)
        wp_coll_rate = result.waypoint_collisions / max(result.total_waypoint_checks, 1)
        mod_rate = result.modified_frames / max(result.total_frames, 1)
        print(f"  {name:25s} {det_rate:7.1%} {fa_rate:8.1%} {wp_coll_rate:7.1%} {mod_rate:7.1%}")


def exp7_multiagent_ablation(loader):
    """Exp 7: Multi-agent repulsion vs single-agent push."""
    print("\n" + "="*70)
    print("  EXP 7: Multi-Agent vs Single-Agent Ablation")
    print("="*70)

    import torch
    from coop_safety.learned.collision_network import CollisionPredictionNetwork
    from coop_safety.learned.hybrid_safety import HybridSafetyConstraint
    from experiments.methods_new_baselines import UniE2EV2XSafety

    v1 = CollisionPredictionNetwork()
    v1.load_state_dict(torch.load("/raid/xuyifan/jiqiuyu/models/collision_net_best.pt",
                                   map_location='cpu', weights_only=False)['model'])
    v1.eval()

    # Compare: our multi-agent vs UniE2EV2X-style single
    methods = {
        "UniE2EV2X (single, 3.0m)": UniE2EV2XSafety(safety_threshold=3.0),
        "Ours-Hybrid (multi-agent)": HybridSafetyConstraint(
            detector_model=v1, base_margin_visible=2.5, base_margin_invisible=4.0,
            detection_threshold=0.3),
    }

    print(f"  {'Method':35s} {'WPColl%':>8s}")
    print("  " + "-"*45)

    for name, method in methods.items():
        wp_coll_total = 0
        wp_total = 0
        for si, s in enumerate(loader.scenarios):
            for fi in range(len(s['frames'])):
                frame = loader.load_frame(si, fi)
                base_wp = simulate_codriving_waypoints(frame)
                if hasattr(method, 'constrain_waypoints'):
                    modified_wp, _ = method.constrain_waypoints(base_wp, frame.perception)
                else:
                    modified_wp = base_wp
                wp_coll_total += check_waypoint_collision(modified_wp, frame)
                wp_total += 10

        wp_rate = wp_coll_total / max(wp_total, 1)
        print(f"  {name:35s} {wp_rate:7.1%}")


def main():
    print("="*70)
    print("  SafeCoDriver: Supplementary Experiments")
    print("="*70)

    loader = DeepAccidentLoader(split='all')
    t0 = time.time()

    exp3_efficiency(loader)
    exp4_scenario_breakdown(loader)
    exp2_threshold_sensitivity(loader)
    exp6_margin_ablation(loader)
    exp1_cooperative_ablation(loader)
    exp5_noise_robustness(loader)
    exp7_multiagent_ablation(loader)

    print(f"\nTotal time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
