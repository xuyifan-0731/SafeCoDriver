#!/usr/bin/env python3
"""Render qualitative TrustCalib case figures from existing DeepAccident pilots."""
from __future__ import annotations

import copy
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.transforms import Affine2D
import numpy as np
import torch


WORK_ROOT = Path(__file__).resolve().parents[1]
SAFECO_ROOT = Path("/raid/xuyifan/jiqiuyu")
sys.path.insert(0, str(SAFECO_ROOT))
sys.path.insert(0, str(WORK_ROOT))

from experiments.deepaccident_loader import DeepAccidentLoader
from experiments.run_deepaccident_unified import check_waypoint_collision, simulate_codriving_waypoints
from prototype.real_coop import filter_sender_visible, load_aligned_source_message, merge_ego_visible_with_real_other
from prototype.run_deepaccident_mixed_pilot import eval_method, load_hybrid, object_impact_guard_perception
from prototype.run_deepaccident_multipeer_pilot import (
    MissingCandidateTemporalTracker,
    MultiPeerEvidenceSupport,
    UnsupportedTemporalTracker,
    _geometry_match,
    agent_key,
    apply_box_margin_guard,
    apply_high_trust_probation,
    apply_missing_object_recovery,
    apply_peer_consensus_smoothing,
    build_peer_messages,
    calibrate_coop_message,
    count_fake,
    count_real_removed,
    parse_delay_grid,
)
from prototype.run_deepaccident_realmultisource_pilot import make_missing_args
from prototype.trust_calib import calibrate_shifted_coop, filter_visible, merge_ego_visible_with_coop_invisible


FIG_DIR = WORK_ROOT / "paper_ready" / "figures"
RESULT_DIR = WORK_ROOT / "results"
if not RESULT_DIR.exists():
    RESULT_DIR = Path("/raid/xuyifan/trusted_coop_perception/results")


SYN_CASES = [
    {
        "name": "case_fake_front_filter",
        "title": "Fake-front object quarantine",
        "mode": "fake_front",
        "scenario": 78,
        "frame": 15,
        "metadata": RESULT_DIR / "baseline_final_v1_20x20_diag" / "metadata.json",
        "caption": "Synthetic fake_front. TrustCalib keeps the injected front object; MultiPeerObjectGuard+BoxGuard removes it.",
    },
    {
        "name": "case_noise_fake_recovery",
        "title": "Noise + fake-front correction",
        "mode": "noise+fake_front",
        "scenario": 48,
        "frame": 12,
        "metadata": RESULT_DIR / "baseline_final_v1_20x20_diag" / "metadata.json",
        "caption": "Synthetic noise+fake_front. Peer consensus suppresses unsupported high-impact artifacts and keeps supported objects.",
    },
    {
        "name": "case_drop_missing_recovery",
        "title": "Dropped blind-spot object recovery",
        "mode": "drop",
        "scenario": 7,
        "frame": 0,
        "metadata": RESULT_DIR / "baseline_final_v1_20x20_diag" / "metadata.json",
        "caption": "Synthetic drop. MissingRecovery admits peer-supported missing objects that recover waypoint safety.",
    },
    {
        "name": "case_collusion_trust_weighting",
        "title": "Trust-weighted collusion guard",
        "mode": "fake_front",
        "scenario": 78,
        "frame": 15,
        "metadata": RESULT_DIR / "collusion_final_trustweighted_v1_20x20_diag" / "metadata.json",
        "caption": "Collusion stress. A low-trust colluding support peer is insufficient to validate the fake object.",
    },
]

REAL_CASES = [
    {
        "name": "case_real_pathrisk_admission",
        "title": "Real multi-source path-risk admission",
        "scenario": 48,
        "frame": 18,
        "metadata": RESULT_DIR / "realmultisource_20x20_pathrisk_min2_thr0" / "metadata.json",
        "caption": "Real aligned multi-source labels. A one-source missing object is admitted only when its oriented-box path margin is collision-critical.",
    }
]


def load_args(path: Path) -> SimpleNamespace:
    meta = json.loads(path.read_text(encoding="utf-8"))
    args = SimpleNamespace(**meta)
    if not hasattr(args, "time_delay_grid_values"):
        args.time_delay_grid_values = parse_delay_grid(getattr(args, "time_delay_grid", "0.0"))
    if isinstance(getattr(args, "support_modes", None), str):
        args.support_modes = [m.strip() for m in args.support_modes.split(",") if m.strip()]
    if isinstance(getattr(args, "support_trusts", None), str):
        args.support_trusts = [float(v.strip()) for v in args.support_trusts.split(",") if v.strip()]
    return args


def get_val_indices(max_scenarios: int) -> list[int]:
    ckpt = torch.load(SAFECO_ROOT / "models" / "collision_net_best.pt", map_location="cpu", weights_only=False)
    val_idx = list(ckpt.get("val_scenario_idx", []))
    if max_scenarios > 0:
        val_idx = val_idx[:max_scenarios]
    return val_idx


def frame_indices_for(args: SimpleNamespace, scenario: dict) -> list[int]:
    total_frames = len(scenario["frames"])
    if getattr(args, "frame_window", "start") == "full":
        return list(range(total_frames))
    if getattr(args, "frame_window", "start") == "accident":
        collision_frame = scenario.get("collision_frame", -1)
        if collision_frame > 0:
            start = max(0, collision_frame - args.accident_window_before)
            end = min(total_frames, collision_frame + args.accident_window_after + 1)
            return list(range(start, end))
    n_frames = total_frames
    if args.max_frames_per_scenario > 0:
        n_frames = min(n_frames, args.max_frames_per_scenario)
    return list(range(n_frames))


def replay_synthetic_case(case: dict, loader: DeepAccidentLoader, hybrid):
    args = load_args(case["metadata"])
    rng = np.random.default_rng(args.seed)
    support_modes = list(args.support_modes)
    val_idx = get_val_indices(args.max_scenarios)
    target = (case["scenario"], case["frame"])

    for si in val_idx:
        temporal_tracker = UnsupportedTemporalTracker(max_dist=args.temporal_match_dist)
        scenario = loader.scenarios[si]
        for fi in frame_indices_for(args, scenario):
            frame = loader.load_frame(si, fi)
            waypoints = simulate_codriving_waypoints(frame)
            primary_msg, support_msgs = build_peer_messages(frame, case["mode"], support_modes, rng, args)
            if (si, fi) != target:
                continue

            ego_only = filter_visible(frame.perception)
            clean_coop = merge_ego_visible_with_coop_invisible(frame.perception, frame.perception)
            primary_raw = merge_ego_visible_with_coop_invisible(frame.perception, primary_msg)
            primary_trust, primary_report = calibrate_coop_message(frame.perception, primary_msg, args)
            support_trust = []
            support_reports = []
            for support_msg in support_msgs:
                perception, support_report = calibrate_coop_message(frame.perception, support_msg, args)
                support_trust.append(perception)
                support_reports.append(support_report)

            support_sender_ids = [f"support_{i}_{support_modes[i]}" for i in range(len(support_modes))]
            evidence = MultiPeerEvidenceSupport(
                support_trust,
                peer_reports=support_reports,
                sender_ids=support_sender_ids,
                sender_trusts=args.support_trusts,
                min_support=args.min_peer_support,
                min_trust_support=args.min_trust_support,
                max_dist=args.peer_match_dist,
            )

            guarded, guarded_report, obj_removed = object_impact_guard_perception(
                ego_only,
                primary_trust,
                primary_report,
                hybrid,
                waypoints,
                evidence_tracker=evidence,
            )
            guarded_report = copy.deepcopy(guarded_report)
            temporal_labels = temporal_tracker.update_and_label(primary_trust, evidence)
            guarded, n_restored, restored_reason = apply_high_trust_probation(
                primary_trust,
                guarded,
                temporal_labels,
                args,
                ego_only,
                hybrid,
                waypoints,
            )
            if n_restored:
                obj_removed = max(0, obj_removed - n_restored)
                guarded_report.n_output_agents = len(guarded.agents)
            guarded, n_box_removed = apply_box_margin_guard(guarded, evidence, args, waypoints)
            if n_box_removed:
                obj_removed += n_box_removed
                guarded_report.action = "object_quarantine"
                guarded_report.n_output_agents = len(guarded.agents)
            guarded, n_smoothed, _ = apply_peer_consensus_smoothing(
                guarded,
                support_trust,
                support_sender_ids,
                args.support_trusts,
                args,
            )
            if n_smoothed:
                guarded_report.n_corrected_agents += n_smoothed
                guarded_report.n_output_agents = len(guarded.agents)

            (
                guarded,
                n_missing_recovered,
                n_missing_candidates,
                n_missing_gt,
                n_missing_tp,
                n_missing_fp,
                missing_records,
            ) = apply_missing_object_recovery(
                case["mode"],
                si,
                fi,
                guarded,
                primary_trust,
                primary_report,
                clean_coop,
                support_trust,
                support_sender_ids,
                args.support_trusts,
                args,
                ego_only,
                hybrid,
                waypoints,
            )
            if n_missing_recovered:
                guarded_report.n_output_agents = len(guarded.agents)

            return {
                "frame": frame,
                "waypoints": waypoints,
                "ego_only": ego_only,
                "clean_coop": clean_coop,
                "primary_raw": primary_raw,
                "primary_trust": primary_trust,
                "primary_report": primary_report,
                "guarded": guarded,
                "guarded_report": guarded_report,
                "evidence": evidence,
                "missing_records": missing_records,
                "stats": {
                    "fake_seen": count_fake(primary_trust),
                    "fake_removed": max(0, count_fake(primary_trust) - count_fake(guarded)),
                    "real_removed": count_real_removed(primary_trust, guarded),
                    "obj_removed": obj_removed,
                    "missing_recovered": n_missing_recovered,
                    "missing_candidates": n_missing_candidates,
                    "missing_gt": n_missing_gt,
                    "missing_tp": n_missing_tp,
                    "missing_fp": n_missing_fp,
                },
                "args": args,
            }

    raise RuntimeError(f"Target synthetic case not found: {target}")


def replay_real_case(case: dict, loader: DeepAccidentLoader, hybrid):
    args = load_args(case["metadata"])
    args.source_roles = [role.strip() for role in args.source_roles if role.strip()]
    if args.primary_role not in args.source_roles:
        args.source_roles.insert(0, args.primary_role)
    support_roles = [role for role in args.source_roles if role != args.primary_role]
    missing_args = make_missing_args(args)
    missing_args.missing_temporal_tracker = (
        MissingCandidateTemporalTracker(max_dist=args.missing_temporal_match_dist)
        if args.enable_missing_temporal_evidence
        else None
    )

    si, fi = case["scenario"], case["frame"]
    frame = loader.load_frame(si, fi)
    waypoints = simulate_codriving_waypoints(frame)
    ego_only = filter_visible(frame.perception)
    clean_coop = merge_ego_visible_with_coop_invisible(frame.perception, frame.perception)

    source_messages = {}
    for role in args.source_roles:
        msg = load_aligned_source_message(loader, si, fi, source_role=role)
        if msg is not None:
            source_messages[role] = filter_sender_visible(msg)
    primary_msg = source_messages.get(args.primary_role)
    if primary_msg is None:
        raise RuntimeError(f"Missing primary real source for scenario={si}, frame={fi}")

    primary_raw = merge_ego_visible_with_real_other(frame.perception, primary_msg)
    primary_trust, primary_report = calibrate_shifted_coop(
        frame.perception,
        primary_msg,
        clean_residual_thr=args.clean_residual_thr,
    )
    support_trust = []
    support_reports = []
    support_ids = []
    for role in support_roles:
        support_msg = source_messages.get(role)
        if support_msg is None:
            continue
        perception, report = calibrate_shifted_coop(
            frame.perception,
            support_msg,
            clean_residual_thr=args.clean_residual_thr,
        )
        support_trust.append(perception)
        support_reports.append(report)
        support_ids.append(role)

    evidence = MultiPeerEvidenceSupport(
        support_trust,
        peer_reports=support_reports,
        sender_ids=support_ids,
        sender_trusts=[1.0] * len(support_trust),
        min_support=args.min_peer_support,
        min_trust_support=args.min_trust_support,
        max_dist=args.peer_match_dist,
    )
    guarded, guarded_report, obj_removed = object_impact_guard_perception(
        ego_only,
        primary_trust,
        primary_report,
        hybrid,
        waypoints,
        evidence_tracker=evidence,
        geom_delta_thr=args.guard_geom_delta_thr,
        mod_delta_thr=args.guard_mod_delta_thr,
    )
    guarded_report = copy.deepcopy(guarded_report)
    guarded, n_box_removed = apply_box_margin_guard(guarded, evidence, missing_args, waypoints)
    if n_box_removed:
        obj_removed += n_box_removed
        guarded_report.action = "object_quarantine"
        guarded_report.n_output_agents = len(guarded.agents)

    (
        guarded,
        n_missing_recovered,
        n_missing_candidates,
        n_missing_gt,
        n_missing_tp,
        n_missing_fp,
        missing_records,
    ) = apply_missing_object_recovery(
        "real_multisource",
        si,
        fi,
        guarded,
        primary_trust,
        primary_report,
        clean_coop,
        support_trust,
        support_ids,
        [1.0] * len(support_trust),
        missing_args,
        ego_only,
        hybrid,
        waypoints,
    )

    return {
        "frame": frame,
        "waypoints": waypoints,
        "ego_only": ego_only,
        "clean_coop": clean_coop,
        "primary_raw": primary_raw,
        "primary_trust": primary_trust,
        "primary_report": primary_report,
        "guarded": guarded,
        "guarded_report": guarded_report,
        "evidence": evidence,
        "missing_records": missing_records,
        "stats": {
            "obj_removed": obj_removed,
            "missing_recovered": n_missing_recovered,
            "missing_candidates": n_missing_candidates,
            "missing_gt": n_missing_gt,
            "missing_tp": n_missing_tp,
            "missing_fp": n_missing_fp,
            "source_available": len(source_messages),
        },
        "args": args,
    }


def agent_matches(agent, candidates) -> bool:
    return _geometry_match(agent, candidates, max_dist=2.5, max_size_diff=3.0) is not None


def draw_agent(ax, agent, *, facecolor, edgecolor, alpha=0.65, linewidth=1.1, zorder=3):
    s = agent.state
    length = max(float(s.length), 0.2)
    width = max(float(s.width), 0.2)
    patch = Rectangle(
        (-length / 2.0, -width / 2.0),
        length,
        width,
        facecolor=facecolor,
        edgecolor=edgecolor,
        alpha=alpha,
        linewidth=linewidth,
        zorder=zorder,
    )
    transform = Affine2D().rotate(float(s.heading)).translate(float(s.x), float(s.y)) + ax.transData
    patch.set_transform(transform)
    ax.add_patch(patch)


def draw_scene(case: dict, data: dict, out_path: Path):
    frame = data["frame"]
    base_wp = data["waypoints"]
    primary_wp, primary_stats, _ = eval_method(load_hybrid(use_v1=not getattr(data["args"], "disable_v1", False)), base_wp, data["primary_trust"])
    final_wp, final_stats, _ = eval_method(load_hybrid(use_v1=not getattr(data["args"], "disable_v1", False)), base_wp, data["guarded"])

    fig, ax = plt.subplots(figsize=(8.2, 5.4), dpi=180)
    ax.set_title(f"{case['title']} | scenario {case['scenario']} frame {case['frame']}", fontsize=10)

    for agent in data["ego_only"].agents:
        if -20 <= agent.state.x <= 95 and -35 <= agent.state.y <= 35:
            draw_agent(ax, agent, facecolor="#d8dde6", edgecolor="#8d99a6", alpha=0.50, linewidth=0.8, zorder=1)

    final_coop = [a for a in data["guarded"].agents if a.source == "coop"]
    primary_coop = [a for a in data["primary_trust"].agents if a.source == "coop"]
    primary_keys = {agent_key(a) for a in primary_coop}

    for agent in primary_coop:
        if not (-20 <= agent.state.x <= 95 and -35 <= agent.state.y <= 35):
            continue
        is_fake = str(agent.state.id) == "fake_front"
        kept = agent_matches(agent, final_coop)
        if is_fake and not kept:
            draw_agent(ax, agent, facecolor="#f94144", edgecolor="#9d0208", alpha=0.82, linewidth=1.4, zorder=5)
            ax.scatter([agent.state.x], [agent.state.y], marker="x", s=90, c="#9d0208", linewidths=2.0, zorder=7)
        elif not kept:
            draw_agent(ax, agent, facecolor="#f8961e", edgecolor="#bc6c25", alpha=0.78, linewidth=1.3, zorder=4)
        else:
            draw_agent(ax, agent, facecolor="#90be6d", edgecolor="#2d6a4f", alpha=0.58, linewidth=0.9, zorder=3)

    for agent in final_coop:
        if agent_key(agent) in primary_keys or agent_matches(agent, primary_coop):
            continue
        if -20 <= agent.state.x <= 95 and -35 <= agent.state.y <= 35:
            draw_agent(ax, agent, facecolor="#43aa8b", edgecolor="#006466", alpha=0.88, linewidth=1.5, zorder=6)
            ax.scatter([agent.state.x], [agent.state.y], marker="*", s=110, c="#006466", zorder=8)

    ego_proxy = SimpleNamespace(
        state=SimpleNamespace(x=0.0, y=0.0, heading=0.0, length=4.5, width=1.8)
    )
    draw_agent(ax, ego_proxy, facecolor="#111827", edgecolor="#111827", alpha=0.95, linewidth=1.2, zorder=9)

    ax.plot(base_wp[:, 0], base_wp[:, 1], "--", color="#6b7280", linewidth=1.2, label="Original path", zorder=2)
    ax.plot(primary_wp[:, 0], primary_wp[:, 1], color="#ef4444", linewidth=1.5, label="After TrustCalib", zorder=6)
    ax.plot(final_wp[:, 0], final_wp[:, 1], color="#2563eb", linewidth=1.8, label="Final output", zorder=7)
    ax.scatter(primary_wp[:, 0], primary_wp[:, 1], s=10, color="#ef4444", zorder=6)
    ax.scatter(final_wp[:, 0], final_wp[:, 1], s=12, color="#2563eb", zorder=7)

    primary_coll = check_waypoint_collision(primary_wp, frame)
    final_coll = check_waypoint_collision(final_wp, frame)
    report = data["primary_report"]
    stats = data["stats"]
    note = (
        f"TrustCalib action={report.action}, residual_after={report.offset.residual_after:.2f} m\n"
        f"WPC {primary_coll}/10 -> {final_coll}/10; "
        f"removed={stats.get('obj_removed', 0)}, recovered={stats.get('missing_recovered', 0)}"
    )
    if "fake_seen" in stats:
        note += f", fake {stats['fake_removed']}/{stats['fake_seen']} removed"
    if "source_available" in stats:
        note += f", sources={stats['source_available']}"
    ax.text(
        0.01,
        0.985,
        note,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=7.5,
        bbox=dict(facecolor="white", edgecolor="#cbd5e1", alpha=0.88, boxstyle="round,pad=0.3"),
    )

    handles = [
        Line2D([0], [0], color="#6b7280", linestyle="--", label="Original path"),
        Line2D([0], [0], color="#ef4444", label="After TrustCalib"),
        Line2D([0], [0], color="#2563eb", label="Final output"),
        Rectangle((0, 0), 1, 1, facecolor="#d8dde6", edgecolor="#8d99a6", label="Ego-visible object"),
        Rectangle((0, 0), 1, 1, facecolor="#90be6d", edgecolor="#2d6a4f", label="Kept coop object"),
        Rectangle((0, 0), 1, 1, facecolor="#f94144", edgecolor="#9d0208", label="Quarantined fake"),
        Rectangle((0, 0), 1, 1, facecolor="#43aa8b", edgecolor="#006466", label="Recovered missing"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=7, framealpha=0.9)

    all_x = [0.0, *base_wp[:, 0], *primary_wp[:, 0], *final_wp[:, 0]]
    all_y = [0.0, *base_wp[:, 1], *primary_wp[:, 1], *final_wp[:, 1]]
    for p in (data["ego_only"], data["primary_trust"], data["guarded"]):
        for a in p.agents:
            if -25 <= a.state.x <= 110 and -45 <= a.state.y <= 45:
                all_x.append(float(a.state.x))
                all_y.append(float(a.state.y))
    xmin, xmax = min(all_x) - 6.0, max(all_x) + 6.0
    ymin, ymax = min(all_y) - 6.0, max(all_y) + 6.0
    if ymax - ymin < 24.0:
        center = 0.5 * (ymin + ymax)
        ymin, ymax = center - 12.0, center + 12.0
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color="#e5e7eb", linewidth=0.6)
    ax.set_xlabel("Longitudinal x (m)", fontsize=8)
    ax.set_ylabel("Lateral y (m)", fontsize=8)
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def write_manifest(rows: list[dict]):
    path = FIG_DIR / "qualitative_case_manifest.md"
    lines = [
        "# Qualitative Case Figures",
        "",
        "These figures are generated by `scripts/render_case_figures.py` from the existing DeepAccident pilot outputs and metadata. They are intended as paper-ready qualitative examples, not as additional metric tables.",
        "",
        "| Figure | Scenario/Frame | Purpose |",
        "|---|---:|---|",
    ]
    for row in rows:
        rel = f"figures/{row['file']}"
        lines.append(f"| `{rel}` | {row['scenario']}/{row['frame']} | {row['caption']} |")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    hybrid = load_hybrid(use_v1=True)
    synthetic_loader = DeepAccidentLoader(split="all", include_invisible=True, include_coop=False)
    real_loader = DeepAccidentLoader(split="all", include_invisible=True, include_coop=True)
    rows = []

    for case in SYN_CASES:
        data = replay_synthetic_case(case, synthetic_loader, hybrid)
        out_path = FIG_DIR / f"{case['name']}.png"
        draw_scene(case, data, out_path)
        rows.append({**case, "file": out_path.name})
        print(f"Wrote {out_path}")

    real_hybrid = load_hybrid(use_v1=False)
    for case in REAL_CASES:
        data = replay_real_case(case, real_loader, real_hybrid)
        out_path = FIG_DIR / f"{case['name']}.png"
        draw_scene(case, data, out_path)
        rows.append({**case, "file": out_path.name})
        print(f"Wrote {out_path}")

    write_manifest(rows)
    print(f"Wrote {FIG_DIR / 'qualitative_case_manifest.md'}")


if __name__ == "__main__":
    main()
