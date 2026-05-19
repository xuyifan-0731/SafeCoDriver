"""Generate GIFs and trajectory plots for all DeepAccident scenarios.

Outputs by default:
  paper/figures/deepaccident/gifs/<scenario>.gif
  paper/figures/deepaccident/trajectories/<scenario>.png
  paper/figures/deepaccident/index.csv

The GIF uses the dataset BEV_instance_camera frames. The trajectory plot uses
the ego_vehicle label files, so trajectories are ego-centric.
"""
from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.deepaccident_loader import DeepAccidentLoader, parse_label


DEFAULT_OUT = ROOT / "paper" / "figures" / "deepaccident"

TYPE_COLORS = {
    "car": "#3f4a56",
    "van": "#5d6772",
    "truck": "#8b6f47",
    "bus": "#7f8c8d",
    "motorcycle": "#8e44ad",
    "pedestrian": "#d35400",
    "ego": "#1565c0",
    "collision": "#d32f2f",
    "invisible": "#f39c12",
}


def slugify(name: str) -> str:
    """Create a stable, filesystem-safe name from a scenario name."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "__", name).strip("_")


def load_font(size: int) -> ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ):
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def scenario_type_dir(scenario: dict[str, Any]) -> Path:
    # data/DeepAccident/<type>/ego_vehicle/label/<scenario>
    return Path(scenario["ego_dir"]).parents[2]


def bev_dir_for(scenario: dict[str, Any]) -> Path:
    type_dir = scenario_type_dir(scenario)
    return type_dir / "ego_vehicle" / "BEV_instance_camera" / Path(scenario["ego_dir"]).name


def label_path_for(scenario: dict[str, Any], frame_name: str) -> Path:
    return Path(scenario["ego_dir"]) / frame_name


def bev_path_for(scenario: dict[str, Any], frame_name: str) -> Path:
    return bev_dir_for(scenario) / f"{Path(frame_name).stem}.npz"


def resize_image(img: Image.Image, size: int) -> Image.Image:
    resampling = getattr(Image, "Resampling", Image).LANCZOS
    img.thumbnail((size, size), resampling)
    canvas = Image.new("RGB", (size, size), (245, 246, 248))
    x = (size - img.width) // 2
    y = (size - img.height) // 2
    canvas.paste(img, (x, y))
    return canvas


def overlay_header(
    img: Image.Image,
    scenario_name: str,
    frame_idx: int,
    n_frames: int,
    is_accident: bool,
    collision_frame: int,
    font: ImageFont.ImageFont,
) -> Image.Image:
    img = img.convert("RGBA")
    draw = ImageDraw.Draw(img)
    header_h = 44
    draw.rectangle((0, 0, img.width, header_h), fill=(0, 0, 0, 150))
    status = "accident" if is_accident else "normal"
    coll = f" collision_frame={collision_frame}" if collision_frame > 0 else ""
    text = f"{scenario_name} | {status} | frame {frame_idx + 1}/{n_frames}{coll}"
    draw.text((10, 12), text, fill=(255, 255, 255, 255), font=font)
    return img.convert("RGB")


def select_frame_indices(scenario: dict[str, Any], collision_window: int | None) -> list[int]:
    """Select either all frames or the representative pre-collision window."""
    n_frames = len(scenario["frames"])
    if n_frames == 0:
        return []
    if not collision_window or collision_window <= 0 or not scenario["is_accident"]:
        return list(range(n_frames))

    collision_frame = int(scenario.get("collision_frame", -1))
    if collision_frame <= 0:
        start = max(0, n_frames - collision_window)
        return list(range(start, n_frames))

    # DeepAccident accident labels often stop before the actual collision frame.
    # Use the available frames closest to the collision.
    end = min(n_frames - 1, collision_frame)
    start = max(0, end - collision_window + 1, collision_frame - collision_window)
    if start > end:
        start = max(0, n_frames - collision_window)
        end = n_frames - 1
    return list(range(start, end + 1))


def make_bev_gif(
    scenario: dict[str, Any],
    scenario_name: str,
    output_path: Path,
    frame_indices: list[int],
    stride: int,
    size: int,
    fps: float,
) -> int:
    frames: list[Image.Image] = []
    font = load_font(max(12, size // 42))
    frame_names = scenario["frames"]
    selected = frame_indices[::stride]

    for frame_idx in selected:
        frame_name = frame_names[frame_idx]
        npz_path = bev_path_for(scenario, frame_name)
        arr = np.load(npz_path)["data"]
        img = Image.fromarray(arr)
        img = resize_image(img, size)
        img = overlay_header(
            img,
            scenario_name,
            frame_idx,
            len(frame_names),
            bool(scenario["is_accident"]),
            int(scenario.get("collision_frame", -1)),
            font,
        )
        frames.append(img)

    if not frames:
        raise RuntimeError(f"No BEV frames found for {scenario_name}")

    duration_ms = max(20, int(1000 / fps))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
    )
    return len(frames)


def collect_tracks(
    scenario: dict[str, Any],
    frame_indices: list[int] | None = None,
) -> dict[str, dict[str, Any]]:
    tracks: dict[str, dict[str, Any]] = {}
    selected = frame_indices if frame_indices is not None else list(range(len(scenario["frames"])))

    for frame_idx in selected:
        frame_name = scenario["frames"][frame_idx]
        label = parse_label(str(label_path_for(scenario, frame_name)), include_invisible=True)
        for agent in label["agents"]:
            if agent["id"] == -100:
                continue
            aid = str(agent["id"])
            track = tracks.setdefault(
                aid,
                {
                    "x": [],
                    "y": [],
                    "frame": [],
                    "type": agent["type"],
                    "visible": [],
                    "collision": [],
                },
            )
            track["x"].append(float(agent["x"]))
            track["y"].append(float(agent["y"]))
            track["frame"].append(frame_idx)
            track["visible"].append(bool(agent["visible"]))
            track["collision"].append(int(agent["collision"]))

    return tracks


def padded_limits(xs: list[float], ys: list[float]) -> tuple[tuple[float, float], tuple[float, float]]:
    xs = xs or [0.0]
    ys = ys or [0.0]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    span = max(x_max - x_min, y_max - y_min, 30.0)
    cx = (x_min + x_max) / 2
    cy = (y_min + y_max) / 2
    half = span / 2 + max(10.0, span * 0.08)
    return (cx - half, cx + half), (cy - half, cy + half)


def plot_trajectories(
    scenario: dict[str, Any],
    scenario_name: str,
    output_path: Path,
    frame_indices: list[int],
) -> tuple[int, int]:
    tracks = collect_tracks(scenario, frame_indices)
    all_x = [0.0]
    all_y = [0.0]
    for tr in tracks.values():
        all_x.extend(tr["x"])
        all_y.extend(tr["y"])

    xlim, ylim = padded_limits(all_x, all_y)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_facecolor("#f7f8fa")
    ax.grid(True, color="#d5dbe1", linewidth=0.6, alpha=0.55)

    ax.scatter([0], [0], marker="*", s=220, color=TYPE_COLORS["ego"],
               edgecolor="white", linewidth=1.2, zorder=8, label="ego")
    ax.annotate("ego", (0, 0), xytext=(6, 6), textcoords="offset points",
                fontsize=8, color=TYPE_COLORS["ego"], weight="bold")

    n_collision_tracks = 0
    for aid, tr in sorted(tracks.items(), key=lambda item: int(item[0])):
        xs = tr["x"]
        ys = tr["y"]
        if len(xs) == 0:
            continue
        has_collision = any(v > 0 for v in tr["collision"])
        ever_invisible = not all(tr["visible"])
        n_collision_tracks += int(has_collision)

        if has_collision:
            color = TYPE_COLORS["collision"]
            linewidth = 2.4
            alpha = 0.95
            zorder = 5
        elif ever_invisible:
            color = TYPE_COLORS["invisible"]
            linewidth = 1.6
            alpha = 0.75
            zorder = 4
        else:
            color = TYPE_COLORS.get(tr["type"], "#607d8b")
            linewidth = 1.2
            alpha = 0.55
            zorder = 3

        linestyle = "--" if ever_invisible and not has_collision else "-"
        ax.plot(xs, ys, linestyle=linestyle, color=color, linewidth=linewidth,
                alpha=alpha, zorder=zorder)
        ax.scatter(xs[0], ys[0], marker="o", s=22, color=color, alpha=0.85,
                   edgecolor="white", linewidth=0.4, zorder=zorder + 1)
        ax.scatter(xs[-1], ys[-1], marker="s", s=24, color=color, alpha=0.85,
                   edgecolor="black", linewidth=0.35, zorder=zorder + 1)

        if has_collision:
            coll_values = [(c, i) for i, c in enumerate(tr["collision"]) if c > 0]
            _, local_idx = min(coll_values, key=lambda item: item[0])
            ax.scatter(xs[local_idx], ys[local_idx], marker="X", s=120,
                       color=TYPE_COLORS["collision"], edgecolor="black",
                       linewidth=0.8, zorder=9)
            ax.annotate(
                aid,
                (xs[local_idx], ys[local_idx]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=7,
                color=TYPE_COLORS["collision"],
                weight="bold",
            )

    handles = [
        Patch(facecolor=TYPE_COLORS["ego"], label="ego"),
        Patch(facecolor=TYPE_COLORS["collision"], label="collision countdown > 0"),
        Patch(facecolor=TYPE_COLORS["invisible"], label="invisible at least once"),
        Patch(facecolor="#607d8b", label="visible non-collision agent"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.9)

    status = "accident" if scenario["is_accident"] else "normal"
    collision_frame = int(scenario.get("collision_frame", -1))
    title = (
        f"{scenario_name}\n"
        f"{status}, selected_frames={len(frame_indices)}/{len(scenario['frames'])}, "
        f"frame_range={frame_indices[0] if frame_indices else 'n/a'}-"
        f"{frame_indices[-1] if frame_indices else 'n/a'}, agents={len(tracks)}, "
        f"collision_frame={collision_frame if collision_frame > 0 else 'n/a'}"
    )
    ax.set_title(title, fontsize=11, weight="bold")
    ax.set_xlabel("ego-centric x (m)")
    ax.set_ylabel("ego-centric y (m)")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return len(tracks), n_collision_tracks


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=None, help="DeepAccident root. Defaults to loader setting.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--stride", type=int, default=2, help="Use every Nth BEV frame in GIFs.")
    parser.add_argument("--size", type=int, default=512, help="GIF canvas size in pixels.")
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--limit", type=int, default=None, help="Optional scenario limit for smoke tests.")
    parser.add_argument("--accident-only", action="store_true", help="Only generate accident scenarios.")
    parser.add_argument(
        "--collision-window",
        type=int,
        default=None,
        help="Use only the N frames closest to the collision frame for accident scenarios.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    if args.stride < 1:
        raise ValueError("--stride must be >= 1")
    if args.size < 128:
        raise ValueError("--size must be >= 128")

    out_dir = Path(args.out_dir)
    gif_dir = out_dir / "gifs"
    traj_dir = out_dir / "trajectories"
    gif_dir.mkdir(parents=True, exist_ok=True)
    traj_dir.mkdir(parents=True, exist_ok=True)

    loader = DeepAccidentLoader(data_root=args.data_root, split="all", include_coop=False)
    scenarios = loader.scenarios
    if args.accident_only:
        scenarios = [s for s in scenarios if s["is_accident"]]
    scenarios = scenarios[: args.limit] if args.limit else scenarios
    rows: list[dict[str, Any]] = []

    print(f"[gen] output: {out_dir}")
    print(f"[gen] scenarios: {len(scenarios)} / {len(loader.scenarios)}")
    print(
        f"[gen] gif stride={args.stride}, size={args.size}, fps={args.fps}, "
        f"accident_only={args.accident_only}, collision_window={args.collision_window}"
    )

    for idx, scenario in enumerate(scenarios, start=1):
        scenario_name = scenario["name"]
        slug = slugify(scenario_name)
        gif_path = gif_dir / f"{slug}.gif"
        traj_path = traj_dir / f"{slug}.png"

        gif_frames = 0
        n_agents = 0
        n_collision_tracks = 0
        status = "ok"
        frame_indices = select_frame_indices(scenario, args.collision_window)

        try:
            if args.overwrite or not gif_path.exists():
                gif_frames = make_bev_gif(
                    scenario,
                    scenario_name,
                    gif_path,
                    frame_indices=frame_indices,
                    stride=args.stride,
                    size=args.size,
                    fps=args.fps,
                )
            else:
                gif_frames = math.ceil(len(frame_indices) / args.stride)

            if args.overwrite or not traj_path.exists():
                n_agents, n_collision_tracks = plot_trajectories(
                    scenario,
                    scenario_name,
                    traj_path,
                    frame_indices=frame_indices,
                )
            else:
                n_agents = len(collect_tracks(scenario, frame_indices))
                n_collision_tracks = -1

        except Exception as exc:
            status = f"error: {exc}"
            print(f"[{idx:03d}/{len(scenarios):03d}] {scenario_name}: {status}")

        rows.append(
            {
                "idx": idx - 1,
                "scenario": scenario_name,
                "is_accident": bool(scenario["is_accident"]),
                "n_frames": len(scenario["frames"]),
                "selected_start": frame_indices[0] if frame_indices else "",
                "selected_end": frame_indices[-1] if frame_indices else "",
                "selected_frames": len(frame_indices),
                "collision_frame": int(scenario.get("collision_frame", -1)),
                "gif_frames": gif_frames,
                "n_agents": n_agents,
                "n_collision_tracks": n_collision_tracks,
                "gif": str(gif_path),
                "trajectory": str(traj_path),
                "status": status,
            }
        )

        print(
            f"[{idx:03d}/{len(scenarios):03d}] {scenario_name}: "
            f"{status}, selected={len(frame_indices)}/{len(scenario['frames'])}, "
            f"gif_frames={gif_frames}, agents={n_agents}"
        )

    index_path = out_dir / "index.csv"
    with index_path.open("w", newline="") as f:
        if rows:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    ok = sum(1 for row in rows if row["status"] == "ok")
    print(f"[gen] done: {ok}/{len(rows)} ok")
    print(f"[gen] index: {index_path}")


if __name__ == "__main__":
    main()
