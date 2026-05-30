"""Synthetic cooperative-message anomaly injection for DeepAccident pilots."""
from __future__ import annotations

import copy
import math

import numpy as np

from coop_safety.interface import Agent, AgentType, PerceptionResult, VehicleState


def clone_perception(perception: PerceptionResult) -> PerceptionResult:
    return copy.deepcopy(perception)


def shift_message(perception: PerceptionResult, dx: float, dy: float) -> PerceptionResult:
    msg = clone_perception(perception)
    for agent in msg.agents:
        agent.state.x += dx
        agent.state.y += dy
        agent.source = "coop_shift"
    return msg


def noise_message(perception: PerceptionResult, sigma: float, rng: np.random.Generator) -> PerceptionResult:
    msg = clone_perception(perception)
    for agent in msg.agents:
        nx, ny = rng.normal(0.0, sigma, size=2)
        agent.state.x += float(nx)
        agent.state.y += float(ny)
        agent.source = "coop_noise"
    return msg


def stale_message(perception: PerceptionResult, delay_s: float) -> PerceptionResult:
    msg = clone_perception(perception)
    for agent in msg.agents:
        agent.state.x -= agent.state.vx * delay_s
        agent.state.y -= agent.state.vy * delay_s
        agent.source = "coop_stale"
    return msg


def drop_invisible_message(
    perception: PerceptionResult,
    drop_rate: float,
    rng: np.random.Generator,
) -> PerceptionResult:
    msg = clone_perception(perception)
    kept = []
    for agent in msg.agents:
        if (not agent.is_visible) and rng.random() < drop_rate:
            continue
        kept.append(agent)
    msg.agents = kept
    for agent in msg.agents:
        agent.source = "coop_drop"
    return msg


def fake_front_message(
    perception: PerceptionResult,
    x: float,
    y: float = 0.0,
    vx: float = 0.0,
    vy: float = 0.0,
) -> PerceptionResult:
    msg = clone_perception(perception)
    fake = Agent(
        state=VehicleState(
            id="fake_front",
            x=x,
            y=y,
            heading=0.0,
            velocity=math.hypot(vx, vy),
            vx=vx,
            vy=vy,
            length=4.5,
            width=1.8,
            vehicle_type="car",
        ),
        agent_type=AgentType.VEHICLE,
        is_visible=False,
        confidence=0.3,
        source="coop_fake",
    )
    msg.agents.append(fake)
    for agent in msg.agents:
        if agent.state.id != "fake_front":
            agent.source = "coop_fake_context"
    return msg


def inject_anomaly(
    perception: PerceptionResult,
    mode: str,
    rng: np.random.Generator,
    ego_horizon_s: float = 1.5,
    shift_x: float = 2.0,
    shift_y: float = 1.0,
    noise_sigma: float = 1.5,
    drop_rate: float = 0.7,
    stale_delay_s: float = 1.0,
) -> PerceptionResult:
    if "+" in mode:
        msg = perception
        for part in [p.strip() for p in mode.split("+") if p.strip()]:
            if part == "clean":
                continue
            msg = inject_anomaly(
                msg,
                mode=part,
                rng=rng,
                ego_horizon_s=ego_horizon_s,
                shift_x=shift_x,
                shift_y=shift_y,
                noise_sigma=noise_sigma,
                drop_rate=drop_rate,
                stale_delay_s=stale_delay_s,
            )
        return msg
    if mode == "clean":
        return clone_perception(perception)
    if mode == "shift":
        return shift_message(perception, shift_x, shift_y)
    if mode == "shift_severe":
        return shift_message(perception, shift_x * 2.0, shift_y * 2.0)
    if mode == "noise":
        return noise_message(perception, noise_sigma, rng)
    if mode == "drop":
        return drop_invisible_message(perception, drop_rate, rng)
    if mode == "stale":
        return stale_message(perception, stale_delay_s)
    if mode == "fake_front":
        ego = perception.ego
        x = max(ego.velocity, 2.0) * ego_horizon_s
        return fake_front_message(perception, x=x, y=0.0, vx=0.0, vy=0.0)
    raise ValueError(f"Unknown anomaly mode: {mode}")
