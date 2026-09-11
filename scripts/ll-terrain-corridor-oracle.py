#!/usr/bin/env python3
"""Independent LL-003D synthetic terrain/protected-zone corridor oracle.

Research-only geometry for exact fixtures. This does not consume real LOLA/GRAIL
products and does not authorize launches.
"""
from __future__ import annotations
import argparse, json, math
from dataclasses import dataclass, asdict

@dataclass(frozen=True)
class TerrainPoint:
    x_m: float
    elevation_m: float
    sigma_m: float = 0.0

@dataclass(frozen=True)
class TrajectoryPoint:
    x_m: float
    y_m: float
    z_m: float
    sigma_m: float = 0.0

@dataclass(frozen=True)
class ProtectedBox:
    zone_id: str
    min_x_m: float
    max_x_m: float
    min_y_m: float
    max_y_m: float
    min_z_m: float
    max_z_m: float

@dataclass(frozen=True)
class CorridorResult:
    min_nominal_clearance_m: float
    min_conservative_clearance_m: float
    closest_x_m: float
    intersects_protected_zone: bool
    intersected_zone_ids: list[str]

def finite(*values: float) -> bool:
    return all(math.isfinite(v) for v in values)

def validate_terrain(points: list[TerrainPoint]) -> None:
    if len(points) < 2:
        raise ValueError("terrain requires at least two points")
    last = None
    for p in points:
        if not finite(p.x_m, p.elevation_m, p.sigma_m) or p.sigma_m < 0:
            raise ValueError("invalid terrain point")
        if last is not None and p.x_m <= last:
            raise ValueError("terrain x must be strictly increasing")
        last = p.x_m

def validate_trajectory(points: list[TrajectoryPoint]) -> None:
    if not points:
        raise ValueError("trajectory cannot be empty")
    last = None
    for p in points:
        if not finite(p.x_m, p.y_m, p.z_m, p.sigma_m) or p.sigma_m < 0:
            raise ValueError("invalid trajectory point")
        if last is not None and p.x_m < last:
            raise ValueError("trajectory x must be nondecreasing")
        last = p.x_m

def interpolate_terrain(points: list[TerrainPoint], x: float) -> tuple[float, float]:
    validate_terrain(points)
    if not math.isfinite(x) or x < points[0].x_m or x > points[-1].x_m:
        raise ValueError("terrain coverage missing")
    if x == points[-1].x_m:
        p = points[-1]
        return p.elevation_m, p.sigma_m
    for a, b in zip(points, points[1:]):
        if a.x_m <= x <= b.x_m:
            t = (x - a.x_m) / (b.x_m - a.x_m)
            return (
                a.elevation_m + t * (b.elevation_m - a.elevation_m),
                a.sigma_m + t * (b.sigma_m - a.sigma_m),
            )
    raise ValueError("terrain coverage missing")

def segment_intersects_box(a: TrajectoryPoint, b: TrajectoryPoint, box: ProtectedBox) -> bool:
    mins = (box.min_x_m, box.min_y_m, box.min_z_m)
    maxs = (box.max_x_m, box.max_y_m, box.max_z_m)
    p0 = (a.x_m, a.y_m, a.z_m)
    p1 = (b.x_m, b.y_m, b.z_m)
    tmin, tmax = 0.0, 1.0
    for i in range(3):
        d = p1[i] - p0[i]
        if abs(d) < 1e-15:
            if p0[i] < mins[i] or p0[i] > maxs[i]:
                return False
            continue
        inv = 1.0 / d
        t1 = (mins[i] - p0[i]) * inv
        t2 = (maxs[i] - p0[i]) * inv
        if t1 > t2:
            t1, t2 = t2, t1
        tmin = max(tmin, t1)
        tmax = min(tmax, t2)
        if tmin > tmax:
            return False
    return True

def validate_box(box: ProtectedBox) -> None:
    vals = (box.min_x_m, box.max_x_m, box.min_y_m, box.max_y_m, box.min_z_m, box.max_z_m)
    if not box.zone_id or not finite(*vals):
        raise ValueError("invalid protected zone")
    if box.min_x_m > box.max_x_m or box.min_y_m > box.max_y_m or box.min_z_m > box.max_z_m:
        raise ValueError("inverted protected-zone bounds")

def evaluate(
    terrain: list[TerrainPoint],
    trajectory: list[TrajectoryPoint],
    boxes: list[ProtectedBox],
    sigma_multiplier: float = 3.0,
) -> CorridorResult:
    validate_terrain(terrain)
    validate_trajectory(trajectory)
    if not math.isfinite(sigma_multiplier) or sigma_multiplier < 0:
        raise ValueError("invalid sigma multiplier")
    for box in boxes:
        validate_box(box)

    min_nominal = math.inf
    min_conservative = math.inf
    closest_x = math.nan
    for p in trajectory:
        elevation, terrain_sigma = interpolate_terrain(terrain, p.x_m)
        nominal = p.z_m - elevation
        conservative = nominal - sigma_multiplier * (terrain_sigma + p.sigma_m)
        if conservative < min_conservative:
            min_conservative = conservative
            min_nominal = nominal
            closest_x = p.x_m

    hit_ids: list[str] = []
    if len(trajectory) == 1:
        a = trajectory[0]
        for box in boxes:
            if (box.min_x_m <= a.x_m <= box.max_x_m and
                box.min_y_m <= a.y_m <= box.max_y_m and
                box.min_z_m <= a.z_m <= box.max_z_m):
                hit_ids.append(box.zone_id)
    else:
        for box in boxes:
            if any(segment_intersects_box(a, b, box) for a, b in zip(trajectory, trajectory[1:])):
                hit_ids.append(box.zone_id)

    return CorridorResult(min_nominal, min_conservative, closest_x, bool(hit_ids), hit_ids)

def self_test() -> None:
    terrain = [
        TerrainPoint(0.0, 0.0, 1.0),
        TerrainPoint(50.0, 20.0, 2.0),
        TerrainPoint(100.0, 0.0, 1.0),
    ]
    trajectory = [
        TrajectoryPoint(0.0, 0.0, 30.0, 0.5),
        TrajectoryPoint(50.0, 0.0, 40.0, 0.5),
        TrajectoryPoint(100.0, 0.0, 30.0, 0.5),
    ]
    result = evaluate(terrain, trajectory, [], 3.0)
    assert abs(result.min_nominal_clearance_m - 20.0) < 1e-12
    assert abs(result.min_conservative_clearance_m - 12.5) < 1e-12
    assert result.closest_x_m == 50.0

    more_uncertain = [
        TrajectoryPoint(p.x_m, p.y_m, p.z_m, p.sigma_m + 2.0)
        for p in trajectory
    ]
    result_more = evaluate(terrain, more_uncertain, [], 3.0)
    assert result_more.min_conservative_clearance_m < result.min_conservative_clearance_m

    try:
        evaluate(terrain, [TrajectoryPoint(101.0, 0.0, 30.0, 0.0)], [], 3.0)
    except ValueError:
        pass
    else:
        raise AssertionError("missing terrain coverage must fail")

    zone = ProtectedBox("habitat", 40.0, 60.0, -5.0, 5.0, 0.0, 100.0)
    hit = evaluate(terrain, trajectory, [zone], 3.0)
    assert hit.intersects_protected_zone and hit.intersected_zone_ids == ["habitat"]

    clear_zone = ProtectedBox("high-box", 40.0, 60.0, -5.0, 5.0, 100.1, 200.0)
    clear = evaluate(terrain, trajectory, [clear_zone], 3.0)
    assert not clear.intersects_protected_zone

    boundary = ProtectedBox("boundary", 50.0, 50.0, 0.0, 0.0, 40.0, 40.0)
    boundary_hit = evaluate(terrain, trajectory, [boundary], 3.0)
    assert boundary_hit.intersects_protected_zone

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--json")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        print("ok")
        return
    if not args.json:
        parser.error("--json or --self-test required")
    payload = json.loads(args.json)
    terrain = [TerrainPoint(**item) for item in payload["terrain"]]
    trajectory = [TrajectoryPoint(**item) for item in payload["trajectory"]]
    boxes = [ProtectedBox(**item) for item in payload.get("protected_boxes", [])]
    result = evaluate(terrain, trajectory, boxes, payload.get("sigma_multiplier", 3.0))
    print(json.dumps(asdict(result), sort_keys=True))

if __name__ == "__main__":
    main()
