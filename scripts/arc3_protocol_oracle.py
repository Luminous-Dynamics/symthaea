#!/usr/bin/env python3
"""Independent ARC3-001 protocol-v1 canonical-byte oracle.

This file intentionally does not import Symthaea code. It implements the frozen
byte contract a second time so Rust and the evidence spec do not share the same
canonicalization helper.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

SCHEMA = b"symthaea.arc3.semantic-observation.v1\0"
ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "docs/arc-agi-3/fixtures/protocol-v1-observation.json"
EXPECTED = ROOT / "docs/arc-agi-3/fixtures/protocol-v1-observation.hex"

TOP_KEYS = {
    "game_id",
    "frame",
    "state",
    "levels_completed",
    "win_levels",
    "action_input",
    "guid",
    "full_reset",
    "available_actions",
}
ACTION_KEYS = {"id", "data", "reasoning"}
STATE_CODE = {"NOT_PLAYED": 0, "NOT_FINISHED": 1, "WIN": 2, "GAME_OVER": 3}
ACTION_ID = {"RESET": 0, **{f"ACTION{i}": i for i in range(1, 8)}}


def require_exact_keys(obj: dict[str, Any], allowed: set[str], where: str) -> None:
    extra = set(obj) - allowed
    if extra:
        raise ValueError(f"unexpected {where} fields: {sorted(extra)}")


def parse_action_kind(raw: Any) -> int:
    if isinstance(raw, int) and not isinstance(raw, bool):
        if 0 <= raw <= 7:
            return raw
        raise ValueError(f"invalid action id: {raw}")
    if isinstance(raw, str):
        try:
            return ACTION_ID[raw.upper()]
        except KeyError as exc:
            raise ValueError(f"invalid action name: {raw}") from exc
    raise ValueError("action id must be integer or string")


def validate_frame(frames: Any) -> list[list[list[int]]]:
    if not isinstance(frames, list) or not frames:
        raise ValueError("frame must contain at least one grid")
    for index, grid in enumerate(frames):
        if not isinstance(grid, list) or not (1 <= len(grid) <= 64):
            raise ValueError(f"invalid grid height at frame {index}")
        if not isinstance(grid[0], list) or not (1 <= len(grid[0]) <= 64):
            raise ValueError(f"invalid grid width at frame {index}")
        width = len(grid[0])
        for row in grid:
            if not isinstance(row, list) or len(row) != width:
                raise ValueError(f"ragged grid at frame {index}")
            for cell in row:
                if not isinstance(cell, int) or isinstance(cell, bool) or not (0 <= cell <= 15):
                    raise ValueError(f"invalid cell at frame {index}")
    return frames


def normalize_action(raw: Any, observation_game_id: str) -> tuple[int, tuple[int, int] | None] | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError("action_input must be an object")
    require_exact_keys(raw, ACTION_KEYS, "action_input")
    kind = parse_action_kind(raw["id"])
    data = raw.get("data", {})
    if not isinstance(data, dict):
        raise ValueError("action_input.data must be an object")
    allowed = {"game_id"} | ({"x", "y"} if kind == 6 else set())
    require_exact_keys(data, allowed, "action data")
    action_game_id = data.get("game_id")
    if action_game_id is not None:
        if not isinstance(action_game_id, str):
            raise ValueError("action game_id must be a string")
        if observation_game_id and action_game_id and action_game_id != observation_game_id:
            raise ValueError("observation/action game_id mismatch")
    if kind != 6:
        return (kind, None)
    x, y = data.get("x"), data.get("y")
    if not all(isinstance(v, int) and not isinstance(v, bool) and 0 <= v <= 63 for v in (x, y)):
        raise ValueError("ACTION6 requires x/y in 0..63")
    return (kind, (x, y))


def canonical_bytes(raw: dict[str, Any]) -> bytes:
    require_exact_keys(raw, TOP_KEYS, "observation")
    game_id = raw.get("game_id", "")
    if not isinstance(game_id, str):
        raise ValueError("game_id must be a string")
    state = raw["state"]
    if state not in STATE_CODE:
        raise ValueError(f"invalid state: {state}")
    levels = raw.get("levels_completed", 0)
    wins = raw.get("win_levels", 0)
    if not all(isinstance(v, int) and not isinstance(v, bool) and 0 <= v <= 254 for v in (levels, wins)):
        raise ValueError("level counts must be integers in 0..254")
    full_reset = raw.get("full_reset", False)
    if not isinstance(full_reset, bool):
        raise ValueError("full_reset must be boolean")
    frames = validate_frame(raw["frame"])
    action = normalize_action(raw.get("action_input"), game_id)

    available_raw = raw.get("available_actions", [])
    if not isinstance(available_raw, list):
        raise ValueError("available_actions must be a list")
    available = sorted(parse_action_kind(v) for v in available_raw)
    if len(available) != len(set(available)):
        raise ValueError("duplicate available action")

    out = bytearray(SCHEMA)
    out.append(STATE_CODE[state])
    out += levels.to_bytes(2, "big")
    out += wins.to_bytes(2, "big")
    out.append(1 if full_reset else 0)
    if action is None:
        out.append(0)
    else:
        kind, coordinates = action
        out.extend((1, kind))
        if coordinates is not None:
            out.extend(coordinates)
    out.append(len(available))
    out.extend(available)
    out += len(frames).to_bytes(4, "big")
    for grid in frames:
        out.extend((len(grid), len(grid[0])))
        for row in grid:
            out.extend(row)
    return bytes(out)


def main() -> None:
    raw = json.loads(FIXTURE.read_text())
    actual = canonical_bytes(raw).hex()
    expected = EXPECTED.read_text().strip()
    if actual != expected:
        raise SystemExit(f"ARC3 protocol oracle FAIL\nexpected={expected}\nactual={actual}")
    print(f"ARC3 protocol oracle PASS ({len(bytes.fromhex(actual))} canonical bytes)")


if __name__ == "__main__":
    main()
