#!/usr/bin/env python3
"""Independent, test-only semantic replay for the Rust synthetic task generator.

This standard-library reference is intentionally not used by production CLI
paths. It independently reconstructs task semantics from the revealed seed and
the exact generator config bytes, then compares parsed task structures. The
Rust verifier remains responsible for byte commitments and canonical-byte
replay; neither implementation proves runner isolation or scientific truth.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

TASK_GENERATOR_DOMAIN = b"symthaea.science_bench_refresh.taskgen.v1\0"
TASK_SET_SCHEMA = "symthaea.science_bench_refresh_task_set.v1"
CONFIG_SCHEMA = "symthaea.science_bench_refresh_generator_config.v1"
U64_MASK = (1 << 64) - 1


def read_bytes(path: str, label: str) -> bytes:
    try:
        return Path(path).read_bytes()
    except OSError as exc:
        raise ValueError(f"cannot read {label} {path}: {exc}") from exc


def parse_json(data: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(data)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError(f"{label} is not valid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def framed_hash(domain: bytes, parts: list[bytes]) -> bytes:
    hasher = hashlib.sha256()
    hasher.update(domain)
    for part in parts:
        hasher.update(len(part).to_bytes(8, "big"))
        hasher.update(part)
    return hasher.digest()


def seed_word(
    seed: bytes,
    generation_id: str,
    family_id: str,
    local_index: int,
    field: str,
) -> int:
    return int.from_bytes(
        framed_hash(
            TASK_GENERATOR_DOMAIN,
            [
                seed,
                generation_id.encode("utf-8"),
                family_id.encode("utf-8"),
                local_index.to_bytes(8, "big"),
                field.encode("utf-8"),
            ],
        )[:8],
        "big",
    )


def unit_fraction(
    seed: bytes, generation_id: str, family_id: str, index: int, field: str
) -> float:
    return (seed_word(seed, generation_id, family_id, index, field) % 1_000_001) / 1_000_000.0


def coefficient(
    seed: bytes, generation_id: str, family_id: str, index: int, field: str
) -> float:
    return float(seed_word(seed, generation_id, family_id, index, field) % 7 - 3)


def observation_noise(
    seed: bytes,
    generation_id: str,
    family_id: str,
    task_index: int,
    sample_index: int,
    amplitude: float,
) -> float:
    if amplitude == 0.0:
        return 0.0
    compound_index = ((task_index * 1_000_003) + sample_index) & U64_MASK
    raw = seed_word(
        seed, generation_id, family_id, compound_index, "observation_noise"
    ) % 2_000_001
    return ((raw / 1_000_000.0) - 1.0) * amplitude


def rust_display_float(value: float) -> str:
    """Match Rust Display for the integral scales/offsets used in instructions."""
    if value.is_integer():
        return format(value, ".0f")
    return repr(value)


def opaque_task_id(seed: bytes, generation_id: str, family_id: str, index: int) -> str:
    digest = framed_hash(
        TASK_GENERATOR_DOMAIN + b"opaque-task-id.v1\0",
        [
            seed,
            generation_id.encode("utf-8"),
            family_id.encode("utf-8"),
            index.to_bytes(8, "big"),
        ],
    )
    return f"task-{digest.hex()}"


def make_task(
    generation_id: str,
    seed: bytes,
    recipe: dict[str, Any],
    index: int,
) -> dict[str, Any]:
    family_id = str(recipe["family_id"])
    kind = str(recipe["kind"])
    instruction = str(recipe["instruction"])
    samples = int(recipe["samples_per_task"])
    x_start = float(recipe["x_start"])
    x_step = float(recipe["x_step"])
    noise_amplitude = float(recipe["noise_amplitude"])
    slope = 1.0 + unit_fraction(seed, generation_id, family_id, index, "slope") * 4.0
    intercept = coefficient(seed, generation_id, family_id, index, "intercept")
    task_id = opaque_task_id(seed, generation_id, family_id, index)

    if kind == "semantic_invariance":
        scales = [0.25, 0.5, 2.0, 4.0]
        scale_index = seed_word(seed, generation_id, family_id, index, "scale") % 4
        scale = scales[scale_index]
        offset = coefficient(seed, generation_id, family_id, index, "coordinate_offset")
        observations = []
        for sample in range(samples):
            latent_x = x_start + float(sample) * x_step
            visible_u = scale * latent_x + offset
            y = slope * latent_x + intercept
            observations.append({"u": visible_u, "y": y})
        visible_slope = slope / scale
        visible_intercept = intercept - slope * offset / scale
        policy_input = {
            "instruction": instruction,
            "coordinate_semantics": (
                f"visible coordinate u = {rust_display_float(scale)} * x + "
                f"({rust_display_float(offset)})"
            ),
            "observations": observations,
            "declared_measurement_noise": 0.0,
        }
        evaluator_only = {
            "target_kind": "unique_law_under_declared_reparameterization",
            "latent_law": {"slope": slope, "intercept": intercept},
            "visible_coordinate_law": {
                "slope": visible_slope,
                "intercept": visible_intercept,
            },
            "inverse_coordinate_transform": {"scale": scale, "offset": offset},
        }
    elif kind == "sampling_noise_shift":
        observations = []
        for sample in range(samples):
            x = x_start + float(sample) * x_step
            err = observation_noise(
                seed, generation_id, family_id, index, sample, noise_amplitude
            )
            observations.append({"x": x, "y": slope * x + intercept + err})
        policy_input = {
            "instruction": instruction,
            "observations": observations,
            "declared_measurement_noise_amplitude": noise_amplitude,
            "sample_budget": samples,
        }
        evaluator_only = {
            "target_kind": "noisy_linear_law",
            "law": {"slope": slope, "intercept": intercept},
            "noise_amplitude": noise_amplitude,
        }
    elif kind == "confounding_null":
        observations = []
        for sample in range(samples):
            z = x_start + float(sample) * x_step
            x_error = observation_noise(
                seed, generation_id, family_id, index, sample * 2, noise_amplitude
            )
            y_error = observation_noise(
                seed, generation_id, family_id, index, sample * 2 + 1, noise_amplitude
            )
            observations.append({"z": z, "x": z + x_error, "y": z + y_error})
        policy_input = {
            "instruction": instruction,
            "observations": observations,
            "causal_warning": "an observed association may be explained by a common cause",
            "sample_budget": samples,
        }
        evaluator_only = {
            "target_kind": "no_direct_effect",
            "common_cause": "z",
            "direct_effect": 0.0,
            "noise_amplitude": noise_amplitude,
        }
    elif kind == "identifiability_abstention":
        narrow_step = x_step / 100.0
        observations = []
        for sample in range(samples):
            x = x_start + float(sample) * narrow_step
            err = observation_noise(
                seed, generation_id, family_id, index, sample, noise_amplitude
            )
            observations.append({"x": x, "y": slope * x + intercept + err})
        roots = [row["x"] for row in observations]
        challenge_index = float(samples) + 10.0
        challenge_x = x_start + challenge_index * x_step
        policy_input = {
            "instruction": instruction,
            "observations": observations,
            "observed_x_span": float(max(samples - 1, 0)) * narrow_step,
            "declared_measurement_noise_amplitude": noise_amplitude,
            "required_response_mode": "report_identifiability_or_abstain",
        }
        evaluator_only = {
            "target_kind": "not_identifiable_from_declared_sample",
            "one_compatible_law": {
                "kind": "linear",
                "slope": slope,
                "intercept": intercept,
            },
            "alternative_compatible_law": {
                "kind": "linear_plus_vanishing_polynomial",
                "base_slope": slope,
                "base_intercept": intercept,
                "coefficient": 1.0,
                "roots": roots,
                "formula": "f_alt(x) = slope*x + intercept + coefficient*product_i(x - observed_x_i)",
            },
            "identifiability_witness": {
                "equal_on_observed_x": True,
                "challenge_x_outside_observed_range": challenge_x,
                "challenge_x_is_observed": False,
                "claim": (
                    "both candidate mean functions agree at every observed x; "
                    "the alternative is not constrained to extrapolate like the linear law"
                ),
            },
            "noise_amplitude": noise_amplitude,
            "claim_limit": "the observations do not uniquely establish a global law",
        }
    else:
        raise ValueError(f"unsupported recipe kind: {kind}")

    return {
        "task_id": task_id,
        "family_id": family_id,
        "policy_input": policy_input,
        "evaluator_only": evaluator_only,
    }


def independently_generate(
    manifest: dict[str, Any], seed: bytes, config: dict[str, Any]
) -> dict[str, Any]:
    generation_id = manifest["generation_id"]
    recipes = config.get("recipes")
    if not isinstance(recipes, list):
        raise ValueError("generator config recipes must be a list")
    canonical_templates = {
        "semantic_invariance": "Recover the relationship invariant under the declared coordinate transformation.",
        "sampling_noise_shift": "Recover the relationship under declared measurement noise and report uncertainty.",
        "confounding_null": "Decide whether the apparent association is a direct effect or explained by a common cause.",
        "identifiability_abstention": "Determine what is identifiable from the narrow observed domain; state an equivalence class or abstain.",
    }
    for recipe in recipes:
        kind = str(recipe["kind"])
        family_id = str(recipe["family_id"])
        if family_id != kind or kind not in canonical_templates:
            raise ValueError("generator recipe family/kind is not in the audited v1 profile")
        if recipe.get("instruction") != canonical_templates[kind]:
            raise ValueError("recipe instruction differs from the frozen, audited policy-visible template")

    recipe_by_family = {str(recipe["family_id"]): recipe for recipe in recipes}
    manifest_families = {
        str(family["family_id"]): int(family["planned_task_count"])
        for family in manifest["task_family_mixture"]
    }
    if set(recipe_by_family) != set(manifest_families):
        raise ValueError("generator config families differ from the manifest")

    tasks: list[dict[str, Any]] = []
    for family in manifest["task_family_mixture"]:
        family_id = str(family["family_id"])
        for index in range(int(family["planned_task_count"])):
            tasks.append(
                make_task(generation_id, seed, recipe_by_family[family_id], index)
            )
    tasks.sort(key=lambda row: row["task_id"])
    return {
        "schema": TASK_SET_SCHEMA,
        "generation_id": generation_id,
        "tasks": tasks,
    }


def main(argv: list[str]) -> int:
    if len(argv) != 4:
        print(
            "usage: reference_replay.py MANIFEST.json SEED.bin "
            "GENERATOR_CONFIG.json GENERATED_TASK_SET.json",
            file=sys.stderr,
        )
        return 2
    manifest_bytes = read_bytes(argv[0], "manifest")
    seed = read_bytes(argv[1], "seed")
    config_bytes = read_bytes(argv[2], "generator config")
    task_set_bytes = read_bytes(argv[3], "generated task set")
    manifest = parse_json(manifest_bytes, "manifest")
    config = parse_json(config_bytes, "generator config")
    observed = parse_json(task_set_bytes, "task set")

    if manifest.get("purpose") != "qualified_run" or manifest.get("lane") != "sealed_refresh":
        raise ValueError("reference replay requires a qualified sealed_refresh manifest")
    if len(seed) < 32:
        raise ValueError("seed must be at least 32 bytes")
    if hashlib.sha256(config_bytes).hexdigest() != manifest["generator"]["configuration_sha256"]:
        raise ValueError("generator config bytes do not match manifest digest")
    if config.get("schema") != CONFIG_SCHEMA:
        raise ValueError("generator config schema mismatch")

    expected = independently_generate(manifest, seed, config)
    if expected != observed:
        raise ValueError("independent semantic replay differs from Rust task-set output")

    ids = [task["task_id"] for task in observed["tasks"]]
    families = Counter(task["family_id"] for task in observed["tasks"])
    expected_counts = {
        family["family_id"]: family["planned_task_count"]
        for family in manifest["task_family_mixture"]
    }
    if len(ids) != len(set(ids)):
        raise ValueError("task IDs are not unique")
    if families != Counter(expected_counts):
        raise ValueError("independent family counts disagree with the manifest")
    if len(observed["tasks"]) != manifest["expected_task_count"]:
        raise ValueError("independent task count disagrees with the manifest")
    for task in observed["tasks"]:
        if task["family_id"] in task["task_id"]:
            raise ValueError("opaque task ID exposes its task family")

    print(
        json.dumps(
            {
                "schema": "symthaea.science_bench_refresh_reference_replay.v1",
                "independent_semantic_match": True,
                "generator_config_sha256": hashlib.sha256(config_bytes).hexdigest(),
                "task_set_sha256": hashlib.sha256(task_set_bytes).hexdigest(),
                "task_count": len(observed["tasks"]),
                "family_counts": dict(sorted(families.items())),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"reference replay failed: {exc}", file=sys.stderr)
        raise SystemExit(2)
