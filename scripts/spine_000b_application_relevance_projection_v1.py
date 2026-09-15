#!/usr/bin/env python3
"""SPINE-000B-R2 deterministic application-relevance projection.

This is measurement-only. It compares qualified I_all and I_withoutS at the
Phase-C source/lowering boundary. It does not fabricate counterfactual runtime
application receipts and claims no actual execution or causal effect.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import spine_000b_integration_lowering_domain_v1 as n2

REGISTRY = Path("docs/research/SPINE_000B_PHASE_C_APPLICATION_REGISTRY_V1.json")
REGISTRY_SCHEMA = "symthaea.spine.000b.phase-c-application-registry.v1"
QUALIFIED_N2 = {"QUALIFIED_IDENTITY", "QUALIFIED_APPLICATION_FINITE"}
CHANNELS = (
    (0, "CONFIDENCE_DELTA", "confidence_delta", "confidence_delta_bits"),
    (1, "LR_MODULATION", "lr_modulation", "lr_modulation_bits"),
    (2, "EXPLORATION_DELTA", "exploration_delta", "exploration_delta_bits"),
    (3, "AROUSAL_DELTA", "arousal_delta", "arousal_delta_bits"),
    (4, "VALENCE_DELTA", "valence_delta", "valence_delta_bits"),
)


def load_registry() -> dict[str, object]:
    raw = json.loads(REGISTRY.read_text(encoding="utf-8"))
    if raw.get("schema") != REGISTRY_SCHEMA:
        raise ValueError("Phase-C registry schema drifted")
    operations: set[str] = set()
    for source in list(raw.get("scalar_sources", [])) + list(raw.get("flag_sources", [])):
        for app in source.get("applications", []):
            operation_id = str(app["operation_id"])
            if operation_id in operations:
                raise ValueError(f"duplicate registry operation_id: {operation_id}")
            operations.add(operation_id)
    return raw


def gate_active(expr: str, active_features: set[str]) -> bool:
    if expr == "ALWAYS":
        return True
    if expr.startswith("all(") and expr.endswith(")"):
        items = [item.strip() for item in expr[4:-1].split(",") if item.strip()]
        if not items:
            raise ValueError("empty all() feature gate")
        return all(item in active_features for item in items)
    if any(ch in expr for ch in "() ,"):
        raise ValueError(f"unsupported feature gate expression: {expr}")
    return expr in active_features


def recompute_changed_mask(all_bits: n2.IntegratedBits, without: n2.IntegratedBits) -> int:
    mask = 0
    for bit, _tag, _field, attr in CHANNELS:
        if getattr(all_bits, attr) != getattr(without, attr):
            mask |= 1 << bit
    return mask


def projection_entry(
    app: dict[str, object],
    *,
    source_tag: str,
    source_flag: int,
    source_condition: str,
    argument_bits: int | None = None,
) -> dict[str, object]:
    entry: dict[str, object] = {
        "operation_id": str(app["operation_id"]),
        "source_tag": source_tag,
        "source_flag": source_flag,
        "source_condition": source_condition,
        "feature_gate": str(app["feature_gate"]),
        "applied_argument_kind": str(app["applied_argument_kind"]),
        "applied_argument_derivation": str(app["applied_argument_derivation"]),
    }
    if argument_bits is not None:
        entry["projected_argument_bits_hex"] = f"{argument_bits:08x}"
    return entry


def classify(
    *,
    integrated_all: n2.IntegratedBits,
    integrated_without: n2.IntegratedBits,
    claimed_changed_channel_mask: int,
    claimed_unique_flags: int,
    claimed_integration_changed: bool,
    active_features: set[str],
) -> dict[str, object]:
    registry = load_registry()
    all_q = n2.classify(integrated_all)
    without_q = n2.classify(integrated_without)
    if all_q.primary_class not in QUALIFIED_N2:
        raise ValueError(f"I_all is outside N2 qualified domain: {all_q.primary_class}")
    if without_q.primary_class not in QUALIFIED_N2:
        raise ValueError(f"I_withoutS is outside N2 qualified domain: {without_q.primary_class}")

    changed_mask = recompute_changed_mask(integrated_all, integrated_without)
    unique_flags = integrated_all.flags & ~integrated_without.flags & 0xFFFF_FFFF
    integration_changed = bool(changed_mask or unique_flags)
    if claimed_changed_channel_mask != changed_mask:
        raise ValueError("changed_channel_mask does not match canonical IntegratedBits")
    if claimed_unique_flags != unique_flags:
        raise ValueError("uniquely_contributed_flags does not match canonical IntegratedBits")
    if claimed_integration_changed != integration_changed:
        raise ValueError("integration_changed does not match P1 semantics")

    scalar_by_tag = {str(item["source_tag"]): item for item in registry["scalar_sources"]}
    flag_by_value = {int(item["value"]): item for item in registry["flag_sources"]}

    actual_projection: list[dict[str, object]] = []
    counter_projection: list[dict[str, object]] = []
    scalar_relevant_ops: set[str] = set()
    flag_relevant_ops: set[str] = set()
    prelowering_only_scalar_mask = 0
    unconsumed_flags = 0
    inactive_flags = 0
    inactive_operation_ids: set[str] = set()

    for bit, tag, field, _attr in CHANNELS:
        bit_mask = 1 << bit
        if not (changed_mask & bit_mask):
            continue
        source = scalar_by_tag.get(tag)
        if source is None:
            raise ValueError(f"registry missing scalar source {tag}")
        apps = source.get("applications", [])
        if len(apps) != 1:
            raise ValueError(f"R2 v1 expects one scalar application for {tag}")
        app = apps[0]
        if not gate_active(str(app["feature_gate"]), active_features):
            inactive_operation_ids.add(str(app["operation_id"]))
            prelowering_only_scalar_mask |= bit_mask
            continue

        all_trigger = bool(all_q.triggered_scalar_mask & bit_mask)
        without_trigger = bool(without_q.triggered_scalar_mask & bit_mask)
        all_arg = all_q.lowered_f32_bits[field]
        without_arg = without_q.lowered_f32_bits[field]

        if all_trigger:
            actual_projection.append(
                projection_entry(
                    app,
                    source_tag=tag,
                    source_flag=0,
                    source_condition="SCALAR_NON_IDENTITY",
                    argument_bits=all_arg,
                )
            )
        if without_trigger:
            counter_projection.append(
                projection_entry(
                    app,
                    source_tag=tag,
                    source_flag=0,
                    source_condition="SCALAR_NON_IDENTITY",
                    argument_bits=without_arg,
                )
            )

        if all_trigger != without_trigger or (all_trigger and all_arg != without_arg):
            scalar_relevant_ops.add(str(app["operation_id"]))
        else:
            prelowering_only_scalar_mask |= bit_mask

    remaining_unique = unique_flags
    for value, source in sorted(flag_by_value.items()):
        if unique_flags & value == 0:
            continue
        remaining_unique &= ~value
        apps = list(source.get("applications", []))
        if not apps:
            unconsumed_flags |= value
            continue
        active_apps = [app for app in apps if gate_active(str(app["feature_gate"]), active_features)]
        inactive_apps = [app for app in apps if not gate_active(str(app["feature_gate"]), active_features)]
        inactive_operation_ids.update(str(app["operation_id"]) for app in inactive_apps)
        if not active_apps:
            inactive_flags |= value
            continue
        for app in active_apps:
            condition = str(app["condition"])
            entry = projection_entry(
                app,
                source_tag="FLAG",
                source_flag=value,
                source_condition=condition,
            )
            if condition == "FLAG_SET":
                actual_projection.append(entry)
            elif condition == "FLAG_CLEAR":
                counter_projection.append(entry)
            else:
                raise ValueError(f"unsupported flag source condition: {condition}")
            flag_relevant_ops.add(str(app["operation_id"]))

    if remaining_unique:
        raise ValueError(f"unique flags contain unregistered bits: 0x{remaining_unique:08x}")

    actual_by_op = {str(item["operation_id"]): item for item in actual_projection}
    counter_by_op = {str(item["operation_id"]): item for item in counter_projection}
    actual_only = sorted(set(actual_by_op) - set(counter_by_op))
    counter_only = sorted(set(counter_by_op) - set(actual_by_op))
    shared_operand_changed: list[str] = []
    shared_operand_equal: list[str] = []
    for operation_id in sorted(set(actual_by_op) & set(counter_by_op)):
        actual_argument = actual_by_op[operation_id].get("projected_argument_bits_hex")
        counter_argument = counter_by_op[operation_id].get("projected_argument_bits_hex")
        if actual_argument != counter_argument:
            shared_operand_changed.append(operation_id)
        else:
            shared_operand_equal.append(operation_id)

    scalar_relevant = bool(scalar_relevant_ops)
    flag_relevant = bool(flag_relevant_ops)
    if not integration_changed:
        relevance_kind = "NO_INTEGRATION_INFLUENCE"
    elif scalar_relevant and flag_relevant:
        relevance_kind = "MIXED"
    elif scalar_relevant:
        relevance_kind = "SCALAR"
    elif flag_relevant:
        relevance_kind = "FLAG"
    else:
        relevance_kind = "NONE"

    actual_projection.sort(key=lambda item: str(item["operation_id"]))
    counter_projection.sort(key=lambda item: str(item["operation_id"]))
    return {
        "schema": "symthaea.spine.000b.application-relevance-projection.v1",
        "authority": "measurement-only",
        "causal_load_claimed": False,
        "actual_execution_claimed": False,
        "counterfactual_execution_claimed": False,
        "active_features": sorted(active_features),
        "integration_changed": integration_changed,
        "changed_channel_mask": changed_mask,
        "uniquely_contributed_flags": unique_flags,
        "relevance_kind": relevance_kind,
        "projection_changed": bool(scalar_relevant or flag_relevant),
        "actual_projected_operations": actual_projection,
        "counterfactual_projected_operations": counter_projection,
        "actual_only_projected_operation_ids": actual_only,
        "counterfactual_only_projected_operation_ids": counter_only,
        "shared_projected_operand_changed_operation_ids": shared_operand_changed,
        "shared_projected_operand_equal_operation_ids": shared_operand_equal,
        "relevant_scalar_operation_ids": sorted(scalar_relevant_ops),
        "relevant_flag_operation_ids": sorted(flag_relevant_ops),
        "prelowering_only_scalar_mask": prelowering_only_scalar_mask,
        "unconsumed_unique_flag_mask": unconsumed_flags,
        "feature_gated_inactive_unique_flag_mask": inactive_flags,
        "feature_gated_inactive_operation_ids": sorted(inactive_operation_ids),
        "i_all_n2": asdict(all_q),
        "i_without_subject_n2": asdict(without_q),
    }


def make_integrated(**kwargs: int) -> n2.IntegratedBits:
    return n2.IntegratedBits(**kwargs)


def self_test() -> None:
    zero = make_integrated()
    result = classify(
        integrated_all=zero,
        integrated_without=zero,
        claimed_changed_channel_mask=0,
        claimed_unique_flags=0,
        claimed_integration_changed=False,
        active_features=set(),
    )
    assert result["relevance_kind"] == "NO_INTEGRATION_INFLUENCE"

    result = classify(
        integrated_all=make_integrated(flags=32, n_contributors=1),
        integrated_without=zero,
        claimed_changed_channel_mask=0,
        claimed_unique_flags=32,
        claimed_integration_changed=True,
        active_features=set(),
    )
    assert result["relevance_kind"] == "NONE"
    assert result["unconsumed_unique_flag_mask"] == 32

    result = classify(
        integrated_all=make_integrated(confidence_delta_bits=n2.f64_bits(0.25), n_contributors=1),
        integrated_without=zero,
        claimed_changed_channel_mask=1,
        claimed_unique_flags=0,
        claimed_integration_changed=True,
        active_features=set(),
    )
    assert result["relevance_kind"] == "SCALAR"
    assert result["actual_only_projected_operation_ids"] == ["feedback.adjust_confidence"]

    tiny = 1.0e-50
    assert tiny != 0.0 and n2.rust_f64_to_f32_bits(tiny) == n2.f32_bits(0.0)
    result = classify(
        integrated_all=make_integrated(confidence_delta_bits=n2.f64_bits(tiny), n_contributors=1),
        integrated_without=zero,
        claimed_changed_channel_mask=1,
        claimed_unique_flags=0,
        claimed_integration_changed=True,
        active_features=set(),
    )
    assert result["relevance_kind"] == "SCALAR"
    assert result["actual_projected_operations"][0]["projected_argument_bits_hex"] == "00000000"

    first = 1.0 + 2.0**-30
    second = 1.0 + 2.0**-29
    result = classify(
        integrated_all=make_integrated(confidence_delta_bits=n2.f64_bits(first), n_contributors=2),
        integrated_without=make_integrated(confidence_delta_bits=n2.f64_bits(second), n_contributors=1),
        claimed_changed_channel_mask=1,
        claimed_unique_flags=0,
        claimed_integration_changed=True,
        active_features=set(),
    )
    assert result["relevance_kind"] == "NONE"
    assert result["prelowering_only_scalar_mask"] == 1

    result = classify(
        integrated_all=make_integrated(confidence_delta_bits=n2.f64_bits(-0.0), n_contributors=2),
        integrated_without=make_integrated(confidence_delta_bits=n2.f64_bits(0.0), n_contributors=1),
        claimed_changed_channel_mask=1,
        claimed_unique_flags=0,
        claimed_integration_changed=True,
        active_features=set(),
    )
    assert result["relevance_kind"] == "NONE"
    assert result["prelowering_only_scalar_mask"] == 1

    result = classify(
        integrated_all=make_integrated(lr_modulation_bits=n2.f64_bits(1.2), n_contributors=2),
        integrated_without=make_integrated(lr_modulation_bits=n2.f64_bits(1.1), n_contributors=1),
        claimed_changed_channel_mask=2,
        claimed_unique_flags=0,
        claimed_integration_changed=True,
        active_features=set(),
    )
    assert result["relevance_kind"] == "SCALAR"
    assert result["shared_projected_operand_changed_operation_ids"] == ["feedback.scale_lr"]

    result = classify(
        integrated_all=make_integrated(flags=16, n_contributors=1),
        integrated_without=zero,
        claimed_changed_channel_mask=0,
        claimed_unique_flags=16,
        claimed_integration_changed=True,
        active_features=set(),
    )
    assert result["relevance_kind"] == "FLAG"
    assert result["actual_only_projected_operation_ids"] == ["feedback.scale_lr.request_rest"]

    result = classify(
        integrated_all=make_integrated(flags=256, n_contributors=1),
        integrated_without=zero,
        claimed_changed_channel_mask=0,
        claimed_unique_flags=256,
        claimed_integration_changed=True,
        active_features={"vision-manifold"},
    )
    assert result["relevance_kind"] == "FLAG"
    assert "quality.set_request_geodesic" in result["actual_only_projected_operation_ids"]
    assert "quality.clear_request_geodesic" in result["counterfactual_only_projected_operation_ids"]

    result = classify(
        integrated_all=make_integrated(flags=256, n_contributors=1),
        integrated_without=zero,
        claimed_changed_channel_mask=0,
        claimed_unique_flags=256,
        claimed_integration_changed=True,
        active_features=set(),
    )
    assert result["relevance_kind"] == "NONE"
    assert result["feature_gated_inactive_unique_flag_mask"] == 256

    result = classify(
        integrated_all=make_integrated(
            exploration_delta_bits=n2.f64_bits(0.2), flags=1, n_contributors=2
        ),
        integrated_without=zero,
        claimed_changed_channel_mask=4,
        claimed_unique_flags=1,
        claimed_integration_changed=True,
        active_features=set(),
    )
    assert result["relevance_kind"] == "MIXED"

    f32_max = n2.f32_from_bits(0x7F7F_FFFF)
    too_large = f32_max + 2.0**103
    try:
        classify(
            integrated_all=make_integrated(
                confidence_delta_bits=n2.f64_bits(too_large), n_contributors=1
            ),
            integrated_without=zero,
            claimed_changed_channel_mask=1,
            claimed_unique_flags=0,
            claimed_integration_changed=True,
            active_features=set(),
        )
    except ValueError as exc:
        assert "N2 qualified domain" in str(exc)
    else:
        raise AssertionError("R2 must reject N2-invalid I_all")

    try:
        classify(
            integrated_all=make_integrated(
                confidence_delta_bits=n2.f64_bits(0.25), n_contributors=1
            ),
            integrated_without=zero,
            claimed_changed_channel_mask=0,
            claimed_unique_flags=0,
            claimed_integration_changed=False,
            active_features=set(),
        )
    except ValueError as exc:
        assert "changed_channel_mask" in str(exc)
    else:
        raise AssertionError("R2 must reject inconsistent P1 receipt summary")


def integrated_from_json(raw: dict[str, object]) -> n2.IntegratedBits:
    return n2.IntegratedBits(**{key: int(value) for key, value in raw.items()})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--input", type=Path)
    args = parser.parse_args()
    if args.self_test:
        self_test()
        print("SPINE-000B-R2 application relevance projection self-test: PASS")
        print("actual_execution_claimed=false")
        print("counterfactual_execution_claimed=false")
        print("causal_load_claimed=false")
        if not args.input:
            return 0
    if args.input:
        raw = json.loads(args.input.read_text(encoding="utf-8"))
        result = classify(
            integrated_all=integrated_from_json(raw["integrated_all"]),
            integrated_without=integrated_from_json(raw["integrated_without_subject"]),
            claimed_changed_channel_mask=int(raw["changed_channel_mask"]),
            claimed_unique_flags=int(raw["uniquely_contributed_flags"]),
            claimed_integration_changed=bool(raw["integration_changed"]),
            active_features={str(item) for item in raw.get("active_features", [])},
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    parser.error("request --self-test and/or --input")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
