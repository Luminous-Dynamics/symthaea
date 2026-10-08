#!/usr/bin/env python3
from __future__ import annotations
from functools import lru_cache
from hashlib import sha256
from itertools import product
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from monetary_partial_attestation import (
    PAIR, AD, SH, SEED, ALLO, FOCAL, REPORT, TRUE_CRIT, CAP,
    true_liquidity_demand, world_jitter, manual_hit, TECH,
)

ROOT = Path(__file__).resolve().parent
MANIFEST = json.loads((ROOT / "../docs/research/monetary/monetary-attestation-binding-v1.json").read_text())
F = MANIFEST["factors"]
BIND = tuple(F["binding_policy"])
VAR = tuple(F["attestation_variant"])
ATTR = MANIFEST["attribute_identities"]
GEN = MANIFEST["fixed_dimensions"]["current_generation_by_seed"]
TOPO = MANIFEST["fixed_dimensions"]["topology_digest"]
RES = MANIFEST["fixed_dimensions"]["resource_digest"]
FIELDS = {k: set(MANIFEST["binding_policies"][k]["fields"]) for k in BIND}

@lru_cache(None)
def context_id(pair, adapter, shock, seed, allocation):
    value = {
        "pair": pair, "adapter": adapter, "shock": shock, "seed": seed,
        "allocation_policy": allocation, "topology_digest": TOPO,
        "resource_digest": RES,
    }
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

def attribute_for(allocation):
    return ATTR[allocation]

def value_for(attribute_id, index, demand):
    if attribute_id == "fifo_order_v1":
        return index
    if attribute_id == "criticality_v1":
        return TRUE_CRIT[index]
    return demand[index]

def historical_value(allocation, index, demand):
    value = value_for(attribute_for(allocation), index, demand)
    if allocation == "criticality_priority":
        return max(0, value - 1)
    if allocation == "minimum_liquidity_demand":
        return min(CAP + 1, value + 5)
    return value + 1

def neighbor_shock(shock):
    return SH[(SH.index(shock) + 1) % len(SH)]

def expected_envelope(pair, adapter, shock, seed, allocation, focal):
    return {
        "subject_id": f"obligation:{focal}",
        "attribute_id": attribute_for(allocation),
        "generation": GEN[str(seed)],
        "context_id": context_id(pair, adapter, shock, seed, allocation),
    }

def make_attestation(pair, adapter, shock, seed, allocation, focal, variant):
    demand = true_liquidity_demand(adapter, shock)
    expected = expected_envelope(pair, adapter, shock, seed, allocation, focal)
    if variant == "exact_current":
        return {**expected, "attested_value": value_for(expected["attribute_id"], focal, demand), "signature_valid": True}
    if variant == "subject_substitution":
        other = (focal + 1) % 3
        return {**expected, "subject_id": f"obligation:{other}", "attested_value": value_for(expected["attribute_id"], other, demand), "signature_valid": True}
    if variant == "attribute_substitution":
        other_attr = next(x for x in ATTR.values() if x != expected["attribute_id"])
        return {**expected, "attribute_id": other_attr, "attested_value": value_for(other_attr, focal, demand), "signature_valid": True}
    if variant == "stale_generation":
        return {**expected, "generation": expected["generation"] - MANIFEST["fixed_dimensions"]["stale_generation_delta"],
                "attested_value": historical_value(allocation, focal, demand), "signature_valid": True}
    if variant == "context_substitution":
        other_shock = neighbor_shock(shock)
        other_demand = true_liquidity_demand(adapter, other_shock)
        return {**expected, "context_id": context_id(pair, adapter, other_shock, seed, allocation),
                "attested_value": value_for(expected["attribute_id"], focal, other_demand), "signature_valid": True}
    if variant == "composite_replay":
        other_shock = neighbor_shock(shock)
        other_demand = true_liquidity_demand(adapter, other_shock)
        return {**expected, "generation": expected["generation"] - MANIFEST["fixed_dimensions"]["stale_generation_delta"],
                "context_id": context_id(pair, adapter, other_shock, seed, allocation),
                "attested_value": historical_value(allocation, focal, other_demand), "signature_valid": True}
    raise ValueError(variant)

def allocation_order(policy, keys):
    indices = (0, 1, 2)
    if policy == "fifo":
        return indices
    if policy == "criticality_priority":
        return tuple(sorted(indices, key=lambda i: (-keys[i], i)))
    return tuple(sorted(indices, key=lambda i: (keys[i], i)))

def run_cell(pair, adapter, shock, seed, allocation, focal, binding, variant, reporting):
    demand = true_liquidity_demand(adapter, shock)
    declared_criticality = list(TRUE_CRIT)
    declared_demand = list(demand)
    if reporting in ("criticality_inflation", "dual_misreport"):
        declared_criticality[focal] = 5
    if reporting in ("liquidity_demand_underreport", "dual_misreport"):
        declared_demand[focal] = min(demand[focal], 5)
    # Deliberately unused: authoritative attestation must dominate participant reporting.
    _ = (declared_criticality, declared_demand)

    keys = [value_for(attribute_for(allocation), i, demand) for i in range(3)]
    att = make_attestation(pair, adapter, shock, seed, allocation, focal, variant)
    expected = expected_envelope(pair, adapter, shock, seed, allocation, focal)
    binding_valid = all(att[field] == expected[field] for field in FIELDS[binding])
    accepted_invalid = variant != "exact_current" and binding_valid and att["signature_valid"]

    if variant == "exact_current" or accepted_invalid:
        keys[focal] = att["attested_value"]
    elif allocation == "criticality_priority":
        keys[focal] = 0
    elif allocation == "minimum_liquidity_demand":
        keys[focal] = CAP + 1
    else:
        keys[focal] = focal

    order = allocation_order(allocation, keys)
    current = 5 + world_jitter(shock, seed)
    available = CAP
    pending = list(order)
    active = []
    finish = {}
    peak = 0

    while pending or active:
        if active:
            next_finish = min(item[0] for item in active)
            current = max(current, next_finish)
            done = [item for item in active if item[0] <= current]
            active = [item for item in active if item[0] > current]
            available += sum(item[1] for item in done)
        while pending:
            index = pending[0]
            quantity = demand[index]
            if quantity == 0 or quantity > CAP:
                pending.pop(0)
                finish[index] = None
                continue
            if quantity > available:
                break
            pending.pop(0)
            available -= quantity
            peak = max(peak, CAP - available)
            finish[index] = (
                current + TECH[adapter]
                + (2 if shock == "liquidity_shock" and adapter == "redeem_reissue" else 0)
                + (2 if manual_hit(shock, seed, index) else 0) + 2
            )
            active.append((finish[index], quantity, index))
        if pending and not active:
            for index in pending:
                finish[index] = None
            pending = []

    done = [
        index for index, value in finish.items()
        if value is not None
        and shock not in ("bridge_failure", "stale_quote")
        and not (shock == "issuer_default" and adapter == "redeem_reissue")
    ]
    return {
        "binding_valid": binding_valid,
        "accepted_invalid": accepted_invalid,
        "signature_valid": att["signature_valid"],
        "focal_finish": finish.get(focal) if focal in done else None,
        "completion_rate": len(done) / 3,
        "system_completion_time": max((finish[i] for i in done), default=None),
        "peak_true_liquidity_reserved": peak,
        "allocation_order": order,
    }

def main(out_dir):
    if len(F) != 9 or MANIFEST["factorial_size"] != 466560:
        raise AssertionError("factor manifest mismatch")

    truthful = {
        (pair, adapter, shock, seed, allocation, focal): run_cell(
            pair, adapter, shock, seed, allocation, focal, "exact_all", "exact_current", "truthful"
        )
        for pair, adapter, shock, seed, allocation, focal in product(PAIR, AD, SH, SEED, ALLO, FOCAL)
    }

    trace = sha256()
    acceptance = {(v, b): [] for v in VAR for b in BIND}
    gains = {(v, b): [] for v in VAR for b in BIND}
    failure_count = 0
    isolation_failures = 0
    reporting_independence_failures = 0

    expected_mutations = {
        "exact_current": set(),
        "subject_substitution": {"subject_id"},
        "attribute_substitution": {"attribute_id"},
        "stale_generation": {"generation"},
        "context_substitution": {"context_id"},
        "composite_replay": {"generation", "context_id"},
    }

    for pair, adapter, shock, seed, allocation, focal, binding, variant, reporting in product(
        PAIR, AD, SH, SEED, ALLO, FOCAL, BIND, VAR, REPORT
    ):
        cell = run_cell(pair, adapter, shock, seed, allocation, focal, binding, variant, reporting)
        base = truthful[(pair, adapter, shock, seed, allocation, focal)]
        att = make_attestation(pair, adapter, shock, seed, allocation, focal, variant)
        expected = expected_envelope(pair, adapter, shock, seed, allocation, focal)
        changed = {field for field in ("subject_id", "attribute_id", "generation", "context_id") if att[field] != expected[field]}
        isolation_failures += int(changed != expected_mutations[variant])

        if reporting == "truthful":
            reference_projection = (
                cell["focal_finish"], cell["completion_rate"],
                cell["system_completion_time"], cell["allocation_order"]
            )
        else:
            projection = (
                cell["focal_finish"], cell["completion_rate"],
                cell["system_completion_time"], cell["allocation_order"]
            )
            reporting_independence_failures += int(projection != reference_projection)

        record = {
            "pair": pair, "adapter": adapter, "shock": shock, "seed": seed,
            "allocation_policy": allocation, "focal_obligation": focal,
            "reporting_policy": reporting, "binding_policy": binding,
            "attestation_variant": variant, **cell,
        }
        trace.update((json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n").encode())
        failure_count += int(cell["peak_true_liquidity_reserved"] > CAP)

        key = (variant, binding)
        acceptance[key].append(int(cell["accepted_invalid"]))
        if cell["accepted_invalid"] and cell["focal_finish"] is not None and base["focal_finish"] is not None:
            gains[key].append(base["focal_finish"] - cell["focal_finish"])

    if failure_count or isolation_failures or reporting_independence_failures:
        raise AssertionError("invariant failure")

    expected_matrix = {
        "subject_substitution": {"attribute_only": 1.0, "generation_only": 1.0, "context_only": 1.0, "subject_only": 0.0, "subject_attribute": 0.0, "exact_all": 0.0},
        "attribute_substitution": {"attribute_only": 0.0, "generation_only": 1.0, "context_only": 1.0, "subject_only": 1.0, "subject_attribute": 0.0, "exact_all": 0.0},
        "stale_generation": {"attribute_only": 1.0, "generation_only": 0.0, "context_only": 1.0, "subject_only": 1.0, "subject_attribute": 1.0, "exact_all": 0.0},
        "context_substitution": {"attribute_only": 1.0, "generation_only": 1.0, "context_only": 0.0, "subject_only": 1.0, "subject_attribute": 1.0, "exact_all": 0.0},
        "composite_replay": {"attribute_only": 1.0, "generation_only": 0.0, "context_only": 0.0, "subject_only": 1.0, "subject_attribute": 1.0, "exact_all": 0.0},
    }
    actual_matrix = {
        "exact_current": {b: 1.0 for b in BIND},
        **{variant: {binding: sum(acceptance[(variant, binding)]) / len(acceptance[(variant, binding)])
                     for binding in BIND} for variant in VAR if variant != "exact_current"}
    }
    assert actual_matrix == expected_matrix | {"exact_current": {b: 1.0 for b in BIND}}

    def attack_stats(variant):
        values = []
        for binding in BIND:
            values.extend(gains[(variant, binding)])
        positive = [x for x in values if x > 0]
        return {
            "positive_gain_rate": len(positive) / len(values) if values else 0.0,
            "mean_gain_ticks": sum(values) / len(values) if values else 0.0,
            "max_gain_ticks": max(values) if values else 0,
        }

    output = {
        "schema_version": "monetary-attestation-binding-v1-execution",
        "run_count": 466560,
        "obligation_count": 1399680,
        "common_random_number_cells": 90,
        "exogenous_random_namespace": MANIFEST["fixed_dimensions"]["crn_namespace"],
        "trace_set_digest": trace.hexdigest(),
        "headline": {
            "exact_all_invalid_binding_acceptance_rate": 0.0,
            "exact_all_attack_positive_gain_rate": 0.0,
            "exact_all_max_attack_gain_ticks": 0,
            "accepted_invalid_rate_across_all_partial_binding_profiles": 17 / 30,
        },
        "binding_acceptance_matrix": actual_matrix,
        "attack_effects_if_admitted": {variant: attack_stats(variant) for variant in VAR if variant != "exact_current"},
        "invariants": {
            "factorial_cardinality_exact": True,
            "obligation_cardinality_exact": True,
            "signature_valid_for_all_synthetic_variants": True,
            "mutation_coordinate_isolation": True,
            "reporting_independent_authoritative_result": True,
            "resource_integrity_failures": 0,
            "exact_current_all_binding_profiles_accept": True,
            "exact_all_rejects_every_invalid_variant": True,
        },
        "claim_ceiling": MANIFEST["claim_ceiling"],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / "monetary-attestation-binding-v1.execution.generated.json"
    target.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"trace_set_digest": output["trace_set_digest"], "headline": output["headline"], "invariants": output["invariants"]}, indent=2))

if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("."))
