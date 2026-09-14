#!/usr/bin/env python3
"""Independent LQCD-021E configuration/measurement provenance oracle."""

import hashlib
import json
import math
import struct
from dataclasses import dataclass, replace

CONFIG_TAG = b"symthaea.lqcd.config-id.v1\x00"
CONFIG_SET_TAG = b"symthaea.lqcd.config-set.v1\x00"
MEASUREMENT_KEY_TAG = b"symthaea.lqcd.measurement-key.v1\x00"
MEASUREMENT_VALUE_TAG = b"symthaea.lqcd.measurement-value.v1\x00"
MEASUREMENT_RECEIPT_TAG = b"symthaea.lqcd.measurement-receipt.v1\x00"
MEASUREMENT_SET_TAG = b"symthaea.lqcd.measurement-set.v1\x00"

PHASE = {"pilot": 2, "final": 3}

def u16(v):
    if not 0 <= v < 1 << 16:
        raise ValueError("u16")
    return v.to_bytes(2, "big")

def u32(v):
    if not 0 <= v < 1 << 32:
        raise ValueError("u32")
    return v.to_bytes(4, "big")

def u64(v):
    if not 0 <= v < 1 << 64:
        raise ValueError("u64")
    return v.to_bytes(8, "big")

def lp(value):
    if not isinstance(value, str) or not value:
        raise ValueError("non-empty string required")
    raw = value.encode("utf-8")
    return u32(len(raw)) + raw

def digest(label, data):
    return hashlib.sha256(label + data).digest()

@dataclass(frozen=True)
class Config:
    campaign_subject: bytes
    phase: int
    chain_id: str
    retained_ordinal: int
    transition_ordinal: int
    generation_subject: bytes
    gauge_field_digest: bytes
    checkpoint_commitment: bytes

def config_preimage(c):
    if len(c.campaign_subject) != 32 or len(c.generation_subject) != 32:
        raise ValueError("subject digest")
    if len(c.gauge_field_digest) != 32 or len(c.checkpoint_commitment) != 32:
        raise ValueError("content/lineage digest")
    if c.phase not in PHASE.values():
        raise ValueError("phase")
    return (
        c.campaign_subject
        + bytes([c.phase])
        + lp(c.chain_id)
        + u64(c.retained_ordinal)
        + u64(c.transition_ordinal)
        + c.generation_subject
        + c.gauge_field_digest
        + c.checkpoint_commitment
    )

def config_id(c):
    return digest(CONFIG_TAG, config_preimage(c))

def config_set_commitment(config_ids):
    if len(set(config_ids)) != len(config_ids):
        raise ValueError("duplicate ConfigId")
    ordered = sorted(config_ids)
    return digest(CONFIG_SET_TAG, u32(len(ordered)) + b"".join(ordered))

@dataclass(frozen=True)
class MeasurementKey:
    config_id: bytes
    measurement_subject: bytes
    ape_id: str
    ape_revision: bytes
    operator_id: str
    operator_revision: bytes
    displacement: tuple
    temporal_extent: int

def key_preimage(k):
    if len(k.config_id) != 32 or len(k.measurement_subject) != 32:
        raise ValueError("key subject digest")
    if len(k.ape_revision) != 32 or len(k.operator_revision) != 32:
        raise ValueError("revision digest")
    dx, dy, dz = k.displacement
    for component in (dx, dy, dz):
        if not -(1 << 15) <= component < (1 << 15):
            raise ValueError("displacement")
    if not 1 <= k.temporal_extent < (1 << 16):
        raise ValueError("T")
    return (
        k.config_id
        + k.measurement_subject
        + lp(k.ape_id)
        + k.ape_revision
        + lp(k.operator_id)
        + k.operator_revision
        + int(dx).to_bytes(2, "big", signed=True)
        + int(dy).to_bytes(2, "big", signed=True)
        + int(dz).to_bytes(2, "big", signed=True)
        + u16(k.temporal_extent)
    )

def key_id(k):
    return digest(MEASUREMENT_KEY_TAG, key_preimage(k))

@dataclass(frozen=True)
class Receipt:
    key: MeasurementKey
    raw_re: float
    raw_im: float
    numerical_profile_id: str
    executable_subject: bytes
    environment_profile_id: str
    shard_id: str
    attempt_id: str

def f64be(value):
    if not math.isfinite(value):
        raise ValueError("non-finite")
    return struct.pack(">d", value)

def semantic_value_preimage(r):
    if len(r.executable_subject) != 32:
        raise ValueError("executable subject")
    return (
        key_id(r.key)
        + f64be(r.raw_re)
        + f64be(r.raw_im)
        + lp(r.numerical_profile_id)
        + r.executable_subject
        + lp(r.environment_profile_id)
    )

def semantic_value_digest(r):
    return digest(MEASUREMENT_VALUE_TAG, semantic_value_preimage(r))

def receipt_digest(r):
    return digest(
        MEASUREMENT_RECEIPT_TAG,
        semantic_value_digest(r) + lp(r.shard_id) + lp(r.attempt_id),
    )

def canonical_measurement_set(expected_key_ids, receipts):
    expected = set(expected_key_ids)
    if len(expected) != len(expected_key_ids):
        raise ValueError("duplicate expected key")
    by_key = {}
    receipt_ids = []
    for receipt in receipts:
        kid = key_id(receipt.key)
        if kid not in expected:
            raise ValueError("unknown measurement key")
        value_digest = semantic_value_digest(receipt)
        receipt_ids.append(receipt_digest(receipt))
        if kid in by_key and by_key[kid] != value_digest:
            raise ValueError("conflicting duplicate measurement")
        by_key[kid] = value_digest
    if set(by_key) != expected:
        raise ValueError("incomplete measurement set")
    ordered = sorted(by_key.items())
    commitment = digest(
        MEASUREMENT_SET_TAG,
        u32(len(ordered))
        + b"".join(kid + value_digest for kid, value_digest in ordered),
    )
    return commitment, sorted(receipt_ids)

def vectors_2528():
    out = []
    for n in range(1, 8):
        out.append((n, 0, 0))
    for n in range(1, 8):
        out.append((n, n, 0))
    for n in range(1, 8):
        out.append((n, n, n))
    for n in range(1, 4):
        out.append((2*n, n, 0))
    assert len(out) == 24
    return out

def fixture_configs(count, phase):
    campaign = hashlib.sha256(b"synthetic-campaign-" + bytes([phase])).digest()
    generation = hashlib.sha256(b"generation-subject-v1").digest()
    configs = []
    for i in range(count):
        configs.append(
            Config(
                campaign_subject=campaign,
                phase=phase,
                chain_id=f"chain-{i % 4:02d}",
                retained_ordinal=i // 4,
                transition_ordinal=8000 + 100 * (i // 4),
                generation_subject=generation,
                gauge_field_digest=hashlib.sha256(b"field-" + u32(i) + bytes([phase])).digest(),
                checkpoint_commitment=hashlib.sha256(b"checkpoint-" + u32(i) + bytes([phase])).digest(),
            )
        )
    return configs

def fixture_keys(configs):
    measurement_subject = hashlib.sha256(b"measurement-subject-v1").digest()
    ape_revision = hashlib.sha256(b"ape-revision-v1").digest()
    operator_revision = hashlib.sha256(b"operator-revision-v1").digest()
    keys = []
    for c in configs:
        cid = config_id(c)
        for displacement in vectors_2528():
            for t in range(1, 9):
                keys.append(
                    MeasurementKey(
                        config_id=cid,
                        measurement_subject=measurement_subject,
                        ape_id="spatial_ape_ehk_polar_v1:alpha=0.7:n=19",
                        ape_revision=ape_revision,
                        operator_id="generalized_bresenham_cubic_ape_spatial_unsmeared_temporal_wilson_v1",
                        operator_revision=operator_revision,
                        displacement=displacement,
                        temporal_extent=t,
                    )
                )
    return keys

def fixture_receipts(keys):
    executable = hashlib.sha256(b"measurement-executable-v1").digest()
    receipts = []
    for i, key in enumerate(keys):
        config_term = (i // 192) / 1024.0
        local = i % 192
        vector_term = (local // 8) / 4096.0
        t_term = (local % 8 + 1) / 65536.0
        raw_re = 0.25 + config_term + vector_term + t_term
        raw_im = ((local % 5) - 2) / 1048576.0
        receipts.append(
            Receipt(
                key=key,
                raw_re=raw_re,
                raw_im=raw_im,
                numerical_profile_id="lqcd_numerical_profile_candidate_v1",
                executable_subject=executable,
                environment_profile_id="symthaea-rust196-linux-x86_64-candidate-v1",
                shard_id=f"shard-{i % 4}",
                attempt_id=f"attempt-{i}",
            )
        )
    return receipts

def must_fail(label, fn):
    try:
        fn()
    except ValueError:
        return "rejected"
    raise AssertionError(label)

def main():
    final_configs = fixture_configs(16, PHASE["final"])
    pilot_configs = fixture_configs(16, PHASE["pilot"])
    final_ids = [config_id(c) for c in final_configs]
    pilot_ids = [config_id(c) for c in pilot_configs]
    assert set(final_ids).isdisjoint(pilot_ids)

    base = final_configs[0]
    config_mutations = {
        "campaign_subject": replace(base, campaign_subject=bytes([base.campaign_subject[0] ^ 1]) + base.campaign_subject[1:]),
        "phase": replace(base, phase=PHASE["pilot"]),
        "chain_id": replace(base, chain_id=base.chain_id + "-mut"),
        "retained_ordinal": replace(base, retained_ordinal=base.retained_ordinal + 1),
        "transition_ordinal": replace(base, transition_ordinal=base.transition_ordinal + 1),
        "generation_subject": replace(base, generation_subject=bytes([base.generation_subject[0] ^ 1]) + base.generation_subject[1:]),
        "gauge_field_digest": replace(base, gauge_field_digest=bytes([base.gauge_field_digest[0] ^ 1]) + base.gauge_field_digest[1:]),
        "checkpoint_commitment": replace(base, checkpoint_commitment=bytes([base.checkpoint_commitment[0] ^ 1]) + base.checkpoint_commitment[1:]),
    }
    base_id = config_id(base)
    for name, mutated in config_mutations.items():
        assert config_id(mutated) != base_id, name

    final_config_set = config_set_commitment(final_ids)
    pilot_config_set = config_set_commitment(pilot_ids)
    assert final_config_set != pilot_config_set

    keys = fixture_keys(final_configs)
    expected_ids = [key_id(k) for k in keys]
    assert len(keys) == 16 * 24 * 8
    assert 4000 * 24 * 8 == 768000
    receipts = fixture_receipts(keys)

    monolithic, _ = canonical_measurement_set(expected_ids, receipts)

    shards = [receipts[i::4] for i in range(4)]
    reordered = list(reversed(shards[2])) + shards[0] + list(reversed(shards[3])) + shards[1]
    sharded, _ = canonical_measurement_set(expected_ids, reordered)
    assert sharded == monolithic

    retry = replace(
        receipts[17],
        shard_id="retry-shard",
        attempt_id="retry-attempt-17",
    )
    assert receipt_digest(retry) != receipt_digest(receipts[17])
    assert semantic_value_digest(retry) == semantic_value_digest(receipts[17])
    retry_commitment, _ = canonical_measurement_set(expected_ids, receipts + [retry])
    assert retry_commitment == monolithic

    conflicting_retry = replace(retry, raw_re=retry.raw_re + 1.0 / 1048576.0)
    conflict = must_fail(
        "conflicting duplicate",
        lambda: canonical_measurement_set(expected_ids, receipts + [conflicting_retry]),
    )
    missing = must_fail(
        "missing key",
        lambda: canonical_measurement_set(expected_ids, receipts[:-1]),
    )
    wrong_operator_key = replace(
        receipts[0].key,
        operator_revision=hashlib.sha256(b"wrong-operator").digest(),
    )
    wrong_operator = must_fail(
        "wrong operator",
        lambda: canonical_measurement_set(
            expected_ids,
            [replace(receipts[0], key=wrong_operator_key)] + receipts[1:],
        ),
    )
    pilot_key = replace(receipts[0].key, config_id=pilot_ids[0])
    pilot_in_final = must_fail(
        "pilot in final",
        lambda: canonical_measurement_set(
            expected_ids,
            [replace(receipts[0], key=pilot_key)] + receipts[1:],
        ),
    )
    extra_key = replace(receipts[0].key, temporal_extent=9)
    unknown_extra = must_fail(
        "unknown extra",
        lambda: canonical_measurement_set(
            expected_ids,
            receipts + [replace(receipts[0], key=extra_key, attempt_id="extra")],
        ),
    )
    duplicate_config_rejected = must_fail(
        "duplicate config id",
        lambda: config_set_commitment(final_ids + [final_ids[0]]),
    )

    result = {
        "oracle_id": "lqcd_config_measurement_provenance_oracle_v1",
        "config_mutation_fields": len(config_mutations),
        "pilot_final_config_sets_disjoint": True,
        "final_fixture_config_count": len(final_configs),
        "measurement_vectors_per_config": 24,
        "temporal_extents_per_vector": 8,
        "measurement_keys_per_config": 192,
        "synthetic_expected_measurement_keys": len(expected_ids),
        "exact_final_design_measurement_keys_for_4000_configs": 768000,
        "final_config_set_commitment": final_config_set.hex(),
        "pilot_config_set_commitment": pilot_config_set.hex(),
        "measurement_set_commitment": monolithic.hex(),
        "shard_partition_order_invariant": sharded == monolithic,
        "retry_receipt_digest_changes": receipt_digest(retry) != receipt_digest(receipts[17]),
        "retry_semantic_value_stable": semantic_value_digest(retry) == semantic_value_digest(receipts[17]),
        "retry_measurement_set_commitment_stable": retry_commitment == monolithic,
        "negative_controls": {
            "conflicting_duplicate": conflict,
            "missing_key": missing,
            "wrong_operator_revision": wrong_operator,
            "pilot_config_in_final": pilot_in_final,
            "unknown_extra_key": unknown_extra,
            "duplicate_config_id": duplicate_config_rejected,
        },
        "scientific_boundary": "configuration_measurement_provenance_only_no_equilibrium_no_physics",
    }
    canonical = json.dumps(result, sort_keys=True, separators=(",", ":")).encode()
    print("ok")
    print("result_sha256=" + hashlib.sha256(canonical).hexdigest())
    print(canonical.decode())

if __name__ == "__main__":
    main()
