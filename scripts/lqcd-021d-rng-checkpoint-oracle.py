#!/usr/bin/env python3
"""Independent LQCD-021D stream-namespace/checkpoint commitment oracle.

This subject freezes phase/purpose stream-ID packing and checkpoint-lineage
commitment semantics. It does not implement ChaCha8 or claim RNG-byte restart
parity; the production Rust tranche must separately prove that seed + stream ID
+ word position reconstruct the exact subsequent RNG bytes.
"""
import hashlib
import json

ORACLE_ID = "lqcd_rng_checkpoint_lineage_v1"
CHECKPOINT_TAG = b"symthaea.lqcd.checkpoint.v1\0"
SEED_TAG = b"symthaea.lqcd.seed.v1\0"

PHASE = {
    "throughput": 1,
    "pilot": 2,
    "final": 3,
    "qualification": 4,
}
PURPOSE = {
    "gauge_transition": 1,
    "diagnostic": 2,
    "analysis": 3,
    "bootstrap": 4,
}


def sha256(data):
    return hashlib.sha256(data).digest()


def hx(data):
    return data.hex()


def pack_stream_id(phase, purpose, ensemble_slot, replica, rank):
    if phase not in PHASE.values() or not 0 < phase < 16:
        raise ValueError("invalid phase")
    if purpose not in PURPOSE.values() or not 0 < purpose < 16:
        raise ValueError("invalid purpose")
    if not 0 <= ensemble_slot < (1 << 24):
        raise ValueError("ensemble_slot out of range")
    if not 0 <= replica < (1 << 16):
        raise ValueError("replica out of range")
    if not 0 <= rank < (1 << 16):
        raise ValueError("rank out of range")
    domain = (phase << 4) | purpose
    return (domain << 56) | (ensemble_slot << 32) | (replica << 16) | rank


def seed_commitment(seed):
    if not isinstance(seed, bytes) or len(seed) != 32:
        raise ValueError("seed must be 32 bytes")
    return sha256(SEED_TAG + seed)


def fixed_digest(label):
    return sha256(label.encode("utf-8"))


def u24(value):
    return value.to_bytes(3, "big")


def checkpoint_bytes(record):
    required = {
        "phase",
        "purpose",
        "ensemble_slot",
        "replica",
        "rank",
        "update_ordinal",
        "word_pos",
        "chain_id",
        "campaign_subject",
        "seed_commitment",
        "gauge_field",
        "sampler_subject",
        "numerical_profile",
        "environment",
        "previous_checkpoint",
    }
    if set(record) != required:
        raise ValueError("checkpoint field set mismatch")

    stream_id = pack_stream_id(
        record["phase"],
        record["purpose"],
        record["ensemble_slot"],
        record["replica"],
        record["rank"],
    )
    if not 0 <= record["update_ordinal"] < (1 << 64):
        raise ValueError("update ordinal out of range")
    if not 0 <= record["word_pos"] < (1 << 128):
        raise ValueError("word position out of range")

    out = bytearray(CHECKPOINT_TAG)
    out += bytes([record["phase"], record["purpose"]])
    out += u24(record["ensemble_slot"])
    out += record["replica"].to_bytes(2, "big")
    out += record["rank"].to_bytes(2, "big")
    out += stream_id.to_bytes(8, "big")
    out += record["update_ordinal"].to_bytes(8, "big")
    out += record["word_pos"].to_bytes(16, "big")
    for name in (
        "chain_id",
        "campaign_subject",
        "seed_commitment",
        "gauge_field",
        "sampler_subject",
        "numerical_profile",
        "environment",
        "previous_checkpoint",
    ):
        value = record[name]
        if not isinstance(value, bytes) or len(value) != 32:
            raise ValueError(name + " must be 32 bytes")
        out += value
    return bytes(out)


def checkpoint_digest(record):
    return sha256(checkpoint_bytes(record))


def make_record(**overrides):
    base = {
        "phase": PHASE["pilot"],
        "purpose": PURPOSE["gauge_transition"],
        "ensemble_slot": 0x021D,
        "replica": 7,
        "rank": 0,
        "update_ordinal": 120,
        "word_pos": 987654321,
        "chain_id": fixed_digest("chain-A"),
        "campaign_subject": fixed_digest("campaign-template"),
        "seed_commitment": seed_commitment(bytes(range(32))),
        "gauge_field": fixed_digest("field@120"),
        "sampler_subject": fixed_digest("cm_heatbath_or_v1"),
        "numerical_profile": fixed_digest("numerical-profile-v1"),
        "environment": fixed_digest("environment-v1"),
        "previous_checkpoint": bytes(32),
    }
    base.update(overrides)
    return base


def verify_lineage(records):
    previous_digest = bytes(32)
    previous_update = None
    for index, record in enumerate(records):
        if record["previous_checkpoint"] != previous_digest:
            raise ValueError(f"predecessor mismatch at {index}")
        if previous_update is not None and record["update_ordinal"] <= previous_update:
            raise ValueError(f"non-monotonic update ordinal at {index}")
        previous_digest = checkpoint_digest(record)
        previous_update = record["update_ordinal"]
    return previous_digest


def main():
    stream_ids = {}
    for phase_name, phase in PHASE.items():
        for purpose_name, purpose in PURPOSE.items():
            key = f"{phase_name}:{purpose_name}"
            stream_ids[key] = pack_stream_id(phase, purpose, 0x021D, 7, 0)
    if len(set(stream_ids.values())) != len(stream_ids):
        raise AssertionError("phase/purpose stream collision")

    if stream_ids["pilot:gauge_transition"] == stream_ids["final:gauge_transition"]:
        raise AssertionError("pilot/final transition stream collision")
    if stream_ids["throughput:gauge_transition"] == stream_ids["pilot:gauge_transition"]:
        raise AssertionError("throughput/pilot transition stream collision")

    genesis = make_record()
    genesis_digest = checkpoint_digest(genesis)

    second = make_record(
        update_ordinal=240,
        word_pos=1987654321,
        gauge_field=fixed_digest("field@240"),
        previous_checkpoint=genesis_digest,
    )
    second_digest = checkpoint_digest(second)

    third = make_record(
        update_ordinal=360,
        word_pos=2987654321,
        gauge_field=fixed_digest("field@360"),
        previous_checkpoint=second_digest,
    )
    third_digest = verify_lineage([genesis, second, third])
    if third_digest != checkpoint_digest(third):
        raise AssertionError("lineage terminal mismatch")

    stale = dict(third)
    stale["previous_checkpoint"] = genesis_digest
    try:
        verify_lineage([genesis, second, stale])
        raise AssertionError("stale predecessor accepted")
    except ValueError as exc:
        if "predecessor mismatch" not in str(exc):
            raise

    fork = dict(second)
    fork["gauge_field"] = fixed_digest("different-field@240")
    fork_digest = checkpoint_digest(fork)
    if fork_digest == second_digest:
        raise AssertionError("field fork did not change checkpoint identity")

    mutations = {
        "phase": PHASE["final"],
        "purpose": PURPOSE["diagnostic"],
        "ensemble_slot": 0x021E,
        "replica": 8,
        "rank": 1,
        "update_ordinal": 121,
        "word_pos": 987654322,
        "chain_id": fixed_digest("chain-B"),
        "campaign_subject": fixed_digest("campaign-template-v2"),
        "seed_commitment": seed_commitment(bytes(reversed(range(32)))),
        "gauge_field": fixed_digest("field@120-mutated"),
        "sampler_subject": fixed_digest("other-sampler"),
        "numerical_profile": fixed_digest("numerical-profile-v2"),
        "environment": fixed_digest("environment-v2"),
        "previous_checkpoint": fixed_digest("different-predecessor"),
    }
    mutation_digests = {}
    for field, value in mutations.items():
        mutated = dict(genesis)
        mutated[field] = value
        digest = checkpoint_digest(mutated)
        if digest == genesis_digest:
            raise AssertionError("mutation did not alter checkpoint: " + field)
        mutation_digests[field] = hx(digest)

    result = {
        "oracle_id": ORACLE_ID,
        "stream_contract": {
            "packing": "[phase:4|purpose:4|ensemble_slot:24|replica:16|rank:16]",
            "pilot_gauge_transition": f"0x{stream_ids['pilot:gauge_transition']:016x}",
            "final_gauge_transition": f"0x{stream_ids['final:gauge_transition']:016x}",
            "throughput_gauge_transition": f"0x{stream_ids['throughput:gauge_transition']:016x}",
            "combination_count": len(stream_ids),
            "unique_stream_id_count": len(set(stream_ids.values())),
        },
        "checkpoint_contract": {
            "tag": CHECKPOINT_TAG.decode("ascii").rstrip("\0"),
            "seed_commitment_tag": SEED_TAG.decode("ascii").rstrip("\0"),
            "genesis_digest": hx(genesis_digest),
            "second_digest": hx(second_digest),
            "third_digest": hx(third_digest),
            "mutation_digests": mutation_digests,
            "stale_predecessor_rejected": True,
            "same_ordinal_field_fork_distinct": True,
        },
        "claim_boundary": {
            "chaCha8_byte_restart_parity_established": False,
            "checkpoint_commitment_and_namespace_semantics_established": True,
        },
    }
    text = json.dumps(result, sort_keys=True, separators=(",", ":"))
    result_digest = hashlib.sha256(text.encode()).hexdigest()
    print("ok")
    print("result_sha256=" + result_digest)
    print(text)


if __name__ == "__main__":
    main()
