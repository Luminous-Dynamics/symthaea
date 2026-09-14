#!/usr/bin/env python3
"""Independent LQCD-021O checkpoint commitment oracle."""
import hashlib
import json
import struct
from dataclasses import dataclass, replace
from typing import Optional

RECORD_TAG = b"symthaea.lqcd.checkpoint.record.v1\x00"
COMMIT_TAG = b"symthaea.lqcd.checkpoint.commitment.v1\x00"
SEED_TAG = b"symthaea.lqcd.seed.commitment.v1\x00"

PHASE = {"throughput": 1, "pilot": 2, "final": 3, "qualification": 4}
PURPOSE = {"gauge_transition": 1, "diagnostic": 2, "analysis": 3, "bootstrap": 4}

def stream_id(phase, purpose, ensemble_slot, replica, rank):
    if not (1 <= phase <= 15 and 1 <= purpose <= 15):
        raise ValueError("invalid phase/purpose")
    if not (0 <= ensemble_slot < (1 << 24)):
        raise ValueError("ensemble_slot out of range")
    if not (0 <= replica < (1 << 16) and 0 <= rank < (1 << 16)):
        raise ValueError("replica/rank out of range")
    return (
        (phase << 60)
        | (purpose << 56)
        | (ensemble_slot << 32)
        | (replica << 16)
        | rank
    )

def seed_commitment(seed):
    if len(seed) != 32:
        raise ValueError("seed must be exactly 32 bytes")
    return hashlib.sha256(SEED_TAG + seed).digest()

@dataclass(frozen=True)
class Checkpoint:
    campaign_subject: bytes
    chain_id: str
    checkpoint_index: int
    update_ordinal: int
    phase: int
    purpose: int
    ensemble_slot: int
    replica: int
    rank: int
    stream_id: int
    rng_word_pos: int
    seed_commitment: bytes
    field_encoding_id: str
    field_sha256: bytes
    sampler_stable_id: str
    sampler_revision: str
    numerical_profile_id: str
    environment_profile_id: str
    predecessor: Optional[bytes]

def _require_id(value):
    if not isinstance(value, str) or not value:
        raise ValueError("required identifier must be non-empty")
    raw = value.encode("utf-8")
    if len(raw) > 0xFFFFFFFF:
        raise ValueError("identifier too long")
    return raw

def validate(checkpoint):
    if len(checkpoint.campaign_subject) != 32:
        raise ValueError("campaign subject must be 32 bytes")
    if len(checkpoint.seed_commitment) != 32:
        raise ValueError("seed commitment must be 32 bytes")
    if len(checkpoint.field_sha256) != 32:
        raise ValueError("field digest must be 32 bytes")
    if checkpoint.predecessor is not None and len(checkpoint.predecessor) != 32:
        raise ValueError("predecessor must be 32 bytes")
    if not (0 <= checkpoint.checkpoint_index < (1 << 64)):
        raise ValueError("checkpoint index out of range")
    if not (0 <= checkpoint.update_ordinal < (1 << 64)):
        raise ValueError("update ordinal out of range")
    if not (0 <= checkpoint.rng_word_pos < (1 << 128)):
        raise ValueError("rng word position out of range")
    expected_stream = stream_id(
        checkpoint.phase,
        checkpoint.purpose,
        checkpoint.ensemble_slot,
        checkpoint.replica,
        checkpoint.rank,
    )
    if checkpoint.stream_id != expected_stream:
        raise ValueError("stream id does not match packed coordinates")
    if checkpoint.checkpoint_index == 0:
        if checkpoint.predecessor is not None:
            raise ValueError("genesis checkpoint must not have predecessor")
    elif checkpoint.predecessor is None:
        raise ValueError("non-genesis checkpoint requires predecessor")
    for value in (
        checkpoint.chain_id,
        checkpoint.field_encoding_id,
        checkpoint.sampler_stable_id,
        checkpoint.sampler_revision,
        checkpoint.numerical_profile_id,
        checkpoint.environment_profile_id,
    ):
        _require_id(value)

def _lp(value):
    raw = _require_id(value)
    return len(raw).to_bytes(4, "big") + raw

def encode(checkpoint):
    validate(checkpoint)
    out = bytearray(RECORD_TAG)
    out += checkpoint.campaign_subject
    out += _lp(checkpoint.chain_id)
    out += checkpoint.checkpoint_index.to_bytes(8, "big")
    out += checkpoint.update_ordinal.to_bytes(8, "big")
    out += bytes((checkpoint.phase, checkpoint.purpose))
    out += checkpoint.ensemble_slot.to_bytes(3, "big")
    out += checkpoint.replica.to_bytes(2, "big")
    out += checkpoint.rank.to_bytes(2, "big")
    out += checkpoint.stream_id.to_bytes(8, "big")
    out += checkpoint.rng_word_pos.to_bytes(16, "big")
    out += checkpoint.seed_commitment
    out += _lp(checkpoint.field_encoding_id)
    out += checkpoint.field_sha256
    out += _lp(checkpoint.sampler_stable_id)
    out += _lp(checkpoint.sampler_revision)
    out += _lp(checkpoint.numerical_profile_id)
    out += _lp(checkpoint.environment_profile_id)
    out += b"\x00" if checkpoint.predecessor is None else b"\x01" + checkpoint.predecessor
    return bytes(out)

def checkpoint_commitment(checkpoint):
    return hashlib.sha256(COMMIT_TAG + encode(checkpoint)).digest()

def _read_lp(data, cursor):
    if cursor + 4 > len(data):
        raise ValueError("truncated length prefix")
    length = int.from_bytes(data[cursor:cursor + 4], "big")
    cursor += 4
    if length == 0 or cursor + length > len(data):
        raise ValueError("invalid length-prefixed identifier")
    raw = data[cursor:cursor + length]
    cursor += length
    return raw.decode("utf-8"), cursor

def decode(data):
    if not data.startswith(RECORD_TAG):
        raise ValueError("wrong record tag")
    cursor = len(RECORD_TAG)

    def take(count):
        nonlocal cursor
        if cursor + count > len(data):
            raise ValueError("truncated checkpoint")
        value = data[cursor:cursor + count]
        cursor += count
        return value

    campaign_subject = take(32)
    chain_id, cursor = _read_lp(data, cursor)
    checkpoint_index = int.from_bytes(take(8), "big")
    update_ordinal = int.from_bytes(take(8), "big")
    phase = take(1)[0]
    purpose = take(1)[0]
    ensemble_slot = int.from_bytes(take(3), "big")
    replica = int.from_bytes(take(2), "big")
    rank = int.from_bytes(take(2), "big")
    packed_stream_id = int.from_bytes(take(8), "big")
    rng_word_pos = int.from_bytes(take(16), "big")
    seed_commit = take(32)
    field_encoding_id, cursor = _read_lp(data, cursor)
    field_digest = take(32)
    sampler_stable_id, cursor = _read_lp(data, cursor)
    sampler_revision, cursor = _read_lp(data, cursor)
    numerical_profile_id, cursor = _read_lp(data, cursor)
    environment_profile_id, cursor = _read_lp(data, cursor)
    predecessor_tag = take(1)[0]
    if predecessor_tag == 0:
        predecessor = None
    elif predecessor_tag == 1:
        predecessor = take(32)
    else:
        raise ValueError("invalid predecessor tag")
    if cursor != len(data):
        raise ValueError("trailing checkpoint bytes")

    checkpoint = Checkpoint(
        campaign_subject=campaign_subject,
        chain_id=chain_id,
        checkpoint_index=checkpoint_index,
        update_ordinal=update_ordinal,
        phase=phase,
        purpose=purpose,
        ensemble_slot=ensemble_slot,
        replica=replica,
        rank=rank,
        stream_id=packed_stream_id,
        rng_word_pos=rng_word_pos,
        seed_commitment=seed_commit,
        field_encoding_id=field_encoding_id,
        field_sha256=field_digest,
        sampler_stable_id=sampler_stable_id,
        sampler_revision=sampler_revision,
        numerical_profile_id=numerical_profile_id,
        environment_profile_id=environment_profile_id,
        predecessor=predecessor,
    )
    validate(checkpoint)
    return checkpoint

def classify_child(current_head, candidate):
    if candidate.checkpoint_index != current_head.checkpoint_index + 1:
        return "StaleOrSkippedIndex"
    if candidate.predecessor != checkpoint_commitment(current_head):
        return "StalePredecessor"
    if candidate.update_ordinal <= current_head.update_ordinal:
        return "NonMonotonicUpdateOrdinal"
    return "EligibleChild"

def main():
    seed = bytes(range(32))
    wrong_seed = bytes(reversed(range(32)))
    campaign_subject = bytes.fromhex(
        "69fde4f216129e45fe3b523977ea00d94630ca684b7ee8c51deec726430fcbf1"
    )
    field_digest = bytes.fromhex(
        "7bd62ca29833be0171a4f3232c03a3edca125aa50b33aeb187e702fb1c8c996b"
    )
    packed = stream_id(PHASE["final"], PURPOSE["gauge_transition"], 0x021D, 7, 0)
    genesis = Checkpoint(
        campaign_subject=campaign_subject,
        chain_id="beta6-final-chain-0007",
        checkpoint_index=0,
        update_ordinal=8000,
        phase=PHASE["final"],
        purpose=PURPOSE["gauge_transition"],
        ensemble_slot=0x021D,
        replica=7,
        rank=0,
        stream_id=packed,
        rng_word_pos=123456789,
        seed_commitment=seed_commitment(seed),
        field_encoding_id="wilson_gauge_field_be_f64_v1",
        field_sha256=field_digest,
        sampler_stable_id="cm_heatbath_or_v1:force=staple:or_sweeps=3:max_attempts=256",
        sampler_revision="56c115b46920fc8b34daa8897051cff59c61a0bf",
        numerical_profile_id="lqcd_numerical_profile_candidate_v1",
        environment_profile_id="symthaea-rust196-linux-x86_64-candidate-v1",
        predecessor=None,
    )
    genesis_bytes = encode(genesis)
    genesis_commit = checkpoint_commitment(genesis)
    assert decode(genesis_bytes) == genesis
    assert encode(decode(genesis_bytes)) == genesis_bytes

    successor = replace(
        genesis,
        checkpoint_index=1,
        update_ordinal=8100,
        rng_word_pos=126789012,
        predecessor=genesis_commit,
    )
    successor_bytes = encode(successor)
    successor_commit = checkpoint_commitment(successor)
    assert decode(successor_bytes) == successor
    assert classify_child(genesis, successor) == "EligibleChild"

    mutations = {
        "campaign_subject": replace(successor, campaign_subject=bytes([successor.campaign_subject[0] ^ 1]) + successor.campaign_subject[1:]),
        "chain_id": replace(successor, chain_id=successor.chain_id + "-mut"),
        "checkpoint_index": replace(successor, checkpoint_index=2),
        "update_ordinal": replace(successor, update_ordinal=8101),
        "phase": replace(successor, phase=PHASE["pilot"], stream_id=stream_id(PHASE["pilot"], successor.purpose, successor.ensemble_slot, successor.replica, successor.rank)),
        "purpose": replace(successor, purpose=PURPOSE["diagnostic"], stream_id=stream_id(successor.phase, PURPOSE["diagnostic"], successor.ensemble_slot, successor.replica, successor.rank)),
        "ensemble_slot": replace(successor, ensemble_slot=0x021E, stream_id=stream_id(successor.phase, successor.purpose, 0x021E, successor.replica, successor.rank)),
        "replica": replace(successor, replica=8, stream_id=stream_id(successor.phase, successor.purpose, successor.ensemble_slot, 8, successor.rank)),
        "rank": replace(successor, rank=1, stream_id=stream_id(successor.phase, successor.purpose, successor.ensemble_slot, successor.replica, 1)),
        "rng_word_pos": replace(successor, rng_word_pos=successor.rng_word_pos + 1),
        "seed_commitment": replace(successor, seed_commitment=bytes([successor.seed_commitment[0] ^ 1]) + successor.seed_commitment[1:]),
        "field_encoding_id": replace(successor, field_encoding_id="wilson_gauge_field_be_f64_v2"),
        "field_sha256": replace(successor, field_sha256=bytes([successor.field_sha256[0] ^ 1]) + successor.field_sha256[1:]),
        "sampler_stable_id": replace(successor, sampler_stable_id=successor.sampler_stable_id + ":mut"),
        "sampler_revision": replace(successor, sampler_revision="0" * 40),
        "numerical_profile_id": replace(successor, numerical_profile_id=successor.numerical_profile_id + "-mut"),
        "environment_profile_id": replace(successor, environment_profile_id=successor.environment_profile_id + "-mut"),
        "predecessor": replace(successor, predecessor=bytes([successor.predecessor[0] ^ 1]) + successor.predecessor[1:]),
    }
    mutation_digests = {}
    for name, mutated in mutations.items():
        digest = checkpoint_commitment(mutated)
        assert digest != successor_commit, name
        mutation_digests[name] = digest.hex()

    assert seed_commitment(seed) == successor.seed_commitment
    assert seed_commitment(wrong_seed) != successor.seed_commitment

    one_bit_field = bytes([field_digest[0] ^ 1]) + field_digest[1:]
    assert one_bit_field != successor.field_sha256

    stale = replace(successor, checkpoint_index=2, update_ordinal=8200, predecessor=b"\x00" * 32)
    assert classify_child(successor, stale) == "StalePredecessor"

    sibling_a = replace(successor, checkpoint_index=2, update_ordinal=8200, predecessor=successor_commit, rng_word_pos=130000000)
    sibling_b = replace(sibling_a, field_sha256=bytes([field_digest[0] ^ 2]) + field_digest[1:])
    assert sibling_a.predecessor == sibling_b.predecessor
    assert sibling_a.checkpoint_index == sibling_b.checkpoint_index
    assert checkpoint_commitment(sibling_a) != checkpoint_commitment(sibling_b)

    negative_controls = {}
    for name, payload in {
        "truncated": successor_bytes[:-1],
        "trailing": successor_bytes + b"\x00",
        "wrong_tag": b"X" + successor_bytes[1:],
    }.items():
        try:
            decode(payload)
        except Exception:
            negative_controls[name] = "rejected"
        else:
            raise AssertionError(name)

    mismatch = replace(successor, stream_id=successor.stream_id ^ 1)
    try:
        encode(mismatch)
    except ValueError:
        negative_controls["stream_coordinate_mismatch"] = "rejected"
    else:
        raise AssertionError("stream mismatch")

    bad_genesis = replace(genesis, predecessor=b"\x00" * 32)
    try:
        encode(bad_genesis)
    except ValueError:
        negative_controls["genesis_with_predecessor"] = "rejected"
    else:
        raise AssertionError("genesis predecessor")

    bad_successor = replace(successor, predecessor=None)
    try:
        encode(bad_successor)
    except ValueError:
        negative_controls["successor_without_predecessor"] = "rejected"
    else:
        raise AssertionError("missing predecessor")

    result = {
        "oracle_id": "lqcd_checkpoint_commitment_oracle_v1",
        "record_tag_hex": RECORD_TAG.hex(),
        "commitment_tag_hex": COMMIT_TAG.hex(),
        "seed_tag_hex": SEED_TAG.hex(),
        "stream_id_hex": f"{packed:016x}",
        "seed_commitment_sha256": seed_commitment(seed).hex(),
        "wrong_seed_matches": False,
        "genesis": {
            "encoded_bytes": len(genesis_bytes),
            "commitment_sha256": genesis_commit.hex(),
            "field_sha256": field_digest.hex(),
        },
        "successor": {
            "encoded_bytes": len(successor_bytes),
            "commitment_sha256": successor_commit.hex(),
            "predecessor_sha256": genesis_commit.hex(),
            "classification": classify_child(genesis, successor),
        },
        "authoritative_field_mutations": len(mutation_digests),
        "mutation_commitments": mutation_digests,
        "negative_controls": negative_controls,
        "stale_predecessor_classification": classify_child(successor, stale),
        "sibling_fork_detected": True,
        "scientific_boundary": "commitment_semantics_only_no_rust_parity_no_filesystem_durability_no_physics",
    }
    canonical = json.dumps(result, sort_keys=True, separators=(",", ":"))
    print("ok")
    print("result_sha256=" + hashlib.sha256(canonical.encode()).hexdigest())
    print(canonical)

if __name__ == "__main__":
    main()
