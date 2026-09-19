#!/usr/bin/env python3
"""Reference implementation for math-structural-compat-wire-v1.

This tool intentionally hashes raw bytes and IEEE-754 bit patterns rather than
human-readable float formatting. It is qualification support, not a scientific
evaluator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import sys
from pathlib import Path
from typing import Any, Iterable

PROFILE = "math-structural-compat-wire-v1"
DOM_HDC = b"MATH-HDC-V1\0"
DOM_SPARSE = b"MATH-AST-SPARSE-V1\0"
DOM_RANK = b"MATH-RANKING-V1\0"
DOM_HDC_SIM = b"MATH-HDC-SIM-TRANSCRIPT-V1\0"
DOM_AST_COS = b"MATH-AST-COSINE-TRANSCRIPT-V1\0"
DOM_HOLDOUT = b"MATH-HOLDOUT-REPR-COMPAT-V1\0"
HEX8 = set("0123456789abcdef")


class WireError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise WireError(message)


def u32le(value: int) -> bytes:
    require(isinstance(value, int) and 0 <= value <= 0xFFFFFFFF, "u32 out of range")
    return struct.pack("<I", value)


def lp_utf8(value: str) -> bytes:
    require(isinstance(value, str) and value != "", "length-prefixed string must be non-empty")
    raw = value.encode("utf-8")
    return u32le(len(raw)) + raw


def sha256_hex(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def parse_hex(value: Any, nbytes: int, where: str) -> bytes:
    require(isinstance(value, str), f"{where}: expected hex string")
    require(len(value) == nbytes * 2, f"{where}: expected {nbytes * 2} lowercase hex chars")
    require(all(ch in HEX8 for ch in value), f"{where}: expected lowercase hex")
    return bytes.fromhex(value)


def bits_u32_le(value_bits: Any, where: str) -> bytes:
    raw = parse_hex(value_bits, 4, where)
    numeric = int.from_bytes(raw, "big")
    value = struct.unpack("<f", numeric.to_bytes(4, "little"))[0]
    require(math.isfinite(value), f"{where}: similarity must be finite")
    require(0.0 <= value <= 1.0, f"{where}: HDC similarity must be in [0,1]")
    return numeric.to_bytes(4, "little")


def bits_u64_le(value_bits: Any, where: str, *, sparse_count: bool = False) -> bytes:
    raw = parse_hex(value_bits, 8, where)
    numeric = int.from_bytes(raw, "big")
    value = struct.unpack("<d", numeric.to_bytes(8, "little"))[0]
    require(math.isfinite(value), f"{where}: value must be finite")
    if sparse_count:
        require(value >= 0.0 and value.is_integer(), f"{where}: sparse feature count must be a non-negative integer-valued f64")
        require(value <= 2**53, f"{where}: sparse count exceeds exact f64 integer range")
    else:
        require(0.0 <= value <= 1.0, f"{where}: cosine must be in [0,1]")
    return numeric.to_bytes(8, "little")


def hdc_digest(raw: bytes) -> str:
    require(isinstance(raw, (bytes, bytearray)) and len(raw) == 2048, "HDC wire requires exactly 2048 bytes")
    return sha256_hex(DOM_HDC + bytes(raw))


def sparse_digest(entries: Iterable[dict[str, Any]]) -> str:
    rows = list(entries)
    require(rows, "sparse map must contain at least one feature")
    parsed: list[tuple[str, bytes]] = []
    seen: set[str] = set()
    for index, row in enumerate(rows):
        require(set(row) == {"key", "value_bits"}, f"sparse entry {index}: unexpected fields")
        key = row["key"]
        require(isinstance(key, str) and key, f"sparse entry {index}: key must be non-empty")
        require(key not in seen, f"duplicate sparse key: {key}")
        seen.add(key)
        parsed.append((key, bits_u64_le(row["value_bits"], f"sparse[{key}]", sparse_count=True)))
    parsed.sort(key=lambda item: item[0])
    wire = bytearray(DOM_SPARSE)
    wire += u32le(len(parsed))
    for key, value_bytes in parsed:
        wire += lp_utf8(key)
        wire += value_bytes
    return sha256_hex(bytes(wire))


def ranking_digest(candidate_ids: Iterable[str]) -> str:
    ids = list(candidate_ids)
    require(ids, "ranking must contain at least one candidate")
    require(len(ids) == len(set(ids)), "ranking candidate ids must be unique")
    wire = bytearray(DOM_RANK)
    wire += u32le(len(ids))
    for candidate_id in ids:
        wire += lp_utf8(candidate_id)
    return sha256_hex(bytes(wire))


def transcript_digest(records: Iterable[dict[str, Any]], *, f32: bool) -> str:
    rows = list(records)
    require(rows, "transcript must contain at least one record")
    wire = bytearray(DOM_HDC_SIM if f32 else DOM_AST_COS)
    wire += u32le(len(rows))
    seen: set[tuple[str, str]] = set()
    for index, row in enumerate(rows):
        require(set(row) == {"case_id", "candidate_id", "value_bits"}, f"transcript record {index}: unexpected fields")
        key = (row["case_id"], row["candidate_id"])
        require(key not in seen, f"duplicate transcript coordinate: {key}")
        seen.add(key)
        wire += lp_utf8(row["case_id"])
        wire += lp_utf8(row["candidate_id"])
        wire += bits_u32_le(row["value_bits"], f"transcript[{key}]") if f32 else bits_u64_le(row["value_bits"], f"transcript[{key}]")
    return sha256_hex(bytes(wire))


def holdout_aggregate_digest(cases: Iterable[dict[str, Any]]) -> str:
    """Aggregate representation digests without labels, ids, scores, or rankings."""
    case_rows = list(cases)
    require(case_rows, "holdout aggregate must contain at least one case")
    wire = bytearray(DOM_HOLDOUT)
    wire += u32le(len(case_rows))
    for case_index, case in enumerate(case_rows):
        require(set(case) == {"representations"}, f"holdout case {case_index}: labels/ids are forbidden")
        reps = case["representations"]
        require(isinstance(reps, list) and reps, f"holdout case {case_index}: representations required")
        wire += u32le(len(reps))
        for rep_index, rep in enumerate(reps):
            require(set(rep) == {"hdc_sha256", "sparse_sha256"}, f"holdout case {case_index} rep {rep_index}: only representation digests allowed")
            wire += parse_hex(rep["hdc_sha256"], 32, f"holdout[{case_index}][{rep_index}].hdc_sha256")
            wire += parse_hex(rep["sparse_sha256"], 32, f"holdout[{case_index}][{rep_index}].sparse_sha256")
    return sha256_hex(bytes(wire))


def check_vectors(doc: dict[str, Any]) -> None:
    require(doc.get("version") == PROFILE, "vector profile mismatch")

    for vector in doc.get("hdc_vectors", []):
        require(set(vector) == {"id", "byte_pattern_hex", "repeat", "expected_sha256"}, "HDC vector fields mismatch")
        pattern = parse_hex(vector["byte_pattern_hex"], len(vector["byte_pattern_hex"]) // 2, f"hdc[{vector['id']}].pattern")
        require(pattern, "HDC byte pattern must be non-empty")
        raw = pattern * vector["repeat"]
        require(hdc_digest(raw) == vector["expected_sha256"], f"HDC vector mismatch: {vector['id']}")

    for vector in doc.get("sparse_maps", []):
        require(sparse_digest(vector["entries"]) == vector["expected_sha256"], f"sparse vector mismatch: {vector['id']}")

    for vector in doc.get("rankings", []):
        require(ranking_digest(vector["candidate_ids"]) == vector["expected_sha256"], f"ranking vector mismatch: {vector['id']}")

    for vector in doc.get("hdc_similarity_transcripts", []):
        require(transcript_digest(vector["records"], f32=True) == vector["expected_sha256"], f"HDC transcript mismatch: {vector['id']}")

    for vector in doc.get("ast_cosine_transcripts", []):
        require(transcript_digest(vector["records"], f32=False) == vector["expected_sha256"], f"AST transcript mismatch: {vector['id']}")

    for vector in doc.get("holdout_aggregates", []):
        require(holdout_aggregate_digest(vector["cases"]) == vector["expected_sha256"], f"holdout aggregate mismatch: {vector['id']}")


def expect_reject(fn, label: str) -> None:
    try:
        fn()
    except WireError:
        return
    raise AssertionError(f"negative self-test unexpectedly accepted: {label}")


def self_test(vectors: dict[str, Any]) -> None:
    check_vectors(vectors)

    sparse = vectors["sparse_maps"][0]["entries"]
    require(sparse_digest(sparse) == sparse_digest(list(reversed(sparse))), "sparse key order must canonicalize")

    expected = vectors["rankings"][0]["expected_sha256"]
    require(ranking_digest(["a1", "a2", "a3"]) == expected, "ranking positive control failed")
    require(ranking_digest(["a2", "a1", "a3"]) != expected, "ranking order must be committed")

    expect_reject(lambda: hdc_digest(b"\x00" * 2047), "short HDC")
    expect_reject(lambda: sparse_digest([{"key": "x", "value_bits": "3ff8000000000000"}]), "fractional sparse count")
    expect_reject(lambda: sparse_digest([{"key": "x", "value_bits": "7ff8000000000000"}]), "NaN sparse count")
    expect_reject(lambda: ranking_digest(["a", "a"]), "duplicate ranking id")
    expect_reject(lambda: transcript_digest([{"case_id": "c", "candidate_id": "a", "value_bits": "7fc00000"}], f32=True), "NaN HDC similarity")
    expect_reject(lambda: holdout_aggregate_digest([{"case_id": "leak", "representations": []}]), "holdout label leak")

    print("PASS: math-structural-compat-wire-v1 reference vectors and negative controls")


def load(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        value = json.load(fh)
    require(isinstance(value, dict), "JSON root must be an object")
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vectors", type=Path, required=True)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    try:
        vectors = load(args.vectors)
        check_vectors(vectors)
        if args.self_test:
            self_test(vectors)
        else:
            print(f"PASS: {args.vectors}")
        return 0
    except (WireError, OSError, json.JSONDecodeError, AssertionError, KeyError, TypeError, ValueError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
