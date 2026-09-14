#!/usr/bin/env python3
"""Independent GEOM D0A3 commitment oracle.

Dependency-free pure-Python BLAKE3 + GEOM commitment encoding. This file does
not import or invoke the Rust `blake3` implementation under qualification.
"""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

IV = [
    0x6A09E667,
    0xBB67AE85,
    0x3C6EF372,
    0xA54FF53A,
    0x510E527F,
    0x9B05688C,
    0x1F83D9AB,
    0x5BE0CD19,
]
MSG_PERMUTATION = [2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8]
CHUNK_START = 1
CHUNK_END = 2
PARENT = 4
ROOT = 8
BLOCK_LEN = 64
CHUNK_LEN = 1024


def rotr32(value: int, count: int) -> int:
    return ((value >> count) | ((value << (32 - count)) & 0xFFFFFFFF)) & 0xFFFFFFFF


def mix(state: list[int], a: int, b: int, c: int, d: int, mx: int, my: int) -> None:
    state[a] = (state[a] + state[b] + mx) & 0xFFFFFFFF
    state[d] = rotr32(state[d] ^ state[a], 16)
    state[c] = (state[c] + state[d]) & 0xFFFFFFFF
    state[b] = rotr32(state[b] ^ state[c], 12)
    state[a] = (state[a] + state[b] + my) & 0xFFFFFFFF
    state[d] = rotr32(state[d] ^ state[a], 8)
    state[c] = (state[c] + state[d]) & 0xFFFFFFFF
    state[b] = rotr32(state[b] ^ state[c], 7)


def round_fn(state: list[int], message: list[int]) -> None:
    mix(state, 0, 4, 8, 12, message[0], message[1])
    mix(state, 1, 5, 9, 13, message[2], message[3])
    mix(state, 2, 6, 10, 14, message[4], message[5])
    mix(state, 3, 7, 11, 15, message[6], message[7])
    mix(state, 0, 5, 10, 15, message[8], message[9])
    mix(state, 1, 6, 11, 12, message[10], message[11])
    mix(state, 2, 7, 8, 13, message[12], message[13])
    mix(state, 3, 4, 9, 14, message[14], message[15])


def permute(message: list[int]) -> list[int]:
    return [message[index] for index in MSG_PERMUTATION]


def words_from_block(block: bytes | bytearray) -> list[int]:
    if len(block) != BLOCK_LEN:
        raise ValueError("BLAKE3 block must be 64 bytes")
    return [int.from_bytes(block[offset : offset + 4], "little") for offset in range(0, 64, 4)]


def compress(
    cv: list[int], block_words: list[int], counter: int, block_len: int, flags: int
) -> list[int]:
    state = list(cv) + IV[:4] + [counter & 0xFFFFFFFF, counter >> 32, block_len, flags]
    message = list(block_words)
    for round_index in range(7):
        round_fn(state, message)
        if round_index != 6:
            message = permute(message)
    for index in range(8):
        state[index] ^= state[index + 8]
        state[index + 8] ^= cv[index]
    return [word & 0xFFFFFFFF for word in state]


class Output:
    def __init__(
        self,
        input_cv: list[int],
        block_words: list[int],
        counter: int,
        block_len: int,
        flags: int,
    ) -> None:
        self.input_cv = list(input_cv)
        self.block_words = list(block_words)
        self.counter = counter
        self.block_len = block_len
        self.flags = flags

    def chaining_value(self) -> list[int]:
        return compress(
            self.input_cv,
            self.block_words,
            self.counter,
            self.block_len,
            self.flags,
        )[:8]

    def root_output_bytes(self, length: int) -> bytes:
        output = bytearray()
        output_block_counter = 0
        while len(output) < length:
            words = compress(
                self.input_cv,
                self.block_words,
                output_block_counter,
                self.block_len,
                self.flags | ROOT,
            )
            output.extend(b"".join(word.to_bytes(4, "little") for word in words))
            output_block_counter += 1
        return bytes(output[:length])


def parent_output(left: list[int], right: list[int]) -> Output:
    return Output(IV, left + right, 0, BLOCK_LEN, PARENT)


def parent_cv(left: list[int], right: list[int]) -> list[int]:
    return parent_output(left, right).chaining_value()


class ChunkState:
    def __init__(self, chunk_counter: int) -> None:
        self.cv = list(IV)
        self.chunk_counter = chunk_counter
        self.block = bytearray(BLOCK_LEN)
        self.block_len = 0
        self.blocks_compressed = 0

    def length(self) -> int:
        return self.blocks_compressed * BLOCK_LEN + self.block_len

    def start_flag(self) -> int:
        return CHUNK_START if self.blocks_compressed == 0 else 0

    def update(self, data: bytes) -> None:
        offset = 0
        while offset < len(data):
            if self.block_len == BLOCK_LEN:
                self.cv = compress(
                    self.cv,
                    words_from_block(self.block),
                    self.chunk_counter,
                    BLOCK_LEN,
                    self.start_flag(),
                )[:8]
                self.blocks_compressed += 1
                self.block = bytearray(BLOCK_LEN)
                self.block_len = 0
            take = min(BLOCK_LEN - self.block_len, len(data) - offset)
            self.block[self.block_len : self.block_len + take] = data[offset : offset + take]
            self.block_len += take
            offset += take

    def output(self) -> Output:
        return Output(
            self.cv,
            words_from_block(self.block),
            self.chunk_counter,
            self.block_len,
            self.start_flag() | CHUNK_END,
        )


class Blake3:
    def __init__(self) -> None:
        self.chunk_state = ChunkState(0)
        self.cv_stack: list[list[int]] = []

    def _add_chunk_cv(self, new_cv: list[int], total_chunks: int) -> None:
        while total_chunks & 1 == 0:
            new_cv = parent_cv(self.cv_stack.pop(), new_cv)
            total_chunks >>= 1
        self.cv_stack.append(new_cv)

    def update(self, data: bytes) -> "Blake3":
        offset = 0
        while offset < len(data):
            if self.chunk_state.length() == CHUNK_LEN:
                chunk_cv = self.chunk_state.output().chaining_value()
                total_chunks = self.chunk_state.chunk_counter + 1
                self._add_chunk_cv(chunk_cv, total_chunks)
                self.chunk_state = ChunkState(total_chunks)
            take = min(CHUNK_LEN - self.chunk_state.length(), len(data) - offset)
            self.chunk_state.update(data[offset : offset + take])
            offset += take
        return self

    def digest(self, length: int = 32) -> bytes:
        output = self.chunk_state.output()
        for left_cv in reversed(self.cv_stack):
            output = parent_output(left_cv, output.chaining_value())
        return output.root_output_bytes(length)

    def hexdigest(self) -> str:
        return self.digest().hex()


def blake3(data: bytes) -> str:
    return Blake3().update(data).hexdigest()


class CommitmentWriter:
    def __init__(self, domain: str) -> None:
        self.hasher = Blake3()
        self.hasher.update(b"SYMTHEAEA-COMMITMENT\x00")
        self.string(domain)

    def u64(self, value: int) -> None:
        self.hasher.update(int(value).to_bytes(8, "little"))

    def bytes(self, value: bytes) -> None:
        self.u64(len(value))
        self.hasher.update(value)

    def string(self, value: str) -> None:
        self.bytes(value.encode("utf-8"))

    def boolean(self, value: bool) -> None:
        self.hasher.update(bytes([1 if value else 0]))

    def optional_string(self, value: str | None) -> None:
        self.boolean(value is not None)
        if value is not None:
            self.string(value.lower())

    def string_vector(self, values: list[str]) -> None:
        self.u64(len(values))
        for value in values:
            self.string(value)

    def finish(self) -> str:
        return self.hasher.hexdigest()


def process_environment_commitment(entries: list[list[str]]) -> str:
    ordered = sorted((key, value) for key, value in entries)
    writer = CommitmentWriter("symthaea:geom:process-environment:v1")
    writer.u64(len(ordered))
    for key, value in ordered:
        writer.string(key)
        writer.string(value)
    return writer.finish()


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def canonical_json_commitment(domain: str, value: object) -> str:
    writer = CommitmentWriter(domain)
    writer.bytes(canonical_json_bytes(value))
    return writer.finish()


def run_environment_commitment(snapshot: dict[str, object]) -> str:
    writer = CommitmentWriter("symthaea:geom:run-environment:v1")
    writer.string("schema")
    writer.string(snapshot["schema"])

    source = snapshot["source"]
    writer.string("subject_sha")
    writer.string(source["subject_sha"].lower())
    writer.string("tree_sha")
    writer.string(source["tree_sha"].lower())
    writer.string("clean_tree")
    writer.boolean(source["clean_tree"])
    writer.string("cargo_lock_blake3")
    writer.string(source["cargo_lock_blake3"].lower())
    writer.string("flake_lock_blake3")
    writer.optional_string(source["flake_lock_blake3"])
    writer.string("rust_toolchain_blake3")
    writer.optional_string(source["rust_toolchain_blake3"])

    runtime = snapshot["runtime"]
    for field in (
        "rustc_version",
        "cargo_version",
        "host_triple",
        "target_triple",
        "os",
        "arch",
        "hardware_identity",
        "thread_policy",
    ):
        writer.string(field)
        writer.string(runtime[field])
    writer.string("cargo_features")
    features = sorted(set(feature.strip() for feature in runtime["cargo_features"] if feature.strip()))
    writer.string_vector(features)

    campaign = snapshot["campaign"]
    for field in (
        "campaign_id",
        "arm_id",
        "analysis_authority_revision",
        "input_schedule_commitment",
        "arm_order_commitment",
    ):
        writer.string(field)
        writer.string(campaign[field])
    writer.string("fixed_utc_hour_bits")
    hour = float(campaign["fixed_utc_hour"])
    if hour == 0.0:
        hour = 0.0
    writer.u64(struct.unpack("<Q", struct.pack("<d", hour))[0])

    for field in (
        "config_commitment",
        "preflight_commitment",
        "process_environment_commitment",
    ):
        writer.string(field)
        writer.string(snapshot[field].lower())
    writer.string("canonical_persistence_root")
    writer.string(snapshot["canonical_persistence_root"])
    return writer.finish()


def evidence_inventory_commitment(files: list[dict[str, str]]) -> tuple[list[tuple[str, int, str]], str]:
    artifacts = []
    for file in files:
        content = bytes.fromhex(file["content_hex"])
        artifacts.append((file["path"], len(content), blake3(content)))
    artifacts.sort()

    writer = CommitmentWriter("symthaea:geom:evidence-inventory:v1")
    writer.string("symthaea.geom.evidence-inventory.v1")
    writer.u64(len(artifacts))
    for path, size, digest in artifacts:
        writer.string(path)
        writer.u64(size)
        writer.string(digest)
    return artifacts, writer.finish()


def require(label: str, observed: str, expected: str) -> None:
    if observed != expected:
        raise SystemExit(f"{label}: expected {expected}, observed {observed}")


def main() -> int:
    # Independent BLAKE3 implementation self-check before any GEOM vector.
    require(
        "BLAKE3 empty",
        blake3(b""),
        "af1349b9f5f9a1a6a0404dea36dcc9499bcb25c9adc112b7cc9a93cae41f3262",
    )
    require(
        "BLAKE3 abc",
        blake3(b"abc"),
        "6437b3ac38465133ffb63b75273a8db548c558465d79db03fd359c6cd5bd9d85",
    )

    root = Path(__file__).resolve().parents[2]
    fixture_path = root / "docs" / "research" / "vectors" / "GEOM_RUN_SEAL_V1.json"
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))

    env = fixture["process_environment"]
    require(
        "process environment",
        process_environment_commitment(env["entries"]),
        env["expected_commitment"],
    )

    canonical = fixture["canonical_json"]
    canonical_utf8 = canonical_json_bytes(canonical["value"]).decode("utf-8")
    require("canonical JSON bytes", canonical_utf8, canonical["expected_canonical_utf8"])
    require(
        "canonical JSON commitment",
        canonical_json_commitment(canonical["domain"], canonical["value"]),
        canonical["expected_commitment"],
    )

    run = fixture["run_environment"]
    require(
        "run environment",
        run_environment_commitment(run["snapshot"]),
        run["expected_commitment"],
    )

    evidence = fixture["evidence"]
    artifacts, inventory = evidence_inventory_commitment(evidence["files"])
    expected_by_path = {item["path"]: item["expected_blake3"] for item in evidence["files"]}
    for path, _size, digest in artifacts:
        require(f"evidence file {path}", digest, expected_by_path[path])
    require("evidence inventory", inventory, evidence["expected_inventory_commitment"])

    print("GEOM D0A3 independent commitment oracle: PASS")
    print(f"process_environment={env['expected_commitment']}")
    print(f"canonical_json={canonical['expected_commitment']}")
    print(f"run_environment={run['expected_commitment']}")
    print(f"evidence_inventory={evidence['expected_inventory_commitment']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
