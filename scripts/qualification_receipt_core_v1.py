#!/usr/bin/env python3
"""Independent reference framing/projection for QualificationReceiptCoreV1.

This script deliberately duplicates the canonical framing in a different language. It consumes
already-validated PassSelectionV2 + ReceiptCandidateGateV1 objects, projects the provider/retry-
neutral portable claim, reproduces the canonical byte transcript, and computes BLAKE3 with a small
reference implementation.

The resulting core is not trusted PASS. It says only that the required recipe observations selected
by the upstream structural layer report Passed. Witnessing, signer authentication and current
admission remain separate theorems.
"""

from __future__ import annotations

import re
import struct
from typing import Any

SCHEMA = "symthaea.qualification-receipt-core.v1"
DOMAIN = b"symthaea.qualification-receipt-core.v1"
PASS_SELECTION_SCHEMA = "symthaea.qualification-pass-selection.v2"
GATE_SCHEMA = "symthaea.qualification-receipt-candidate-gate.v1"
SELECTION_DISPOSITION = "SelectedRequiredRecipesReportedPassedV1"
CROSS_CUTTING_DISPOSITION = "NoGenericCrossCuttingRulesV1"
NON_CLAIM_TAGS = [
    "AttemptHistoryExcludedFromSemanticIdentity",
    "NoCurrentAdmission",
    "NoDetachedAttestation",
    "NoMergeOrExecutionAuthority",
    "NoProviderAuthenticity",
    "NoScientificValidity",
    "NoTrustedPassEstablished",
]

_SHA256_ID = re.compile(r"^sha256:[0-9a-f]{64}$")
_RECIPE_ID = re.compile(r"^(?:sha256:[0-9a-f]{64}|git-blob-sha1:[0-9a-f]{40})$")

_IV = [
    0x6A09E667,
    0xBB67AE85,
    0x3C6EF372,
    0xA54FF53A,
    0x510E527F,
    0x9B05688C,
    0x1F83D9AB,
    0x5BE0CD19,
]
_MSG_PERMUTATION = [2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8]
_CHUNK_START = 1
_CHUNK_END = 2
_PARENT = 4
_ROOT = 8
_BLOCK_LEN = 64
_CHUNK_LEN = 1024
_MASK32 = 0xFFFFFFFF


class ReceiptCoreError(ValueError):
    pass


def _require_exact_keys(value: Any, expected: set[str], where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ReceiptCoreError(f"{where}: expected object")
    observed = set(value)
    missing = sorted(expected - observed)
    unknown = sorted(observed - expected)
    if missing or unknown:
        raise ReceiptCoreError(f"{where}: schema mismatch missing={missing} unknown={unknown}")
    return value


def _require_sha256_id(value: Any, where: str) -> str:
    if not isinstance(value, str) or _SHA256_ID.fullmatch(value) is None:
        raise ReceiptCoreError(f"{where}: expected sha256:<64 lowercase hex>")
    return value


def _require_recipe_id(value: Any, where: str) -> str:
    if not isinstance(value, str) or _RECIPE_ID.fullmatch(value) is None:
        raise ReceiptCoreError(f"{where}: invalid recipe identity")
    return value


def project_core(selection: Any, gate: Any) -> dict[str, Any]:
    """Project a portable untrusted receipt core from matching structural candidate objects."""
    selection = _require_exact_keys(
        selection,
        {
            "schema",
            "admission_subject_id",
            "qualification_subject_id",
            "qualification_profile_id",
            "input_closure_id",
            "qualification_environment_id",
            "attempt_history_id",
            "selected_recipe_attempts",
            "cross_cutting_evidence",
            "non_claims",
            "pass_selection_id",
        },
        "pass_selection",
    )
    gate = _require_exact_keys(
        gate,
        {
            "schema",
            "qualification_subject_id",
            "qualification_profile_id",
            "attempt_history_id",
            "pass_selection_id",
            "support_closure_id",
            "required_recipe_ids",
            "disposition",
            "non_claims",
            "receipt_candidate_id",
        },
        "receipt_candidate_gate",
    )

    if selection["schema"] != PASS_SELECTION_SCHEMA:
        raise ReceiptCoreError("pass_selection.schema: unsupported")
    if gate["schema"] != GATE_SCHEMA:
        raise ReceiptCoreError("receipt_candidate_gate.schema: unsupported")
    if gate["disposition"] != "WitnessRequired":
        raise ReceiptCoreError("receipt_candidate_gate.disposition: WitnessRequired required")
    for field in (
        "qualification_subject_id",
        "qualification_profile_id",
        "attempt_history_id",
        "pass_selection_id",
    ):
        if selection[field] != gate[field]:
            raise ReceiptCoreError(f"{field}: selection/gate mismatch")

    if selection["cross_cutting_evidence"]:
        raise ReceiptCoreError("V1 receipt core requires no generic cross-cutting evidence")

    selected = selection["selected_recipe_attempts"]
    if not isinstance(selected, list) or not selected:
        raise ReceiptCoreError("selected_recipe_attempts: non-empty array required")

    recipes: list[dict[str, str]] = []
    for index, item in enumerate(selected):
        item = _require_exact_keys(
            item,
            {
                "recipe_id",
                "attempt_subject_id",
                "attempt_registration_id",
                "attempt_observation_id",
                "evidence_content_ids",
            },
            f"selected_recipe_attempts[{index}]",
        )
        recipes.append(
            {
                "recipe_id": _require_recipe_id(item["recipe_id"], f"recipes[{index}].recipe_id"),
                "attempt_subject_id": _require_sha256_id(
                    item["attempt_subject_id"], f"recipes[{index}].attempt_subject_id"
                ),
            }
        )

    recipes.sort(key=lambda item: (item["recipe_id"], item["attempt_subject_id"]))
    recipe_ids = [item["recipe_id"] for item in recipes]
    if len(recipe_ids) != len(set(recipe_ids)):
        raise ReceiptCoreError("recipe identities must be unique")
    if recipe_ids != gate["required_recipe_ids"]:
        raise ReceiptCoreError("required recipe set differs from selected recipe theorem set")

    core = {
        "schema": SCHEMA,
        "qualification_subject_id": _require_sha256_id(
            selection["qualification_subject_id"], "qualification_subject_id"
        ),
        "qualification_profile_id": _require_sha256_id(
            selection["qualification_profile_id"], "qualification_profile_id"
        ),
        "input_closure_id": _require_sha256_id(selection["input_closure_id"], "input_closure_id"),
        "qualification_environment_id": _require_sha256_id(
            selection["qualification_environment_id"], "qualification_environment_id"
        ),
        "recipes": recipes,
        "selection_disposition": SELECTION_DISPOSITION,
        "cross_cutting_disposition": CROSS_CUTTING_DISPOSITION,
        "non_claims": list(NON_CLAIM_TAGS),
    }
    core["qualification_receipt_id"] = receipt_id(core)
    return core


def frame_core(core: Any) -> bytes:
    core = _require_exact_keys(
        core,
        {
            "schema",
            "qualification_subject_id",
            "qualification_profile_id",
            "input_closure_id",
            "qualification_environment_id",
            "recipes",
            "selection_disposition",
            "cross_cutting_disposition",
            "non_claims",
        },
        "receipt_core",
    )
    if core["schema"] != SCHEMA:
        raise ReceiptCoreError("receipt_core.schema: unsupported")
    if core["selection_disposition"] != SELECTION_DISPOSITION:
        raise ReceiptCoreError("receipt_core.selection_disposition: unsupported")
    if core["cross_cutting_disposition"] != CROSS_CUTTING_DISPOSITION:
        raise ReceiptCoreError("receipt_core.cross_cutting_disposition: unsupported")
    if core["non_claims"] != NON_CLAIM_TAGS:
        raise ReceiptCoreError("receipt_core.non_claims: exact V1 non-claim tags required")

    recipes = core["recipes"]
    if not isinstance(recipes, list) or not recipes:
        raise ReceiptCoreError("receipt_core.recipes: non-empty array required")
    normalized: list[dict[str, str]] = []
    for index, item in enumerate(recipes):
        item = _require_exact_keys(item, {"recipe_id", "attempt_subject_id"}, f"recipes[{index}]")
        normalized.append(
            {
                "recipe_id": _require_recipe_id(item["recipe_id"], f"recipes[{index}].recipe_id"),
                "attempt_subject_id": _require_sha256_id(
                    item["attempt_subject_id"], f"recipes[{index}].attempt_subject_id"
                ),
            }
        )
    normalized.sort(key=lambda item: (item["recipe_id"], item["attempt_subject_id"]))
    recipe_ids = [item["recipe_id"] for item in normalized]
    if len(recipe_ids) != len(set(recipe_ids)):
        raise ReceiptCoreError("receipt_core.recipes: duplicate recipe identity")

    out = bytearray()
    _frame(out, DOMAIN)
    _frame(out, SCHEMA.encode())
    for field in (
        "qualification_subject_id",
        "qualification_profile_id",
        "input_closure_id",
        "qualification_environment_id",
    ):
        _frame(out, _require_sha256_id(core[field], field).encode())
    out.extend(struct.pack("<Q", len(normalized)))
    for item in normalized:
        _frame(out, item["recipe_id"].encode())
        _frame(out, item["attempt_subject_id"].encode())
    _frame(out, SELECTION_DISPOSITION.encode())
    _frame(out, CROSS_CUTTING_DISPOSITION.encode())
    out.extend(struct.pack("<Q", len(NON_CLAIM_TAGS)))
    for tag in NON_CLAIM_TAGS:
        _frame(out, tag.encode())
    return bytes(out)


def receipt_id(core: dict[str, Any]) -> str:
    return "blake3:" + blake3_256(frame_core(core)).hex()


def _frame(out: bytearray, data: bytes) -> None:
    out.extend(struct.pack("<Q", len(data)))
    out.extend(data)


def _rotr32(value: int, amount: int) -> int:
    return ((value >> amount) | (value << (32 - amount))) & _MASK32


def _g(state: list[int], a: int, b: int, c: int, d: int, mx: int, my: int) -> None:
    state[a] = (state[a] + state[b] + mx) & _MASK32
    state[d] = _rotr32(state[d] ^ state[a], 16)
    state[c] = (state[c] + state[d]) & _MASK32
    state[b] = _rotr32(state[b] ^ state[c], 12)
    state[a] = (state[a] + state[b] + my) & _MASK32
    state[d] = _rotr32(state[d] ^ state[a], 8)
    state[c] = (state[c] + state[d]) & _MASK32
    state[b] = _rotr32(state[b] ^ state[c], 7)


def _round(state: list[int], message: list[int]) -> None:
    _g(state, 0, 4, 8, 12, message[0], message[1])
    _g(state, 1, 5, 9, 13, message[2], message[3])
    _g(state, 2, 6, 10, 14, message[4], message[5])
    _g(state, 3, 7, 11, 15, message[6], message[7])
    _g(state, 0, 5, 10, 15, message[8], message[9])
    _g(state, 1, 6, 11, 12, message[10], message[11])
    _g(state, 2, 7, 8, 13, message[12], message[13])
    _g(state, 3, 4, 9, 14, message[14], message[15])


def _compress(chaining_value: list[int], block_words: list[int], counter: int, block_len: int, flags: int) -> list[int]:
    state = list(chaining_value) + _IV[:4] + [
        counter & _MASK32,
        (counter >> 32) & _MASK32,
        block_len,
        flags,
    ]
    message = list(block_words)
    for round_index in range(7):
        _round(state, message)
        if round_index != 6:
            message = [message[index] for index in _MSG_PERMUTATION]
    for index in range(8):
        state[index] ^= state[index + 8]
        state[index + 8] ^= chaining_value[index]
    return [word & _MASK32 for word in state]


def _block_words(block: bytes) -> list[int]:
    return list(struct.unpack("<16I", block + b"\0" * (_BLOCK_LEN - len(block))))


class _Output:
    def __init__(self, input_cv: list[int], block_words: list[int], counter: int, block_len: int, flags: int):
        self.input_cv = list(input_cv)
        self.block_words = list(block_words)
        self.counter = counter
        self.block_len = block_len
        self.flags = flags

    def chaining_value(self) -> list[int]:
        return _compress(self.input_cv, self.block_words, self.counter, self.block_len, self.flags)[:8]

    def root_bytes(self, size: int) -> bytes:
        output = bytearray()
        output_block_counter = 0
        while len(output) < size:
            words = _compress(
                self.input_cv,
                self.block_words,
                output_block_counter,
                self.block_len,
                self.flags | _ROOT,
            )
            output.extend(struct.pack("<16I", *words))
            output_block_counter += 1
        return bytes(output[:size])


def _chunk_output(chunk: bytes, counter: int) -> _Output:
    cv = list(_IV)
    block_count = max(1, (len(chunk) + _BLOCK_LEN - 1) // _BLOCK_LEN)
    for block_index in range(block_count - 1):
        block = chunk[block_index * _BLOCK_LEN : (block_index + 1) * _BLOCK_LEN]
        flags = _CHUNK_START if block_index == 0 else 0
        cv = _compress(cv, _block_words(block), counter, _BLOCK_LEN, flags)[:8]
    last = chunk[(block_count - 1) * _BLOCK_LEN :]
    flags = _CHUNK_END | (_CHUNK_START if block_count == 1 else 0)
    return _Output(cv, _block_words(last), counter, len(last), flags)


def _parent_output(left_cv: list[int], right_cv: list[int]) -> _Output:
    return _Output(list(_IV), list(left_cv) + list(right_cv), 0, _BLOCK_LEN, _PARENT)


def blake3_256(data: bytes) -> bytes:
    """Unkeyed BLAKE3, 32-byte output. Reference implementation for conformance only."""
    chunk_count = max(1, (len(data) + _CHUNK_LEN - 1) // _CHUNK_LEN)
    stack: list[list[int]] = []
    for chunk_index in range(chunk_count - 1):
        chunk = data[chunk_index * _CHUNK_LEN : (chunk_index + 1) * _CHUNK_LEN]
        cv = _chunk_output(chunk, chunk_index).chaining_value()
        total_chunks = chunk_index + 1
        while total_chunks & 1 == 0:
            cv = _parent_output(stack.pop(), cv).chaining_value()
            total_chunks >>= 1
        stack.append(cv)
    final_chunk = data[(chunk_count - 1) * _CHUNK_LEN :]
    output = _chunk_output(final_chunk, chunk_count - 1)
    for left_cv in reversed(stack):
        output = _parent_output(left_cv, output.chaining_value())
    return output.root_bytes(32)
