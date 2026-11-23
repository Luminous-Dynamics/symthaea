# Copyright (C) 2026 Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Thin LongMemEval-V2 Memory API bridge for the Rust MACE BM25 sidecar.

This first arm retrieves text and returns the original screenshots attached to
selected trajectory states. It accepts query_image for API compatibility but
does not use query-image pixels for retrieval.
"""

from __future__ import annotations

import atexit
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import stat
import struct
import subprocess
import tempfile
import threading
from typing import Any

from memory_modules.memory import Memory, MemoryContextItem, register_memory, require


_ALLOWED_PARAMS = {
    "sidecar_binary",
    "source_screenshot_root",
    "workspace_dir",
    "domain",
    "run_id",
    "environment_revision_sha256",
    "top_k",
    "slice_radius",
    "neighbor_radius",
    "bm25_k1",
    "bm25_b",
}
_ALLOWED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}
MAX_SCREENSHOT_BYTES = 32 * 1024 * 1024
MAX_TRAJECTORY_SCREENSHOT_BYTES = 512 * 1024 * 1024


@register_memory
class MaceLongMemEvalV2Memory(Memory):
    """Official LongMemEval-V2 plugin backed by a persistent Rust sidecar."""

    memory_type = "mace_lmev2_bm25_text"

    def __init__(self, memory_params: dict[str, object]) -> None:
        require(isinstance(memory_params, dict), "MACE memory_params must be an object")
        unexpected = sorted(set(memory_params) - _ALLOWED_PARAMS)
        require(not unexpected, f"Unexpected MACE memory_params keys: {unexpected}")

        sidecar_value = memory_params.get("sidecar_binary")
        source_root_value = memory_params.get("source_screenshot_root")
        workspace_value = memory_params.get("workspace_dir")
        domain = memory_params.get("domain", "web")
        run_id = memory_params.get("run_id")
        environment_revision_sha256 = memory_params.get("environment_revision_sha256")
        require(isinstance(sidecar_value, str) and sidecar_value, "sidecar_binary is required")
        require(isinstance(source_root_value, str) and source_root_value, "source_screenshot_root is required")
        require(isinstance(workspace_value, str) and workspace_value, "workspace_dir is required")
        require(domain in {"web", "enterprise"}, "domain must be web or enterprise")
        require(
            isinstance(run_id, str) and run_id.strip()
            and not any(ord(character) < 32 or ord(character) == 127 for character in run_id),
            "run_id must be a non-empty printable identifier",
        )
        require(
            isinstance(environment_revision_sha256, str)
            and len(environment_revision_sha256) == 64
            and all(character in "0123456789abcdef" for character in environment_revision_sha256),
            "environment_revision_sha256 must be lowercase SHA-256",
        )

        self.sidecar_binary = Path(sidecar_value).expanduser().resolve(strict=True)
        require(self.sidecar_binary.is_file(), "sidecar_binary must be a file")
        require(os.access(self.sidecar_binary, os.X_OK), "sidecar_binary must be executable")
        self._sidecar_binary_stat = self.sidecar_binary.stat()
        require(stat.S_ISREG(self._sidecar_binary_stat.st_mode), "sidecar_binary must be a regular file")
        self.sidecar_binary_sha256 = self._sha256_file(self.sidecar_binary)

        self.source_screenshot_root = Path(source_root_value).expanduser().resolve(strict=True)
        require(self.source_screenshot_root.is_dir(), "source_screenshot_root must be a directory")
        require(
            hasattr(os, "O_NOFOLLOW") and hasattr(os, "O_DIRECTORY"),
            "secure screenshot ingress requires O_NOFOLLOW and O_DIRECTORY support",
        )
        root_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        if hasattr(os, "O_CLOEXEC"):
            root_flags |= os.O_CLOEXEC
        self._source_root_fd = os.open(self.source_screenshot_root, root_flags)
        require(
            stat.S_ISDIR(os.fstat(self._source_root_fd).st_mode),
            "pinned screenshot root descriptor is not a directory",
        )
        self.workspace_dir = Path(workspace_value).expanduser().resolve()
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        self.asset_root = self.workspace_dir / "assets"
        self.asset_root.mkdir(parents=True, exist_ok=True)
        self.candidate_evidence_dir = self.workspace_dir / "candidate_evidence"
        self.candidate_evidence_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.chmod(self.candidate_evidence_dir, 0o700)
        self.retrieval_trace_dir = self.workspace_dir / "retrieval_traces"
        self.retrieval_trace_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.chmod(self.retrieval_trace_dir, 0o700)

        self.domain = str(domain)
        self.run_id = run_id
        self.environment_revision_sha256 = environment_revision_sha256
        self.top_k = self._bounded_int(memory_params.get("top_k", 8), "top_k", 1, 100)
        self.slice_radius = self._bounded_int(memory_params.get("slice_radius", 1), "slice_radius", 0, 32)
        self.neighbor_radius = self._bounded_int(
            memory_params.get("neighbor_radius", 1), "neighbor_radius", 0, 32
        )
        self.bm25_k1 = self._bounded_float(memory_params.get("bm25_k1", 1.2), "bm25_k1", minimum=0.0)
        self.bm25_b = self._bounded_float(memory_params.get("bm25_b", 0.75), "bm25_b", minimum=0.0)
        require(self.bm25_k1 > 0.0, "bm25_k1 must be greater than zero")
        require(self.bm25_b <= 1.0, "bm25_b must be <= 1")

        self.backend_revision_sha256 = hashlib.sha256(
            json.dumps(
                {
                    "schema_version": "mace_lmev2_backend_profile_v1",
                    "memory_type": self.memory_type,
                    "domain": self.domain,
                    "sidecar_binary_sha256": self.sidecar_binary_sha256,
                    "top_k": self.top_k,
                    "slice_radius": self.slice_radius,
                    "neighbor_radius": self.neighbor_radius,
                    "bm25_k1": self.bm25_k1,
                    "bm25_b": self.bm25_b,
                    "tokenizer_version": "unicode_alphanumeric_v1",
                    "retrieval_mode": "text_bm25_with_original_image_evidence",
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()

        self._lock = threading.RLock()
        self._process: subprocess.Popen[str] | None = None
        self._poisoned = False
        self._closed = False
        self._trajectory_lines: list[str] = []
        self._inserted_ids: set[str] = set()
        super().__init__(memory_params)
        atexit.register(self.close)

    def _verify_sidecar_binary(self) -> None:
        current = self.sidecar_binary.stat()
        identity_before = (
            self._sidecar_binary_stat.st_dev,
            self._sidecar_binary_stat.st_ino,
            self._sidecar_binary_stat.st_size,
            self._sidecar_binary_stat.st_mtime_ns,
            self._sidecar_binary_stat.st_ctime_ns,
        )
        identity_now = (
            current.st_dev,
            current.st_ino,
            current.st_size,
            current.st_mtime_ns,
            current.st_ctime_ns,
        )
        require(
            stat.S_ISREG(current.st_mode) and identity_now == identity_before,
            "sidecar binary changed after its backend revision was pinned",
        )
        require(
            self._sha256_file(self.sidecar_binary) == self.sidecar_binary_sha256,
            "sidecar binary digest changed after backend revision pinning",
        )

    @staticmethod
    def _sha256_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _write_once(directory: Path, filename: str, payload: bytes) -> None:
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.chmod(directory, 0o700)
        directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        if hasattr(os, "O_CLOEXEC"):
            directory_flags |= os.O_CLOEXEC
        directory_fd = os.open(directory, directory_flags)
        file_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
        if hasattr(os, "O_CLOEXEC"):
            file_flags |= os.O_CLOEXEC
        try:
            try:
                file_fd = os.open(filename, file_flags, 0o600, dir_fd=directory_fd)
            except FileExistsError:
                read_flags = os.O_RDONLY | os.O_NOFOLLOW
                if hasattr(os, "O_CLOEXEC"):
                    read_flags |= os.O_CLOEXEC
                existing_fd = os.open(filename, read_flags, dir_fd=directory_fd)
                try:
                    metadata = os.fstat(existing_fd)
                    require(stat.S_ISREG(metadata.st_mode), "candidate receipt is not a regular file")
                    chunks: list[bytes] = []
                    while True:
                        chunk = os.read(existing_fd, 1024 * 1024)
                        if not chunk:
                            break
                        chunks.append(chunk)
                    require(
                        b"".join(chunks) == payload,
                        "candidate invocation ID was already bound to different evidence",
                    )
                finally:
                    os.close(existing_fd)
                return

            try:
                view = memoryview(payload)
                written = 0
                while written < len(view):
                    count = os.write(file_fd, view[written:])
                    require(count > 0, "short write while persisting candidate evidence")
                    written += count
                os.fsync(file_fd)
            finally:
                os.close(file_fd)
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)

    @staticmethod
    def _candidate_set_digest_from_records(records: list[dict[str, Any]]) -> str:
        require(bool(records), "candidate-set digest requires a non-empty record set")
        ordered = sorted(
            records,
            key=lambda record: (
                record["retrieval_channel"],
                record["retrieval_rank"],
                record["evidence_id"],
            ),
        )
        digest = hashlib.sha256()
        digest.update(b"luminous:mace:candidate-evidence-set:v1\0")

        def hash_field(value: str) -> None:
            encoded = value.encode("utf-8")
            digest.update(struct.pack(">Q", len(encoded)))
            digest.update(encoded)

        first = ordered[0]
        for field in (
            "run_id",
            "candidate_id",
            "backend_id",
            "backend_revision_sha256",
            "query_scope",
            "query_purpose",
            "query_sha256",
        ):
            hash_field(first[field])

        digest.update(struct.pack(">Q", len(ordered)))
        for record in ordered:
            for field in (
                "evidence_id",
                "source_id",
                "source_sha256",
                "payload_sha256",
                "payload_ref",
                "modality",
                "retrieval_channel",
            ):
                hash_field(record[field])
            digest.update(struct.pack(">I", record["retrieval_rank"]))
            score = record.get("retrieval_score")
            if score is None:
                digest.update(b"\x00")
            else:
                digest.update(b"\x01")
                digest.update(struct.pack(">d", float(score)))
            parents = sorted(record["parent_evidence_ids"])
            digest.update(struct.pack(">Q", len(parents)))
            for parent in parents:
                hash_field(parent)
            receipt = record.get("derivation_receipt_sha256")
            if receipt is None:
                digest.update(b"\x00")
            else:
                digest.update(b"\x01")
                hash_field(receipt)
        return digest.hexdigest()

    @staticmethod
    def _validate_candidate_evidence_graph(records: list[dict[str, Any]]) -> None:
        require(bool(records), "candidate evidence graph must be non-empty")
        common_fields = (
            "run_id",
            "candidate_id",
            "backend_id",
            "backend_revision_sha256",
            "query_scope",
            "query_purpose",
            "query_sha256",
        )
        first = records[0]
        ids: set[str] = set()
        ranks: set[tuple[str, int]] = set()
        for record in records:
            require(
                all(record.get(field) == first.get(field) for field in common_fields),
                "candidate evidence batch contains mixed identity",
            )
            evidence_id = record.get("evidence_id")
            require(isinstance(evidence_id, str) and evidence_id.strip(),
                    "candidate evidence ID is invalid")
            require(evidence_id not in ids, "candidate evidence contains duplicate IDs")
            ids.add(evidence_id)
            channel = record.get("retrieval_channel")
            rank = record.get("retrieval_rank")
            require(isinstance(channel, str) and channel.strip(),
                    "candidate retrieval channel is invalid")
            require(
                isinstance(rank, int) and not isinstance(rank, bool)
                and 0 < rank <= 0xFFFFFFFF,
                "candidate retrieval rank is invalid",
            )
            key = (channel, rank)
            require(key not in ranks, "candidate evidence contains duplicate channel/rank")
            ranks.add(key)
            parents = record.get("parent_evidence_ids")
            require(isinstance(parents, list) and all(
                isinstance(parent, str) and parent.strip() for parent in parents
            ), "candidate parent list is invalid")
            require(len(set(parents)) == len(parents), "candidate parent list contains duplicates")
            require(parents == sorted(parents), "candidate parent list is not canonical")
            receipt = record.get("derivation_receipt_sha256")
            if parents:
                require(
                    isinstance(receipt, str)
                    and len(receipt) == 64
                    and all(character in "0123456789abcdef" for character in receipt),
                    "derived candidate is missing a valid derivation receipt",
                )

        for record in records:
            for parent in record["parent_evidence_ids"]:
                require(parent in ids, "candidate evidence graph has an unresolved parent")

    @staticmethod
    def _retrieval_trace_digest_from_json(trace: dict[str, Any]) -> str:
        digest = hashlib.sha256()
        digest.update(b"luminous:mace:retrieval-trace:v1\0")

        def hash_field(value: str) -> None:
            encoded = value.encode("utf-8")
            digest.update(struct.pack(">Q", len(encoded)))
            digest.update(encoded)

        for field in (
            "run_id",
            "query_invocation_id",
            "domain_id",
            "environment_revision_sha256",
            "tool_profile_sha256",
            "query_sha256",
        ):
            value = trace.get(field)
            require(isinstance(value, str), f"retrieval trace {field} must be a string")
            hash_field(value)

        candidate_set = trace.get("candidate_set_sha256")
        if candidate_set is None:
            digest.update(b"\x00")
        else:
            require(
                isinstance(candidate_set, str)
                and len(candidate_set) == 64
                and all(character in "0123456789abcdef" for character in candidate_set),
                "retrieval trace candidate digest must be lowercase SHA-256",
            )
            digest.update(b"\x01")
            hash_field(candidate_set)

        stop_tags = {
            "evidence_sufficient": 1,
            "budget_exhausted": 2,
            "no_candidate_evidence": 3,
            "tool_error": 4,
            "abstained": 5,
        }
        stop_reason = trace.get("stop_reason")
        require(stop_reason in stop_tags, "retrieval trace stop reason is invalid")
        digest.update(struct.pack(">B", stop_tags[stop_reason]))

        def integer_field(field: str, width: str = ">Q") -> int:
            value = trace.get(field)
            require(
                isinstance(value, int) and not isinstance(value, bool)
                and 0 <= value <= (0xFFFFFFFF if width == ">I" else 0xFFFFFFFFFFFFFFFF),
                f"retrieval trace {field} is invalid",
            )
            digest.update(struct.pack(width, value))
            return value

        budget = integer_field("tool_call_budget", ">I")
        used = integer_field("tool_calls_used", ">I")
        require(budget > 0 and used <= budget, "retrieval trace tool budget is invalid")
        integer_field("elapsed_ns")
        integer_field("inspected_bytes")
        complete = trace.get("inspected_bytes_complete")
        require(isinstance(complete, bool), "retrieval trace byte-accounting flag must be boolean")
        digest.update(b"\x01" if complete else b"\x00")
        integer_field("created_at_unix")

        operations = trace.get("operations")
        require(isinstance(operations, list) and len(operations) == used,
                "retrieval trace operation count mismatch")
        digest.update(struct.pack(">Q", len(operations)))
        outcome_tags = {"evidence_observed": 1, "no_evidence": 2, "tool_error": 3}
        all_bytes_measured = True
        elapsed_sum = 0
        bytes_sum = 0
        observed_any = False
        last_outcome: str | None = None
        for operation in operations:
            require(isinstance(operation, dict), "retrieval trace operation must be an object")
            for field in ("tool_id", "tool_revision_sha256", "action_template_sha256"):
                value = operation.get(field)
                require(isinstance(value, str), f"retrieval operation {field} must be a string")
                hash_field(value)
            outcome = operation.get("outcome")
            require(outcome in outcome_tags, "retrieval operation outcome is invalid")
            digest.update(struct.pack(">B", outcome_tags[outcome]))
            elapsed = operation.get("elapsed_ns")
            inspected = operation.get("inspected_bytes")
            require(isinstance(elapsed, int) and not isinstance(elapsed, bool) and elapsed >= 0,
                    "retrieval operation elapsed_ns is invalid")
            require(isinstance(inspected, int) and not isinstance(inspected, bool) and inspected >= 0,
                    "retrieval operation inspected_bytes is invalid")
            digest.update(struct.pack(">Q", elapsed))
            digest.update(struct.pack(">Q", inspected))
            elapsed_sum += elapsed
            bytes_sum += inspected
            measured = operation.get("inspected_bytes_measured")
            require(isinstance(measured, bool), "retrieval operation byte-accounting flag is invalid")
            all_bytes_measured = all_bytes_measured and measured
            digest.update(b"\x01" if measured else b"\x00")
            evidence_ids = operation.get("observed_evidence_ids")
            require(isinstance(evidence_ids, list), "retrieval operation evidence IDs must be a list")
            require(all(isinstance(item, str) for item in evidence_ids),
                    "retrieval operation evidence ID must be a string")
            if outcome == "evidence_observed":
                require(bool(evidence_ids), "evidence_observed operation must name evidence IDs")
                observed_any = True
            else:
                require(not evidence_ids, "non-evidence operation cannot name observed evidence IDs")
            last_outcome = outcome
            ordered_ids = sorted(evidence_ids)
            digest.update(struct.pack(">Q", len(ordered_ids)))
            for evidence_id in ordered_ids:
                hash_field(evidence_id)

        require(elapsed_sum == trace["elapsed_ns"], "retrieval trace elapsed accounting mismatch")
        require(bytes_sum == trace["inspected_bytes"], "retrieval trace byte accounting mismatch")
        require(all_bytes_measured == complete, "retrieval trace byte completeness mismatch")
        if stop_reason == "evidence_sufficient":
            require(candidate_set is not None and observed_any,
                    "evidence_sufficient trace must bind a non-empty candidate set")
        elif stop_reason == "no_candidate_evidence":
            require(candidate_set is None and not observed_any,
                    "no_candidate_evidence trace cannot contain candidates")
        elif stop_reason == "budget_exhausted":
            require(used == budget, "budget_exhausted trace did not use its full budget")
        elif stop_reason == "tool_error":
            require(last_outcome == "tool_error", "tool_error trace must end in a tool error")
        return digest.hexdigest()

    def _persist_retrieval_trace(
        self,
        candidate_id: str,
        query: str,
        response: dict[str, Any],
    ) -> None:
        trace = response.get("retrieval_trace")
        trace_digest = response.get("retrieval_trace_sha256")
        require(isinstance(trace, dict), "sidecar retrieval_trace must be an object")
        require(trace.get("schema_version") == "mace_retrieval_trace_receipt_v1",
                "retrieval trace schema mismatch")
        query_sha256 = hashlib.sha256(query.encode("utf-8")).hexdigest()
        require(trace.get("run_id") == self.run_id, "retrieval trace run mismatch")
        require(trace.get("query_invocation_id") == candidate_id,
                "retrieval trace invocation ID mismatch")
        require(trace.get("domain_id") == self.domain, "retrieval trace domain mismatch")
        require(
            trace.get("environment_revision_sha256") == self.environment_revision_sha256,
            "retrieval trace environment revision mismatch",
        )
        require(trace.get("tool_profile_sha256") == self.backend_revision_sha256,
                "retrieval trace tool-profile mismatch")
        require(trace.get("query_sha256") == query_sha256, "retrieval trace query digest mismatch")
        require(
            trace.get("candidate_set_sha256") == response.get("candidate_set_sha256"),
            "retrieval trace candidate-set digest mismatch",
        )
        require(
            isinstance(trace_digest, str)
            and len(trace_digest) == 64
            and all(character in "0123456789abcdef" for character in trace_digest),
            "retrieval trace digest must be lowercase SHA-256",
        )
        recomputed = self._retrieval_trace_digest_from_json(trace)
        require(recomputed == trace_digest == trace.get("receipt_sha256"),
                "retrieval trace receipt digest mismatch")

        invocation_digest = hashlib.sha256(candidate_id.encode("utf-8")).hexdigest()
        trace_bytes = json.dumps(
            trace, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        self._write_once(
            self.retrieval_trace_dir,
            f"{invocation_digest}.trace.json",
            trace_bytes,
        )

    def _verify_persisted_retrieval_traces(self) -> None:
        directory = self.retrieval_trace_dir
        metadata = directory.lstat()
        require(stat.S_ISDIR(metadata.st_mode), "retrieval trace root must be a real directory")
        os.chmod(directory, 0o700)
        trace_files: set[str] = set()
        for path in directory.iterdir():
            entry = path.lstat()
            require(not stat.S_ISLNK(entry.st_mode), "retrieval trace directory contains a symlink")
            require(stat.S_ISREG(entry.st_mode), "retrieval trace directory contains a non-file entry")
            require(path.name.endswith(".trace.json"), "retrieval trace directory contains an unknown file")
            try:
                trace = json.loads(path.read_bytes())
            except (json.JSONDecodeError, UnicodeDecodeError) as error:
                raise RuntimeError(f"saved retrieval trace is invalid JSON: {path.name}") from error
            require(isinstance(trace, dict), "saved retrieval trace must be an object")
            require(trace.get("schema_version") == "mace_retrieval_trace_receipt_v1",
                    "saved retrieval trace schema mismatch")
            candidate_id = trace.get("query_invocation_id")
            require(isinstance(candidate_id, str) and candidate_id.strip(),
                    "saved retrieval trace has no invocation ID")
            invocation_digest = hashlib.sha256(candidate_id.encode("utf-8")).hexdigest()
            require(path.name == f"{invocation_digest}.trace.json",
                    "saved retrieval trace filename does not match invocation ID")
            require(trace.get("run_id") == self.run_id, "saved retrieval trace run mismatch")
            require(trace.get("domain_id") == self.domain, "saved retrieval trace domain mismatch")
            require(trace.get("environment_revision_sha256") == self.environment_revision_sha256,
                    "saved retrieval trace environment mismatch")
            require(trace.get("tool_profile_sha256") == self.backend_revision_sha256,
                    "saved retrieval trace tool-profile mismatch")
            require(
                isinstance(trace.get("query_sha256"), str)
                and len(trace["query_sha256"]) == 64
                and all(character in "0123456789abcdef" for character in trace["query_sha256"]),
                "saved retrieval trace query digest is invalid",
            )
            require(
                self._retrieval_trace_digest_from_json(trace) == trace.get("receipt_sha256"),
                "saved retrieval trace receipt digest mismatch",
            )
            candidate_receipt_path = (
                self.candidate_evidence_dir / f"{invocation_digest}.receipt.json"
            )
            require(candidate_receipt_path.is_file(),
                    "saved retrieval trace has no paired candidate receipt")
            candidate_receipt = json.loads(candidate_receipt_path.read_text(encoding="utf-8"))
            require(candidate_receipt.get("candidate_id") == candidate_id,
                    "saved retrieval trace candidate receipt ID mismatch")
            require(candidate_receipt.get("query_sha256") == trace.get("query_sha256"),
                    "saved retrieval trace query digest mismatch")
            require(candidate_receipt.get("backend_revision_sha256") == trace.get("tool_profile_sha256"),
                    "saved retrieval trace execution-profile mismatch")
            require(candidate_receipt.get("candidate_set_sha256") == trace.get("candidate_set_sha256"),
                    "saved retrieval trace candidate-set mismatch")
            trace_files.add(path.name)

        candidate_receipts = {
            path.name.replace(".receipt.json", ".trace.json")
            for path in self.candidate_evidence_dir.iterdir()
            if path.name.endswith(".receipt.json")
        }
        require(trace_files == candidate_receipts,
                "retrieval trace receipts and candidate receipts are not one-to-one")

    def _persist_candidate_envelope(
        self,
        candidate_id: str,
        query: str,
        response: dict[str, Any],
    ) -> None:
        candidate_jsonl = response.get("candidate_evidence_jsonl")
        record_count = response.get("candidate_record_count")
        candidate_set_sha256 = response.get("candidate_set_sha256")
        require(
            isinstance(candidate_jsonl, str),
            "sidecar response candidate_evidence_jsonl must be a string",
        )
        require(
            isinstance(record_count, int) and not isinstance(record_count, bool) and record_count >= 0,
            "sidecar response candidate_record_count must be a non-negative integer",
        )

        query_sha256 = hashlib.sha256(query.encode("utf-8")).hexdigest()
        lines = [line for line in candidate_jsonl.splitlines() if line.strip()]
        require(len(lines) == record_count, "candidate record count does not match JSONL")
        parsed_records: list[dict[str, Any]] = []
        if record_count == 0:
            require(
                candidate_set_sha256 is None and not candidate_jsonl,
                "empty candidate result must not carry a candidate-set digest",
            )
        else:
            require(
                isinstance(candidate_set_sha256, str)
                and len(candidate_set_sha256) == 64
                and all(character in "0123456789abcdef" for character in candidate_set_sha256),
                "candidate-set digest must be lowercase SHA-256",
            )
            for ordinal, line in enumerate(lines):
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise RuntimeError(f"invalid candidate JSONL record {ordinal}: {error}") from error
                require(isinstance(record, dict), f"candidate JSONL record {ordinal} must be an object")
                parsed_records.append(record)
                require(
                    record.get("schema_version") == "mace_candidate_evidence_v0_1"
                    and record.get("record_type") == "candidate_evidence",
                    "candidate record schema/type mismatch",
                )
                require(record.get("modality") in {"text", "image", "audio", "video", "structured"},
                        "candidate modality is invalid")
                require(
                    isinstance(record.get("evidence_id"), str) and record["evidence_id"].strip()
                    and isinstance(record.get("source_id"), str) and record["source_id"].strip()
                    and isinstance(record.get("payload_ref"), str) and record["payload_ref"].strip(),
                    "candidate evidence identity/reference is invalid",
                )
                for digest_field in ("backend_revision_sha256", "query_sha256", "source_sha256", "payload_sha256"):
                    digest_value = record.get(digest_field)
                    require(
                        isinstance(digest_value, str)
                        and len(digest_value) == 64
                        and all(character in "0123456789abcdef" for character in digest_value),
                        f"candidate {digest_field} is invalid",
                    )
                rank = record.get("retrieval_rank")
                require(isinstance(rank, int) and not isinstance(rank, bool) and rank > 0,
                        "candidate retrieval_rank is invalid")
                score = record.get("retrieval_score")
                require(
                    score is None
                    or (isinstance(score, (int, float)) and not isinstance(score, bool)
                        and math.isfinite(float(score))),
                    "candidate retrieval_score is invalid",
                )
                require(record.get("run_id") == self.run_id, "candidate run_id mismatch")
                require(record.get("candidate_id") == candidate_id, "candidate invocation ID mismatch")
                require(
                    record.get("backend_id") == "mace_lmev2_bm25_text",
                    "candidate backend ID mismatch",
                )
                require(
                    record.get("backend_revision_sha256") == self.backend_revision_sha256,
                    "candidate backend execution-profile digest mismatch",
                )
                require(record.get("query_sha256") == query_sha256, "candidate query digest mismatch")
                require(
                    record.get("query_scope") == f"longmemeval-v2:domain:{self.domain}",
                    "candidate query scope mismatch",
                )
                require(
                    record.get("query_purpose") == "benchmark:memory-context",
                    "candidate query purpose mismatch",
                )
            self._validate_candidate_evidence_graph(parsed_records)
            recomputed_candidate_set_sha256 = self._candidate_set_digest_from_records(parsed_records)
            require(
                recomputed_candidate_set_sha256 == candidate_set_sha256,
                "candidate-set digest mismatch",
            )

        invocation_digest = hashlib.sha256(candidate_id.encode("utf-8")).hexdigest()
        candidate_filename = f"{invocation_digest}.jsonl"
        candidate_bytes = candidate_jsonl.encode("utf-8")
        receipt = {
            "schema_version": "mace_lmev2_candidate_receipt_v1",
            "run_id": self.run_id,
            "candidate_id": candidate_id,
            "query_sha256": query_sha256,
            "domain": self.domain,
            "backend_id": "mace_lmev2_bm25_text",
            "backend_revision_sha256": self.backend_revision_sha256,
            "sidecar_binary_sha256": self.sidecar_binary_sha256,
            "candidate_record_count": record_count,
            "candidate_set_sha256": candidate_set_sha256,
            "candidate_jsonl_sha256": hashlib.sha256(candidate_bytes).hexdigest(),
            "candidate_file": candidate_filename,
            "retrieval_mode": "text_bm25_with_original_image_evidence",
        }
        receipt_bytes = json.dumps(
            receipt, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        self._write_once(self.candidate_evidence_dir, candidate_filename, candidate_bytes)
        self._write_once(
            self.candidate_evidence_dir,
            f"{invocation_digest}.receipt.json",
            receipt_bytes,
        )

    def _verify_persisted_candidate_evidence(self) -> None:
        """Validate saved candidate files against their run-bound local receipts.

        This detects partial writes and accidental/tampering edits to one side
        of the pair. The receipt and payload share a workspace and are not an
        independent signature or protection against an attacker rewriting both.
        """
        directory = self.candidate_evidence_dir
        metadata = directory.lstat()
        require(stat.S_ISDIR(metadata.st_mode), "candidate evidence root must be a real directory")
        os.chmod(directory, 0o700)

        receipts: dict[str, dict[str, Any]] = {}
        expected_candidate_files: set[str] = set()
        for path in directory.iterdir():
            entry = path.lstat()
            require(not stat.S_ISLNK(entry.st_mode), "candidate evidence directory contains a symlink")
            require(stat.S_ISREG(entry.st_mode), "candidate evidence directory contains a non-file entry")
            if path.name.endswith(".jsonl"):
                continue
            require(path.name.endswith(".receipt.json"), "candidate evidence directory contains an unknown file")
            receipt_bytes = path.read_bytes()
            try:
                receipt = json.loads(receipt_bytes)
            except json.JSONDecodeError as error:
                raise RuntimeError(f"saved candidate receipt is invalid JSON: {path.name}") from error
            require(isinstance(receipt, dict), "saved candidate receipt must be an object")
            candidate_id = receipt.get("candidate_id")
            require(isinstance(candidate_id, str) and candidate_id.strip(),
                    "saved candidate receipt has no candidate_id")
            invocation_digest = hashlib.sha256(candidate_id.encode("utf-8")).hexdigest()
            expected_receipt_name = f"{invocation_digest}.receipt.json"
            expected_payload_name = f"{invocation_digest}.jsonl"
            require(path.name == expected_receipt_name, "saved candidate receipt filename does not match ID")
            require(candidate_id not in receipts, "duplicate saved candidate invocation receipt")
            require(receipt.get("schema_version") == "mace_lmev2_candidate_receipt_v1",
                    "saved candidate receipt schema mismatch")
            require(receipt.get("run_id") == self.run_id, "saved candidate receipt run mismatch")
            require(receipt.get("domain") == self.domain, "saved candidate receipt domain mismatch")
            require(receipt.get("backend_id") == "mace_lmev2_bm25_text",
                    "saved candidate receipt backend mismatch")
            require(receipt.get("backend_revision_sha256") == self.backend_revision_sha256,
                    "saved candidate receipt execution-profile digest mismatch")
            require(receipt.get("sidecar_binary_sha256") == self.sidecar_binary_sha256,
                    "saved candidate receipt binary digest mismatch")
            require(receipt.get("candidate_file") == expected_payload_name,
                    "saved candidate receipt payload filename mismatch")
            require(receipt.get("retrieval_mode") == "text_bm25_with_original_image_evidence",
                    "saved candidate receipt retrieval mode mismatch")
            require(
                isinstance(receipt.get("query_sha256"), str)
                and len(receipt["query_sha256"]) == 64
                and all(character in "0123456789abcdef" for character in receipt["query_sha256"]),
                "saved candidate query digest is invalid",
            )
            require(
                isinstance(receipt.get("candidate_jsonl_sha256"), str)
                and len(receipt["candidate_jsonl_sha256"]) == 64
                and all(character in "0123456789abcdef" for character in receipt["candidate_jsonl_sha256"]),
                "saved candidate payload digest is invalid",
            )
            count = receipt.get("candidate_record_count")
            require(isinstance(count, int) and not isinstance(count, bool) and count >= 0,
                    "saved candidate record count is invalid")
            candidate_set_sha256 = receipt.get("candidate_set_sha256")
            if count == 0:
                require(candidate_set_sha256 is None, "empty saved candidate set must not have a digest")
            else:
                require(
                    isinstance(candidate_set_sha256, str)
                    and len(candidate_set_sha256) == 64
                    and all(character in "0123456789abcdef" for character in candidate_set_sha256),
                    "saved candidate-set digest is invalid",
                )

            payload_path = directory / expected_payload_name
            try:
                payload_stat = payload_path.lstat()
            except FileNotFoundError as error:
                raise RuntimeError("saved candidate receipt refers to a missing payload") from error
            require(stat.S_ISREG(payload_stat.st_mode), "saved candidate payload is not a regular file")
            payload = payload_path.read_bytes()
            require(
                hashlib.sha256(payload).hexdigest() == receipt["candidate_jsonl_sha256"],
                "saved candidate JSONL digest mismatch",
            )
            lines = [line for line in payload.decode("utf-8").splitlines() if line.strip()]
            require(len(lines) == count, "saved candidate JSONL count mismatch")
            saved_records: list[dict[str, Any]] = []
            for ordinal, line in enumerate(lines):
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise RuntimeError(f"saved candidate JSONL record {ordinal} is invalid") from error
                require(isinstance(record, dict), "saved candidate JSONL record must be an object")
                saved_records.append(record)
                require(record.get("schema_version") == "mace_candidate_evidence_v0_1"
                        and record.get("record_type") == "candidate_evidence",
                        "saved candidate record schema/type mismatch")
                require(record.get("run_id") == self.run_id, "saved candidate record run mismatch")
                require(record.get("candidate_id") == candidate_id, "saved candidate record ID mismatch")
                require(record.get("query_sha256") == receipt["query_sha256"],
                        "saved candidate record query digest mismatch")
                require(record.get("backend_revision_sha256") == self.backend_revision_sha256,
                        "saved candidate record execution-profile digest mismatch")
                require(record.get("backend_id") == "mace_lmev2_bm25_text",
                        "saved candidate record backend mismatch")
                require(record.get("query_scope") == f"longmemeval-v2:domain:{self.domain}",
                        "saved candidate record scope mismatch")
                require(record.get("query_purpose") == "benchmark:memory-context",
                        "saved candidate record purpose mismatch")
            if count:
                self._validate_candidate_evidence_graph(saved_records)
                require(
                    self._candidate_set_digest_from_records(saved_records) == candidate_set_sha256,
                    "saved candidate-set digest mismatch",
                )
            receipts[candidate_id] = receipt
            expected_candidate_files.add(expected_payload_name)

        actual_candidate_files = {
            path.name for path in directory.iterdir() if path.name.endswith(".jsonl")
        }
        require(
            actual_candidate_files == expected_candidate_files,
            "candidate evidence directory contains orphan or unreferenced payloads",
        )

    @staticmethod
    def _bounded_int(value: object, name: str, minimum: int, maximum: int) -> int:
        require(isinstance(value, int) and not isinstance(value, bool), f"{name} must be an integer")
        require(minimum <= value <= maximum, f"{name} must be in {minimum}..={maximum}")
        return value

    @staticmethod
    def _bounded_float(value: object, name: str, minimum: float) -> float:
        require(
            isinstance(value, (int, float)) and not isinstance(value, bool),
            f"{name} must be numeric",
        )
        normalized = float(value)
        require(normalized == normalized and abs(normalized) != float("inf"), f"{name} must be finite")
        require(normalized >= minimum, f"{name} must be >= {minimum}")
        return normalized

    def _start_sidecar(self) -> None:
        if self._closed:
            raise RuntimeError("MACE sidecar has been closed")
        if self._poisoned:
            raise RuntimeError(
                "MACE sidecar outcome is unknown; refusing further operations until reload/reconciliation"
            )
        if self._process is not None:
            if self._process.poll() is None:
                return
            raise RuntimeError("MACE Rust sidecar exited unexpectedly; refusing silent restart")

        self._verify_sidecar_binary()
        env = os.environ.copy()
        env.update(
            {
                "MACE_LMEV2_DOMAIN": self.domain,
                "MACE_LMEV2_RUN_ID": self.run_id,
                "MACE_LMEV2_ENVIRONMENT_REVISION_SHA256": self.environment_revision_sha256,
                "MACE_LMEV2_BACKEND_REVISION_SHA256": self.backend_revision_sha256,
                "MACE_LMEV2_POLICY_TRACE_DIR": str(self.retrieval_trace_dir.resolve()),
                "MACE_LMEV2_DATASET_ROOT": str(self.asset_root.resolve()),
                "MACE_LMEV2_TOP_K": str(self.top_k),
                "MACE_LMEV2_SLICE_RADIUS": str(self.slice_radius),
                "MACE_LMEV2_NEIGHBOR_RADIUS": str(self.neighbor_radius),
                "MACE_LMEV2_BM25_K1": str(self.bm25_k1),
                "MACE_LMEV2_BM25_B": str(self.bm25_b),
            }
        )
        self._process = subprocess.Popen(
            [str(self.sidecar_binary)],
            cwd=str(self.workspace_dir),
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )

    def _poison_sidecar(self) -> None:
        """Stop an ambiguous sidecar and forbid silent reuse or restart."""
        self._poisoned = True
        process = self._process
        if process is None or process.poll() is not None:
            return
        try:
            process.terminate()
            process.wait(timeout=2)
        except (OSError, subprocess.TimeoutExpired):
            try:
                process.kill()
                process.wait(timeout=2)
            except (OSError, subprocess.TimeoutExpired):
                pass

    def _rpc(self, request: dict[str, object]) -> dict[str, Any]:
        with self._lock:
            self._start_sidecar()
            assert self._process is not None
            assert self._process.stdin is not None
            assert self._process.stdout is not None
            if self._process.poll() is not None:
                self._poisoned = True
                raise RuntimeError("MACE Rust sidecar is not running")
            try:
                self._process.stdin.write(
                    json.dumps(request, ensure_ascii=False, separators=(",", ":")) + "\n"
                )
                self._process.stdin.flush()
                response_line = self._process.stdout.readline()
            except (OSError, ValueError, UnicodeError) as error:
                self._poison_sidecar()
                raise RuntimeError(
                    "MACE sidecar RPC outcome is unknown; refusing further operations"
                ) from error
            if not response_line:
                self._poison_sidecar()
                raise RuntimeError(
                    "MACE sidecar closed its response stream; operation outcome is unknown"
                )
            try:
                response = json.loads(response_line)
            except json.JSONDecodeError as error:
                self._poison_sidecar()
                raise RuntimeError(
                    f"MACE Rust sidecar returned invalid JSON; outcome is unknown: {error}"
                ) from error
            if isinstance(response, dict) and response.get("ok") is False:
                message = response.get("error", "request rejected")
                if isinstance(message, str) and "outcome may be unknown" in message:
                    self._poison_sidecar()
                    raise RuntimeError(
                        "MACE sidecar mutation outcome is unknown; process poisoned until reload/reconciliation"
                    )
                raise RuntimeError(f"MACE Rust sidecar rejected request: {message}")
            if not isinstance(response, dict) or response.get("ok") is not True:
                self._poison_sidecar()
                raise RuntimeError("MACE Rust sidecar returned an invalid acknowledgement")
            return response

    def _resolve_source_screenshot(self, reference: str) -> Path:
        require(isinstance(reference, str) and reference.strip(), "screenshot reference must be non-empty")
        raw = Path(reference)
        candidates = [raw] if raw.is_absolute() else [
            self.source_screenshot_root / raw,
            self.source_screenshot_root / "screenshots" / raw,
        ]
        for candidate in candidates:
            try:
                resolved = candidate.resolve(strict=True)
            except (OSError, RuntimeError):
                continue
            require(
                resolved.is_relative_to(self.source_screenshot_root),
                f"Screenshot resolves outside source_screenshot_root: {reference}",
            )
            require(resolved.is_file(), f"Screenshot is not a regular file: {reference}")
            return resolved
        raise RuntimeError(f"Could not resolve screenshot under configured source root: {reference}")

    def _read_source_screenshot(self, resolved: Path) -> bytes:
        """Read an image through the pinned root fd without following swapped symlinks."""
        try:
            relative = resolved.relative_to(self.source_screenshot_root)
        except ValueError as error:
            raise RuntimeError("Resolved screenshot is outside the pinned source root") from error
        parts = relative.parts
        require(
            bool(parts) and all(part not in {"", ".", ".."} for part in parts),
            "Resolved screenshot has unsafe path components",
        )
        require(
            getattr(self, "_source_root_fd", -1) >= 0,
            "Pinned screenshot root descriptor is unavailable",
        )

        directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        file_flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_NONBLOCK", 0)
        if hasattr(os, "O_CLOEXEC"):
            directory_flags |= os.O_CLOEXEC
            file_flags |= os.O_CLOEXEC

        opened_directories: list[int] = []
        current_directory_fd = self._source_root_fd
        try:
            for component in parts[:-1]:
                next_fd = os.open(
                    component,
                    directory_flags,
                    dir_fd=current_directory_fd,
                )
                opened_directories.append(next_fd)
                current_directory_fd = next_fd

            image_fd = os.open(
                parts[-1],
                file_flags,
                dir_fd=current_directory_fd,
            )
            try:
                before = os.fstat(image_fd)
                require(stat.S_ISREG(before.st_mode), "Screenshot is not a regular file")
                require(
                    before.st_size <= MAX_SCREENSHOT_BYTES,
                    f"Screenshot exceeds {MAX_SCREENSHOT_BYTES} bytes",
                )

                chunks: list[bytes] = []
                total = 0
                while total <= MAX_SCREENSHOT_BYTES:
                    chunk = os.read(
                        image_fd,
                        min(1024 * 1024, MAX_SCREENSHOT_BYTES + 1 - total),
                    )
                    if not chunk:
                        break
                    chunks.append(chunk)
                    total += len(chunk)
                require(
                    total <= MAX_SCREENSHOT_BYTES,
                    f"Screenshot exceeds {MAX_SCREENSHOT_BYTES} bytes",
                )

                after = os.fstat(image_fd)
                before_identity = (
                    before.st_dev,
                    before.st_ino,
                    before.st_size,
                    before.st_mtime_ns,
                    before.st_ctime_ns,
                )
                after_identity = (
                    after.st_dev,
                    after.st_ino,
                    after.st_size,
                    after.st_mtime_ns,
                    after.st_ctime_ns,
                )
                require(
                    before_identity == after_identity and total == before.st_size,
                    "Screenshot changed while it was being read",
                )
                return b"".join(chunks)
            finally:
                os.close(image_fd)
        except OSError as error:
            raise RuntimeError(
                "Screenshot path changed or could not be opened safely without following links"
            ) from error
        finally:
            for directory_fd in reversed(opened_directories):
                os.close(directory_fd)

    def _prepare_trajectory(self, trajectory: dict[str, object]) -> tuple[str, str, str]:
        require(isinstance(trajectory, dict), "trajectory must be an object")
        trajectory_id = trajectory.get("id")
        states_value = trajectory.get("states")
        require(isinstance(trajectory_id, str) and trajectory_id.strip(), "trajectory id is required")
        require(isinstance(states_value, list) and states_value, "trajectory states must be a non-empty list")
        require(
            "domain" not in trajectory or trajectory.get("domain") == self.domain,
            "trajectory domain does not match configured domain",
        )

        canonical_original = json.dumps(
            trajectory, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        original_digest = hashlib.sha256(canonical_original).hexdigest()
        trajectory_key = hashlib.sha256(trajectory_id.encode("utf-8")).hexdigest()
        staging_parent = self.asset_root / ".staging"
        staging_parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=trajectory_key[:16] + "-", dir=staging_parent))
        destination = self.asset_root / trajectory_key
        if destination.exists():
            shutil.rmtree(staging, ignore_errors=True)
            raise RuntimeError(f"Duplicate trajectory asset directory: {trajectory_id}")

        try:
            normalized = json.loads(json.dumps(trajectory, ensure_ascii=False))
            normalized["domain"] = self.domain
            normalized["mace_original_trajectory_sha256"] = original_digest
            normalized["mace_original_trajectory_json"] = canonical_original.decode("utf-8")
            normalized_states: list[dict[str, object]] = []
            total_screenshot_bytes = 0
            for ordinal, original_state in enumerate(states_value):
                require(isinstance(original_state, dict), f"trajectory state {ordinal} must be an object")
                state = dict(original_state)
                screenshot_value = state.get("screenshot")
                require(
                    isinstance(screenshot_value, str) and screenshot_value.strip(),
                    f"trajectory state {ordinal} requires a screenshot reference",
                )
                source = self._resolve_source_screenshot(screenshot_value)
                image_bytes = self._read_source_screenshot(source)
                total_screenshot_bytes += len(image_bytes)
                require(
                    total_screenshot_bytes <= MAX_TRAJECTORY_SCREENSHOT_BYTES,
                    f"trajectory screenshots exceed {MAX_TRAJECTORY_SCREENSHOT_BYTES} bytes",
                )
                image_digest = hashlib.sha256(image_bytes).hexdigest()
                suffix = source.suffix.lower()
                if suffix not in _ALLOWED_IMAGE_SUFFIXES:
                    suffix = ".img"
                filename = f"state_{ordinal:06d}{suffix}"
                staged_file = staging / filename
                staged_file.write_bytes(image_bytes)
                state["screenshot"] = f"{trajectory_key}/{filename}"
                state["mace_screenshot_sha256"] = image_digest
                normalized_states.append(state)
            normalized["states"] = normalized_states
            serialized = json.dumps(
                normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            )
            os.replace(staging, destination)
            return trajectory_id, trajectory_key, serialized
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise

    def insert(self, trajectory: dict[str, object]) -> None:
        with self._lock:
            trajectory_id, trajectory_key, serialized = self._prepare_trajectory(trajectory)
            if trajectory_id in self._inserted_ids:
                shutil.rmtree(self.asset_root / trajectory_key, ignore_errors=True)
                raise RuntimeError(f"Duplicate trajectory insert attempted: {trajectory_id}")
            try:
                acknowledgement = self._rpc({"op": "insert", "trajectory": json.loads(serialized)})
                if acknowledgement.get("inserted") != 1:
                    self._poison_sidecar()
                    raise RuntimeError("MACE sidecar insert acknowledgement is not exact")
            except Exception:
                shutil.rmtree(self.asset_root / trajectory_key, ignore_errors=True)
                raise
            self._inserted_ids.add(trajectory_id)
            self._trajectory_lines.append(serialized)

    def query(
        self,
        query: str,
        query_image: str | None = None,
    ) -> list[MemoryContextItem]:
        require(isinstance(query, str) and query.strip(), "query must be non-empty")
        query_context = self.get_query_context()
        candidate_id = query_context.get("query_invocation_id")
        require(
            isinstance(candidate_id, str) and candidate_id.strip(),
            "official LongMemEval-V2 query_invocation_id is required for evidence binding",
        )
        response = self._rpc(
            {
                "op": "query",
                "query": query,
                "query_image": query_image,
                "candidate_id": candidate_id,
            }
        )
        try:
            self._persist_candidate_envelope(candidate_id, query, response)
            self._persist_retrieval_trace(candidate_id, query, response)
        except Exception:
            # The sidecar may already have durably recorded the trace. Do not
            # continue serving queries on a partially reconciled invocation.
            self._poison_sidecar()
            raise
        items = response.get("items")
        if not isinstance(items, list):
            self._poison_sidecar()
            raise RuntimeError("MACE sidecar query acknowledgement has no item list")
        validated: list[MemoryContextItem] = []
        for index, item in enumerate(items):
            require(isinstance(item, dict), f"sidecar context item {index} must be an object")
            item_type = item.get("type")
            value = item.get("value")
            require(item_type in {"text", "image"}, f"invalid sidecar context type at {index}")
            require(isinstance(value, str) and value.strip(), f"invalid sidecar context value at {index}")
            if item_type == "image":
                require(Path(value).is_file(), f"resolved screenshot is unavailable: {value}")
            validated.append({"type": item_type, "value": value})
        return validated

    def _save_backend(self, output_dir: Path) -> None:
        state_dir = output_dir / "mace_lmev2"
        state_dir.mkdir(parents=True, exist_ok=True)
        (state_dir / "trajectories.jsonl").write_text(
            "".join(line + "\n" for line in self._trajectory_lines),
            encoding="utf-8",
        )
        saved_assets = state_dir / "assets"
        if saved_assets.exists():
            shutil.rmtree(saved_assets)
        shutil.copytree(self.asset_root, saved_assets, ignore=shutil.ignore_patterns(".staging"))
        saved_evidence = state_dir / "candidate_evidence"
        if saved_evidence.exists():
            shutil.rmtree(saved_evidence)
        shutil.copytree(self.candidate_evidence_dir, saved_evidence)
        saved_traces = state_dir / "retrieval_traces"
        if saved_traces.exists():
            shutil.rmtree(saved_traces)
        shutil.copytree(self.retrieval_trace_dir, saved_traces)

    def _load_backend(self, input_dir: Path) -> None:
        state_dir = input_dir / "mace_lmev2"
        trajectories_path = state_dir / "trajectories.jsonl"
        if not trajectories_path.is_file():
            return
        saved_assets = state_dir / "assets"
        require(saved_assets.is_dir(), "saved MACE asset directory is missing")
        if self.asset_root.exists():
            shutil.rmtree(self.asset_root)
        shutil.copytree(saved_assets, self.asset_root)
        saved_evidence = state_dir / "candidate_evidence"
        require(saved_evidence.is_dir(), "saved candidate evidence root is missing")
        saved_evidence_stat = saved_evidence.lstat()
        require(stat.S_ISDIR(saved_evidence_stat.st_mode),
                "saved candidate evidence root must be a real directory")
        if self.candidate_evidence_dir.exists():
            shutil.rmtree(self.candidate_evidence_dir)
        shutil.copytree(saved_evidence, self.candidate_evidence_dir)
        self._verify_persisted_candidate_evidence()

        saved_traces = state_dir / "retrieval_traces"
        require(saved_traces.is_dir(), "saved retrieval trace root is missing")
        saved_traces_stat = saved_traces.lstat()
        require(stat.S_ISDIR(saved_traces_stat.st_mode),
                "saved retrieval trace root must be a real directory")
        if self.retrieval_trace_dir.exists():
            shutil.rmtree(self.retrieval_trace_dir)
        shutil.copytree(saved_traces, self.retrieval_trace_dir)
        self._verify_persisted_retrieval_traces()
        self._trajectory_lines = []
        self._inserted_ids = set()
        for line in trajectories_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            trajectory = json.loads(line)
            trajectory_id = trajectory.get("id")
            require(isinstance(trajectory_id, str), "saved trajectory ID is invalid")
            acknowledgement = self._rpc({"op": "insert", "trajectory": trajectory})
            if acknowledgement.get("inserted") != 1:
                self._poison_sidecar()
                raise RuntimeError("MACE sidecar replay acknowledgement is not exact")
            self._inserted_ids.add(trajectory_id)
            self._trajectory_lines.append(line)

    def close(self) -> None:
        with self._lock:
            self._closed = True
            process = self._process
            self._process = None
            if process is not None:
                if process.stdin is not None:
                    try:
                        process.stdin.close()
                    except OSError:
                        pass
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.terminate()
                    try:
                        process.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
            root_fd = getattr(self, "_source_root_fd", -1)
            if root_fd >= 0:
                os.close(root_fd)
                self._source_root_fd = -1

