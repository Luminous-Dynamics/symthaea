"""Thin LongMemEval-V2 Memory adapter for the Rust lexical evidence worker.

Install this file as memory_modules/broca_rust.py inside a pinned
LongMemEval-V2 checkout and register it at the bottom of memory_modules/memory.py.
The worker remains the retrieval implementation; this shim only handles lifecycle,
bounded JSONL IPC, official context shape validation, and persistence/replay.
"""
from __future__ import annotations

import atexit
import hashlib
import json
import os
from pathlib import Path
import queue
import shutil
import subprocess
import tempfile
import threading
from typing import Any

from .memory import Memory, MemoryContextItem, register_memory, require


PROTOCOL_ID = "BROCA-LME-V2-BM25-WORKER-V1"
MAX_WORKER_LINE_BYTES = 32 * 1024 * 1024
WORKER_MANIFEST_NAME = "broca_worker_manifest.json"
TRAJECTORY_JOURNAL_NAME = "broca_trajectory_journal.jsonl"


@register_memory
class BrocaRustMemory(Memory):
    """Official-harness adapter for a persistent, lexical-only Rust worker."""

    memory_type = "broca_rust"

    def __init__(self, memory_params: dict[str, object]) -> None:
        super().__init__(memory_params)
        allowed = {
            "worker_path",
            "scope_id",
            "timeout_seconds",
            "max_jsonl_line_bytes",
            "bm25_params",
        }
        unexpected = sorted(set(memory_params) - allowed)
        require(not unexpected, f"broca_rust contains unexpected keys: {unexpected}")

        raw_worker = memory_params.get("worker_path")
        require(isinstance(raw_worker, str) and raw_worker.strip(), "worker_path must be a non-empty path")
        self.worker_path = Path(raw_worker).expanduser().resolve()
        require(self.worker_path.is_file(), f"worker binary not found: {self.worker_path}")
        require(os.access(self.worker_path, os.X_OK), f"worker binary is not executable: {self.worker_path}")

        raw_scope = memory_params.get("scope_id")
        require(
            isinstance(raw_scope, str) and raw_scope in {"web", "enterprise"},
            "scope_id must be exactly 'web' or 'enterprise'",
        )
        self.scope_id = raw_scope

        raw_timeout = memory_params.get("timeout_seconds", 60.0)
        require(
            isinstance(raw_timeout, (int, float)) and not isinstance(raw_timeout, bool)
            and 0.1 <= float(raw_timeout) <= 3600.0,
            "timeout_seconds must be in [0.1, 3600]",
        )
        self.timeout_seconds = float(raw_timeout)

        raw_max_line = memory_params.get("max_jsonl_line_bytes", MAX_WORKER_LINE_BYTES)
        require(
            isinstance(raw_max_line, int) and not isinstance(raw_max_line, bool)
            and 1024 <= raw_max_line <= MAX_WORKER_LINE_BYTES,
            f"max_jsonl_line_bytes must be in [1024, {MAX_WORKER_LINE_BYTES}]",
        )
        self.max_jsonl_line_bytes = raw_max_line

        raw_bm25 = memory_params.get("bm25_params", {})
        require(isinstance(raw_bm25, dict), "bm25_params must be an object")
        require(
            not (set(raw_bm25) - {"k1", "b", "max_items"}),
            "bm25_params may contain only k1, b, max_items",
        )
        self.bm25_params = {
            "k1": float(raw_bm25.get("k1", 1.2)),
            "b": float(raw_bm25.get("b", 0.75)),
            "max_items": int(raw_bm25.get("max_items", 5)),
        }
        require(self.bm25_params["k1"] > 0 and self.bm25_params["k1"] < 100,
                "bm25_params.k1 must be in (0, 100)")
        require(0 <= self.bm25_params["b"] <= 1,
                "bm25_params.b must be in [0, 1]")
        require(1 <= self.bm25_params["max_items"] <= 100,
                "bm25_params.max_items must be in [1, 100]")

        self.worker_sha256 = self._sha256_file(self.worker_path)
        self._lock = threading.RLock()
        self._response_queue: queue.Queue[bytes | BaseException | None] = queue.Queue()
        self._request_sequence = 0
        self._failed = False
        self._closed = False
        self._trajectory_ids: set[str] = set()
        self._trajectory_count = 0
        self._audit_by_invocation: dict[str, dict[str, Any]] = {}
        self._journal = tempfile.TemporaryFile(mode="w+t", encoding="utf-8", newline="\n")

        self._process = subprocess.Popen(
            [str(self.worker_path)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            bufsize=-1,
        )
        require(self._process.stdin is not None and self._process.stdout is not None,
                "failed to create worker standard streams")
        self._reader_thread = threading.Thread(
            target=self._read_responses,
            name="broca-lme-worker-stdout",
            daemon=True,
        )
        self._reader_thread.start()

        try:
            self._rpc({
                "op": "init",
                "scope_id": self.scope_id,
                "now_unix_ms": 0,
                "bm25": self.bm25_params,
            })
        except BaseException:
            self._abort_worker()
            raise
        atexit.register(self.close)

    @property
    def memory_config(self) -> dict[str, object]:
        return {
            "memory_type": self.memory_type,
            "memory_params": {
                "worker_path": str(self.worker_path),
                "scope_id": self.scope_id,
                "timeout_seconds": self.timeout_seconds,
                "max_jsonl_line_bytes": self.max_jsonl_line_bytes,
                "bm25_params": dict(self.bm25_params),
            },
        }

    def insert(self, trajectory: dict[str, object]) -> None:
        require(isinstance(trajectory, dict), "trajectory must be an object")
        trajectory_id = trajectory.get("id")
        require(isinstance(trajectory_id, str) and trajectory_id.strip(),
                "trajectory.id must be a non-empty string")
        with self._lock:
            require(trajectory_id not in self._trajectory_ids,
                    f"duplicate trajectory insert attempted: {trajectory_id}")

            serialized = json.dumps(
                trajectory,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            encoded = (serialized + "\n").encode("utf-8")
            require(len(encoded) <= self.max_jsonl_line_bytes,
                    "trajectory exceeds configured JSONL line bound")

            response = self._rpc({"op": "insert", "trajectory": trajectory})
            inserted = response.get("inserted_evidence_items")
            require(isinstance(inserted, int) and inserted > 0,
                    "worker insert receipt must report positive inserted_evidence_items")

            # Only journal a trajectory after the worker atomically accepted it.
            self._journal.write(serialized + "\n")
            self._journal.flush()
            self._trajectory_ids.add(trajectory_id)
            self._trajectory_count += 1

    def query(
        self,
        query: str,
        query_image: str | None = None,
    ) -> list[MemoryContextItem]:
        require(isinstance(query, str) and query.strip(), "broca_rust query must be non-empty")
        response = self._rpc({"op": "query", "query": query, "query_image": query_image})
        context = response.get("context")
        require(isinstance(context, list), "worker query response context must be a list")

        checked: list[MemoryContextItem] = []
        for index, item in enumerate(context):
            require(isinstance(item, dict), f"context item {index} must be an object")
            require(set(item) == {"type", "value"},
                    f"context item {index} must contain exactly type and value")
            kind = item.get("type")
            value = item.get("value")
            require(kind in {"text", "image"}, f"context item {index} has invalid type")
            require(isinstance(value, str) and value.strip(),
                    f"context item {index} value must be a non-empty string")
            if kind == "image":
                image_path = Path(value)
                require(image_path.is_absolute() and image_path.is_file(),
                        f"context item {index} image path does not exist")
            checked.append({"type": kind, "value": value})

        query_context = self.get_query_context()
        invocation_id = query_context.get("query_invocation_id")
        if invocation_id:
            self._audit_by_invocation[invocation_id] = {
                "protocol_id": PROTOCOL_ID,
                "scope_id": self.scope_id,
                "retrieval_type": "lexical_bm25",
                "worker_sha256": self.worker_sha256,
                "audit": response.get("audit", []),
                "image_query_received_but_not_used": bool(
                    response.get("image_query_received_but_not_used", False)
                ),
            }
        return checked

    def post_query_hook(
        self,
        *,
        query: str,
        query_image: str | None,
        memory_context: list[MemoryContextItem],
    ) -> dict[str, object] | None:
        invocation_id = self.get_query_context().get("query_invocation_id")
        if not invocation_id:
            return {
                "protocol_id": PROTOCOL_ID,
                "retrieval_type": "lexical_bm25",
                "audit_available": False,
            }
        with self._lock:
            return self._audit_by_invocation.pop(invocation_id, {
                "protocol_id": PROTOCOL_ID,
                "retrieval_type": "lexical_bm25",
                "audit_available": False,
            })

    def configure_runtime(self, **kwargs: object) -> None:
        # No mutable query policy is accepted from runtime/harness metadata.
        # This worker's scope and ranking configuration are fixed at init.
        return None

    def _rpc(self, payload: dict[str, object]) -> dict[str, Any]:
        with self._lock:
            require(not self._closed, "worker is closed")
            require(not self._failed, "worker is in a fail-closed state")
            require(self._process.poll() is None, "worker process exited")
            self._request_sequence += 1
            request_id = str(self._request_sequence)
            request = {"request_id": request_id, **payload}
            line = (
                json.dumps(request, ensure_ascii=True, sort_keys=True,
                           separators=(",", ":"), allow_nan=False) + "\n"
            ).encode("utf-8")
            require(len(line) <= self.max_jsonl_line_bytes,
                    "request exceeds configured JSONL line bound")
            try:
                assert self._process.stdin is not None
                self._process.stdin.write(line)
                self._process.stdin.flush()
            except (BrokenPipeError, OSError) as error:
                self._abort_worker()
                raise RuntimeError(f"failed to write worker request: {error}") from error

            try:
                response_line = self._response_queue.get(timeout=self.timeout_seconds)
            except queue.Empty as error:
                self._abort_worker()
                raise TimeoutError(
                    f"worker timed out after {self.timeout_seconds:.3f}s for request {request_id}"
                ) from error

            if isinstance(response_line, BaseException):
                self._abort_worker()
                raise RuntimeError(f"worker stdout failed: {response_line}") from response_line
            if response_line is None:
                self._abort_worker()
                raise RuntimeError("worker closed stdout before replying")
            if len(response_line) > self.max_jsonl_line_bytes or not response_line.endswith(b"\n"):
                self._abort_worker()
                raise RuntimeError("worker response is oversized or not newline-terminated")
            try:
                response = json.loads(response_line)
            except (json.JSONDecodeError, UnicodeDecodeError) as error:
                self._abort_worker()
                raise RuntimeError(f"worker returned malformed JSON: {error}") from error
            if not isinstance(response, dict):
                self._abort_worker()
                raise RuntimeError("worker response must be a JSON object")
            if response.get("request_id") != request_id:
                self._abort_worker()
                raise RuntimeError(
                    f"worker response ID mismatch: expected {request_id}, "
                    f"got {response.get('request_id')!r}"
                )
            if response.get("ok") is not True:
                raise RuntimeError(
                    f"worker rejected request {request_id}: "
                    f"{response.get('error', 'unknown error')}"
                )
            result = response.get("result")
            if not isinstance(result, dict):
                self._abort_worker()
                raise RuntimeError("worker response result must be a JSON object")
            return result

    def _read_responses(self) -> None:
        assert self._process.stdout is not None
        try:
            while True:
                line = self._process.stdout.readline(self.max_jsonl_line_bytes + 1)
                if not line:
                    self._response_queue.put(None)
                    return
                self._response_queue.put(line)
        except BaseException as error:
            self._response_queue.put(error)

    def _abort_worker(self) -> None:
        with self._lock:
            self._failed = True
            if self._process.poll() is None:
                self._process.terminate()
                try:
                    self._process.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    self._process.kill()
                    self._process.wait(timeout=1.0)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            try:
                if self._process.stdin is not None:
                    self._process.stdin.close()
                if self._process.poll() is None:
                    self._process.terminate()
                    try:
                        self._process.wait(timeout=1.0)
                    except subprocess.TimeoutExpired:
                        self._process.kill()
                        self._process.wait(timeout=1.0)
            finally:
                self._journal.close()

    def _save_backend(self, output_dir: Path) -> None:
        journal_path = output_dir / TRAJECTORY_JOURNAL_NAME
        self._journal.flush()
        self._journal.seek(0)
        with journal_path.open("w", encoding="utf-8", newline="\n") as destination:
            shutil.copyfileobj(self._journal, destination)
            destination.flush()
            os.fsync(destination.fileno())
        self._journal.seek(0, os.SEEK_END)

        journal_sha = self._sha256_file(journal_path)
        manifest = {
            "protocol_id": PROTOCOL_ID,
            "scope_id": self.scope_id,
            "worker_sha256": self.worker_sha256,
            "trajectory_count": self._trajectory_count,
            "journal_file": TRAJECTORY_JOURNAL_NAME,
            "journal_sha256": journal_sha,
            "bm25_params": dict(self.bm25_params),
        }
        manifest_path = output_dir / WORKER_MANIFEST_NAME
        with manifest_path.open("w", encoding="utf-8", newline="\n") as file:
            file.write(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
            file.flush()
            os.fsync(file.fileno())

    def _load_backend(self, input_dir: Path) -> None:
        manifest_path = input_dir / WORKER_MANIFEST_NAME
        journal_path = input_dir / TRAJECTORY_JOURNAL_NAME
        require(manifest_path.is_file(), f"missing worker manifest: {manifest_path}")
        require(journal_path.is_file(), f"missing trajectory journal: {journal_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        require(isinstance(manifest, dict), "worker manifest must be an object")
        require(manifest.get("protocol_id") == PROTOCOL_ID, "unsupported worker manifest protocol")
        require(manifest.get("scope_id") == self.scope_id, "saved memory scope differs from configured scope")
        require(manifest.get("worker_sha256") == self.worker_sha256, "worker binary digest differs from saved memory")
        require(manifest.get("bm25_params") == self.bm25_params, "BM25 configuration differs from saved memory")
        require(manifest.get("journal_file") == TRAJECTORY_JOURNAL_NAME, "unexpected trajectory journal filename")
        require(self._sha256_file(journal_path) == manifest.get("journal_sha256"),
                "trajectory journal digest mismatch")

        loaded_count = 0
        with journal_path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                require(len(line.encode("utf-8")) <= self.max_jsonl_line_bytes,
                        f"journal line {line_number} exceeds configured size bound")
                if not line.strip():
                    raise RuntimeError(f"blank trajectory journal line {line_number}")
                trajectory = json.loads(line)
                require(isinstance(trajectory, dict), f"journal line {line_number} must be an object")
                self.insert(trajectory)
                loaded_count += 1
        require(loaded_count == manifest.get("trajectory_count"),
                "trajectory journal count differs from manifest")

    @staticmethod
    def _sha256_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
