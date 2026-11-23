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
import os
from pathlib import Path
import shutil
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
        require(isinstance(sidecar_value, str) and sidecar_value, "sidecar_binary is required")
        require(isinstance(source_root_value, str) and source_root_value, "source_screenshot_root is required")
        require(isinstance(workspace_value, str) and workspace_value, "workspace_dir is required")
        require(domain in {"web", "enterprise"}, "domain must be web or enterprise")

        self.sidecar_binary = Path(sidecar_value).expanduser().resolve(strict=True)
        require(self.sidecar_binary.is_file(), "sidecar_binary must be a file")
        require(os.access(self.sidecar_binary, os.X_OK), "sidecar_binary must be executable")

        self.source_screenshot_root = Path(source_root_value).expanduser().resolve(strict=True)
        require(self.source_screenshot_root.is_dir(), "source_screenshot_root must be a directory")
        self.workspace_dir = Path(workspace_value).expanduser().resolve()
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        self.asset_root = self.workspace_dir / "assets"
        self.asset_root.mkdir(parents=True, exist_ok=True)

        self.domain = str(domain)
        self.top_k = self._bounded_int(memory_params.get("top_k", 8), "top_k", 1, 100)
        self.slice_radius = self._bounded_int(memory_params.get("slice_radius", 1), "slice_radius", 0, 32)
        self.neighbor_radius = self._bounded_int(
            memory_params.get("neighbor_radius", 1), "neighbor_radius", 0, 32
        )
        self.bm25_k1 = self._bounded_float(memory_params.get("bm25_k1", 1.2), "bm25_k1", minimum=0.0)
        self.bm25_b = self._bounded_float(memory_params.get("bm25_b", 0.75), "bm25_b", minimum=0.0)
        require(self.bm25_k1 > 0.0, "bm25_k1 must be greater than zero")
        require(self.bm25_b <= 1.0, "bm25_b must be <= 1")

        self._lock = threading.RLock()
        self._process: subprocess.Popen[str] | None = None
        self._poisoned = False
        self._closed = False
        self._trajectory_lines: list[str] = []
        self._inserted_ids: set[str] = set()
        super().__init__(memory_params)
        atexit.register(self.close)

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

        env = os.environ.copy()
        env.update(
            {
                "MACE_LMEV2_DOMAIN": self.domain,
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
                source_size = source.stat().st_size
                require(
                    source_size <= MAX_SCREENSHOT_BYTES,
                    f"screenshot {ordinal} exceeds {MAX_SCREENSHOT_BYTES} bytes",
                )
                image_bytes = source.read_bytes()
                require(
                    len(image_bytes) == source_size,
                    f"screenshot {ordinal} changed while it was being copied",
                )
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
        response = self._rpc(
            {
                "op": "query",
                "query": query,
                "query_image": query_image,
            }
        )
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
        self._trajectory_lines = []
        self._inserted_ids = set()
        for line in trajectories_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            trajectory = json.loads(line)
            trajectory_id = trajectory.get("id")
            require(isinstance(trajectory_id, str), "saved trajectory ID is invalid")
            self._rpc({"op": "insert", "trajectory": trajectory})
            self._inserted_ids.add(trajectory_id)
            self._trajectory_lines.append(line)

    def close(self) -> None:
        with self._lock:
            self._closed = True
            process = self._process
            self._process = None
            if process is None:
                return
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

