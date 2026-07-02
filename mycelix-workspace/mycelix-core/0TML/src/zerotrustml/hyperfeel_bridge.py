"""
Rust-first HyperFeel bridge for ZeroTrustML.

This module provides a thin Python wrapper around the Rust
`mycelix-hyperfeel` crate via the `hyperfeel_cli` binary. It is
designed to be:

- Minimal and dependency-light
- Easy to swap to FFI later if desired
- Safe to ignore on systems where Rust is not installed
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from typing import List, Optional, Sequence


class HyperFeelUnavailable(RuntimeError):
    """Raised when the `hyperfeel_cli` binary is not available."""


@dataclass
class HyperFeelConfig:
    """
    Configuration for calling the Rust HyperFeel CLI.

    Attributes:
        binary: Optional explicit path to the `hyperfeel_cli` binary.
                If None, `$PATH` will be searched.
    """

    binary: Optional[str] = None

    def resolve_binary(self) -> str:
        """Return the resolved path to the CLI or raise if missing."""
        if self.binary is not None:
            return self.binary

        path = shutil.which("hyperfeel_cli")
        if path is None:
            raise HyperFeelUnavailable(
                "hyperfeel_cli binary not found on PATH. "
                "Build it with `cargo build --release` in "
                "`Mycelix-Core/0TML/mycelix-hyperfeel` and ensure "
                "the resulting binary is on PATH."
            )
        return path


def encode_gradient(
    gradient: Sequence[float],
    config: Optional[HyperFeelConfig] = None,
) -> dict:
    """
    Encode a dense gradient into a HyperGradient using Rust HyperFeel.

    Args:
        gradient: 1D iterable of floats (e.g. numpy array or list).
        config: Optional HyperFeelConfig specifying the CLI location.

    Returns:
        A dict corresponding to the serialized HyperGradient from Rust.

    Raises:
        HyperFeelUnavailable: if the CLI binary is not available.
        RuntimeError: on non-zero exit or invalid JSON.
    """
    cfg = config or HyperFeelConfig()
    binary = cfg.resolve_binary()

    payload = {"gradient": list(float(x) for x in gradient)}
    proc = subprocess.run(
        [binary, "encode"],
        input=json.dumps(payload).encode("utf-8"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    if proc.returncode != 0:
        raise RuntimeError(
            f"hyperfeel_cli encode failed (code={proc.returncode}): "
            f"{proc.stderr.decode('utf-8', errors='ignore')}"
        )

    try:
        return json.loads(proc.stdout.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"invalid JSON from hyperfeel_cli encode: {exc}") from exc


def aggregate_hypergradients(
    hypergradients: List[dict],
    config: Optional[HyperFeelConfig] = None,
) -> dict:
    """
    Aggregate multiple HyperGradients via Rust HyperFeel.

    Args:
        hypergradients: List of dicts previously returned by `encode_gradient`
                        (or compatible structures).
        config: Optional HyperFeelConfig specifying the CLI location.

    Returns:
        A dict representing the aggregated HyperGradient.

    Raises:
        HyperFeelUnavailable: if the CLI binary is not available.
        RuntimeError: on non-zero exit or invalid JSON.
    """
    if not hypergradients:
        raise ValueError("hypergradients list must not be empty")

    cfg = config or HyperFeelConfig()
    binary = cfg.resolve_binary()

    proc = subprocess.run(
        [binary, "aggregate"],
        input=json.dumps(hypergradients).encode("utf-8"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    if proc.returncode != 0:
        raise RuntimeError(
            f"hyperfeel_cli aggregate failed (code={proc.returncode}): "
            f"{proc.stderr.decode('utf-8', errors='ignore')}"
        )

    try:
        return json.loads(proc.stdout.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"invalid JSON from hyperfeel_cli aggregate: {exc}") from exc

