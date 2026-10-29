#!/usr/bin/env python3
"""Differentially compare Symthaea's independent Huber-Catoni equations to upstream.

The upstream GPL-3.0 checkout is an external test dependency. This file neither
vendors nor imports upstream code into the product; it loads that checkout only
when this benchmark is explicitly run.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np

from huber_catoni_reference import catoni_huber_cs

DEFAULT_EPSILONS = (0.025, 0.05, 0.12, 0.18)
DEFAULT_CHECKPOINTS = (1, 2, 4, 8, 16, 32, 64, 128)
SUPPORT = (0.0, 1.0)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def clip_interval(lower: float, upper: float,
                  support: tuple[float, float] = SUPPORT) -> tuple[float, float]:
    """Intersect reference endpoints with the known clean-mean support."""
    lo, hi = support
    if not lo < hi:
        raise ValueError("support must be strictly increasing")
    return max(lo, float(lower)), min(hi, float(upper))


def compare_bounds(expected: tuple[float, float],
                    observed: tuple[float, float]) -> dict:
    """Return endpoint errors; infinities are not silently considered equal."""
    errors = []
    nonfinite_mismatch = False
    for lhs, rhs in zip(expected, observed, strict=True):
        if lhs == rhs:
            errors.append(0.0)
        elif not (np.isfinite(lhs) and np.isfinite(rhs)):
            errors.append(None)
            nonfinite_mismatch = True
        else:
            errors.append(abs(float(lhs) - float(rhs)))
    finite_errors = [error for error in errors if error is not None]
    return {
        "lower_abs_error": errors[0],
        "upper_abs_error": errors[1],
        "max_abs_error": max(finite_errors) if not nonfinite_mismatch else None,
        "nonfinite_mismatch": nonfinite_mismatch,
    }


def make_streams(max_n: int, seed: int) -> dict[str, np.ndarray]:
    if max_n < 1:
        raise ValueError("max_n must be positive")
    rng = np.random.default_rng(seed)
    return {
        "bernoulli_037": rng.binomial(1, 0.37, size=max_n).astype(float),
        "alternating_01": np.resize(np.array([0.0, 1.0]), max_n),
        "constant_040": np.full(max_n, 0.4, dtype=float),
    }


def load_reference(reference_dir: Path):
    root = reference_dir.resolve()
    source = root / "robustconfseq.py"
    toolkit = root / "toolkit.py"
    if not source.is_file() or not toolkit.is_file():
        raise FileNotFoundError(
            f"{root} must be a catoni-confseq checkout containing robustconfseq.py and toolkit.py"
        )
    # The benchmark imports upstream only from the caller-supplied checkout.
    sys.path.insert(0, str(root))
    module = importlib.import_module("robustconfseq")
    return module.RCS_Generator, source, toolkit, root


def reference_commit(root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def run(reference_dir: Path, *, max_n: int = 128, seed: int = 20261031,
        alpha: float = 0.05, moment_bound: float = 0.25,
        epsilons: tuple[float, ...] = DEFAULT_EPSILONS,
        tolerance: float = 5e-4) -> dict:
    if not (0.0 < alpha < 0.5):
        raise ValueError("alpha must be in (0, 0.5)")
    if moment_bound <= 0 or tolerance <= 0:
        raise ValueError("moment_bound and tolerance must be positive")
    if not epsilons or any(not (0.0 < e < 1.0) for e in epsilons):
        raise ValueError("this differential harness requires positive epsilons in (0, 1)")
    if any(2.0 < 1.0 + 1.5 * e for e in epsilons):
        raise ValueError("epsilon exceeds the p=2 breakdown condition")
    checkpoints = tuple(n for n in DEFAULT_CHECKPOINTS if n <= max_n)
    if not checkpoints or checkpoints[-1] != max_n:
        checkpoints = tuple(sorted(set((*checkpoints, max_n))))

    reference_cls, upstream_source, upstream_toolkit, root = load_reference(reference_dir)
    streams = make_streams(max_n, seed)
    cases = []
    for stream_name, samples in streams.items():
        for eps in epsilons:
            # Separate one-sided upstream CSs at alpha/2 avoid relying on the
            # upstream two-sided wrapper's alpha-allocation convention.
            ref_lower = reference_cls(
                eps=eps, moment=moment_bound, alpha=alpha / 2.0, p=2.0, side="r"
            )
            ref_upper = reference_cls(
                eps=eps, moment=moment_bound, alpha=alpha / 2.0, p=2.0, side="l"
            )
            for n, sample in enumerate(samples, start=1):
                low_out = ref_lower.observe(float(sample))
                high_out = ref_upper.observe(float(sample))
                if n not in checkpoints:
                    continue
                upstream_bounds = clip_interval(
                    float(low_out["lcb"]), float(high_out["ucb"])
                )
                independent_bounds = catoni_huber_cs(
                    samples[:n], alpha=alpha, eps=eps,
                    moment_bound=moment_bound, p=2.0, bisections=48,
                    support=SUPPORT,
                )
                errors = compare_bounds(independent_bounds, upstream_bounds)
                cases.append({
                    "stream": stream_name, "epsilon": eps, "n": n,
                    "independent_bounds": list(independent_bounds),
                    "upstream_clipped_bounds": list(upstream_bounds),
                    **errors,
                    "within_tolerance": (
                        not errors["nonfinite_mismatch"]
                        and errors["max_abs_error"] is not None
                        and errors["max_abs_error"] <= tolerance
                    ),
                })

    finite_errors = [c["max_abs_error"] for c in cases if c["max_abs_error"] is not None]
    max_error = max(finite_errors, default=0.0)
    failures = [c for c in cases if not c["within_tolerance"]]
    nonfinite_mismatches = sum(bool(c["nonfinite_mismatch"]) for c in cases)
    implementation_path = Path(__file__).with_name("huber_catoni_reference.py")
    return {
        "schema": "symthaea-huber-catoni-reference-crosscheck-v1",
        "status": "PASS_REFERENCE_EQUIVALENCE" if not failures else "FAIL_REFERENCE_EQUIVALENCE",
        "scope": "equation/root equivalence only; not a coverage proof or SOTA benchmark",
        "protocol": {
            "seed": seed, "max_n": max_n, "checkpoints": list(checkpoints),
            "streams": list(streams), "epsilon_grid": list(epsilons),
            "alpha_two_sided": alpha, "alpha_per_tail": alpha / 2.0,
            "clean_central_variance_bound": moment_bound, "p": 2.0,
            "clean_mean_support": list(SUPPORT), "absolute_endpoint_tolerance": tolerance,
            "upstream_reference_commit": reference_commit(root),
            "upstream_robustconfseq_sha256": sha256_file(upstream_source),
            "upstream_toolkit_sha256": sha256_file(upstream_toolkit),
            "independent_equation_source_sha256": sha256_file(implementation_path),
            "reference_license": "GPL-3.0; external benchmark dependency only, not vendored",
        },
        "summary": {
            "comparison_count": len(cases), "failure_count": len(failures),
            "nonfinite_mismatch_count": nonfinite_mismatches,
            "max_abs_finite_endpoint_error": max_error,
        },
        "cases": cases,
        "limitations": [
            "positive epsilon only; epsilon=0 has a separate lambda/alpha convention",
            "three deterministic bounded streams and finite checkpoints validate implementation agreement, not statistical coverage",
            "correct output agreement does not validate a claimed contamination or moment bound",
        ],
        "environment": {"python": platform.python_version(), "numpy": np.__version__},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", type=Path, required=True,
                        help="external local checkout of HJ-Wg/catoni-confseq")
    parser.add_argument("--max-n", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20261031)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--moment-bound", type=float, default=0.25)
    parser.add_argument("--epsilons", default="0.025,0.05,0.12,0.18")
    parser.add_argument("--tolerance", type=float, default=5e-4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    epsilons = tuple(float(x) for x in args.epsilons.split(",") if x.strip())
    result = run(
        args.reference_dir, max_n=args.max_n, seed=args.seed, alpha=args.alpha,
        moment_bound=args.moment_bound, epsilons=epsilons, tolerance=args.tolerance,
    )
    encoded = (json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(encoded)
    print(json.dumps({
        "output": str(args.output), "bytes": len(encoded),
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "status": result["status"], **result["summary"],
    }, indent=2))
    return 0 if result["status"] == "PASS_REFERENCE_EQUIVALENCE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
