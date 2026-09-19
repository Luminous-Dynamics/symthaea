#!/usr/bin/env python3
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
"""DE-001A1C released-Cobaya one-call fixed-point executor.

This executable performs exactly one released Cobaya DESI DR2 BAO likelihood
call after validating a frozen A1C authorization receipt and all hash-addressed
evidence needed to dereference its transitive provenance.

Exit codes:
  0: PASS fixed-point reproduction
  1: NEGATIVE fixed-point reproduction
  2: INVALID execution/evidence
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.abc
import importlib.metadata
import inspect
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any

PROTOCOL = "DE-001A1C-COBAYA-RESULT-v1"
AUTHORITY = "released-likelihood-fixed-point-reproduction-only"
AUTH_PROTOCOL = "DE-001A1C-EXECUTION-AUTHORIZATION-v1"
A1Q_PROTOCOL = "DE-001A1Q-BUNDLE-INTEGRITY-v1"
A1E_PROTOCOL = "DE-001A1E-ENVIRONMENT-EVIDENCE-v1"
CONTRACT_PROTOCOL = "DE-001A1C-CONTRACT-CONSISTENCY-v1"
MANIFEST_PROTOCOL = "DE-001A1C-COBAYA-FIXED-POINT-v1"
EXECUTOR_PROTOCOL = "DE-001A1C-COBAYA-EXECUTOR-v1"

COBAYA_VERSION = "3.6.2"
COBAYA_SOURCE_COMMIT = "899f30a49f85de610dac321e91a1af50018e56aa"
LIKELIHOOD_YAML_SHA256 = "fd7e9bf2dcf5ffee90a9a30b18227f4337d6d5c1978782c63513cbe0d8280daa"

OMEGA_M = 0.29717787
H_R_D_MPC = 101.54786
RDRAG_GAUGE_MPC = 100.0
REFERENCE_CHI2 = 10.282299
REPRODUCTION_TOLERANCE = 0.01
INTERNAL_CHI2_TOLERANCE = 1.0e-10
C_KM_S = 299792.458

LIKELIHOOD_CALLS = 0

MEAN_ROLE = "dataset-mean"
MEAN_SIZE = 472
MEAN_SHA256 = "9ac154ab583ce759c0f7eef3c978c7c70a6ead2d18774caceadf1a350a640585"
COV_ROLE = "dataset-covariance"
COV_SIZE = 2547
COV_SHA256 = "252a143274c8a07c78694c119617d36594f6d7965d00319ca611c6ffb886e509"

MANIFEST_RELATIVE = Path(
    "crates/domains/symthaea-cosmology-research/references/"
    "de001a_a1c_cobaya_fixed_point_v1.json"
)
AUTH_SPEC_RELATIVE = Path(
    "crates/domains/symthaea-cosmology-research/references/"
    "de001a_a1c_authorization_v1.json"
)
EXECUTOR_SPEC_RELATIVE = Path(
    "crates/domains/symthaea-cosmology-research/references/"
    "de001a_a1c_executor_v1.json"
)

AUTH_KEYS = {
    "protocol",
    "verdict",
    "scientific_claim",
    "authority",
    "a1c_execution_authorized",
    "subject_head",
    "subject_tree",
    "authorization_spec_sha256",
    "a1c_manifest_sha256",
    "a1q_receipt_sha256",
    "a1e_receipt_sha256",
    "a1c_contract_receipt_sha256",
    "a1q_reproduction_verdict",
    "environment_definition_sha256",
    "likelihood_call_budget",
    "sampler_allowed",
    "minimizer_allowed",
    "optimization_allowed",
    "parameter_mutation_allowed",
    "network_allowed",
    "camb_allowed",
    "runtime_package_installation_allowed",
    "authorization_scope",
    "stateful_global_single_use_claimed",
}


class InvalidExecution(RuntimeError):
    pass


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise InvalidExecution(f"duplicate JSON key: {key}")
        out[key] = value
    return out


def _read_regular(path: Path, max_bytes: int = 8 * 1024 * 1024) -> bytes:
    try:
        st = path.lstat()
    except OSError as exc:
        raise InvalidExecution(f"{path}: metadata failed: {exc}") from exc
    if path.is_symlink():
        raise InvalidExecution(f"{path}: symlinks are forbidden")
    if not path.is_file():
        raise InvalidExecution(f"{path}: not a regular file")
    if st.st_size > max_bytes:
        raise InvalidExecution(f"{path}: file exceeds {max_bytes} bytes")
    data = path.read_bytes()
    if len(data) != st.st_size:
        raise InvalidExecution(f"{path}: size changed while reading")
    return data


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_json(path: Path) -> tuple[bytes, dict[str, Any]]:
    data = _read_regular(path)
    try:
        value = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(
                InvalidExecution(f"non-finite JSON constant: {value}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise InvalidExecution(f"{path}: invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise InvalidExecution(f"{path}: top-level JSON must be an object")
    return data, value


def _run(*args: str) -> str:
    try:
        completed = subprocess.run(
            args,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            shell=False,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        stderr = getattr(exc, "stderr", "") or ""
        raise InvalidExecution(f"command failed: {args!r}: {stderr.strip()}") from exc
    return completed.stdout.strip()


def _repository_identity() -> tuple[Path, str, str]:
    root = Path(_run("git", "rev-parse", "--show-toplevel"))
    head = _run("git", "rev-parse", "HEAD")
    tree = _run("git", "rev-parse", "HEAD^{tree}")
    if not root.is_dir() or len(head) not in (40, 64) or len(tree) not in (40, 64):
        raise InvalidExecution("invalid Git repository identity")
    return root, head, tree


def _require_clean_tracked_worktree() -> None:
    subprocess.run(["git", "diff", "--exit-code"], check=True, stdout=subprocess.DEVNULL)
    subprocess.run(
        ["git", "diff", "--cached", "--exit-code"],
        check=True,
        stdout=subprocess.DEVNULL,
    )


def _require_exact_keys(value: dict[str, Any], expected: set[str], label: str) -> None:
    actual = set(value)
    if actual != expected:
        raise InvalidExecution(
            f"{label} field set mismatch: missing={sorted(expected - actual)!r}, "
            f"unexpected={sorted(actual - expected)!r}"
        )


def _require_str(value: dict[str, Any], key: str) -> str:
    result = value.get(key)
    if not isinstance(result, str):
        raise InvalidExecution(f"{key}: expected string")
    return result


def _require_bool(value: dict[str, Any], key: str) -> bool:
    result = value.get(key)
    if not isinstance(result, bool):
        raise InvalidExecution(f"{key}: expected boolean")
    return result


def _require_int(value: dict[str, Any], key: str) -> int:
    result = value.get(key)
    if isinstance(result, bool) or not isinstance(result, int):
        raise InvalidExecution(f"{key}: expected integer")
    return result


def _validate_authorization(
    auth_bytes: bytes,
    auth: dict[str, Any],
    root: Path,
    head: str,
    tree: str,
) -> str:
    _require_exact_keys(auth, AUTH_KEYS, "authorization receipt")
    for key, expected in (
        ("protocol", AUTH_PROTOCOL),
        ("verdict", "PASS"),
        ("scientific_claim", "NONE"),
        ("authority", "fixed-point-execution-authorization-only"),
        ("subject_head", head),
        ("subject_tree", tree),
        ("authorization_scope", "one-fixed-point-executor-process"),
    ):
        if _require_str(auth, key) != expected:
            raise InvalidExecution(f"authorization receipt {key} mismatch")

    if not _require_bool(auth, "a1c_execution_authorized"):
        raise InvalidExecution("authorization receipt does not authorize A1C")
    if _require_int(auth, "likelihood_call_budget") != 1:
        raise InvalidExecution("authorization likelihood-call budget is not exactly one")
    for key in (
        "sampler_allowed",
        "minimizer_allowed",
        "optimization_allowed",
        "parameter_mutation_allowed",
        "network_allowed",
        "camb_allowed",
        "runtime_package_installation_allowed",
        "stateful_global_single_use_claimed",
    ):
        if _require_bool(auth, key):
            raise InvalidExecution(f"authorization unexpectedly permits/claims {key}")

    reproduction = _require_str(auth, "a1q_reproduction_verdict")
    if reproduction not in {"PASS", "NEGATIVE"}:
        raise InvalidExecution("authorization embeds invalid A1Q reproduction verdict")

    auth_spec_bytes = _read_regular(root / AUTH_SPEC_RELATIVE)
    manifest_bytes = _read_regular(root / MANIFEST_RELATIVE)
    if _sha256(auth_spec_bytes) != _require_str(auth, "authorization_spec_sha256"):
        raise InvalidExecution("authorization specification hash mismatch")
    if _sha256(manifest_bytes) != _require_str(auth, "a1c_manifest_sha256"):
        raise InvalidExecution("A1C manifest hash mismatch")

    return _sha256(auth_bytes)


def _validate_transitive_receipts(
    auth: dict[str, Any],
    a1q_path: Path,
    a1e_path: Path,
    contract_path: Path,
) -> dict[str, str]:
    a1q_bytes, a1q = _read_json(a1q_path)
    a1e_bytes, a1e = _read_json(a1e_path)
    contract_bytes, contract = _read_json(contract_path)

    if _sha256(a1q_bytes) != _require_str(auth, "a1q_receipt_sha256"):
        raise InvalidExecution("A1Q receipt hash does not match authorization")
    if _sha256(a1e_bytes) != _require_str(auth, "a1e_receipt_sha256"):
        raise InvalidExecution("A1E receipt hash does not match authorization")
    if _sha256(contract_bytes) != _require_str(auth, "a1c_contract_receipt_sha256"):
        raise InvalidExecution("A1C contract receipt hash does not match authorization")

    if _require_str(a1q, "protocol") != A1Q_PROTOCOL or _require_str(a1q, "verdict") != "PASS":
        raise InvalidExecution("hash-bound A1Q receipt is not PASS")
    if _require_str(a1q, "reproduction_verdict") != _require_str(
        auth, "a1q_reproduction_verdict"
    ):
        raise InvalidExecution("A1Q reproduction verdict differs from authorization")

    if _require_str(a1e, "protocol") != A1E_PROTOCOL or _require_str(a1e, "verdict") != "PASS":
        raise InvalidExecution("hash-bound A1E receipt is not PASS")
    if not _require_bool(a1e, "environment_reuse_authorized"):
        raise InvalidExecution("hash-bound A1E receipt does not authorize environment reuse")
    if _require_bool(a1e, "a1c_execution_authorized"):
        raise InvalidExecution("A1E improperly claims direct A1C execution authority")

    if (
        _require_str(contract, "protocol") != CONTRACT_PROTOCOL
        or _require_str(contract, "verdict") != "PASS"
    ):
        raise InvalidExecution("hash-bound A1C contract receipt is not PASS")
    if _require_bool(contract, "execution_authorized"):
        raise InvalidExecution("A1C contract improperly claims execution authority")

    return {
        "a0_receipt_sha256": _require_str(a1q, "a0_receipt_sha256"),
        "environment_qualification_receipt_sha256": _require_str(
            a1e, "qualification_receipt_sha256"
        ),
        "environment_versions_sha256": _require_str(a1e, "versions_sha256"),
        "point_manifest_sha256": _require_str(contract, "a1r_manifest_sha256"),
    }


def _validate_executor_spec(root: Path) -> tuple[dict[str, Any], str]:
    spec_bytes, spec = _read_json(root / EXECUTOR_SPEC_RELATIVE)
    if spec.get("schema_version") != 1:
        raise InvalidExecution("executor spec schema version mismatch")
    for key, expected in (
        ("protocol", EXECUTOR_PROTOCOL),
        ("status", "implementation-frozen-unqualified"),
        ("scientific_claim", "NONE"),
        ("authority", AUTHORITY),
        ("result_protocol", PROTOCOL),
    ):
        if _require_str(spec, key) != expected:
            raise InvalidExecution(f"executor spec {key} mismatch")
    script_rel = _require_str(spec, "script_path")
    script_bytes = _read_regular(root / script_rel)
    if _sha256(script_bytes) != _require_str(spec, "script_sha256"):
        raise InvalidExecution("executor script hash differs from frozen executor spec")

    policy = spec.get("execution_policy")
    if not isinstance(policy, dict):
        raise InvalidExecution("executor spec missing execution_policy")
    if policy.get("likelihood_calls") != 1:
        raise InvalidExecution("executor spec call budget is not exactly one")
    for key in (
        "sampler_forbidden",
        "minimizer_forbidden",
        "optimization_forbidden",
        "parameter_mutation_forbidden",
        "network_forbidden",
        "camb_forbidden",
        "runtime_package_installation_forbidden",
    ):
        if policy.get(key) is not True:
            raise InvalidExecution(f"executor spec does not freeze {key}=true")
    if float(spec.get("internal_chi2_tolerance", -1.0)) != INTERNAL_CHI2_TOLERANCE:
        raise InvalidExecution("executor internal consistency tolerance mismatch")
    return spec, _sha256(spec_bytes)


def _validate_manifest(root: Path) -> tuple[dict[str, Any], str]:
    manifest_bytes, manifest = _read_json(root / MANIFEST_RELATIVE)
    if manifest.get("schema_version") != 1:
        raise InvalidExecution("A1C manifest schema mismatch")
    for key, expected in (
        ("protocol", MANIFEST_PROTOCOL),
        ("scientific_claim", "NONE"),
        ("authority", AUTHORITY),
        ("status", "preregistered-contract-only"),
    ):
        if _require_str(manifest, key) != expected:
            raise InvalidExecution(f"A1C manifest {key} mismatch")

    subject = manifest.get("subject")
    provider = manifest.get("provider")
    source = manifest.get("source")
    policy = manifest.get("execution_policy")
    if not all(isinstance(value, dict) for value in (subject, provider, source, policy)):
        raise InvalidExecution("A1C manifest nested contract missing")

    expected_subject = {
        "omega_m": OMEGA_M,
        "h_r_d_mpc": H_R_D_MPC,
        "reference_chi2_bao": REFERENCE_CHI2,
        "absolute_tolerance": REPRODUCTION_TOLERANCE,
        "mean_role": MEAN_ROLE,
        "mean_size": MEAN_SIZE,
        "mean_sha256": MEAN_SHA256,
        "covariance_role": COV_ROLE,
        "covariance_size": COV_SIZE,
        "covariance_sha256": COV_SHA256,
        "row_count": 13,
    }
    for key, expected in expected_subject.items():
        if subject.get(key) != expected:
            raise InvalidExecution(f"A1C frozen subject mismatch: {key}")

    if (
        source.get("cobaya_version") != COBAYA_VERSION
        or source.get("cobaya_source_commit") != COBAYA_SOURCE_COMMIT
        or source.get("likelihood_definition_sha256") != LIKELIHOOD_YAML_SHA256
    ):
        raise InvalidExecution("A1C Cobaya source identity mismatch")
    if (
        provider.get("rdrag_gauge_mpc") != RDRAG_GAUGE_MPC
        or provider.get("speed_of_light_km_s") != C_KM_S
        or provider.get("camb_forbidden") is not True
    ):
        raise InvalidExecution("A1C provider identity mismatch")
    if (
        policy.get("likelihood_calls") != 1
        or policy.get("sampler_forbidden") is not True
        or policy.get("minimizer_forbidden") is not True
        or policy.get("optimization_forbidden") is not True
        or policy.get("network_forbidden") is not True
    ):
        raise InvalidExecution("A1C execution policy mismatch")

    return manifest, _sha256(manifest_bytes)


def _validate_a0_file(path: Path, role: str, size: int, digest: str) -> bytes:
    if path.name != role:
        raise InvalidExecution(f"{role}: expected basename {role!r}, got {path.name!r}")
    data = _read_regular(path)
    if len(data) != size or _sha256(data) != digest:
        raise InvalidExecution(f"{role}: A0 size/hash mismatch")
    return data


class _ForbiddenCambFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> Any:
        if fullname == "camb" or fullname.startswith("camb."):
            raise ImportError("CAMB import forbidden by DE-001A1C execution contract")
        if fullname == "cobaya.theories.camb" or fullname.startswith(
            "cobaya.theories.camb."
        ):
            raise ImportError("Cobaya CAMB theory import forbidden by DE-001A1C contract")
        return None


def _network_audit(event: str, args: tuple[Any, ...]) -> None:
    if event.startswith("socket."):
        raise InvalidExecution(f"network/socket operation forbidden: {event}")


def _map_redshift(z: Any, fn: Any) -> Any:
    import numpy as np

    arr = np.asarray(z, dtype=float)
    if arr.ndim == 0:
        return float(fn(float(arr)))
    values = [fn(float(value)) for value in arr.flat]
    return np.asarray(values, dtype=float).reshape(arr.shape)


class FlatLambdaBaoProvider:
    def __init__(self, omega_m: float, h_r_d_mpc: float, rdrag_mpc: float) -> None:
        self.omega_m = float(omega_m)
        self.h_r_d_mpc = float(h_r_d_mpc)
        self.rdrag_mpc = float(rdrag_mpc)
        self.h0 = 100.0 * self.h_r_d_mpc / self.rdrag_mpc
        self._distance_cache: dict[float, float] = {}

    def _e(self, z: float) -> float:
        if not math.isfinite(z) or z < 0.0:
            raise InvalidExecution(f"invalid redshift: {z}")
        return math.sqrt(self.omega_m * (1.0 + z) ** 3 + (1.0 - self.omega_m))

    def _comoving_distance(self, z: float) -> float:
        if z not in self._distance_cache:
            from scipy.integrate import quad

            integral, error = quad(
                lambda value: 1.0 / self._e(float(value)),
                0.0,
                z,
                epsabs=1.0e-11,
                epsrel=1.0e-11,
                limit=200,
            )
            if not math.isfinite(integral) or not math.isfinite(error):
                raise InvalidExecution("non-finite adaptive-quadrature result")
            self._distance_cache[z] = C_KM_S / self.h0 * integral
        return self._distance_cache[z]

    def get_angular_diameter_distance(self, z: Any) -> Any:
        return _map_redshift(z, lambda value: self._comoving_distance(value) / (1.0 + value))

    def get_Hubble(self, z: Any, units: str = "km/s/Mpc") -> Any:
        if units != "km/s/Mpc":
            raise InvalidExecution(f"unexpected Hubble units request: {units}")
        return _map_redshift(z, lambda value: self.h0 * self._e(value))

    def get_param(self, name: str) -> float:
        if name != "rdrag":
            raise InvalidExecution(f"unexpected provider parameter request: {name}")
        return self.rdrag_mpc


def _canonical_float_vector_sha256(values: list[float]) -> str:
    encoded = json.dumps(
        values, allow_nan=False, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return _sha256(encoded)


def _execute_likelihood(
    mean_path: Path, cov_path: Path
) -> tuple[float, float, list[float], str, str, str]:
    global LIKELIHOOD_CALLS
    sys.meta_path.insert(0, _ForbiddenCambFinder())
    sys.addaudithook(_network_audit)

    if importlib.metadata.version("cobaya") != COBAYA_VERSION:
        raise InvalidExecution("runtime Cobaya version mismatch")

    import numpy as np
    from cobaya.likelihoods.bao.desi_dr2.desi_bao_all import desi_bao_all

    if "camb" in sys.modules or "cobaya.theories.camb" in sys.modules:
        raise InvalidExecution("CAMB was imported before A1C evaluation")

    class_path = Path(inspect.getfile(desi_bao_all)).resolve()
    yaml_path = class_path.with_suffix(".yaml")
    yaml_bytes = _read_regular(yaml_path)
    if _sha256(yaml_bytes) != LIKELIHOOD_YAML_SHA256:
        raise InvalidExecution("installed DESI DR2 likelihood YAML hash mismatch")

    if mean_path.parent.resolve() != cov_path.parent.resolve():
        raise InvalidExecution("A0 mean and covariance must share one artifact directory")
    artifact_dir = mean_path.parent.resolve()

    likelihood = desi_bao_all(
        {
            "path": str(artifact_dir),
            "measurements_file": MEAN_ROLE,
            "cov_file": COV_ROLE,
        },
        name="bao.desi_dr2",
        packages_path=str(artifact_dir),
        timing=False,
        standalone=True,
    )
    if (
        likelihood.measurements_file != MEAN_ROLE
        or likelihood.cov_file != COV_ROLE
        or float(likelihood.rs_fid) != 1.0
        or float(likelihood.rs_rescale) != 1.0
    ):
        raise InvalidExecution("released Cobaya likelihood initialization drifted")

    requirements = likelihood.get_requirements()
    if set(requirements) != {"angular_diameter_distance", "Hubble", "rdrag"}:
        raise InvalidExecution(f"unexpected Cobaya BAO requirements: {requirements!r}")

    provider = FlatLambdaBaoProvider(OMEGA_M, H_R_D_MPC, RDRAG_GAUGE_MPC)
    likelihood.initialize_with_provider(provider)

    LIKELIHOOD_CALLS += 1
    logp = float(likelihood.logp())
    if LIKELIHOOD_CALLS != 1:
        raise InvalidExecution("likelihood call count differs from authorization budget")
    if not math.isfinite(logp):
        raise InvalidExecution("Cobaya returned non-finite BAO logp")

    chi2 = -2.0 * logp
    predictions: list[float] = []
    for _, row in likelihood.data.iterrows():
        value = float(likelihood.theory_fun(float(row["z"]), str(row["observable"])))
        if not math.isfinite(value):
            raise InvalidExecution("non-finite Cobaya prediction")
        predictions.append(value)
    if len(predictions) != 13:
        raise InvalidExecution("Cobaya prediction vector does not have 13 entries")

    observed = likelihood.data["value"].to_numpy(dtype=float)
    prediction_array = np.asarray(predictions, dtype=float)
    residual = prediction_array - observed
    residual_chi2 = float(residual.dot(likelihood.invcov).dot(residual))
    if not math.isfinite(residual_chi2):
        raise InvalidExecution("non-finite residual chi2")
    if abs(residual_chi2 - chi2) > INTERNAL_CHI2_TOLERANCE:
        raise InvalidExecution(
            "Cobaya logp and explicit residual/covariance chi2 disagree beyond "
            "the frozen engineering tolerance"
        )

    if "camb" in sys.modules or "cobaya.theories.camb" in sys.modules:
        raise InvalidExecution("CAMB was imported during A1C evaluation")

    return (
        logp,
        chi2,
        predictions,
        _canonical_float_vector_sha256(predictions),
        _sha256(yaml_bytes),
        _sha256(_read_regular(class_path)),
    )


def _write_result(result: dict[str, Any]) -> None:
    json.dump(
        result,
        sys.stdout,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    sys.stdout.write("\n")
    sys.stdout.flush()


def _invalid(error: Exception) -> int:
    _write_result(
        {
            "protocol": PROTOCOL,
            "verdict": "INVALID",
            "scientific_claim": "NONE",
            "authority": AUTHORITY,
            "likelihood_call_count": LIKELIHOOD_CALLS,
            "error": str(error),
        }
    )
    return 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("authorization_receipt", type=Path)
    parser.add_argument("a1q_receipt", type=Path)
    parser.add_argument("a1e_receipt", type=Path)
    parser.add_argument("a1c_contract_receipt", type=Path)
    parser.add_argument("dataset_mean", type=Path)
    parser.add_argument("dataset_covariance", type=Path)
    args = parser.parse_args()

    try:
        root, head_before, tree_before = _repository_identity()
        _require_clean_tracked_worktree()

        auth_bytes, auth = _read_json(args.authorization_receipt)
        authorization_sha256 = _validate_authorization(
            auth_bytes, auth, root, head_before, tree_before
        )
        transitive = _validate_transitive_receipts(
            auth, args.a1q_receipt, args.a1e_receipt, args.a1c_contract_receipt
        )
        executor_spec, executor_spec_sha256 = _validate_executor_spec(root)
        _, manifest_sha256 = _validate_manifest(root)

        if manifest_sha256 != _require_str(auth, "a1c_manifest_sha256"):
            raise InvalidExecution("authorization and canonical A1C manifest diverged")

        _validate_a0_file(args.dataset_mean, MEAN_ROLE, MEAN_SIZE, MEAN_SHA256)
        _validate_a0_file(args.dataset_covariance, COV_ROLE, COV_SIZE, COV_SHA256)

        logp, chi2, predictions, prediction_sha256, yaml_sha256, class_source_sha256 = (
            _execute_likelihood(args.dataset_mean, args.dataset_covariance)
        )

        head_after = _run("git", "rev-parse", "HEAD")
        tree_after = _run("git", "rev-parse", "HEAD^{tree}")
        _require_clean_tracked_worktree()
        if head_after != head_before or tree_after != tree_before:
            raise InvalidExecution("repository HEAD/TREE changed during A1C execution")

        absolute_delta = abs(chi2 - REFERENCE_CHI2)
        verdict = "PASS" if absolute_delta <= REPRODUCTION_TOLERANCE else "NEGATIVE"

        result = {
            "protocol": PROTOCOL,
            "verdict": verdict,
            "scientific_claim": "NONE",
            "authority": AUTHORITY,
            "subject_head": head_before,
            "subject_tree": tree_before,
            "authorization_receipt_sha256": authorization_sha256,
            "authorization_spec_sha256": _require_str(auth, "authorization_spec_sha256"),
            "executor_spec_sha256": executor_spec_sha256,
            "executor_script_sha256": _require_str(executor_spec, "script_sha256"),
            "a1c_manifest_sha256": manifest_sha256,
            "a1q_receipt_sha256": _require_str(auth, "a1q_receipt_sha256"),
            "a1e_receipt_sha256": _require_str(auth, "a1e_receipt_sha256"),
            "a1c_contract_receipt_sha256": _require_str(
                auth, "a1c_contract_receipt_sha256"
            ),
            "a0_receipt_sha256": transitive["a0_receipt_sha256"],
            "environment_receipt_sha256": transitive[
                "environment_qualification_receipt_sha256"
            ],
            "environment_versions_sha256": transitive["environment_versions_sha256"],
            "point_manifest_sha256": transitive["point_manifest_sha256"],
            "cobaya_version": COBAYA_VERSION,
            "cobaya_source_commit": COBAYA_SOURCE_COMMIT,
            "installed_likelihood_yaml_sha256": yaml_sha256,
            "installed_likelihood_class_source_sha256": class_source_sha256,
            "dataset_mean_sha256": MEAN_SHA256,
            "dataset_covariance_sha256": COV_SHA256,
            "omega_m": OMEGA_M,
            "h_r_d_mpc": H_R_D_MPC,
            "rdrag_gauge_mpc": RDRAG_GAUGE_MPC,
            "h0_gauge_km_s_mpc": 100.0 * H_R_D_MPC / RDRAG_GAUGE_MPC,
            "likelihood_call_budget": 1,
            "likelihood_call_count": LIKELIHOOD_CALLS,
            "logp_bao": logp,
            "chi2_bao": chi2,
            "reference_chi2_bao": REFERENCE_CHI2,
            "absolute_delta_chi2": absolute_delta,
            "absolute_tolerance": REPRODUCTION_TOLERANCE,
            "internal_chi2_tolerance": INTERNAL_CHI2_TOLERANCE,
            "prediction_vector": predictions,
            "prediction_vector_sha256": prediction_sha256,
            "a1q_reproduction_verdict": _require_str(auth, "a1q_reproduction_verdict"),
        }
        _write_result(result)
        return 0 if verdict == "PASS" else 1
    except (
        InvalidExecution,
        OSError,
        subprocess.CalledProcessError,
        ValueError,
        TypeError,
        KeyError,
        ImportError,
    ) as exc:
        return _invalid(exc)


if __name__ == "__main__":
    raise SystemExit(main())
