#!/usr/bin/env python3
import hashlib
import json
import math
from pathlib import Path

CORPUS = Path("docs/release/evidence/ion-001b-synthetic-transport-corpus-v1.json")
EXPECTED_SCHEMA = "ion-001b-synthetic-transport-corpus-v1"
EXPECTED_ARCH = "6f235ec28910050fd111eb7021153ed4ffe72668"
EXPECTED_SHA256 = "aac932a68fe18c7614cae0868bc83613e4c1b07ffd0c72ede96e9654f9e047e6"
EXPECTED_GIT_BLOB = "2585fdc0f1b257f56a260da84bb32ef5935f1374"
EXPECTED_VECTOR_KEYS = [
    "subject_identity_valid",
    "neb_barrier_admissible",
    "trajectory_transport_admissible",
    "diffusion_admissible",
    "ionic_conductivity_admissible",
    "experimental_conductivity_admissible",
    "transport_capability_supported",
    "replication_supported",
    "scope_transfer_supported",
    "selectivity_supported",
    "prospective_scoring_supported",
    "new_lineage_required",
]
EXPECTED_CASE_IDS = [f"ION-001B-{i:02d}" for i in range(1, 28)]


def fail(message):
    raise SystemExit(f"ERROR: {message}")


def git_blob_sha(raw):
    header = f"blob {len(raw)}\0".encode()
    return hashlib.sha1(header + raw).hexdigest()


def derive(f):
    out = {key: False for key in EXPECTED_VECTOR_KEYS}
    subject = f.get("subject_state_bound") is True
    out["subject_identity_valid"] = subject

    out["new_lineage_required"] = subject and any(
        f.get(key) is True
        for key in (
            "subject_state_changed_after_commitment",
            "analysis_window_changed_after_result",
            "eis_fit_profile_changed_after_normalization",
            "geometry_factor_changed_after_result",
        )
    )

    out["neb_barrier_admissible"] = (
        subject
        and f.get("neb_converged") is True
        and f.get("path_endpoints_match") is True
        and f.get("barrier_profile_bound") is True
    )

    transport = (
        subject
        and f.get("trajectory_complete") is True
        and f.get("model_applicable") is True
        and f.get("transport_calibrated") is True
    )
    out["transport_capability_supported"] = transport
    out["trajectory_transport_admissible"] = transport

    diffusion = (
        transport
        and f.get("long_time_linear_regime") is True
        and f.get("carrier_events_sufficient") is True
    )
    out["diffusion_admissible"] = diffusion

    conductivity = (
        diffusion
        and f.get("conductivity_profile") == "nernst_einstein_uncorrelated_v1"
        and all(k in f for k in ("carrier_density_m3", "species_charge_c", "diffusion_m2_s", "temperature_k"))
        and f.get("temperature_k", 0) > 0
    )
    out["ionic_conductivity_admissible"] = conductivity

    experimental = (
        subject
        and f.get("experimental_sample_lineage_bound") is True
        and f.get("raw_eis_artifact_bound") is True
        and f.get("instrument_calibration_bound") is True
        and f.get("eis_fit_profile_bound") is True
        and f.get("geometry_factor_bound") is True
        and f.get("eis_fit_profile_changed_after_normalization") is not True
        and f.get("geometry_factor_changed_after_result") is not True
    )
    out["experimental_conductivity_admissible"] = experimental

    out["replication_supported"] = (
        transport
        and f.get("replicate_count", 0) >= 2
        and f.get("independent_initializations", 0) >= 2
    )

    out["scope_transfer_supported"] = (
        subject
        and f.get("scope") is not None
        and f.get("scope") == f.get("requested_scope")
        and (conductivity or experimental)
    )

    out["selectivity_supported"] = (
        subject
        and f.get("competitive_multi_species_evidence") is True
        and f.get("selectivity_profile_bound") is True
    )

    out["prospective_scoring_supported"] = (
        subject
        and f.get("prospective_commitment_precedes_outcome") is True
        and f.get("complete_terminal_census") is True
    )

    return out


def expected_metrics(f, derived):
    out = {}
    if derived["ionic_conductivity_admissible"]:
        kb = 1.380649e-23
        n = float(f["carrier_density_m3"])
        q = float(f["species_charge_c"])
        diffusion = float(f["diffusion_m2_s"])
        temperature = float(f["temperature_k"])
        out["ionic_conductivity_s_m"] = n * q * q * diffusion / (kb * temperature)
    return out


def main():
    raw = CORPUS.read_bytes()
    sha256 = hashlib.sha256(raw).hexdigest()
    blob = git_blob_sha(raw)
    if sha256 != EXPECTED_SHA256:
        fail(f"SHA-256 mismatch: {sha256}")
    if blob != EXPECTED_GIT_BLOB:
        fail(f"Git blob mismatch: {blob}")

    doc = json.loads(raw)
    if doc.get("schema") != EXPECTED_SCHEMA:
        fail("schema mismatch")
    if doc.get("architecture_head") != EXPECTED_ARCH:
        fail("architecture head mismatch")
    if doc.get("authority") != "representation_only_no_scientific_authority":
        fail("authority mismatch")
    if doc.get("vector_keys") != EXPECTED_VECTOR_KEYS:
        fail("vector key/order mismatch")

    cases = doc.get("cases")
    if not isinstance(cases, list) or [c.get("id") for c in cases] != EXPECTED_CASE_IDS:
        fail("case census/order mismatch")

    for case in cases:
        facts = case.get("facts")
        if not isinstance(facts, dict):
            fail(f"{case.get('id')}: missing facts")
        derived = derive(facts)
        expected = set(case.get("expected_true", []))
        actual = {k for k, value in derived.items() if value}
        if actual != expected:
            fail(f"{case['id']}: vector mismatch actual={sorted(actual)} expected={sorted(expected)}")

        metrics = expected_metrics(facts, derived)
        recorded = case.get("expected_metrics", {})
        if set(metrics) != set(recorded):
            fail(f"{case['id']}: metric-key mismatch actual={sorted(metrics)} expected={sorted(recorded)}")
        for key, value in metrics.items():
            if not math.isclose(value, float(recorded[key]), rel_tol=1e-12, abs_tol=1e-15):
                fail(f"{case['id']}: {key} mismatch actual={value!r} expected={recorded[key]!r}")

    print(f"ok cases={len(cases)} sha256={sha256} git_blob={blob}")


if __name__ == "__main__":
    main()
