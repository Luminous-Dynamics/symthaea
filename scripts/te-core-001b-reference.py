#!/usr/bin/env python3
import hashlib
import json
import math
from pathlib import Path

CORPUS = Path("docs/release/evidence/te-core-001b-synthetic-evidence-corpus-v1.json")
EXPECTED_SCHEMA = "te-core-001b-synthetic-evidence-corpus-v1"
EXPECTED_ARCH = "5731c227d0822568908df79dccdc74a5f23b1b62"
EXPECTED_SHA256 = "24e395f3b821ce00487e70c52941678825f7fc20a730d7993aa3a7ba8b8b0620"
EXPECTED_GIT_BLOB = "7a177f9a63d404f4a90a9cd8a7c945a9410b566f"
EXPECTED_VECTOR_KEYS = [
    "subject_identity_valid",
    "power_factor_admissible",
    "zt_derivation_admissible",
    "electronic_transport_admissible",
    "lattice_thermal_transport_admissible",
    "stability_compatible",
    "experimental_composition_admissible",
    "holdout_classification_valid",
    "prospective_history_admissible",
    "new_lineage_required",
]
EXPECTED_CASE_IDS = [f"TE-CORE-001B-{i:02d}" for i in range(1, 28)]


def fail(message):
    raise SystemExit(f"ERROR: {message}")


def git_blob_sha(raw):
    return hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()


def derive(f):
    out = {key: False for key in EXPECTED_VECTOR_KEYS}
    subject = f.get("subject_state_bound") is True
    out["subject_identity_valid"] = subject

    electronic = (
        subject
        and f.get("electronic_transport_profile_bound") is True
        and f.get("scattering_profile_bound") is True
        and f.get("sigma_is_absolute") is True
    )
    out["electronic_transport_admissible"] = electronic

    compatible = (
        subject
        and f.get("same_material_state") is True
        and f.get("same_temperature") is True
        and f.get("same_carrier_state") is True
        and (
            f.get("same_transport_direction") is True
            or f.get("tensor_mixing_profile_allows") is True
        )
    )
    power_factor = (
        compatible
        and electronic
        and "seebeck_v_k" in f
        and "sigma_s_m" in f
    )
    out["power_factor_admissible"] = power_factor

    lattice = (
        subject
        and f.get("lattice_transport_profile_bound") is True
        and f.get("qmesh_converged") is True
    )
    out["lattice_thermal_transport_admissible"] = lattice
    out["stability_compatible"] = subject and f.get("dynamic_stability_supported") is True

    if (
        power_factor
        and lattice
        and all(key in f for key in ("kappa_e_w_mk", "kappa_l_w_mk", "temperature_k"))
        and float(f["temperature_k"]) > 0
        and float(f["kappa_e_w_mk"]) + float(f["kappa_l_w_mk"]) > 0
    ):
        pf = float(f["seebeck_v_k"]) ** 2 * float(f["sigma_s_m"])
        zt = pf * float(f["temperature_k"]) / (
            float(f["kappa_e_w_mk"]) + float(f["kappa_l_w_mk"])
        )
        stored_mismatch = (
            "stored_zt" in f
            and not math.isclose(float(f["stored_zt"]), zt, rel_tol=1e-12, abs_tol=1e-15)
        )
        out["zt_derivation_admissible"] = not stored_mismatch

    out["experimental_composition_admissible"] = (
        subject
        and f.get("experimental_components_present") is True
        and f.get("same_specimen_or_declared_compatibility") is True
        and f.get("measurement_profiles_bound") is True
        and f.get("same_temperature") is True
        and f.get("same_carrier_state") is True
        and f.get("same_transport_direction") is True
    )

    if f.get("claimed_unseen_material_holdout") is True:
        train = set(f.get("train_parent_compounds", []))
        test = set(f.get("test_parent_compounds", []))
        out["holdout_classification_valid"] = (
            subject
            and bool(train)
            and bool(test)
            and train.isdisjoint(test)
            and f.get("split_only_by_temperature") is not True
        )

    out["prospective_history_admissible"] = (
        subject
        and f.get("prospective_complete_history") is True
        and f.get("filtered_candidate_omitted") is not True
    )
    out["new_lineage_required"] = (
        subject and f.get("scattering_profile_changed_after_result") is True
    )
    return out


def metrics(f, derived):
    out = {}
    if derived["power_factor_admissible"]:
        out["power_factor_w_mk2"] = float(f["seebeck_v_k"]) ** 2 * float(f["sigma_s_m"])
    if derived["zt_derivation_admissible"]:
        out["zt"] = (
            out["power_factor_w_mk2"]
            * float(f["temperature_k"])
            / (float(f["kappa_e_w_mk"]) + float(f["kappa_l_w_mk"]))
        )
    return out


def validate(raw):
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
    if not isinstance(cases, list) or [case.get("id") for case in cases] != EXPECTED_CASE_IDS:
        fail("case census/order mismatch")

    for case in cases:
        facts = case.get("facts")
        if not isinstance(facts, dict):
            fail(f"{case.get('id')}: missing facts")
        derived = derive(facts)
        actual = {key for key, value in derived.items() if value}
        expected = set(case.get("expected_true", []))
        if actual != expected:
            fail(f"{case['id']}: vector mismatch actual={sorted(actual)} expected={sorted(expected)}")

        actual_metrics = metrics(facts, derived)
        expected_metrics = case.get("expected_metrics", {})
        if set(actual_metrics) != set(expected_metrics):
            fail(f"{case['id']}: metric-key mismatch actual={sorted(actual_metrics)} expected={sorted(expected_metrics)}")
        for key, value in actual_metrics.items():
            if not math.isclose(value, float(expected_metrics[key]), rel_tol=1e-12, abs_tol=1e-15):
                fail(f"{case['id']}: {key} mismatch actual={value!r} expected={expected_metrics[key]!r}")

    return len(cases), sha256, blob


def main():
    cases, sha256, blob = validate(CORPUS.read_bytes())
    print(f"ok cases={cases} sha256={sha256} git_blob={blob}")


if __name__ == "__main__":
    main()
