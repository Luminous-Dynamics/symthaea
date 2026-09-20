#!/usr/bin/env python3
import argparse
import hashlib
import json
import re
from pathlib import Path

DOMAIN = "symthaea.se001q.repair-evidence-join.v1"
SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def sha_bytes(data):
    return "sha256:" + hashlib.sha256(data).hexdigest()


def die(message):
    raise SystemExit(message)


def load_json(path):
    raw = Path(path).read_bytes()
    return json.loads(raw), raw


def reject_authority(value, path="$" ):
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            if key == "repair_authority":
                die(f"forbidden repair_authority field at {child_path}")
            if key == "repair_authority_claim" and child != "NONE":
                die(f"repair authority violation at {child_path}")
            if key == "qualification_claim" and child != "NONE":
                die(f"qualification authority violation at {child_path}")
            if key == "sufficient_for_repair_grant" and child is not False:
                die(f"repair-grant sufficiency violation at {child_path}")
            reject_authority(child, child_path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            reject_authority(child, f"{path}[{index}]")


def require_non_authorizing_authority(container, what):
    authority = container.get("authority")
    if not isinstance(authority, dict):
        die(f"{what} authority object missing")
    if (
        authority.get("sufficient_for_repair_grant") is not False
        or authority.get("qualification_claim") != "NONE"
        or authority.get("repair_authority_claim") != "NONE"
    ):
        die(f"{what} authority tuple mismatch")
    return authority


def require_sha256(value, what):
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        die(f"invalid {what}")
    return value


def require_git_sha(value, what):
    if not isinstance(value, str) or not GIT_SHA_RE.fullmatch(value):
        die(f"invalid {what}")
    return value


def require_positive_decimal(value, what):
    text = str(value)
    if not text.isdigit() or int(text) <= 0:
        die(f"invalid {what}")
    return text


def verify_content_addressed(obj, schema, id_key, what):
    if obj.get("schema") != schema:
        die(f"{what} schema mismatch")
    identity = obj.get("identity")
    if not isinstance(identity, dict):
        die(f"{what} identity missing")
    expected = sha_bytes(canonical(identity))
    if obj.get(id_key) != expected:
        die(f"{what} content-addressed identity mismatch")
    return identity


def verify_provenance(obj, contract, label):
    identity = verify_content_addressed(
        obj,
        contract["required_provenance_schema"],
        "provenance_witness_id",
        f"provenance {label}",
    )
    reject_authority(obj)
    require_non_authorizing_authority(identity, f"provenance {label}")

    if identity.get("provenance_strength") != contract["required_provenance_strength"]:
        die(f"provenance {label} strength mismatch")

    run = identity.get("run")
    job = identity.get("job")
    artifact = identity.get("artifact")
    if not all(isinstance(item, dict) for item in (run, job, artifact)):
        die(f"provenance {label} missing run/job/artifact identity")

    run_id = require_positive_decimal(run.get("id"), f"provenance {label} run id")
    run_attempt = require_positive_decimal(run.get("attempt"), f"provenance {label} run attempt")
    pr_number = require_positive_decimal(run.get("pr_number"), f"provenance {label} PR number")
    head_sha = require_git_sha(run.get("head_sha"), f"provenance {label} head SHA")

    if run.get("event") != "pull_request" or run.get("status") != "completed" or run.get("conclusion") != "success":
        die(f"provenance {label} run state mismatch")
    if run.get("workflow_path") != contract["expected_workflow_path"]:
        die(f"provenance {label} workflow mismatch")

    require_positive_decimal(job.get("id"), f"provenance {label} job id")
    job_attempt = require_positive_decimal(job.get("attempt"), f"provenance {label} job attempt")
    if job_attempt != run_attempt:
        die(f"provenance {label} job/run attempt mismatch")
    if job.get("status") != "completed" or job.get("conclusion") != "success":
        die(f"provenance {label} job state mismatch")
    if not isinstance(job.get("steps"), list) or not job["steps"]:
        die(f"provenance {label} missing job steps")

    artifact_id = require_positive_decimal(artifact.get("id"), f"provenance {label} artifact id")
    workflow_run_id = require_positive_decimal(
        artifact.get("workflow_run_id"),
        f"provenance {label} artifact workflow run id",
    )
    api_size = int(require_positive_decimal(artifact.get("size_in_bytes"), f"provenance {label} API artifact size"))
    archive_size = int(
        require_positive_decimal(
            artifact.get("downloaded_archive_bytes"),
            f"provenance {label} downloaded artifact size",
        )
    )
    if api_size != archive_size:
        die(f"provenance {label} artifact size mismatch")

    api_digest = require_sha256(artifact.get("api_digest"), f"provenance {label} API artifact digest")
    archive_digest = require_sha256(
        artifact.get("downloaded_archive_sha256"),
        f"provenance {label} downloaded artifact digest",
    )
    if api_digest != archive_digest:
        die(f"provenance {label} transport digest mismatch")
    if artifact.get("expired") is not False:
        die(f"provenance {label} artifact expired")
    if workflow_run_id != run_id:
        die(f"provenance {label} artifact/run mismatch")
    if require_git_sha(
        artifact.get("workflow_run_head_sha"),
        f"provenance {label} artifact head SHA",
    ) != head_sha:
        die(f"provenance {label} artifact/head mismatch")

    return {
        "witness_id": obj["provenance_witness_id"],
        "identity": identity,
        "run_id": run_id,
        "run_attempt": run_attempt,
        "pr_number": pr_number,
        "artifact_id": artifact_id,
        "head_sha": head_sha,
        "artifact_digest": api_digest,
        "artifact_bytes": archive_size,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--contract", required=True)
    ap.add_argument("--corroboration", required=True)
    ap.add_argument("--provenance-a", required=True)
    ap.add_argument("--provenance-b", required=True)
    ap.add_argument("--output", required=True)
    ns = ap.parse_args()

    contract, contract_bytes = load_json(ns.contract)
    corroboration, _ = load_json(ns.corroboration)
    provenance_a, _ = load_json(ns.provenance_a)
    provenance_b, _ = load_json(ns.provenance_b)

    reject_authority(contract)
    if contract.get("schema") != "symthaea.se001q.repair-evidence-join-contract.v1":
        die("bad join contract schema")
    if contract.get("domain") != DOMAIN:
        die("bad join contract domain")
    require_non_authorizing_authority(contract, "join contract")
    expected_subject_sha = require_git_sha(contract.get("expected_subject_sha"), "expected subject SHA")
    expected_source_lock = require_sha256(
        contract.get("expected_source_lock_sha256"),
        "expected source lock digest",
    )

    corroboration_identity = verify_content_addressed(
        corroboration,
        contract["required_corroboration_schema"],
        "corroboration_id",
        "corroboration",
    )
    reject_authority(corroboration)
    require_non_authorizing_authority(corroboration_identity, "corroboration")

    if corroboration_identity.get("result") != contract["required_corroboration_result"]:
        die("corroboration result mismatch")
    if corroboration_identity.get("subject_sha") != expected_subject_sha:
        die("corroboration subject mismatch")
    if corroboration_identity.get("source_lock_sha256") != expected_source_lock:
        die("corroboration source lock mismatch")
    if corroboration_identity.get("experiment_id") != contract["expected_experiment_id"]:
        die("corroboration experiment mismatch")
    experiment_sha = require_sha256(
        corroboration_identity.get("experiment_sha256"),
        "corroboration experiment digest",
    )
    generated_lock = require_sha256(
        corroboration_identity.get("generated_lock_sha256"),
        "corroboration generated lock digest",
    )
    if generated_lock == expected_source_lock:
        die("corroboration generated lock equals source lock")

    runs = corroboration_identity.get("runs")
    if not isinstance(runs, list) or len(runs) != 2:
        die("corroboration must contain exactly two runs")

    normalized_runs = []
    for index, run in enumerate(runs):
        if not isinstance(run, dict):
            die(f"corroboration run {index} malformed")
        normalized_runs.append(
            {
                **run,
                "run_id": require_positive_decimal(run.get("run_id"), f"corroboration run {index} id"),
                "artifact_id": require_positive_decimal(
                    run.get("artifact_id"),
                    f"corroboration run {index} artifact id",
                ),
                "verifier_sha": require_git_sha(
                    run.get("verifier_sha"),
                    f"corroboration run {index} verifier SHA",
                ),
                "artifact_zip_sha256": require_sha256(
                    run.get("artifact_zip_sha256"),
                    f"corroboration run {index} artifact digest",
                ),
                "observation_id": require_sha256(
                    run.get("observation_id"),
                    f"corroboration run {index} observation id",
                ),
                "manifest_id": require_sha256(
                    run.get("manifest_id"),
                    f"corroboration run {index} manifest id",
                ),
                "lock_delta_witness_id": require_sha256(
                    run.get("lock_delta_witness_id"),
                    f"corroboration run {index} lock witness id",
                ),
            }
        )

    if len({run["run_id"] for run in normalized_runs}) != 2:
        die("corroboration run IDs are not distinct")
    if len({run["artifact_id"] for run in normalized_runs}) != 2:
        die("corroboration artifact IDs are not distinct")
    if len({run["verifier_sha"] for run in normalized_runs}) != 2:
        die("corroboration verifier generations are not distinct")

    prov = [
        verify_provenance(provenance_a, contract, "A"),
        verify_provenance(provenance_b, contract, "B"),
    ]
    if prov[0]["witness_id"] == prov[1]["witness_id"]:
        die("provenance witnesses are not distinct")
    if prov[0]["run_id"] == prov[1]["run_id"]:
        die("provenance run IDs are not distinct")
    if prov[0]["artifact_id"] == prov[1]["artifact_id"]:
        die("provenance artifact IDs are not distinct")

    provenance_by_run = {item["run_id"]: item for item in prov}
    if len(provenance_by_run) != 2:
        die("provenance run map is not one-to-one")

    pairings = []
    for run in sorted(normalized_runs, key=lambda item: item["run_id"]):
        run_id = run["run_id"]
        if run_id not in provenance_by_run:
            die(f"missing provenance witness for corroborated run {run_id}")
        p = provenance_by_run[run_id]

        if p["artifact_id"] != run["artifact_id"]:
            die(f"artifact ID mismatch for run {run_id}")
        if p["head_sha"] != run["verifier_sha"]:
            die(f"verifier/head SHA mismatch for run {run_id}")
        if p["artifact_digest"] != run["artifact_zip_sha256"]:
            die(f"artifact digest mismatch for run {run_id}")

        pairings.append(
            {
                "run_id": run_id,
                "run_attempt": p["run_attempt"],
                "pr_number": p["pr_number"],
                "artifact_id": run["artifact_id"],
                "artifact_bytes": p["artifact_bytes"],
                "verifier_sha": run["verifier_sha"],
                "artifact_zip_sha256": run["artifact_zip_sha256"],
                "provenance_witness_id": p["witness_id"],
                "observation_id": run["observation_id"],
                "manifest_id": run["manifest_id"],
                "lock_delta_witness_id": run["lock_delta_witness_id"],
                "verification_schema": run.get("verification_schema"),
                "capture_protocol": run.get("capture_protocol"),
            }
        )

    if {item["run_id"] for item in pairings} != set(provenance_by_run):
        die("unused provenance witness")

    identity = {
        "domain": DOMAIN,
        "contract_sha256": sha_bytes(contract_bytes),
        "corroboration_id": corroboration["corroboration_id"],
        "subject_sha": expected_subject_sha,
        "experiment_id": corroboration_identity["experiment_id"],
        "experiment_sha256": experiment_sha,
        "source_lock_sha256": expected_source_lock,
        "generated_lock_sha256": generated_lock,
        "pairings": pairings,
        "candidate_lock_transform": {
            "path": "Cargo.lock",
            "source_sha256": expected_source_lock,
            "generated_sha256": generated_lock,
            "authorized": False,
        },
        "result": contract["result"],
        "authority": {
            "meaning": contract["authority"]["meaning"],
            "sufficient_for_repair_grant": False,
            "qualification_claim": "NONE",
            "repair_authority_claim": "NONE",
        },
    }
    output = {
        "schema": DOMAIN,
        "repair_evidence_join_id": sha_bytes(canonical(identity)),
        "identity": identity,
    }
    reject_authority(output)
    require_non_authorizing_authority(identity, "repair evidence join")

    output_path = Path(ns.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "schema": DOMAIN,
                "result": identity["result"],
                "repair_evidence_join_id": output["repair_evidence_join_id"],
                "corroboration_id": identity["corroboration_id"],
                "source_lock_sha256": identity["source_lock_sha256"],
                "generated_lock_sha256": identity["generated_lock_sha256"],
                "sufficient_for_repair_grant": False,
                "qualification_claim": "NONE",
                "repair_authority_claim": "NONE",
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
