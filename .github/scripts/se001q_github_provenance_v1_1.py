#!/usr/bin/env python3
import argparse, hashlib, json, re
from pathlib import Path

DOMAIN = "symthaea.se001q.github-provenance.v1.1"
SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def sha_bytes(data):
    return "sha256:" + hashlib.sha256(data).hexdigest()


def die(message):
    raise SystemExit(message)


def load_json(path):
    raw = Path(path).read_bytes()
    return json.loads(raw), raw


def reject(value, path="$"):
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            if key == "repair_authority":
                die(f"forbidden repair_authority at {child_path}")
            if key == "repair_authority_claim" and child != "NONE":
                die(f"repair authority violation at {child_path}")
            if key == "qualification_claim" and child != "NONE":
                die(f"qualification authority violation at {child_path}")
            reject(child, child_path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            reject(child, f"{path}[{index}]")


def one(items, predicate, what):
    matches = [item for item in items if predicate(item)]
    if len(matches) != 1:
        die(f"expected exactly one {what}, found {len(matches)}")
    return matches[0]


def main():
    ap = argparse.ArgumentParser()
    for name in (
        "contract",
        "run-json",
        "jobs-json",
        "artifacts-json",
        "artifact-archive",
        "expected-run-id",
        "expected-pr-number",
        "expected-head-sha",
        "expected-workflow-path",
        "expected-job-name",
        "expected-artifact-name",
        "output",
    ):
        ap.add_argument("--" + name, required=True)
    ns = ap.parse_args()

    contract, contract_bytes = load_json(ns.contract)
    run, run_bytes = load_json(ns.run_json)
    jobs, jobs_bytes = load_json(ns.jobs_json)
    artifacts, artifacts_bytes = load_json(ns.artifacts_json)
    archive_path = Path(ns.artifact_archive)
    archive_bytes = archive_path.read_bytes()

    for obj in (contract, run, jobs, artifacts):
        reject(obj)

    if contract.get("schema") != "symthaea.se001q.github-provenance-contract.v1.1":
        die("bad contract schema")
    if contract.get("domain") != DOMAIN:
        die("bad contract domain")
    if contract.get("authority", {}).get("sufficient_for_repair_grant") is not False:
        die("contract authority violation")

    run_id = int(ns.expected_run_id)
    pr_number = int(ns.expected_pr_number)
    run_attempt = int(run.get("run_attempt", 0))

    if (
        run.get("id") != run_id
        or run.get("event") != contract["required_event"]
        or run.get("status") != contract["required_run_status"]
        or run.get("conclusion") != contract["required_run_conclusion"]
    ):
        die("run state mismatch")
    if run.get("head_sha") != ns.expected_head_sha or run.get("path") != ns.expected_workflow_path:
        die("run identity mismatch")
    if run_attempt < 1:
        die("bad run attempt")

    pr = one(run.get("pull_requests") or [], lambda item: item.get("number") == pr_number, "matching pull request")
    if (pr.get("head") or {}).get("sha") != ns.expected_head_sha:
        die("PR head mismatch")

    job = one(
        jobs.get("jobs") or [],
        lambda item: item.get("name") == ns.expected_job_name and item.get("run_id") == run_id,
        "matching job",
    )
    if job.get("status") != contract["required_job_status"] or job.get("conclusion") != contract["required_job_conclusion"]:
        die("job state mismatch")
    if contract.get("require_job_attempt_match") and int(job.get("run_attempt", 0)) != run_attempt:
        die("job attempt mismatch")
    steps = job.get("steps")
    if contract.get("require_nonempty_job_steps") and not steps:
        die("job steps missing")

    artifact = one(
        artifacts.get("artifacts") or [],
        lambda item: item.get("name") == ns.expected_artifact_name,
        "matching artifact",
    )
    if contract.get("require_artifact_not_expired") and artifact.get("expired") is not False:
        die("artifact expired")
    if int(artifact.get("size_in_bytes", 0)) <= 0:
        die("artifact empty")

    api_digest = artifact.get("digest")
    if not isinstance(api_digest, str) or not SHA256_RE.fullmatch(api_digest):
        die("artifact digest invalid")
    if not api_digest.startswith(contract["require_artifact_digest_prefix"]):
        die("artifact digest prefix mismatch")

    workflow_run = artifact.get("workflow_run")
    if contract.get("require_artifact_workflow_run_binding"):
        if not isinstance(workflow_run, dict):
            die("artifact workflow_run binding missing")
        if workflow_run.get("id") != run_id:
            die("artifact run mismatch")
        if workflow_run.get("head_sha") != ns.expected_head_sha:
            die("artifact head mismatch")
        if workflow_run.get("head_branch") is not None and workflow_run.get("head_branch") != run.get("head_branch"):
            die("artifact branch mismatch")

    if not archive_bytes:
        die("downloaded artifact archive empty")
    archive_digest = sha_bytes(archive_bytes)
    if contract.get("require_artifact_archive_digest_match") and archive_digest != api_digest:
        die("downloaded artifact archive digest mismatch")

    snapshots = {
        "run_json_sha256": sha_bytes(run_bytes),
        "jobs_json_sha256": sha_bytes(jobs_bytes),
        "artifacts_json_sha256": sha_bytes(artifacts_bytes),
    }
    if contract.get("require_distinct_api_snapshots") and len(set(snapshots.values())) != 3:
        die("snapshot digest collision")

    identity = {
        "domain": DOMAIN,
        "source": contract["source"],
        "contract_sha256": sha_bytes(contract_bytes),
        "run": {
            "id": str(run_id),
            "attempt": str(run_attempt),
            "event": run.get("event"),
            "status": run.get("status"),
            "conclusion": run.get("conclusion"),
            "head_sha": run.get("head_sha"),
            "head_branch": run.get("head_branch"),
            "workflow_path": run.get("path"),
            "pr_number": str(pr_number),
        },
        "job": {
            "id": str(job.get("id")),
            "attempt": str(job.get("run_attempt")),
            "name": job.get("name"),
            "status": job.get("status"),
            "conclusion": job.get("conclusion"),
            "steps": [
                {
                    "name": step.get("name"),
                    "status": step.get("status"),
                    "conclusion": step.get("conclusion"),
                    "number": step.get("number"),
                }
                for step in (steps or [])
            ],
        },
        "artifact": {
            "id": str(artifact.get("id")),
            "name": artifact.get("name"),
            "size_in_bytes": artifact.get("size_in_bytes"),
            "api_digest": api_digest,
            "downloaded_archive_sha256": archive_digest,
            "downloaded_archive_bytes": len(archive_bytes),
            "expired": artifact.get("expired"),
            "workflow_run_id": str(workflow_run.get("id")),
            "workflow_run_head_sha": workflow_run.get("head_sha"),
            "workflow_run_head_branch": workflow_run.get("head_branch"),
        },
        "api_snapshots": snapshots,
        "provenance_strength": "API_CONSISTENCY_PLUS_ARTIFACT_DIGEST_BINDING_NOT_GITHUB_SIGNATURE",
        "authority": {
            "meaning": "execution provenance and downloaded-artifact transport consistency only",
            "sufficient_for_repair_grant": False,
            "qualification_claim": "NONE",
            "repair_authority_claim": "NONE",
        },
    }
    output = {
        "schema": DOMAIN,
        "provenance_witness_id": sha_bytes(canonical(identity)),
        "identity": identity,
    }
    reject(output)

    output_path = Path(ns.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "schema": DOMAIN,
                "result": "PASS",
                "provenance_witness_id": output["provenance_witness_id"],
                "run_id": str(run_id),
                "run_attempt": str(run_attempt),
                "artifact_id": str(artifact.get("id")),
                "artifact_digest": api_digest,
                "downloaded_archive_sha256": archive_digest,
                "sufficient_for_repair_grant": False,
                "qualification_claim": "NONE",
                "repair_authority_claim": "NONE",
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
