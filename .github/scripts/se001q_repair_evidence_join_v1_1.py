#!/usr/bin/env python3
import argparse, base64, hashlib, json, re, zipfile
from pathlib import Path

DOMAIN = "symthaea.se001q.repair-evidence-join.v1.1"
SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")
GITSHA = re.compile(r"^[0-9a-f]{40}$")


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def digest(data):
    return "sha256:" + hashlib.sha256(data).hexdigest()


def die(msg):
    raise SystemExit(msg)


def load(path):
    raw = Path(path).read_bytes()
    return json.loads(raw), raw


def reject_authority(value, path="$" ):
    if isinstance(value, dict):
        for key, child in value.items():
            here = f"{path}.{key}"
            if key == "repair_authority":
                die(f"forbidden repair_authority field at {here}")
            if key == "repair_authority_claim" and child != "NONE":
                die(f"repair authority violation at {here}")
            if key == "qualification_claim" and child != "NONE":
                die(f"qualification authority violation at {here}")
            if key == "sufficient_for_repair_grant" and child is not False:
                die(f"repair-grant sufficiency violation at {here}")
            reject_authority(child, here)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            reject_authority(child, f"{path}[{index}]")


def non_authorizing(obj, what):
    authority = obj.get("authority")
    if not isinstance(authority, dict):
        die(f"{what} authority object missing")
    if (
        authority.get("sufficient_for_repair_grant") is not False
        or authority.get("qualification_claim") != "NONE"
        or authority.get("repair_authority_claim") != "NONE"
    ):
        die(f"{what} authority tuple mismatch")


def sha256(value, what):
    if not isinstance(value, str) or not SHA256.fullmatch(value):
        die(f"invalid {what}")
    return value


def gitsha(value, what):
    if not isinstance(value, str) or not GITSHA.fullmatch(value):
        die(f"invalid {what}")
    return value


def positive_decimal(value, what):
    value = str(value)
    if not value.isdigit() or int(value) <= 0:
        die(f"invalid {what}")
    return value


def content_addressed(obj, schema, id_key, what):
    if obj.get("schema") != schema:
        die(f"{what} schema mismatch")
    identity = obj.get("identity")
    if not isinstance(identity, dict):
        die(f"{what} identity missing")
    if obj.get(id_key) != digest(canonical(identity)):
        die(f"{what} content-addressed identity mismatch")
    return identity


def verify_diagnostic_witnesses(doc, archive_path, contract):
    reject_authority(doc)
    if doc.get("schema") != contract["required_diagnostic_witness_set_schema"]:
        die("diagnostic witness set schema mismatch")
    identity = doc.get("identity")
    if not isinstance(identity, dict):
        die("diagnostic witness set identity missing")
    if doc.get("set_id") != digest(canonical(identity)):
        die("diagnostic witness set content-addressed identity mismatch")
    if doc["set_id"] != contract["expected_diagnostic_witness_set_id"]:
        die("diagnostic witness set ID mismatch")
    if identity.get("qualification_claim") != "NONE" or identity.get("repair_authority_claim") != "NONE":
        die("diagnostic witness set authority boundary mismatch")
    if identity.get("subject_sha") != contract["expected_subject_sha"]:
        die("diagnostic witness subject mismatch")
    if identity.get("verifier_sha") != contract["expected_diagnostic_verifier_sha"]:
        die("diagnostic witness verifier mismatch")
    if identity.get("source_evidence_manifest_id") != contract["expected_source_evidence_manifest_id"]:
        die("diagnostic witness source manifest mismatch")

    archive = Path(archive_path)
    archive_bytes = archive.read_bytes()
    archive_sha = digest(archive_bytes)
    if archive_sha != contract["expected_source_artifact_zip_sha256"]:
        die("diagnostic source archive digest mismatch")

    with zipfile.ZipFile(archive) as zf:
        manifest = json.loads(zf.read("evidence/manifest.json"))
        manifest_identity = dict(manifest)
        manifest_id = manifest_identity.pop("manifest_id", None)
        if manifest_id != digest(canonical(manifest_identity)):
            die("diagnostic source evidence manifest identity mismatch")
        if manifest_id != contract["expected_source_evidence_manifest_id"]:
            die("diagnostic source evidence manifest ID mismatch")
        if manifest.get("subject_sha") != contract["expected_subject_sha"]:
            die("diagnostic source manifest subject mismatch")
        if manifest.get("verifier_sha") != contract["expected_diagnostic_verifier_sha"]:
            die("diagnostic source manifest verifier mismatch")

        witnesses = doc.get("witnesses")
        gates = contract["expected_diagnostic_gates"]
        expected_ids = contract["expected_diagnostic_witness_ids"]
        if not isinstance(witnesses, list) or len(witnesses) != len(gates):
            die("diagnostic witness count mismatch")

        actual_ids = []
        actual_gates = []
        for index, witness in enumerate(witnesses):
            if witness.get("schema") != "symthaea.diagnostic-witness.v1":
                die(f"diagnostic witness {index} schema mismatch")
            wi = witness.get("identity")
            if not isinstance(wi, dict):
                die(f"diagnostic witness {index} identity missing")
            witness_id = digest(canonical(wi))
            if witness.get("witness_id") != witness_id:
                die(f"diagnostic witness {index} identity mismatch")
            if witness_id != expected_ids[index]:
                die(f"diagnostic witness {index} unexpected ID")
            actual_ids.append(witness_id)

            source = wi.get("source")
            evidence = wi.get("evidence")
            rule = wi.get("rule")
            conclusion = wi.get("conclusion")
            if not all(isinstance(x, dict) for x in (source, evidence, rule, conclusion)):
                die(f"diagnostic witness {index} malformed")
            gate = source.get("gate_id")
            if gate != gates[index]:
                die(f"diagnostic witness {index} gate mismatch")
            actual_gates.append(gate)

            if source.get("subject_sha") != contract["expected_subject_sha"]:
                die(f"diagnostic witness {gate} subject mismatch")
            if source.get("verifier_sha") != contract["expected_diagnostic_verifier_sha"]:
                die(f"diagnostic witness {gate} verifier mismatch")
            if source.get("original_classification_state") != "FAIL_UNCLASSIFIED":
                die(f"diagnostic witness {gate} rewrites source classification")
            sha256(source.get("observation_id"), f"{gate} observation ID")
            sha256(source.get("classification_id"), f"{gate} classification ID")

            if conclusion.get("failure_class") != contract["required_failure_class"]:
                die(f"diagnostic witness {gate} failure class mismatch")
            if conclusion.get("confidence") != "EXACT_RETAINED_BYTE_MATCH":
                die(f"diagnostic witness {gate} confidence mismatch")
            if conclusion.get("qualification_claim") != "NONE" or conclusion.get("repair_authority_claim") != "NONE":
                die(f"diagnostic witness {gate} authority boundary mismatch")
            if evidence.get("exit_code") != 101 or rule.get("required_exit_code") != 101:
                die(f"diagnostic witness {gate} exit-code mismatch")
            if evidence.get("stream") != "stderr":
                die(f"diagnostic witness {gate} stream mismatch")
            if rule.get("rule_id") != "cargo-locked-update-required-v1":
                die(f"diagnostic witness {gate} rule mismatch")

            excerpt = base64.b64decode(evidence.get("excerpt_base64", ""), validate=True)
            if digest(excerpt) != sha256(evidence.get("excerpt_sha256"), f"{gate} excerpt digest"):
                die(f"diagnostic witness {gate} excerpt hash mismatch")
            text = excerpt.decode("utf-8")
            if not text.startswith("cannot update the lock file "):
                die(f"diagnostic witness {gate} prefix mismatch")
            if not text.endswith(rule.get("required_excerpt_utf8_suffix", "")):
                die(f"diagnostic witness {gate} suffix mismatch")
            byte_range = evidence.get("byte_range")
            if not isinstance(byte_range, dict):
                die(f"diagnostic witness {gate} byte range missing")
            start, end = byte_range.get("start"), byte_range.get("end_exclusive")
            if not isinstance(start, int) or not isinstance(end, int) or start < 0 or end < start:
                die(f"diagnostic witness {gate} byte range invalid")
            if end - start != len(excerpt):
                die(f"diagnostic witness {gate} byte range length mismatch")

            stderr = zf.read(f"evidence/{gate}/stderr.log")
            if digest(stderr) != sha256(evidence.get("stream_sha256"), f"{gate} stderr digest"):
                die(f"diagnostic witness {gate} retained stream hash mismatch")
            if stderr[start:end] != excerpt:
                die(f"diagnostic witness {gate} retained byte slice mismatch")

            provenance = witness.get("provenance")
            if not isinstance(provenance, dict):
                die(f"diagnostic witness {gate} provenance missing")
            if str(provenance.get("github_run_id")) != contract["expected_source_run_id"]:
                die(f"diagnostic witness {gate} source run mismatch")
            if str(provenance.get("github_run_attempt")) != contract["expected_source_run_attempt"]:
                die(f"diagnostic witness {gate} source run attempt mismatch")
            if str(provenance.get("artifact_id")) != contract["expected_source_artifact_id"]:
                die(f"diagnostic witness {gate} source artifact mismatch")
            if provenance.get("artifact_zip_sha256") != archive_sha:
                die(f"diagnostic witness {gate} source artifact digest mismatch")
            if provenance.get("evidence_manifest_id") != manifest_id:
                die(f"diagnostic witness {gate} source manifest provenance mismatch")

    if actual_ids != identity.get("witness_ids"):
        die("diagnostic witness set ID ordering mismatch")
    if actual_gates != contract["expected_diagnostic_gates"]:
        die("diagnostic witness gate ordering mismatch")
    return {
        "set_id": doc["set_id"],
        "failure_class": contract["required_failure_class"],
        "source_run_id": contract["expected_source_run_id"],
        "source_run_attempt": contract["expected_source_run_attempt"],
        "source_artifact_id": contract["expected_source_artifact_id"],
        "source_archive_sha256": archive_sha,
        "source_archive_bytes": len(archive_bytes),
        "source_evidence_manifest_id": contract["expected_source_evidence_manifest_id"],
        "witness_ids": actual_ids,
    }


def verify_provenance(obj, contract, label):
    identity = content_addressed(obj, contract["required_provenance_schema"], "provenance_witness_id", f"provenance {label}")
    reject_authority(obj)
    non_authorizing(identity, f"provenance {label}")
    if identity.get("provenance_strength") != contract["required_provenance_strength"]:
        die(f"provenance {label} strength mismatch")
    run, job, artifact = identity.get("run"), identity.get("job"), identity.get("artifact")
    if not all(isinstance(x, dict) for x in (run, job, artifact)):
        die(f"provenance {label} missing run/job/artifact identity")
    run_id = positive_decimal(run.get("id"), f"provenance {label} run id")
    attempt = positive_decimal(run.get("attempt"), f"provenance {label} run attempt")
    pr = positive_decimal(run.get("pr_number"), f"provenance {label} PR number")
    head = gitsha(run.get("head_sha"), f"provenance {label} head SHA")
    if run.get("event") != "pull_request" or run.get("status") != "completed" or run.get("conclusion") != "success":
        die(f"provenance {label} run state mismatch")
    if run.get("workflow_path") != contract["expected_workflow_path"]:
        die(f"provenance {label} workflow mismatch")
    positive_decimal(job.get("id"), f"provenance {label} job id")
    if positive_decimal(job.get("attempt"), f"provenance {label} job attempt") != attempt:
        die(f"provenance {label} job/run attempt mismatch")
    if job.get("status") != "completed" or job.get("conclusion") != "success" or not job.get("steps"):
        die(f"provenance {label} job state/steps mismatch")
    artifact_id = positive_decimal(artifact.get("id"), f"provenance {label} artifact id")
    if positive_decimal(artifact.get("workflow_run_id"), f"provenance {label} artifact run id") != run_id:
        die(f"provenance {label} artifact/run mismatch")
    api_size = int(positive_decimal(artifact.get("size_in_bytes"), f"provenance {label} API artifact size"))
    archive_size = int(positive_decimal(artifact.get("downloaded_archive_bytes"), f"provenance {label} archive size"))
    if api_size != archive_size:
        die(f"provenance {label} artifact size mismatch")
    api_digest = sha256(artifact.get("api_digest"), f"provenance {label} API artifact digest")
    if sha256(artifact.get("downloaded_archive_sha256"), f"provenance {label} archive digest") != api_digest:
        die(f"provenance {label} transport digest mismatch")
    if artifact.get("expired") is not False:
        die(f"provenance {label} artifact expired")
    if gitsha(artifact.get("workflow_run_head_sha"), f"provenance {label} artifact head SHA") != head:
        die(f"provenance {label} artifact/head mismatch")
    return {
        "witness_id": obj["provenance_witness_id"], "run_id": run_id, "run_attempt": attempt,
        "pr_number": pr, "artifact_id": artifact_id, "head_sha": head,
        "artifact_digest": api_digest, "artifact_bytes": archive_size,
    }


def main():
    ap = argparse.ArgumentParser()
    for name in ("contract", "diagnostic-witnesses", "diagnostic-source-archive", "corroboration", "provenance-a", "provenance-b", "output"):
        ap.add_argument("--" + name, required=True)
    ns = ap.parse_args()
    contract, contract_bytes = load(ns.contract)
    diagnostic_witnesses, _ = load(ns.diagnostic_witnesses)
    corroboration, _ = load(ns.corroboration)
    provenance_a, _ = load(ns.provenance_a)
    provenance_b, _ = load(ns.provenance_b)

    reject_authority(contract)
    if contract.get("schema") != "symthaea.se001q.repair-evidence-join-contract.v1.1" or contract.get("domain") != DOMAIN:
        die("bad join contract")
    non_authorizing(contract, "join contract")
    subject = gitsha(contract.get("expected_subject_sha"), "expected subject SHA")
    source_lock = sha256(contract.get("expected_source_lock_sha256"), "expected source lock digest")
    diagnostic = verify_diagnostic_witnesses(diagnostic_witnesses, ns.diagnostic_source_archive, contract)

    ci = content_addressed(corroboration, contract["required_corroboration_schema"], "corroboration_id", "corroboration")
    reject_authority(corroboration)
    non_authorizing(ci, "corroboration")
    if ci.get("result") != contract["required_corroboration_result"] or ci.get("subject_sha") != subject:
        die("corroboration result/subject mismatch")
    if ci.get("source_lock_sha256") != source_lock or ci.get("experiment_id") != contract["expected_experiment_id"]:
        die("corroboration source/experiment mismatch")
    experiment_sha = sha256(ci.get("experiment_sha256"), "corroboration experiment digest")
    generated_lock = sha256(ci.get("generated_lock_sha256"), "corroboration generated lock digest")
    if generated_lock == source_lock:
        die("corroboration generated lock equals source lock")

    runs = ci.get("runs")
    if not isinstance(runs, list) or len(runs) != 2:
        die("corroboration must contain exactly two runs")
    norm = []
    for i, run in enumerate(runs):
        if not isinstance(run, dict):
            die(f"corroboration run {i} malformed")
        norm.append({**run,
            "run_id": positive_decimal(run.get("run_id"), f"run {i} id"),
            "artifact_id": positive_decimal(run.get("artifact_id"), f"run {i} artifact id"),
            "verifier_sha": gitsha(run.get("verifier_sha"), f"run {i} verifier SHA"),
            "artifact_zip_sha256": sha256(run.get("artifact_zip_sha256"), f"run {i} artifact digest"),
            "observation_id": sha256(run.get("observation_id"), f"run {i} observation ID"),
            "manifest_id": sha256(run.get("manifest_id"), f"run {i} manifest ID"),
            "lock_delta_witness_id": sha256(run.get("lock_delta_witness_id"), f"run {i} lock witness ID"),
        })
    if len({r["run_id"] for r in norm}) != 2 or len({r["artifact_id"] for r in norm}) != 2 or len({r["verifier_sha"] for r in norm}) != 2:
        die("corroboration independence predicate failed")

    prov = [verify_provenance(provenance_a, contract, "A"), verify_provenance(provenance_b, contract, "B")]
    if len({p["witness_id"] for p in prov}) != 2 or len({p["run_id"] for p in prov}) != 2 or len({p["artifact_id"] for p in prov}) != 2:
        die("provenance independence predicate failed")
    by_run = {p["run_id"]: p for p in prov}
    if len(by_run) != 2:
        die("provenance run map is not one-to-one")

    pairings = []
    for run in sorted(norm, key=lambda r: r["run_id"]):
        p = by_run.get(run["run_id"])
        if p is None:
            die(f"missing provenance for run {run['run_id']}")
        if p["artifact_id"] != run["artifact_id"] or p["head_sha"] != run["verifier_sha"] or p["artifact_digest"] != run["artifact_zip_sha256"]:
            die(f"paired provenance mismatch for run {run['run_id']}")
        pairings.append({
            "run_id": run["run_id"], "run_attempt": p["run_attempt"], "pr_number": p["pr_number"],
            "artifact_id": run["artifact_id"], "artifact_bytes": p["artifact_bytes"],
            "verifier_sha": run["verifier_sha"], "artifact_zip_sha256": run["artifact_zip_sha256"],
            "provenance_witness_id": p["witness_id"], "observation_id": run["observation_id"],
            "manifest_id": run["manifest_id"], "lock_delta_witness_id": run["lock_delta_witness_id"],
            "verification_schema": run.get("verification_schema"), "capture_protocol": run.get("capture_protocol"),
        })
    if {p["run_id"] for p in pairings} != set(by_run):
        die("unused provenance witness")

    identity = {
        "domain": DOMAIN, "contract_sha256": digest(contract_bytes),
        "diagnostic_witness_set_id": diagnostic["set_id"], "diagnostic_failure_class": diagnostic["failure_class"],
        "diagnostic_source": {
            "run_id": diagnostic["source_run_id"], "run_attempt": diagnostic["source_run_attempt"],
            "artifact_id": diagnostic["source_artifact_id"], "artifact_zip_sha256": diagnostic["source_archive_sha256"],
            "artifact_bytes": diagnostic["source_archive_bytes"], "evidence_manifest_id": diagnostic["source_evidence_manifest_id"],
            "witness_ids": diagnostic["witness_ids"],
        },
        "corroboration_id": corroboration["corroboration_id"], "subject_sha": subject,
        "experiment_id": ci["experiment_id"], "experiment_sha256": experiment_sha,
        "source_lock_sha256": source_lock, "generated_lock_sha256": generated_lock, "pairings": pairings,
        "candidate_lock_transform": {"path": "Cargo.lock", "source_sha256": source_lock, "generated_sha256": generated_lock, "authorized": False},
        "result": contract["result"],
        "authority": {"meaning": contract["authority"]["meaning"], "sufficient_for_repair_grant": False, "qualification_claim": "NONE", "repair_authority_claim": "NONE"},
    }
    output = {"schema": DOMAIN, "repair_evidence_join_id": digest(canonical(identity)), "identity": identity}
    reject_authority(output)
    non_authorizing(identity, "repair evidence join")
    out = Path(ns.output); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "schema": DOMAIN, "result": identity["result"], "repair_evidence_join_id": output["repair_evidence_join_id"],
        "diagnostic_witness_set_id": identity["diagnostic_witness_set_id"], "corroboration_id": identity["corroboration_id"],
        "source_lock_sha256": source_lock, "generated_lock_sha256": generated_lock,
        "sufficient_for_repair_grant": False, "qualification_claim": "NONE", "repair_authority_claim": "NONE",
    }, sort_keys=True))


if __name__ == "__main__":
    main()
