#!/usr/bin/env python3
import argparse, hashlib, json, subprocess, tempfile
from pathlib import Path

CAPTURE_PROTOCOL = "symthaea.se001q.lock-delta.capture.v1.1"
LOCK_DIAGNOSTICS = (b"cannot update the lock file", b"lock file", b"needs to be updated")


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def sha_bytes(data):
    return "sha256:" + hashlib.sha256(data).hexdigest()


def sha_file(path):
    return sha_bytes(Path(path).read_bytes())


def die(message):
    raise SystemExit(message)


def git_text(root, *args):
    proc = subprocess.run(["git", "-C", str(root), *args], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode:
        die(f"git {' '.join(args)} failed: {proc.stderr.decode(errors='replace')}")
    return proc.stdout.decode().strip()


def status_paths(raw):
    paths = []
    for line in raw.decode().splitlines():
        if not line:
            continue
        path = line[3:]
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        paths.append(path)
    return sorted(paths)


def expected_argv(exp, section, ident):
    for item in exp[section]:
        if item["id"] == ident:
            return item["argv"]
    die(f"missing experiment command {section}/{ident}")


def verify_command(root, rel, argv):
    gate = root / rel
    result = json.loads((gate / "command-result.json").read_text(encoding="utf-8"))
    stdout = (gate / "stdout.log").read_bytes()
    stderr = (gate / "stderr.log").read_bytes()
    exit_code = int((gate / "exit-code.txt").read_text(encoding="utf-8").strip())
    if result.get("argv") != argv or result.get("cwd") != ".":
        die(f"command contract mismatch: {rel}")
    if result.get("exit_code") != exit_code:
        die(f"exit-code mismatch: {rel}")
    if result.get("stdout_sha256") != sha_bytes(stdout):
        die(f"stdout digest mismatch: {rel}")
    if result.get("stderr_sha256") != sha_bytes(stderr):
        die(f"stderr digest mismatch: {rel}")
    return result, stderr


def verify_patch(before_path, after_path, patch_path):
    with tempfile.TemporaryDirectory(prefix="se001q-lock-verify-") as td:
        root = Path(td)
        subprocess.run(["git", "init", "-q"], cwd=root, check=True)
        (root / "Cargo.lock").write_bytes(Path(before_path).read_bytes())
        proc = subprocess.run(["git", "apply", "--binary", str(Path(patch_path).resolve())], cwd=root, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if proc.returncode:
            die(f"lock patch does not apply: {proc.stderr.decode(errors='replace')}")
        if (root / "Cargo.lock").read_bytes() != Path(after_path).read_bytes():
            die("lock patch result does not equal retained after bytes")


def reject_authority(value, path="$"):
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            if key == "repair_authority":
                die(f"forbidden repair_authority field at {child_path}")
            if key == "repair_authority_claim" and child != "NONE":
                die(f"repair authority violation at {child_path}")
            if key == "qualification_claim" and child != "NONE":
                die(f"qualification authority violation at {child_path}")
            reject_authority(child, child_path)
    elif isinstance(value, list):
        for idx, child in enumerate(value):
            reject_authority(child, f"{path}[{idx}]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evidence", required=True)
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--subject", required=True)
    ap.add_argument("--capture-runner", required=True)
    ap.add_argument("--verifier-root", required=True)
    ns = ap.parse_args()

    root = Path(ns.evidence).resolve()
    experiment_path = Path(ns.experiment).resolve()
    subject = Path(ns.subject).resolve()
    capture_runner = Path(ns.capture_runner).resolve()
    verifier_root = Path(ns.verifier_root).resolve()

    exp = json.loads(experiment_path.read_text(encoding="utf-8"))
    if exp.get("schema") != "symthaea.se001q.lock-delta-experiment.v1":
        die("bad experiment schema")
    reject_authority(exp)

    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    reject_authority(summary)
    reject_authority(manifest)
    if summary.get("schema") != "symthaea.se001q.lock-delta-observation.v1":
        die("bad summary schema")
    if summary.get("observation_id") != sha_bytes(canonical(summary["identity"])):
        die("summary id mismatch")
    if manifest.get("schema") != "symthaea.se001q.lock-delta-manifest.v1":
        die("bad manifest schema")
    if manifest.get("manifest_id") != sha_bytes(canonical(manifest["identity"])):
        die("manifest id mismatch")

    si = summary["identity"]
    mi = manifest["identity"]
    if si.get("capture_protocol") != CAPTURE_PROTOCOL or mi.get("capture_protocol") != CAPTURE_PROTOCOL:
        die("capture protocol mismatch")
    if mi.get("observation_id") != summary["observation_id"]:
        die("manifest observation mismatch")

    experiment_sha = sha_file(experiment_path)
    capture_runner_sha = sha_file(capture_runner)
    verifier_sha = git_text(verifier_root, "rev-parse", "HEAD")
    if si.get("experiment_sha256") != experiment_sha or mi.get("experiment_sha256") != experiment_sha:
        die("experiment digest mismatch")
    if si.get("capture_runner_sha256") != capture_runner_sha or mi.get("capture_runner_sha256") != capture_runner_sha:
        die("capture runner digest mismatch")
    if si.get("verifier_sha") != verifier_sha or mi.get("verifier_sha") != verifier_sha:
        die("verifier commit mismatch")

    expected_files = {entry["path"]: entry for entry in mi["files"]}
    actual_files = {path.relative_to(root).as_posix(): path for path in root.rglob("*") if path.is_file() and path.name != "manifest.json"}
    if set(expected_files) != set(actual_files):
        die("manifest file set mismatch")
    for rel, path in actual_files.items():
        entry = expected_files[rel]
        if entry["sha256"] != sha_file(path) or entry["bytes"] != path.stat().st_size:
            die(f"manifest mismatch: {rel}")

    if git_text(subject, "rev-parse", "HEAD") != exp["subject"]["sha"]:
        die("live subject head mismatch")
    if git_text(subject, "rev-parse", "HEAD^{tree}") != exp["subject"]["tree"]:
        die("live subject tree mismatch")
    live_status = subprocess.run(["git", "-C", str(subject), "status", "--porcelain=v1", "--untracked-files=all"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True).stdout
    if live_status:
        die("live frozen subject not clean")
    for rel, expected_hash in exp["expected_inputs"].items():
        if sha_file(subject / rel) != expected_hash:
            die(f"live frozen input mismatch: {rel}")

    online_before = root / "Cargo.lock.online-before"
    offline_before = root / "Cargo.lock.offline-before"
    online_after = root / "Cargo.lock.online-after"
    offline_after = root / "Cargo.lock.offline-after"
    before_hash = sha_file(online_before)
    if before_hash != sha_file(offline_before):
        die("fresh-worktree source locks differ")
    if before_hash != exp["expected_inputs"]["Cargo.lock"]:
        die("unexpected source lock")
    online_hash = sha_file(online_after)
    offline_hash = sha_file(offline_after)

    verify_patch(online_before, online_after, root / "Cargo.lock.online.patch")
    verify_patch(offline_before, offline_after, root / "Cargo.lock.offline.patch")
    online_result, _ = verify_command(root, "online-all-targets", expected_argv(exp, "resolution_probes", "online-all-targets"))
    offline_result, _ = verify_command(root, "offline-reproduction-all-targets", expected_argv(exp, "resolution_probes", "offline-reproduction-all-targets"))

    online_paths = status_paths((root / "git-status.online-after-resolution.txt").read_bytes())
    offline_paths = status_paths((root / "git-status.offline-after-resolution.txt").read_bytes())
    online_final_paths = status_paths((root / "git-status.online-final.txt").read_bytes())
    offline_final_paths = status_paths((root / "git-status.offline-final.txt").read_bytes())

    expected_nonlock = {key: value for key, value in exp["expected_inputs"].items() if key != "Cargo.lock"}
    online_nonlock = json.loads((root / "nonlock-input-hashes.online.json").read_text(encoding="utf-8"))
    offline_nonlock = json.loads((root / "nonlock-input-hashes.offline.json").read_text(encoding="utf-8"))
    source_preserved = online_nonlock == expected_nonlock and offline_nonlock == expected_nonlock

    counter = []
    previous_lock_hash = offline_hash
    for item in exp["counterfactual_locked_gates"]:
        rel = "counterfactual-" + item["id"]
        result, stderr = verify_command(root, rel, item["argv"])
        lock_before = sha_file(root / rel / "Cargo.lock.before")
        lock_after = sha_file(root / rel / "Cargo.lock.after")
        if lock_before != previous_lock_hash:
            die(f"counterfactual lock chain discontinuity: {item['id']}")
        counter.append({
            "gate_id": item["id"],
            "result": result,
            "lock_sha256_before": lock_before,
            "lock_sha256_after": lock_after,
            "lock_unchanged": lock_before == lock_after,
            "lock_update_diagnostic_present": any(x in stderr for x in LOCK_DIAGNOSTICS),
        })
        previous_lock_hash = lock_after

    delta_exists = online_hash != before_hash and offline_hash != before_hash
    reproduced = online_hash == offline_hash
    tracked_safe = online_paths == ["Cargo.lock"] and offline_paths == ["Cargo.lock"] and online_final_paths == ["Cargo.lock"] and offline_final_paths == ["Cargo.lock"]
    counter_lock_stable = all(item["lock_unchanged"] for item in counter)
    lock_boundary_cleared = all(not item["lock_update_diagnostic_present"] for item in counter)

    if delta_exists and reproduced and tracked_safe and counter_lock_stable and lock_boundary_cleared and source_preserved:
        expected_status = "REPRODUCIBLE_LOCK_DELTA"
    elif not delta_exists:
        expected_status = "NO_REPRODUCIBLE_LOCK_DELTA"
    else:
        expected_status = "INCOMPLETE_OR_NONREPRODUCIBLE"

    projections = {
        "before_lock_sha256": before_hash,
        "online_after_lock_sha256": online_hash,
        "offline_after_lock_sha256": offline_hash,
        "online_changed_tracked_paths": online_paths,
        "offline_changed_tracked_paths": offline_paths,
        "online_final_changed_paths": online_final_paths,
        "offline_final_changed_paths": offline_final_paths,
        "online_resolution_result": online_result,
        "offline_resolution_result": offline_result,
        "counterfactual_locked_gates": counter,
        "source_preserved": source_preserved,
        "frozen_subject_unchanged": True,
        "lock_delta_reproduced_offline": reproduced and delta_exists,
        "counterfactual_lock_stable": counter_lock_stable,
        "lock_boundary_cleared": lock_boundary_cleared,
        "status": expected_status,
    }
    for key, value in projections.items():
        if si.get(key) != value:
            die(f"summary projection mismatch: {key}")

    witness_path = root / "lock-delta-witness.json"
    if expected_status == "REPRODUCIBLE_LOCK_DELTA":
        if not witness_path.exists():
            die("missing lock delta witness")
        witness = json.loads(witness_path.read_text(encoding="utf-8"))
        reject_authority(witness)
        if witness.get("schema") != "symthaea.lock-delta-witness.v1" or witness.get("witness_id") != sha_bytes(canonical(witness["identity"])):
            die("lock witness identity mismatch")
        expected_witness = {
            "domain": "symthaea.lock-delta-witness.v1",
            "capture_protocol": CAPTURE_PROTOCOL,
            "subject_sha": exp["subject"]["sha"],
            "source_lock_sha256": before_hash,
            "generated_lock_sha256": offline_hash,
            "online_lock_patch_sha256": sha_file(root / "Cargo.lock.online.patch"),
            "offline_lock_patch_sha256": sha_file(root / "Cargo.lock.offline.patch"),
            "online_resolution_result_sha256": sha_file(root / "online-all-targets/command-result.json"),
            "offline_resolution_result_sha256": sha_file(root / "offline-reproduction-all-targets/command-result.json"),
            "changed_tracked_paths": ["Cargo.lock"],
            "offline_reproduction": "PASS",
            "counterfactual_lock_stability": "PASS",
            "lock_boundary_cleared": "PASS",
            "source_preservation": "PASS",
            "qualification_claim": "NONE",
            "repair_authority_claim": "NONE",
        }
        if witness["identity"] != expected_witness:
            die("lock witness semantic mismatch")
        witness_id = witness["witness_id"]
    else:
        if witness_path.exists():
            die("lock witness exists without reproducible lock delta")
        witness_id = None

    print(json.dumps({
        "schema": "symthaea.se001q.lock-delta-verification.v2",
        "result": "PASS",
        "status": expected_status,
        "observation_id": summary["observation_id"],
        "manifest_id": manifest["manifest_id"],
        "lock_delta_witness_id": witness_id,
        "qualification_claim": "NONE",
        "repair_authority_claim": "NONE",
    }, sort_keys=True))


if __name__ == "__main__":
    main()
