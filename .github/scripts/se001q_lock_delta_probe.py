#!/usr/bin/env python3
import argparse, hashlib, json, os, shutil, subprocess
from pathlib import Path

CAPTURE_PROTOCOL = "symthaea.se001q.lock-delta.capture.v1.1"
LOCK_DIAGNOSTICS = (b"cannot update the lock file", b"lock file", b"needs to be updated")


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def sha_bytes(data):
    return "sha256:" + hashlib.sha256(data).hexdigest()


def sha_file(path):
    return sha_bytes(Path(path).read_bytes())


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run(argv, cwd, outdir, env=None):
    outdir.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(argv, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    (outdir / "stdout.log").write_bytes(proc.stdout)
    (outdir / "stderr.log").write_bytes(proc.stderr)
    (outdir / "exit-code.txt").write_text(str(proc.returncode) + "\n", encoding="utf-8")
    command = {
        "argv": argv,
        "cwd": ".",
        "exit_code": proc.returncode,
        "stdout_sha256": sha_bytes(proc.stdout),
        "stderr_sha256": sha_bytes(proc.stderr),
    }
    write_json(outdir / "command-result.json", command)
    return command, proc.stderr


def git(cwd, *args, check=True):
    proc = subprocess.run(["git", "-C", str(cwd), *args], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if check and proc.returncode:
        raise SystemExit(f"git {' '.join(args)} failed: {proc.stderr.decode(errors='replace')}")
    return proc


def git_text(cwd, *args):
    return git(cwd, *args).stdout.decode().strip()


def input_hashes(root, keys):
    return {key: sha_file(Path(root) / key) for key in keys}


def status_bytes(root):
    return git(root, "status", "--porcelain=v1", "--untracked-files=all").stdout


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


def command_contract(exp, section, ident):
    for item in exp[section]:
        if item["id"] == ident:
            return item["argv"]
    raise SystemExit(f"missing experiment command {section}/{ident}")


def snapshot_lock(source, dest):
    shutil.copy2(source / "Cargo.lock", dest)
    return sha_file(dest)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", required=True)
    ap.add_argument("--verifier-root", required=True)
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--work-root", required=True)
    ns = ap.parse_args()

    subject = Path(ns.subject).resolve()
    verifier_root = Path(ns.verifier_root).resolve()
    experiment_path = Path(ns.experiment).resolve()
    output = Path(ns.output).resolve()
    workroot = Path(ns.work_root).resolve()
    exp = json.loads(experiment_path.read_text(encoding="utf-8"))

    if exp.get("schema") != "symthaea.se001q.lock-delta-experiment.v1":
        raise SystemExit("bad experiment schema")
    if exp.get("qualification_claim") != "NONE" or exp.get("repair_authority_claim") != "NONE":
        raise SystemExit("authority boundary violated")

    subject_sha = exp["subject"]["sha"]
    subject_tree = exp["subject"]["tree"]
    if git_text(subject, "rev-parse", "HEAD") != subject_sha:
        raise SystemExit("subject head mismatch")
    if git_text(subject, "rev-parse", "HEAD^{tree}") != subject_tree:
        raise SystemExit("subject tree mismatch")
    if status_bytes(subject):
        raise SystemExit("subject not clean")
    if input_hashes(subject, exp["expected_inputs"]) != exp["expected_inputs"]:
        raise SystemExit("subject input hash mismatch")

    verifier_sha = git_text(verifier_root, "rev-parse", "HEAD")
    experiment_sha = sha_file(experiment_path)
    capture_runner_sha = sha_file(Path(__file__).resolve())

    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    if workroot.exists():
        shutil.rmtree(workroot)
    workroot.mkdir(parents=True)

    online = workroot / "online"
    offline = workroot / "offline"
    git(subject, "worktree", "add", "--detach", str(online), subject_sha)
    git(subject, "worktree", "add", "--detach", str(offline), subject_sha)

    try:
        before_online = snapshot_lock(online, output / "Cargo.lock.online-before")
        before_offline = snapshot_lock(offline, output / "Cargo.lock.offline-before")
        if before_online != before_offline:
            raise SystemExit("fresh worktrees started from different lock bytes")
        before_lock = before_online
        env = os.environ.copy()
        env["CARGO_TERM_COLOR"] = "never"

        online_argv = command_contract(exp, "resolution_probes", "online-all-targets")
        online_res, _ = run(online_argv, online, output / "online-all-targets", env)
        online_lock = snapshot_lock(online, output / "Cargo.lock.online-after")
        (output / "Cargo.lock.online.patch").write_bytes(git(online, "diff", "--binary", "--", "Cargo.lock").stdout)
        online_status = status_bytes(online)
        (output / "git-status.online-after-resolution.txt").write_bytes(online_status)
        online_changed = status_paths(online_status)

        offline_argv = command_contract(exp, "resolution_probes", "offline-reproduction-all-targets")
        offline_res, _ = run(offline_argv, offline, output / "offline-reproduction-all-targets", env)
        offline_lock = snapshot_lock(offline, output / "Cargo.lock.offline-after")
        (output / "Cargo.lock.offline.patch").write_bytes(git(offline, "diff", "--binary", "--", "Cargo.lock").stdout)
        offline_status = status_bytes(offline)
        (output / "git-status.offline-after-resolution.txt").write_bytes(offline_status)
        offline_changed = status_paths(offline_status)

        counter = []
        for item in exp["counterfactual_locked_gates"]:
            gate_dir = output / ("counterfactual-" + item["id"])
            gate_dir.mkdir(parents=True, exist_ok=True)
            lock_before = snapshot_lock(offline, gate_dir / "Cargo.lock.before")
            result, stderr = run(item["argv"], offline, gate_dir, env)
            lock_after = snapshot_lock(offline, gate_dir / "Cargo.lock.after")
            counter.append({
                "gate_id": item["id"],
                "result": result,
                "lock_sha256_before": lock_before,
                "lock_sha256_after": lock_after,
                "lock_unchanged": lock_before == lock_after,
                "lock_update_diagnostic_present": any(x in stderr for x in LOCK_DIAGNOSTICS),
            })

        final_online_status = status_bytes(online)
        final_offline_status = status_bytes(offline)
        (output / "git-status.online-final.txt").write_bytes(final_online_status)
        (output / "git-status.offline-final.txt").write_bytes(final_offline_status)

        nonlock_keys = [key for key in exp["expected_inputs"] if key != "Cargo.lock"]
        online_nonlock = input_hashes(online, nonlock_keys)
        offline_nonlock = input_hashes(offline, nonlock_keys)
        write_json(output / "nonlock-input-hashes.online.json", online_nonlock)
        write_json(output / "nonlock-input-hashes.offline.json", offline_nonlock)
        expected_nonlock = {key: value for key, value in exp["expected_inputs"].items() if key != "Cargo.lock"}

        subject_clean = status_bytes(subject) == b""
        subject_head = git_text(subject, "rev-parse", "HEAD") == subject_sha
        subject_tree_ok = git_text(subject, "rev-parse", "HEAD^{tree}") == subject_tree
        source_preserved = online_nonlock == expected_nonlock and offline_nonlock == expected_nonlock
        delta_exists = online_lock != before_lock and offline_lock != before_lock
        reproduced = online_lock == offline_lock
        tracked_safe = (
            online_changed == ["Cargo.lock"]
            and offline_changed == ["Cargo.lock"]
            and status_paths(final_online_status) == ["Cargo.lock"]
            and status_paths(final_offline_status) == ["Cargo.lock"]
        )
        counter_lock_stable = all(item["lock_unchanged"] for item in counter)
        lock_boundary_cleared = all(not item["lock_update_diagnostic_present"] for item in counter)

        if delta_exists and reproduced and tracked_safe and counter_lock_stable and lock_boundary_cleared and source_preserved and subject_clean and subject_head and subject_tree_ok:
            status = "REPRODUCIBLE_LOCK_DELTA"
        elif not delta_exists:
            status = "NO_REPRODUCIBLE_LOCK_DELTA"
        else:
            status = "INCOMPLETE_OR_NONREPRODUCIBLE"

        toolchain = {}
        for name, argv in {
            "rustc": ["rustc", "--version", "--verbose"],
            "cargo": ["cargo", "--version"],
            "clippy": ["cargo", "clippy", "--version"],
        }.items():
            proc = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
            toolchain[name] = proc.stdout.decode(errors="replace").strip()

        identity = {
            "domain": "symthaea.se001q.lock-delta-observation.v1",
            "capture_protocol": CAPTURE_PROTOCOL,
            "capture_runner_sha256": capture_runner_sha,
            "experiment_sha256": experiment_sha,
            "verifier_sha": verifier_sha,
            "experiment_id": exp["experiment_id"],
            "subject": exp["subject"],
            "parent_evidence": exp["parent_evidence"],
            "toolchain": toolchain,
            "before_lock_sha256": before_lock,
            "online_after_lock_sha256": online_lock,
            "offline_after_lock_sha256": offline_lock,
            "online_changed_tracked_paths": online_changed,
            "offline_changed_tracked_paths": offline_changed,
            "online_final_changed_paths": status_paths(final_online_status),
            "offline_final_changed_paths": status_paths(final_offline_status),
            "online_resolution_result": online_res,
            "offline_resolution_result": offline_res,
            "counterfactual_locked_gates": counter,
            "source_preserved": source_preserved,
            "frozen_subject_unchanged": subject_clean and subject_head and subject_tree_ok,
            "lock_delta_reproduced_offline": reproduced and delta_exists,
            "counterfactual_lock_stable": counter_lock_stable,
            "lock_boundary_cleared": lock_boundary_cleared,
            "status": status,
            "qualification_claim": "NONE",
            "repair_authority_claim": "NONE",
        }
        summary = {"schema": "symthaea.se001q.lock-delta-observation.v1", "observation_id": sha_bytes(canonical(identity)), "identity": identity}
        write_json(output / "summary.json", summary)

        if status == "REPRODUCIBLE_LOCK_DELTA":
            witness_identity = {
                "domain": "symthaea.lock-delta-witness.v1",
                "capture_protocol": CAPTURE_PROTOCOL,
                "subject_sha": subject_sha,
                "source_lock_sha256": before_lock,
                "generated_lock_sha256": offline_lock,
                "online_lock_patch_sha256": sha_file(output / "Cargo.lock.online.patch"),
                "offline_lock_patch_sha256": sha_file(output / "Cargo.lock.offline.patch"),
                "online_resolution_result_sha256": sha_file(output / "online-all-targets/command-result.json"),
                "offline_resolution_result_sha256": sha_file(output / "offline-reproduction-all-targets/command-result.json"),
                "changed_tracked_paths": ["Cargo.lock"],
                "offline_reproduction": "PASS",
                "counterfactual_lock_stability": "PASS",
                "lock_boundary_cleared": "PASS",
                "source_preservation": "PASS",
                "qualification_claim": "NONE",
                "repair_authority_claim": "NONE",
            }
            witness = {"schema": "symthaea.lock-delta-witness.v1", "witness_id": sha_bytes(canonical(witness_identity)), "identity": witness_identity}
            write_json(output / "lock-delta-witness.json", witness)

        files = []
        for path in sorted(p for p in output.rglob("*") if p.is_file() and p.name != "manifest.json"):
            files.append({"path": path.relative_to(output).as_posix(), "sha256": sha_file(path), "bytes": path.stat().st_size})
        manifest_identity = {
            "domain": "symthaea.se001q.lock-delta-manifest.v1",
            "capture_protocol": CAPTURE_PROTOCOL,
            "subject_sha": subject_sha,
            "verifier_sha": verifier_sha,
            "experiment_sha256": experiment_sha,
            "capture_runner_sha256": capture_runner_sha,
            "observation_id": summary["observation_id"],
            "files": files,
            "qualification_claim": "NONE",
            "repair_authority_claim": "NONE",
        }
        manifest = {"schema": "symthaea.se001q.lock-delta-manifest.v1", "manifest_id": sha_bytes(canonical(manifest_identity)), "identity": manifest_identity}
        write_json(output / "manifest.json", manifest)

        witness_path = output / "lock-delta-witness.json"
        witness_id = json.loads(witness_path.read_text(encoding="utf-8"))["witness_id"] if witness_path.exists() else None
        print(json.dumps({
            "result": "CAPTURE_COMPLETE",
            "status": status,
            "observation_id": summary["observation_id"],
            "manifest_id": manifest["manifest_id"],
            "lock_delta_witness": witness_id,
            "qualification_claim": "NONE",
            "repair_authority_claim": "NONE",
        }, sort_keys=True))
    finally:
        git(subject, "worktree", "remove", "--force", str(online), check=False)
        git(subject, "worktree", "remove", "--force", str(offline), check=False)


if __name__ == "__main__":
    main()
