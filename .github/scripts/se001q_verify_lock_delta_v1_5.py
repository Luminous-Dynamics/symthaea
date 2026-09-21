#!/usr/bin/env python3
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


SCHEMA = "symthaea.se001q.lock-delta-verification.v2.1"


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def sha_bytes(data):
    return "sha256:" + hashlib.sha256(data).hexdigest()


def sha_file(path):
    return sha_bytes(Path(path).read_bytes())


def die(message: str) -> None:
    raise SystemExit(message)


def probe(argv):
    proc = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if proc.returncode:
        die(f"toolchain probe failed: {' '.join(argv)}")
    return proc.stdout.decode(errors="replace").strip()


def rustc_release(text):
    releases = [line[len("release: "):].strip() for line in text.splitlines() if line.startswith("release: ")]
    if len(releases) != 1 or not releases[0]:
        die("could not parse unique rustc release")
    return releases[0]


def simple_release(text, tool):
    parts = text.split()
    if len(parts) < 2 or parts[0] != tool:
        die(f"could not parse {tool} release")
    return parts[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--evidence", required=True)
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--subject", required=True)
    ap.add_argument("--capture-runner", required=True)
    ap.add_argument("--verifier-root", required=True)
    ns = ap.parse_args()

    verifier_root = Path(ns.verifier_root).resolve()
    base_verifier = verifier_root / ".github/scripts/se001q_verify_lock_delta.py"
    cmd = [
        sys.executable,
        str(base_verifier),
        "--evidence", ns.evidence,
        "--experiment", ns.experiment,
        "--subject", ns.subject,
        "--capture-runner", ns.capture_runner,
        "--verifier-root", ns.verifier_root,
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode:
        sys.stderr.buffer.write(proc.stderr)
        sys.stdout.buffer.write(proc.stdout)
        raise SystemExit(proc.returncode)

    try:
        base_result = json.loads(proc.stdout.decode(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        die(f"base verifier output is not one JSON object: {exc}")
    if base_result.get("schema") != "symthaea.se001q.lock-delta-verification.v2":
        die("unexpected base verification schema")
    if base_result.get("result") != "PASS":
        die("base verification did not pass")
    if base_result.get("qualification_claim") != "NONE" or base_result.get("repair_authority_claim") != "NONE":
        die("base verification authority boundary violated")

    experiment_path = Path(ns.experiment).resolve()
    experiment = json.loads(experiment_path.read_text(encoding="utf-8"))
    summary = json.loads((Path(ns.evidence) / "summary.json").read_text(encoding="utf-8"))
    expected_channel = experiment.get("toolchain_channel")
    if expected_channel != "1.96.0":
        die("unexpected experiment toolchain channel")

    live_toolchain = {
        "rustc": probe(["rustc", "--version", "--verbose"]),
        "cargo": probe(["cargo", "--version"]),
        "clippy": probe(["cargo", "clippy", "--version"]),
    }
    capture_toolchain = summary.get("identity", {}).get("toolchain")
    if not isinstance(capture_toolchain, dict):
        die("capture toolchain missing")

    rust_release = rustc_release(live_toolchain["rustc"])
    cargo_release = simple_release(live_toolchain["cargo"], "cargo")
    clippy_release = simple_release(live_toolchain["clippy"], "clippy")
    channel_parts = expected_channel.split(".")
    if len(channel_parts) != 3 or not all(part.isdigit() for part in channel_parts):
        die("unexpected experiment toolchain channel format")
    expected_clippy_release = f"0.{channel_parts[0]}.{channel_parts[1]}"

    predicates = {
        "capture_matches_live": capture_toolchain == live_toolchain,
        "rustc_release_matches_channel": rust_release == expected_channel,
        "cargo_release_matches_channel": cargo_release == expected_channel,
        "clippy_release_matches_channel": clippy_release == expected_clippy_release,
    }
    failed = [name for name, value in predicates.items() if not value]
    if failed:
        die("toolchain verification failed: " + ", ".join(failed))

    identity = {
        "domain": SCHEMA,
        "base_verification": base_result,
        "experiment_sha256": sha_file(experiment_path),
        "experiment_toolchain_channel": expected_channel,
        "capture_toolchain": capture_toolchain,
        "live_toolchain": live_toolchain,
        "parsed_releases": {
            "rustc": rust_release,
            "cargo": cargo_release,
            "clippy": clippy_release,
        },
        "expected_clippy_release": expected_clippy_release,
        "toolchain_predicates": predicates,
        "qualification_claim": "NONE",
        "repair_authority_claim": "NONE",
    }
    result = {
        "schema": SCHEMA,
        "verification_id": sha_bytes(canonical(identity)),
        "identity": identity,
    }
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
