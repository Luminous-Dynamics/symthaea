#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

import qualify_vart_world_creative_001_instrument_source_closure as q


def run(*argv: str, cwd: Path | None = None) -> str:
    proc = subprocess.run(list(argv), cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"{argv!r} failed: {proc.stderr}")
    return proc.stdout.strip()


def commit(repo: Path, content: str) -> tuple[str, str]:
    (repo / "instrument.txt").write_text(content, encoding="utf-8")
    run("git", "add", "instrument.txt", cwd=repo)
    run("git", "commit", "-q", "-m", content, cwd=repo)
    return run("git", "rev-parse", "HEAD", cwd=repo), run("git", "rev-parse", "HEAD^{tree}", cwd=repo)


def receipt(path: Path, head: str, tree: str) -> None:
    env = {"python_version": "3.x", "platform": "synthetic"}
    env_digest = hashlib.sha256(json.dumps(env, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    obj = {
        "schema": "symthaea.vart-world-creative-001.instrument-qualification.v1",
        "experiment_id": q.EXPERIMENT_ID,
        "status": "qualified",
        "instrument_source": {"head": head, "tree": tree, "dirty": False},
        "instrument_manifest_sha256": "a" * 64,
        "instrument_environment": env,
        "instrument_environment_digest": env_digest,
        "all_suites_pass": True,
        "confirmatory_execution_authorized": False,
        "claim_authorized": False,
    }
    path.write_text(json.dumps(obj, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")


def expect_reject(fn, fragment: str) -> None:
    try:
        fn()
    except q.InstrumentClosureError as exc:
        assert fragment in str(exc), (fragment, exc)
        return
    raise AssertionError(f"expected rejection containing {fragment}")


with tempfile.TemporaryDirectory(prefix="vart-instrument-source-test-") as td:
    root = Path(td)
    origin = root / "origin.git"
    repo = root / "instrument"
    run("git", "init", "--bare", "-q", str(origin))
    run("git", "init", "-q", str(repo))
    run("git", "config", "user.email", "vart@example.invalid", cwd=repo)
    run("git", "config", "user.name", "VART Test", cwd=repo)
    head, tree = commit(repo, "qualified instrument")
    run("git", "branch", "instrument-qualified", cwd=repo)
    run("git", "remote", "add", "origin", str(origin), cwd=repo)
    run("git", "push", "-q", "origin", "instrument-qualified:refs/heads/instrument-qualified", cwd=repo)
    qualification = root / "qualification.json"
    receipt(qualification, head, tree)

    def qualify(**overrides):
        kwargs = {
            "repo": repo,
            "remote_name": "origin",
            "repository_full_name": "example/vart-instrument",
            "ref": "refs/heads/instrument-qualified",
            "qualification_receipt": qualification,
        }
        kwargs.update(overrides)
        return q.qualify(**kwargs)

    result = qualify()
    assert result["status"] == "qualified"
    assert result["instrument_source"] == {"head": head, "tree": tree}
    assert result["remote"]["fresh_checkout_verified"] is True
    assert result["qualification"]["all_suites_pass"] is True

    # IC1 — dirty instrument source cannot close.
    dirty = repo / "dirty.tmp"
    dirty.write_text("dirty\n", encoding="utf-8")
    expect_reject(qualify, "instrument checkout is dirty")
    dirty.unlink()

    # IC2 — an unpushed local instrument commit cannot masquerade as the durable ref.
    newer_head, newer_tree = commit(repo, "unpushed instrument")
    receipt(qualification, newer_head, newer_tree)
    expect_reject(qualify, "remote ref HEAD mismatch")
    run("git", "reset", "--hard", "-q", head, cwd=repo)
    receipt(qualification, head, tree)

    # IC3 — qualification receipt from another source identity rejects.
    receipt(qualification, "b" * 40, "c" * 40)
    expect_reject(qualify, "instrument qualification source identity mismatch")
    receipt(qualification, head, tree)

    # IC4 — a qualification receipt without a complete suite pass is not admissible.
    obj = json.loads(qualification.read_text())
    obj["all_suites_pass"] = False
    qualification.write_text(json.dumps(obj, sort_keys=True, separators=(",", ":")) + "\n")
    expect_reject(qualify, "instrument qualification receipt is not qualified")
    receipt(qualification, head, tree)

    # IC5 — authority escalation inside the qualification receipt rejects.
    obj = json.loads(qualification.read_text())
    obj["confirmatory_execution_authorized"] = True
    qualification.write_text(json.dumps(obj, sort_keys=True, separators=(",", ":")) + "\n")
    expect_reject(qualify, "instrument qualification authority violation")

print("PASS: instrument fresh-checkout closure acceptance + IC1-IC5 rejection")
