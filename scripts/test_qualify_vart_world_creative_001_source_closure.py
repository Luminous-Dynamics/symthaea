#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import qualify_vart_world_creative_001_source_closure as q


def run(*argv: str, cwd: Path | None = None) -> str:
    proc = subprocess.run(list(argv), cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"{argv!r} failed: {proc.stderr}")
    return proc.stdout.strip()


def commit(repo: Path, name: str, content: str) -> tuple[str, str]:
    (repo / name).write_text(content, encoding="utf-8")
    run("git", "add", name, cwd=repo)
    run("git", "commit", "-q", "-m", f"commit {name} {content}", cwd=repo)
    return run("git", "rev-parse", "HEAD", cwd=repo), run("git", "rev-parse", "HEAD^{tree}", cwd=repo)


def expect_reject(fn, fragment: str) -> None:
    try:
        fn()
    except q.SourceClosureError as exc:
        assert fragment in str(exc), (fragment, exc)
        return
    raise AssertionError(f"expected rejection containing {fragment}")


with tempfile.TemporaryDirectory(prefix="vart-source-closure-test-") as td:
    root = Path(td)
    origin = root / "origin.git"
    repo = root / "subject"
    run("git", "init", "--bare", "-q", str(origin))
    run("git", "init", "-q", str(repo))
    run("git", "config", "user.email", "vart@example.invalid", cwd=repo)
    run("git", "config", "user.name", "VART Test", cwd=repo)

    baseline_head, baseline_tree = commit(repo, "Cargo.lock", "baseline\n")
    # The production qualifier is pinned to the real v0.5-A baseline. For this
    # isolated mechanism test, substitute the synthetic baseline constants.
    q.V05_HEAD = baseline_head
    q.V05_TREE = baseline_tree

    pilot_head, pilot_tree = commit(repo, "runtime.txt", "pilot\n")
    confirm_head, confirm_tree = commit(repo, "runtime.txt", "confirmatory\n")
    run("git", "branch", "confirmatory", cwd=repo)
    run("git", "remote", "add", "origin", str(origin), cwd=repo)
    run("git", "push", "-q", "origin", "confirmatory:refs/heads/confirmatory", cwd=repo)

    env_receipt = root / "environment.json"
    qual_receipt = root / "qualification.json"
    env_receipt.write_text('{"environment":"synthetic"}\n', encoding="utf-8")
    qual_receipt.write_text('{"qualification":"synthetic"}\n', encoding="utf-8")

    def qualify(**overrides):
        kwargs = {
            "repo": repo,
            "remote_name": "origin",
            "repository_full_name": "example/vart-subject",
            "ref": "refs/heads/confirmatory",
            "pilot_head": pilot_head,
            "pilot_tree": pilot_tree,
            "parent_v05a_head": baseline_head,
            "parent_v05a_tree": baseline_tree,
            "lock_files": ["Cargo.lock"],
            "environment_receipt": env_receipt,
            "qualification_receipt": qual_receipt,
        }
        kwargs.update(overrides)
        return q.qualify(**kwargs)

    result = qualify()
    assert result["status"] == "qualified"
    assert result["confirmatory_source"]["head"] == confirm_head
    assert result["confirmatory_source"]["tree"] == confirm_tree
    assert result["remote"]["fresh_checkout_verified"] is True
    assert result["pilot_predecessor"]["is_ancestor_of_confirmatory_source"] is True
    assert result["confirmatory_execution_authorized"] is False

    # SC1 — dirty subject checkout cannot be qualified.
    dirty = repo / "untracked.tmp"
    dirty.write_text("dirty\n", encoding="utf-8")
    expect_reject(qualify, "subject source checkout is dirty")
    dirty.unlink()

    # SC2 — a clean local HEAD that was not pushed to the frozen remote ref rejects.
    unpushed_head, _ = commit(repo, "runtime.txt", "unpushed\n")
    assert unpushed_head != confirm_head
    expect_reject(qualify, "remote ref HEAD mismatch")
    run("git", "reset", "--hard", "-q", confirm_head, cwd=repo)

    # SC3 — pilot digest/tree substitution is detected before ancestry admission.
    expect_reject(lambda: qualify(pilot_tree="0" * 40), "pilot predecessor TREE mismatch")

    # SC4 — a real commit from a side branch is not accepted as the pilot predecessor.
    run("git", "checkout", "-q", "-b", "side", baseline_head, cwd=repo)
    side_head, side_tree = commit(repo, "side.txt", "side\n")
    run("git", "checkout", "-q", "confirmatory", cwd=repo)
    expect_reject(lambda: qualify(pilot_head=side_head, pilot_tree=side_tree), "pilot predecessor is not an ancestor")

    # SC5 — the qualified baseline identity cannot be silently substituted.
    expect_reject(
        lambda: qualify(parent_v05a_head="0" * 40, parent_v05a_tree="1" * 40),
        "qualified v0.5-A parent identity mismatch",
    )

    # SC6 — reproduction evidence is required, not a boolean label.
    missing_env = root / "missing-environment.json"
    expect_reject(lambda: qualify(environment_receipt=missing_env), "missing environment receipt")

print("PASS: subject source closure canonical fresh-fetch acceptance + SC1-SC6 rejection")
