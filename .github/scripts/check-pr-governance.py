#!/usr/bin/env python3
"""Fail-closed structural governance for exact pull-request change sets.

The validator binds classification to immutable event base/head SHAs.  It checks
Class A structural process only; scientific adequacy, test execution, full CI,
and merge authorization remain separate evidence layers.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

SHA40 = re.compile(r"^[0-9a-f]{40}$")
HEADING_RE = re.compile(r"(?m)^#{1,6}\s+")
PLACEHOLDER_MARKERS = (
    "[Title]",
    "YYYY-MM-DD",
    "Proposed | Accepted | Deprecated | Superseded",
    "EXAMPLE_THRESHOLD",
    "Citation(s) justifying the change.",
    "How to revert this change if problems are discovered.",
)
REQUIRED_ADR_SECTIONS = (
    "### Scientific Basis",
    "## Impact Analysis",
    "### Risk Register Impact",
    "## Test Evidence",
    "## Rollback Plan",
)
SAFETY_PREFIXES = (
    "safety:", "safety(", "ethics:", "ethics(",
    "emergency-safety:", "emergency-safety(",
)
GOVERNANCE_PREFIXES = (
    "governance:", "governance(", "emergency-safety:", "emergency-safety(",
)
REQUIRED_CLASS_A_ROOTS = (
    ("prefix", "src/cognitive_loop/thresholds/", "safety"),
    ("exact", "src/cognitive_loop/threshold_overrides.rs", "safety"),
    ("exact", "crates/core/symthaea-types/src/threshold_overrides.rs", "safety"),
    ("exact", "src/cognitive_loop/ethics_engine.rs", "safety"),
    ("exact", "src/safety/agent.rs", "safety"),
    ("exact", "scripts/cls_promote_candidate.sh", "safety"),
    ("exact", "docs/compliance/GOVERNANCE_CHARTER.md", "governance"),
    ("exact", ".github/governance-change-policy-v1.json", "governance"),
    ("exact", ".github/scripts/check-pr-governance.py", "governance"),
    ("exact", ".github/workflows/pr-governance.yml", "governance"),
    ("exact", ".github/workflows/pr-governance-root.yml", "governance"),
)
EXPECTED_ADR_PATH_PREFIXES = ("docs/compliance/adr/",)
EXPECTED_PREFIX_POLICY = {
    "safety": [
        "safety:", "safety(...):", "ethics:", "ethics(...):",
        "emergency-safety:", "emergency-safety(...):",
    ],
    "governance": [
        "governance:", "governance(...):",
        "emergency-safety:", "emergency-safety(...):",
    ],
}
EXPECTED_CLASS_B_STATE = "deferred-pending-gov-policy-001"


class GovernanceError(ValueError):
    pass


def git(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(
        ["git", *args],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if check and proc.returncode != 0:
        detail = proc.stderr.strip() or proc.stdout.strip() or f"git exited {proc.returncode}"
        raise GovernanceError(f"git {' '.join(args)}: {detail}")
    return proc


def run_git(*args: str) -> str:
    return git(*args).stdout


def git_ok(*args: str) -> bool:
    return git(*args, check=False).returncode == 0


def require_sha(name: str, value: str) -> str:
    value = value.strip().lower()
    if not SHA40.fullmatch(value):
        raise GovernanceError(f"{name} must be a lowercase 40-hex commit SHA")
    if not git_ok("cat-file", "-e", f"{value}^{{commit}}"):
        raise GovernanceError(f"{name} does not resolve to a commit: {value}")
    return value


def load_policy(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GovernanceError(f"{path}: {exc}") from exc
    if not isinstance(data, dict) or data.get("schema") != "symthaea-pr-governance-change-policy-v1":
        raise GovernanceError("unexpected governance policy schema")

    entries = data.get("class_a_paths")
    if not isinstance(entries, list) or not entries:
        raise GovernanceError("class_a_paths must be a non-empty list")
    normalized: set[tuple[str, str, str]] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise GovernanceError("every class_a_paths entry must be an object")
        match = entry.get("match")
        path_value = entry.get("path")
        authority = entry.get("authority")
        if match not in {"exact", "prefix"}:
            raise GovernanceError("class_a path match must be exact or prefix")
        if not isinstance(path_value, str) or not path_value:
            raise GovernanceError("class_a path must be non-empty")
        if authority not in {"safety", "governance"}:
            raise GovernanceError("class_a authority must be safety or governance")
        normalized.add((str(match), path_value, str(authority)))

    missing = [root for root in REQUIRED_CLASS_A_ROOTS if root not in normalized]
    if missing:
        raise GovernanceError(f"governance policy removed required Class A roots: {missing}")

    if data.get("class_b_policy_state") != EXPECTED_CLASS_B_STATE:
        raise GovernanceError("Class B must remain deferred until GOV-POLICY-001 is resolved")
    if data.get("class_b_paths") != []:
        raise GovernanceError("Class B paths must remain unenforced while policy is deferred")
    if data.get("adr_path_prefixes") != list(EXPECTED_ADR_PATH_PREFIXES):
        raise GovernanceError("ADR path roots drifted from the validator contract")
    if data.get("required_adr_sections") != list(REQUIRED_ADR_SECTIONS):
        raise GovernanceError("required ADR sections drifted from the validator contract")
    if data.get("class_a_prefix_policy") != EXPECTED_PREFIX_POLICY:
        raise GovernanceError("Class A commit-prefix policy drifted from the validator contract")
    nonclaims = data.get("authority_nonclaims")
    if not isinstance(nonclaims, list) or len(nonclaims) < 4:
        raise GovernanceError("authority_nonclaims must preserve explicit evidence limits")
    if not all(isinstance(item, str) and item.strip() for item in nonclaims):
        raise GovernanceError("authority_nonclaims entries must be non-empty strings")
    return data


def classify(path: str, policy: dict[str, Any]) -> str | None:
    for entry in policy["class_a_paths"]:
        matched = path == entry["path"] if entry["match"] == "exact" else path.startswith(entry["path"])
        if matched:
            return str(entry["authority"])
    return None


def root_exists(commit: str, match: str, path: str) -> bool:
    if match == "exact":
        return git_ok("cat-file", "-e", f"{commit}:{path}")
    out = run_git("ls-tree", "-r", "--name-only", commit, "--", path)
    return any(line.startswith(path) for line in out.splitlines())


ROOT_WORKFLOW_PATH = ".github/workflows/pr-governance-root.yml"
REQUIRED_ROOT_WORKFLOW_SNIPPETS = (
    "pull_request_target:",
    "permissions:",
    "contents: read",
    "ref: ${{ github.event.pull_request.base.sha }}",
    "fetch-depth: 0",
    "persist-credentials: false",
    'refs/pull/${PR_NUMBER}/head:refs/remotes/pr-governance/${PR_NUMBER}/head',
    'test "${fetched_head}" = "${PR_HEAD_SHA}"',
    "python3 .github/scripts/check-pr-governance.py",
)
FORBIDDEN_ROOT_WORKFLOW_SNIPPETS = (
    "ref: ${{ github.event.pull_request.head.sha }}",
    "ref: refs/pull/${{ github.event.pull_request.number }}/merge",
    "repository: ${{ github.event.pull_request.head.repo.full_name }}",
    "allow-unsafe-pr-checkout: true",
    "gh pr checkout",
    "git checkout \\${PR_HEAD_SHA}",
    "git checkout refs/pull/",
    "actions/download-artifact",
    "filter:",
)

def validate_root_workflow_text(text: str) -> None:
    missing = [snippet for snippet in REQUIRED_ROOT_WORKFLOW_SNIPPETS if snippet not in text]
    if missing:
        raise GovernanceError(
            f"{ROOT_WORKFLOW_PATH}: trusted-base workflow contract missing: {missing}"
        )
    forbidden = [snippet for snippet in FORBIDDEN_ROOT_WORKFLOW_SNIPPETS if snippet in text]
    checkout_lines = [
        line.strip() for line in text.splitlines()
        if line.strip().startswith("uses: actions/checkout@")
    ]
    if any(
        not re.fullmatch(r"uses: actions/checkout@[0-9a-f]{40}", line)
        for line in checkout_lines
    ):
        raise GovernanceError(f"{ROOT_WORKFLOW_PATH}: actions/checkout must use a full commit SHA")
    if forbidden:
        raise GovernanceError(
            f"{ROOT_WORKFLOW_PATH}: forbidden untrusted-code pattern: {forbidden}"
        )

def validate_root_workflow_contract(base: str) -> None:
    validate_root_workflow_text(read_at_commit(base, ROOT_WORKFLOW_PATH))
def validate_declared_roots(
    base: str,
    head: str,
    exists_fn: Any = root_exists,
) -> None:
    for match, path, _authority in REQUIRED_CLASS_A_ROOTS:
        if not (exists_fn(base, match, path) or exists_fn(head, match, path)):
            raise GovernanceError(
                f"required Class A root is absent from both event base and head: {match}:{path}"
            )


def require_changed_adr(class_a_detected: bool, adr_paths: list[str]) -> None:
    if class_a_detected and not adr_paths:
        raise GovernanceError(
            "Class A changes require a changed ADR-NNN*.md present in the PR head"
        )


def parse_name_status(text: str) -> list[str]:
    paths: list[str] = []
    for raw in text.splitlines():
        if not raw.strip():
            continue
        fields = raw.split("\t")
        status = fields[0]
        kind = status[:1]
        if kind in {"R", "C"}:
            if len(fields) != 3:
                raise GovernanceError(f"unexpected rename/copy record: {raw!r}")
            paths.extend((fields[1], fields[2]))
        else:
            if len(fields) != 2:
                raise GovernanceError(f"unexpected name-status record: {raw!r}")
            paths.append(fields[1])
    return sorted(set(paths))


def require_complete_object_graph(git_runner: Any = git) -> None:
    """Reject external or substitutable object sources before topology validation."""
    for variable in (
        "GIT_OBJECT_DIRECTORY",
        "GIT_ALTERNATE_OBJECT_DIRECTORIES",
        "GIT_COMMON_DIR",
    ):
        if os.environ.get(variable):
            raise GovernanceError(
                f"governance validation forbids {variable}; object authority must remain local and self-contained"
            )

    shallow = git_runner("rev-parse", "--is-shallow-repository", check=False)
    if shallow.returncode != 0:
        raise GovernanceError("unable to determine repository shallow state")
    if shallow.stdout.strip().lower() != "false":
        raise GovernanceError(
            "governance validation requires a complete commit history; repository is shallow"
        )

    partial = git_runner("config", "--get", "extensions.partialClone", check=False)
    if partial.returncode not in {0, 1}:
        raise GovernanceError("unable to determine Git partial-clone state")
    if partial.returncode == 0 and partial.stdout.strip():
        raise GovernanceError(
            "governance validation requires a complete object graph; repository is a partial clone"
        )

    promisors = git_runner(
        "config", "--get-regexp", r"^remote\..*\.promisor$", check=False
    )
    if promisors.returncode not in {0, 1}:
        raise GovernanceError("unable to determine Git promisor-remote state")
    if promisors.returncode == 0:
        for line in promisors.stdout.splitlines():
            _key, _sep, value = line.partition(" ")
            if value.strip().lower() == "true":
                raise GovernanceError(
                    "governance validation requires a complete object graph; "
                    "repository has a promisor remote"
                )

    alternates = git_runner(
        "config", "--get", "core.alternateRefsCommand", check=False
    )
    if alternates.returncode not in {0, 1}:
        raise GovernanceError("unable to determine alternate-ref command state")
    if alternates.returncode == 0 and alternates.stdout.strip():
        raise GovernanceError(
            "governance validation forbids alternate-ref commands in the authority repository"
        )

    replace_refs = git_runner(
        "for-each-ref", "--format=%(refname)", "refs/replace/"
    )
    if replace_refs.strip():
        raise GovernanceError(
            "governance validation forbids refs/replace because Git may substitute "
            "replacement objects for ordinary object reads"
        )


def require_complete_history() -> None:
    """Backward-compatible alias for the complete-object-graph guard."""
    require_complete_object_graph()


def validate_exact_base_head_ancestry(base: str, head: str) -> None:
    if base == head:
        raise GovernanceError("base_sha and head_sha must identify distinct commits")
    if not git_ok("merge-base", "--is-ancestor", base, head):
        merge_base = run_git("merge-base", base, head).strip() or "<none>"
        raise GovernanceError(
            "head_sha must descend from base_sha for exact base-to-head governance; "
            f"merge_base={merge_base}"
        )


def changed_paths(base: str, head: str) -> list[str]:
    validate_exact_base_head_ancestry(base, head)
    out = run_git(
        "diff", "--name-status", "-M", "--diff-filter=ACDMRT", base, head
    )
    return parse_name_status(out)


def unique_non_merge_commits(base: str, head: str) -> list[str]:
    out = run_git("rev-list", "--reverse", "--no-merges", head, "--not", base)
    return [line.strip() for line in out.splitlines() if line.strip()]


def commit_changed_paths(commit: str) -> list[str]:
    out = run_git(
        "diff-tree", "--root", "--no-commit-id", "--name-status", "-r", "-M",
        "--diff-filter=ACDMRT", commit,
    )
    return parse_name_status(out)


def history_topology_self_test() -> None:
    """Exercise exact base/head authority against adversarial Git histories."""
    with tempfile.TemporaryDirectory() as td:
        worktree = Path(td)
        def run_local(*args: str) -> str:
            proc = subprocess.run(
                ["git", *args], cwd=worktree, check=False,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            )
            if proc.returncode != 0:
                detail = proc.stderr.strip() or proc.stdout.strip()
                raise AssertionError(f"fixture git command failed: {args}: {detail}")
            return proc.stdout.strip()

        run_local("init", "--quiet")
        run_local("config", "user.name", "Governance Self-Test")
        run_local("config", "user.email", "governance-self-test@example.invalid")
        (worktree / "tracked.txt").write_text("base\\n", encoding="utf-8")
        run_local("add", "tracked.txt")
        run_local("commit", "--quiet", "-m", "base")
        base = run_local("rev-parse", "HEAD")

        (worktree / "tracked.txt").write_text("feature\\n", encoding="utf-8")
        run_local("commit", "--quiet", "-am", "feature")
        feature = run_local("rev-parse", "HEAD")
        validate_exact_base_head_ancestry(base, feature)

        try:
            validate_exact_base_head_ancestry(base, base)
        except GovernanceError:
            pass
        else:
            raise AssertionError("identical base/head must fail closed")

        run_local("checkout", "--quiet", base)
        (worktree / "side.txt").write_text("side\\n", encoding="utf-8")
        run_local("add", "side.txt")
        run_local("commit", "--quiet", "-m", "side")
        side = run_local("rev-parse", "HEAD")
        run_local("checkout", "--quiet", feature)
        run_local("merge", "--quiet", "--no-ff", side, "-m", "merge")
        merge = run_local("rev-parse", "HEAD")
        validate_exact_base_head_ancestry(base, merge)
        assert run_local("rev-parse", f"{merge}^1") == feature
        assert run_local("rev-parse", f"{merge}^2") == side

        try:
            validate_exact_base_head_ancestry(feature, base)
        except GovernanceError:
            pass
        else:
            raise AssertionError("reverse ancestry must fail closed")

        (worktree / "tracked.txt").write_text("rebased\\n", encoding="utf-8")
        run_local("commit", "--quiet", "-am", "rebased")
        rebased = run_local("rev-parse", "HEAD")
        try:
            validate_exact_base_head_ancestry(side, rebased)
        except GovernanceError:
            pass
        else:
            raise AssertionError("non-ancestor rewritten history must fail closed")

        changed = run_local("diff", "--name-only", base, merge).splitlines()
        assert "side.txt" in changed
        assert "tracked.txt" in changed

        # Shallow history must not be accepted as a substitute for complete
        # ancestry. Git explicitly treats shallow commits as roots, so a
        # topology-sensitive validator must detect that boundary and fail
        # closed rather than silently reasoning over truncated history.
        shallow = worktree / "shallow"
        run_local("clone", "--quiet", "--depth", "1", f"file://{worktree}", str(shallow))
        shallow_cmd = lambda *args: subprocess.run(
            ["git", *args], cwd=shallow, check=False,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        proc = shallow_cmd("merge-base", "--is-ancestor", base, merge)
        assert proc.returncode != 0
        assert (shallow / ".git" / "shallow").exists()

        def local_git(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
            proc = subprocess.run(
                ["git", *args], cwd=worktree, check=False,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            )
            if check and proc.returncode != 0:
                detail = proc.stderr.strip() or proc.stdout.strip()
                raise AssertionError(f"fixture git command failed: {args}: {detail}")
            return proc

        assert require_complete_object_graph(local_git) is None
        saved_env = {key: os.environ.get(key) for key in (
            "GIT_OBJECT_DIRECTORY",
            "GIT_ALTERNATE_OBJECT_DIRECTORIES",
            "GIT_COMMON_DIR",
        )}
        try:
            os.environ["GIT_ALTERNATE_OBJECT_DIRECTORIES"] = str(worktree / "objects")
            try:
                require_complete_object_graph(local_git)
            except GovernanceError:
                pass
            else:
                raise AssertionError("alternate object environment must fail closed")
        finally:
            for key, value in saved_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

        run_local("config", "extensions.partialClone", "origin")
        try:
            require_complete_object_graph(local_git)
        except GovernanceError:
            pass
        else:
            raise AssertionError("partial-clone extension must fail closed")
        run_local("config", "--unset", "extensions.partialClone")

        run_local("remote", "add", "origin", "https://example.invalid/symthaea.git")
        run_local("config", "remote.origin.promisor", "true")
        try:
            require_complete_object_graph(local_git)
        except GovernanceError:
            pass
        else:
            raise AssertionError("promisor remote must fail closed")
        run_local("config", "--unset", "remote.origin.promisor")

        run_local("config", "core.alternateRefsCommand", "echo refs/heads/main")
        try:
            require_complete_object_graph(local_git)
        except GovernanceError:
            pass
        else:
            raise AssertionError("alternate-ref command must fail closed")
        run_local("config", "--unset", "core.alternateRefsCommand")

        run_local("replace", base, feature)
        try:
            require_complete_object_graph(local_git)
        except GovernanceError:
            pass
        else:
            raise AssertionError("replace refs must fail closed")
        run_local("replace", "-d", base)


def read_at_commit(commit: str, path: str) -> str:
    return run_git("show", f"{commit}:{path}")


def object_exists(commit: str, path: str) -> bool:
    return git_ok("cat-file", "-e", f"{commit}:{path}")


def is_adr_path(path: str, policy: dict[str, Any]) -> bool:
    prefixes = policy["adr_path_prefixes"]
    return (
        path.endswith(".md")
        and Path(path).name.startswith("ADR-")
        and any(path.startswith(str(prefix)) for prefix in prefixes)
    )


def section_body(text: str, heading: str) -> str:
    marker = re.search(rf"(?m)^{re.escape(heading)}\s*$", text)
    if marker is None:
        raise GovernanceError(f"ADR missing required section: {heading}")
    start = marker.end()
    following = HEADING_RE.search(text, start)
    body = text[start: following.start() if following else len(text)].strip()
    if not body:
        raise GovernanceError(f"ADR section is empty: {heading}")
    return body


def validate_adr(path: str, text: str) -> None:
    if not re.search(r"(?mi)^\*\*Change Class\*\*:\s*A(?:\s|\(|$)", text):
        raise GovernanceError(f"{path}: ADR must declare Change Class A")
    for marker in PLACEHOLDER_MARKERS:
        if marker in text:
            raise GovernanceError(f"{path}: unresolved template placeholder: {marker}")
    for heading in REQUIRED_ADR_SECTIONS:
        body = section_body(text, heading)
        if len(re.sub(r"\s+", " ", body)) < 12:
            raise GovernanceError(f"{path}: section too small to be meaningful: {heading}")


def approved_subject(subject: str, authorities: set[str]) -> bool:
    subject = subject.strip().lower()
    if authorities == {"safety"}:
        return subject.startswith(SAFETY_PREFIXES)
    if authorities == {"governance"}:
        return subject.startswith(GOVERNANCE_PREFIXES)
    if authorities == {"safety", "governance"}:
        return subject.startswith(("emergency-safety:", "emergency-safety("))
    raise GovernanceError(f"unexpected Class A authority set: {sorted(authorities)}")


def validate_change_set(base: str, head: str, policy: dict[str, Any]) -> dict[str, Any]:
    require_complete_history()
    validate_declared_roots(base, head)
    validate_root_workflow_contract(base)
    validate_exact_base_head_ancestry(base, head)
    paths = changed_paths(base, head)
    class_a = [(path, classify(path, policy)) for path in paths]
    class_a = [(path, authority) for path, authority in class_a if authority is not None]
    receipt: dict[str, Any] = {
        "changeset_subject": "EXACT_EVENT_BASE_HEAD",
        "base_sha": base,
        "head_sha": head,
        "changed_path_count": len(paths),
        "deletions_and_renames_governed": True,
        "policy_integrity": "PASS",
        "declared_root_existence": "PASS",
        "class_a_detected": bool(class_a),
        "class_a_paths": [path for path, _ in class_a],
        "class_b_policy": "DEFERRED_GOV_POLICY_001",
    }
    if not class_a:
        receipt["class_a_structural_policy"] = "NOT_APPLICABLE"
        return receipt

    adr_paths = [
        path for path in paths
        if is_adr_path(path, policy) and object_exists(head, path)
    ]
    require_changed_adr(bool(class_a), adr_paths)
    for adr in adr_paths:
        validate_adr(adr, read_at_commit(head, adr))

    class_a_paths = {path for path, _ in class_a}
    contributing = 0
    for commit in unique_non_merge_commits(base, head):
        touched = set(commit_changed_paths(commit)) & class_a_paths
        if not touched:
            continue
        contributing += 1
        authorities = {classify(path, policy) for path in touched}
        authorities.discard(None)
        subject = run_git("show", "-s", "--format=%s", commit).strip()
        if not approved_subject(subject, {str(authority) for authority in authorities}):
            raise GovernanceError(
                f"{commit}: Class A commit subject lacks approved prefix for "
                f"{sorted(str(authority) for authority in authorities)}: {subject!r}"
            )
    if contributing == 0:
        raise GovernanceError("Class A diff exists but no unique non-merge Class A commit was identified")

    receipt.update({
        "class_a_structural_policy": "PASS",
        "adr_paths": adr_paths,
        "class_a_commit_prefixes": "PASS",
        "scientific_adequacy": "NOT_ESTABLISHED_BY_THIS_CHECK",
        "test_execution": "NOT_ESTABLISHED_BY_THIS_CHECK",
        "full_ci_status": "NOT_ESTABLISHED_BY_THIS_CHECK",
        "merge_authorization": "NOT_ESTABLISHED_BY_THIS_CHECK",
    })
    return receipt


def self_test() -> None:
    policy = {
        "schema": "symthaea-pr-governance-change-policy-v1",
        "class_a_paths": [
            {"match": match, "path": path, "authority": authority}
            for match, path, authority in REQUIRED_CLASS_A_ROOTS
        ],
        "class_b_policy_state": EXPECTED_CLASS_B_STATE,
        "class_b_paths": [],
        "adr_path_prefixes": list(EXPECTED_ADR_PATH_PREFIXES),
        "class_a_prefix_policy": EXPECTED_PREFIX_POLICY,
        "required_adr_sections": list(REQUIRED_ADR_SECTIONS),
        "authority_nonclaims": ["one", "two", "three", "four"],
    }
    assert classify("src/safety/agent.rs", policy) == "safety"
    assert classify("src/cognitive_loop/thresholds/moral.rs", policy) == "safety"
    assert classify("ordinary.rs", policy) is None
    assert approved_subject("safety(core): tighten", {"safety"})
    assert require_complete_object_graph() is None
    assert not approved_subject("governance(ci): wrong", {"safety"})
    assert approved_subject("governance(ci): tighten", {"governance"})
    assert approved_subject("emergency-safety(ci): coordinated", {"safety", "governance"})
    assert not approved_subject("safety(core): mixed", {"safety", "governance"})
    assert not approved_subject("governance(ci): mixed", {"safety", "governance"})
    assert parse_name_status("M\ta.rs\nD\tb.rs\nR100\told.rs\tnew.rs\n") == [
        "a.rs", "b.rs", "new.rs", "old.rs"
    ]
    assert parse_name_status("C100\tcopy-source.rs\tcopy-target.rs\nT\ttype-change.rs\n") == [
        "copy-source.rs", "copy-target.rs", "type-change.rs"
    ]
    try:
        parse_name_status("R100\tonly-one-path.rs\n")
    except GovernanceError:
        pass
    else:
        raise AssertionError("malformed rename records must be rejected")

    # Adversarial state-transition matrix: protected-root deletion, rename,
    # copy, and mixed-authority transitions all fail closed.
    assert parse_name_status("D\tsrc/safety/agent.rs\n") == [
        "src/safety/agent.rs"
    ]
    assert parse_name_status(
        "R100\tsrc/safety/agent.rs\tsrc/safety/renamed.rs\n"
    ) == [
        "src/safety/agent.rs",
        "src/safety/renamed.rs",
    ]
    assert parse_name_status(
        "C100\tsrc/safety/agent.rs\tsrc/safety/copied.rs\n"
    ) == [
        "src/safety/agent.rs",
        "src/safety/copied.rs",
    ]
    assert approved_subject("emergency-safety(root): coordinated", {"safety", "governance"})
    assert not approved_subject("safety(root): coordinated", {"safety", "governance"})
    assert not approved_subject("governance(root): coordinated", {"safety", "governance"})

    # State-transition matrix: every protected root must exist on at least one
    # side of the exact event boundary.
    root = REQUIRED_CLASS_A_ROOTS[0]
    states = ({"base"}, {"head"}, {"base", "head"})

    def synthetic_exists(state: set[str]):
        def exists(commit: str, match: str, path: str) -> bool:
            return commit in state if (match, path) == (root[0], root[1]) else True
        return exists

    for state in states:
        validate_declared_roots("base", "head", synthetic_exists(state))
    try:
        validate_declared_roots("base", "head", synthetic_exists(set()))
    except GovernanceError:
        pass
    else:
        raise AssertionError("protected root absent from both sides must fail closed")

    require_changed_adr(True, ["docs/compliance/adr/ADR-002-state.md"])
    try:
        require_changed_adr(True, [])
    except GovernanceError:
        pass
    else:
        raise AssertionError("Class A change without changed ADR must fail closed")
    require_changed_adr(False, [])

    history_topology_self_test()\n\n    trusted_workflow = "\n".join(REQUIRED_ROOT_WORKFLOW_SNIPPETS) + "\nuses: actions/checkout@11d5960a326750d5838078e36cf38b85af677262"
    validate_root_workflow_text(trusted_workflow)
    for forbidden in FORBIDDEN_ROOT_WORKFLOW_SNIPPETS:
        try:
            validate_root_workflow_text(f"{trusted_workflow}\n{forbidden}")
        except GovernanceError:
            pass
        else:
            raise AssertionError(
                f"root workflow forbidden pattern must fail closed: {forbidden!r}"
            )
    try:
        validate_root_workflow_text(
            trusted_workflow.replace("actions/checkout@", "actions/checkout@v6")
        )
    except GovernanceError:
        pass
    else:
        raise AssertionError("unpinned checkout must fail closed")

    with tempfile.TemporaryDirectory() as td:
        policy_path = Path(td) / "policy.json"
        policy_path.write_text(json.dumps(policy), encoding="utf-8")
        load_policy(policy_path)
        weakened = json.loads(json.dumps(policy))
        weakened["class_a_paths"] = [
            entry for entry in weakened["class_a_paths"]
            if entry["path"] != ".github/governance-change-policy-v1.json"
        ]
        policy_path.write_text(json.dumps(weakened), encoding="utf-8")
        try:
            load_policy(policy_path)
        except GovernanceError:
            pass
        else:
            raise AssertionError("policy self-root removal must fail closed")

    valid_adr = """# ADR-001: Gate
**Date**: 2026-09-21
**Status**: Proposed
**Change Class**: A (Safety-Critical)

### Scientific Basis
Empirical repository evidence demonstrates the governance defect.

## Impact Analysis
The change affects pull-request governance only.

### Risk Register Impact
No existing product risk is changed; merge-governance risk is reduced.

## Test Evidence
Embedded validator self-tests cover positive and negative structural cases.

## Rollback Plan
Revert the governance commit and restore the previous advisory check.
"""
    validate_adr("docs/compliance/adr/ADR-001-gate.md", valid_adr)
    try:
        validate_adr(
            "docs/compliance/adr/ADR-001-gate.md",
            valid_adr.replace(
                "Empirical repository evidence demonstrates the governance defect.",
                "Citation(s) justifying the change.",
            ),
        )
    except GovernanceError:
        pass
    else:
        raise AssertionError("placeholder ADR must be rejected")
    print("pr_governance_self_test=PASS")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-sha")
    parser.add_argument("--head-sha")
    parser.add_argument(
        "--policy", type=Path,
        default=Path(".github/governance-change-policy-v1.json"),
    )
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    try:
        if args.self_test:
            self_test()
            return 0
        if not args.base_sha or not args.head_sha:
            parser.error("--base-sha and --head-sha are required unless --self-test is used")
        policy = load_policy(args.policy)
        base = require_sha("base_sha", args.base_sha)
        head = require_sha("head_sha", args.head_sha)
        receipt = validate_change_set(base, head, policy)
    except GovernanceError as exc:
        print("pr_governance=FAIL")
        print(f"reason={exc}")
        return 2

    print("pr_governance=PASS")
    for key, value in receipt.items():
        if isinstance(value, list):
            value = ",".join(value)
        elif isinstance(value, bool):
            value = str(value).lower()
        print(f"{key}={value}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
