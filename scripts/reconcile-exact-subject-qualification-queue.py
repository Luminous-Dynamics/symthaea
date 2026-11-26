#!/usr/bin/env python3
"""Audit/cancel stale exact-subject qualification runs without Actions runners.

Dry-run by default. V1 considers only queued/requested/waiting/pending pull_request
runs for reference/custody/oracle workflows with exactly one explicit PR
association. The PR must still be open in this repository, the run head must be
older than the PR's current head, and the current workflow bytes must satisfy the
canonical latest-head policy before cancellation is even proposed.

Destructive mode additionally requires explicit --run allowlisting for every run
that may be cancelled.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

API_ROOT = "https://api.github.com"
API_VERSION = "2022-11-28"
USER_AGENT = "symthaea-exact-subject-queue-reconciler/1"
DEFAULT_REPO = "Luminous-Dynamics/symthaea"
STATES = ("queued", "requested", "waiting", "pending")
HINTS = ("reference", "custody", "oracle")

CANONICAL_IF = (
    "github.event_name != 'pull_request' || "
    "(github.event.pull_request.draft == false && github.event.action != 'closed')"
)
CANONICAL_GROUP = (
    "group: ${{ github.workflow }}-${{ github.event_name == 'workflow_dispatch' "
    "&& github.run_id || github.event.pull_request.number || github.ref }}"
)
CHECKOUT_NAME = "      - name: Checkout exact subject"
VERIFY_NAME = "      - name: Verify exact subject checkout"
EXACT_REF = "ref: ${{ github.event.pull_request.head.sha || github.sha }}"
FETCH_DEPTH = "fetch-depth: 0"
NO_PERSIST = "persist-credentials: false"
EXPECTED_ENV = "EXPECTED_HEAD: ${{ github.event.pull_request.head.sha || github.sha }}"
HEAD_ASSERT = 'run: test \"$(git rev-parse HEAD)\" = \"$EXPECTED_HEAD\"'

CHECKOUT_LINE = re.compile(r"^\s+uses:\s*actions/checkout@([^\s#]+)\s*$", re.MULTILINE)
PINNED_40 = re.compile(r"^[0-9a-f]{40}$")
WRITE_PERMISSION = re.compile(r"^\s+[A-Za-z0-9_-]+:\s*write\s*$", re.MULTILINE)
JOB_PERMISSION = re.compile(r"^    permissions:\s*", re.MULTILINE)
RUNS_ON = re.compile(r"^    runs-on:\s*\S", re.MULTILINE)
TIMEOUT = re.compile(r"^    timeout-minutes:\s*[1-9][0-9]*\s*$", re.MULTILINE)


class ApiError(RuntimeError):
    def __init__(self, status: int, message: str):
        super().__init__(f"GitHub API {status}: {message}")
        self.status = status


@dataclass(frozen=True)
class Candidate:
    run_id: int
    workflow_path: str
    status: str
    created_at: str
    stale_head_sha: str
    pr_number: int
    current_head_sha: str


class Client:
    def __init__(self, token: str | None):
        self.token = token

    def request(self, method: str, path: str) -> Any:
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": API_VERSION,
            "User-Agent": USER_AGENT,
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        req = urllib.request.Request(f"{API_ROOT}{path}", headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                payload = response.read()
                return json.loads(payload.decode()) if payload else None
        except urllib.error.HTTPError as error:
            raw = error.read().decode(errors="replace")
            try:
                message = json.loads(raw).get("message", raw)
            except json.JSONDecodeError:
                message = raw
            raise ApiError(error.code, str(message)) from error

    def get(self, path: str) -> Any:
        return self.request("GET", path)

    def post(self, path: str) -> Any:
        return self.request("POST", path)

    def paginate(self, path: str, key: str) -> list[Any]:
        sep = "&" if "?" in path else "?"
        page = 1
        out: list[Any] = []
        while True:
            batch = self.get(f"{path}{sep}per_page=100&page={page}")[key]
            if not isinstance(batch, list):
                raise RuntimeError(f"expected list for {key}")
            out.extend(batch)
            if len(batch) < 100:
                return out
            page += 1


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", default=DEFAULT_REPO)
    p.add_argument("--apply", action="store_true")
    p.add_argument(
        "--run",
        type=int,
        action="append",
        default=[],
        help="exact stale run ID allowed for destructive cancellation; repeatable",
    )
    p.add_argument("--max-cancellations", type=int, default=50)
    p.add_argument("--rate-reserve", type=int, default=200)
    p.add_argument("--workflow", action="append", default=[])
    p.add_argument("--receipt", type=Path)
    p.add_argument("--self-test", action="store_true")
    return p.parse_args()


def exact_path(path: str) -> bool:
    stem = Path(path).stem.lower()
    return path.startswith(".github/workflows/") and any(hint in stem for hint in HINTS)


def _top_level_section(text: str, key: str) -> list[str]:
    lines = text.splitlines()
    marker = f"{key}:"
    try:
        start = next(i for i, line in enumerate(lines) if line == marker)
    except StopIteration:
        return []
    out: list[str] = []
    for line in lines[start + 1 :]:
        if line and not line.startswith(" ") and not line.lstrip().startswith("#"):
            break
        out.append(line)
    return out


def _pull_request_section(text: str) -> str:
    on = _top_level_section(text, "on")
    for index, line in enumerate(on):
        if re.fullmatch(r"  pull_request:\s*", line):
            block = [line]
            for candidate in on[index + 1 :]:
                if candidate and len(candidate) - len(candidate.lstrip(" ")) <= 2:
                    break
                block.append(candidate)
            return "\n".join(
                line.split("#", 1)[0] for line in block if line.split("#", 1)[0].strip()
            )
    return ""


def _named_step(text: str, name: str) -> str | None:
    lines = text.splitlines()
    marker = f"      - name: {name}"
    positions = [i for i, line in enumerate(lines) if line == marker]
    if len(positions) != 1:
        return None
    start = positions[0]
    end = len(lines)
    for i in range(start + 1, len(lines)):
        if re.match(r"^      - (?:name|uses|run):", lines[i]):
            end = i
            break
        if lines[i] and len(lines[i]) - len(lines[i].lstrip(" ")) <= 4:
            end = i
            break
    return "\n".join(lines[start:end]) + "\n"


def current_policy_valid(text: str) -> bool:
    pr = _pull_request_section(text)
    if not pr or any(event not in pr for event in ("ready_for_review", "converted_to_draft", "closed")):
        return False

    permissions = [
        line for line in _top_level_section(text, "permissions")
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if permissions != ["  contents: read"] or WRITE_PERMISSION.search(text) or JOB_PERMISSION.search(text):
        return False

    concurrency = [
        line.strip() for line in _top_level_section(text, "concurrency")
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if concurrency != [CANONICAL_GROUP, "cancel-in-progress: true"]:
        return False

    if len(RUNS_ON.findall(text)) != 1 or TIMEOUT.search(text) is None:
        return False
    admission_lines = [
        line.strip() for line in text.splitlines()
        if line.strip().startswith("if:")
    ]
    if admission_lines.count(f"if: {CANONICAL_IF}") != 1:
        return False

    checkout = _named_step(text, "Checkout exact subject")
    verify = _named_step(text, "Verify exact subject checkout")
    if checkout is None or verify is None:
        return False

    all_checkouts = CHECKOUT_LINE.findall(text)
    checkout_uses = CHECKOUT_LINE.findall(checkout)
    if len(all_checkouts) != 1 or len(checkout_uses) != 1:
        return False
    if PINNED_40.fullmatch(checkout_uses[0]) is None:
        return False
    if any(needle not in checkout for needle in (EXACT_REF, FETCH_DEPTH, NO_PERSIST)):
        return False
    return EXPECTED_ENV in verify and HEAD_ASSERT in verify


def associated_pr(run: dict[str, Any]) -> int | None:
    prs = run.get("pull_requests") or []
    if len(prs) != 1 or not isinstance(prs[0].get("number"), int):
        return None
    return int(prs[0]["number"])


def same_repo_open(pr: dict[str, Any], repo: str) -> bool:
    return (
        pr.get("state") == "open"
        and pr.get("head", {}).get("repo", {}).get("full_name") == repo
        and isinstance(pr.get("head", {}).get("sha"), str)
    )


def candidate_from(
    run: dict[str, Any],
    pr: dict[str, Any],
    repo: str,
    workflow_text: str,
    workflow_allowlist: set[str],
    run_allowlist: set[int],
) -> Candidate | None:
    path = str(run.get("path") or "")
    run_id = int(run.get("id", -1))
    if workflow_allowlist and path not in workflow_allowlist:
        return None
    if run_allowlist and run_id not in run_allowlist:
        return None
    if not exact_path(path) or run.get("event") != "pull_request":
        return None
    if str(run.get("status")) not in STATES:
        return None
    number = associated_pr(run)
    if number is None or int(pr.get("number", -1)) != number:
        return None
    if not same_repo_open(pr, repo):
        return None
    stale = str(run.get("head_sha") or "")
    current = str(pr["head"]["sha"])
    if not stale or stale == current or not current_policy_valid(workflow_text):
        return None
    return Candidate(
        run_id,
        path,
        str(run["status"]),
        str(run.get("created_at") or ""),
        stale,
        number,
        current,
    )


def fetch_workflow_text(client: Client, repo: str, path: str, ref: str) -> str:
    p = urllib.parse.quote(path, safe="/")
    r = urllib.parse.quote(ref, safe="")
    payload = client.get(f"/repos/{repo}/contents/{p}?ref={r}")
    if payload.get("type") != "file" or payload.get("encoding") != "base64":
        raise RuntimeError(f"unexpected contents payload for {path}@{ref}")
    encoded = "".join(str(payload["content"]).split())
    return base64.b64decode(encoded, validate=True).decode()


def discover(
    client: Client,
    repo: str,
    workflow_allowlist: set[str],
    run_allowlist: set[int],
) -> tuple[list[Candidate], int]:
    runs: dict[int, dict[str, Any]] = {}
    for state in STATES:
        for run in client.paginate(
            f"/repos/{repo}/actions/runs?event=pull_request&status={state}",
            "workflow_runs",
        ):
            path = str(run.get("path") or "")
            run_id = int(run.get("id", -1))
            if workflow_allowlist and path not in workflow_allowlist:
                continue
            if run_allowlist and run_id not in run_allowlist:
                continue
            if exact_path(path):
                runs[run_id] = run

    prs: dict[int, dict[str, Any]] = {}
    texts: dict[tuple[str, str], str] = {}
    out: list[Candidate] = []
    for run in runs.values():
        number = associated_pr(run)
        if number is None:
            continue
        pr = prs.get(number)
        if pr is None:
            pr = client.get(f"/repos/{repo}/pulls/{number}")
            prs[number] = pr
        if not same_repo_open(pr, repo):
            continue
        current = str(pr["head"]["sha"])
        key = (str(run["path"]), current)
        if key not in texts:
            try:
                texts[key] = fetch_workflow_text(client, repo, key[0], key[1])
            except (ApiError, RuntimeError, UnicodeDecodeError, ValueError):
                continue
        item = candidate_from(
            run, pr, repo, texts[key], workflow_allowlist, run_allowlist
        )
        if item:
            out.append(item)
    out.sort(key=lambda item: (item.created_at, item.run_id))
    return out, len(runs)


def rate_limit(client: Client) -> tuple[int, str]:
    core = client.get("/rate_limit")["resources"]["core"]
    return int(core["remaining"]), datetime.fromtimestamp(
        int(core["reset"]), tz=timezone.utc
    ).isoformat()


def safe_limit(requested: int, reserve: int, remaining: int) -> int:
    # run + PR + current workflow + cancellation per destructive iteration
    return min(requested, max(0, (remaining - reserve) // 4))


def still_valid(
    client: Client,
    repo: str,
    c: Candidate,
    workflow_allowlist: set[str],
    run_allowlist: set[int],
) -> bool:
    run = client.get(f"/repos/{repo}/actions/runs/{c.run_id}")
    if int(run.get("id", -1)) != c.run_id or str(run.get("head_sha") or "") != c.stale_head_sha:
        return False
    if associated_pr(run) != c.pr_number or str(run.get("status")) not in STATES:
        return False
    pr = client.get(f"/repos/{repo}/pulls/{c.pr_number}")
    if not same_repo_open(pr, repo) or str(pr["head"]["sha"]) != c.current_head_sha:
        return False
    try:
        text = fetch_workflow_text(client, repo, c.workflow_path, c.current_head_sha)
    except (ApiError, RuntimeError, UnicodeDecodeError, ValueError):
        return False
    return candidate_from(run, pr, repo, text, workflow_allowlist, run_allowlist) is not None


def tool_identity() -> tuple[str, int]:
    raw = Path(__file__).read_bytes()
    return hashlib.sha256(raw).hexdigest(), len(raw)


def safe_workflow() -> str:
    return """name: Example reference

on:
  pull_request:
    types: [opened, synchronize, reopened, ready_for_review, converted_to_draft, closed]
  workflow_dispatch:

permissions:
  contents: read

concurrency:
  group: ${{ github.workflow }}-${{ github.event_name == 'workflow_dispatch' && github.run_id || github.event.pull_request.number || github.ref }}
  cancel-in-progress: true

jobs:
  validate:
    if: github.event_name != 'pull_request' || (github.event.pull_request.draft == false && github.event.action != 'closed')
    runs-on: ubuntu-latest
    timeout-minutes: 5
    steps:
      - name: Checkout exact subject
        uses: actions/checkout@11d5960a326750d5838078e36cf38b85af677262
        with:
          ref: ${{ github.event.pull_request.head.sha || github.sha }}
          fetch-depth: 0
          persist-credentials: false
      - name: Verify exact subject checkout
        env:
          EXPECTED_HEAD: ${{ github.event.pull_request.head.sha || github.sha }}
        run: test \"$(git rev-parse HEAD)\" = \"$EXPECTED_HEAD\"
      - run: python3 scripts/reference.py
"""


def self_test() -> int:
    valid = safe_workflow()
    if not current_policy_valid(valid):
        return 1
    pr = {
        "number": 17,
        "state": "open",
        "head": {"sha": "new", "repo": {"full_name": DEFAULT_REPO}},
    }
    run = {
        "id": 101,
        "path": ".github/workflows/example-reference.yml",
        "event": "pull_request",
        "status": "queued",
        "created_at": "2026-09-27T00:00:00Z",
        "head_sha": "old",
        "pull_requests": [{"number": 17}],
    }
    if candidate_from(run, pr, DEFAULT_REPO, valid, set(), set()) is None:
        return 1
    current = json.loads(json.dumps(run))
    current["head_sha"] = "new"
    if candidate_from(current, pr, DEFAULT_REPO, valid, set(), set()):
        return 1
    fork = json.loads(json.dumps(pr))
    fork["head"]["repo"]["full_name"] = "someone/fork"
    if candidate_from(run, fork, DEFAULT_REPO, valid, set(), set()):
        return 1
    ambiguous = json.loads(json.dumps(run))
    ambiguous["pull_requests"].append({"number": 18})
    if candidate_from(ambiguous, pr, DEFAULT_REPO, valid, set(), set()):
        return 1
    if candidate_from(run, pr, DEFAULT_REPO, valid, set(), {999}):
        return 1

    hostile_workflows = (
        valid.replace("cancel-in-progress: true", "cancel-in-progress: false"),
        valid.replace(
            "      - name: Verify exact subject checkout\n",
            "      - name: Shadow checkout\n"
            "        uses: actions/checkout@v4\n"
            "      - name: Verify exact subject checkout\n",
        ),
        valid.replace(
            "github.event.action != 'closed')",
            "github.event.action != 'closed') || true",
        ),
        valid.replace("contents: read", "contents: write"),
    )
    if any(current_policy_valid(item) for item in hostile_workflows):
        return 1
    if safe_limit(50, 200, 5000) != 50 or safe_limit(50, 200, 240) != 10:
        return 1
    print("Exact-subject qualification queue reconciler self-test: PASS")
    return 0


def main() -> int:
    args = parse_args()
    if args.self_test:
        return self_test()
    if "/" not in args.repo or args.rate_reserve < 0 or not 1 <= args.max_cancellations <= 200:
        print("invalid arguments", file=sys.stderr)
        return 2
    if len(args.run) != len(set(args.run)) or any(run_id <= 0 for run_id in args.run):
        print("--run values must be unique positive integers", file=sys.stderr)
        return 2

    token = os.environ.get("GITHUB_TOKEN")
    if args.apply and not token:
        print("--apply requires GITHUB_TOKEN with Actions write access", file=sys.stderr)
        return 2
    if args.apply and not args.run:
        print("--apply requires at least one explicit --run allowlist entry", file=sys.stderr)
        return 2

    workflow_allowlist = set(args.workflow)
    run_allowlist = set(args.run)
    client = Client(token)
    candidates, scoped = discover(client, args.repo, workflow_allowlist, run_allowlist)
    remaining, reset_at = rate_limit(client)
    limit = (
        safe_limit(args.max_cancellations, args.rate_reserve, remaining)
        if args.apply
        else min(args.max_cancellations, len(candidates))
    )
    actions: list[dict[str, Any]] = []
    cancelled = preserved = gone = 0
    rate_limited = False

    for c in candidates[:limit]:
        record = asdict(c)
        if not args.apply:
            record["action"] = "would_cancel_stale_exact_subject_run"
            actions.append(record)
            continue
        if not still_valid(client, args.repo, c, workflow_allowlist, run_allowlist):
            preserved += 1
            record["action"] = "preserved_after_live_recheck"
            actions.append(record)
            continue
        try:
            client.post(f"/repos/{args.repo}/actions/runs/{c.run_id}/cancel")
            cancelled += 1
            record["action"] = "cancel_requested"
            actions.append(record)
        except ApiError as error:
            if error.status == 409:
                gone += 1
                record["action"] = "already_non_cancellable"
                actions.append(record)
            elif error.status in (403, 429):
                rate_limited = True
                record["action"] = f"rate_limited_{error.status}"
                actions.append(record)
                break
            else:
                raise
        time.sleep(1.0)

    tool_sha256, tool_bytes = tool_identity()
    receipt = {
        "schema": "symthaea.ci.exact-subject-queue-reconciliation.v1",
        "authority": "scheduler-operations-only",
        "scientific_claim": "NONE",
        "generated_at_utc": datetime.now(tz=timezone.utc).isoformat(),
        "repository": args.repo,
        "apply": args.apply,
        "workflow_allowlist": sorted(workflow_allowlist),
        "run_allowlist": sorted(run_allowlist),
        "statuses": list(STATES),
        "scoped_live_runs_examined": scoped,
        "stale_policy_verified_candidates": len(candidates),
        "requested_max": args.max_cancellations,
        "effective_max": limit,
        "rate_remaining_before_mutations": remaining,
        "rate_reset_at": reset_at,
        "rate_reserve": args.rate_reserve,
        "cancel_requested": cancelled,
        "preserved_after_live_recheck": preserved,
        "already_non_cancellable": gone,
        "rate_limited": rate_limited,
        "tool_sha256": tool_sha256,
        "tool_bytes": tool_bytes,
        "python_version": sys.version.split()[0],
        "actions": actions,
    }
    rendered = json.dumps(receipt, indent=2, sort_keys=True)
    print(rendered)
    if args.receipt:
        args.receipt.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ApiError as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(1)
