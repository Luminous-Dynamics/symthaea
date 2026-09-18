#!/usr/bin/env python3
"""Out-of-band recovery for a saturated draft full-CI queue.

This is the bootstrap counterpart to `.github/workflows/draft-ci-governor.yml`.
Unlike the governor it runs from an operator shell, so it does not need an Actions
runner in order to cancel queued Actions work.

Safety properties:
- dry-run unless `--apply` is explicit;
- only the named full-CI workflow is considered;
- only open, same-repository draft PRs are eligible;
- in-progress runs are excluded unless `--include-in-progress` is explicit;
- each PR is re-read immediately before cancellation;
- destructive mode keeps a configurable GitHub API rate-budget reserve;
- every invocation emits a machine-readable receipt.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

API_ROOT = "https://api.github.com"
DEFAULT_REPO = "Luminous-Dynamics/symthaea"
DEFAULT_WORKFLOW = "ci.yml"
DEFAULT_STATES = ("queued", "requested", "waiting", "pending")
API_VERSION = "2022-11-28"
USER_AGENT = "symthaea-draft-ci-bootstrap-recovery/1"


class GitHubApiError(RuntimeError):
    def __init__(self, status: int, message: str, headers: dict[str, str] | None = None):
        super().__init__(f"GitHub API {status}: {message}")
        self.status = status
        self.headers = headers or {}


@dataclass(frozen=True)
class Candidate:
    run_id: int
    status: str
    created_at: str
    head_sha: str
    head_branch: str
    pr_number: int


class GitHubClient:
    def __init__(self, token: str | None):
        self.token = token

    def request(self, method: str, path: str, body: dict[str, Any] | None = None) -> Any:
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": API_VERSION,
            "User-Agent": USER_AGENT,
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        data = None
        if body is not None:
            data = json.dumps(body).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(
            f"{API_ROOT}{path}", data=data, headers=headers, method=method
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                payload = response.read()
                if not payload:
                    return None
                return json.loads(payload.decode("utf-8"))
        except urllib.error.HTTPError as error:
            payload = error.read().decode("utf-8", errors="replace")
            try:
                message = json.loads(payload).get("message", payload)
            except json.JSONDecodeError:
                message = payload
            raise GitHubApiError(
                error.code,
                str(message),
                {key.lower(): value for key, value in error.headers.items()},
            ) from error

    def get(self, path: str) -> Any:
        return self.request("GET", path)

    def post(self, path: str) -> Any:
        return self.request("POST", path)

    def paginate(self, path: str, item_key: str | None = None) -> list[Any]:
        separator = "&" if "?" in path else "?"
        page = 1
        items: list[Any] = []
        while True:
            payload = self.get(f"{path}{separator}per_page=100&page={page}")
            batch = payload[item_key] if item_key else payload
            if not isinstance(batch, list):
                raise RuntimeError(f"expected list from paginated endpoint {path}")
            items.extend(batch)
            if len(batch) < 100:
                return items
            page += 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--workflow", default=DEFAULT_WORKFLOW)
    parser.add_argument("--apply", action="store_true", help="perform cancellations")
    parser.add_argument(
        "--include-in-progress",
        action="store_true",
        help="also consider in-progress draft full-CI runs",
    )
    parser.add_argument("--max-cancellations", type=int, default=100)
    parser.add_argument("--rate-reserve", type=int, default=200)
    parser.add_argument(
        "--receipt",
        type=Path,
        help="optional path for the JSON reconciliation receipt",
    )
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def same_repo_draft(pr: dict[str, Any], full_name: str) -> bool:
    return (
        pr.get("state") == "open"
        and bool(pr.get("draft"))
        and pr.get("head", {}).get("repo", {}).get("full_name") == full_name
    )


def associated_pr_number(
    run: dict[str, Any],
    open_by_number: dict[int, dict[str, Any]],
    open_by_sha: dict[str, dict[str, Any]],
    same_repo_by_branch: dict[str, list[dict[str, Any]]],
) -> int | None:
    associated = run.get("pull_requests") or []
    if associated:
        number = associated[0].get("number")
        # An explicit GitHub association is authoritative. Never rebind an old
        # run to a different PR merely because the original PR has since closed.
        return number if number in open_by_number else None

    by_sha = open_by_sha.get(run.get("head_sha", ""))
    if by_sha:
        return int(by_sha["number"])
    by_branch = same_repo_by_branch.get(run.get("head_branch", ""), [])
    if len(by_branch) == 1:
        return int(by_branch[0]["number"])
    return None


def discover_candidates(
    client: GitHubClient,
    repo: str,
    workflow: str,
    statuses: Iterable[str],
) -> tuple[list[Candidate], int]:
    owner, name = repo.split("/", 1)
    full_name = f"{owner}/{name}"
    encoded_workflow = urllib.parse.quote(workflow, safe="")

    runs_by_id: dict[int, dict[str, Any]] = {}
    for status in statuses:
        runs = client.paginate(
            f"/repos/{repo}/actions/workflows/{encoded_workflow}/runs"
            f"?event=pull_request&status={urllib.parse.quote(status)}",
            "workflow_runs",
        )
        for run in runs:
            runs_by_id[int(run["id"])] = run

    open_prs = client.paginate(f"/repos/{repo}/pulls?state=open")
    open_by_number = {int(pr["number"]): pr for pr in open_prs}
    open_by_sha = {pr.get("head", {}).get("sha", ""): pr for pr in open_prs}
    same_repo_by_branch: dict[str, list[dict[str, Any]]] = {}
    for pr in open_prs:
        if pr.get("head", {}).get("repo", {}).get("full_name") != full_name:
            continue
        branch = pr.get("head", {}).get("ref", "")
        same_repo_by_branch.setdefault(branch, []).append(pr)

    priority = {status: index for index, status in enumerate(statuses)}
    candidates: list[Candidate] = []
    for run in runs_by_id.values():
        number = associated_pr_number(
            run, open_by_number, open_by_sha, same_repo_by_branch
        )
        if number is None:
            continue
        pr = open_by_number[number]
        if not same_repo_draft(pr, full_name):
            continue
        candidates.append(
            Candidate(
                run_id=int(run["id"]),
                status=str(run["status"]),
                created_at=str(run["created_at"]),
                head_sha=str(run["head_sha"]),
                head_branch=str(run.get("head_branch") or ""),
                pr_number=number,
            )
        )

    candidates.sort(
        key=lambda candidate: (
            priority.get(candidate.status, 999),
            candidate.created_at,
            candidate.run_id,
        )
    )
    return candidates, len(runs_by_id)


def read_rate_limit(client: GitHubClient) -> tuple[int, str]:
    payload = client.get("/rate_limit")
    core = payload["resources"]["core"]
    remaining = int(core["remaining"])
    reset = datetime.fromtimestamp(int(core["reset"]), tz=timezone.utc).isoformat()
    return remaining, reset


def rate_safe_limit(requested: int, reserve: int, remaining: int) -> int:
    # One live PR re-read + one cancellation request per destructive iteration.
    safe = max(0, (remaining - reserve) // 2)
    return min(requested, safe)


def self_test() -> int:
    same_repo = {
        "state": "open",
        "draft": True,
        "number": 7,
        "head": {"repo": {"full_name": DEFAULT_REPO}, "sha": "abc", "ref": "topic"},
    }
    ready = json.loads(json.dumps(same_repo))
    ready["draft"] = False
    fork = json.loads(json.dumps(same_repo))
    fork["head"]["repo"]["full_name"] = "someone/fork"
    if not same_repo_draft(same_repo, DEFAULT_REPO):
        print("self-test failed: same-repo draft rejected", file=sys.stderr)
        return 1
    if same_repo_draft(ready, DEFAULT_REPO):
        print("self-test failed: ready PR accepted", file=sys.stderr)
        return 1
    if same_repo_draft(fork, DEFAULT_REPO):
        print("self-test failed: fork PR accepted", file=sys.stderr)
        return 1
    run = {"pull_requests": [{"number": 7}], "head_sha": "abc", "head_branch": "topic"}
    if associated_pr_number(run, {7: same_repo}, {"abc": same_repo}, {"topic": [same_repo]}) != 7:
        print("self-test failed: authoritative PR association lost", file=sys.stderr)
        return 1
    if associated_pr_number(run, {}, {"abc": same_repo}, {"topic": [same_repo]}) is not None:
        print("self-test failed: closed associated PR was rebound", file=sys.stderr)
        return 1
    if rate_safe_limit(100, 200, 5000) != 100:
        print("self-test failed: normal authenticated rate budget was miscomputed", file=sys.stderr)
        return 1
    if rate_safe_limit(100, 200, 250) != 25:
        print("self-test failed: rate reserve was not preserved", file=sys.stderr)
        return 1
    print("Draft CI bootstrap recovery self-test: PASS")
    return 0


def main() -> int:
    args = parse_args()
    if args.self_test:
        return self_test()
    if args.max_cancellations < 1 or args.max_cancellations > 500:
        print("--max-cancellations must be in [1, 500]", file=sys.stderr)
        return 2
    if args.rate_reserve < 0:
        print("--rate-reserve must be non-negative", file=sys.stderr)
        return 2
    if "/" not in args.repo:
        print("--repo must be OWNER/REPO", file=sys.stderr)
        return 2

    token = os.environ.get("GITHUB_TOKEN")
    if args.apply and not token:
        print("--apply requires GITHUB_TOKEN with Actions write access", file=sys.stderr)
        return 2

    statuses = list(DEFAULT_STATES)
    if args.include_in_progress:
        statuses.append("in_progress")

    client = GitHubClient(token)
    candidates, live_workflow_runs = discover_candidates(
        client, args.repo, args.workflow, statuses
    )
    remaining, reset_at = read_rate_limit(client)
    effective_max = (
        rate_safe_limit(args.max_cancellations, args.rate_reserve, remaining)
        if args.apply
        else min(args.max_cancellations, len(candidates))
    )
    selected = candidates[:effective_max]

    cancelled = 0
    already_gone = 0
    no_longer_eligible = 0
    rate_limited = False
    actions: list[dict[str, Any]] = []

    for candidate in selected:
        if not args.apply:
            actions.append(
                {
                    "run_id": candidate.run_id,
                    "pr": candidate.pr_number,
                    "status": candidate.status,
                    "created_at": candidate.created_at,
                    "head_sha": candidate.head_sha,
                    "action": "would_cancel",
                }
            )
            continue

        live_pr = client.get(f"/repos/{args.repo}/pulls/{candidate.pr_number}")
        if not same_repo_draft(live_pr, args.repo):
            no_longer_eligible += 1
            actions.append(
                {
                    "run_id": candidate.run_id,
                    "pr": candidate.pr_number,
                    "action": "preserved_live_pr_changed",
                }
            )
            continue
        try:
            client.post(f"/repos/{args.repo}/actions/runs/{candidate.run_id}/cancel")
            cancelled += 1
            actions.append(
                {
                    "run_id": candidate.run_id,
                    "pr": candidate.pr_number,
                    "action": "cancel_requested",
                }
            )
        except GitHubApiError as error:
            if error.status == 409:
                already_gone += 1
                actions.append(
                    {
                        "run_id": candidate.run_id,
                        "pr": candidate.pr_number,
                        "action": "already_non_cancellable",
                    }
                )
            elif error.status in (403, 429):
                rate_limited = True
                actions.append(
                    {
                        "run_id": candidate.run_id,
                        "pr": candidate.pr_number,
                        "action": f"rate_limited_{error.status}",
                    }
                )
                break
            else:
                raise
        time.sleep(1.0)

    receipt = {
        "schema": "symthaea.ci.draft-backlog-bootstrap-recovery.v1",
        "repository": args.repo,
        "workflow": args.workflow,
        "apply": args.apply,
        "statuses": statuses,
        "include_in_progress": args.include_in_progress,
        "live_workflow_runs_examined": live_workflow_runs,
        "eligible_same_repo_draft_runs": len(candidates),
        "requested_max": args.max_cancellations,
        "effective_max": effective_max,
        "rate_remaining_before_mutations": remaining,
        "rate_reset_at": reset_at,
        "rate_reserve": args.rate_reserve,
        "cancel_requested": cancelled,
        "already_non_cancellable": already_gone,
        "no_longer_eligible": no_longer_eligible,
        "rate_limited": rate_limited,
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
    except GitHubApiError as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(1)
