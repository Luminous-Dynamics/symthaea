#!/usr/bin/env python3
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Safely identify and optionally cancel superseded queued GitHub Actions runs.

The script is intentionally conservative:
- dry-run by default;
- only considers pull_request runs unless --include-push is supplied;
- never touches workflow_dispatch, schedule, or the newest run in a group;
- only cancels a run when a strictly newer run exists for the same workflow and PR/branch;
- requires both --apply and --yes before mutating GitHub state.

It uses the authenticated `gh` CLI and never reads or prints an auth token.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

QUEUE_STATES = {"queued", "pending", "requested", "waiting"}
UTC = dt.timezone.utc


@dataclass(frozen=True)
class Run:
    run_id: int
    workflow_id: int
    workflow_name: str
    event: str
    head_branch: str
    created_at: dt.datetime
    group_subject: str

    @property
    def group_key(self) -> tuple[int, str, str]:
        return (self.workflow_id, self.event, self.group_subject)


def gh(*args: str) -> str:
    completed = subprocess.run(
        ["gh", *args],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"gh {' '.join(args)} failed: {detail}")
    return completed.stdout


def detect_repo() -> str:
    value = gh("repo", "view", "--json", "nameWithOwner", "--jq", ".nameWithOwner").strip()
    if "/" not in value:
        raise RuntimeError(f"could not determine owner/repo from gh: {value!r}")
    return value


def parse_time(value: str) -> dt.datetime:
    return dt.datetime.fromisoformat(value.replace("Z", "+00:00"))


def subject_for(raw: dict[str, Any]) -> str | None:
    event = raw.get("event")
    if event == "pull_request":
        prs = raw.get("pull_requests") or []
        if prs and isinstance(prs[0], dict) and prs[0].get("number") is not None:
            return f"pr:{prs[0]['number']}"
        branch = raw.get("head_branch")
        return f"branch:{branch}" if branch else None
    if event == "push":
        branch = raw.get("head_branch")
        return f"branch:{branch}" if branch else None
    return None


def fetch_recent_runs(repo: str, pages: int) -> list[dict[str, Any]]:
    runs: dict[int, dict[str, Any]] = {}
    for page in range(1, pages + 1):
        payload = json.loads(
            gh(
                "api",
                "-H",
                "Accept: application/vnd.github+json",
                f"repos/{repo}/actions/runs?per_page=100&page={page}",
            )
        )
        page_runs = payload.get("workflow_runs") or []
        if not page_runs:
            break
        for raw in page_runs:
            if isinstance(raw, dict) and isinstance(raw.get("id"), int):
                runs[raw["id"]] = raw
    return list(runs.values())


def normalize_runs(raw_runs: list[dict[str, Any]], include_push: bool) -> list[Run]:
    allowed_events = {"pull_request"}
    if include_push:
        allowed_events.add("push")

    normalized: list[Run] = []
    for raw in raw_runs:
        status = str(raw.get("status") or "")
        event = str(raw.get("event") or "")
        if status not in QUEUE_STATES or event not in allowed_events:
            continue
        subject = subject_for(raw)
        created = raw.get("created_at")
        if subject is None or not isinstance(created, str):
            continue
        normalized.append(
            Run(
                run_id=int(raw["id"]),
                workflow_id=int(raw.get("workflow_id") or 0),
                workflow_name=str(raw.get("name") or "unknown"),
                event=event,
                head_branch=str(raw.get("head_branch") or ""),
                created_at=parse_time(created),
                group_subject=subject,
            )
        )
    return normalized


def superseded_runs(runs: list[Run], min_age_seconds: int) -> list[tuple[Run, Run]]:
    groups: dict[tuple[int, str, str], list[Run]] = defaultdict(list)
    for run in runs:
        groups[run.group_key].append(run)

    now = dt.datetime.now(UTC)
    candidates: list[tuple[Run, Run]] = []
    for group in groups.values():
        if len(group) < 2:
            continue
        ordered = sorted(group, key=lambda item: (item.created_at, item.run_id))
        newest = ordered[-1]
        for old in ordered[:-1]:
            age = (now - old.created_at.astimezone(UTC)).total_seconds()
            if age >= min_age_seconds:
                candidates.append((old, newest))
    return sorted(candidates, key=lambda pair: (pair[0].created_at, pair[0].run_id))


def cancel(repo: str, run_id: int) -> None:
    gh(
        "api",
        "--method",
        "POST",
        "-H",
        "Accept: application/vnd.github+json",
        f"repos/{repo}/actions/runs/{run_id}/cancel",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", help="owner/repo; defaults to the current gh repository")
    parser.add_argument("--pages", type=int, default=10, help="recent 100-run pages to inspect")
    parser.add_argument(
        "--min-age-seconds",
        type=int,
        default=120,
        help="do not cancel very recent superseded runs (default: 120)",
    )
    parser.add_argument(
        "--include-push",
        action="store_true",
        help="also consider superseded push runs; pull_request only by default",
    )
    parser.add_argument("--max-cancel", type=int, default=100, help="hard mutation cap")
    parser.add_argument("--apply", action="store_true", help="perform cancellations")
    parser.add_argument(
        "--yes",
        action="store_true",
        help="required together with --apply; prevents accidental mutation",
    )
    args = parser.parse_args()

    if shutil.which("gh") is None:
        print("ERROR: GitHub CLI (`gh`) is required", file=sys.stderr)
        return 2
    if args.pages < 1 or args.pages > 50:
        print("ERROR: --pages must be between 1 and 50", file=sys.stderr)
        return 2
    if args.min_age_seconds < 0:
        print("ERROR: --min-age-seconds must be non-negative", file=sys.stderr)
        return 2
    if args.max_cancel < 1:
        print("ERROR: --max-cancel must be positive", file=sys.stderr)
        return 2
    if args.apply != args.yes:
        print("ERROR: mutation requires BOTH --apply and --yes", file=sys.stderr)
        return 2

    try:
        repo = args.repo or detect_repo()
        raw = fetch_recent_runs(repo, args.pages)
        active = normalize_runs(raw, args.include_push)
        candidates = superseded_runs(active, args.min_age_seconds)
    except (RuntimeError, ValueError, json.JSONDecodeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    print(f"Repository: {repo}")
    print(f"Inspected recent runs: {len(raw)}")
    print(f"Queued/requested/waiting candidates in scope: {len(active)}")
    print(f"Safely superseded runs: {len(candidates)}")
    print()

    for old, newest in candidates[: args.max_cancel]:
        age = int((dt.datetime.now(UTC) - old.created_at.astimezone(UTC)).total_seconds())
        print(
            f"{old.run_id:>12}  {old.workflow_name:<28.28}  "
            f"{old.group_subject:<12} age={age:>6}s  -> keep {newest.run_id}"
        )

    if len(candidates) > args.max_cancel:
        print(f"... {len(candidates) - args.max_cancel} more omitted by --max-cancel")

    selected = candidates[: args.max_cancel]
    if not args.apply:
        print("\nDRY RUN: no GitHub state changed.")
        print("To cancel exactly the listed superseded runs, repeat with --apply --yes.")
        return 0

    cancelled = 0
    for old, _newest in selected:
        try:
            cancel(repo, old.run_id)
        except RuntimeError as error:
            # A concurrent GitHub cancellation/supersession is benign; report it
            # rather than broadening the cancellation selection.
            print(f"WARN: could not cancel {old.run_id}: {error}", file=sys.stderr)
            continue
        cancelled += 1
        print(f"cancelled {old.run_id}")

    print(f"\nCancelled {cancelled} superseded run(s); newest runs were preserved.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
