#!/usr/bin/env python3
"""Read-only audit of the active pull-request frontier.

This tool deliberately uses GitHub GET endpoints only. It does not close, retarget,
label, merge, comment on, or otherwise mutate pull requests.

The strongest v1 ancestry relation is intentionally narrow:

    child.base.ref == parent.head.ref
    AND
    child.base.sha == parent.head.sha

A matching branch name with a different SHA is reported as drift, not ancestry.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from typing import Any, Iterable

API_VERSION = "2022-11-28"
DEFAULT_REPO = "Luminous-Dynamics/symthaea"
DEFAULT_BRANCH = "main"


@dataclasses.dataclass(frozen=True)
class Pull:
    number: int
    title: str
    draft: bool
    base_ref: str
    base_sha: str
    head_ref: str
    head_sha: str
    html_url: str
    body: str
    created_at: str
    updated_at: str

    @classmethod
    def from_api(cls, raw: dict[str, Any]) -> "Pull":
        return cls(
            number=int(raw["number"]),
            title=str(raw.get("title") or ""),
            draft=bool(raw.get("draft", False)),
            base_ref=str(raw["base"]["ref"]),
            base_sha=str(raw["base"]["sha"]),
            head_ref=str(raw["head"]["ref"]),
            head_sha=str(raw["head"]["sha"]),
            html_url=str(raw.get("html_url") or ""),
            body=str(raw.get("body") or ""),
            created_at=str(raw.get("created_at") or ""),
            updated_at=str(raw.get("updated_at") or ""),
        )


@dataclasses.dataclass(frozen=True)
class ParentRelation:
    kind: str
    parent_number: int | None = None
    parent_head_sha: str | None = None
    reason: str | None = None


@dataclasses.dataclass
class PullAudit:
    pull: Pull
    relation: ParentRelation
    depth: int | None = None
    root_number: int | None = None
    cycle: bool = False
    self_declared_qualification_only: bool = False
    changed_files: list[str] | None = None
    workflow_patch_capsule: bool | None = None
    current_head_run_count: int | None = None
    current_head_queued_run_count: int | None = None
    current_head_in_progress_run_count: int | None = None


class GitHubReadOnlyClient:
    def __init__(self, repo: str, token: str | None = None) -> None:
        self.repo = repo
        self.token = token
        self.remaining: int | None = None

    def _request_json(
        self,
        path: str,
        params: dict[str, Any] | None = None,
    ) -> tuple[Any, dict[str, str]]:
        query = urllib.parse.urlencode(params or {})
        url = f"https://api.github.com/repos/{self.repo}/{path}"
        if query:
            url += f"?{query}"

        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": API_VERSION,
            "User-Agent": "symthaea-pr-frontier-audit-v1",
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"

        req = urllib.request.Request(url, headers=headers, method="GET")
        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                payload = json.load(response)
                response_headers = {
                    key.lower(): value for key, value in response.headers.items()
                }
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")
            raise RuntimeError(
                f"GitHub GET failed: {exc.code} {url}: {detail}"
            ) from exc

        remaining = response_headers.get("x-ratelimit-remaining")
        if remaining is not None:
            try:
                self.remaining = int(remaining)
            except ValueError:
                self.remaining = None

        return payload, response_headers

    def _paginate(
        self,
        path: str,
        params: dict[str, Any] | None = None,
    ) -> Iterable[dict[str, Any]]:
        page = 1
        while True:
            page_params = dict(params or {})
            page_params.update({"per_page": 100, "page": page})
            payload, _ = self._request_json(path, page_params)
            if not isinstance(payload, list):
                raise RuntimeError(
                    f"Expected list from GET {path}, got {type(payload).__name__}"
                )
            for item in payload:
                if not isinstance(item, dict):
                    raise RuntimeError(f"Expected object in GET {path} response")
                yield item
            if len(payload) < 100:
                return
            page += 1

    def list_open_pulls(self) -> list[Pull]:
        return [
            Pull.from_api(raw)
            for raw in self._paginate(
                "pulls",
                {"state": "open", "sort": "created", "direction": "asc"},
            )
        ]

    def list_pull_files(self, number: int) -> list[str]:
        return [
            str(raw["filename"])
            for raw in self._paginate(f"pulls/{number}/files")
        ]

    def count_current_head_runs(self, head_sha: str) -> tuple[int, int, int]:
        payload, _ = self._request_json(
            "actions/runs",
            {
                "head_sha": head_sha,
                "event": "pull_request",
                "per_page": 100,
                "page": 1,
            },
        )
        runs = payload.get("workflow_runs", [])
        if not isinstance(runs, list):
            raise RuntimeError("Malformed Actions workflow-runs response")
        queued = sum(1 for run in runs if run.get("status") == "queued")
        in_progress = sum(1 for run in runs if run.get("status") == "in_progress")
        total = int(payload.get("total_count", len(runs)))
        return total, queued, in_progress


def infer_parent_relations(
    pulls: list[Pull],
    default_branch: str,
) -> dict[int, ParentRelation]:
    heads: dict[str, list[Pull]] = defaultdict(list)
    for pull in pulls:
        heads[pull.head_ref].append(pull)

    relations: dict[int, ParentRelation] = {}
    for child in pulls:
        if child.base_ref == default_branch:
            relations[child.number] = ParentRelation(kind="default_branch_root")
            continue

        candidates = [
            pull
            for pull in heads.get(child.base_ref, [])
            if pull.number != child.number
        ]
        if not candidates:
            relations[child.number] = ParentRelation(
                kind="unresolved_base",
                reason=(
                    f"base branch {child.base_ref!r} is not the head branch "
                    "of another open PR"
                ),
            )
            continue

        if len(candidates) > 1:
            relations[child.number] = ParentRelation(
                kind="ambiguous_base",
                reason=(
                    f"{len(candidates)} open PRs advertise head branch "
                    f"{child.base_ref!r}"
                ),
            )
            continue

        parent = candidates[0]
        if child.base_sha == parent.head_sha:
            relations[child.number] = ParentRelation(
                kind="exact_open_parent",
                parent_number=parent.number,
                parent_head_sha=parent.head_sha,
            )
        else:
            relations[child.number] = ParentRelation(
                kind="base_ref_sha_drift",
                parent_number=parent.number,
                parent_head_sha=parent.head_sha,
                reason=(
                    f"child base SHA {child.base_sha} != open parent head SHA "
                    f"{parent.head_sha}"
                ),
            )

    return relations


def annotate_lineages(audits: dict[int, PullAudit]) -> None:
    for number, audit in audits.items():
        seen: list[int] = []
        cursor = number
        depth = 0

        while True:
            if cursor in seen:
                cycle_start = seen.index(cursor)
                for member in seen[cycle_start:]:
                    audits[member].cycle = True
                    audits[member].depth = None
                    audits[member].root_number = None
                audit.depth = None
                audit.root_number = None
                break
            seen.append(cursor)

            relation = audits[cursor].relation
            if (
                relation.kind != "exact_open_parent"
                or relation.parent_number is None
            ):
                audit.depth = depth
                audit.root_number = cursor
                break

            parent_number = relation.parent_number
            if parent_number not in audits:
                audit.depth = depth
                audit.root_number = cursor
                break

            depth += 1
            cursor = parent_number


def looks_self_declared_qualification_only(pull: Pull) -> bool:
    text = f"{pull.title}\n{pull.body}".lower()
    markers = (
        "qualification-only",
        "qualification only",
        "does not modify product source",
        "no product source",
        "candidate patch plus",
        "exact candidate patch",
    )
    return any(marker in text for marker in markers)


def is_workflow_patch_capsule(paths: list[str]) -> bool:
    """Recognize only the strict current workflow+immutable-patch capsule shape."""
    if not paths:
        return False
    has_workflow = False
    has_patch = False
    for path in paths:
        if path.startswith(".github/workflows/") and path.endswith((".yml", ".yaml")):
            has_workflow = True
            continue
        if path.startswith("docs/release/evidence/") and path.endswith(".patch"):
            has_patch = True
            continue
        return False
    return has_workflow and has_patch


def choose_enrichment_candidates(audits: dict[int, PullAudit]) -> list[int]:
    def score(audit: PullAudit) -> tuple[int, int, int]:
        relation_bonus = 2 if audit.relation.kind == "exact_open_parent" else 0
        qualification_bonus = 3 if audit.self_declared_qualification_only else 0
        return (
            qualification_bonus + relation_bonus,
            relation_bonus,
            audit.pull.number,
        )

    return [
        audit.pull.number
        for audit in sorted(audits.values(), key=score, reverse=True)
    ]


def to_document(
    audits: dict[int, PullAudit],
    repo: str,
    default_branch: str,
    remaining: int | None,
) -> dict[str, Any]:
    values = sorted(audits.values(), key=lambda audit: audit.pull.number)
    relation_counts = Counter(audit.relation.kind for audit in values)
    exact_edges = [
        {"child": audit.pull.number, "parent": audit.relation.parent_number}
        for audit in values
        if audit.relation.kind == "exact_open_parent"
    ]
    max_depth = max(
        (audit.depth or 0 for audit in values if not audit.cycle),
        default=0,
    )

    roots = sorted(
        audit.pull.number
        for audit in values
        if audit.relation.kind != "exact_open_parent" and not audit.cycle
    )

    lineages = []
    for root in roots:
        members = sorted(
            audit.pull.number for audit in values if audit.root_number == root
        )
        if len(members) > 1:
            lineages.append({"root": root, "members": members, "size": len(members)})

    rows = []
    for audit in values:
        rows.append(
            {
                "number": audit.pull.number,
                "title": audit.pull.title,
                "draft": audit.pull.draft,
                "base_ref": audit.pull.base_ref,
                "base_sha": audit.pull.base_sha,
                "head_ref": audit.pull.head_ref,
                "head_sha": audit.pull.head_sha,
                "relation": audit.relation.kind,
                "parent_pr": audit.relation.parent_number,
                "relation_reason": audit.relation.reason,
                "depth": audit.depth,
                "root_pr": audit.root_number,
                "cycle": audit.cycle,
                "self_declared_qualification_only": (
                    audit.self_declared_qualification_only
                ),
                "changed_files": audit.changed_files,
                "workflow_patch_capsule": audit.workflow_patch_capsule,
                "current_head_run_count": audit.current_head_run_count,
                "current_head_queued_run_count": (
                    audit.current_head_queued_run_count
                ),
                "current_head_in_progress_run_count": (
                    audit.current_head_in_progress_run_count
                ),
                "url": audit.pull.html_url,
            }
        )

    return {
        "schema": "symthaea.pr-frontier-audit.v1",
        "repository": repo,
        "default_branch": default_branch,
        "read_only": True,
        "ancestry_rule": (
            "child.base.ref == parent.head.ref AND "
            "child.base.sha == parent.head.sha"
        ),
        "summary": {
            "open_prs": len(values),
            "draft_prs": sum(1 for audit in values if audit.pull.draft),
            "ready_prs": sum(1 for audit in values if not audit.pull.draft),
            "exact_open_parent_edges": relation_counts["exact_open_parent"],
            "default_branch_roots": relation_counts["default_branch_root"],
            "base_ref_sha_drift": relation_counts["base_ref_sha_drift"],
            "ambiguous_base": relation_counts["ambiguous_base"],
            "unresolved_base": relation_counts["unresolved_base"],
            "self_declared_qualification_only": sum(
                1 for audit in values if audit.self_declared_qualification_only
            ),
            "proven_workflow_patch_capsules": sum(
                1 for audit in values if audit.workflow_patch_capsule is True
            ),
            "max_exact_stack_depth": max_depth,
            "multi_pr_exact_lineages": len(lineages),
            "github_rate_limit_remaining": remaining,
        },
        "exact_edges": exact_edges,
        "lineages": sorted(
            lineages,
            key=lambda lineage: (-lineage["size"], lineage["root"]),
        ),
        "pulls": rows,
    }


def render_markdown(doc: dict[str, Any]) -> str:
    summary = doc["summary"]
    lines = [
        "# Symthaea Active PR Frontier Audit v1",
        "",
        f"Repository: `{doc['repository']}`",
        "",
        (
            "This report is read-only. It makes no closure, retargeting, merge, "
            "label, or supersession decision."
        ),
        "",
        "## Summary",
        "",
        f"- Open PRs: **{summary['open_prs']}**",
        (
            f"- Draft / ready: **{summary['draft_prs']} / "
            f"{summary['ready_prs']}**"
        ),
        f"- Exact open-parent edges: **{summary['exact_open_parent_edges']}**",
        f"- Multi-PR exact lineages: **{summary['multi_pr_exact_lineages']}**",
        f"- Maximum exact stack depth: **{summary['max_exact_stack_depth']}**",
        f"- Base-ref/SHA drift cases: **{summary['base_ref_sha_drift']}**",
        f"- Ambiguous bases: **{summary['ambiguous_base']}**",
        f"- Unresolved bases: **{summary['unresolved_base']}**",
        (
            "- Self-declared qualification-only PRs: "
            f"**{summary['self_declared_qualification_only']}**"
        ),
        (
            "- Proven workflow+patch capsules among enriched PRs: "
            f"**{summary['proven_workflow_patch_capsules']}**"
        ),
        "",
        "## Ancestry theorem",
        "",
        "An edge is called exact only when:",
        "",
        "```text",
        "child.base.ref == parent.head.ref",
        "AND",
        "child.base.sha == parent.head.sha",
        "```",
        "",
        (
            "A matching branch name with a different SHA is reported as drift "
            "and does not receive ancestry credit."
        ),
        "",
        "## Largest exact lineages",
        "",
    ]
    for lineage in doc["lineages"][:50]:
        members = " -> ".join(f"#{number}" for number in lineage["members"])
        lines.append(f"- {members}")
    if not doc["lineages"]:
        lines.append("- None")

    lines.extend(
        [
            "",
            "## Pull requests",
            "",
            (
                "| PR | Draft | Relation | Parent | Depth | Qualification-only "
                "text | Proven capsule | Runs | Title |"
            ),
            "|---:|:---:|---|---:|---:|:---:|:---:|---:|---|",
        ]
    )
    for row in doc["pulls"]:
        parent = f"#{row['parent_pr']}" if row["parent_pr"] is not None else ""
        depth = "" if row["depth"] is None else str(row["depth"])
        declared = "yes" if row["self_declared_qualification_only"] else ""
        capsule = (
            "yes"
            if row["workflow_patch_capsule"] is True
            else ("no" if row["workflow_patch_capsule"] is False else "")
        )
        runs = (
            ""
            if row["current_head_run_count"] is None
            else str(row["current_head_run_count"])
        )
        title = str(row["title"]).replace("|", "\\|")
        lines.append(
            f"| #{row['number']} | {'yes' if row['draft'] else 'no'} | "
            f"{row['relation']} | {parent} | {depth} | {declared} | "
            f"{capsule} | {runs} | {title} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "```text",
            "exact stack relation != recommendation to close",
            "qualification-only wording != proven qualification capsule",
            "workflow+patch capsule != qualified candidate",
            "queued workflow != executed evidence",
            "closed-as-superseded != erased research lineage",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--default-branch", default=DEFAULT_BRANCH)
    parser.add_argument("--token-env", default="GITHUB_TOKEN")
    parser.add_argument(
        "--enrich-files",
        action="store_true",
        help="Fetch changed-file lists for selected PRs.",
    )
    parser.add_argument(
        "--enrich-runs",
        action="store_true",
        help=(
            "Fetch current-head pull_request workflow-run counts for selected PRs."
        ),
    )
    parser.add_argument(
        "--enrich-limit",
        type=int,
        default=50,
        help=(
            "Maximum PRs to enrich. Candidates are prioritized by "
            "qualification-only text and exact stacking."
        ),
    )
    parser.add_argument("--format", choices=("json", "markdown"), default="json")
    parser.add_argument("--output", help="Write output to this path instead of stdout.")
    parser.add_argument(
        "--allow-unauthenticated-enrichment",
        action="store_true",
        help=(
            "Permit enrichment without an API token. This can exhaust GitHub's "
            "low unauthenticated rate limit."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    if args.enrich_limit < 0:
        raise SystemExit("--enrich-limit must be >= 0")

    token = os.environ.get(args.token_env) or None
    if (
        (args.enrich_files or args.enrich_runs)
        and not token
        and not args.allow_unauthenticated_enrichment
    ):
        raise SystemExit(
            "Refusing API enrichment without a token. Set the requested token "
            "environment variable or pass --allow-unauthenticated-enrichment "
            "explicitly."
        )

    client = GitHubReadOnlyClient(args.repo, token=token)
    pulls = client.list_open_pulls()
    relations = infer_parent_relations(pulls, args.default_branch)

    audits = {
        pull.number: PullAudit(
            pull=pull,
            relation=relations[pull.number],
            self_declared_qualification_only=(
                looks_self_declared_qualification_only(pull)
            ),
        )
        for pull in pulls
    }
    annotate_lineages(audits)

    if args.enrich_files or args.enrich_runs:
        candidates = choose_enrichment_candidates(audits)[: args.enrich_limit]
        for number in candidates:
            audit = audits[number]
            if args.enrich_files:
                audit.changed_files = client.list_pull_files(number)
                audit.workflow_patch_capsule = is_workflow_patch_capsule(
                    audit.changed_files
                )
            if args.enrich_runs:
                total, queued, in_progress = client.count_current_head_runs(
                    audit.pull.head_sha
                )
                audit.current_head_run_count = total
                audit.current_head_queued_run_count = queued
                audit.current_head_in_progress_run_count = in_progress

    doc = to_document(audits, args.repo, args.default_branch, client.remaining)
    text = (
        json.dumps(doc, indent=2, sort_keys=True) + "\n"
        if args.format == "json"
        else render_markdown(doc) + "\n"
    )

    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(text)
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
