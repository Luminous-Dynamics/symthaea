#!/usr/bin/env python3
"""Resolve one GitHub Actions event into Symthaea's CI lifecycle mode.

This module is deliberately independent of PyYAML and GitHub-specific libraries so
its routing semantics can be unit-tested without executing a workflow.

Fail-safe rule:

    unknown / malformed event state -> full-premerge

The only mode allowed to suppress the repository-wide integration matrix is an
explicit draft pull-request event whose payload says ``draft: true``.
"""

from __future__ import annotations

import argparse
import dataclasses
import enum
import json
import pathlib
import sys
from typing import Any, Mapping


class QualificationMode(str, enum.Enum):
    ITERATION = "iteration"
    FULL_PREMERGE = "full-premerge"
    MAIN = "main"
    SCHEDULED = "scheduled"
    MANUAL_FULL = "manual-full"


@dataclasses.dataclass(frozen=True)
class ScopeDecision:
    mode: QualificationMode
    run_full: bool
    reason: str
    fail_safe: bool = False

    def github_outputs(self) -> str:
        return "\n".join(
            (
                f"mode={self.mode.value}",
                f"run_full={'true' if self.run_full else 'false'}",
                f"fail_safe={'true' if self.fail_safe else 'false'}",
                f"reason={self.reason}",
            )
        )


def _full(
    mode: QualificationMode,
    reason: str,
    *,
    fail_safe: bool = False,
) -> ScopeDecision:
    return ScopeDecision(
        mode=mode,
        run_full=True,
        reason=reason,
        fail_safe=fail_safe,
    )


def _pull_request_draft(payload: Mapping[str, Any]) -> bool | None:
    pull = payload.get("pull_request")
    if not isinstance(pull, Mapping):
        return None
    draft = pull.get("draft")
    if isinstance(draft, bool):
        return draft
    return None


def resolve_scope(
    *,
    event_name: str,
    ref: str,
    payload: Mapping[str, Any] | None = None,
) -> ScopeDecision:
    """Resolve one event into a fail-safe CI lifecycle decision.

    Only an explicitly draft ``pull_request`` payload may yield ``iteration``.
    Every unknown, malformed, or unsupported state receives full verification.
    """

    payload = payload or {}
    event_name = event_name.strip()
    ref = ref.strip()

    if event_name == "workflow_dispatch":
        return _full(
            QualificationMode.MANUAL_FULL,
            "manual workflow_dispatch is a deliberate full qualification",
        )

    if event_name == "schedule":
        return _full(
            QualificationMode.SCHEDULED,
            "scheduled regression retains the complete integration surface",
        )

    if event_name == "push":
        if ref == "refs/heads/main":
            return _full(
                QualificationMode.MAIN,
                "push to main requires complete post-merge integration",
            )
        return _full(
            QualificationMode.FULL_PREMERGE,
            "unexpected non-main push failed toward full verification",
            fail_safe=True,
        )

    if event_name == "pull_request":
        draft = _pull_request_draft(payload)
        action = payload.get("action")
        action_text = action if isinstance(action, str) else "unknown"

        if draft is True:
            return ScopeDecision(
                mode=QualificationMode.ITERATION,
                run_full=False,
                reason=(
                    "explicit draft pull request: run iteration invariants and "
                    "independent focused workflows only"
                ),
            )

        if draft is False:
            return _full(
                QualificationMode.FULL_PREMERGE,
                f"non-draft pull request action={action_text} requires full premerge CI",
            )

        return _full(
            QualificationMode.FULL_PREMERGE,
            "pull_request payload lacks a boolean draft field; fail toward full verification",
            fail_safe=True,
        )

    return _full(
        QualificationMode.FULL_PREMERGE,
        f"unknown event {event_name!r}; fail toward full verification",
        fail_safe=True,
    )


def load_event_payload(path: str | None) -> Mapping[str, Any]:
    if not path:
        return {}
    raw = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("GitHub event payload must be a JSON object")
    return raw


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-name", required=True)
    parser.add_argument("--ref", default="")
    parser.add_argument("--event-json")
    parser.add_argument(
        "--format",
        choices=("github-output", "json", "mode"),
        default="github-output",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    try:
        payload = load_event_payload(args.event_json)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        # A malformed event file must never turn a full event into iteration.
        decision = _full(
            QualificationMode.FULL_PREMERGE,
            f"event payload could not be decoded: {type(exc).__name__}",
            fail_safe=True,
        )
    else:
        decision = resolve_scope(
            event_name=args.event_name,
            ref=args.ref,
            payload=payload,
        )

    if args.format == "mode":
        print(decision.mode.value)
    elif args.format == "json":
        print(
            json.dumps(
                {
                    "mode": decision.mode.value,
                    "run_full": decision.run_full,
                    "reason": decision.reason,
                    "fail_safe": decision.fail_safe,
                },
                sort_keys=True,
            )
        )
    else:
        print(decision.github_outputs())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
