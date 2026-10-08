#!/usr/bin/env python3
"""Dependency-free reference oracle for institutional-evolution fixtures.

It intentionally implements the frozen transition semantics independently
of the Rust kernel. It is a conformance oracle, not a performance model.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REQUIRED_AUTHORIZER = {
    "operational": "collective_choice",
    "collective_choice": "constitutional",
    "constitutional": "meta_constitutional",
    "meta_constitutional": None,
}

RULE_KEYS = (
    "operational",
    "collective_choice",
    "constitutional",
    "meta_constitutional",
)


class OracleError(Exception):
    pass


def disposition_for(events: list[dict], initial_rules: dict[str, str]) -> tuple[str, dict[str, object]]:
    current_profile = "profile-v0"
    rules = dict(initial_rules)
    proposals: dict[str, dict] = {}
    decisions: dict[str, dict] = {}
    last = "Initial"
    last_authorizing_rule_hash = None

    for event in events:
        kind = event.get("type")
        if kind == "Proposal":
            proposal_id = event["proposal_id"]
            if proposal_id in proposals:
                return "DuplicateProposal", {"current_profile": current_profile}
            if event["parent_institution_hash"] != current_profile:
                return "ParentMismatch", {"current_profile": current_profile}
            if event["rule_level"] == "constitutional":
                return "ConstitutionalMutationDisabled", {"current_profile": current_profile}
            if event["rule_level"] == "meta_constitutional":
                return "MetaConstitutionalMutationDisabled", {"current_profile": current_profile}
            proposals[proposal_id] = event
            last = "Proposed"
            continue

        if kind == "Adoption":
            proposal_id = event["proposal_id"]
            proposal = proposals.get(proposal_id)
            if proposal is None:
                return "UnknownProposal", {"current_profile": current_profile}
            if proposal_id in decisions:
                return "AlreadyDecided", {"current_profile": current_profile}
            if proposal["parent_institution_hash"] != current_profile:
                return "ParentMismatch", {"current_profile": current_profile}
            if not event["adopted"]:
                decisions[proposal_id] = event
                last = "Rejected"
                continue

            if not event.get("authority") or not event.get("authorizing_rule_hash"):
                return "MissingAuthority", {"current_profile": current_profile}
            required_level = REQUIRED_AUTHORIZER[proposal["rule_level"]]
            if required_level is not None:
                if event.get("authorizing_rule_level") != required_level:
                    return "WrongAuthorityLevel", {"current_profile": current_profile}
                if rules[required_level] != event["authorizing_rule_hash"]:
                    return "AuthorityRuleNotCurrent", {"current_profile": current_profile}
                if event["authorizing_rule_hash"] == proposal["candidate_rule_hash"]:
                    return "SelfModificationUnauthorized", {"current_profile": current_profile}
            decisions[proposal_id] = event
            last_authorizing_rule_hash = event.get("authorizing_rule_hash")
            last = "Adopted"
            continue

        if kind == "Implementation":
            proposal_id = event["proposal_id"]
            proposal = proposals.get(proposal_id)
            decision = decisions.get(proposal_id)
            if proposal is None:
                return "UnknownProposal", {"current_profile": current_profile}
            if not decision or not decision.get("adopted"):
                return "NotAdopted", {"current_profile": current_profile}
            if proposal["parent_institution_hash"] != current_profile:
                return "ParentMismatch", {"current_profile": current_profile}
            if rules[proposal["rule_level"]] == proposal["candidate_rule_hash"]:
                return "AlreadyImplemented", {"current_profile": current_profile}
            rules[proposal["rule_level"]] = proposal["candidate_rule_hash"]
            current_profile = proposal["candidate_institution_hash"]
            last = "Implemented"
            continue

        raise OracleError(f"unknown event type: {kind!r}")

    observables = {
        "current_profile": current_profile,
        "proposal_count": len(proposals),
        "disposition": last,
        "authorizing_rule_hash": last_authorizing_rule_hash,
    }
    for level, key in zip(RULE_KEYS, ("operational_rule", "collective_choice_rule", "constitutional_rule", "meta_constitutional_rule")):
        observables[key] = rules[level]
    return last, observables


def check_case(case: dict, initial_rules: dict[str, str], rejected: bool) -> None:
    disposition, observed = disposition_for(case["events"], initial_rules)
    if rejected:
        expected = case["expected_disposition"]
        if disposition != expected:
            raise OracleError(f'{case["id"]}: expected {expected}, got {disposition}')
        return
    for key, expected in case["expected"].items():
        if observed.get(key) != expected:
            raise OracleError(f'{case["id"]}: {key}: expected {expected!r}, got {observed.get(key)!r}')


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {Path(sys.argv[0]).name} FIXTURES.json", file=sys.stderr)
        return 2
    payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    initial_rules = payload["initial_rule_set"]
    for case in payload["valid"]:
        check_case(case, initial_rules, rejected=False)
    for case in payload["rejected"]:
        check_case(case, initial_rules, rejected=True)
    print(f"verified {len(payload['valid'])} valid + {len(payload['rejected'])} rejected institutional fixtures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
