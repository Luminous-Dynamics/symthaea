#!/usr/bin/env python3
"""Static authoring validator for the CEB branch-aware continuity development pack.

This validates public documentation structure and claim boundaries only. It does not
parse Git repositories, compute merge bases, construct repository graphs, evaluate
candidate agents, execute code, or establish evidence validity.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

PROTOCOL = "docs/research/CEB_BRANCH_AWARE_EVIDENCE_CONTINUITY_PROTOCOL_V0_2026-11-08.md"
CHALLENGE = "docs/research/CEB_BRANCH_AWARE_EVIDENCE_CONTINUITY_CHALLENGE_V0_2026-11-08.md"
KEY = "docs/research/CEB_BRANCH_AWARE_EVIDENCE_CONTINUITY_KEY_V0_2026-11-08.md"
CASE_IDS = [f"BHC-S{i:02d}" for i in range(1, 9)]
CASE_MARKERS = {
    "BHC-S01": ("fork", "branch-local"),
    "BHC-S02": ("first-parent", "second parent"),
    "BHC-S03": ("cherry-pick", "patch identity"),
    "BHC-S04": ("merge resolution", "combined"),
    "BHC-S05": ("rebase", "tree identity"),
    "BHC-S06": ("revert", "historical exposure"),
    "BHC-S07": ("global authority", "branch-scoped"),
    "BHC-S08": ("missing second parent", "unresolved"),
}
DISPOSITIONS = ("retain", "revalidate", "invalidate", "supersede", "unresolved")
PROTOCOL_PHRASES = (
    "#10630", "#10631", "#10173", "#10314", "#10064", "#10369",
    "not a new repository graph", "git commit", "full parent", "branch scope",
    "merge reconciliation", "cherry-pick", "backport", "rebase", "revert",
    "clean full-recomputation", "not a sealed holdout", "not a new metric lane",
    "no candidate-controlled code", "sequence-complete rate",
    "evidence-disposition macro-f1", "stale-evidence use rate",
    "false-invalidation rate", "revalidation cost per accepted sequence",
    "repair survival rate", "evidence-lineage completeness",
)


def extract_blocks(text: str, pattern: str) -> tuple[list[str], list[str]]:
    matches = list(re.finditer(pattern, text, re.MULTILINE))
    ids: list[str] = []
    blocks: list[str] = []
    for index, match in enumerate(matches):
        ids.append(match.group(1))
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        blocks.append(text[match.start():end])
    return ids, blocks


def validate(protocol: str, challenge: str, key: str) -> list[str]:
    errors: list[str] = []
    challenge_ids, challenge_blocks = extract_blocks(challenge, r"^## (BHC-S\d{2})\b")
    key_ids, key_blocks = extract_blocks(key, r"^## \d+\. (BHC-S\d{2})\b")
    if challenge_ids != CASE_IDS:
        errors.append(f"challenge cases must be BHC-S01..BHC-S08 in order; got {challenge_ids}")
    if key_ids != CASE_IDS:
        errors.append(f"key cases must be BHC-S01..BHC-S08 in order; got {key_ids}")

    challenge_map = dict(zip(challenge_ids, challenge_blocks))
    key_map = dict(zip(key_ids, key_blocks))
    for case_id in CASE_IDS:
        challenge_block = challenge_map.get(case_id, "")
        key_block = key_map.get(case_id, "")
        for phrase in ("Initial state:", "Branch events:", "Claims:", "Agent-visible task:"):
            if phrase not in challenge_block:
                errors.append(f"{case_id}: missing {phrase}")
        for phrase in ("Expected dispositions:", "Reason:", "Discriminating evidence:"):
            if phrase not in key_block:
                errors.append(f"{case_id} key: missing {phrase}")
        for phrase in CASE_MARKERS[case_id]:
            if phrase.lower() not in challenge_block.lower():
                errors.append(f"{case_id}: missing scenario marker {phrase!r}")
            if phrase.lower() not in key_block.lower():
                errors.append(f"{case_id} key: missing scenario marker {phrase!r}")

    for phrase in PROTOCOL_PHRASES:
        if phrase.lower() not in protocol.lower():
            errors.append(f"protocol missing required concept: {phrase}")
    for disposition in DISPOSITIONS:
        if disposition not in protocol:
            errors.append(f"protocol missing continuity disposition: {disposition}")

    if "not a sealed holdout" not in challenge.lower():
        errors.append("challenge must disclose that it is not a sealed holdout")
    if "not secure blinding" not in key.lower():
        errors.append("public key must disclose that it is not secure blinding")
    if "no candidate-controlled code" not in protocol.lower():
        errors.append("protocol must preserve the no-candidate-execution boundary")

    return errors


def minimal_pack() -> tuple[str, str, str]:
    protocol = " ".join(PROTOCOL_PHRASES + DISPOSITIONS) + " no candidate-controlled code"
    challenge_blocks = []
    key_blocks = []
    for index, case_id in enumerate(CASE_IDS, 1):
        markers = " ".join(CASE_MARKERS[case_id])
        challenge_blocks.append(
            f"## {case_id}\nInitial state: x\nBranch events: y\nClaims: z\n"
            f"Agent-visible task: do not guess\n{markers}\n"
        )
        key_blocks.append(
            f"## {index}. {case_id}\nExpected dispositions: unresolved\n"
            f"Reason: x\nDiscriminating evidence: y\n{markers}\n"
        )
    challenge = "\n".join(challenge_blocks) + "\nnot a sealed holdout"
    key = "\n".join(key_blocks) + "\nnot secure blinding"
    return protocol, challenge, key


def self_test() -> list[str]:
    protocol, challenge, key = minimal_pack()
    if validate(protocol, challenge, key):
        return ["valid minimal branch-aware pack rejected"]
    mutations = (
        ("case removed", challenge.replace("## BHC-S03", "## BHC-SXX", 1), key, protocol),
        ("second-parent control removed", challenge.replace("second parent", "unknown parent", 1), key, protocol),
        ("merge-resolution control removed", challenge.replace("merge resolution", "integration event", 1), key, protocol),
        ("branch-scope protocol removed", protocol.replace("branch scope", "scope", 1), challenge, key),
        ("clean oracle removed", protocol.replace("clean full-recomputation", "incremental guess", 1), challenge, key),
        ("continuity disposition removed", protocol.replace("supersede", "missing", 1), challenge, key),
        ("sealed-boundary removed", challenge.replace("not a sealed holdout", "development set", 1), key, protocol),
        ("blinding-boundary removed", key.replace("not secure blinding", "secure key", 1), challenge, protocol),
        ("execution-boundary removed", protocol.replace("no candidate-controlled code", "candidate execution permitted", 1), challenge, key),
        ("S07 global-scope distinction removed", challenge.replace("global authority", "authority", 1), key, protocol),
    )
    for label, mutated_challenge, mutated_key, mutated_protocol in mutations:
        if not validate(mutated_protocol, mutated_challenge, mutated_key):
            return [f"negative self-test accepted mutation: {label}"]
    return []


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        failures = self_test()
        if failures:
            for failure in failures:
                print(f"SELF_TEST_FAIL: {failure}", file=sys.stderr)
            return 1
        print("CEB_BRANCH_AWARE_CONTINUITY_SELF_TEST_PASS: 10 negative controls")

    documents = []
    for relative in (PROTOCOL, CHALLENGE, KEY):
        try:
            documents.append((args.repo_root / relative).read_text(encoding="utf-8"))
        except OSError as exc:
            print(f"ERROR: cannot read {relative}: {exc}", file=sys.stderr)
            return 1

    failures = validate(*documents)
    if failures:
        for failure in failures:
            print(f"ERROR: {failure}", file=sys.stderr)
        return 1

    print("CEB_BRANCH_AWARE_CONTINUITY_PACK_PASS: cases=8 dispositions=5")
    print(
        "CEB_BRANCH_AWARE_CONTINUITY_CLAIM_BOUNDARY: static documentation validation only; "
        "no Git repository history, candidate code, builds, tests, graders or model evaluations executed"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
