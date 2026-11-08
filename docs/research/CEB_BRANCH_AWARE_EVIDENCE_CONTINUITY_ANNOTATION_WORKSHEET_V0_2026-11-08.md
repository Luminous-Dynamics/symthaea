# CEB Branch-Aware Evidence Continuity Annotation Worksheet V0 — 2026-11-08

**Status:** Blank worksheet for public synthetic development cases. No annotations have been collected.  
**Owner:** CEB-CONTINUITY-001 #10630.  
**Blinding limitation:** The challenge and answer key are public. This worksheet can preserve independent first-pass judgments, but the exercise is **not securely blinded** and cannot establish confirmatory inter-rater reliability.

## Reviewer and lock metadata

Use a separate copy for each reviewer. A coordinator should randomize case order and reviewer assignments without changing case content. Reviewers must not see one another's annotations before first-pass lock. Since the key is public, record whether it was previously opened or discussed; a no-key instruction is not cryptographic blinding.

| Field | Reviewer record |
|---|---|
| Reviewer pseudonym / ID | |
| Assignment batch and randomized case order | |
| Assignment issued (UTC) | |
| First-pass submitted (UTC) | |
| Locked artifact SHA-256 | |
| Public key previously viewed or discussed? | |
| Conflicts / relevant prior exposure | |
| Protocol / challenge revision | |

The coordinator computes and records the locked artifact digest after submission. Do not invent a digest before the artifact exists. Corrections after lock are appended; preserve original first-pass answers.

## Decision vocabulary

Use exactly one primary #10630 disposition for each claim × target-tip pair: **retain**, **revalidate**, **invalidate**, **supersede**, or **unresolved**. Keep historical truth separate from current decision-use validity. Missing ancestry is not an absent event. Branch labels, patch IDs, commit messages and path overlap/non-overlap do not establish semantic equivalence by themselves.

## Per-claim annotation template

Copy this table for every claim × target-tip pair.

| Field | Annotation |
|---|---|
| Case ID / claim ID | |
| Repository identity | |
| Target branch label (informational only) | |
| Target resolved tip commit | |
| Target tree identity / digest | |
| Complete known parent commit set | |
| Missing ancestry / shallow-history limitation | |
| Claim proposition and claim kind | |
| Original claim branch/revision scope | |
| Configuration / target / feature scope | |
| Contract / policy / authority epoch | |
| Time interval / historical-vs-current applicability | |
| Support-closure identity | |
| Supporting and contradictory evidence identities | |
| Relevant event IDs and event kinds | |
| Branch-local, transferable, or global scope | |
| Primary continuity disposition | retain / revalidate / invalidate / supersede / unresolved |
| Impact witness (required for non-retain) | |
| Unaffectedness / equivalence witness (required for retain/transfer on mixed-impact cases) | |
| Minimum discriminating evidence request | |
| Predicted result A and resulting disposition | |
| Predicted result B and resulting disposition | |
| Actual result (leave blank before evidence is acquired) | |
| Revalidation / escalation action | |
| Unresolved rationale / open-world remainder | |
| Human minutes and other known review costs | |
| Reviewer rationale and ambiguity notes | |

## BHC-S01 — Fork with branch-local change

Claims: C1-main at main tip; C2-feature at feature tip.

- Did the fork itself change the proposition, or only its applicability?
- Which exact worker-ingress changes enter C2-feature's support closure?
- What would be sufficient to transfer or reject the main-branch claim?

## BHC-S02 — First-parent view omits relevant second parent

Claims: C1-release and downstream C2-route at merge result.

- Is the second parent observed, absent, or unavailable?
- What exact parent/tree/merge-resolution evidence is missing?
- Which disposition is justified before the ancestry gap is closed?

## BHC-S03 — Cherry-pick/backport with same patch identity

Claims: C1-main and C2-release.

- Which configuration, dependency, caller-contract and neighboring-code differences matter?
- Which main observations can be reused, and what equivalence proof is required?
- What is the minimum destination-branch revalidation plan?

## BHC-S04 — Safe parents, unsafe merge resolution

Claims: C1-A, C2-B and C3-merge.

- Which parent claims remain historically true?
- What behavior is introduced by the merge resolution?
- Which target and preservation oracles independently establish the merged result?

## BHC-S05 — Rebase/squash changes commit identity

Claims: C1 source-behavior and C2 artifact/provenance.

- Is the source tree exactly identical, and is contract/configuration scope unchanged?
- Which claim depends on commit, builder or artifact identity?
- What remains valid without transferring provenance incorrectly?

## BHC-S06 — Revert/reapply and historical exposure

Claims: C1 current safety, C2 no vulnerable release ever deployed, C3 R1 exposure.

- What is true about the current tip?
- What is historically true about the deployed interval?
- Which deployment/artifact/event identities establish exposure?

## BHC-S07 — Global authority versus branch-scoped policy

Claims: C1 current authorization and C2 branch configuration.

- What is the revocation's exact global or scoped applicability?
- Which branch-local configuration changed, and where?
- Are authenticity, historical validity and current authorization kept distinct?

## BHC-S08 — Missing second parent and incomplete ancestry

Claims: C1 counter-check presence and C2 merged authorization safety.

- Which parent, tree or merge-resolution artifact is unavailable?
- What minimum evidence would close the ancestry gap?
- Why would retaining from a clean first-parent path overstate the evidence?

## Independent-review and adjudication protocol

1. Assign two reviewers independently; randomize order and balance the eight cases.
2. Give each reviewer the same frozen public challenge and protocol revision and only the intended agent-visible evidence view.
3. Preserve first-pass records, timestamps and hashes before peer discussion.
4. Do not expose one reviewer's labels to the other before both locks. Do not overwrite first-pass labels during adjudication.
5. Record disagreements by claim identity, target tip, disposition, branch scope, parent-set completeness, evidence transfer, impact witness, unaffectedness witness and missingness interpretation.
6. A third expert may adjudicate after both locks. Preserve both original annotations and append adjudication separately.
7. Because the key is public, report this as rubric calibration/usability—not securely blinded reliability.
8. Do not fabricate annotations, reviewer identities, timestamps, hashes or adjudication. This file intentionally contains no completed judgments.

## Reporting

Report case and claim denominators, reviewer-level counts, disagreement categories, adjudication rate and time per case. Sibling claims, branch tips, commits and events from one synthetic anchor are clustered, not independent observations. This public eight-case pack is **not a sealed holdout**.
