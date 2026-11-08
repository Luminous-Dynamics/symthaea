# CEB Branch-Aware Evidence Continuity Challenge Key V0 — 2026-11-08

**Status:** Public development key. This file is public and is **not secure blinding**. Do not use it for a blinded reliability estimate or confirmatory holdout.

## 1. BHC-S01 — Fork with branch-local change

**Expected dispositions:** C1-main = retain for its declared main/M0 scope, subject to no global event; C2-feature = revalidate or invalidate depending on the exact changed behavior, and must not inherit C1-main unchanged.

**Reason:** A fork copies history, not applicability. The feature branch has a new source event in the authorization closure; main's documentation-only event is irrelevant to the scoped authorization claim.

**Discriminating evidence:** Exact feature tip/tree, worker ingress caller/operation closure, and target/preservation checks for the alternate path.

## 2. BHC-S02 — First-parent view omits relevant second parent

**Expected dispositions:** C1-release = unresolved until full parent ancestry and merge tree are retrieved; C2-route = unresolved or revalidate after C1's closure is complete.

**Reason:** First-parent and path-simplified history are incomplete views, not proof that no relevant event occurred. A missing second-parent event must not be treated as an absent event.

**Discriminating evidence:** Both parent identities, merge commit tree, relevant second-parent diff and caller-contract source anchor.

## 3. BHC-S03 — Cherry-pick/backport with same patch identity

**Expected dispositions:** C1-main = retain within main scope; C2-release = revalidate. Do not transfer the main test/build result solely on patch identity.

**Reason:** A cherry-pick has distinct destination context. Feature selection, dependency version, caller contract and neighboring code differ. Patch identity is not behavioral equivalence.

**Discriminating evidence:** Destination-branch build world, dependency identity, call-site closure and parser boundary tests under feature set B/D1.

## 4. BHC-S04 — Safe parents, unsafe merge resolution

**Expected dispositions:** C1-A and C2-B = retain historically for their parent tips; C3-merge = invalidate if the supplied merged tree confirms the dropped guard, otherwise revalidate from the merge tree.

**Reason:** Two safe parent states do not imply a safe combined state. The merge resolution creates a new behavioral state. Parent tests cannot substitute for a merged-result witness.

**Discriminating evidence:** Merge commit tree, conflict-resolution diff, complete mutation path and independent authorization plus retry/preservation oracles.

## 5. BHC-S05 — Rebase/squash changes commit identity

**Expected dispositions:** C1 = retain if exact tree, contract P, configuration W and behavioral scope are independently shown equivalent; C2 = revalidate or supersede because the exact artifact/provenance claim is bound to commit C and no C-prime build provenance exists.

**Reason:** The tree identity can remain unchanged and preserve a source-behavior proposition without preserving commit- or artifact-identity propositions. A changed SHA alone is not proof of changed behavior; matching tree bytes alone is not proof of artifact provenance.

**Discriminating evidence:** Exact tree digest and scope comparison for C1; new artifact subject/build provenance and canonical receipt for C2.

## 6. BHC-S06 — Revert/reapply and historical exposure

**Expected dispositions:** C1 = retain for current tip after adequate current-state verification; C2 = invalidate; C3 = retain if incident records bind R1, deployment identity and exposure interval.

**Reason:** The revert restores current source state but does not erase a historical deployed vulnerable interval. Present-state safety and historical-exposure claims are different propositions.

**Discriminating evidence:** Exact current tip/tree and current tests for C1; deployment/event timeline and artifact identity for C2/C3.

## 7. BHC-S07 — Global authority versus branch-scoped policy

**Expected dispositions:** C1 = invalidate for current decision use on every branch within the global revocation scope, while preserving historical authenticity; C2 = retain only for branches whose configuration scope is unaffected, otherwise revalidate.

**Reason:** A global authority event propagates across branches even without a branch commit. A branch-scoped configuration edit must not be propagated to unrelated release branches.

**Discriminating evidence:** Revocation authority/scope/time, signer identity, branch-specific configuration identity and policy epoch.

## 8. BHC-S08 — Missing second parent and incomplete ancestry

**Expected dispositions:** C1-merge = unresolved; C2-merge = unresolved. Do not assign retain from the clean first-parent path.

**Reason:** The missing second parent and merge-resolution record are open-world gaps. The correct action is to obtain discriminating evidence or escalate, not infer a favorable merge.

**Discriminating evidence:** Complete parent DAG, merge-resolution diff/tree and source anchor for the counter-check. If unavailable, preserve unresolved.

## Key-wide scoring notes

- Score each claim × branch-tip pair, not each commit as an independent observation.
- Preserve historical truth separately from current decision-use authority.
- Count false cross-branch transfers, merge-inheritance errors and false global propagation as distinct diagnostics.
- A correct unresolved disposition is preferable to fabricated certainty, but always-unresolved must fail coverage/utility requirements.
- The key is public and **not secure blinding**; human annotation before key exposure is needed for any meaningful calibration exercise.
