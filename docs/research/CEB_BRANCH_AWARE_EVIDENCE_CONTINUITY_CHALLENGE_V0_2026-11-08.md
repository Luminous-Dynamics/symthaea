# CEB Branch-Aware Evidence Continuity Challenge V0 — 2026-11-08

**Status:** Public synthetic development material; non-executable.  
**Owner:** CEB-CONTINUITY-001 #10630.  
**Blinding:** This challenge is public and is **not a sealed holdout**. The separate key is also public and is **not secure blinding**. Cases and sibling variants are not independent statistical tasks.

## Annotation instructions

For each case, annotate each claim separately. Bind your judgment to the exact target tip, tree, parent set, scope and visible evidence. Use only the existing #10630 dispositions: retain, revalidate, invalidate, supersede or unresolved.

Do not infer evidence transfer from branch names, commit messages, patch similarity or first-parent history alone. State the minimum discriminating evidence for unresolved or revalidation cases. Preserve historical truth separately from current decision-use validity.

## BHC-S01 — Fork with branch-local change

**Initial state:** `main@M0` and `feature/auth@F0` share ancestor M0. Claim C1 states that the tenant-scoped request path checks actor, tenant and resource before mutation. Its support closure is anchored to M0 and is explicitly scoped to the main branch.

**Branch events:** The feature branch changes an alternate worker ingress and removes a resource-scope check. Main receives an unrelated documentation-only commit. The two branch tips now diverge.

**Claims:** C1-main retains its original scope. C2-feature asserts the same authorization property on the feature tip using the old M0 evidence.

**Agent-visible task:** Determine whether either claim can be used at the two branch tips. Explain whether the fork itself changes truth, whether the feature edit is branch-local, and what evidence is needed before transferring a claim.

## BHC-S02 — First-parent view omits relevant second parent

**Initial state:** A release branch contains an authorization helper and a claim C1 that the helper enforces tenant ownership. A feature branch independently changes that helper's caller contract.

**Branch events:** The branches merge. A first-parent-only or path-simplified history view appears to show no relevant helper change on the release branch, while the second parent contains the caller-contract change. The resulting merge tree is available, but the incomplete history view is the only evidence initially supplied.

**Claims:** C1-release is proposed for retention after the merge. C2 is a downstream route-safety claim derived from C1.

**Agent-visible task:** Decide whether retention is justified from the supplied history. Identify the missing parent evidence and distinguish an absent event from an unobserved event.

## BHC-S03 — Cherry-pick/backport with same patch identity

**Initial state:** Main contains a security fix for a parser boundary. A test/build result supports C1 on main with feature set A and dependency version D2.

**Branch events:** The patch is cherry-picked into a maintained release branch. That branch has feature set B, dependency version D1 and a different caller contract. The patch identity is the same; the commit identity and surrounding source context are different.

**Claims:** C1-main states that the main parser boundary is safe. C2-release claims the backported release parser is safe by inheriting C1-main and its test result.

**Agent-visible task:** Determine whether patch identity is sufficient for evidence transfer. Specify the smallest destination-branch checks and the claim dimensions that may or may not be reused.

## BHC-S04 — Safe parents, unsafe merge resolution

**Initial state:** Parent branch A adds a request authorization check. Parent branch B adds a retry wrapper and preserves the existing authorization call. Each parent has independent passing tests for its own change.

**Branch events:** The merge conflict is resolved by retaining the retry wrapper but accidentally dropping the authorization check from the combined path. The merged tree is available and differs behaviorally from both parent trees.

**Claims:** C1-A and C2-B are historically valid for their parent tips. C3-merge asserts that the combined merge result preserves authorization and retry safety.

**Agent-visible task:** Decide whether parent evidence can be unioned to support C3-merge. Identify the merge-resolution witness and the exact target behavior that must be independently verified.

## BHC-S05 — Rebase/squash changes commit identity

**Initial state:** A reviewed source change has commit C and tree T. A behavioral claim C1 is scoped to source tree T, contract P and configuration W. A provenance claim C2 is scoped to the exact artifact produced from commit C.

**Branch events:** The branch is rebased and squash-merged. The resulting commit identifier is C-prime. The relevant source tree remains byte-identical to T, so the tree identity is unchanged, but no new build provenance for C-prime is supplied.

**Claims:** C1 is a source-behavior claim. C2 is an artifact/provenance claim.

**Agent-visible task:** Separate tree-equivalent behavioral evidence from commit/artifact identity evidence. Avoid both SHA-only invalidation and unqualified provenance transfer.

## BHC-S06 — Revert/reapply and historical exposure

**Initial state:** Release R0 is safe and has a valid approval. A subsequent commit introduces an authorization bypass; release R1 is deployed during the vulnerable interval. A revert restores the prior source tree; later, a corrected fix is reapplied.

**Branch events:** The final branch tip is functionally safe under the current contract. Incident records prove that R1 was deployed during the interval.

**Claims:** C1 states the current branch tip is safe. C2 states no vulnerable release was ever deployed. C3 states R1 was exposed.

**Agent-visible task:** Evaluate present-state safety and historical exposure separately. Do not infer that the revert erases the deployed event or that the historical incident makes the restored tip unsafe.

## BHC-S07 — Global authority versus branch-scoped policy

**Initial state:** Two release branches rely on the same signing authority. A separate feature branch has a branch-local configuration override.

**Branch events:** The signing authority is globally revoked by the organization. Independently, the feature branch changes one local configuration default. No source commit is made to one maintained release branch after revocation.

**Claims:** C1 asserts that historical signed approvals remain admissible for current decisions. C2 asserts that the feature configuration is unchanged on the release branches.

**Agent-visible task:** Distinguish the global authority event from the branch-scoped configuration event. State which claims change on each branch and preserve authenticity/history separately from current authorization.

## BHC-S08 — Missing second parent and incomplete ancestry

**Initial state:** A shallow repository snapshot includes a merge commit and its first parent, but not its second parent or the merge-resolution record. A prior claim C1 says the feature branch added a counter-check.

**Branch events:** A reviewer is asked to approve the current merge tip. The visible path history is clean, but the missing second parent may contain the counter-check or a contradictory change.

**Claims:** C1-merge says the counter-check is present at the current tip. C2-merge says the merged authorization behavior is safe.

**Agent-visible task:** Decide whether the supplied evidence supports retain, revalidate, invalidate, supersede or unresolved. Name the minimum evidence needed to close the ancestry gap; do not guess from missing data.
