# CEB Review Claim-Support Closure Challenge V0 — 2026-11-08

**Status:** Public synthetic development challenge; source-only; not admitted, executable, scored or sealed.
**Protocol:** [Claim-Support Closure and Selective Invalidation Protocol V0](CEB_REVIEW_CLAIM_SUPPORT_CLOSURE_AND_SELECTIVE_INVALIDATION_PROTOCOL_V0_2026-11-08.md).
**Key:** [Separate public answer key](CEB_REVIEW_CLAIM_SUPPORT_CLOSURE_KEY_V0_2026-11-08.md). It is not secure blinding or a sealed holdout.
**Composition:** #10630 evidence continuity, #10173 repository-twin/change-impact facts, #9112 decision-use validity, #10610 rerun/regrade/reuse policy.

## Reviewer instructions

For every case:
1. Identify each claim and its current decision-use scope.
2. Enumerate the minimum complete support closure, including source, contract, assumption, environment, authority and evaluator/method dependencies where relevant.
3. Identify the exact changed event and changed dependency nodes.
4. Produce the impact or unaffectedness witness with typed dependency paths.
5. Assign one #10630 disposition per claim: retain / revalidate / invalidate / supersede / unresolved.
6. Separate engineering-claim validity from benchmark rerun/regrade/reuse.
7. State the uninspected/open-world remainder and claim ceiling.

Do not infer semantic independence merely from different files. Do not turn missing dependency knowledge into retain. A static source claim and a runtime claim may have different support closures. Include known counter-evidence, failed checks and material rejected alternatives; do not let mutually dependent claims stand as their own independent support.

## CCL-S01 — direct source-anchor edit with a derived claim

**Initial claims:**

- C1: “Function parse_window rejects end values below start.” Scope: source-level behavior of parse_window at tree T0.
- C2: “The CLI option is documented as an inclusive end bound.” Scope: user-visible option contract at T0.

**Support closure:**

- C1: source anchor src/window.rs::parse_window; direct caller src/cli.rs::parse_args; contract clause docs/cli.md §2.1.
- C2: contract anchor docs/cli.md §2.1; CLI help string src/cli.rs::WINDOW_HELP.
- No shared source hunk is declared between parse_window and the help string.

**Change event E1:** a source edit changes the parse_window predicate but does not touch the caller, CLI help, or documentation.

**Reviewer question:** Which claim is affected? Is C2 safely retained because it states the documented option contract, or must it be revalidated because the implementation contract may have changed? What evidence distinguishes a historically true documentation claim from current implementation conformance?

## CCL-S02 — shared upstream dependency with different downstream claims

**Initial claims:**

- C1: “HTTP requests are authenticated using the current tenant key.”
- C2: “Background exports are authenticated using the current tenant key.”

**Support closure:**

- C1: request middleware auth/http.rs; shared key provider security/key_provider.rs::current_key; tenant-key rotation policy security/policy.md.
- C2: export worker jobs/export.rs; the same key provider security/key_provider.rs::current_key; tenant-key rotation policy.
- C1 and C2 share the provider and policy nodes, but have different consuming paths and operations.

**Change event E2:** the key provider changes its fallback behavior when the active tenant key is unavailable. The new behavior can return a process-wide bootstrap key for HTTP requests, while exports explicitly deny on missing tenant key. The event evidence includes the provider diff and one caller-specific policy contract, but does not include the complete worker implementation.

**Reviewer question:** Which closures contain the changed provider node? Can C2 be retained based on the declared policy, or is the worker path evidence incomplete? Why is “the change is only in the provider file” not enough to assign one disposition to both claims?

## CCL-S03 — build/feature transition with a source-level claim

**Initial claims:**

- C1: “The default build selects strict-policy for Linux x86_64.”
- C2: “The source tree contains both strict and permissive implementations.”

**Support closure:**

- C1: exact Cargo invocation, workspace feature defaults, target triple, cfg selection, compiler invocation, artifact digest and build provenance.
- C2: exact source tree digest and the two cfg-gated definitions; no artifact selection claim.

**Change event E3:** a Cargo manifest change removes strict-policy from default features. No Rust source file changes. The new manifest is available, but no new build or artifact provenance is supplied.

**Reviewer question:** Which claim is invalidated, which can remain historically true, and which requires revalidation? Does C2 remain valid as a source-structure claim even though C1's build-world selection changes?

## CCL-S04 — verifier-only update

**Initial claims:**

- C1: “Artifact A was built from source tree T and passed verifier V1 under environment E.”
- C2: “The reviewed source at T contains the checked authorization guard.”

**Support closure:**

- C1: source tree T; build recipe and toolchain; artifact digest A; verifier V1 identity; test inputs; environment E; original run receipt.
- C2: source anchor at T; contract clause and scope; no dependency on verifier V1's verdict.

**Change event E4:** verifier V1 is found to have a false-negative bug and is replaced by V2. Candidate source T, artifact A, build recipe and environment E are byte-identical. V2's new evaluation of the retained artifact is not yet run.

**Reviewer question:** What must be regraded, what must not be rerun by default, and which source-level claim can remain? Do not confuse invalidation of a grade with invalidation of the underlying source claim.

## CCL-S05 — authority-root revocation

**Initial claims:**

- C1: “Signer S approved artifact A on date D.”
- C2: “Artifact A was produced from source tree T by build B.”

**Support closure:**

- C1: signed approval statement; signer identity S; trust root R; signature-verification policy; time D.
- C2: provenance statement linking A to T and B; build record; artifact digest A.
- C2 does not depend on signer S or trust root R.

**Change event E5:** trust root R is revoked because the signing key is compromised. The historical signed bytes and artifact provenance remain unchanged.

**Reviewer question:** Does the historical fact “a signature exists” become false? Is C1 still admissible for current authorization decisions? Can C2 be retained if its provenance chain is independently valid and does not rely on R?

## CCL-S06 — metadata-only event with incomplete closure

**Initial claims:**

- C1: “The release checklist records that the compatibility suite passed for release R.”
- C2: “The API remains compatible with all supported downstream callers.”

**Support closure:**

- C1: retained run receipt; exact compatibility-suite version; source tree; target/configuration; artifact identity.
- C2: API contract; complete supported-caller inventory; compatibility test coverage; feature and downstream dependency matrix.
- The caller inventory is not supplied. C2's current closure is explicitly incomplete.

**Change event E6:** a documentation page title and display label are corrected; source tree, receipt, test suite, build configuration and artifact digests are unchanged. The event does not touch the undocumented caller inventory.

**Reviewer question:** Can C1 be retained by exact input/receipt identity? Can C2 also be retained because the edit is metadata-only, or must it remain unresolved because its support closure was already incomplete? Do not allow a benign event to cure a pre-existing completeness gap.

## Submission worksheet

| Field | Reviewer response |
|---|---|
| Case ID | |
| Claim ID | |
| Claim kind and decision-use scope | |
| Support closure nodes | |
| Typed dependency edges | |
| Closure completeness argument | |
| Change event and changed nodes | |
| Impact witness / path | |
| Unaffectedness witness / path | |
| #10630 disposition | retain / revalidate / invalidate / supersede / unresolved |
| #10610 treatment | rerun / regrade / reuse / assurance re-evaluation / not applicable |
| Open-world remainder | |
| Claim ceiling | |

## Development boundary

Six cases are six development anchors, not independent statistical samples. The answer key is public; this challenge is not a sealed holdout. No candidate code, builds, tests, PoCs, hidden labels, grader, receipt or production event is executed.
