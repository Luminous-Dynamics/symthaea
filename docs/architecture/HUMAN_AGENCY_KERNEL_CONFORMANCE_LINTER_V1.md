# Human Agency Kernel — Conformance Manifest Linter v1

Status: HAK-006 implementation candidate / audit-only tooling

Parent: HAK-005 Proof Obligations & Conformance.

## 1. Purpose

HAK-006 turns a narrow subset of HAK-005's evidence-accounting rules into executable tooling.

The tool deliberately does **not** answer:

```text
is this authority legitimate?
is this system safe?
is this policy ethical?
may this machine act?
```

It answers a narrower question:

> Is this conformance manifest internally honest and structurally consistent with the evidence claims it makes?

Core theorem:

```text
LintPass != HAKQualification
```

The linter is therefore an audit aid, not an authorization oracle.

## 2. Files

```text
scripts/hak_conformance_lint.py
tests/python/test_hak_conformance_lint.py
docs/architecture/hak/conformance-profile-v1.schema.json
docs/architecture/hak/examples/fabrication-partition-lease.profile.json
```

The JSON Schema documents structural shape. The Python linter adds cross-field and dependency semantics that JSON Schema alone should not pretend to establish.

## 3. First executable invariants

The first implementation intentionally validates only evidence bookkeeping.

### HAK-LINT-001 — exact implementation lineage

If an implementation commit is supplied, it must be an exact lowercase 40-hex commit identifier.

E5+ qualification requires such an exact implementation commit.

```text
branch name / latest / main
!= exact-head identity
```

### HAK-LINT-002 — evidence-bearing statuses need evidence references

Statuses that claim observed or executed evidence require at least one explicit reference:

```text
SourceObserved
TestSourceObserved
PropertyViolated
Qualified
Queued
InfrastructureFailed
TestFailed
```

This does not prove that a reference is trustworthy; it prevents evidence-bearing claims with no named evidence.

### HAK-LINT-003 — only `Qualified` may claim an evidence tier

```text
TestSourceObserved + evidence_tier=E2 -> reject
Queued + evidence_tier=E5            -> reject
```

This freezes two HAK-005 distinctions:

```text
TestSourceObserved != E2
CIQueued != E5
```

### HAK-LINT-004 — explicit `NotApplicable`

`NotApplicable` must include a reason.

The linter does not decide whether the reason is correct.

### HAK-LINT-005 — findings and blockers are explicit

```text
OpenFinding -> finding_ref required
BlockedBy   -> blocker obligation IDs required
```

Referenced dependencies/blockers must exist.

### HAK-LINT-006 — policy-sensitive claims name policy lineage

A policy-sensitive obligation must bind either a profile-wide or local policy lineage.

```text
SameCode + DifferentAuthorityPolicy
!= SameQualifiedBoundary
```

### HAK-LINT-007 — dependency graph is closed and acyclic

Unknown dependency identifiers, self-dependencies, and dependency cycles are rejected.

This does not prove the declared graph is complete.

### HAK-LINT-008 — critical-path closure

If an end-to-end claim is marked `Qualified`, every declared critical obligation must itself be qualified at the claim's required evidence tier or stronger.

Therefore:

```text
CriticalUnknown
+ QualifiedEndToEndClaim
-> lint error
```

This prevents test-count optimism from hiding one unqualified load-bearing join.

## 4. What the linter intentionally does not do

Version 1 does not:

- fetch CI providers;
- verify evidence-reference authenticity;
- cryptographically validate evidence artifacts;
- inspect source code;
- infer missing obligations;
- judge whether `NotApplicable` is legitimate;
- decide constitutional legitimacy;
- assess human worth, consciousness, reputation, or civic standing;
- grant or revoke runtime permissions;
- issue HAK certificates;
- query a model for safety truth.

Those would either require separate trusted evidence resolvers or would cross the line into an authority oracle.

## 5. Example

The included fabrication lease profile is intentionally honest:

```text
lease provenance        -> SourceObserved
overlap regression test -> TestSourceObserved
effect reachability     -> Unknown
rights floor            -> NotApplicable(reason)
end-to-end claim        -> Unknown
```

That manifest should lint successfully.

This demonstrates:

```text
HonestUnknown != LintFailure
```

The linter rejects contradictory evidence accounting, not uncertainty.

## 6. Usage

```bash
python scripts/hak_conformance_lint.py \
  docs/architecture/hak/examples/fabrication-partition-lease.profile.json
```

Machine-readable output:

```bash
python scripts/hak_conformance_lint.py \
  docs/architecture/hak/examples/fabrication-partition-lease.profile.json \
  --json
```

Success prints:

```text
VALID: ...
LintPass != HAKQualification
```

and exits `0`.

A lint violation exits `1`.

## 7. Qualification obligations for HAK-006 itself

HAK-006 must obey HAK-005.

At minimum:

```text
HAK-LINTER-SEM-001
Lint success is never represented as runtime authority qualification.

HAK-LINTER-EVID-001
A non-qualified status cannot claim an execution evidence tier.

HAK-LINTER-EVID-002
E5+ requires an exact implementation commit and explicit evidence refs.

HAK-LINTER-COMP-001
A qualified end-to-end claim cannot bypass an unqualified declared critical obligation.

HAK-LINTER-GRAPH-001
Dependency cycles/unknown dependencies are rejected.

HAK-LINTER-POLICY-001
Policy-sensitive claims cannot omit policy lineage.
```

Required evidence should include negative tests and exact-head hosted execution.

The fact that the linter can lint its own manifest would not establish correctness.

```text
SelfAcceptance != IndependentQualification
```

## 8. Future safe extensions

Potential later additions that preserve the audit-only boundary:

- resolvers that verify CI/evidence references without changing authority;
- exact policy digests rather than free-form lineage strings;
- reproducibility capsule references;
- SARIF output for code-review UX;
- graph visualization;
- mutation-test evidence references;
- environment/lock digests;
- explicit supersession/withdrawal of stale evidence;
- signed audit manifests.

Each extension should distinguish:

```text
validating evidence metadata
```

from:

```text
deciding runtime authorization
```

## 9. Non-claims

This tranche does not claim:

- a lint-clean profile is HAK conformant;
- a HAK-conformant system is universally safe or ethical;
- all obligations can be machine-checked;
- evidence tiers form one universal assurance score;
- all domains should use the same runtime authority representation.

The intended role is narrower:

> Make it mechanically harder for evidence bookkeeping to say more than the evidence actually supports.
