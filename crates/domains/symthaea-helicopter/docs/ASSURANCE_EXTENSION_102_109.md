# Helicopter Assurance Extension 102–109

This extension strengthens independent verification, safety-case upkeep, recovery authority, service resilience, diversity evidence, rare-event analysis, evidence longevity, and release authorization. It does **not** claim certification or flight readiness. Every gate retains distinct pass, fail, and incomplete outcomes.

## Patch 102 — Independent implementation verification

`independent_verification` requires complete result matrices across declared test vectors and implementation identities. Shared source or dependency digests, insufficient organizational independence, algorithm correlation for safety-critical vectors, output disagreement, and excessive metric deltas remain visible.

## Patch 103 — Safety-case maintenance

`safety_case_maintenance` treats assurance artifacts as revisioned and time-bounded. Changed assumptions and hazards propagate through trace links, and downstream claims or deployments must be reviewed after the triggering change.

## Patch 104 — Secure recovery authority

`secure_recovery` validates a ground-only, output-disarmed, short-lived recovery request. Actions are allow-listed and require physical-presence evidence plus authenticated approvals across required roles and organizations.

## Patch 105 — Essential-service resilience

`service_resilience` evaluates service dependencies, outage limits, recovery objectives, cycles, evidence, and fallback independence. An essential service does not count as available merely because a fallback was declared.

## Patch 106 — Software diversity assurance

`software_diversity` records source lineage, algorithms, languages, compilers, dependency graphs, teams, training data, and hardware architecture. It quantifies pairwise diversity without claiming statistical independence.

## Patch 107 — Rare-event campaign evidence

`rare_event_campaign` evaluates importance-sampled campaigns using declared target and proposal probabilities, normalized weights, effective sample size, family coverage, indeterminate outcomes, and bounded unsafe-probability gates.

## Patch 108 — Evidence-schema migration integrity

`evidence_schema_migration` requires an allowed, deterministic migration chain with source/target digests, record accounting, declared lossy fields, validation evidence, and rollback coverage.

## Patch 109 — Independent release authorization

`independent_release_authorization` binds approvals and evidence to one candidate digest, enforces separation of duties, rejects author self-approval, requires independent organizations, and prevents expired or failed evidence from authorizing release.

## Full-workspace verification

Run from the Symthaea workspace root:

```bash
cargo fmt --all -- --check
cargo check -p symthaea-helicopter --all-targets --all-features
cargo test -p symthaea-helicopter --all-features
cargo clippy -p symthaea-helicopter --all-targets --all-features -- -D warnings
```

The standalone archive does not include all path dependencies required for these commands.
