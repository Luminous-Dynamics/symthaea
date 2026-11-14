# PLASMA-EVIDENCE-COMPOSITION-001 — method-qualified transfer evidence

**Date:** 2026-11-14  
**Status:** S0 composition contract and synthetic audit  
**Issue:** [#12033](https://github.com/Luminous-Dynamics/symthaea/issues/12033)  
**Composes with:** [PLASMA-CLAIM-TYPE-001 #12026](https://github.com/Luminous-Dynamics/symthaea/issues/12026) and [PLASMA-TRANSFER-001 #12002](https://github.com/Luminous-Dynamics/symthaea/issues/12002)

## Executive thesis

A qualification-transfer bridge must not be admitted because a JSON record contains an endpoint name, confidence interval, specimen IDs and a SHA-shaped digest.

The underlying observation must establish that the selected method and exact edition actually measured that endpoint for the declared material pair and geometry, under a traceable protocol, on the same physical specimen cohort the bridge claims to compare.

The paradigm shift is from **endpoint-named evidence** to **method-qualified observation receipts**.

## Verified gap

The S0 transfer audit in draft [PR #12006](https://github.com/Luminous-Dynamics/symthaea/pull/12006), branch `research/plasma-transfer-v0-1`, validates the structural shape of endpoint interval records, specimen IDs, independence groups, protocol IDs and artifact digest strings.

The method-to-claim audit in [PR #12028](https://github.com/Luminous-Dynamics/symthaea/pull/12028), branch `research/plasma-claim-types-v0-1`, separately checks test-method edition, evidence kind, material class, specimen geometry, endpoint and permitted claim.

Neither audit alone proves that the method-qualified observation is the exact source observation used by a transfer interval. This increment composes them without adding another method registry or replacing the existing transfer audit.

## S0 composition contract

For every required transfer endpoint, the fixture provides:
- exact baseline and candidate profile material-pair class and geometry;
- one baseline and one candidate method-qualified observation receipt;
- exact endpoint identity;
- exact baseline and candidate specimen lists;
- test method, edition, evidence kind and claim type;
- protocol identity and digest;
- raw-data artifact digest.

The composition audit invokes the shared pure method-scope evaluator from `plasma_claim_scope.rs`. It does not accept a caller-supplied `supported=true` flag or a self-attested disposition.

### Required invariants

1. Every required endpoint has exactly one endpoint-evidence record.
2. No unrequested endpoint appears in the bridge evidence.
3. Both observation receipts name the exact transfer endpoint.
4. Both method receipts are supported by the shared edition/material/geometry/endpoint/claim registry.
5. The baseline receipt matches the baseline material/geometry profile; the candidate receipt matches the candidate profile.
6. Baseline and candidate use the same test method, edition, evidence kind, endpoint, requested claim, protocol identity and protocol digest for that comparison.
7. Bridge specimen IDs exactly match the method receipt specimen IDs; baseline and candidate cohorts are distinct and non-overlapping.
8. Receipt identities and raw-data artifact digests cannot be reused across baseline/candidate or required endpoints.
9. A BLOCKED claim receipt blocks transfer; a REJECTED method/claim receipt rejects transfer.
10. Transfer PASS is only a method-scope composition result. Existing qualification-scope, exact joint-interaction bridge and equivalence-interval rules still have to pass independently.

### Disposition composition

- **PASS:** every required endpoint is backed by distinct baseline and candidate receipts admitted by the shared method-scope evaluator, and all lineage/matching checks pass.
- **BLOCKED:** a required endpoint or receipt is missing, the method edition is unknown, or required protocol/evidence fields are incomplete.
- **REJECTED:** a known method/material/geometry/endpoint/claim mismatch exists, cohorts overlap, receipt/artifact reuse occurs, or the baseline/candidate comparison protocol is knowingly inconsistent.

A rejected endpoint cannot be rescued by a good confidence interval. A structurally complete interval cannot rescue an unsupported observation.

## Frozen synthetic fixture

Run:

```bash
cargo run -p xtask -- plasma-evidence-composition-audit
```

Fixture: `data/research/plasma_evidence_composition_fixture_v0_1.json`

The current fixture has **15 synthetic known-answer cases**:
- 2 PASS;
- 3 BLOCKED;
- 10 REJECTED.

Challenge families include:
- valid rigid-plastic D3163 baseline/candidate evidence;
- valid FRP-to-FRP D5868 baseline/candidate evidence;
- contact angle misrepresented as lap shear;
- lap shear escalated to a service-life claim;
- unknown test-method edition;
- material-pair and geometry mismatches;
- missing protocol digest;
- receipt/bridge specimen mismatch;
- overlapping baseline/candidate specimens;
- missing required endpoint;
- receipt reuse between baseline and candidate;
- different protocols used in the nominally matched comparison;
- endpoint evidence reused to stand in for a different outcome.

The exact committed fixture was fetched back from the branch and independently replayed in JavaScript: **15/15 expected dispositions matched** (2 PASS, 3 BLOCKED, 10 REJECTED).

## Baselines and benchmark targets

Compare three strategies on identical fixtures:

1. **Endpoint-name-only:** checks only that each required endpoint label exists.
2. **Method-name-only:** checks a method identifier but ignores exact edition, evidence kind, material pair, geometry and specimen identity.
3. **Composed method-qualified receipts:** applies the existing exact-edition scope evaluator and binds the receipt to the transfer endpoint, cohort, profile and protocol.

Prospective targets:
- zero transfer PASS on planted test-method/material/geometry/endpoint mismatches;
- 100% detection of missing required endpoint receipts;
- 100% detection of receipt-to-bridge specimen identity mismatch;
- zero reuse of one observation/artifact to qualify both baseline and candidate;
- zero implicit conversion from wettability to bond strength, lap shear to fatigue/lifetime, or conditioning practice to retained-strength measurement;
- 100% of PASS outputs have exact method edition and explicit claim ceiling.

Report false-PASS rate separately from BLOCKED rate. Abstention on unknown evidence is correct. Do not combine this with the statistical quality of equivalence intervals or physical transfer.

## SOTA and industry positioning

Industrial plasma vendors provide valuable process telemetry and traceability; the benchmark does not compete by rebuilding their generator controls. The Luminous opportunity is the independent downstream qualification layer.

Likewise, standards identify different evidence roles:
- ASTM D3163-01R23 is the active rigid-plastic single-lap comparative method.
- ASTM D5868-01R23 is the active FRP-to-FRP / FRP-to-metal lap-shear comparative method.
- ASTM D2093-03R25 is a surface-preparation practice.
- ASTM D1151-00R22 and D1183-03R19 are environmental conditioning practices.
- ASTM D4896-01R24 is an interpretation guide for single-lap results, not a strength-test method or a design-allowable authorization.

This composition prevents a valid conditioning reference or surface-preparation record from being mistaken for the product measurement needed to qualify a transfer.

## Ownership and non-duplication

- #12026 / #12028 owns exact method-to-claim scope and the reusable claim-scope evaluator.
- #12002 / #12006 owns qualification scope, bridge evidence, joint-change interactions and equivalence interval structure.
- #11952 owns spatial support and measurement registration.
- #11971 owns survival, aging, censoring and lifetime evidence.
- #11907 owns decision-focused experiment selection.
- #11784 / #11787 provides an existing interlaboratory round-robin and fixture-metrology precedent. Do not duplicate its generic lab/fixture/cross-lab identity engine for plasma.

This S0 module is an integration contract. It does not introduce a new test-method registry, statistics engine, survival estimator, optimizer, materials registry, calibration authority or quality-release system.

## Next integration step

After the method-scope PR and the transfer PR are reconciled against the same main commit, wire this composition evaluator directly into the actual `plasma_transfer` bridge-evidence path so the transfer audit cannot emit PASS without method-qualified observation receipts.

The present increment demonstrates the composed contract on synthetic transfer cases; it does **not** yet modify `plasma_transfer.rs` or make #12006 enforce these receipts.

## Validation boundary

- Source, fixture and command were fetched back from the exact branch.
- Independent synthetic replay: **14/14** expected dispositions matched.
- Five Rust unit tests are authored but **not executed**.
- Rust formatting, compilation, Clippy and exact-head CI remain unverified.
- Hash validation is syntactic; referenced data bytes are not retrieved, authenticated or recalculated.
- No confidence intervals are calculated by this module.
- No physical equivalence, plasma performance, standard compliance, laboratory competence, product qualification, service life, production savings or release authority is claimed.
