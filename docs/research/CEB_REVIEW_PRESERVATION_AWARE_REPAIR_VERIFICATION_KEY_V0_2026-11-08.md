# CEB Review Preservation-Aware Repair Verification Challenge Key V0 — 2026-11-08

**Status:** Public synthetic development answer key; not hidden ground truth, secure blinding or a sealed holdout.
**Challenge:** [Preservation-Aware Repair Verification Challenge V0](CEB_REVIEW_PRESERVATION_AWARE_REPAIR_VERIFICATION_CHALLENGE_V0_2026-11-08.md).
**Protocol:** [Preservation-Aware Repair Verification Protocol V0](CEB_REVIEW_PRESERVATION_AWARE_REPAIR_VERIFICATION_PROTOCOL_V0_2026-11-08.md).

## 1. General adjudication rule

The canonical variant is designed to conform and the mutant to violate the stated requirement. Reviewers should be rewarded for stating exactly why, not for proposing extra hardening or stylistic changes.

A valid answer includes both:
- a target witness that demonstrates the mutant's contract violation; and
- a preservation witness that proves the repair does not break the nearest valid behavior.

Do not let a proposed patch define its own oracle. Do not infer broader correctness from the supplied example inputs. If implementation details required by the contract are omitted (for example, durable audit semantics), preserve the unresolved authority/evidence gap.

## 2. PAR-S01 — half-open interval

**Canonical disposition:** conforms.
**Mutant disposition:** violates.
**Target witness:** start=10, end=20, x=20. The mutant returns true although the end is excluded by contract.
**Preservation witness:** start=10, end=20, x=10. Both implementations include the start; a patch that changes >= to > would break valid behavior.
**Minimal discriminating oracle:** assert false at x=end and true at x=start; include one interior and one exterior point to check the other bounds.
**Minimal patch:** change <= to < in the upper-bound predicate.
**Over-correction trap:** do not reject the canonical implementation for not including the endpoint. The contract deliberately defines a half-open interval.
**Claim ceiling:** the supplied predicate conforms to the stated interval contract for the tested arithmetic/domain. Integer overflow and invalid start/end ordering require separate contract clauses if they are in scope.

## 3. PAR-S02 — checked arithmetic

**Canonical disposition:** conforms.
**Mutant disposition:** violates.
**Target witness:** current=u32::MAX, delta=1. The mutant returns u32::MAX, silently saturating instead of returning Overflow.
**Preservation witness:** current=10, delta=5. Both should return 15; a patch that always returns Overflow would be a false repair.
**Minimal discriminating oracle:** one in-range addition and one exact overflow boundary. Also test current=u32::MAX, delta=0 as a boundary-preservation control.
**Minimal patch:** use checked_add and propagate Error::Overflow on None.
**Over-correction trap:** adding a saturating fallback is not a more correct implementation under this contract.
**Claim ceiling:** the checked-add logic satisfies the stated u32 behavior. It does not by itself prove persistence, transactionality or concurrent update safety.

## 4. PAR-S03 — audited cross-tenant administrator access

**Canonical disposition:** conforms conditionally on the audit API contract.
**Mutant disposition:** violates.
**Target witness:** actor is global admin, resource belongs to another tenant, and audit recording succeeds. The mutant allows access without producing the required audit event.
**Preservation witness:** a global administrator's audited cross-tenant read remains permitted; same-tenant owner access remains allowed; ordinary cross-tenant access remains denied.
**Minimal discriminating oracle:** test the three policy branches and assert that the cross-tenant administrator path records the correct actor/resource audit before the protected read.
**Minimal patch:** record and require successful audit before returning permission.
**Critical evidence gap:** the snippet calls audit.record_cross_tenant_read before returning true, but the implementation/contract of that method is not supplied. A source call establishes invocation order in this helper, not durable commit, read ordering or production execution. To assert durability, inspect the audit API transaction/commit semantics and the actual protected-read call chain.
**Over-correction trap:** denying all cross-tenant administrator access would violate the explicit policy exception.
**Claim ceiling:** source-level control-flow conformance conditional on the audit API's documented durability and call-order guarantees; no incident claim is established.

## 5. PAR-S04 — absent versus explicit empty configuration

**Canonical disposition:** conforms.
**Mutant disposition:** violates.
**Target witness:** supplied=None and service_default=[read]. The mutant returns an empty vector instead of inheriting [read].
**Preservation witness:** supplied=Some(empty). The canonical returns empty, preserving an explicit deny-all selection. A proposed patch that unconditionally returns defaults would re-enable permissions.
**Additional neighbor:** supplied=Some([write]) must remain exactly [write].
**Minimal discriminating oracle:** test None, Some(empty), and Some(non-empty) separately.
**Minimal patch:** use the supplied default only when the option is None; preserve Some(empty) as explicit input.
**Over-correction trap:** do not collapse missing and empty into one “no scopes” state.
**Claim ceiling:** correct Option semantics for the supplied vectors; downstream authorization enforcement and default provenance require separate evidence.

## 6. PAR-S05 — logical-operation idempotency

**Canonical disposition:** conforms to the stated key-identity contract.
**Mutant disposition:** violates.
**Target witness:** two deliveries with logical_id=L and attempt_id values 1 and 2. The mutant creates two keys; the canonical maps both deliveries to L.
**Preservation witness:** two different logical IDs L1 and L2 must remain distinct, even if their attempt IDs match.
**Minimal discriminating oracle:** assert repeated deliveries of one logical ID produce equal keys and distinct logical IDs produce unequal keys.
**Minimal patch:** use logical_id alone as the idempotency key; retain attempt_id separately for tracing.
**Over-correction trap:** do not remove logical identity or deduplicate unrelated operations merely because they share an attempt number.
**Claim ceiling:** the key function conforms to the stated identity rule. It does not prove atomic deduplication, durable storage, retention horizon or transaction correctness.

## 7. PAR-S06 — absent object versus storage error

**Canonical disposition:** conforms.
**Mutant disposition:** violates.
**Target witness:** store.get returns Err(ConnectionLost). The mutant returns NotFound and suppresses the infrastructure failure.
**Preservation witness:** store.get returns Ok(None). Both should return NotFound; changing all errors into Storage would misclassify genuinely absent records.
**Minimal discriminating oracle:** test Ok(Some(value)), Ok(None), and Err(StorageFailure) as three distinct outcomes.
**Minimal patch:** preserve the explicit match arms and propagate storage failures.
**Over-correction trap:** do not turn absence into a backend error or mask backend errors as absence.
**Claim ceiling:** correct propagation for the supplied Result<Option<T>, E> contract. Retry policy, alerting and transaction semantics remain outside this snippet.

## 8. Cross-case reviewer errors

- Requiring an inclusive endpoint despite an explicit half-open contract (PAR-S01).
- Treating saturation as a safer replacement when the contract requires overflow rejection (PAR-S02).
- Blocking every cross-tenant administrator action rather than enforcing the documented audited exception (PAR-S03).
- Treating missing and explicitly empty configuration as equivalent (PAR-S04).
- Putting delivery attempt identity into an operation-level idempotency key (PAR-S05).
- Collapsing storage failure into record absence (PAR-S06).
- Calling a patch verified because it passes only the target test and omitting the preservation oracle.
- Treating a model-generated test as independent simply because it is executable.
- Treating a source-level ordering statement as proof of durable runtime ordering or incident occurrence.

## 9. Adjudicator checklist

For each reviewer, preserve:
1. canonical and mutant verdicts;
2. exact target witness and preservation witness;
3. requirement authority and scope;
4. independent target/preservation oracle pair;
5. whether the proposed patch is minimal and within contract;
6. uncovered conditions, environment and claim ceiling;
7. disagreements about specification ambiguity or evidence authority.

Do not merge the first-pass forms or replace a reviewer’s original answer with the final adjudication.

## 10. Claim boundary

This key is a synthetic expected-answer guide. It reports no model score, no human reliability estimate, no repair verification result and no production behavior. The six cases are development-only and permanently ineligible as a sealed holdout.
