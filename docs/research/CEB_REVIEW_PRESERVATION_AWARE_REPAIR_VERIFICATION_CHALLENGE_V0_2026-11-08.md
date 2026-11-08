# CEB Review Preservation-Aware Repair Verification Challenge V0 — 2026-11-08

**Status:** Public synthetic development challenge; source-only; no candidate execution or hidden labels.
**Protocol:** [Preservation-Aware Repair Verification Protocol V0](CEB_REVIEW_PRESERVATION_AWARE_REPAIR_VERIFICATION_PROTOCOL_V0_2026-11-08.md).
**Key:** [Separate public development answer key](CEB_REVIEW_PRESERVATION_AWARE_REPAIR_VERIFICATION_KEY_V0_2026-11-08.md). It is not secure blinding or a sealed holdout.
**Use:** Complete independent first-pass worksheets before looking at the key.

## Reviewer instructions

For each case, assess both the canonical and mutant implementation under the same frozen requirement. Do not assume that code is defective because it could be made more defensive, more elegant or more general.

For each variant, record:
1. requirement clause and relevant source anchor;
2. verdict: conforms / violates / under-specified / unresolved;
3. target witness for any violation;
4. nearest preservation witness (valid behavior that must not be broken);
5. the minimum test/property pair that discriminates the canonical and mutant versions;
6. the smallest justified patch, or state that no patch is warranted;
7. residual oracle gap and claim ceiling.

The examples are deliberately small and synthetic. All snippets are illustrations; none is executable benchmark input in this public development pack.

## PAR-S01 — inclusive end boundary

**Normative contract:** A requested interval is half-open: it includes start and excludes end. For any integer x, the predicate is true exactly when start <= x < end.

**Canonical implementation:**

~~~rust
fn contains(start: i64, end: i64, x: i64) -> bool {
    x >= start && x < end
}
~~~

**Mutant implementation:**

~~~rust
fn contains(start: i64, end: i64, x: i64) -> bool {
    x >= start && x <= end
}
~~~

**Preservation boundary:** x == start must be included; x == end must be excluded; values strictly inside must be included; values outside must be excluded.

**Reviewer question:** Is the canonical implementation defective because it excludes the endpoint? Does the mutant violate the explicit half-open contract? What target and preservation inputs distinguish them?

## PAR-S02 — overflow must be rejected, not saturated

**Normative contract:** If adding delta to current exceeds the representable u32 range, the operation returns Overflow and does not silently clamp the value. Otherwise it returns the exact sum.

**Canonical implementation:**

~~~rust
fn add_balance(current: u32, delta: u32) -> Result<u32, Error> {
    current.checked_add(delta).ok_or(Error::Overflow)
}
~~~

**Mutant implementation:**

~~~rust
fn add_balance(current: u32, delta: u32) -> Result<u32, Error> {
    Ok(current.saturating_add(delta))
}
~~~

**Preservation boundary:** A valid non-overflowing addition must return the exact sum. The error path must remain explicit. Do not reject the canonical implementation for not providing a saturating fallback when the contract forbids saturation.

**Reviewer question:** What is the target witness? Which ordinary in-range input is the preservation witness? Why is “more forgiving” not necessarily more correct?

## PAR-S03 — cross-tenant administrator access requires an audit record

**Normative contract:** A tenant-scoped owner may access resources in their tenant. A global administrator may access another tenant only when an audit record is durably accepted before the protected read. All other cross-tenant access is denied.

**Canonical implementation:**

~~~rust
fn may_read(actor: &Actor, resource: &Resource, audit: &Audit) -> Result<bool> {
    if actor.tenant_id == resource.tenant_id {
        return Ok(actor.is_owner);
    }
    if actor.is_global_admin {
        audit.record_cross_tenant_read(actor.id, resource.id)?;
        return Ok(true);
    }
    Ok(false)
}
~~~

**Mutant implementation:**

~~~rust
fn may_read(actor: &Actor, resource: &Resource, audit: &Audit) -> Result<bool> {
    if actor.tenant_id == resource.tenant_id {
        return Ok(actor.is_owner);
    }
    if actor.is_global_admin {
        return Ok(true);
    }
    Ok(false)
}
~~~

**Preservation boundary:** Same-tenant owner access remains allowed. A global administrator is not categorically forbidden from cross-tenant access; the mandatory audit ordering is the condition. Ordinary cross-tenant users remain denied.

**Reviewer question:** How do you avoid both errors: falsely flagging the permitted audited administrator path, and missing the mutant's audit omission? Is a source-level call enough to establish durable audit-before-read ordering?

## PAR-S04 — omitted configuration and explicit empty configuration differ

**Normative contract:** None means inherit the service default scopes, which are [read]. Some(empty) means explicitly grant no scopes. Some(non-empty) means use exactly the supplied scopes.

**Canonical implementation:**

~~~rust
fn effective_scopes(
    supplied: Option<Vec<Scope>>,
    service_default: Vec<Scope>,
) -> Vec<Scope> {
    supplied.unwrap_or(service_default)
}
~~~

**Mutant implementation:**

~~~rust
fn effective_scopes(
    supplied: Option<Vec<Scope>>,
    _service_default: Vec<Scope>,
) -> Vec<Scope> {
    supplied.unwrap_or_default()
}
~~~

**Preservation boundary:** An omitted value inherits [read]. An explicit empty list stays empty. An explicit [write] list stays [write]. Do not normalize absence and emptiness into the same state.

**Reviewer question:** Which input exposes the mutant's defect? Which neighboring input ensures a proposed repair does not accidentally re-enable permissions for an explicit empty list?

## PAR-S05 — idempotency is per logical operation, not per delivery attempt

**Normative contract:** A logical transfer may be delivered more than once. It must apply at most once per logical operation ID. Attempt IDs identify deliveries for tracing but do not create a new business operation.

**Canonical implementation:**

~~~rust
fn dedupe_key(msg: &Message) -> LogicalOperationId {
    msg.logical_id
}
~~~

**Mutant implementation:**

~~~rust
fn dedupe_key(msg: &Message) -> (LogicalOperationId, AttemptId) {
    (msg.logical_id, msg.attempt_id)
}
~~~

**Preservation boundary:** Repeated delivery attempts with the same logical ID must map to the same idempotency key. Two distinct logical IDs must remain distinct even if they share an attempt number.

**Reviewer question:** What is the minimal counterexample? What nearby valid case demonstrates that removing attempt ID from the key does not merge distinct logical operations?

## PAR-S06 — missing record and storage failure are distinct outcomes

**Normative contract:** If the object is absent, return NotFound. If the storage backend fails, propagate the storage failure; never disguise an infrastructure error as absence.

**Canonical implementation:**

~~~rust
fn load_object(id: Id, store: &Store) -> Result<Object, Error> {
    match store.get(id) {
        Ok(Some(value)) => Ok(value),
        Ok(None) => Err(Error::NotFound),
        Err(error) => Err(Error::Storage(error)),
    }
}
~~~

**Mutant implementation:**

~~~rust
fn load_object(id: Id, store: &Store) -> Result<Object, Error> {
    match store.get(id) {
        Ok(Some(value)) => Ok(value),
        Ok(None) | Err(_) => Err(Error::NotFound),
    }
}
~~~

**Preservation boundary:** A genuinely absent record remains NotFound. A backend failure remains distinguishable and available for retry/alerting. Do not “fix” the canonical code by converting all errors into successful empty values or by treating every failure as an outage.

**Reviewer question:** Which two distinct outcomes are collapsed by the mutant? What test pair demonstrates the target defect while preserving correct not-found semantics?

## Submission worksheet

| Field | Reviewer response |
|---|---|
| Case ID | |
| Canonical verdict | conforms / violates / under-specified / unresolved |
| Mutant verdict | conforms / violates / under-specified / unresolved |
| Normative clause / authority | |
| Target witness | |
| Preservation witness | |
| Minimal target oracle | |
| Minimal preservation oracle | |
| Proposed patch / no-patch rationale | |
| Other invariant at risk | |
| Oracle independence and coverage gap | |
| Claim kind and claim ceiling | |
| Adjudication disagreement | |

## Development boundary

Six paired cases are six development anchors, not a sample for reliability estimates. The challenge and answer key are public and permanently ineligible as a sealed CEB holdout. No candidate code, compiler, test, PoC, benchmark runner, hidden grader or production system is executed. The existing three primary CEB-REVIEW measures remain unchanged.
