# symthaea-evidence-lifecycle

Time- and scope-aware lifecycle assurance for verified `symthaea-formal-safety` evidence.

A verified receipt is not automatically valid forever or everywhere. This crate binds a receipt to:

- the exact `SafetyCase::contract_digest()`,
- an explicit applicability interval,
- durable deployment/configuration applicability references,
- auditable lifecycle events.

Supported lifecycle events are:

- **revocation** — permanently withdraw a receipt,
- **supersession** — replace an old receipt with another already-verified applicable receipt,
- **contradiction** — record materially conflicting evidence,
- **contradiction resolution** — record reviewed disposition of the conflict.

An unresolved contradiction blocks readiness even when another favorable receipt exists. Resolving a contradiction does not reactivate the contradicted receipt; another active verified receipt is still required.

Expired, revoked, and superseded historical receipts do not by themselves block readiness when another currently applicable receipt satisfies the same obligation. Structural lifecycle inconsistencies fail closed as `Invalid`.

The lifecycle report is evidence/readiness metadata only and never grants physical authority.

```bash
cargo test -p symthaea-evidence-lifecycle
```
