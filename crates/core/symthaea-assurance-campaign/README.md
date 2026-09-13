# symthaea-assurance-campaign

ASSURE-002 adds preregistered campaign identity and evidence-admission ordering above the qualified ASSURE-000/001 layers.

The crate deliberately does **not** use a self-declared timestamp as preregistration authority. Instead, exact plan-registration, evidence-production, and evidence-admission statements are bound to externally supplied monotonic ordering receipts carrying a source identity, epoch, sequence, statement commitment, and external receipt commitment.

Core semantics:

```text
exact ASSURE-001 subject
+ exact claim
+ exact campaign plan
+ empty pre-evidence root
+ ordered registration receipt
    -> current preregistered campaign

current registration
< ordered evidence production
< ordered evidence admission
    -> preregistered admitted evidence
```

A successor plan starts a fresh empty evidence root. Forked registration lineages, withdrawn current registrations, stale ledgers, post-hoc evidence, cross-lineage ordering, unregistered evidence kinds, duplicate evidence, and non-monotonic ordering fail closed.

ASSURE-002 validates canonical bindings and relative ordering only. It does not authenticate the external ordering provider, infer evidence support, resolve replication, certify a system, or grant deployment/action authority.

See `ASSURE_002.md` for the complete theorem and nonclaims.
