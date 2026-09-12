# symthaea-evidence-deep-readiness

Composite readiness for safety evidence after lifecycle, deployment scope, quarantine, trusted time, verifier diversity, and atomic facet coverage.

The core invariant is that deeper assurance gates operate only on receipts that are actually active for the exact deployment at the assessed trusted time. Historical receipts that are expired, contradicted, revoked, superseded, quarantined, or bound to another configuration cannot be counted toward verifier diversity or atomic coverage.

Readiness must hold at both ends of the trusted clock-uncertainty interval:

1. lifecycle / deployment / quarantine / strict readiness
2. verifier common-cause diversity
3. atomic evidence-facet coverage

A receipt that expires inside the interval may leave ordinary strict readiness intact through another receipt, yet still cause verifier-diversity or atomic-coverage requirements to fail. This crate detects that case.

The composition is monotonic and grants no physical authority.

```bash
cargo test -p symthaea-evidence-deep-readiness
```
