# symthaea-evidence-deployment-scope

Machine-checkable deployment/configuration scope for lifecycle-managed safety evidence.

A safety-contract digest identifies the reviewed subject and obligation set, but it does not by itself identify the exact software/model/calibration configuration currently deployed. This crate adds that second scope boundary.

A `DeploymentEvidenceContext` carries:

- deployment id,
- exact configuration digest,
- optional exact model-manifest digest,
- optional exact calibration-manifest digest,
- evidence references describing the current deployment context.

A lifecycle-scoped safety receipt can be bound to that exact context. During assessment, receipts from a different deployment or configuration are retained as history but are excluded from current readiness.

Configuration/model/calibration drift therefore removes old evidence from the active readiness set instead of silently reusing it.

Historical lifecycle events for receipts belonging to another configuration are not allowed to poison the current configuration's assessment; only events targeting currently matched receipts enter the downstream lifecycle evaluator.

This crate never grants physical authority.

```bash
cargo test -p symthaea-evidence-deployment-scope
```
