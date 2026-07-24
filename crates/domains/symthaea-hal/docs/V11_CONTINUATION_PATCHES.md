# Symthaea HAL v11 continuation patches

The v11 series continues from hardened v10 and contains patches 0121-0138.

| Patch | Purpose |
|---|---|
| 0121 | Bound fleet operation during network partitions. |
| 0122 | Persist anti-rollback partition checkpoints. |
| 0123 | Sign narrowly scoped partition continuation permits. |
| 0124 | Add signed irreversible device decommissioning. |
| 0125 | Bind replacement devices to retired identities. |
| 0126 | Require dual-party custody handoff acceptance. |
| 0127 | Sign safety-relevant component identities. |
| 0128 | Verify signed transitive device bills of materials. |
| 0129 | Link component swaps to consecutive signed BOMs. |
| 0130 | Classify and bound long-horizon safety evidence. |
| 0131 | Gate deletion behind legal holds and signed receipts. |
| 0132 | Hash-link evidence segments across fault domains. |
| 0133 | Require periodic signed media-scrub evidence. |
| 0134 | Assemble cross-device lifecycle assurance bundles. |
| 0135 | Require lifecycle provenance and retention closure at admission. |
| 0136 | Reconcile offline execution before fleet rejoin. |
| 0137 | Add the offline fleet lifecycle verifier. |
| 0138 | Publish deployment policies and lifecycle operating guidance. |

The series is intentionally additive. Existing v10 formats and fingerprints
remain unchanged; v11 artifacts use new domain-separated `sha256:` identities.
