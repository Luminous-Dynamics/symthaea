# v9 Continuation Patch Series

The v9 series continues after hardened-v8 patches `0071` through `0086`.

| Patch | Purpose |
|---|---|
| 0087 | Chunked cryptographic identity for deployable artifact bytes |
| 0088 | Signed, linked deployment update plans |
| 0089 | Durable A/B trial and rollback state machine |
| 0090 | Signed generation-linked runtime safety policies |
| 0091 | Durable runtime-policy anti-rollback checkpoint |
| 0092 | Hash-chained boot epochs and startup outcomes |
| 0093 | Bounded resumable manifest-bound update receiver |
| 0094 | Opaque externally encrypted incident escrow binding |
| 0095 | Multi-node, multi-fault-domain authority continuity proof |
| 0096 | Signed health-window evidence for update confirmation |
| 0097 | v9 boot admission for update and runtime-policy evidence |
| 0098 | Explicit operator quorums for update, transition, and escrow actions |
| 0099 | Offline update, policy, boot, and escrow verifier CLIs |
| 0100 | Full v9 evidence inputs in `hal-security-admission` |
| 0101 | Deployment policies, examples, and operating documentation |

The patches are intentionally separable. A deployment can review artifact
identity and A/B state before enabling remote transfer, and can introduce signed
runtime policies independently from incident escrow.
