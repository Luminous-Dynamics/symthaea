# symthaea-replicator-safety

> **Status: REFERENCE SAFETY SEMANTICS — NOT PRODUCTION-ADMITTED**

This crate defines the deterministic, deny-by-default constitutional authority semantics for the Symthaea Replicator Safety Kernel (RSK).

It contains **no physical replication mechanism, molecular design, biological implementation, fabrication recipe, or autonomous manufacturing path**.

## What this crate establishes

The foundational invariant is:

> **Physical creation does not confer authority.**

The reference evaluator models whether a subject may receive bounded replication authority under an explicit grant and current safety conditions. It checks, among other things:

- exact subject and lineage binding;
- inherited capability ceilings;
- risk-class restrictions;
- direct-child, descendant, depth, and resource ceilings;
- grant validity interval and generation;
- safety-case and containment-envelope digest binding;
- monitoring health/freshness;
- quarantine and revocation precedence;
- independent-approval count;
- fail-closed behavior for missing, stale, or contradictory authority facts.

The evaluator is deliberately deterministic and side-effect free.

## What this crate does **not** prove yet

The current public structs are reference inputs. They model facts that a future trusted adapter must verify; their mere construction is not proof that those facts are authentic.

In particular, this crate does **not yet cryptographically verify**:

- that a `ReplicationGrant` was signed by an authorized external issuer;
- that `GrantIssuerClass::ExternalIndependent` actually represents an independent trust domain;
- that a quorum count corresponds to distinct verified signers/failure domains;
- that monitoring/containment observations came from an authenticated independent runtime monitor;
- that supplied wall-clock time has monotonic continuity;
- that an evidence digest corresponds to a verified durable safety-case snapshot;
- that a caller cannot fabricate reference-only evidence structs in its own process.

For that reason, these types must not be treated as a complete production trust boundary merely because `evaluate_replication_authority` returns `Allow`.

## Production-admission direction

Before any higher-level autonomous fabrication system may rely on RSK for production replication authority, the surrounding system must provide at least:

1. **Verified grants** — a `VerifiedReplicationGrant`-style type constructible only after signature, trust-root, role, generation, subject, lineage, and validity verification.
2. **Verified quorum evidence** — distinct signer identities and relevant organizational/key/failure-domain evidence, not a caller-supplied count alone.
3. **Verified runtime safety evidence** — authenticated monitor/containment evidence whose provenance is independent of the autonomy requesting authority.
4. **Trusted time continuity** — monotonic/epoch-bound time evidence so clock rollback cannot revive expired authority.
5. **Exact intent/action binding** — the authorization token must commit to every authority-relevant value consumed at commit time, including the requested resource amount and operation identity.
6. **Durable replay-verifiable lineage state** — canonical event encoding, predecessor hashes, replay-from-genesis, transactional compare-and-append, checkpoints, and fork containment.
7. **Recovery separation** — quarantine/revocation/fork recovery requires an externally governed fresh-epoch ceremony; a controlled subject cannot self-recover authority.
8. **Adversarial/property verification** — state-machine, crash, replay, fork, stale-evidence, and common-mode-failure tests must pass under the pinned toolchain.

See the RSK architecture documents under `docs/architecture/replicator-safety/` for the normative design and promotion gates.

## Intended use today

Appropriate uses:

- executable policy/reference semantics;
- unit, integration, property, and formal-model comparison;
- simulation and safety-case development;
- designing verified adapters around the constitutional core.

Not appropriate today:

- treating raw caller-created grants or monitor structs as authenticated production evidence;
- directly gating hazardous physical actuation;
- bypassing the ledger/durable-evidence/recovery layers;
- assuming the crate is certified, formally verified, or production admitted.

The crate is intentionally marked `publish = false` while these admission gates remain open.
