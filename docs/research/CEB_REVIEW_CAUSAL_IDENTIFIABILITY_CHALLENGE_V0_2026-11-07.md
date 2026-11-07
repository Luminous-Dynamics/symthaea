# CEB Review Causal-Identifiability Development Challenge V0 — 2026-11-07

**Status:** Public synthetic development challenge; source-only; not admitted, executable, sealed or scored.
**Protocol:** [Causal-Identifiability Protocol V0](CEB_REVIEW_CAUSAL_IDENTIFIABILITY_PROTOCOL_V0_2026-11-07.md).
**Key:** [Separate public answer key](CEB_REVIEW_CAUSAL_IDENTIFIABILITY_KEY_V0_2026-11-07.md); conceal it until first-pass forms are locked. The key is public, so this is not secure blinding.
**Annotation:** Use the existing [Dual-Annotator Calibration and Reliability Pack](CEB_REVIEW_DUAL_ANNOTATOR_CALIBRATION_AND_RELIABILITY_PACK_V0_2026-11-07.md).

## Reviewer instructions

For each case, state the exact claim/query; list the material causal models compatible with the supplied evidence; identify a query/intervention on which alternatives disagree; assign one disposition (identified, partially_identified, non_identifiable, contradicted, unresolved); request the minimum safe additional evidence; and state the current claim ceiling, authority class and scope.

Do not infer runtime behavior from source fragments. Do not invent probabilities for alternative models. Do not confuse a possible path with an observed path. The answer key is separate and public.

## CID-S01 — omitted authorization helper and caller context

**Family:** Missing-context ambiguity / authority-flow uncertainty.

**Claim:** Does the supplied request handler establish that every resource mutation is authorized for the requesting tenant?

**Claim kind:** Source structural possibility (and completeness of the in-scope mutation-path inventory). This is not an incident-attribution claim.

**Evidence packet:**

```rust
fn submit(req: Request, store: &Store) -> Result<()> {
    if auth::check(req.actor, req.tenant, req.resource)? {
        store.mutate(req.resource, req.payload)?;
    }
    Ok(())
}
```

The packet omits the implementation of auth::check, its backing policy store, and other callers of store.mutate. No runtime trace, policy contract or caller inventory is supplied.

Two compatible completions:

```rust
// Completion A
fn check(actor: UserId, tenant: TenantId, resource: ResourceId) -> Result<bool> {
    policy.current(actor, tenant, resource)
}
```

```rust
// Completion B
fn check(actor: UserId, _tenant: TenantId, _resource: ResourceId) -> Result<bool> {
    global_user_allowlist.contains(actor)
}
```

**Reviewer question:** What does the handler prove? Can it prove tenant/resource authorization, or only that a helper gate precedes this particular mutation call?

## CID-S02 — configuration-dependent authorization implementation

**Family:** Build/feature ambiguity / environment applicability.

**Claim:** Does the supplied workspace source prove that the shipped binary uses the fail-closed implementation?

**Claim kind:** Build-world selection. The question is which conditional implementation is selected in the artifact, not whether either source branch is executable in principle.

**Evidence packet:**

```rust
#[cfg(feature = "strict-policy")]
fn missing_policy() -> Decision {
    Decision::Deny
}

#[cfg(not(feature = "strict-policy"))]
fn missing_policy() -> Decision {
    Decision::Allow
}
```

The source tree includes both implementations. The exact Cargo invocation, resolved feature set, target triple, build profile and artifact identity are not supplied.

Two compatible build worlds: (A) strict-policy is enabled and the fail-closed definition is selected; (B) strict-policy is not enabled and the fail-open definition is selected. No default-feature behavior is stated in the packet.

**Reviewer question:** Can the reviewer state which implementation is active in the shipped binary? What exact build/artifact evidence would identify the selected definition?

## CID-S03 — workload is a common cause of cache evictions and request failures

**Family:** Common-cause confounding / observational equivalence.

**Claim:** Do cache evictions cause the request failures?

**Claim kind:** Actual-cause attribution for the observed requests, plus identifiability of an eviction intervention effect under the competing common-cause models.

**Evidence packet:**

```rust
fn handle(req: Request, cache: &Cache) -> Result<Response> {
    let entry = cache.get(req.key)?;
    match entry {
        Some(value) => execute(req, value),
        None => Err(Error::Unavailable),
    }
}
```

A synthetic monitoring summary says that, in the examined observation window, most requests with Error::Unavailable were also preceded by a cache eviction. The summary does not include tenant workload, queue pressure, cache-capacity changes or per-request interleaving. No controlled intervention or randomized rollout is supplied.

Two compatible models:
- Model A: cache eviction removes a required entry; the subsequent miss directly produces the observed error.
- Model B: a latent high-load context causes both cache eviction and an independent dependency/queue failure; eviction is correlated with the error but is not its cause.

**Reviewer question:** Does temporal precedence identify cache eviction as the cause? What additional observation or safe intervention would distinguish A from B?

## CID-S04 — overdetermined failure with two sufficient routes

**Family:** Overdetermination / alternative sufficient mechanisms.

**Claim:** Does removing the legacy retry path eliminate the observed duplicate-transfer failure?

**Claim kind:** Source-level structural sufficiency and residual-path analysis; separately, actual-cause attribution to one route in the reported incident.

**Evidence packet:**

```rust
fn process(msg: Message, cfg: Config) -> Result<()> {
    if cfg.legacy_retry && !msg.has_stable_id {
        apply_transfer(msg)?;
    }

    if cfg.replay_fallback && !ledger.has_commit(msg.logical_id) {
        apply_transfer(msg)?;
    }

    Ok(())
}
```

The frozen contract says that either call to apply_transfer can independently duplicate a transfer when its gate is active. The incident record says only that a duplicate transfer occurred. It does not identify which call executed, whether both executed, or which condition values were active.

Two compatible incident models:
- Model A: the legacy retry gate was active; replay fallback was inactive.
- Model B: the legacy retry gate was inactive; replay fallback was active.

Both paths are independently sufficient. A third possibility is that both were active.

**Reviewer question:** Can the reviewer identify one actual cause from the incident outcome alone? Does removing only one path guarantee that the duplicate cannot recur?

## CID-S05 — source permits a stale-authority race but event order is absent

**Family:** Temporal ambiguity / happens-before uncertainty.

**Claim:** Did the observed worker mutation execute after authority revocation?

**Claim kind:** Distinguish source structural possibility from execution realization and incident attribution. The supplied code may permit the ordering without showing that it occurred.

**Evidence packet:**

```rust
fn submit(req: Request, queue: &Queue) -> Result<()> {
    let allowed = authorize(req.actor, req.tenant, req.resource)?;
    queue.enqueue(Job { req, allowed })
}

fn worker(job: Job) -> Result<()> {
    if job.allowed {
        mutate(job.req)
    } else {
        Err(Error::Denied)
    }
}
```

A revocation handler can change current authorization state. The packet supplies no transaction/queue ordering contract, event timestamps, worker trace, synchronization edge or authorization epoch.

Two possible orders:
- enqueue with allowed=true → worker mutates → revoke;
- enqueue with allowed=true → revoke → worker mutates from the stale boolean.

**Reviewer question:** What can be said about the code's possible race? Can the reviewer assert that this particular incident followed the unsafe ordering?

## CID-S06 — bundled repair does not identify the necessary repair component

**Family:** Repair entanglement / causal attribution ambiguity.

**Claim:** Which change was necessary to remove the target duplicate-transfer mechanism?

**Claim kind:** Actual-cause attribution to a repair component and identifiability of that component's minimal intervention effect; the incident-level outcome is separately attributed to the bundled revision.

Before revision:

```rust
let key = (msg.logical_id, msg.attempt_id);
if !seen.contains(key) {
    apply_transfer(msg)?;
    seen.insert(key);
}
```

After revision:

```rust
let key = msg.logical_id;
if !seen.contains(key) {
    apply_transfer(msg)?;
    seen.insert_with_retention(key, REDELIVERY_HORIZON);
}
```

The same historical change also modifies queue acknowledgement behavior, the retry horizon and the deduplication-store implementation. The only supplied outcome is that the duplicate was not observed in the subsequent sample. No isolated commits, component ablations, independent replay or workload-matched control is supplied.

Two compatible explanations:
- Model A: stable logical-operation identity is necessary and sufficient under the declared retention contract.
- Model B: acknowledgement/transaction ordering or store durability removed the duplicate; the identity change is correlated but is not the decisive cause.

**Reviewer question:** Can the reviewer attribute the improvement to the key change alone? What minimal disentanglement or ablation would identify the necessary component?

## Submission worksheet

| Field | Reviewer response |
|---|---|
| Case ID | |
| Exact claim/query | |
| Observed evidence and authority | |
| Explicit assumptions and scope | |
| Candidate causal model | |
| Material compatible alternative(s) | |
| Disagreement witness (query/intervention) | |
| Predicted outcomes under each model | |
| Disposition | |
| Minimum additional evidence | |
| Safety/authority/cost note | |
| Current claim ceiling | |

## Development use and claim boundary

The six cases are synthetic and public. They are designed to expose distinct non-identifiability patterns, not estimate their prevalence in real code review. The answer key is public and is not a sealed holdout. A reviewer who requests evidence that does not actually separate the supplied models has not solved the case.

No candidate code, Cargo command, build, test, PoC, grader, hidden label or production intervention is executed. This challenge does not establish model performance. The existing three primary CEB-REVIEW measures remain unchanged.
