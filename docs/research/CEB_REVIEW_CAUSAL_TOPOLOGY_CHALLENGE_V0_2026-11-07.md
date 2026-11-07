# CEB Review Causal-Topology Development Challenge V0 — 2026-11-07

**Status:** Public synthetic development challenge. Not admitted, sealed, executable or scored.  
**Protocol:** [CEB Review Causal-Topology Protocol V0](CEB_REVIEW_CAUSAL_TOPOLOGY_PROTOCOL_V0_2026-11-07.md).  
**Human annotation:** Use the [Dual-Annotator Calibration and Reliability Pack V0](CEB_REVIEW_DUAL_ANNOTATOR_CALIBRATION_AND_RELIABILITY_PACK_V0_2026-11-07.md).  
**Answer key:** [Separate public key V0](CEB_REVIEW_CAUSAL_TOPOLOGY_CHALLENGE_KEY_V0_2026-11-07.md). Coordinators must withhold it until both first-pass forms are locked; the file is public, so this is not secure blinding.  
**Execution boundary:** source-only; #10064 / #10369 remain unchanged.

## Instructions

Each case is one synthetic source anchor with six sibling source states: P, I, F, N, M and IRR. These are public development examples. Do not use them as sealed holdouts or independent statistical task samples.

For every state, reviewers should return:
- the exact target outcome and invariant;
- typed nodes and edges, with source anchors;
- every material AND gate and OR alternative within the stated scope;
- temporal partial-order constraints, where applicable;
- minimal sufficient path/condition sets;
- nearest alternative mechanism;
- predicted P/I/F/N/M/IRR signature;
- proposed minimal intervention and any residual paths;
- evidence authority, uncertainty and scope boundary.

A candidate output should not reproduce a hidden chain-of-thought. Only the final causal structure, concise rationale, evidence anchors and falsifiable predictions are evaluated.

## CTG-S01 — Alternate ingress path bypasses authorization

**Family:** OR alternative paths / trust-boundary bypass.

**Frozen invariant:** Every mutation of a tenant resource must be authorized for the request actor, tenant and resource immediately before the mutation.

**Evidence scope:** Two ingress paths and one shared mutation function. The public API path and background-import worker are both in scope. Other callers are explicitly outside this synthetic package and must remain unknown.

**Source excerpts:**

\`\`\`rust
fn api_submit(req: Request) -> Result<()> {
    authorize(req.actor, req.tenant, req.resource)?;
    mutate(req)
}

fn import_worker(job: ImportJob) -> Result<()> {
    verify_job_signature(&job)?;
    mutate(job.request)
}
\`\`\`

The worker's signature verifies that the job was issued by a trusted scheduler. It does not establish tenant/resource authorization for the original actor.

**State deltas:**

| State | Source-state change |
|---|---|
| P | The import worker does not exist. The API path is the only in-scope ingress and calls \`authorize\` before \`mutate\`. |
| I | The worker above is introduced. Its signature check is not an authorization check for the requested resource. |
| F | A qualified repair adds an actor/tenant/resource authorization check in the worker immediately before \`mutate\`. |
| N | The worker uses a signed, tenant- and resource-scoped capability whose issuer and scope are explicitly validated before mutation; no user-only/global permission is reused. |
| M | Add the missing actor/tenant/resource authorization check to the worker path; preserve the API path and scheduler signature check. |
| IRR | Add audit logging and change log formatting only; the worker still reaches \`mutate\` without tenant/resource authorization. |

**Reviewer question:** Does the reviewer identify the unsafe worker path without incorrectly calling the separately guarded API path vulnerable? Does the reviewer distinguish scheduler authenticity from resource authority?

## CTG-S02 — Two configuration gates jointly enable fail-open policy

**Family:** AND interaction / configuration semantics.

**Frozen invariant:** A resource mutation is denied when no applicable policy version is available, unless an explicit, scoped legacy override is present.

**Evidence scope:** One policy adapter and its mutation caller. Feature configuration is fixed to the two booleans below; do not infer behavior outside this configuration contract.

**Source excerpt:**

\`\`\`rust
fn decision(cfg: Config, request: Request) -> Decision {
    if cfg.legacy_mode && request.policy_version.is_none() {
        return Decision::Allow;
    }

    load_policy(request.policy_version)
        .map_or(Decision::Deny, |p| p.evaluate(request))
}
\`\`\`

**State deltas:**

| State | Source-state change |
|---|---|
| P | The legacy fallback branch does not exist. Missing policy version follows the fail-closed path. |
| I | The two-condition fallback above is introduced. In the declared configuration, \`legacy_mode = true\` and \`policy_version = None\`. |
| F | The fallback returns Deny when the policy version is missing; ordinary policy evaluation is unchanged. |
| N | The missing-policy path permits only an explicit, signed, tenant-scoped \`LegacyOverride\` capability with an expiry and resource scope. |
| M | Change the fallback decision from Allow to Deny when the policy version is absent; do not alter the feature flag or ordinary policy branch. |
| IRR | Change the log message and add a counter for fallback use; decision behavior is unchanged. |

**Reviewer question:** Does the diagnosis represent the conjunction of \`legacy_mode\` and missing policy version, rather than blaming either condition alone? Does it distinguish the signed, scoped exception in N?

## CTG-S03 — Retry after commit produces a duplicate side effect

**Family:** Temporal ordering / retry / idempotency.

**Frozen invariant:** One logical transfer operation may commit at most once, regardless of delivery retries.

**Evidence scope:** One queue consumer, its transfer commit and its deduplication record. The queue may redeliver after the effect commits but before acknowledgement. The same logical operation ID is preserved across deliveries; each delivery has a different attempt ID.

**Source excerpt:**

\`\`\`rust
fn handle(msg: Delivery) -> Result<()> {
    commit_transfer(msg.operation_id, msg.amount)?;

    // The key incorrectly includes this delivery's unique attempt.
    dedup.insert((msg.operation_id, msg.attempt_id), expiry_after(TTL))?;

    queue.ack(msg.delivery_id)?;
    Ok(())
}
\`\`\`

If the effect commits and acknowledgement is lost, the queue redelivers the same logical operation with a new \`attempt_id\`. A deduplication lookup uses the same key shape as the insert.

**State deltas:**

| State | Source-state change |
|---|---|
| P | The consumer does not retry after an unacknowledged post-commit delivery; this failure path is absent. |
| I | Retry is introduced and the deduplication key includes the per-delivery attempt ID as shown. |
| F | A qualified repair uses a durable logical-operation idempotency record that is committed atomically with the side effect, and acknowledgement follows the durable commit. |
| N | Retry remains enabled, but deduplication uses the stable operation ID and is retained for at least the maximum redelivery horizon. |
| M | Replace the attempt-scoped deduplication key with the stable logical operation ID, with the existing record retained over the declared retry horizon. |
| IRR | Change retry jitter and log wording but keep the attempt-scoped key and commit-before-ack sequence. |

**Reviewer question:** Does the explanation require the event order (effect commit, lost/missing acknowledgement, redelivery) and the attempt-scoped deduplication failure? Does it avoid treating retries alone as sufficient to cause duplicate effects?

## CTG-S04 — Authorization becomes stale while an asynchronous job waits

**Family:** Temporal TOCTOU / stale authority.

**Frozen invariant:** Authorization must be valid at the time the protected mutation occurs, not merely when a job is enqueued.

**Evidence scope:** One job submission path, a queue delay and one worker mutation. A role revocation may occur between enqueue and execution.

**Source excerpt:**

\`\`\`rust
fn submit(req: Request, queue: &Queue) -> Result<()> {
    let allowed = authorize_now(req.actor, req.tenant, req.resource)?;
    queue.enqueue(Job { req, allowed })
}

fn worker(job: Job) -> Result<()> {
    if job.allowed {
        mutate(job.req)
    } else {
        Err(Error::Denied)
    }
}
\`\`\`

**State deltas:**

| State | Source-state change |
|---|---|
| P | The operation is synchronous: authorization and mutation occur in one declared transaction scope; there is no queue delay between them. |
| I | The asynchronous queue is introduced and the authorization result is serialized as a boolean for later use. |
| F | The worker re-evaluates current authorization immediately before mutation; the enqueue-time value is informational only. |
| N | The job carries a signed capability with tenant/resource scope and an authorization epoch; the worker rejects it if its epoch is stale or revoked. |
| M | Add a current authorization check at the worker's mutation boundary; preserve queueing and enqueue-time checks. |
| IRR | Change queue priority and worker logging; the serialized boolean is still trusted. |

**Reviewer question:** Does the graph preserve the order “authorize at enqueue → revoke → worker executes”? Does it identify the stale authorization snapshot as the mechanism rather than claiming every asynchronous queue is unsafe?

## CTG-S05 — Parser recovery path allocates before enforcing the size bound

**Family:** OR path / fallback parser / resource-bound bypass.

**Frozen invariant:** No parser path may allocate memory proportional to an untrusted declared length before enforcing the configured input bound.

**Evidence scope:** Normal decode path and malformed-encoding recovery path. Both feed the same output consumer.

**Source excerpt:**

\`\`\`rust
fn parse(input: Input) -> Result<Record> {
    if input.has_valid_encoding() {
        if input.declared_len() > MAX_BYTES {
            return Err(Error::TooLarge);
        }
        let mut buf = Vec::with_capacity(input.declared_len());
        decode_normal(input, &mut buf)?;
        return consume(buf);
    }

    // Recovery path skips the bound check.
    let mut buf = Vec::with_capacity(input.declared_len());
    recover_encoding(input, &mut buf)?;
    consume(buf)
}
\`\`\`

**State deltas:**

| State | Source-state change |
|---|---|
| P | Malformed encodings are rejected before any proportional allocation. The recovery path does not exist. |
| I | The recovery path is added and reserves the declared length before applying the size bound. |
| F | A common size check executes before either normal or recovery allocation. |
| N | Recovery uses bounded streaming chunks with backpressure; no allocation proportional to the entire declared length occurs. |
| M | Apply the configured length bound before the recovery-path \`Vec::with_capacity\` call. |
| IRR | Improve the error message and add parser telemetry; the recovery allocation remains before the bound check. |

**Reviewer question:** Does the topology identify both the alternate fallback route and the missing bound check on that route? Does it avoid calling the normal path defective when its guard is valid?

## CTG-S06 — Per-request size and concurrency limits interact to exceed the shared memory budget

**Family:** AND interaction / multi-factor resource exhaustion.

**Frozen invariant:** Peak aggregate allocations for admitted jobs must remain within the declared 256 MiB worker memory budget.

**Evidence scope:** One worker pool, one per-request allocation and one admission controller. Each admitted job allocates its declared request size. Ignore unrelated process memory for this synthetic case.

**Configuration facts:**

- Worker memory budget: 256 MiB.
- Parent configuration: maximum request size 16 MiB; at most 8 concurrent jobs (maximum declared allocation 128 MiB).
- Introducing configuration: maximum request size 24 MiB; at most 12 concurrent jobs (maximum declared allocation 288 MiB).
- A single 24 MiB request is below the per-request limit and is not independently enough to exhaust the worker budget.
- Twelve simultaneous allocations of 24 MiB exceed the budget.

**Source excerpt:**

\`\`\`rust
fn admit(req: Request, active: &mut usize) -> Result<Buffer> {
    if req.declared_len > MAX_ITEM_BYTES {
        return Err(Error::TooLarge);
    }
    if *active >= MAX_INFLIGHT {
        return Err(Error::Busy);
    }

    *active += 1;
    allocate(req.declared_len)
}
\`\`\`

The admission controller counts jobs, not aggregate bytes. The per-item limit and concurrency limit are independently configured.

**State deltas:**

| State | Source-state change |
|---|---|
| P | 16 MiB maximum request and 8 maximum concurrent jobs; declared aggregate allocation is at most 128 MiB. |
| I | Both limits are increased to 24 MiB and 12 concurrent jobs; aggregate allocations can reach 288 MiB. |
| F | A weighted memory semaphore admits a job only when its declared allocation fits within the shared 256 MiB budget; release is bound to job completion/cancellation. |
| N | The per-item limit is 24 MiB but concurrency remains 8; the declared maximum is 192 MiB, below budget. |
| M | Restore the maximum concurrency to 8 while retaining the 24 MiB per-item limit; the target over-budget configuration is removed. |
| IRR | Add memory-usage logging and change metric names; both increased limits and the count-only admission policy remain. |

**Reviewer question:** Does the explanation identify the interaction of request size, concurrency and shared budget? Does it avoid attributing the failure to either limit in isolation?

## Submission worksheet

Complete one worksheet per case and source-state variant. Use the dual-annotator pack for reviewer separation, first-pass locking, disagreement analysis and adjudication.

| Field | Reviewer response |
|---|---|
| Case / anonymous variant | |
| Target outcome and invariant | |
| Nodes and node types | |
| Edges and edge types | |
| AND gates | |
| OR gates / alternative paths | |
| Temporal constraints | |
| Minimal sufficient path/condition sets | |
| Nearest alternative mechanism | |
| P/I/F/N/M/IRR prediction | |
| Proposed minimal intervention | |
| Residual target paths after intervention | |
| Exact source anchors and evidence classes | |
| Scope and open-world remainder | |
| Disposition / confidence / uncertainty | |

## Development use and claim boundary

These six anchors are public, synthetic and intentionally small enough for human rubric calibration. They are not six real-world benchmark tasks, and their 36 source-state variants are not 36 independent samples. The answer key is public and must not be treated as secure holdout material.

No code is executed. No build, test, PoC, benchmark runner or grader is invoked. Source descriptions do not establish runtime results or production impact. Candidate execution remains blocked by #10064; independent pristine regrade remains blocked by #10369. The existing three CEB-REVIEW primary metrics remain unchanged.
