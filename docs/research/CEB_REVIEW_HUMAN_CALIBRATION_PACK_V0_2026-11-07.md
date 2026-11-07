# CEB Review Human Calibration Pack V0 — 2026-11-07

**Status:** Public synthetic development/calibration material. Not an admitted task set, hidden label set, sealed holdout, or performance benchmark.  
**Purpose:** Let two independent reviewers practice the existing CEB finding-adjudication card on source-only counterfactual cases before any candidate evaluation.  
**Related:** [Causal-diagnosis preregistration](CEB_REVIEW_CAUSAL_DIAGNOSIS_PREREGISTRATION_V0_2026-11-07.md), [26-scenario challenge](CEB_REVIEW_EVIDENCE_GROUNDING_CHALLENGE_V0_2026-11-06.md), [finding adjudication card](CEB_REVIEW_FINDING_ADJUDICATION_CARD_V0_2026-11-07.md).  
**Execution:** None. Snippets are inert source excerpts, not complete crates. Do not compile, run, benchmark or turn them into admitted task fixtures.

## 1. Instructions for independent reviewers

This is a *calibration exercise*, not a blind holdout. The answer key is a separate public development file. For an independent dry run, reviewers should receive only this pack and the blank worksheet; the maintainer must retain the key until both worksheets are submitted. Because both documents are public, this procedure does not establish sealed/blind benchmark security.

Review each case independently. Do not discuss the case with the other reviewer until both worksheets are complete. Use only the specification and source variants below; no external assumptions, execution or tool-derived runtime claims are allowed.

For each neutral variant ID (X1–X6):

1. Decide whether the **specific target mechanism** in the proposed finding is live: `yes / no / unresolved`.
2. Give the causal reason: trigger/precondition → relevant path/state → operation → violated condition → consequence.
3. Identify evidence authority. These excerpts support source-level findings only; do not claim compiler or runtime observation.
4. State whether a nearby but different issue is present. Do not confuse a valid non-target issue with a target hit.
5. Give confidence and the smallest additional evidence that would resolve any uncertainty.

Then classify the candidate finding across the six variants:
- target mechanism present;
- target mechanism absent;
- target replaced by a neighboring mechanism;
- evidence insufficient.

Use the [CEB Review Finding Adjudication Card](CEB_REVIEW_FINDING_ADJUDICATION_CARD_V0_2026-11-07.md). Record disagreements verbatim; do not resolve them by majority vote alone.

## 2. Case CAL-01 — authorization before persistent mutation

### Normative contract

A caller without write permission must not cause a persistent document mutation. A denied request must leave the stored document unchanged. The authorization decision is scoped to the target document and caller.

### Candidate finding to assess

> The write operation is performed before the write-permission check. A caller who is later denied can still cause a persistent mutation.

The question is whether this causal mechanism is present in each variant, not whether the code style is ideal.

### Source variants

#### Variant X1

```rust
fn publish(actor: &Actor, doc: &Document, db: &mut Db) -> Result<(), Denied> {
    if !actor.can_read(doc.tenant_id, doc.id) {
        return Err(Denied);
    }
    db.put(doc.tenant_id, doc.id, doc.body.clone());
    Ok(())
}
```

#### Variant X2

```rust
fn publish(actor: &Actor, doc: &Document, db: &mut Db) -> Result<(), Denied> {
    if !actor.can_write(doc.tenant_id, doc.id) {
        return Err(Denied);
    }
    db.put(doc.tenant_id, doc.id, doc.body.clone());
    Ok(())
}
```

#### Variant X3

```rust
fn publish(actor: &Actor, doc: &Document, db: &mut Db) -> Result<(), Denied> {
    db.put(doc.tenant_id, doc.id, doc.body.clone());
    if !actor.can_write(doc.tenant_id, doc.id) {
        return Err(Denied);
    }
    Ok(())
}
```

#### Variant X4

```rust
fn publish(actor: &Actor, doc: &Document, db: &mut Db) -> Result<(), Denied> {
    let pending = doc.body.clone();
    if actor.can_write(doc.tenant_id, doc.id) {
        db.put(doc.tenant_id, doc.id, pending);
        Ok(())
    } else {
        Err(Denied)
    }
}
```

#### Variant X5

```rust
fn publish(actor: &Actor, doc: &Document, db: &mut Db) -> Result<(), Denied> {
    db.put(doc.tenant_id, doc.id, doc.body.clone());
    if !actor.can_write(doc.tenant_id, doc.id) {
        audit::record_denied_publish(actor.id, doc.id, "permission denied");
        return Err(Denied);
    }
    Ok(())
}
```

#### Variant X6

```rust
fn publish(actor: &Actor, doc: &Document, db: &mut Db) -> Result<(), Denied> {
    if !actor.can_write(doc.tenant_id, doc.id) {
        return Err(Denied);
    }
    let body = doc.body.clone();
    db.put(doc.tenant_id, doc.id, body);
    Ok(())
}
```

### Reviewer worksheet

| Variant | Target live? yes/no/unresolved | Causal explanation and source anchors | Non-target issue, if any | Confidence |
|---|---|---|---|---|
| X1 | | | | |
| X2 | | | | |
| X3 | | | | |
| X4 | | | | |
| X5 | | | | |
| X6 | | | | |

## 3. Case CAL-02 — enforce untrusted length before proportional allocation

### Normative contract

The packet's declared length is untrusted input. If it exceeds `MAX_PACKET_BYTES`, the decoder must return an error **before any allocation proportional to that declared length**. The packet may be empty; an empty payload is valid under this contract. Checksum validity is a separate property.

### Candidate finding to assess

> The decoder allocates a buffer using the untrusted declared length before enforcing the maximum-size bound. An attacker can cause a large allocation before the function rejects the packet.

Do not confuse empty-packet handling or checksum validation with the target size-check ordering.

### Source variants

#### Variant X1

```rust
fn decode_packet(input: &[u8]) -> Result<Packet, DecodeError> {
    let declared = read_u32_be(input)?;
    let mut payload = vec![0_u8; declared as usize];
    if declared > MAX_PACKET_BYTES {
        return Err(DecodeError::TooLarge);
    }
    fill_payload(input, &mut payload)?;
    verify_checksum(input, &payload)?;
    Ok(Packet { payload })
}
```

#### Variant X2

```rust
fn decode_packet(input: &[u8]) -> Result<Packet, DecodeError> {
    let declared = read_u32_be(input)?;
    if declared > MAX_PACKET_BYTES {
        return Err(DecodeError::TooLarge);
    }
    let mut payload = vec![0_u8; declared as usize];
    fill_payload(input, &mut payload)?;
    verify_checksum(input, &payload)?;
    Ok(Packet { payload })
}
```

#### Variant X3

```rust
fn decode_packet(input: &[u8]) -> Result<Packet, DecodeError> {
    let declared = read_u32_be(input)?;
    if declared == 0 {
        return Err(DecodeError::EmptyPacket);
    }
    if declared > MAX_PACKET_BYTES {
        return Err(DecodeError::TooLarge);
    }
    let mut payload = vec![0_u8; declared as usize];
    fill_payload(input, &mut payload)?;
    verify_checksum(input, &payload)?;
    Ok(Packet { payload })
}
```

#### Variant X4

```rust
fn decode_packet(input: &[u8]) -> Result<Packet, DecodeError> {
    let declared = read_u32_be(input)?;
    let mut payload = vec![0_u8; declared as usize];
    fill_payload(input, &mut payload)?;
    if !verify_checksum(input, &payload)? {
        return Err(DecodeError::BadChecksum);
    }
    if declared > MAX_PACKET_BYTES {
        return Err(DecodeError::TooLarge);
    }
    Ok(Packet { payload })
}
```

#### Variant X5

```rust
fn decode_packet(input: &[u8]) -> Result<Packet, DecodeError> {
    let declared = read_u32_be(input)?;
    let size = usize::try_from(declared).map_err(|_| DecodeError::TooLarge)?;
    if size > MAX_PACKET_BYTES as usize {
        return Err(DecodeError::TooLarge);
    }
    let mut payload = Vec::with_capacity(size);
    payload.resize(size, 0_u8);
    fill_payload(input, &mut payload)?;
    verify_checksum(input, &payload)?;
    Ok(Packet { payload })
}
```

#### Variant X6

```rust
fn decode_packet(input: &[u8]) -> Result<Packet, DecodeError> {
    let declared = read_u32_be(input)?;
    let size = usize::try_from(declared).map_err(|_| DecodeError::TooLarge)?;
    if size > MAX_PACKET_BYTES as usize {
        return Err(DecodeError::TooLarge);
    }
    let mut payload = vec![0_u8; size];
    fill_payload(input, &mut payload)?;
    verify_checksum(input, &payload)?;
    Ok(Packet { payload })
}
```

### Reviewer worksheet

| Variant | Target live? yes/no/unresolved | Causal explanation and source anchors | Non-target issue, if any | Confidence |
|---|---|---|---|---|
| X1 | | | | |
| X2 | | | | |
| X3 | | | | |
| X4 | | | | |
| X5 | | | | |
| X6 | | | | |

## 4. Case CAL-03 — tenant-scoped authorization cache identity

### Normative contract

Authorization decisions are scoped to the tuple `(tenant_id, user_id, resource_id)`. The same user identifier can exist in different tenants, and policy may differ by tenant. A cache hit for one tenant must never authorize a request in another tenant.

### Candidate finding to assess

> The global authorization cache omits tenant identity from its key. A decision cached for one tenant can be reused for the same user/resource identifiers in another tenant.

The causal path crosses the cache definition and its caller. A reviewer must inspect both; a local key declaration alone is insufficient if the backing map is separately partitioned by tenant.

### Source variants

#### Variant X1 — cache implementation

```rust
struct PolicyCache {
    by_tenant: HashMap<TenantId, HashMap<(UserId, ResourceId), Decision>>,
}

impl PolicyCache {
    fn get(&self, tenant: TenantId, user: UserId, resource: ResourceId) -> Option<Decision> {
        self.by_tenant.get(&tenant)?.get(&(user, resource)).copied()
    }
}
```

Caller:

```rust
let decision = cache.get(request.tenant_id, request.user_id, request.resource_id);
```

#### Variant X2 — cache implementation

```rust
struct PolicyCache {
    entries: HashMap<(UserId, ResourceId), Decision>,
}

impl PolicyCache {
    fn get(&self, user: UserId, resource: ResourceId) -> Option<Decision> {
        self.entries.get(&(user, resource)).copied()
    }
}
```

Caller:

```rust
let decision = cache.get(request.user_id, request.resource_id);
```

#### Variant X3 — cache implementation

```rust
struct PolicyCache {
    entries: HashMap<(TenantId, UserId, ResourceId), Decision>,
}

impl PolicyCache {
    fn get(&self, tenant: TenantId, user: UserId, resource: ResourceId) -> Option<Decision> {
        self.entries.get(&(tenant, user, resource)).copied()
    }
}
```

Caller:

```rust
let decision = cache.get(request.tenant_id, request.user_id, request.resource_id);
```

#### Variant X4 — cache implementation

```rust
struct PolicyCache {
    entries: HashMap<(UserId, ResourceId), (TenantId, Decision)>,
}

impl PolicyCache {
    fn get(&self, tenant: TenantId, user: UserId, resource: ResourceId) -> Option<Decision> {
        let (cached_tenant, decision) = self.entries.get(&(user, resource))?;
        (*cached_tenant == tenant).then_some(*decision)
    }
}
```

Caller:

```rust
let decision = cache.get(request.tenant_id, request.user_id, request.resource_id);
```

#### Variant X5 — cache implementation

```rust
struct PolicyCache {
    entries: HashMap<(UserId, ResourceId), Decision>,
}

impl PolicyCache {
    fn get(&self, user: UserId, resource: ResourceId) -> Option<Decision> {
        self.entries.get(&(user, resource)).copied()
    }

    fn insert(&mut self, user: UserId, resource: ResourceId, decision: Decision) {
        self.entries.insert((user, resource), decision);
    }
}
```

Caller:

```rust
let decision = cache.get(request.user_id, request.resource_id);
if decision.is_none() {
    cache.insert(request.user_id, request.resource_id, evaluate_policy(&request));
}
```

#### Variant X6 — cache implementation

```rust
struct PolicyCache {
    entries: HashMap<(TenantId, UserId, ResourceId), (Decision, Instant)>,
}

impl PolicyCache {
    fn get(&self, tenant: TenantId, user: UserId, resource: ResourceId) -> Option<Decision> {
        let (decision, expiry) = self.entries.get(&(tenant, user, resource))?;
        (*expiry > Instant::now()).then_some(*decision)
    }
}
```

Caller:

```rust
let decision = cache.get(request.tenant_id, request.user_id, request.resource_id);
```

### Reviewer worksheet

| Variant | Target live? yes/no/unresolved | Causal explanation and source anchors | Non-target issue, if any | Confidence |
|---|---|---|---|---|
| X1 | | | | |
| X2 | | | | |
| X3 | | | | |
| X4 | | | | |
| X5 | | | | |
| X6 | | | | |

## 5. Cross-case adjudication questions

After independently labeling all 18 variants, answer:

1. Which variants are false friends: nearby code or a similar symptom without the target causal mechanism?
2. Which variants alter the necessary causal condition, and which alter irrelevant detail?
3. Where is caller context essential to establish the diagnosis?
4. Which conclusions are source-declared, syntax-derived or inferred? None of these excerpts supply compiler/runtime/intervention observations.
5. Which claim would you leave unresolved without additional source context?
6. Did any reviewer mistake a source location match for a correct mechanism?
7. Did any reviewer treat a valid non-target concern as a target hit?
8. Did the two reviewers disagree on severity, causal identity, evidence authority or disposition? Preserve each disagreement separately.

## 6. Submission and agreement recording

Each reviewer submits one completed copy of the finding adjudication card per candidate finding plus the 18-row worksheet. The maintainer records:
- raw independent labels before discussion;
- exact disagreements by case/variant/label dimension;
- adjudication outcome and supporting evidence;
- time spent per case;
- whether a third expert was needed;
- rubric wording that caused ambiguity.

Do not collapse agreement into a single percentage without reporting the number of cases and label prevalence. For rare labels, report raw counts and an appropriate chance-corrected statistic only where its assumptions are defensible. If reviewers agree because the examples are trivial, that does not validate the rubric on real repository tasks.

## 7. Claim boundary

This pack contains three synthetic Rust-like review examples and 18 shuffled source variants. It is a public calibration aid only. It has no benchmark task IDs, no repository slots, no hidden labels, no Cargo manifests, no reference patches, no executable oracles, no PoCs and no run receipts. It is permanently ineligible as a sealed CEB holdout. No code was compiled or executed. Its answer key is maintained in a separate public development document.
