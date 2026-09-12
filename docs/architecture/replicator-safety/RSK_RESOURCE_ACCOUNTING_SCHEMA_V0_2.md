# Replicator Safety Kernel — Resource Accounting Schema v0.2

Status: **normative semantic-identity contract; not production admission**

This document refines the RSK resource-accounting identity so the exact numeric dimension identifiers consumed by the Rust semantic TCB are committed by the same scheme digest as their human-readable semantic meanings.

It contains no physical replication mechanism, material recipe, manufacturing process, biological design, or physical resource model.

---

## 1. Governing correction

The v0.1 reference profile committed semantic dimension identifiers such as:

```text
budget.compute
budget.energy
```

while the Rust reference layer separately used numeric identifiers such as `0` and `1`.

That separation leaves an unacceptable semantic-identity ambiguity: the same advertised scheme digest could be paired with a different runtime numeric mapping.

The v0.2 rule is therefore:

```text
ResourceAccountingSchemeId
    = Digest(canonical schema bytes including semantic ID AND runtime numeric ID)
```

No mutable side table may define the authoritative runtime mapping.

---

## 2. Canonical v0.2 dimension

Every dimension in the production-target v0.2 profile contains at least:

```text
{
    id,
    numeric_id,
    unit,
    scale,
    minimum,
    maximum,
    required,
    rounding,
    aggregation
}
```

`numeric_id` is an unsigned 16-bit identity in the current Rust TCB profile.

It is semantic identity, not array position.

Numeric IDs:

- MUST be unique inside one scheme;
- MAY be sparse;
- MUST NOT be inferred from list position;
- MUST NOT be reassigned without producing a new scheme identity;
- MUST NOT be supplied by mutable registry metadata outside the digested schema bytes.

---

## 3. Canonical ordering

The canonical schema dimension array remains ordered by canonical semantic `id`.

This means array order and runtime numeric identity are intentionally separate concepts.

The verifier checks:

- semantic IDs are canonical and strictly ordered;
- semantic IDs are unique;
- numeric IDs are valid `u16` values;
- numeric IDs are unique;
- all other v0.1 dimensional semantic constraints remain valid.

A runtime implementation may materialize a second numeric-ID-sorted representation after verification, but that representation MUST derive only from the verified canonical schema.

---

## 4. Scheme identity

For the abstract v0.2 golden profile:

```text
budget.compute -> numeric_id 0
budget.energy  -> numeric_id 1
```

canonical SHA-256 scheme identity is:

```text
8386b56b11818273612cc9f15e6c6fa8cd19bbf9aa2c7a3476022d2f7ef0f1e1
```

Changing only the runtime mapping to:

```text
budget.compute -> 1
budget.energy  -> 0
```

produces a different identity:

```text
04f5f4cce1d7f819becf84fdff236ff6db695e0590c37c665994487d2b5ff024
```

The exact values are test vectors, not production resource semantics.

---

## 5. Historical v0.1 behavior

The v0.1 resource schema remains valid historical/reference evidence.

It MUST NOT be retroactively interpreted as committing numeric IDs that were not present in its canonical bytes.

Reference tooling may continue to parse and validate v0.1 evidence so old receipts remain understandable.

Production-target runtime binding requires v0.2 or a later explicitly admitted schema profile.

Conceptually:

```text
validate_resource_schema(v1)                  -> historical/reference OK
require_runtime_bound_resource_schema(v1)     -> DENY
require_runtime_bound_resource_schema(v2)     -> structurally eligible
```

Structural eligibility is still not verified registry provenance or replication authority.

---

## 6. Runtime vector binding

A runtime resource vector is meaningful only together with one exact v0.2 scheme ID.

Conceptually:

```text
ResourceVector {
    scheme_id,
    quantities: [
        (numeric_id, amount),
        ...
    ]
}
```

For one verified v0.2 scheme:

- every numeric ID in the vector MUST resolve to exactly one schema dimension;
- required schema dimensions MUST be present;
- duplicate numeric IDs are invalid;
- vector numeric IDs are canonical and strictly increasing;
- per-dimension min/max and representation rules still apply;
- an unknown numeric ID fails closed.

---

## 7. Registry composition

The verified schema registry defined by `RSK_VERIFIED_SCHEMA_REGISTRY_V0_1.md` must verify the exact canonical v0.2 bytes and scheme digest before a runtime numeric mapping is trusted.

The registry does not supply an independent mutable mapping.

The authoritative mapping is extracted from the already verified schema bytes.

Thus:

```text
VerifiedSchemaRegistrySnapshot
    -> exact v0.2 schema bytes
    -> exact scheme digest
    -> exact semantic ID <-> numeric ID mapping
    -> structural value validation
```

Any disagreement at any step fails closed.

---

## 8. Migration from v0.1

v0.1 and v0.2 are different accounting schemes.

A v0.1 grant, budget, checkpoint, authorization or ledger state MUST NOT be relabeled as v0.2 merely because the operator believes the intended mapping was `0/1`.

Migration requires explicit cross-schema evidence under the existing conservative transition contract.

At minimum the transition must bind:

- source v0.1 scheme identity;
- target v0.2 scheme identity;
- exact semantic dimension correspondence;
- exact target numeric IDs;
- source remaining allowance;
- conservative target image;
- rounding and dimensional rules;
- evidence/policy identity;
- validity scope;
- applicable ancestor scopes.

Default:

```text
No verified transition -> No authority carryover
```

---

## 9. Remaining-authority rule

Migration continues to operate on **remaining authority**, not independently converted limits and consumed counters.

For each applicable source scope:

```text
source_remaining = source_limit - source_consumed
```

A target v0.2 state is admissible only if separately verified policy establishes:

```text
target_remaining <= ConservativeImage(source_remaining)
```

The v0.2 schema itself does not manufacture that proof.

---

## 10. Durable evidence

Once v0.2 enters production authority paths, durable evidence must bind its exact scheme identity through:

- grants;
- lineage hard policy;
- requests;
- evaluated authorization;
- committed actions;
- ancestor budget accounting;
- ledger events;
- checkpoints/state digests;
- safety-case snapshots;
- recovery/epoch transitions;
- admitted release/runtime configuration.

A numeric resource vector without its exact scheme identity is incomplete authority evidence.

---

## 11. Golden-vector requirement

The committed v0.2 golden corpus is:

`golden/RSK_SEMANTIC_SCHEMA_GOLDEN_V0_2.json`

It must demonstrate at least:

- the expected v0.2 scheme digest;
- runtime semantic-name -> numeric-ID mapping;
- digest sensitivity to numeric-ID remapping;
- valid semantic-name vectors;
- valid numeric-ID vectors;
- required-dimension failure;
- expected remaining amounts.

Python and Rust implementations must converge on the same canonical bytes, digest and mapping before production integration.

---

## 12. Non-amplification

Moving to v0.2 does not itself grant authority.

In particular, the v0.2 scheme cannot:

- create a replication grant;
- increase a capability set;
- reset resource consumption;
- clear quarantine/revocation;
- extend an expiry;
- satisfy quorum;
- authorize a cross-schema translation;
- make a historical v0.1 state current.

---

## 13. Production gate

Until exact-head executed evidence demonstrates Python/Rust agreement and the v0.2 identity is threaded through every positive-authority and durable-state surface, production admission remains:

**DENIED / NOT YET ELIGIBLE**.
