# Engineering Assurance V2 — Pre-Merge Authority Audit

Status: **draft protocol review / merge blockers**

This document reviews the plan-bound ETK V2 composition introduced by #1864.
It is a source/design audit, not runtime qualification and not evidence that the
current branch compiles or passes CI.

## Authority ladder under review

```text
semantic evidence plan
!= plan-bound admitted evidence
!= plan-bound historical discharge receipt
!= current plan-bound discharge fact
!= complete requirement-verification contract
!= current requirement satisfaction
!= qualified design
!= certification / manufacturing / deployment / actuation authority
```

V2 materially improves the earlier composition prototype by making the unit of
verification an explicit `(requirement->obligation relationship, evidence plan)`
member rather than merely an obligation revision. This permits one obligation to
require several independent plans/load cases while preventing an exact plan from
being counted twice.

## Invariant A — authority-critical digest syntax

A field used as an authority-bearing content digest must not be accepted merely
because it is non-empty.

Current status:

- rendered solver input is bound through `Sha256DigestV1` in the semantic plan;
- the solver output digest is still inherited from `SimulationEvidence` as an
  arbitrary non-empty string and is then included in the V2 admission identity.

Required before protocol freeze:

```text
non-empty provenance label != content digest
canonical digest syntax != authenticated artifact
```

At minimum, V2 must reject solver output content identity that is not canonical
`sha256:<64 lowercase hex>`. Authentication/trust of the producer remains a
separate later theorem.

After changing the fixed output fixtures, recompute exactly once:

```text
plan-bound admitted evidence
  -> historical V2 discharge receipt
  -> current plan-bound discharge fact
  -> current requirement-satisfaction receipt
```

Request/policy/plan identities should remain unchanged because output content is
a result property, not a plan premise.

## Invariant B — capability identity must remain auditable

Content-addressing a rich preimage and then discarding the witness fields makes
later audit unnecessarily dependent on external reconstruction.

Every authority-bearing V2 capability should retain enough immutable data to
reconstruct or export the exact semantic preimage used for its identity, while
keeping construction private and providing no JSON/data -> capability path.

### Plan-bound admitted evidence

The admission identity commits:

- plan ID;
- candidate artifact label;
- source-lineage label;
- convergence/confidence;
- normalized run uncertainty;
- all normalized result metrics/uncertainty;
- warnings;
- solver backend/version;
- exact input/output content identities;
- parser version.

The capability currently retains only its V2 ID, plan ID, and lower V1 admitted
capability. The lower capability does not retain all V2 result/provenance fields.

Required improvement: retain an immutable normalized V2 admission record (or an
equivalent typed witness structure), and expose only a data-only
`audit_record_v2()` view. The V2 identity must be recomputable from that record.

### Plan-bound historical discharge receipt

The receipt identity commits the exact plan-bound admitted-evidence ID. Preserve
that witness in the receipt and expose it through a read-only getter/audit view.

### Current plan-bound discharge fact

The fact identity commits the historical V2 receipt ID plus exact plan/request/
policy/validity/currentness context. Preserve the witness receipt identity in the
fact so present authority has an explicit historical provenance edge.

### Requirement verification contract

The contract identity commits decomposition policy and decomposition acceptance
identities. Preserve both in the typed contract and expose them in a data-only
audit record; do not retain only the final contract hash.

### Requirement satisfaction receipt

The satisfaction identity commits:

- verification contract ID;
- current requirement/subject/twin;
- higher currentness assertion;
- exact current-discharge facts actually used for every required plan.

Preserve the higher currentness assertion and canonical used-fact witness set in
the typed receipt. A later auditor should not have to guess which facts produced
a valid satisfaction identity.

## Invariant C — constructor authority remains one-way

Authority output types must keep private fields and must not implement generic
Serde deserialization or public unchecked constructors.

Data-only audit records are allowed because:

```text
audit representation != authority capability
```

A caller may serialize/export the evidence trail, but must traverse the actual
ETK constructors again to regain authority.

## Invariant D — no fake public parsers for authority IDs

The current V2 `digest_id!` macro gives every generated ID type a public `parse`
method and makes authority-output ID parsers return an error at runtime via a
boolean macro argument.

This is fail-closed, but it is a confusing public API and may trigger strict
lint diagnostics due to constant-condition expansion.

Required improvement: split the macro/API into two categories:

- **authority-output IDs**: private/internal `from_digest`, public read-only
  `as_str`/Display, no public parser;
- **non-authoritative premise IDs** (decomposition policy/acceptance/currentness
  references): explicit checked public parser.

The type system should communicate capability direction instead of relying on a
runtime parser that can never succeed.

## Invariant E — exact-plan completeness

Keep the V2 rule:

```text
verification member = relationship + exact evidence plan
```

Do **not** regress to uniqueness by obligation or relationship. One relationship
may require multiple plans. Duplicate exact plan IDs in one `AllOf` contract
must fail closed.

A current fact for the correct obligation under the wrong plan must never satisfy
the required member.

## Invariant F — lower ETK is defense in depth, not V2 identity material

ETK-2B V1 admission/currentness remains valuable as a lower independent gate.
However its admitted identity uses the older JSON-number representation.

Keep this split:

```text
V1 capability success = mandatory lower gate
V1 capability ID      = NOT hashed into V2 authority identity
```

This prevents serializer-specific V1 float spelling from contaminating the V2
cross-language identity contract while preserving independent lower rejection.

## Invariant G — warnings remain authority-relevant

The V2 semantic plan owns warning policy. `DenyAny` must reject any warning;
`ReviewRequired` must not silently auto-admit; `AllowExact` must admit only the
explicitly allowed warning semantics.

Before protocol freeze, decide whether duplicate identical warnings are a
semantic no-op or a distinct event. If a no-op, canonicalize/deduplicate; if
distinct, document that multiplicity is intentionally identity-bearing.

## Required qualification sequence

Before #1864 can claim merge-ready ETK V2 authority semantics:

1. close canonical solver-output digest syntax;
2. retain reconstructable immutable witness data in every authority capability;
3. split authority-ID construction from premise-ID parsing;
4. resolve warning multiplicity semantics;
5. re-freeze the independent V2 oracle vectors once;
6. update the Rust public-API parity vectors to the exact same values;
7. run exact-head Cargo tests and Clippy;
8. execute the independent checked-in oracle bytes;
9. record exact commit/blob/runtime evidence;
10. only then consider the vectors protocol-stable.

## Deliberate nonclaims

Even after these invariants hold, V2 still does **not** by itself establish:

- solver correctness or physical truth;
- authentication of provenance/currentness/acceptance artifacts;
- evidence-source independence;
- correctness of the requirement derivation;
- correctness/authenticity of decomposition approval;
- complete design qualification;
- certification, manufacturing, deployment, or actuation authority.

Those remain separate authority transitions rather than implications of a hash.
