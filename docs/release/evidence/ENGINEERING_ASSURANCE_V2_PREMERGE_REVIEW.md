# Engineering Assurance V2 — Pre-Merge Authority Audit

Status: **source blockers implemented; runtime qualification pending**

This document reviews the plan-bound ETK V2 composition introduced by #1864.
It distinguishes source-level closure from execution evidence. The current design
has been hardened against the audit findings below, but that does **not** imply
that the Rust branch compiles, passes Clippy/tests, or is merge-qualified.

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

The unit of V2 completeness is an explicit verification member:

```text
(requirement -> obligation relationship, exact evidence plan)
```

One obligation may therefore require multiple distinct plans/load cases. An
exact plan may not be counted twice.

## Source invariant A — authority-critical content IDs

Implemented in the current source candidate:

- rendered solver input is a typed `Sha256DigestV1` plan premise;
- solver-result input digest must parse as canonical lowercase
  `sha256:<64 hex>` and must equal the exact rendered-input plan premise;
- solver-result output digest must also parse as canonical lowercase SHA-256;
- malformed/noncanonical output content identity fails before V2 authority is
  minted.

The rule remains:

```text
non-empty provenance label != content digest
canonical digest syntax != authenticated artifact
```

Digest syntax/content addressing does not authenticate the producer or establish
physical truth.

## Source invariant B — authority identity remains auditable

Implemented in the current source candidate. Every authority-bearing V2 object
retains the witness material needed to reconstruct/export its own hash preimage,
while fields remain private and there is no generic data -> capability parser.

### Plan-bound admitted evidence

Retains:

- exact plan identity;
- canonical solver-output digest;
- complete normalized V2 admission preimage;
- lower ETK V1 admitted capability as defense in depth.

`audit_record_v2()` exposes the retained normalized preimage as data only.

### Plan-bound historical discharge receipt

Retains and exports the exact plan-bound admitted-evidence witness ID in addition
to exact plan/obligation identity.

### Current plan-bound discharge fact

Retains and exports:

- exact historical V2 receipt witness;
- plan, obligation, request, policy, validity and currentness identities;
- exact requirement/subject/twin context.

### Requirement verification contract

Retains and exports decomposition-policy and decomposition-acceptance identities,
plus the canonical exact-plan member set.

### Requirement satisfaction receipt

Retains and exports:

- verification-contract ID;
- current requirement/subject/twin;
- higher currentness assertion;
- exact canonical set of current-discharge facts actually used.

## Source invariant C — authority construction is one-way

Implemented in the current source candidate:

- authority-bearing fields are private;
- authority output types have internal/private digest constructors;
- no generic Serde deserialization or public unchecked authority constructor is
  provided;
- data-only audit records do not confer authority.

```text
audit representation != authority capability
```

## Source invariant D — premise parsing is separate from authority IDs

Implemented in the current source candidate.

The previous boolean-controlled `digest_id!` parser pattern was replaced with two
API classes:

- authority-output IDs: private `from_digest`, public read-only `as_str`/Display,
  **no public parser**;
- non-authoritative premise identities: explicit checked public parser.

Capability direction is therefore expressed by the type API rather than a public
parser that can only fail.

## Source invariant E — exact-plan completeness

Implemented and retained:

```text
verification member = relationship + exact evidence plan
```

The same relationship/obligation may occur under several distinct plans. Only a
duplicate exact evidence-plan identity in one `AllOf` contract fails closed.

A current fact for the correct obligation under the wrong plan cannot satisfy the
required member.

## Source invariant F — lower ETK remains defense in depth

Implemented and retained:

```text
ETK V1 capability success = mandatory lower gate
ETK V1 capability ID      = NOT V2 hash material
```

This preserves independent lower admission/currentness rejection while keeping
serializer-specific V1 float spelling out of the V2 cross-language identity.

## Source invariant G — warnings are authority-relevant

Implemented in the current source candidate:

- warnings are canonicalized in sorted order;
- duplicate identical warning occurrences are rejected rather than silently
  collapsed;
- `DenyAny` rejects any warning;
- `ReviewRequired` cannot silently auto-admit warnings;
- `AllowExact` admits only explicit allowed warning text.

This makes warning multiplicity semantics explicit for V2.

## Re-frozen candidate V2 vectors

Canonical solver output content IDs changed only the result-dependent lineage.
Request/policy/plan/contract identities remain unchanged.

```text
request-v2
sha256:695db7bfd3570d020ecef240303d4ba7cc8fc8ef461f4a7acf1c08491c75f165

policy-v2
sha256:7f58b186d470cd62256df2aa14f6be85e35a52ba7de0b91755e4fed3cebe1a09

plan A
sha256:3b55051507d38ebde23b9f5b5e6ad03c1cadae81ba56abde25ff3b7213ce030b

plan B
sha256:926d82f36922fa3e38ac369c3e841462ea9024460ceb8f83c7007908c2e010e7

admitted A
sha256:daca55e4fdd2c173606e035b63c9721d6ed86c121d28543e3f326366881ce6ed

admitted B
sha256:7f9a4bc46a34f3d97f1fcfbbd4b3479a61de0d601545677dbbf765321d4ddc3b

receipt A
sha256:391e20b8697ad72aa29de0b99b6d317d2dc704542ea88a0e081578ccf720561d

receipt B
sha256:b0a2d81d49dd5bba72a6d1b885d75953768fa0e31ab97d66042a98ee4190ed18

current fact A
sha256:a258c2988a48f27de28d9f0f712f01a3235702ae473537bbe8487f38e2f96b6e

current fact B
sha256:36050614ef2fbe51c52c229e77d5a0616b29a4e38f24e52cfc2511a92df864d3

AllOf contract
sha256:79a8a3e5cda3f89ff66c4f5954f52fb92e3b3e33418d50230b6fd7311be1d800

current requirement satisfaction
sha256:883d74833c424d3a5dfa0a29519fb69a5d2a2e53ecfd0c85629e399ed1b5dce8
```

These are **candidate production vectors** until exact-head Rust execution
reproduces them.

## Independent exact-byte reference evidence

The hardened independent Python oracle was executed locally before check-in with:

- built-in `--self-test`: PASS;
- `python3 -m py_compile`: PASS;
- raw file SHA-256:
  `9f0852271bda6acbabf238f720ea5a97dfa9f7f05cf0a1ab20b6ce1d6f38ad57`;
- locally computed Git blob SHA-1:
  `0c7f364e6f6480b2cd5f92116782dd58e84cfd60`.

GitHub reported the checked-in oracle blob as exactly:

`0c7f364e6f6480b2cd5f92116782dd58e84cfd60`

Therefore the checked-in reference bytes are the bytes that passed the local
self-test and syntax compilation. This is **reference execution evidence**, not
production Rust qualification and not scientific/physical validation.

## Remaining qualification gates

Before #1864 can claim merge-ready ETK V2 authority semantics:

1. exact-head Rust source must compile;
2. exact-head public-API parity tests must reproduce every re-frozen vector;
3. exact-head Clippy/format/workspace gates must pass;
4. independent reference CI should execute successfully where applicable;
5. exact commit/run evidence must be recorded without mixing superseded heads;
6. only then may these candidate V2 vectors be treated as protocol-stable.

No local Rust toolchain was available during this audit, so no local Cargo or
Clippy success is claimed.

## Deliberate nonclaims

Even after qualification, V2 does **not** by itself establish:

- solver correctness or physical truth;
- authentication of provenance/currentness/acceptance artifacts;
- evidence-source independence;
- correctness of requirement derivation;
- correctness/authenticity of decomposition approval;
- complete design qualification;
- certification, manufacturing, deployment, or actuation authority.

Those remain distinct authority transitions rather than implications of a hash.
