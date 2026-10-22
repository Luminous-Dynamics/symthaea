# Engineering Trust Kernel — Semantic Evidence Plan Oracle V1

**Status:** independent reference semantics with exact-byte local execution evidence  
**Branch:** `engineering/etk-3b-evidence-plan-oracle`  
**Base:** `main` at `ada0d92fce3f0ff10883061b0608edfd37b16317`

## Theorem boundary

```text
matching labels != matching engineering semantics

evidence plan != admitted evidence != discharge receipt
```

This oracle freezes the ETK-3B canonical semantic-identity layer. It answers which exact requirement, subject, twin, request, evidence policy, validity domain, currentness assertion, proof obligation, and rendered solver input are bound together before evidence admission is even possible.

It grants no evidence-admission, obligation-discharge, qualification, deployment, fabrication, or actuation authority.

## Independence

`scripts/etk-semantic-evidence-plan-oracle.py` is standard-library Python and imports no Symthaea code. It is based directly on `main`, independently of production ETK-3B PR #1760.

Production code must reproduce the contract independently rather than invoking the reference script as an authority source.

## Canonical identities

The positive fixture freezes this complete lineage:

```text
requirement_revision_id
sha256:e340c5030eebc978c41443ffd64f340dc5febad31e376080340bacfceda60faa

subject_revision_id
sha256:38a9505d423fa3464020107b4e8abc8acc6ac6af33bcc98a2418480e1d33390e

twin_revision_id
sha256:19d558d6e7579f0f44c71398c7ddac659677aca0fa0e65ed46227670e4876cd9

request_revision_id
sha256:aec313ebf10fd7b611f7ff8d6239cf0b4006edde6ae580e7c800ecca8398decc

evidence_policy_revision_id
sha256:ec15bcd4e0adbb135e14287b0791f07b5200faa18a702689b7b131a94b9543e3

validity_domain_revision_id
sha256:90bff4adf8917f2ae071f405d5c46afb310b12a98dd2e4eec845de1261a24890

currentness_assertion_id
sha256:f85bd7d5129b08a3256cfdd5b3506f3cae1dbc625b097a8bf95bf1f6fe709dd9

obligation_revision_id
sha256:743d2c13cfcc52bdfb4cfd9a4a836ed0806ace7b2a68b4c7284b1910d8863c29

evidence_plan_id
sha256:17b14595f14fcf06c5f8f7c38f3953702d5cf93b5e9b40a106e638ed85337f62
```

The proof-obligation vector uses fixed UUID `00000000-0000-4000-8000-000000000042`, allowing independent implementations to reproduce the exact obligation snapshot.

## Intended canonicalization

The reference contract treats these order changes as semantic no-ops:

- accepted-requirement structural-invariant ordering;
- simulation-parameter ordering when names are unique;
- requested-metric ordering;
- validity-dimension map ordering;
- exact warning allow-list ordering.

Duplicates are rejected rather than silently collapsed. Semantic mutations change their owning revision and therefore change the composed evidence-plan identity.

## Authority-critical content

The fixture and canonicalization bind:

- accepted requirement statement, criticality, evidence kind, invariants, domain, and acceptance-record digest;
- subject namespace/key and subject-state digest;
- twin kind, state digest, schema digest, and optional parent revision;
- simulation domain, solver family, objective, parameters, units, provenance text, uncertainty, and requested metrics;
- metric acceptance predicate, uncertainty budgets, external-solver requirement, and warning policy;
- model revision, solver-configuration revision, domain-specific validity dimensions;
- currentness attestation digest and observation timestamp;
- proof-obligation UUID, claim, and expected evidence kind;
- expected rendered solver-input SHA-256.

## Explicit trust limitation

A syntactically valid SHA-256 digest is only an identity. This oracle does **not** prove that the digest was produced by a trusted actor, that its referenced artifact is authentic, or that the artifact is physically true.

Therefore:

```text
content-addressed != authenticated != qualified
```

Acceptance-record authorization, provenance authentication, twin/model attestation, and currentness authentication remain separate ETK theorems.

## Warning semantics

The current simulation bridge exposes warnings as unstructured strings. ETK-3B therefore makes warning treatment part of the evidence-policy identity rather than silently ignoring warnings.

V1 reference modes are:

- `deny_any`;
- `review_required`;
- transitional `allow_exact` with a canonical exact-message set.

Structured warning codes should supersede exact-message allow-listing in a later version.

## Adversarial self-test

The built-in self-test checks:

- canonical equivalence under all intended reorderings;
- requirement semantic mutation changes the plan;
- currentness refresh changes currentness and plan IDs while preserving the twin ID;
- duplicate simulation parameters are denied;
- duplicate requested metrics are denied;
- requirement/request domain mismatch is denied;
- policy metric absent from requested outputs is denied;
- malformed/non-canonical digest syntax is denied;
- exact-warning allow-list order is canonical.

## Exact-byte local execution evidence

The checked-in script was executed locally with Python 3.13.5 using:

```text
python scripts/etk-semantic-evidence-plan-oracle.py --self-test
python -m py_compile scripts/etk-semantic-evidence-plan-oracle.py
```

The raw script SHA-256 is:

```text
bc41df5f96d7fe5877ce3b40a915a86333f56ef594bb7ef28e4e2f1882b4ee24
```

For exact-byte comparison with Git, the local file's canonical Git blob object SHA-1 was computed over `blob <length>\0 || bytes` and is:

```text
a381ec6b549a6fe4f64ee2de4fd2727f67cba2a7
```

GitHub reports the checked-in script blob as exactly:

```text
a381ec6b549a6fe4f64ee2de4fd2727f67cba2a7
```

Therefore the checked-in script bytes are the exact bytes that produced the frozen vectors and passed the local self-test / `py_compile`. This is execution evidence for the reference oracle only, not qualification of production Rust or repository-wide CI.

## Production parity target

Production ETK-3B PR #1760 should reproduce all nine frozen identities through public Rust APIs using the same fixed fixture. Production code must not call this Python oracle.

The parity theorem should additionally require semantic mutations to affect only the expected portion of the lineage—for example, currentness refresh should change `currentness_assertion_id` and `evidence_plan_id` without changing the bound twin revision.

## Deliberate nonclaims

This oracle does not establish evidence admission, solver correctness, physical truth, authenticated source provenance, authenticated currentness, evidence independence, safety-case closure, design qualification, certification, deployment approval, manufacturing approval, or physical actuation authority.
