# SCIP Text-to-Claim Corpus Manifest — V19 Evidence Note

## Purpose

V18 preregisters the natural-language text-to-claim extraction benchmark. V19 freezes the corpus sealing and split-reconstruction contract that must exist before any candidate extractor can be tuned against calibration data or evaluated against confirmatory data.

The key separation is:

```text
adjudicated case corpus
    + precommitted split seed
    -> deterministic stratified manifest
    != annotation correctness
    != extractor quality
    != surface fidelity
    != confirmatory execution authority
```

## Exact stratification

The V18 floor is made exact rather than approximate.

For each of the eleven semantic dimensions:

- 52 positive cases;
- 52 negative cases;
- 20 positive + 20 negative calibration cases;
- 32 positive + 32 negative confirmatory cases.

For each of eight discourse families:

- 27 total cases;
- 11 calibration;
- 16 confirmatory.

This yields exactly:

```text
528 calibration
832 confirmatory
1360 total
```

No post-hoc balancing is needed.

## Outcome-independent split reconstruction

V19 uses a 32-byte seed committed before case generation. The seed is revealed only after the corpus manifest is sealed.

Within each exact stratum, assignment is reconstructed by sorting cases using:

```text
HMAC-SHA256(seed,
    "symthaea-scip-text-claim-split-rank-v1\0"
    || assignment_key_sha256_bytes)
```

with `assignment_key_sha256` as the deterministic tie breaker.

The first 20 cases in each dimension/polarity stratum and first 11 cases in each discourse stratum become calibration; the remainder become confirmatory.

A count-preserving swap between calibration and confirmatory therefore fails once the committed seed is revealed, even if every visible global and stratum count remains correct.

## Case identity

`case_id` is SHA-256 over a domain-separated canonical JSON projection of post-adjudication case content.

The following are deliberately excluded from case identity:

```text
split
assignment_key_sha256
```

Therefore the same adjudicated case cannot acquire a new case identity merely by moving between splits or receiving a different assignment key.

Changing adjudicated semantic content changes case identity.

## Cross-split leakage controls

The manifest rejects cross-split reuse of:

- template identity;
- named-entity tuple identity;
- numeric tuple identity;
- exact sentence identity.

Assignment keys and case IDs must also be globally unique.

## Domain separation

The policy uses actual NUL-terminated domains, represented in JSON as `\u0000`, for:

- seed commitment;
- split ranking;
- case identity.

This avoids ambiguity between printable backslash-zero and an actual NUL byte.

## Policy identity

Frozen semantic V19 policy identity:

```text
9a30ba36b3e872aaec349647f71433579bc0932f24c96335bb32afe77498ffc0
```

It binds exact V18 preregistration identity:

```text
96e2ec5e1fad213f4261405c22d1f80cb6f1d106fd5621481eb0e95a6cca4530
```

## Evidence status

The validator and policy bytes were locally exercised before Git object construction. The final hostile-input harness uploaded to Git differs from the earlier local pre-upload harness bytes, so V19 deliberately does **not** claim local exact-byte execution for that final harness.

Instead the hosted workflow binds the final hostile harness by exact Git blob identity:

```text
4c1865fc341a3d926799b8a245bb329ba9cccd61
```

and executes that exact blob. Hosted exact-head PASS is therefore required before the final adversarial harness contributes qualification evidence.

This distinction is intentional:

```text
locally similar harness passed
!= final Git harness qualified
```

## Non-claims

V19 is corpus-governance / measurement infrastructure only. It does not establish annotation correctness, candidate quality, text-to-claim extraction correctness, natural-language semantic fidelity, factual truth, or any runtime/action authority. It does not authorize confirmatory evaluation.
