# Qualification Identity Census V1

Date: 2026-09-09
Status: architecture census; non-authorizing
Series: SCI-Q2

## Purpose

This document inventories identity and digest mechanisms already present in Symthaea before a qualification receipt is allowed to bind candidate artifacts or execution environments.

The immediate motivation is SCI-Q1 (`QualificationReceiptV1` candidate): a scientific disposition can only become useful for later materialization if a verifier can prove which exact candidate, base, qualification instrument, profile, and execution environment the disposition refers to.

This census does not define or mint materialization authority. It deliberately stops before a universal identity implementation.

## Central non-equivalence

The system MUST preserve these distinctions:

`logging fingerprint != byte digest != structured commitment != Git object identity != provenance/authentication != execution identity != authority`

A digest proves equality only under the encoding and algorithm that produced it. It does not by itself establish authorship, authenticity, scientific validity, successful execution, or permission to act.

## Current identity mechanisms

| Mechanism | Location | What it identifies | Safe authority | Must not be used as |
|---|---|---|---|---|
| `config_hash(Debug)` | `symthaea-evidence-plane` | Best-effort local config fingerprint | Logs, deduplication, dashboards | Cryptographic or cross-version authority identity |
| raw-byte SHA-256 | `symthaea-fabrication-kernel::crypto_digest` | Exact byte sequence | Tamper-evident byte equality when algorithm/bytes are known | Authentication or semantic document identity by itself |
| canonical JSON SHA-256 | `symthaea-muse::evidence_digest` | Canonically serialized structured value | Structured evidence commitments within its defined encoding | Universal byte identity or universal serializer without qualification |
| study artifact bundle commitment | `symthaea-muse::study_artifact` | Frozen study authorities, environment identities, files, and records under one commitment schema | Muse study artifact integrity under that schema | Generic qualification artifact identity without migration audit |
| externally supplied `ArtifactDigest` | `symthaea-subterranean::update_control` | Opaque 32-byte digest supplied by a verifier outside the crate | Safe update-state comparison after external verification | Proof that bytes were actually hashed or authenticated by this module |
| path -> raw SHA-256 manifest | `xtask::manifest` | Selected repository files and their raw bytes | Packaging/integrity manifest | Full source-tree or execution-environment identity |
| promotion-script SHA-256 | `scripts/*promote_candidate.sh` | Candidate and active checkpoint bytes | Candidate-vs-active byte comparison | Scientific qualification or source-tree equivalence |
| Git commit/tree/blob object IDs | repository/Git | Git objects under the repository object format | Source-history and tree identity | A fixed SHA-256 digest type or external artifact authentication |

### Evidence-plane config fingerprint

`symthaea-evidence-plane::config_hash` is intentionally based on `DefaultHasher` over `Debug` formatting. Its own documentation says it is suitable for logging/deduplication/dashboard identity and is not cryptographic or guaranteed stable across Rust versions.

Decision: retain this mechanism for its current operational purpose. It MUST NOT participate in `QualificationArtifactIdentity`, `QualificationExecutionIdentity`, receipt authority, or materialization authorization.

### Fabrication raw-byte SHA-256

`symthaea-fabrication-kernel::crypto_digest` implements a typed SHA-256 digest over raw bytes. Its module-level authority boundary is correct: a digest authenticates bytes only when anchored by a trusted signature or independently trusted channel.

Decision: raw-byte SHA-256 is a strong semantic candidate for qualification patch/workflow/lockfile byte commitments. This census does not move or re-export the fabrication implementation because cross-domain API extraction has not itself been qualified.

### Muse canonical structured commitments

`symthaea-muse::evidence_digest` canonicalizes JSON object-key ordering and hashes the resulting bytes with RustCrypto SHA-256. Muse then uses explicit commitment structures to omit self-digest fields and commit to the exact authority-bearing fields of a record.

`study_artifact` is especially relevant: its bundle commitment binds frozen manifest, methodology, blinded schedule, production plan, renderer binary, render environment, soundfont, and artifact records.

Decision: this is a strong pattern for a future qualification profile/environment commitment. It remains Muse-local for now. Repeated calls inside Muse are one domain reusing one mechanism, not independent cross-domain recurrence.

### Subterranean update identity and transition separation

`symthaea-subterranean::update_control` carries artifact/configuration/rollback digests in an update manifest and uses them inside a guarded state transition. Its documentation explicitly delegates actual artifact-byte and signature verification to an external layer.

Decision: preserve the architectural lesson, not the type: identity verification and action authority are separate stages. A future materialization token should be minted only after artifact identity verification, but the generic qualification layer should not reuse `ArtifactDigest([u8; 32])` because that type does not encode the digest algorithm or Git object format.

## Identity classes that must remain distinct

### 1. Raw byte identity

Question answered: "Are these exact bytes the same under algorithm A?"

Examples:
- candidate patch bytes;
- workflow file bytes;
- `Cargo.lock` bytes;
- external tool binary bytes;
- generated evidence artifact bytes.

Required metadata:
- algorithm identifier;
- digest bytes;
- encoding/domain when the byte stream is not self-evident.

### 2. Canonical structured identity

Question answered: "Do these structured values have the same meaning under canonicalization schema S?"

This requires an explicit schema/canonicalization version. Serializing a Rust value and hashing it is not automatically a stable public protocol.

Candidate uses:
- qualification profile semantic commitment;
- execution-environment manifest commitment;
- artifact-identity envelope commitment.

### 3. Git object identity

Question answered: "Which exact Git object is this?"

Git object identity is repository/object-format specific. Qualification types MUST NOT assume every Git OID is 32 bytes. A Git identity should retain at least:
- object format / algorithm;
- object kind where required (`commit`, `tree`, `blob`);
- hexadecimal/object bytes.

Candidate uses:
- base commit identity;
- resulting source-tree identity;
- optional qualification-branch commit identity.

### 4. Provenance/authentication

Question answered: "Why should this identity statement be trusted?"

A content digest alone does not answer this. Trust may come from a signed statement, a trusted repository channel, an authenticated operator, transparency log, independent witness, or other qualified provenance source.

SCI-Q3 should bind content first. Provenance/signature policy should remain a separate theorem so the system can distinguish "same bytes" from "trusted source of those bytes."

### 5. Execution identity

Question answered: "Under what exact instrument and environment were these observations produced?"

A GitHub Actions run ID or workflow name is not sufficient. `ubuntu-latest` is a moving label, not a stable environment identity.

At minimum, a future execution envelope should account for:
- workflow bytes or canonical workflow commitment;
- immutable action identities;
- compiler/toolchain identity (`rustc -vV`, Cargo version, target);
- dependency lock identity;
- runner OS/image/build identity where available;
- Nix flake/derivation identity when execution occurs under Nix;
- relevant environment variables and feature flags;
- external binaries/tools and their identities;
- seeds and deterministic configuration where relevant;
- declared input artifact identities.

## Qualification artifact envelope: proposed minimum semantics

A future `QualificationArtifactIdentityV1` should conceptually bind:

| Field | Why it is required |
|---|---|
| schema/version | Prevent reinterpretation under a changed identity protocol |
| repository identity | Prevent an otherwise valid Git OID from being silently interpreted in another repository context |
| base Git commit OID | Same patch applied to a different base is not the same candidate |
| candidate patch raw-byte digest | Bind the exact proposed mutation bytes |
| resulting Git tree OID | Bind the exact source state produced after applying the patch |
| changed-path set/manifest | Useful explicit scope evidence; not a substitute for the tree identity |
| qualification workflow raw-byte digest | Bind the instrument definition used to collect observations |
| qualification profile semantic commitment | Bind which checks were premise/scientific/admissibility and which were required |

A patch digest without a base is insufficient. A base plus patch is stronger but still benefits from an independently recomputed result-tree identity. The verifier should reject if the applied result does not equal the bound tree.

## Qualification execution envelope: proposed minimum semantics

A future `QualificationExecutionIdentityV1` should conceptually bind:

| Field | Meaning |
|---|---|
| workflow commitment | Exact instrument source |
| action commitments | Exact third-party action implementations, not only moving major-version tags |
| repository/base/candidate identity | Execution subject |
| toolchain commitment | rustc/cargo/target/component versions |
| dependency commitment | lockfile and relevant vendored/source dependencies |
| runner/environment commitment | OS/image/build/Nix derivation or equivalent |
| external-tool commitments | Lean, Z3, Workbench, FluidSynth, or other experiment-specific executables |
| input commitments | datasets, fixtures, models, configs, seeds |
| output/evidence commitments | observations and retained evidence artifacts |

This should be profile-extensible: not every experiment needs every external tool, but a profile must say which identities are required rather than allowing silent omission.

## Materialization theorem boundary

The eventual action path should require all of the following independently:

1. The qualification receipt validates against its trusted profile.
2. The receipt disposition is `Qualified`.
3. The receipt is bound to the exact artifact identity envelope.
4. The artifact envelope verifies against the intended repository/base/patch/result tree.
5. The execution envelope verifies the instrument/environment identity required by the profile.
6. Candidate bytes have not drifted since qualification.
7. A dedicated materialization policy/verifier mints an opaque, bounded action capability.

Even this establishes only that a specific candidate passed a specific qualified procedure in a specific environment. It does not establish external scientific truth, independent replication, consensus, or unlimited action authority.

## Failure semantics for identity verification

Identity mismatch MUST NOT be mapped to `CandidateRejected` scientific failure.

Suggested separation:
- candidate bytes/tree mismatch -> identity verification failure;
- workflow/profile mismatch -> instrument identity failure;
- environment mismatch -> execution identity failure or incomplete qualification;
- authentication/provenance failure -> provenance failure;
- scientific check failure -> candidate scientific rejection;
- repository hygiene failure -> candidate hygiene failure.

This preserves the SCI-Q1 invariant that observation, interpretation, identity, and authority are different layers.

## Anti-patterns to reject

The qualification architecture should explicitly reject these shortcuts:

- using `config_hash` as cryptographic authority;
- trusting a workflow run conclusion without per-check semantic interpretation;
- treating a workflow name as workflow identity;
- treating `actions/checkout@v4` or another moving tag as an immutable action identity;
- treating `ubuntu-latest` as an exact environment identity;
- hashing a patch without binding its base;
- binding base+patch without checking the resulting tree;
- treating SHA-256 equality as authentication;
- treating a signature as scientific truth;
- forcing Git OIDs into a fixed `[u8; 32]` digest type;
- silently changing profile roles while retaining the same profile version/commitment;
- allowing a serialized "authorized" boolean/enum to substitute for verifier-owned capability issuance.

## Reuse decisions

### Safe to reuse conceptually now

- raw-byte cryptographic digest semantics;
- Muse's explicit commitment-struct pattern;
- Subterranean's separation of digest verification from guarded state transition;
- Git-native commit/tree identity for source state;
- verifier-owned opaque authority tokens from SCI-Q1.

### Do not unify yet

- the concrete fabrication and Muse digest APIs;
- canonical JSON as a universal Symthaea serialization protocol;
- `ArtifactDigest([u8; 32])` as a universal identity type;
- execution identity across all scientific domains before profiling actual external-tool/environment needs.

## Proposed sequence

### SCI-Q1 — Qualification semantics

`Observation -> profile-relative disposition -> verifier-owned reporting token`

Candidate: PR #1082. This stage deliberately has no materialization authority.

### SCI-Q2 — Identity census

This document. Establish non-equivalences, existing mechanisms, safe reuse boundaries, and minimum artifact/execution envelope semantics. Non-authorizing.

### SCI-Q3 — Artifact identity qualification

Qualify a minimal algorithm-aware raw-byte digest + Git-object-aware artifact envelope for one real qualification candidate. Required adversarial cases:
- same patch, different base;
- one-byte patch mutation;
- base+patch that does not produce bound tree;
- workflow-byte mutation;
- profile semantic mutation;
- malformed/unsupported digest algorithm;
- repository-context mismatch.

No materialization token yet.

### SCI-Q4 — Execution identity qualification

Bind the exact qualification instrument and environment for one real lane. Replace mutable action/environment labels with evidence-bearing identities where technically possible. Distinguish unavailable environment evidence from candidate failure.

No materialization token yet.

### SCI-Q5 — Receipt-to-artifact binding

Prove that observations/disposition from one receipt cannot be replayed for a different artifact, base, workflow, profile, or execution envelope.

### SCI-Q6 — Materialization capability

Only after Q1-Q5 qualify should a verifier be allowed to mint an opaque, single-purpose materialization capability. The capability should bind the destination/base and exact result identity and should be invalid after any byte drift.

### SCI-Q7 — Provenance/authentication

Add trusted-source/signature/transparency requirements appropriate to the action surface. Keep provenance authority distinct from scientific disposition.

## Immediate engineering consequence

Current qualification workflows may continue to serve as experimental qualification instruments, but their successful conclusions must not be interpreted as exact execution-environment proof until SCI-Q4 exists. In particular, mutable action tags and rolling runner labels are known identity gaps to close, not reasons to rewrite SCI-Q1 while its exact head is under qualification.

## Exit criteria for this census

SCI-Q2 is complete when:
- the current identity mechanisms are accurately classified;
- no non-cryptographic fingerprint is promoted into authority;
- byte, structured, Git, provenance, execution, and authority identities remain distinct;
- Q3/Q4 minimum envelopes and adversarial tests are explicit;
- no production identity implementation or materialization authority is introduced by this document.
