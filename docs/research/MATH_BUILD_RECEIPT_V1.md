# MATH-BUILD-002A — Build Receipt v1

## Status

Draft authority contract for provider-independent mathematical build evidence.

This tranche changes no Cargo dependency and no `Cargo.lock` bytes. The normative semantic validator is:

`.github/scripts/validate-math-build-receipt.py`

The JSON interchange description is:

`.github/schemas/math-build-receipt-v1.schema.json`

The JSON Schema is structural. The validator is authoritative for role transitions, receipt hashing, sorted collections, required build gates, and claim ceilings.

## Core rule

```text
valid receipt
    != successful receipt
    != artifact contained in source
    != exact-head build qualification
    != formal authority
    != mathematical authority
```

A receipt may be structurally valid while recording a failed execution. Failure evidence is still evidence and must not be discarded merely because it cannot promote authority.

## Roles

### `artifact_generation`

Records an execution that generated a candidate artifact outside the frozen source subject, such as MATH-BUILD-001Q generating a reconciled `Cargo.lock`.

A generation receipt may never claim:

- `artifact_contained_in_source_subject`;
- `subject_build_qualified`;
- formal authority;
- mathematical authority.

A `pass` generation receipt means only that the declared generation/evaluation recipe passed its recorded gates and produced the bound candidate artifact.

### `artifact_replay`

Binds a generated artifact to the later repaired source subject.

A replay `pass` requires:

```text
generated_artifact_sha256 == replayed_artifact_sha256
```

and may assert only that the artifact is contained in the repaired source subject. Replay alone can never assert build qualification.

A one-byte artifact difference requires a failing replay receipt and a new qualification cycle; it cannot be waived by reviewer judgment.

### `exact_head_build_qualification`

Runs the build recipe against a subject where the accepted artifact is already committed.

For MATH-BUILD-002 v1, a passing package-scoped receipt requires at least these named gates:

```text
cargo_metadata_locked
cargo_fmt
cargo_test_locked_offline
cargo_clippy_locked_offline
```

All recorded gates must pass for receipt outcome `pass`.

This role may assert package/profile-scoped `subject_build_qualified=true`. It may not assert whole-workspace, Lean/formal, specification, novelty, or mathematical authority.

## Canonical receipt digest

`receipt_sha256` is SHA-256 over the canonical JSON object with the `receipt_sha256` member omitted.

Canonicalization is:

1. reject all floating-point values;
2. encode UTF-8 JSON;
3. sort object keys lexicographically;
4. use compact separators with no insignificant whitespace;
5. preserve array order;
6. require authority-bearing arrays such as changed paths and gates to be sorted and duplicate-free where the validator specifies this.

The v1 schema forbids floats specifically to avoid cross-runtime numeric serialization ambiguity. A future schema requiring non-integer numerical evidence must define its canonical numeric encoding explicitly rather than silently relaxing this rule.

## Authority-bearing vs provenance-only fields

Subject identities, artifact digests, recipe semantics, dependency identities, gate identities/results, and predecessor receipt digests are authority-bearing.

Provider name, run ID, and timestamps are retained for provenance but do not by themselves change execution-semantic equivalence or grant authority.

Authentication of an envelope or signature proves who/what attested to bytes; it does not prove that the receipt is semantically valid or authorized for promotion.

## Provider independence

The validator is Python-stdlib-only by design. It must remain runnable without Cargo dependency resolution so it cannot create or repair the dependency graph it is judging.

Future producers may be:

- GitHub Actions;
- Nix-backed local execution;
- another CI provider;
- a Rust tool;
- an in-toto/SLSA attestation producer.

All producers must converge on the same canonical receipt semantics.

## Standards envelope

A future adapter may carry the canonical receipt in an in-toto Statement / SLSA-style provenance envelope and may authenticate that envelope with DSSE/Sigstore.

Those layers are orthogonal:

```text
canonical Symthaea receipt semantics
        ↓ optionally wrapped by
in-toto / SLSA provenance envelope
        ↓ optionally authenticated by
DSSE / Sigstore
```

Neither wrapping nor signing may promote a failing or weaker receipt role into a stronger Symthaea authority state.

## Transition chain

```text
FrozenSourceSubject
        ↓ generation PASS
GeneratedArtifactCandidate
        ↓ exact-byte replay PASS
RepairedContainedSubject
        ↓ exact-head build PASS
PackageScopedBuildQualified
```

Forbidden shortcuts:

```text
ArtifactGenerationReceipt -X-> PackageScopedBuildQualified
ArtifactReplayReceipt     -X-> PackageScopedBuildQualified
Any build receipt         -X-> FormalAuthority
Any build receipt         -X-> MathematicalAuthority
```

## Relationship to current math stack

For MATH-BUILD-001 / #3828:

1. #3879 may eventually emit an `artifact_generation` receipt for its generated lock candidate;
2. the evidence artifact must be inspected before replay;
3. exact accepted lock bytes are replayed into the repaired #3792 subject;
4. the replay produces an `artifact_replay` receipt;
5. the repaired #3792 exact head runs package-scoped qualification and may produce `exact_head_build_qualification`;
6. only then may downstream mathematical branches be re-frozen onto that build-qualified subject.

The later Lean-bridge dependency transition is a distinct lock transition and requires its own generation/replay/exact-head receipt lineage.

## Nonclaims

MATH-BUILD-002A does not itself execute Cargo, repair `Cargo.lock`, qualify `symthaea-math-research`, authenticate Lean, validate a mathematical formalization, establish reviewer independence, establish novelty, or prove mathematical truth.
