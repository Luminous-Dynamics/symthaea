# Broca UniMorph compiler contract

This document defines the evidence boundary for the deterministic UniMorph-style morphology compiler.

## Source format

The accepted input shape is a selected single-line record:

lemma<TAB>form<TAB>feature1;feature2;...

The source record is treated as UTF-8 bytes. Only a terminal LF or CRLF is removed for parsing; any remaining LF or CR is rejected. The three parsed fields are trimmed.

## Normalization

The current Symthaea adapter sorts feature tokens lexicographically and rejects duplicate or empty tokens before recording that policy as:

`trim-one-line-ending-sort-feature-tokens-sort-output-rules-v1`

This is a Symthaea-local deterministic normalization policy.

It is **not** the official UniMorph canonicalizer.

The official UniMorph canonicalizer places the part-of-speech tag first, then universal tags by the category they represent, then language-specific tags lexicographically. It also checks for conflicts and inconsistencies.

Therefore a compiler receipt using the current adapter must never describe its normalization as "UniMorph canonical form".

## Executable projection

The adapter converts only transformations directly representable by the narrow executable operation vocabulary:

- identity
- suffix append
- prefix prepend

It does not infer replacement, stem alternation, reduplication, deletion, or other morphophonological processes.

Unsupported transformations fail closed.

## Provenance

The current compilation witness schema is `broca-morphophonological-compilation-witness-v3`.
Adding implementation-identity fields is a schema change and therefore invalidates older witness encodings rather than silently interpreting them as complete.

A source-backed compilation witness binds:

Source slices are record selections, not arbitrary byte windows: each selected range must begin at the start of the artifact or immediately after an LF, and end at the artifact end or immediately before an LF. This prevents a valid-looking TSV row from being carved out of the middle of a larger source line.

The `record_id` on a source slice is a caller-supplied label. The cryptographic identity of the selected bytes is the per-slice BLAKE3 digest plus the exact byte range; the label itself is not treated as an intrinsic upstream identifier.

For the declared UniMorph compiler, source mapping is exact: the executable rule set must contain exactly one source-bound rule for every selected source record, and cannot add unbound executable rules while still claiming the same compilation witness.

- the exact source artifact digest
- exact selected source byte ranges
- exact selected-record digests
- exact source-record-to-rule identities
- compiler identity and version
- normalization policy
- output executable rule-set digest
- transformation digest

The witness can re-execute the current adapter against the exact selected bytes and requires the exact executable output and witness to reproduce.

## Evidence levels

The following claims are distinct:

1. **Artifact identity** — the supplied bytes match a recorded digest.
2. **Source selection** — the selected byte ranges and per-record digests match the artifact.
3. **Compiler replay** — the current adapter reproduces the exact executable representation.
4. **Morphological replay** — the executable rules reproduce the recorded surface forms.
5. **Linguistic validity** — the upstream resource or resulting grammar is linguistically correct.

The current compiler provides evidence for levels 1–4 when the relevant bytes and witnesses are supplied.

It does not establish level 5.

## External corpus caution

UniMorph language repositories can carry known data-quality issues. An external resource must therefore be treated as provenance-bearing input, not as an assertion of linguistic ground truth.

The repository's issue tracker is part of the surrounding evidence context; individual source revisions should be pinned and independently replayable.

## Frozen snapshot requirement

Before an external UniMorph snapshot is admitted as a repository fixture, capture:

- observed immutable upstream commit or release
- language/version metadata where available
- exact raw artifact bytes and digest
- source URI
- license/attribution evidence
- exact selected byte ranges and digests
- non-empty source slices; a zero-byte selected record is invalid
- exact compiler/normalization version
- exact compiler implementation, parser, and build-context revisions
- exact emitted rule-set digest
- reproducible compiler witness

Do not invent a revision from a crawl timestamp or mutable branch name.

## Claim ceiling

A passing compiler witness means that the declared deterministic transformation can be reproduced from the declared source bytes.

It does not mean:

- the source dataset is complete
- the source dataset is error-free
- the source labels are correct
- the derived rule system is a complete morphology engine
- the resulting pronunciation is correct or natural
- the resulting speech is human-appropriate

Those require separate evidence and qualification.

## Implementation identity

UniMorph compiler receipts carry three generated identities:

- `compiler_implementation_revision`: a content-addressed BLAKE3 identity of the complete `lexical_binding.rs` compiler module plus the build-time identity mechanism.
- `source_parser_revision`: a content-addressed BLAKE3 identity of the explicitly delimited accepted source-format parser surface plus the build-time identity mechanism.
- `compiler_build_context_revision`: a content-addressed BLAKE3 identity of the Broca crate manifest, workspace manifest, checked-in `Cargo.lock`, pinned `rust-toolchain.toml`, repository-local `.cargo/config.toml` and `.cargo/config` presence/content, actual `rustc --version --verbose` identity, actual `cargo --version --verbose` identity, configured rustc/workspace-wrapper values, enabled Cargo feature set, all `CARGO_CFG_*` values, Cargo profile/debug/optimization/job settings, `CARGO_ENCODED_RUSTFLAGS`, target triple, and build host.

The build-context identity is deliberately separate from executable source identity. Rust source alone is not a complete reproducibility boundary when dependency resolution, the actual compiler or Cargo binary, compiler wrappers, feature selection, compiler flags, or compilation target changes.

Generic compiler witnesses do not borrow these UniMorph identities; their fields remain explicitly absent until a compiler-family-specific implementation identity is defined. A generic/non-UniMorph witness carrying any of these UniMorph identity fields is rejected rather than silently accepting an uncontracted identity format.

These revisions are generated by `crates/domains/symthaea-broca/build.rs` during compilation. They are not manually maintained semantic labels and are not derived from a crawl timestamp, branch name, mutable external revision, or undeclared dependency/toolchain state.

Current UniMorph replay requires all three generated identities to equal the currently compiled implementation/build context, including the actual rustc identity, target, host, and enabled Cargo features used to compile the witness. A witness with unchanged executable output but tampered implementation, parser, or build-context revision therefore fails closed.

The repository currently pins Rust `1.96.0` in `rust-toolchain.toml` and maintains a workspace `Cargo.lock`. CI independently installed and reported rustc `1.96.0` for the prior qualification run; the receipt now binds the actual verbose rustc and Cargo identities instead of relying only on the requested channel.

Changing the compiler module, source parser surface, revision-generation mechanism, manifests, lockfile, or pinned toolchain invalidates prior compiler receipts by construction. This is intentionally conservative: these identities are provenance, not a claim of linguistic correctness.


The duplicate-token rejection is intentionally syntactic. It does not infer semantic conflicts between different UniMorph dimensions; those remain part of the official taxonomy/canonicalizer qualification tracked separately.


Revision commitments are not self-certifying: validation first requires the generated current implementation and parser revisions, so recomputing the transformation digest with a forged revision does not make a witness acceptable.


Structural source-artifact validation intentionally does not require the recorded compiler revisions to equal the current implementation/context; this preserves historical witness inspectability. Current compiler replay is the stronger admission operation and requires the generated implementation, parser, and build-context identities to equal the current implementation/context before re-execution.


## Qualification trigger boundary

The Broca Feature Matrix is intentionally triggered by changes to the compiler implementation, its declared dependencies/toolchain, the frozen UniMorph manifests, and the build configuration inputs bound into the compiler build-context identity. In particular, `docs/broca/**`, `rust-toolchain.toml`, and both repository-local Cargo config spellings are qualification-triggering paths.

The workflow also disables persisted checkout credentials for the qualification jobs and grants the workflow only `contents: read`; the freeze audit does not require repository write access.

This trigger surface is part of the evidence boundary: changing a bound qualification input without re-running the exact-head audit is treated as an invalid qualification state rather than an implicit continuation of the previous receipt.
