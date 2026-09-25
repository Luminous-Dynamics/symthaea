# SYM-FV-001A — exact Aeneas extraction probe v1

This tranche answers one bounded question:

> Can the exact production `symthaea-core` scalar BinaryHV binding kernel be translated through the pinned Charon/Aeneas stack into Lean and typechecked against that same Aeneas backend?

It does not prove the HDC algebra and it does not prove the SIMD-dispatched production `bind()` path.

## Frozen source subject

- source commit: `458c7b98d81c64b9361e252f85ef9d45132e6682`
- source path: `crates/core/symthaea-core/src/hdc/binary_hv.rs`
- blob: `22a56cfaedf5b0cb7d8ff8b73c41ed4caf8d7056`
- item: `BinaryHV::bind_scalar`
- Charon translation root: `crate::hdc::binary_hv::_::bind_scalar`

The `_` segment is deliberate. At the pinned Charon revision, inherent methods have a distinct impl/type segment in their full item name (for example, Charon's own tests render `test_crate::{Generic<T>...}::get` and match it with `test_crate::_::get`). The Rust-source-looking path `crate::...::BinaryHV::bind_scalar` is therefore not used as an unverified assumption.

The qualifier also requires the qualification head to carry the same source blob, requires exactly one `fn bind_scalar` definition in that source file, uses `--start-from` so this item is the translation root rather than merely an opacity whitelist, and requires the retained LLBC to contain the selected `bind_scalar` item name. A proof-infrastructure change therefore cannot silently rebind or broaden the Rust subject.

## Frozen translator stack

- Aeneas: `6939df045f741c2d4afb927a7adc021752562ef6`
- Charon: `62585970fc75f61d83c7898ff8dfdd7edaa3c073`, verified from that Aeneas revision's `flake.lock`
- Lean: `leanprover/lean4:v4.31.0`, verified from that Aeneas revision's `backends/lean/lean-toolchain`

The workflow invokes Charon and Aeneas through the exact Aeneas Nix flake revision. The checked-out Aeneas source is used only to verify the lock identities and to provide the matching Lean backend to Lake.

## Qualification sequence

1. verify source commit/blob/path/item invariants and exactly one source `bind_scalar` definition;
2. verify exact Aeneas, locked Charon, and Lean identities;
3. run Charon with `--preset=aeneas` and `--start-from crate::hdc::binary_hv::_::bind_scalar`;
4. require the retained LLBC to contain the selected item name, then retain and SHA-256 the LLBC;
5. translate the LLBC with the Lean backend;
6. retain and hash every emitted Lean file;
7. census any `_Template.lean` external-model boundary;
8. if no external model is required, build the generated project with Lake against the exact pinned Aeneas Lean backend.

An emitted external-model template is classified as `UnsupportedDependencyBoundary`, not patched over with an unreviewed axiom.

## Result classes

The receipt terminates in exactly one of:

- `ExtractedAndLeanTypechecked`
- `UnsupportedRustConstruct`
- `UnsupportedDependencyBoundary`
- `TranslatorFailure`
- `LeanGenerationFailure`
- `LeanTypecheckFailure`
- `EnvironmentFailure`

The current qualifier uses conservative classifications. In particular, an unexpected Charon failure is recorded as `TranslatorFailure` until its retained log justifies a more specific unsupported-feature classification.

## Claim ceiling

A successful run establishes extraction feasibility for this exact source subject and exact toolchain only.

`ExtractedAndLeanTypechecked` does **not** establish:

- the abstract HDC laws;
- Rust-to-abstract-spec refinement;
- equivalence of `bind_scalar` and the SIMD-dispatched `bind()`;
- rustc/LLVM or native-binary correctness;
- empirical claims about HDC cognition or intelligence.

`SYM-FV-003` / #5716 owns the first `ExtractedSourceRefinement` theorem joining this artifact to the abstract Lean specification.
