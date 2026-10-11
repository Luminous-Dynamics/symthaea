# Qualification Receipt-Core Framing Migration Note V1

Status: architecture correction only. No PASS, migration, or qualification authority is established by this document.

Related: #2469, #3742, #3746, #3747, #3748.

## Correction

The historical positive qualification receipt core is cross-language consistent, but it is not encoded under the later normative qualification framing protocol.

Historical Python reference:

```text
tools/qualification-receipt-core-v1@54e6fc3a688b04cbd4ea4b32e9c1b91454093aee
scripts/qualification_receipt_core_v1.py
Git blob 4784ac1fd7fc6d0036d6d9f8baa607467e0ed1f2
```

Historical Rust implementation:

```text
tools/qualification-receipt-core-rust-v1@765fdedefd72e5042f458f682c1e9a176ff5a7ff
crates/core/symthaea-qualification-receipt-core/src/lib.rs
Git blob 1010fe1146afd0b9169f4f3213f11ff896ea22d0
```

They reproduce the same historical V1 canonical transcript using custom little-endian u64 length/count framing and BLAKE3 identity.

The later one-framing rule in #2469 instead identifies `qualification_framing_v1.py` as the single normative generic qualification framing protocol:

```text
tools/qualification-semantic-convergence-v2@5d951664ff2608d231da258b2a29db56d07e83f5
scripts/qualification_framing_v1.py
Git blob ab5f50d9d2e68fadd42f171e66855eba60d03930
magic SYMQFRM1
```

Therefore:

```text
Python V1 == Rust V1
```

may establish historical cross-language receipt-core equivalence, but:

```text
Python V1 == Rust V1
-/-> #2469 normative framing compliance
```

## Required consequence

The historical receipt core must not be marked a normative landing primitive merely because its Python and Rust implementations agree.

Before becoming normative it must either:

1. migrate into a new explicit receipt-core schema/domain whose semantic identity is framed through the #2469 protocol, with an exact V1 -> V2 migration witness; or
2. remain HistoricalOnly / CompatibilityAdapter while a separately defined normative receipt core is created under #2469.

Do not reuse the V1 schema/domain for different bytes.

Historical V1 receipt IDs remain historical V1 identities and are never silently reinterpreted.

## Migration prerequisites

A V1 -> normative successor bridge should require exact evidence for:

- self-consistent historical V1 receipt identity;
- normative subject/profile/recipe semantic projections;
- normative input closure;
- verified qualification environment realization;
- exact attempt-subject migration for selected recipes;
- support-closure / receipt-candidate relation required by the source V1 construction;
- explicit treatment of every material field.

Unknown correspondence fails closed.

## Claim ceiling

Neither historical cross-language agreement nor a future framing migration establishes provider authenticity, trusted execution by itself, scientific validity, current admission, domain claim interpretation, merge authority, deployment authority, or physical authority.

#3748 owns the executable migration program.
