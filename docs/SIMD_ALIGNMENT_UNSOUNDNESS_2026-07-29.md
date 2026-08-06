# Unsound alignment assumption in `symthaea-core::hdc::simd_ops`

**Severity: memory-safety defect (undefined behaviour) in the workspace's most-depended-on
crate.** Reproduced as a SIGSEGV; demonstrated concretely without writing any `unsafe` code.

**Status: FIXED 2026-07-29.** 28 aligned loads → unaligned; 24 misaligned `u64` dereferences →
`read_unaligned`; regression test added and proven to fail against the unfixed code.
Measurements below.

Found 2026-07-29 while trial-extracting the HDC seam
(`SYMTHAEA_CORE_EXTRACTION_ANALYSIS_2026-07-29.md`). The extraction compiled cleanly; the bug
surfaced when the extracted crate's own tests were run.

## The defect

The AVX2 paths in `simd_ops.rs` use **aligned** load intrinsics:

```rust
// simd_ops.rs:103
unsafe fn bind_avx2(a: &[u8; 2048], b: &[u8; 2048], result: &mut [u8; 2048]) {
    let a_ptr = a.as_ptr() as *const __m256i;
    let b_ptr = b.as_ptr() as *const __m256i;
    // Source arrays are align(32) from BinaryHV, so use aligned loads
    let a0 = _mm256_load_si256(a_ptr.add(i));   // requires 32-byte alignment
    let b0 = _mm256_load_si256(b_ptr.add(i));
```

`_mm256_load_si256` faults unless its operand is 32-byte aligned. The comment's justification is
real but does not hold at the type level:

- `BinaryHV` **is** `#[repr(align(32))]` (`binary_hv.rs:51`), so `&some_binary_hv.0` is aligned.
- But these functions take **`&[u8; 2048]`**, whose alignment is **1**. The invariant lives in a
  *container type that is absent from the signature*.

The invariant is therefore lost the moment a bare array is produced — and the crate's own API
produces one:

```rust
// simd_ops.rs:630 — returns a BARE [u8; 2048]; the align(32) guarantee does not travel with it
pub fn invert_simd(a: &[u8; 2048]) -> [u8; 2048] { … }
```

## Reproduction

`hdc::simd_ops::tests::test_simd_inverse_properties` — a test that already exists in
`symthaea-core` — performs exactly this sequence:

```rust
let a   = BinaryHV::random(42);
let inv = invert_simd(&a.0);          // bare [u8; 2048]  → alignment 1 by type
let xor = bind_simd(&a.0, &inv);      // → _mm256_load_si256 on &inv → SIGSEGV
```

Observed in the trial crate:

```
signal: 11, SIGSEGV: invalid memory reference
```

Isolated with `--test-threads=1`; the neighbouring `test_bundle_simd_matches_scalar` passes and
was a red herring from parallel output interleaving.

### Demonstration without `unsafe`

A probe that only *measures* the alignment of values the public API returns:

```
BinaryHV field &a.0 -> alignment  64   (repr(align(32)) applies)
invert_simd()  &inv -> alignment  16   (bare [u8; 2048] == align 1)

bind_simd(&a.0, &inv) issues _mm256_load_si256 on a 16-byte-aligned pointer.
That instruction requires 32-byte alignment. This is the SIGSEGV.
```

The 16 is incidental — the compiler happened to place the local there. That is the nature of the
defect: **it is layout-dependent UB**, so it can pass in one build and fault in another. It is
not a bug that manifests reliably, which is precisely why it has survived.

## Scope

`grep -c "_mm256_load_si256\|_mm_load_si128\|_mm256_store_si256\|_mm_store_si128" simd_ops.rs`
→ **28 sites**, spanning at least `bind_avx2` (l.113), the Hamming path (l.242), the
matching-bits path (l.380) and `invert_avx2` (l.655). Every one of them takes `&[u8; 2048]`
rather than `&BinaryHV`, so none is protected by the alignment attribute its comment cites.

Not all call sites are unsound. Passing `&binary_hv.0` is fine. Only paths that route a bare
array back in are affected — but nothing in the type system distinguishes the two, and the
crate's own test suite exercises the unsound one.

## Confirmed: the same test passes in `symthaea-core` and faults in the extracted crate

This was measured, not inferred:

```
$ cargo test -p symthaea-core --lib hdc::simd_ops::tests
test result: ok. 14 passed; 0 failed; 2 ignored          # test_simd_inverse_properties PASSES

$ cargo test -- --test-threads=1                          # extracted crate, same source files
test hdc::simd_ops::tests::test_simd_inverse_properties ... SIGSEGV
```

**Identical source. Opposite outcomes.** In `symthaea-core` the local happens to land on a
32-byte boundary and the aligned load succeeds; under the extracted crate's codegen it lands on
16 and faults.

The consequence worth stating plainly: **for this defect, a green test suite is not evidence of
soundness.** `symthaea-core` has run this test successfully for as long as it has existed, and
that told nobody anything, because the failure mode is a property of memory layout rather than
of program logic. Any future refactor — reordering fields, changing an inlining decision, adding
a local — can flip it, in either direction, with no source change to the SIMD code at all.

This is also an argument *for* extraction beyond compile times: a smaller crate changes layout
and surfaces assumptions that a monolith's incidental arrangement was hiding.

## The fix as applied

All 28 `_mm256_load_si256` → `_mm256_loadu_si256`.

**A second instance of the same false premise was found while fixing the first.** The two
`matching_bits_*_popcnt` functions cast `&[u8; 2048]` to `*const u64` and dereference, under a
`SAFETY:` comment making the identical `repr(align(32))` claim. `u64` reads need 8-byte
alignment; the parameter guarantees 1. x86 tolerates misaligned integer loads in hardware so this
one would not fault, but it was UB all the same. 24 dereferences rewritten to `read_unaligned()`
(identical instruction on x86, no precondition). All four stale comments replaced.

### Regression test, proven to have power

`simd_entry_points_accept_deliberately_misaligned_inputs` constructs a buffer at exactly
`16 mod 32` — legal for the parameter type — and asserts that precondition before proceeding, so
it cannot silently decay into a test that passes only because a local landed well.

| | result |
|---|---|
| test against **unfixed** code | **SIGSEGV** |
| test against fixed code | **15 passed, 0 failed** |

### Performance: measured, and indistinguishable

Two copies of the trial crate differing only in the load intrinsics; the crate's own
`bench_simd_vs_scalar`, three `--release` runs each with `target-cpu=native`:

| arm | bind (ns) | similarity (ns) |
|---|---|---|
| aligned (before) | 850, 311, 359 | 343, 391, 256 |
| unaligned (after) | 456, 403, 433 | 347, 445, 310 |

**Within-arm spread exceeds the between-arm difference in both metrics.** The honest reading is
that this measurement cannot resolve a difference between aligned and unaligned loads here — not
that the fix is provably free. Host load was 39–49 with 16 concurrent sessions; a quiet-machine
measurement would be needed to say more, and nothing about this fix depends on saying more.

### Unrelated pre-existing finding, NOT addressed

The benchmark's own assertion `SIMD bind should be faster than scalar` **failed in all six runs,
in both arms**: `bind_simd` measured 2–4× *slower* than the scalar path. That is independent of
this fix (both arms fail it) and plausibly correct behaviour — LLVM auto-vectorises the scalar
XOR well, and this file already records the same conclusion for `bundle`
("Hand-written AVX2 intrinsics ... do NOT outperform well-optimized scalar code"). The benchmark
is `#[ignore]`d, so this assertion has never run in CI. Worth a dedicated look on a quiet
machine; deliberately out of scope here.

This follows the codebase's own convention rather than a preference: `simd_continuous.rs`
already uses `_mm256_loadu_ps`/`_mm_loadu_ps` at 53 sites, and `simd_ops.rs` already used
`_mm256_storeu_si256` for its stores and `_mm512_loadu_si512` on the AVX-512 path. The aligned
loads were the outlier in a file that had otherwise settled on unaligned.

A stricter alternative is to change the signatures to `&BinaryHV`, making the alignment
invariant type-enforced. That is more invasive (it touches every caller) but eliminates the
class of defect rather than this instance, and is worth considering if these functions are
meant to stay `pub`.

**Do not fix by adding a debug assertion.** The path is UB in release builds, which is where it
matters.

## Reproducing

The trial crate is a scratch artifact, not checked in. To rebuild it: copy
`hdc/{unified_hv,binary_hv,hdc_ltc_unified,simd_continuous,simd_ops,simd_detect}.rs` and
`genesis.rs` into a fresh crate preserving the `crate::hdc::…` / `crate::genesis::…` module
paths, add `serde`/`serde_arrays`/`rand`/`sha3`/`blake3`, then
`cargo test -- --test-threads=1`.
