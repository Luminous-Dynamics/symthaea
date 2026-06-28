# Alpha.6 Test Fix

Alpha.6 fixes a narrow regression discovered by running the crate tests inside the real Symthaea workspace.

## What failed

`tests::encode_decode_packet` asserted that PSNR must be finite.

For a perfect reconstruction, mean squared error is zero. In the standard PSNR definition, zero MSE yields positive infinity, not a finite value.

That means the codec path was behaving acceptably; the test assertion was too strict.

## Fix

The test now rejects `NaN`, accepts `+inf`, and otherwise requires a high reconstruction PSNR for the full-coefficient 8x8 round trip.

A dedicated regression test documents the invariant:

```rust
psnr_identical_images_is_infinite
```

## Correct verification command

From the Symthaea workspace root:

```bash
cargo test -p symthaea-visual-compression-probe
```

Avoid:

```bash
cargo test symthaea-visual-compression-probe
```

That form filters test names and can compile unrelated workspace targets.
