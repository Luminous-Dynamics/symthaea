# Alpha.8 Scope Fix

Alpha.7 introduced a regression test named `packet_hash_survives_text_roundtrip` to confirm that packet hashes survive `.svmp` text serialization.

The test accidentally called `fixture_image()`, which was not defined inside the crate unit-test module. Alpha.8 fixes this by using the existing `tiny_image()` helper already used by the rest of the unit tests.

## Expected Verification

```bash
cargo test -p symthaea-visual-compression-probe
```

This release changes only test code and documentation. Runtime codec behavior, packet format, CLI behavior, and fixture packets are unchanged from alpha.7.
