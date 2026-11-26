# Repair Locality Reference

This zero-dependency Rust crate contains two deliberately separate pieces:

1. A deterministic access-cost scorer for a supplied repair helper plan.
2. A small synthetic [7,4] locally repairable code matrix for finite algebra checks.

The access scorer does **not** prove that a helper set reconstructs the requested bytes. A repair can be accepted only when an independent algebra verifier reports success, all helpers are available, and the reconstructed bytes match the scorer-held expected digest. The current crate records an explicit independent-algebra result rather than implementing every external ECC family.

## Synthetic local-repair matrix

Data rows:
- D0 = [1,0,0,0]
- D1 = [0,1,0,0]
- D2 = [0,0,1,0]
- D3 = [0,0,0,1]

Parity rows:
- P0 = D0 + D1
- P1 = D2 + D3
- G = D0 + 2D1 + 4D2 + 8D3 over GF(256), polynomial 0x11d.

Local groups are (D0,D1,P0) and (D2,D3,P1), with G as a global parity row. The intended repair of one data symbol reads its peer plus local parity (two helpers). The global row supports selected multi-erasure cases.

The exact matrix is checked by exhaustive row-rank enumeration:
- every one of the 21 two-erasure patterns leaves rank 4;
- at least one three-erasure pattern is unrecoverable.

This is a synthetic [7,4,3]-style candidate, not a published or standards-conformant LRC implementation. It has 3 parity symbols for 4 data symbols (75% parity/data overhead), compared with the custom MDS [6,4] reference's 2 parity symbols for 4 data symbols (50%). It therefore trades lower single-data repair reads for additional storage overhead. The fixture does not claim an equal-rate result.

## Run locally

```bash
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test
```

This session's environment has Python 3.13.5 but no Cargo/rustc/rustfmt, so these commands have not been executed here. The tests are authored only.

## Metric definitions

- read amplification = total helper bytes read / requested bytes repaired;
- helper read count = number of source fragments read;
- unique regions = unique region IDs among the ordered helper reads;
- layer transitions = adjacent helper reads whose declared layer differs.

These are abstract counters. They are not physical seek, optical FOV, stage travel, tape mount, HDD rotation, SSD page or DNA sequencing costs. Those require medium-specific qualified adapters.
