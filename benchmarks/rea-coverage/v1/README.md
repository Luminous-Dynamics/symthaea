# ENG-REA-003 v1: extraction-coverage integrity contract

**Status:** public development known-answer corpus; local contract verifier passed.  
**Parent benchmark:** [ENG-REA-002 #8057](https://github.com/Luminous-Dynamics/symthaea/issues/8057)  
**Integration boundary:** [ENG-REA-001 #8022](https://github.com/Luminous-Dynamics/symthaea/issues/8022)  
**Upstream candidate fix:** [morluto/rea#4953](https://github.com/morluto/rea/pull/4953), still an open draft when this corpus was authored.

## Problem

A reverse-engineering report must not label a nonempty selected input `coverage: complete` when no chunk accounts for its bytes and no gap reports them. A second failure mode is to find a missing span but map it to the wrong original-file offset when the caller selected a subrange. These are evidence-integrity defects: a downstream diagnoser could mistake missing analysis for negative evidence.

An upstream source audit on 2026-10-18 found that `morluto/rea` `src/firmware/FirmwareReports.ts` at blob `f06129aa2894e1a03a5ad573c90e432cbddd1712` calculated coverage from depth limits, diagnostics, unknown handlers and encrypted chunks, without testing the complement of reported selected-input chunk ranges. Upstream draft PR #4953 documents the zero-chunk reproduction and proposes an `unreported-input-region` gap. This Symthaea corpus adds a downstream-independent contract; it does not assume that PR is merged or provider-tested.

## Contract

All chunk and selected-input gap offsets use zero-based half-open ranges relative to the selected input. `root_file_range.offset` maps the same bytes back to the original source and must equal `selection_offset + range.offset`.

For a nonempty selection of length `L`:

1. Sort chunk intervals and compute their union within `[0, L)`.
2. Every uncovered byte interval in the complement must appear exactly once as an `unreported-input-region` gap. No covered byte may be reported as unreported.
3. Every such gap must preserve its length and map to the correct original-file offset.
4. Coverage is `partial` if any selected bytes are unreported, any extraction gap or partial signal exists, or any chunk has an unknown handler or is encrypted. A zero-length selection is invalid. Negative, fractional, empty-length, or out-of-bounds chunk ranges are malformed evidence and must fail closed.
5. `complete` means only that this report's declared extraction conditions have no gaps or partial signals. It does **not** prove that firmware was fully reconstructed, that all formats were recognized, or that runtime behavior is equivalent.

## Cases

`cases.json` holds 16 normalized-report examples; `oracle.json` separately records expected dispositions. The set covers whole-input and selected-subrange zero-chunk false-completeness, out-of-order adjacent and overlapping chunks, an interior uncovered span, correctly mapped gaps, unknown/encrypted chunks, depth-limit signals, missing-extractor diagnostics, empty selections, out-of-bounds chunks, mismatched root offsets and spurious gaps.

| Target | Acceptance gate |
|---|---|
| False completeness on nonempty zero-chunk input | 0 cases accepted |
| Missing selected-input bytes | Exact complement recovered, 100% |
| Source-range mapping | 100% exact root offsets and lengths |
| Invalid input, fractional byte ranges and malformed chunk ranges | 100% fail closed |
| Full known-chunk coverage | No spurious `unreported-input-region` gap |
| Determinism | Same manifest + oracle yields the same case dispositions and file digests |
| Small-domain exhaustive coverage | All 510 byte masks for lengths 1–8 agree with an independent bytewise-gap oracle |
| Interval representation invariance | All 33,866 interval subsets for lengths 1–5 pass in original, reversed and duplicated forms (101,598 representations) |

These are contract targets and the current public fixture results, not measurements of a real REA provider.

## Run

Requires `jq` and `sha256sum` only; it does not require Node, Binwalk, Unblob, a decompiler, or a repository build.

```sh
benchmarks/rea-coverage/v1/verify.sh
benchmarks/rea-coverage/v1/property-sweep.sh
benchmarks/rea-coverage/v1/representation-sweep.sh
```

Recorded fixed-corpus result: 16/16 case dispositions matched the oracle: 5 internally consistent reports accepted, 8 inconsistent reports rejected, 1 empty input marked invalid, and 2 malformed reports rejected. The independent `property-sweep.sh` passed all 510 coverage masks for selected-input lengths 1–8, with source-offset checks. `representation-sweep.sh` then exhaustively checked all 33,866 possible subsets of byte intervals for lengths 1–5 in three representations each: original order, reversed order, and duplicate intervals. All 101,598 representations matched the independent bytewise gap-run oracle. Exact tested file identities and run limits are recorded in `receipt.json`.

## Next evidence gate

Run the pinned upstream REA fixture/oracle lane against the actual Unblob version with both an unrecognized whole input and an unrecognized selected subrange. Compare the current-main behavior with the candidate fix in #4953, checking CLI and MCP output parity, exact selected/root ranges, and source-byte immutability. Only after that should a Rust-native Symthaea importer bind this output to input digests, actual executable/tool identity, version, limitations and explicit partial coverage.

**Not done here:** no REA process or provider, Binwalk, Unblob, Ghidra, emulator, recovered/vendor firmware, MCU, physical device, or hardware-write path was run. The 16-case result and the exhaustive sweeps are executable checks of this synthetic normalized-report contract only. The oracle is public, so it is a regression corpus rather than a blind leaderboard.
