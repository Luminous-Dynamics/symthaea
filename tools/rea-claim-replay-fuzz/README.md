# REA claim replay fuzz qualification

This is a standalone cargo-fuzz workspace. It is not a production dependency and does not change claim evaluation semantics.

## Targets

- claim_request: feeds arbitrary bounded bytes to the raw claim request parser.
- claim_result: uses a fixed valid no-premise claim and feeds arbitrary bounded bytes to the result parser.
- framed_replay: parses a small ASCII length-framed envelope and invokes the full replay API with bounded request, result and Evidence slices.

## Frame format

The framed target accepts:

1. Decimal claim byte length, newline, then exactly that many claim bytes.
2. Decimal result byte length, newline, then exactly that many result bytes.
3. Decimal Evidence record count, newline.
4. For each record, decimal Evidence byte length, newline, then exactly that many Evidence bytes.

The format is deliberately inspectable so public seeds can be reviewed without a binary generator. It rejects noncanonical decimal lengths, overflow, truncation, excess record counts and trailing bytes before production replay is called.

## Bounds

- Max fuzzer input: 1 MiB.
- Request: 1 MiB.
- Result: 4 MiB.
- Evidence aggregate: 128 MiB (the frame cap means the fuzzer currently exercises at most 1 MiB).
- Evidence records per framed fuzz input: 8.
- Production verifier record cap: 4,096.
- Per-target PR smoke: 45 seconds.
- Per-target scheduled/manual campaign: 120 seconds.
- Per-input timeout: 5 seconds.
- Runner RSS ceiling: 512 MiB.
- Fuzzer: cargo-fuzz 0.13.2.
- Toolchain: nightly-2026-11-03.

## Evidence and limits

Fuzz runs record exact checked-out commit, toolchain, fuzzer version, lockfile digest, corpus/seed digests, runner OS, duration, executions per second, instrumentation output, peak RSS and crash artifacts.

A no-crash campaign is only bounded evidence about the exact target build, corpus, runner and time budget. It is not parser certification, proof of semantic correctness, or proof of absence of vulnerabilities. Every crash or sanitizer finding must be minimized and retained as a regression before closure.
