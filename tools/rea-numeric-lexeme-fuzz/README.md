# REA raw numeric lexeme fuzz qualification

This is a standalone, test-only cargo-fuzz workspace for the raw token auditor in tools/rea-numeric-lexeme-audit. It is not a production dependency and does not change Evidence identity, JCS behavior or claim dispositions.

## Target

raw_numeric_lexemes feeds arbitrary bounded raw bytes to audit_json. For every accepted JSON document, the support layer repeats the audit to require deterministic output, checks stable warning ordering, then adds only outer JSON whitespace. Warning paths, original integer lexemes and visited-value counts must remain unchanged, while the exact-source SHA-256 must change.

Malformed, duplicate-key, over-limit and invalid JSON are ordinary typed parser outcomes. They must not panic.

## Bounds

- Fuzzer maximum input: 1 MiB (stricter than the audit's 64 MiB production limit).
- Auditor limits remain: 64 MiB input, nesting depth 128, 1,000,000 visited values, 100,000 warnings.
- Per-input timeout: 5 seconds.
- Runner RSS ceiling: 512 MiB.
- PR smoke: 60 seconds.
- Scheduled/manual run: 300 seconds.
- Toolchain: nightly-2026-11-03.
- cargo-fuzz: 0.13.2.

## Seeds

The corpus includes safe/unsafe integer boundaries, negative and large values, nested arrays, decimal/exponent/string controls and escaped-equivalent duplicate keys.

## Evidence

The workflow records exact source commit, compiler, cargo-fuzz version, both lockfile hashes, seed digests, runner OS, duration, execution count, throughput, peak RSS, output hash and crash artifacts.

A clean bounded run qualifies only this exact target build, seed corpus, runner and time budget. It does not prove parser security, exhaustiveness over all JSON inputs, semantic correctness or numeric truth. Every crash must be minimized and retained as a regression.
