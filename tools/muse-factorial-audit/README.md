# Muse Factorial Audit Harness

This is a small, dependency-free Rust research workspace for the MUSE-CAUSAL-001 factorial response-surface protocol.

## Why it is isolated

The monorepo root Cargo workspace currently has an unresolved member-glob / missing-manifest failure. This tool declares its own workspace so its pure design and analysis code can be compiled independently of the root workspace. It does **not** exclude production crates, rewrite root workspace rules, or claim that the composer has executed.

## Protocol

The frozen preregistration is:

- `docs/research/MUSE_CAUSAL_FACTORIAL_PROTOCOL_V1.json`
- GitHub tracking issue: https://github.com/Luminous-Dynamics/symthaea/issues/12841

It specifies four three-level factors:

| Factor | Low | Centre | High |
|---|---:|---:|---:|
| Valence | -1 | 0 | +1 |
| Arousal | 0 | 0.5 | 1 |
| Energy | 0 | 0.5 | 1 |
| Requested bars | 8 | 16 | 24 |

There are 81 conditions in each style/seed block. Discovery covers all 29 executable styles and four fixed seed blocks (9,396 planned composer calls). Confirmation uses six preselected grammar-route styles and twelve disjoint seeds (5,832 planned composer calls). The combined plan contains 15,228 composer calls and is marked `preregistered_not_executed`.

The six confirmation styles were selected for route diversity: Classical / PeriodSentence, Fugue / Contrapuntal, Passacaglia / GroundVariation, Blues / BluesCallResponse, AfroCuban / GrooveCycle, and HindustaniInspired / RagaModalArc. Passacaglia is deliberately retained as a known zero-supported-axis negative-control profile. All 29 styles—including RenaissancePolyphony—remain in discovery.

## Analysis kernel

`src/lib.rs` provides:

- Deterministic case generation and stable case IDs.
- Exact materialized-case validation (missing, duplicated, unexpected or mutated cases reject).
- A strict 81-cell style/seed/metric block contract.
- 32 primary contrasts: eight orthogonal main effects and 24 pairwise interactions.
- Across-seed mean, sample standard deviation, minimum and maximum.
- 48 three-/four-factor higher-order diagnostic contrasts, kept separate from the primary terms.
- Adversarial tests for case coverage, duplicates, non-finite responses, row-order determinism and known synthetic main, pairwise and higher-order effects.

For a 3-level factor, the orthogonal contrast bases are linear `[-1, 0, 1]` and quadratic `[1, -2, 1]`. The pairwise contrast coefficient is:

`sum(y × w) / sum(w²)`

where `w` is the product of the one- or two-factor contrast weights. The kernel reports coefficients and seed dispersion; it deliberately does not fabricate p-values or confidence intervals. Terms of order three and four are diagnostics, not part of the primary comparison target.

## Run independently

```sh
cargo test --manifest-path tools/muse-factorial-audit/Cargo.toml

# Materialize the frozen case identities in deterministic CSV form.
cargo run --manifest-path tools/muse-factorial-audit/Cargo.toml --bin emit-cases -- discovery > /tmp/muse-discovery-cases.csv
cargo run --manifest-path tools/muse-factorial-audit/Cargo.toml --bin emit-cases -- confirmation > /tmp/muse-confirmation-cases.csv
```

The harness has no third-party crates. It can also be tested directly with `rustc --test tools/muse-factorial-audit/src/lib.rs` in a Rust environment. The case exporter emits stable IDs plus the four factor-level codes; it does not call the composer.

## Integration boundary

This is **not yet** the composer execution adapter. A later adapter must:

1. Materialize these exact cases into `MusicalIntent` values.
2. Call the pinned `compose_with_grammar_plan` route.
3. Run the existing symbolic validators and capture each required metric.
4. Bind each row to the exact style, seed, case ID, input, source revision and output digest.
5. Feed complete scalar metric blocks to `estimate_block`.
6. Keep generation errors, categorical outputs, not-applicable channels and missing metrics explicit.

The case generator and contrast engine are ready to be connected; no composer output, causal effect or musical-quality result is asserted by this repository addition.

## Evidence labels

- **Implemented:** standalone factorial case generator, exact coverage validator, contrast kernel and unit-test source.
- **Source-checked:** factor/style/seed choices and source blob digests in the preregistration.
- **Not executed:** Rust tests, composer calls, discovery sweep, confirmation sweep, rendering and listening study.


## Import and analyze composer results

After an adapter emits one outcome row per expected case and registered metric,
use the strict importer described in [RESULTS_SCHEMA.md](RESULTS_SCHEMA.md):

```sh
cargo run --manifest-path tools/muse-factorial-audit/Cargo.toml --bin analyze-results -- \
  discovery observations.csv note_density,mean_pitch coverage.csv effects.csv summary.csv
```

The metric list is explicit so wholly absent metrics are visible as 81/81 missing
within every expected block. Incomplete blocks remain in coverage reports and
produce no contrast estimates; the command exits non-zero after writing reports.
Missing, failed, and not-applicable values are never imputed as zero.


## Composer execution adapter

The next stage is the source-pinned symbolic composer adapter. It validates the
complete frozen case manifest, invokes each style's canonical grammar-plan
composition path, hashes each serialized score, and emits 18 symbolic metrics.
See [COMPOSER_ADAPTER.md](COMPOSER_ADAPTER.md) for the exact run commands,
metric registry, evidence boundary, and workspace prerequisite.

The adapter source is an example under
`crates/domains/symthaea-music-theory/examples/`; it cannot be compiled until
the separate root Cargo workspace member-exclusion repair is applied. No
composer execution or Rust test result is claimed by this PR.
