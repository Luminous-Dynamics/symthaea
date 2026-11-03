# ETH-023: fail-closed MC input semantic contract auditor

**Status:** Python reference parser/auditor and finite fixture/fuzz tests committed. Rust port and differential comparison against official solver parsers remain open.

## 1. Why input semantics are part of exactness

Model counting, weighted model counting, projected model counting, projected weighted model counting and algebraic model counting are different outputs. Choosing the wrong algebra after silently misclassifying the input can produce a numerically precise answer to the wrong problem.

The MC 2026 tracks include #SAT, negative-weight WMC, PMC, mixed WMC/PMC/PWMC Track 4 and complex-field AMC Track 5B:
- https://mccompetition.org/2026/mc_description.html

The public Ganak 2026 submission reported a Track 4 implementation issue: it trusted the type header despite a mixed benchmark set, which caused some instances to be misrouted:
- https://github.com/meelgroup/mccomp-ganak-appmc-2026

The official MC data format v1.2 requires the solver to handle the type line or return an error; it also specifies exact weight/projection directives, literal complement rules and warnings for weights on non-projection variables:
- https://mccompetition.org/assets/files/mccomp_format_25.pdf

## 2. A real semantic ambiguity: missing projection set

The v1.2 format PDF states that in PMC, listing every variable using show denotes ordinary model counting; listing no variables is a satisfiability-decision query. Ganak's current README instead says that an absent projection set means all variables are projected:
- https://github.com/meelgroup/ganak

These two policies produce different results. The auditor therefore has three explicit profiles:
- strict: missing PMC projection set remains ambiguous;
- mccomp_v1_2: follows the published 2025 PDF decision-query interpretation;
- ganak_2026_compat: follows Ganak's current documented all-variables projection default.

The profile is required input to semantic resolution. The parser does not choose one interpretation silently. The PWMC profile requires an explicit projection set in this initial auditor.

The result preserves both the raw declared mode and an explicit `effective_output_contract`. For PMC with no `show`: strict mode returns `ambiguous` and no effective contract; the v1.2 profile resolves to `sat_decision`; the Ganak compatibility profile resolves to `projected_model_count` with all variables projected. An explicit partial projection resolves to `projected_model_count`, while showing every variable resolves to `model_count`. Downstream execution must route on the effective contract, never the raw header alone.

## 3. Implemented reference

File: reference/mc_input_contract_auditor.py
Results: reference/mc_input_contract_results.json

The parser handles:
- DIMACS p cnf header and exact clause-count validation;
- one type declaration from mc, wmc, pmc, pwmc/wpmc or amc-complex;
- projection declarations, potentially spread across multiple show lines;
- exact rational decimal/fraction weights;
- exact complex rational coefficients a+bi;
- literal/domain bounds, duplicate/conflicting metadata, clause termination and unsupported directives;
- weight complements and effective defaults;
- separate status values valid, invalid, ambiguous and unsupported, with independent warning/error/unresolved channels.

Effective weight rules:
- WMC/PWMC variables without explicit literal weights default to 1 for both polarities.
- If one literal receives 0<w<1 and the opposite is absent, the v1.2 complement rule infers 1-w exactly.
- A nonpositive literal weight without its opposite is rejected.
- A weight >=1 without its opposite is left unresolved rather than guessed.
- An AMC variable with no weights defaults to 1+0i for each polarity; if one polarity is supplied, the other is required.
- Weights declared for non-projection variables under PWMC produce a warning per the format profile.

## 4. Executed finite tests

Deterministic seed: 0xE7023.

- 34/34 fixture expectations passed.
- 2,000 random malformed/mixed-input strings were audited without an uncaught exception.
- Strict, mccomp_v1_2 and ganak_2026_compat profiles exercised.
- Mutation cases include duplicate/conflicting headers, header order, unsupported directives, invalid literals/variables, wrong clause counts, malformed rational/complex weights, polarity-pair rules and mode/directive contradictions.
- Effective-weight checks verify exact 1/4 -> 3/4 complement inference, all-one WMC defaults and complex AMC defaults.

These are reference parser tests, not a complete official format parser, official fixture match, Ganak differential run, Rust implementation, solver, or performance result.

## 5. Industry benchmark and acceptance plan

Compare parser decisions against:
1. official MC 2026 public instances (all five tracks);
2. Ganak's input parser for the identical raw input;
3. the official format's versioned requirements.

Do not make Ganak the sole semantic oracle: it is a necessary implementation baseline, but its own submission documents a Track 4 routing issue and its README's no-show PMC default differs from the 2025 format text.

Record:
- accepted/rejected/ambiguous/unsupported classification;
- effective weights and projection set;
- parser runtime;
- all warnings and errors;
- whether the downstream solver output contract matches the input semantics.

Acceptance:
- zero ambiguous or contradictory inputs accepted as a valid executable contract;
- 100% agreement on frozen official/adversarial fixtures after the selected format profile is fixed;
- no silent fallback for unsupported exact coefficient syntax;
- Rust port passes the same golden corpus;
- parser-time overhead reported separately from solving.

## 6. Claim ceiling

This implementation is a small semantic auditor. It is not yet validated against the complete official MC2026 corpus, it has not been differentially run against Ganak, and it does not solve or count any instance.
