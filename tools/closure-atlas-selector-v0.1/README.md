# Closure Atlas Selector v0.1

**Status:** Pure Rust reference selector with 29 authored synthetic tests. No plasma closure, solver or learned router is implemented. Rust compilation and hosted CI remain pending exact-head validation.

## Purpose

The selector implements the first gate from Symthaea #12918: choose a specialist only when exactly one independently qualified profile matches the declared physical regime, measured parameter intervals, solver build and Runge–Kutta tableau.

A profile is eligible only when:
- qualification status is Qualified;
- model, source, solver, tableau, independent qualification and rights evidence digests have SHA-256 shape;
- dataset rights are Verified, identify a license and explicitly grant research execution;
- physics-model, regime, dimensionality, solver-build and tableau identities match exactly;
- the local parameter observation set exactly matches the profile's declared parameter envelope names and units;
- every conservative measurement interval lies wholly inside every inclusive profile envelope.

v0.1 uses integer-scaled parameter values and exact unit-string equality. It performs no floating-point boundary comparisons and no unit conversion. A later version may add dimensionally correct conversion only with explicit conversion evidence and tested canonicalization.

## Fail-closed outcomes

- **Selected:** exactly one fully eligible specialist and no unresolved possible competitor.
- **Unknown:** local descriptors or uncertainty are insufficient to prove eligibility. No model is selected.
- **Abstained:** no qualified profile matches; the state is demonstrably outside the declared profile set or a required gate fails.
- **Ambiguous:** multiple qualified profiles match. v0.1 does not invent a score or choose arbitrarily.

An observation interval that is entirely outside a profile is rejected. An interval that touches/crosses an envelope edge without being wholly inside is Unknown. This avoids selecting a model when uncertainty straddles its qualified boundary.

The decision report contains sorted candidate IDs, a per-profile eligibility decision and rejection reasons. The pure selector makes no external calls and does not create authority.

## Physical qualification evidence

A profile now declares five gate records, each binding a gate-contract digest and its evidence digest:
- conservation (the exact mass/momentum/energy balance convention for the declared equations);
- frame symmetry (the regime-appropriate Galilean, Lorentz or explicitly scoped transformation family);
- entropy admissibility (the exact entropy variable, production convention and sign/realizability condition);
- well-posedness (the applicable hyperbolicity/characteristic analysis);
- spurious-mode stability (the declared growing-mode/stability analysis).

Every gate must be explicit. Pass requires evidence; Fail prevents selection; Unknown yields no selection; NotApplicable requires both evidence and a non-empty rationale. NotApplicable is not a shortcut: the evidence must establish why the gate does not apply to this model/regime. A missing gate also returns Unknown.

The selector validates declarations and digest shapes only. It does not execute conservation, symmetry, entropy, hyperbolicity or spectral tests, authenticate the evidence issuer, or determine whether a NotApplicable rationale is scientifically defensible. The gate contract digest must bind regime-specific definitions; for example, Galilean and Lorentz transformations are not interchangeable. Do not impose a universal entropy sign rule without the corresponding model-specific thermodynamic derivation.

## Synthetic fixture

The JSON fixture is entirely synthetic and exists only to check the interchange schema. Its synthetic rights declaration is not a claim about any external dataset. No real plasma data, closure artifact or solver digest is included.

## Limits and claim ceiling

The selector validates caller-supplied declarations. It does not authenticate profile issuers or verify that hashes refer to real artifacts; classify physical regimes from raw fields; establish that the declared envelope is scientifically correct; train, infer or execute a closure; compare expert outputs; learn a router; switch models inside a PDE solver; prove switching stability; or establish fusion-device validity.

The independent referee and Symtropy #1685 run receipt remain external qualification/observation owners. Mycelix #5900 remains experiment-provenance owner. A passing unit test demonstrates selector semantics only.

## Run

    cargo fmt --manifest-path tools/closure-atlas-selector-v0.1/Cargo.toml -- --check
    cargo test --manifest-path tools/closure-atlas-selector-v0.1/Cargo.toml --locked
    cargo clippy --manifest-path tools/closure-atlas-selector-v0.1/Cargo.toml --locked --all-targets -- -D warnings
