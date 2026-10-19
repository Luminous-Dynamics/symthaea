# Marine voyage trace benchmark (V0)

This small Rust crate converts a canonical synchronized trace of propulsion and auxiliary electrical input power into separate and total energy bands. It is the data-collection bridge for [MARINE-TRACE-001](https://github.com/Luminous-Dynamics/symthaea/issues/8102) and is intended to compose with [MARINE-ENERGY-BUDGET-001](https://github.com/Luminous-Dynamics/symthaea/issues/8092).

## Run

    cargo test --locked --manifest-path crates/tools/marine-voyage-trace-bench/Cargo.toml

    cargo run --locked --manifest-path crates/tools/marine-voyage-trace-bench/Cargo.toml --example synthetic

The example uses synthetic values only. No result from it is physical evidence.

## What it does

- Requires synchronized propulsion and auxiliary electrical input power in watts at one declared energy-boundary convention.
- Requires start/end coverage, strictly increasing integer-millisecond timestamps, a declared maximum sample gap, a source-trace SHA-256 identifier, mission/condition identifiers, a SHA-256 digest of the delivered-service contract, a meter calibration-epoch identifier, and an explicit absolute-bound method identifier.
- Integrates between adjacent samples using the trapezoidal rule (piecewise-linear interpolation).
- Carries nominal and conservative lower/upper watt-hour totals for propulsion, auxiliaries and the whole electrical boundary.
- Returns Indeterminate for missing channel measurements, incomplete endpoints, or gaps larger than the declared limit; returns Invalid for malformed metadata, invalid numbers or nonmonotonic timestamps.

The trace digest is only a caller-supplied identifier. This crate does not authenticate the source file or sensor, and it does not correct measured data for wind, waves, current, depth, draft or loading.

## Energy and uncertainty contract

For consecutive sample powers P0 and P1, duration delta-t in milliseconds, energy is:

    E_Wh = ((P0 + P1) / 2) * (delta-t / 3,600,000)

The absolute per-sample bound is propagated by integrating its lower and upper power envelopes. Lower power is bounded at zero because this schema represents nonnegative electrical consumption. These are declared arithmetic bounds, not probability distributions or calibrated confidence intervals. Correlation, unobserved transients and sensor bias are not inferred by the crate.

A missing auxiliary channel is never silently interpreted as zero. An over-limit gap is never integrated across. The maximum gap must be set from the acquisition plan before looking at results.

## How this fits marine validation practice

The current ITTC procedure for ship speed/power trials is [7.5-04-01-01.1, Revision 08 (effective 2024)](https://www.ittc.info/media/11972/75-04-01-011.pdf). ISO 15016:2025 defines preparation, execution, reporting and correction of speed-trial data for displacement ships of 50–500 m length between perpendiculars ([official ISO record](https://www.iso.org/standard/82080.html)). ISO 19030-1:2016, reviewed and confirmed current in 2022, provides general principles and indicators for measuring hull/propeller performance changes on the same ship over time ([official ISO record](https://www.iso.org/standard/63774.html)).

This crate implements none of those standards. It supplies only a strict trace arithmetic/quality contract. Raw voyage traces, corrected speed-trial records, simulation output and synthetic fixtures must remain separately labelled. Equivalent mission, condition, energy-boundary and uncertainty semantics are required before energy totals can be compared using issue #8092.


## Matched-trial eligibility (paired_trial module)

A follow-on comparator adapter can validate whether a baseline run and a candidate run are eligible for the existing mission-energy comparison. It requires matching campaign/pair identities, mission, delivered-service digest, environment/condition digest, electrical measurement boundary, meter calibration epoch, and uncertainty-method identifier. Candidate/control identities must differ, both traces must be complete, and run order must be explicitly recorded.

Campaign audit reports eligible and ineligible pairs, complete/indeterminate/invalid run counts, and baseline-first versus candidate-first pair counts. It always preserves unmatched runs. Seeing both order directions is only an order-coverage diagnostic; it does not prove randomization, causal attribution, sufficient sample size, or calibrated statistical uncertainty. A valid pair means only EligibleForEnergyComparison, not a saving or validated engineering result.

The pairing contract is not a correction method. It does not alter trace values to compensate for wind, waves, current, water depth, trim, displacement, hull fouling, route drift or trial-order effects. Choose an applicable external procedure (for example ISO 15016:2025 for displacement ships in its stated scope) and keep corrected records distinct from raw traces.


## Frozen multi-operating-point envelope audit

The `operating_envelope` module checks evidence coverage for a predeclared set of operating cases. Each case declares the mission ID, service-contract digest and exact external condition/profile identity. The envelope manifest declares its digest, campaign, electrical measurement boundary, meter calibration epoch and uncertainty-bound method. Every required case must have exactly one eligible paired-trial result, and all cases must compare the same baseline design ID against the same candidate design ID.

The audit rechecks that each energy pair matches the frozen mission/service/condition identities, measurement boundary, calibration epoch, uncertainty method and distinct trial order. It reports each missing, duplicate, unexpected, incomparable or fingerprint-mismatched case. A single favorable condition cannot compensate for missing evidence at another required point. The audit does not compute a saving, combine unlike operating points into one weighted score, select a design, establish a robust Pareto relation, or validate the physics. Feed the corresponding matched traces into the existing energy-budget comparator and retain each case's result separately for the shared robust-frontier analysis.

Suggested operating cases depend on the vessel class and mission. A profile may include low-speed control, transit/cruise, required maximum service speed and relevant environmental/loading corners, but it must not blindly impose every regime on every craft. Freeze the required cases before reviewing candidate outcomes. Digests are caller-provided identifiers; they are not proof that a source, geometry or environmental measurement is authentic.


## Fidelity-aware experiment ledger

The `fidelity_ledger` module accounts for analytical, low-fidelity simulation, high-fidelity simulation and physical-test attempts as distinct **declared evaluation classes**. It does not assert that one class is universally more accurate: the frozen profile states the required class, evidence origin, method ID and exact revision for each candidate / operating case. A low-cost result cannot satisfy a profile requirement for another class or method.

Every attempt is a separate receipt. Costs use positive integer units from one frozen `cost_basis_id`; unknown cost is not zero, and failed / censored / out-of-domain attempts still count toward campaign cost if their cost is known and comparable. Missing/zero/mismatched costs make the total campaign cost unauditable. Configuration digests must remain identical for a candidate across its required operating cases. A complete evaluation also requires an output digest.

Source ID, source-reference digest and explicit rights status are recorded. Only evidence marked permitted for the declared use can satisfy a release requirement; restricted or unknown rights do not become usable merely because a simulation or experiment succeeded. Hashes are identifiers, not proof of origin/authenticity or a license.

A requirement with `required_successes = 0` declares an allowed/ledgered screening tier that is not itself a release gate. A positive count is the number of distinct complete receipts required for that exact candidate/configuration/case/class/origin/method revision. No inferred fidelity hierarchy, posterior probability, value-of-information score, design ranking or physical qualification is produced.

This ledger composes with [MARINE-EXP-001 #7641](https://github.com/Luminous-Dynamics/symthaea/issues/7641) and shared [SE-OPT-BENCH-001 #7803](https://github.com/Luminous-Dynamics/symthaea/issues/7803). When evaluating policies, compare at the same end-to-end budget and include failed calls plus the later high-fidelity or physical-validation cost needed to release a low-fidelity-selected candidate. This is motivated by the 2026 multi-fidelity consequential-improvement study, which explicitly treats accuracy, validity, cost, and deferred high-fidelity validation cost as part of evaluation selection: https://link.springer.com/article/10.1007/s11081-026-10078-5. The 2026 constrained-multi-fidelity study is methodological prior art only; its aircraft examples do not establish marine benefit: https://arxiv.org/abs/2603.28987.

## Marine reference-data boundaries

The official [ITTC benchmark repository](https://www.ittc.info/benchmark-repository/) lists separate reference families for ship resistance/propulsion, manoeuvring predictions, and cavitation/noise round-robin work. A fidelity receipt must bind its specific source/method/revision and relevant operating case; a generic label such as "CFD" does not qualify a result across those different physical questions. The ITTC resistance/propulsion benchmark procedure is [7.5-03-02-02, 2024 Revision 03](https://www.ittc.info/search/?q=Self-Propulsion).

## Claim ceiling

A Complete result means only that the canonical trace passed this crate's metadata, time-coverage, numeric and integration gates. It does not validate the measurement system, prove a hydrodynamic improvement, show corrected speed-power performance, establish lifecycle emissions, demonstrate vessel safety, or certify compliance.


## Reference-data intake and leakage-resistant validation

The `reference_intake` module audits whether a declared reference source is ready for a model/data comparison. It requires inspected raw bytes and the exact artifact SHA-256, a documented rights decision, reviewed observable names/units/definitions, and a frozen split over operating-condition groups. It fails closed if calibration and hold-out groups overlap, groups are undeclared/unassigned, or the model was not declared frozen before hold-out access.

Two comparison dispositions are kept separate:
- `ReadyForUncertaintyAwareComparison`: measurement-uncertainty metadata is reported and reviewed.
- `ReadyForDescriptiveComparisonOnly`: data is otherwise admitted but the source does not report measurement uncertainty. This cannot support uncertainty-coverage claims.
- `Blocked`: missing/unreviewed/invalid intake or split metadata.

A passing result only permits a declared comparison to begin. Hashes do not authenticate the publisher or prove a license; the freeze flag is an attestation, not proof of blinding. This module does not download data, perform unit conversions, execute CFD, validate the model or establish physical performance.

### Content-bound SHA-256 contract

The intake audit now uses the maintained Rust `sha2` crate rather than a hand-written cryptographic primitive. The caller must provide the actual raw artifact bytes and model/configuration bytes; the audit recomputes their SHA-256 and rejects mismatches against the declared digests. The versioned canonical dataset manifest binds the raw-byte digest, citation/source revision, rights reference/status, uncertainty-status declaration, observable contracts, and operating-group inventory. The canonical split digest additionally binds that dataset manifest, model/configuration bytes, the frozen-before-holdout declaration, and normalized calibration/hold-out groups.

Length-prefixed fields prevent delimiter collisions. Set-valued observables and group lists are canonically sorted, so permuting equivalent input vectors does not change the digest; changing bytes, units, rights, source revision, model configuration or a split assignment does change it. The digest field itself is excluded from its own canonical preimage.

This still cannot prove publisher authenticity, legal rights, or that a team never inspected held-out results before declaring the model frozen. Those require independently reviewed evidence and access controls. SHA-256 establishes byte identity and manifest consistency, not truth.


### Candidate reference sources: discovery only

- The [ITTC benchmark repository](https://ittc.info/benchmark-repository/) is a directory of separate resistance/propulsion, manoeuvring and cavitation/noise source families; ITTC notes link maintenance and data property are not its responsibility. It is not itself a measured dataset.
- A 2026 paper on the [Duisburg Test Case in shallow water](https://doi.org/10.1080/09377255.2026.2644800) reports publicly available experimental resistance and self-propulsion data for depth-to-draught ratios 4.0, 2.0 and 1.2. The raw artifacts, hashes, reuse terms, units and uncertainty metadata have **not** been inspected by this crate and must remain blocked until reviewed.
- The [ITTC CFD-validation benchmark procedure](https://ittc.info/search/?q=Self-Propulsion) is a process/reference source, not a substitute for raw measurement data.

Do not tune a solver on the hold-out condition. Prefer grouped leave-one-operating-condition-out evaluation and report error separately for each observable and condition. If a data source omits uncertainty information, report descriptive residuals only; do not manufacture confidence levels. Unit-test fixtures are synthetic placeholders and never count as physical evidence.
