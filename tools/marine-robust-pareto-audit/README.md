# Marine robust Pareto audit

A standalone Rust decision audit for comparing propulsion candidates on a frozen cargo-service functional unit. It applies non-compensable hard gates before interval-robust Pareto comparison. It does not produce a weighted score.

## Run

~~~sh
cargo test --manifest-path tools/marine-robust-pareto-audit/Cargo.toml
cargo run --manifest-path tools/marine-robust-pareto-audit/Cargo.toml -- \
  docs/research/marine/fixtures/synthetic-marine-robust-pareto-v0.1.0.json
~~~

The synthetic fixture must return INCOMPLETE and must not qualify a candidate for the robust frontier.

## Whole-vessel objective vector

The integrated scenario compares eight separate objectives on the frozen cargo-work unit: primary energy, well-to-wake GHG, underwater noise at 1 kHz, 2–5 kHz and 30–80 kHz, lifecycle cost, maintenance hours and availability. The three acoustic bands remain distinct; no broadband aggregate or scalar weighted score hides band-specific degradation. The sample 30–80 kHz limit and 95% availability floor are scenario-specific test values, not universal regulatory limits.

## Decision rules

- Candidate service must match the frozen mission, vessel, route, measurement window, service/environment fingerprints and cargo-work unit.
- Candidate cargo-work interval must fit the declared service equivalence tolerance.
- Each objective metric must be present for every candidate, with matching units, measurement method and expanded-uncertainty confidence level across the comparison.
- Every candidate metric carries method, raw data, calibration/validation, timebase and operating-point references.
- Every hard gate is non-compensable. Metric upper-bound gates pass only if the whole interval is below the threshold, fail only if the whole interval is above, and otherwise hold as unresolved. Lower-bound gates use the symmetric interval rule. External biosecurity gates consume an exact required audit contract/requirement receipt.
- A failed hard gate excludes a candidate regardless of energy or cost. An unresolved/missing gate holds the candidate and never contributes it to the Pareto frontier.
- For a minimizing objective, A is certainly no worse only if A's upper interval bound is <= B's lower bound. For maximizing objectives, A's lower bound must be >= B's upper bound.
- Dominance requires certainty on every objective and strict separation on at least one. A confirmed trade-off is retained; overlapping/touching intervals do not create a ranking.
- The output is pairwise relations plus a robust nondominated set with uncertainty status. No single weighted score is generated.

## Integrated marine evidence profile

The included scenario now requires external non-compensable receipts for organism release, in-water cleaning authorization and captured waste, acoustic scale-transfer review, source-state contrast, cavitation intermittency, vibration–hydrophone coherence and shaft-rotation-synchronous modulation. These seven receipts supplement—not replace—the numeric acoustic limit and cargo-service/objective intervals.

The synthetic fixture marks each of these external receipts unresolved, and the dedicated workflow asserts that neither synthetic candidate enters the robust frontier. A receipt may be marked satisfied only when the exact audit contract, requirement reference and reviewed receipt digest match. A completed software audit is still not a physical performance or ecological-effect claim; the relevant measurement result and gate threshold must be separately declared.

## Marine environmental evidence gates

The scenario can require exact external audit contract/receipt pairs as non-compensable gates. The integrated example now registers:
- biofouling organism-release control;
- in-water cleaning authorization and waste capture;
- acoustic model-to-full-scale transfer review;
- controlled acoustic source-state contrast;
- cavitation intermittency distribution;
- vibration–hydrophone coherence characterization;
- shaft-rotation-synchronous modulation characterization.

These are evidence-quality gates, not automatic proof that a candidate is environmentally superior. A missing, unresolved or mismatched receipt holds the candidate out of the robust Pareto frontier, irrespective of energy or cost savings. The actual underwater-noise objective and biosecurity thresholds remain separate declared gates. Synthetic fixture receipts are placeholders for known-answer tests only; they do not represent physical evidence.

Each external gate receipt is bound to the exact candidate it qualifies: candidate ID, vessel revision, propeller revision, operating-point receipt digest, service fingerprint and environment fingerprint must match the candidate record byte-for-byte. This prevents a valid-looking pass receipt for one propulsion configuration or voyage context from being reused for another. These are binding checks over declared references; they do not authenticate the audit publisher or verify the raw receipt bytes.

## Interpretation limits

The audit tests declared comparison contracts and computes interval relations. It does not authenticate sensor certificates or raw bytes, validate physical measurements, acoustic propagation, ecology, CFD, full-scale transfer or regulatory compliance. A SHA-256 digest is a reference, not proof of authenticity.

A confirmed nondominated candidate is not automatically the best engineering choice; it is only not robustly dominated within the declared objective set and evidence envelope. Held candidates and unresolved pairwise comparisons remain explicit.

## Benchmark progression

B0 known-answer fixtures: interval arithmetic, hard-gate truth tables, Pareto dominance/trade-off and no false rankings under overlap.
B1 matched model-scale/bench trials with independently measured cargo work.
B2 cross-facility and held-out vessel comparison using the same service fingerprint and declared acoustic bands.
B3 ITTC Nawigator XXI Round Robin acoustic metric harmonization where data rights permit.
B4 held-out full-scale service campaigns with load, route, water condition, hull condition and maintenance state.

Research targets, not achievements: zero false hard-gate passes; zero robust dominance claims unless all objective intervals separate in the required directions; 100% metric/gate provenance; 100% service/unit matching; and a 10% primary-energy-per-cargo-work improvement campaign target only after every environmental and reliability gate passes.
