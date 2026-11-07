# Marine autonomy-vector benchmark (V0)

A standalone, dependency-free Rust contract for checking whether declared resource inventories cover a target mission duration one resource dimension at a time.

It deliberately does not expose a scalar "autonomy days" result.

## Run

~~~sh
cargo test --locked --offline --manifest-path crates/tools/marine-autonomy-vector-bench/Cargo.toml
cargo run --locked --offline --manifest-path crates/tools/marine-autonomy-vector-bench/Cargo.toml --example thirty_day_vector
~~~

The example uses synthetic values. It is a contract / arithmetic demonstration, not a vessel or life-support validation.

## V0 inputs

A case binds:
- a frozen profile ID and SHA-256 profile-manifest identity;
- an exact configuration digest, operating-case ID and condition digest;
- a target duration in days;
- the complete required resource-dimension roster;
- exactly one input row per required dimension.

Each row carries its unit, usable stock interval, net daily depletion interval, exact subject bindings, evidence reference and evidence class.

The supplied usable stock is already net of unusable heel, inaccessible inventory and separately held emergency reserve. The supplied net depletion rate is already the result of the upstream process model. The crate does not derive either value.

For waste-storage rows, the rate means the net consumption of remaining storage headroom after declared treatment and retained residuals. Do not credit any overboard discharge unless its legality and engineering applicability are separately established; that decision is outside this crate.

## Interval arithmetic

Given stock [S_low, S_high] and non-negative depletion rate [r_low, r_high]:

- minimum endurance = S_low / r_high, if r_high > 0;
- maximum endurance = S_high / r_low, if r_low > 0;
- zero upper depletion rate yields no finite lower-endurance bound in this static interval model;
- zero lower depletion rate yields no finite upper-endurance bound.
- zero stock combined with zero upper depletion is ambiguous for a required dimension and returns Indeterminate; the model does not infer sufficiency from an empty store.

A dimension is sufficient only when its conservative minimum endurance meets the target. It is insufficient only when even its maximum endurance falls short. Otherwise it is indeterminate.

The report lists every required dimension and its own interval/disposition. The overall disposition says whether all dimensions cover, at least one fails, the result is indeterminate, or the inputs are invalid. It does not collapse resource dimensions into one number.

## Explicit boundaries

This package does not:
- calculate propulsion, hotel or battery energy (compose with #8092/#8102);
- calculate water demand, desalination, water quality, sewage treatment, nutrition or crop production;
- authenticate measurements, verify sensor calibration or recalculate supplied digests;
- decide legal waste discharge, potable-water safety, nutritional adequacy, fire/flood safety, stability, structure, seaworthiness, class/flag compliance or operating authority;
- turn assumed, synthetic, model-estimated, externally sourced or measured inputs into a claim of physical autonomy.

Evidence classes are preserved, not graded. A Measured label does not establish measurement accuracy. Digests are identity strings, not source authentication.

## Frozen fixture corpus

The fixtures directory contains an eight-case synthetic campaign: adequate coverage, water-limited coverage, a scalar-average false pass, food-nutrient bottleneck, missing required dimension, zero depletion, stale condition binding and inverted stock interval. These records are review fixtures; they are not physical data and are not automatically promoted to measured evidence.

## Benchmarks

Contract targets:
- 100% agreement with independent hand-calculated interval fixtures;
- zero missing, duplicate, unexpected or stale resource dimensions accepted as complete;
- zero false coverage outcomes in frozen negative controls;
- deterministic output under input permutation;
- every supplied row and finding accounted for.

Runtime / allocation benchmarks should be recorded only after local execution on frozen 8 / 32 / 128 / 512-dimension workloads. No throughput claim is made here.

The intended comparison with a naive scalar/minimum-days approach is correctness: a long food duration cannot compensate for a short potable-water or waste-storage limit.
