# Marine acoustic exposure ledger audit

A standalone Rust audit that converts evidence-bound source levels and declared frequency-dependent transmission-loss model outputs into received-level intervals, per-contribution SEL, cumulative ship-only SEL and maximum simultaneous received level at explicit receiver locations.

## Run

~~~sh
cargo test --manifest-path tools/marine-acoustic-exposure-ledger-audit/Cargo.toml
cargo run --manifest-path tools/marine-acoustic-exposure-ledger-audit/Cargo.toml -- \
  docs/research/marine/fixtures/synthetic-marine-acoustic-exposure-ledger-v0.1.0.json
~~~

The synthetic fixture must return INCOMPLETE as physical evidence.

## Arithmetic and decision contract

- Received SPL = declared source level at 1 m − declared transmission loss.
- Contribution SEL = received SPL + 10 log10(duration in seconds / 1 s).
- Cumulative ship-only SEL combines segment SELs in the linear energy domain.
- Maximum individual-contribution SPL and maximum simultaneous ship SPL are reported separately. Simultaneous SPL uses a time-boundary sweep; only events active at the same time are energy-summed.
- The period must explicitly declare the source-energy summation basis as incoherent or phase-averaged. If this basis is unknown, per-contribution results remain available but aggregate ship SEL, maximum simultaneous level and exposure-limit decisions are withheld.
- Source-level and transmission-loss expanded uncertainties must carry matching coverage levels. V0 conservatively adds their bounds; it does not assume statistical independence. Ambient samples carry their own coverage level, which must remain consistent across a receiver/band reference series.
- Receiver range is checked against geodetic horizontal distance plus vertical source/receiver separation.
- Source-to-propagation clock alignment has its own receipt digest, separate from both the source-level and propagation timebase receipts.
- Ambient reference SEL is calculated and reported separately. It is not added to ship-only SEL because shipping contributions must not be double-counted as “ambient-only” evidence.
- Overlapping repeated ambient reference windows for the same receiver/band are rejected rather than summed twice.
- Every band ID has one canonical centre frequency across source events, propagation contributions, ambient records and exposure limits. Quantity IDs explicitly distinguish source level, received SPL and SEL.
- A declared upper SEL threshold passes only when the entire ship-only cumulative SEL interval is below/equal to it, fails only when the lower bound exceeds it, and is otherwise unresolved. Synthetic, rights-uncleared or unreviewed source evidence cannot produce a threshold pass or fail.

## Interpretation and limits

This ledger consumes source level and transmission loss; it does not implement BELLHOP, parabolic-equation, ray-tracing or other underwater propagation solvers. It does not infer auditory weighting, species sensitivity, masking, TTS/PTS or biological impact. It does not prove that a source-level measurement is single-source or ecologically beneficial. Input hashes are references, not authentication.

The output is a frequency-resolved source-to-receiver ship-noise accounting result with separate ambient reference. It is not a regulatory compliance certificate or a marine-mammal dose/impact assessment.
