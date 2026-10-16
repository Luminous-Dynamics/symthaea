# Symthaea Futures: Prediction-Market Integrity Lab

**Status:** early research harness; synthetic fixtures only; not a trading system.

This crate is a narrow audit kernel for four structural questions:

1. Does a large count of forecast accounts conceal repeated reports from the same declared source group?
2. How much gross turnover is explicitly same-beneficial-owner, assuming the supplied ownership labels are already qualified?
3. Does an event contract specify the fields required for a structured resolution policy?
4. How much of a terminal outcome-aligned price move was already present before a supplied public-information timestamp, under declared scope conditions?
5. How sensitive is that descriptive statistic to several plausible timestamps, with explicit exclusions rather than zero-filled missing values?

It deliberately does not download venue data, execute orders, infer criminal intent, prove source independence, authenticate identities, resolve events, or claim that the remaining volume is economically meaningful risk transfer.

## Run

From the Symthaea workspace:

~~~sh
cargo fmt --manifest-path crates/domains/symthaea-futures-market-integrity/Cargo.toml --check
cargo test --manifest-path crates/domains/symthaea-futures-market-integrity/Cargo.toml
cargo clippy --manifest-path crates/domains/symthaea-futures-market-integrity/Cargo.toml --all-targets -- -D warnings
~~~

These commands are proposed reproduction commands. Do not infer they passed until the exact commit has completed the relevant checks.

## Design boundaries

- Source-capped pooling is a sensitivity baseline. Equal weighting of declared source groups is not claimed to be universally optimal.
- Source groups are labels supplied to the harness; a production system must bind them to evidence and handle unknown/correlated sources conservatively.
- Same-owner turnover is only a structural alert. It cannot detect transactions routed through distinct owners, infer intent, or prove net risk transfer.
- Resolution-policy qualification checks field completeness and simple bounds. It does not authenticate a source, freeze contract bytes, establish independent adjudication, or interpret the law.
- Information-timing diagnostics depend on an independently qualified public-event timestamp, declared edge exclusions, and sufficient terminal movement toward the realized outcome. They are not an insider-trading probability or proof of misconduct; timestamp error and event selection can invalidate the interpretation.
- Timestamp sensitivity reports variation over a finite, strictly ordered candidate sample only; it is not a continuous uncertainty bound unless the candidate set is separately justified as complete for the declared timestamp interval.
- All outputs are diagnostics, never a trade recommendation or autonomous market action.

See docs/research/finance/PREDICTION_MARKET_EPISTEMIC_INFRASTRUCTURE_V1.md for the wider research agenda and benchmark targets.
