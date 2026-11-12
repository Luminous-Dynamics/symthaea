# ROB-CAUSAL-MISSION-REPLAY-001

Owner: Symthaea issue #11662.

This standalone Rust reference performs an offline audit of whether a mission's actuator-applied commands are causally linked to the observations, decision/calibration state, optional inference request, action prefix, independent projection disposition and physical-state transition records that precede them.

## What it composes

- Symtropy #4713: source-to-actuator Age of Information and command-bound lineage receipts.
- Symthaea #11646 and #11655: decision-sufficiency horizon and simultaneous episode-level calibration.
- Symthaea #11627 / #11652: constrained optional fleet inference admission.
- Symtropy #2836 / #4117 / #4382: controller-boundary dispatch, certificate validity and policy projection.
- Symtropy #4555 and #4588: persistent reserve state and observation-constrained recovery options.

It consumes these owners' records; it does not reimplement or overrule their models or gates.

## Replay chain

For each actuator-applied command, join:

    source observations
      -> decision + calibrated regret bound
      -> optional remote request / admission / response (if used)
      -> action proposal bound to actually executed prefix
      -> independently supplied projection / assurance disposition
      -> actuator application + source-to-actuator AoI interval
      -> persistent physical reserve transition
      -> terminal safe-stop / retreat evidence

The replay validates event sequence order, exact IDs/digests, clock and boot-epoch identity, selected action and validity horizon, remote request/admission chronology, executed-prefix identity, independent gate disposition, reserve-state digest continuity, complete per-node thermal identity coverage, recovery-topology status and terminal closure.

## Clock and age semantics

Each event carries bounded timestamps [t-, t+] in one declared clock and boot epoch. For source observation interval [s-, s+] and actuator application interval [a-, a+]:

    AoI_min = a- - s+
    AoI_max = a+ - s-

If the same-clock event order cannot be established (a- < s+), the age is Unknown. Cross-clock or cross-boot subtraction is never attempted. This reference does not synchronize clocks or validate timestamp truth.

## Outcomes

- Closed: all required structural relationships in this supplied trace are closed.
- Unknown: a required relationship or evidence item cannot be established.
- Rejected: a deterministic causal, identity, ordering, digest, action, admission or reserve-chain violation exists.

The report always carries actuation_authorized = false. Closed means only that the supplied finite trace satisfies this reference's structural contract. It is not an actuation permit, safety certificate, physical validation or runtime assurance monitor.

## Run

    cargo fmt --manifest-path research/rob-causal-mission-replay-001/Cargo.toml -- --check
    cargo test --locked --manifest-path research/rob-causal-mission-replay-001/Cargo.toml
    cargo clippy --locked --manifest-path research/rob-causal-mission-replay-001/Cargo.toml --all-targets -- -D warnings

## Causal closure measurement

The replay now emits a **causal-closure metric** in addition to the mission-level Closed / Unknown / Rejected disposition:

- denominator: every actuator-application record supplied, including Unknown and Invalid attempts;
- linked applications: complete observation, decision/calibration, optional request/admission/response, proposal/prefix, assurance and application lineage;
- causally closed applications: linked applications with a corresponding physical transition, continuous predecessor-state chain, complete declared thermal-node manifest, certain timestamp order and known recovery topology;
- `closure_ppm`: floor(causally_closed / applications_examined × 1,000,000), or `None` when no applications are supplied.

Terminal safe-stop, energy closure, command response and other physical qualification domains remain separately reported. This avoids presenting a high application-lineage rate as a complete physical mission pass. The metric is a proposed Luminous benchmark, not an industry-standard leaderboard score.

The event records now bind policy digest to the decision and action chunk, actuator/controller configuration digests to the application, and thermal sensor calibration to the physical transition. These are checked by the independent replay before an application can count as linked/closed.

Current source-level corpus: 25 core causal-replay tests and 17 single-joint metrology-join tests. Exact-head Rust 1.96 CI remains the execution gate.
