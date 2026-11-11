# AIRROAD Source Capability Oracle

A read-only Rust adapter that joins the multi-propulsor DC-bus demand report to a state-conditioned, evidence-bound battery/source capability profile.

## Ownership boundary

- The electric rotorcraft drive/bus reference in `electric-rotorcraft-drive-oracle` (#11642) supplies simultaneous bus power, current, voltage, energy and exact interval identity.
- AIRROAD battery capability semantics and mission benchmark #7805 own the battery model, usable-energy, SOC/SOH, reserve, voltage/current and temperature-dependent source capability.
- The coupled electro-thermal mission solver #11411 owns iterative battery/propulsion/thermal feedback.
- Cross-fidelity validation #11445 owns analytic, differential and physical holdout promotion gates.

This crate does not implement battery electrochemistry, OCV, resistance, SOC/SOH propagation, voltage sag, cell imbalance, thermal runaway or charging. It only checks that an upstream battery/source model's capability result applies to the exact bus request and covers the whole mission plus protected reserve.

## Contract

Each source profile is bound to:
- battery/source instance and exact bus source-boundary ID;
- bus and aircraft configuration;
- as-installed pack configuration;
- mission-profile digest;
- source model identity/revision and model-output/evidence manifests;
- one interval record for each exact bus segment;
- source-state and operating-context identity at each interval;
- lower/nominal/upper usable energy above the hard BMS floor and protected end reserve.

For each interval and adverse/nominal/favourable scenario, the upstream source model supplies:
- the exact bus power/current/voltage it assessed;
- maximum deliverable terminal power and current for that interval;
- minimum deliverable terminal voltage under that demand;
- the duration for which that capability result is valid;
- the capability evidence-manifest digest.

The adapter requires the assessed operating point to exactly match the corresponding bus-segment report. It then compares demand with the supplied capability values. Missing profiles or interval evidence remain UNKNOWN; malformed capability values are a non-pass; it never falls back to static nameplate limits. It relies on the upstream bus oracle to produce a internally consistent bus report and does not recompute branch/bus aggregation.

## Scenario and reserve semantics

Adverse demand is checked against the adverse source capability, lower usable-energy bound and upper protected reserve. Nominal and favourable cases use their corresponding source/demand cases. This is a rectangular envelope when correlation is not supplied; it is not a probability or confidence interval.

Mission energy is taken directly from the bus report. It already includes propulsion, TMS and auxiliary loads. The adapter does not add those loads again.

## Dispositions

- `FeasibleWithinSuppliedCapability`: all required interval checks and mission reserve inequality pass.
- `Infeasible`: at least one supplied interval limit or the mission energy/reserve inequality fails.
- `UnknownMissingCapability`: the source profile, interval, operating state or evidence needed for qualification is absent.
- `NonPassIdentityMismatch`: source/configuration/mission/operating point identities conflict.
- `NonPassInvalidCapability`: malformed, non-finite or out-of-domain source capability values are supplied.

None is a battery-safety, aircraft airworthiness or certification finding.

## Limitations

The first version evaluates the capability bounds supplied for each exact interval. It does not solve battery source impedance, transient DC-link dynamics, rolling peak-duty/recovery windows, charge/thermal reset between sorties or multi-bus shared-source aggregation. A source supplying several buses requires a qualified aggregate source profile; duplicate source IDs are rejected rather than counted independently.

## Verification

```bash
cargo fmt --manifest-path tools/airroad-source-capability-oracle/Cargo.toml -- --check
cargo test --locked --manifest-path tools/airroad-source-capability-oracle/Cargo.toml
cargo clippy --locked --manifest-path tools/airroad-source-capability-oracle/Cargo.toml --all-targets -- -D warnings
```

The current source contains 16 synthetic tests. They validate only the adapter's identity, exact-point, limit and reserve arithmetic semantics. No battery capability or aircraft performance result is established.
