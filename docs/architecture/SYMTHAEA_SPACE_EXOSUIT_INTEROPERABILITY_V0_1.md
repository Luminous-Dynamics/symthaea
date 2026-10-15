# Space Exosuit suit–vehicle interoperability contracts v0.1

**Status:** Software-level compatibility and evidence-traceability research. A matching record is not proof of physical, electrical, pressure, thermal, breathable-gas or human-rated compatibility. No control or actuator interface is provided.

**Research track:** [SX-030 — versioned suit–vehicle interoperability contracts and electrostatic grounding evidence](https://github.com/Luminous-Dynamics/symthaea/issues/7517).

**Implementation:** Draft branch `feat/space-exosuit-interface-compatibility`, stacked on [SX-028 specialized EVA profiles PR #7505](https://github.com/Luminous-Dynamics/symthaea/pull/7505).

## Why this matters now

NASA's April 20, 2026 OIG report IG-26-006 recommended a plan for interoperability standards between Artemis lunar vehicles and spacesuits. The report described compatibility concerns as NASA shifts toward a broader commercial space economy. NASA's International Deep Space Standards deliberately define interoperable interfaces rather than prescribing full system designs, with separate examples for avionics, communications, power, software, ECLSS, docking, robotics and thermal systems.

NASA also opened the Lunar Grounding Challenge on October 5, 2026, with submissions due January 15, 2027. NASA describes tribocharging from walking and plasma charging in lunar shadow/polar regions; accumulated suit charge can create a hazardous electrostatic discharge when a suited crew member contacts a vehicle. The challenge explicitly treats this as a suit/lander interaction problem, not simply a dust-retention problem.

Sources:
- NASA OIG, next-generation suit acquisition and interoperability recommendation: https://oig.nasa.gov/news/spacesuit-development-delays-could-impact-artemis-timeline-and-iss-operations/
- NASA OIG report IG-26-006: https://oig.nasa.gov/wp-content/uploads/2026/04/final-report-ig-26-006-nasas-acquisition-of-next-generation-spacesuit-services.pdf
- NASA International Deep Space Standards: https://www.nasa.gov/international-deep-space-standards/
- NASA Lunar Grounding Challenge (published October 6, 2026): https://www.nasa.gov/directorates/stmd/prizes-challenges-crowdsourcing-program/center-of-excellence-for-collaborative-innovation-coeci/lunar-grounding-challenge/
- NASA-STD-3001 Volume 2 Revision F (active, dated July 14, 2026): https://standards.nasa.gov/node/237

## Interface contract model

`suit_interop.rs` models a component manifest as versioned endpoints with:
- stable logical interface ID and endpoint ID;
- explicit interface kind and provider/consumer/bidirectional role;
- exact contract revision;
- schema/ICD SHA-256 digest;
- connector-profile ID;
- applicable environment set;
- declared evidence level, reference, and required/optional status.

Interface kinds are deliberately distinct: pressure boundary, breathable gas, coolant, thermal exchange, electrical power, grounding/ESD, communications, telemetry/data, mechanical capture, restraint/anchor, consumable transfer, dust service and rescue handling. The SX-028 EVA profile model also declares `ElectrostaticChargeManagement` separately from `DustMitigation` for lunar-surface and dust-intensive work hypotheses, so dust protection is not silently treated as a substitute for charge/grounding work.

The caller supplies a non-empty operating-environment context and minimum evidence floor. The evaluator checks both directions for required consumer/bidirectional endpoints and reports missing peers, duplicate/ambiguous providers, kind/role mismatches, revision or schema-digest mismatch, connector mismatch, unsupported environment, and evidence below policy. All findings are canonicalized before reporting.

No version coercion is performed: matching labels with different digests do not match; different contract revisions do not match; a connector profile does not match merely because its text is similar. Future adapters require their own declared mapping/version/evidence and are intentionally not inferred in this first tranche.

The new `GroundingAndEsd` interface kind and `LunarShadowOrPolarRegion` environment let a mission manifest require a declared interface contract in that context. They do not prescribe a discharge voltage/current, discharge device, sequencing operation, or operational safety decision. Those must come from approved engineering requirements and instrumented bench evidence. A compatible software record must never be treated as permission to touch a vehicle or to initiate grounding.

## Interpreting outcomes

- `MeetsDeclaredCompatibilityPolicy`: the supplied declared contract and evidence fields agree for the specific environment and evidence floor. It does not prove hardware compatibility.
- `RequiredInterfacesBlocked`: at least one required endpoint is missing, ambiguous, mismatched or below policy.
- `NoRequiredInterfacesToVerify`: there was no required consumer/bidirectional endpoint; this is not a compatibility pass.
- `represents_physical_qualification` is always false.

Evidence level is a caller-declared maturity label. A URI/report reference is stored but not authenticated or independently reviewed by this module. A SHA-256 digest detects content changes; it does not establish that the underlying interface design is sound, reviewed or authorized.

## Verification plan

Unit tests exercise matched grounding contracts in a lunar-polar environment, absent grounding peer, same logical ID/different schema digest, revision/connector mismatches, unsupported environment, duplicate providers, insufficient evidence, directionality, invalid digest/duplicate endpoint IDs, empty environment context, input ordering and changed-manifest receipt rejection.

Run on a configured workspace:

```sh
cargo test -p symthaea-exoskeleton suit_interop
```

Tests are authored but need execution on the exact PR head. A passing software test establishes only these encoded contract checks.

## Platform-level paradigm shift

Treat the suit, lander/vehicle, habitat, rover, service station and rescue equipment as independently developed systems that negotiate *explicit and versioned interface contracts*. Keep commonality at the interface boundary so multiple hardware providers can compete, while every physical safety claim remains tied to the relevant interface-control document and test evidence. The goal is controlled interoperability, not assumed modularity.
