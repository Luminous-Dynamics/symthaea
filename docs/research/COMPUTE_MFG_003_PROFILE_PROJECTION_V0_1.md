# COMPUTE-MFG-003 — Computer-profile projection into CIV-BOOT closure (v0.1)

**Status:** draft, synthetic structural qualification only  
**Tracking:** [Symthaea COMPUTE-MFG-003 #7681](https://github.com/Luminous-Dynamics/symthaea/issues/7681)  
**Computer reproduction campaign:** [COMPUTE-MFG-002 #7658](https://github.com/Luminous-Dynamics/symthaea/issues/7658)  
**Computer-domain owner:** [COMPUTE-MFG-001 #5796](https://github.com/Luminous-Dynamics/symthaea/issues/5796) / [001A #5798](https://github.com/Luminous-Dynamics/symthaea/issues/5798)  
**Atlas source:** [Mycelix CIV-BOOT-003 draft PR #5064](https://github.com/Luminous-Dynamics/mycelix/pull/5064)

## Why this slice exists

The Mycelix computer-reproduction corpus describes computer-domain outcomes (repair, assembly, host-assisted copy, host-removal copy, independent transfer, and PCB/die branches). CIV-BOOT's Rust kernel describes a narrower theorem: structural productive capability closure, operational reachability with declared imports, and dependency leverage.

This slice maps **declared computer profile facts** into `ClosureProblemV1` and checks the generic kernel's structural result. It does not duplicate the closure algorithm and does not translate a structural reachability result into a physical C4/C5 pass.

## Files

- `crates/domains/symthaea-capability-closure/fixtures/computer-profile-projection-v0.1.json`
  - eight frozen scenario fixtures;
  - seven structurally valid inputs and one unresolved-input preflight case;
  - source-pin to the Mycelix Atlas corpus blob: `e9a7a8bce54d0ec89151ee011a5bb3d7002b5134`.
- `crates/domains/symthaea-capability-closure/examples/computer_profile_projection.rs`
  - qualification-only Rust executable;
  - evaluates each valid `ClosureProblemV1` through `evaluate_reachability` and `evaluate_import_leverage`;
  - compares the actual canonical JSON to preregistered expected output;
  - rejects unresolved/unknown required inputs before invoking V1, whose capability-presence contract is boolean;
  - verifies declaration-order equivalence cases return identical normalized results.

## Frozen scenario coverage

1. Locally fabricable support parts coexist with an imported-compute-dependent computer.
2. An imported finished computer can be operationally reachable without a route to reproduce a computer.
3. A bounded locally buildable controller remains locally closed while a high-compute workstation remains import-dependent.
4. Missing a required CPU module blocks board/computer routes without erasing local enclosure capability.
5. A cycle between controller and firmware toolchain cannot bootstrap itself.
6. Shuffled declarations must have exactly the same canonical report as fixture 1.
7. Unknown compute-module stock is held as `UnresolvedInput`; it is not passed to a boolean reachability kernel.
8. Board assembly from imported bare PCB/ICs does not imply local die-fabrication capability.

## Run

```sh
cargo run --locked -p symthaea-capability-closure --example computer_profile_projection
```

A process exit code of zero means only that these structural fixtures matched this Rust implementation's frozen expected vectors. It is not a differential comparison against the Python reference, a hardware test, or proof of a physical computer-reproduction capability.

## Required interpretation

- **Local enclosure** may be `LocallyClosed` while a **whole computer** is `ImportDependent`.
- A **finished imported computer** may be operationally reachable while reproduction is `Unavailable`.
- A declared imported capability is an abstract presence assertion in this synthetic scenario; it is not a verified quantity, supplier, lot, lead time, substitute, or stock runway.
- Removing a declared imported capability estimates structural reachability loss in the graph only. It does not estimate cost, strategic criticality, or economic impact.
- The generic kernel cannot encode **unknown inventory** as a third truth value. The projection therefore rejects that case before kernel evaluation rather than inventing evidence.
- A structural C5 route does not witness disconnecting the original host or a second physical machine actually producing another. C4/C5 remain physical acceptance gates owned by #7658 and require exact BOM/build/test receipts.
- PCB fabrication and semiconductor die fabrication remain independent closure branches; neither is inferred from C0–C5.
- The existing 20-fixture CIV-BOOT-002D corpus and its blob pin are unchanged.

## Qualification status

This change adds the executable projection and fixture source. It has **not** been compiled or run in this environment. Do not treat this file's expected values as a PASS receipt. Before qualification, run the command above at the exact branch head, then pin the candidate SHA, fixture blob SHA, Cargo/Rust toolchain, command and complete output. Follow with differential comparison and only later with witnessed physical C4/C5 runs.
