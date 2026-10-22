# First-Aid Workflow Invariant Coverage Map v0

**Status:** engineering coverage inventory for synthetic process-integrity tests.  
**Claim ceiling:** this map describes which software checks exist; it is not clinical validation, protocol approval, medical-device certification, or proof of patient safety.

## Why this crosswalk exists

Three forms of evidence now coexist in the research branch:

1. The **typed trace checker** evaluates ordered event histories and emits violation codes.
2. A separately structured **JSON-value reference interpreter** checks the same frozen corpus and selected adversarial mutations.
3. A finite **abstract state model** exhaustively explores 84 reachable states and 1,344 state/action transitions for its declared action space.

They are not interchangeable. The state model is compiled only for tests and covers a smaller state space. A mutation being represented in the matrix means the expected result is declared and mapped to a test or workflow gate; the new exact-head workflow must pass before treating the mapping as verified evidence.

## Invariant crosswalk

| Invariant / violation | Typed trace checker | JSON reference interpreter | Finite state model | Relevant cases | Remaining qualification |
|---|---|---|---|---|---|
| Invalid bundle verification | Yes | Yes | Yes | MUT-001, MUT-014, MUT-015, MUT-027 | Expand generated testing over all bundle IDs and combinations of session, expiry, digest, and revocation state |
| Render outside an active session | Yes | Yes | Yes | MUT-001, MUT-027 | Check every transition path that can produce a render, including replay and duplicate events |
| Render after revocation | Yes | Yes | Yes | MUT-008, MUT-017 | Add generated sequences with repeated revocations, multiple bundles, and arbitrary new-session boundaries |
| Render without matching verified bundle | Yes | Yes | Yes | MUT-002, MUT-009, MUT-013, MUT-014, MUT-015, MUT-026, MUT-027, MUT-028 | Compare ordered event outcomes for two or more bundle IDs and all resume/interruption combinations |
| Uncertainty lost at handoff | Yes | Yes | Yes | MUT-005 corpus boundary; MUT-016 re-entry | The finite model only represents one pending uncertainty flag; define behavior for multiple sources and partial preservation |
| Uncertainty lost at session boundary | Yes | Yes | Yes | MUT-025 | Add generated cases where preserved and unpreserved uncertainties coexist across successive boundaries |
| Unknown state normalized to ordinary state | Yes | Yes, at event-semantic layer | No | BASE-FA-TRACE-004 | Clarify whether this event represents a domain-level unknown rather than schema-invalid input; add model coverage if the semantics are retained |
| Stale/unknown resource presented as confirmed | Yes | Yes | No | BASE-FA-TRACE-006, MUT-010 | Add unknown resource states and all freshness/presentation combinations to a separate reference model |
| Offline core-network dependency | Yes | Yes | No | BASE-FA-TRACE-007, MUT-023 | Model online/offline/intermittent connectivity explicitly and test when network is optional vs required |
| Intermittent core-network dependency | Yes | Yes | No | MUT-003 | Generate flapping network histories, timeouts, and recovery transitions; current model only classifies static connectivity |
| Sensitive-data egress without consent scope | Yes | Yes | No | BASE-FA-TRACE-009, MUT-024 | Model consent scope, payload sensitivity, revocation, and egress as transitions rather than a single event predicate |

## What is verified vs what is only represented

- **Declared and mapped:** the machine-readable matrix contains 40 entries—12 frozen traces and 28 mutation/control cases. The Rust matrix runner executes 26 cases and the workflow gates cover two separately.
- **Differential implementation check:** a separate JSON-value-based interpreter is compared on the frozen traces and a defined mutation suite. Because both are maintained here and share expected fixtures, agreement is not independent ground truth.
- **Finite state exploration:** the test-only state model checks its 16-action abstract system exhaustively, with baseline counts of 84 reachable states and 1,344 state/action transitions. Those counts are tied to the current abstract model, not to all production events.
- **Independent JS audit performed during authoring:** a separate interpreter written for an analysis pass compared all 40 declared matrix outcomes against the frozen corpus and current matrix operators; 40 matched and zero mismatched. This is a one-off authoring-time result, not a committed evaluator, not the Rust workflow, and not a clinical oracle.

## Prioritized next steps

1. **Expand the finite model** to freshness, connectivity, consent-scoped egress, and unknown-state semantics while controlling state explosion with explicit finite domains and documented abstraction.
2. **Generate traces from the abstract model**, then check whether trace-checker outputs preserve the same safety properties. Preserve each counterexample as a named regression case.
3. **Add property-based sequence generation** and measure coverage by transition pair/sequence depth, not just line or branch coverage.
4. **Introduce independently adjudicated benign controls** so false-positive burden can be estimated; the current 12-trace corpus is too small to support robust rate estimates.
5. **Commission independent domain review** of the invariant catalog. Technical agreement between implementations cannot establish that the invariants are clinically complete or correctly scoped.

Until those steps are complete, do not claim that the finite-state model proves all trace invariants or that the overall system is clinically safe.
