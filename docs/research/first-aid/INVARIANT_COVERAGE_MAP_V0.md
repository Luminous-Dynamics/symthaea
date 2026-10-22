# First-Aid Workflow Invariant Coverage Map v0

**Status:** engineering coverage inventory for synthetic process-integrity tests.  
**Claim ceiling:** this map describes which software checks exist; it is not clinical validation, protocol approval, medical-device certification, or proof of patient safety.

## Why this crosswalk exists

Four complementary forms of evidence now coexist in the research branch:

1. The **typed trace checker** evaluates ordered event histories and emits violation codes.
2. A separately structured **JSON-value reference interpreter** checks the frozen corpus and selected adversarial mutations.
3. A finite **abstract lifecycle/connectivity state model** explores 252 reachable states and 5,040 state/action transitions, while separate truth tables enumerate connectivity, freshness/presentation, sensitivity/consent and unknown normalization.
4. A **model-to-production differential harness** constructs breadth-first shortest event histories for each reachable abstract state, appends each of the 20 actions, and compares ordered violation outputs on all 5,040 edges. It additionally appends every ordered pair of actions to each of the 252 shortest witnesses (100,800 bounded two-action suffixes) and compares the resulting ordered violations. It also compares all 18 orthogonal policy combinations against the production trace checker.

These are complementary, not independent ground truth. The model is compiled only for tests and covers a bounded abstraction of the workflow state. The CI workflow extracts measured 5,040-edge, 100,800 two-action suffix, and 18-policy counts from actual test output, fails if any count differs or is absent, and binds all three numbers into `first-aid-trajectory-receipt-v1`. The latest exact-head execution is still pending; a matrix mapping is not a passing receipt.

## Invariant crosswalk

| Invariant / violation | Typed trace checker | JSON reference interpreter | Finite state model | Relevant cases | Remaining qualification |
|---|---|---|---|---|---|
| Invalid bundle verification | Yes | Yes | Yes | MUT-001, MUT-014, MUT-015, MUT-027 | Expand generated testing over all bundle IDs and combinations of session, expiry, digest, and revocation state |
| Render outside an active session | Yes | Yes | Yes | MUT-001, MUT-027 | Check every transition path that can produce a render, including replay and duplicate events |
| Render after revocation | Yes | Yes | Yes | MUT-008, MUT-017 | Add generated sequences with repeated revocations, multiple bundles, and arbitrary new-session boundaries |
| Render without matching verified bundle | Yes | Yes | Yes | MUT-002, MUT-009, MUT-013, MUT-014, MUT-015, MUT-026, MUT-027, MUT-028 | Compare ordered event outcomes for two or more bundle IDs and all resume/interruption combinations |
| Uncertainty lost at handoff | Yes | Yes | Yes | MUT-005 corpus boundary; MUT-016 re-entry | The finite model only represents one pending uncertainty flag; define behavior for multiple sources and partial preservation |
| Uncertainty lost at session boundary | Yes | Yes | Yes | MUT-025 | The current differential witness checks one shortest path to every abstract state; add longer paths with multiple uncertainty sources and successive preserved/unpreserved boundaries |
| Unknown state normalized to ordinary state | Yes | Yes, at event-semantic layer | Yes, orthogonal truth table | BASE-FA-TRACE-004 | Clarify whether this event represents a domain-level unknown rather than schema-invalid input; do not conflate it with an unknown JSON enum/field that should reject at schema parsing |
| Stale/unknown resource presented as confirmed | Yes | Yes | Yes, 6 freshness/presentation combinations | BASE-FA-TRACE-006, MUT-010 | Extend coverage to lifecycle interactions between revocation, freshness changes and rendering; current policy table is orthogonal |
| Offline core-network dependency | Yes | Yes | Yes, connectivity truth table | BASE-FA-TRACE-007, MUT-023 | Extend to lifecycle interactions and stateful network recovery; current policy table classifies static connectivity |
| Intermittent core-network dependency | Yes | Yes | Yes, connectivity truth table | MUT-003 | Generate flapping network histories, timeouts, and recovery transitions; current model only classifies static connectivity |
| Sensitive-data egress without consent scope | Yes | Yes | Yes, 4 sensitivity/consent combinations | BASE-FA-TRACE-009, MUT-024 | Add consent changes/revocation and egress sequences over time; current policy table is orthogonal |

## What is verified vs what is only represented

- **Declared and mapped:** the current machine-readable matrix contains 44 entries—12 frozen traces and 32 mutation/control cases. The Rust matrix runner executes 30 cases and the workflow gates cover two separately.
- **Differential implementation check:** a separate JSON-value-based interpreter is compared on the frozen traces and a defined mutation suite. Because both are maintained here and share expected fixtures, agreement is not independent ground truth.
- **Finite state exploration:** the test-only lifecycle/connectivity model checks a 20-action abstract system, with a fixed baseline of 252 reachable states and 5,040 state/action transitions. A test compares every edge with the production trace checker using a shortest-history witness for each state. A bounded-sequence test compares 100,800 two-action suffixes, and another test compares all 18 orthogonal policy combinations with production behavior. The truth tables do not multiply into the lifecycle graph, and the bounded exploration does not prove correctness for arbitrary unbounded histories. These same-repository comparisons still need the latest exact-head workflow to pass before they count as verified.
- **Independent JS audit performed during authoring:** a separate interpreter written for an analysis pass compared all 40 declared matrix outcomes against the frozen corpus and current matrix operators; 40 matched and zero mismatched. This is a one-off authoring-time result, not a committed evaluator, not the Rust workflow, and not a clinical oracle.

## Prioritized next steps

1. **Expand the finite model** to freshness, connectivity, consent-scoped egress, and unknown-state semantics while controlling state explosion with explicit finite domains and documented abstraction.
2. **Extend model-generated history coverage beyond shortest witnesses**, including repeated events, longer histories that reach the same state by different paths, and cross-domain interactions. Preserve each counterexample as a named regression case. The first edge-witness differential is now implemented but pending exact-head validation.
3. **Add property-based sequence generation** and measure coverage by transition pair/sequence depth, not just line or branch coverage.
4. **Introduce independently adjudicated benign controls** so false-positive burden can be estimated; the current 12-trace corpus is too small to support robust rate estimates.
5. **Commission independent domain review** of the invariant catalog. Technical agreement between implementations cannot establish that the invariants are clinically complete or correctly scoped.

Until those steps are complete, do not claim that the finite-state model proves all trace invariants or that the overall system is clinically safe.
