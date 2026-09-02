# VART-002 Derivation Provenance & Shadow Recall v1

Status: development contract; not a preregistration; confirmatory execution unauthorized.

## Purpose

VART-001 motivates testing whether epistemically typed memory is causally useful. VART-002 must not jump directly from a promising benchmark result to production filtering. This contract defines the intermediate instrumentation needed to make provenance derivation auditable and to measure the behavioral distance between current raw episodic recall and provenance-safe recall before enforcement.

## 1. Derivation provenance

A provenance label is insufficient if the system cannot explain how the labeled object was derived.

Every new derivation-capable integration should be able to emit an `EpistemicDerivationReceipt` binding:

- exact child subject SHA-256;
- exact parent subject SHA-256 values;
- transform kind;
- transform implementation identity;
- transform version;
- event time when available.

`derive_with_receipt` is intentionally incapable of creating `PhysicalGrounded` or `DigitalCommitted` objects. Grounding still requires subject-bound `GroundingEvidence`.

Counterfactual taint propagates through derivation and cannot be cleared by a derivation receipt.

## 2. Composite cognitive episodes

Current episodic records contain both encoded input/perception and Symthaea's own cognitive output. The entire `Episode` must therefore not be labeled `PhysicalGrounded` merely because some input originated in perception.

Until component-level provenance is explicitly wired:

- historical/legacy episodes remain effective `Unknown` unless explicitly attached;
- runtime code must not synthesize grounded episode provenance from heuristics;
- a future component derivation should bind perception/input provenance, internal-output provenance, and the final episode subject separately or through an equivalent auditable composite receipt.

## 3. Shadow recall

Shadow recall is behavior-neutral instrumentation.

For the exact same similarity-ranked episodic candidate set it records what the following views would admit:

- `GroundedHistory`;
- `GroundedOrImported`;
- `CounterfactualOnly`.

The shadow audit records at least:

- similarity-eligible count;
- provenance-admitted count;
- would-return count after top-k;
- exclusions due to `Unknown`;
- exclusions due to active counterfactual taint;
- exclusions due to incompatible known domain;
- top-k truncation;
- overlap with production raw top-k;
- whether enforcement would change selection.

Shadow mode MUST NOT:

- replace the production recall result;
- alter prediction confidence;
- alter proposal construction;
- alter action authority;
- alter replay/consolidation;
- expose hidden VART fixtures, seeds, or outcomes.

## 4. Promotion gate: shadow -> enforcement

Provenance filtering may become behaviorally load-bearing only after all of the following are true:

1. perception/input provenance is subject-bound rather than inferred from a code path name;
2. internal transformations emit derivation receipts or an equivalent auditable chain;
3. composite episodic semantics do not label internal cognitive output as externally grounded;
4. legacy records remain fail-closed as `Unknown`;
5. sidecar/persistence restoration is paired and tested;
6. shadow audit has run on fresh DEVART workloads;
7. raw-vs-safe selection divergence is characterized by domain and cause;
8. no hidden VART-002 confirmatory material has been used to tune the filter;
9. an explicit development decision records the retrieval mode to enforce;
10. VART-002 preregistration remains prospective relative to hidden confirmatory outcomes.

## 5. What shadow-mode data may be used for

Allowed development uses:

- finding missing provenance instrumentation;
- measuring legacy `Unknown` prevalence;
- finding accidental counterfactual/history mixing;
- performance/cost profiling of provenance filtering;
- designing DEVART regression fixtures;
- validating audit accounting.

Forbidden uses:

- tuning against VART-001 confirmatory outcomes;
- inspecting hidden VART-002 worlds to choose thresholds;
- changing a future preregistered metric direction after outcome inspection;
- converting a shadow audit into an efficacy claim.

## 6. Claim boundary

A shadow recall result can establish only that a provenance policy would have selected a different memory set under a development workload. It does not establish that the alternate selection is more intelligent, safer, or more effective.

Those are VART-002 scientific questions and require fresh hidden evidence.
