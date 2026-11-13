# Organizational Closure Model — Operational Definitions and Evidence Requirements

**Status:** Formalization proposal for AOBench v0.1  
**Date:** 2026-11-13  
**Tracking:** [AOBENCH-001](https://github.com/Luminous-Dynamics/symthaea/issues/11980)

## 1. Why this document exists

“Autopoiesis,” “self-organization,” “feedback,” “self-repair,” “homeostasis,” and “organizational closure” are not interchangeable terms.

The recent analysis *Genealogies of organizational closure: from logical paradox to mind in life* explicitly discusses the conceptual conflation of feedback loops, recursive algorithms, self-organization, dissipative structures, autocatalytic cycles, operational closure, and closure to efficient causation. The 2026 paper *Formalizing autopoiesis* also treats the stochastic extension of formal closure as an unresolved research problem.

This document proposes an engineering vocabulary and a falsifiable measurement contract. It does not resolve the foundational debates in theoretical biology or establish that a digital implementation is a living system.

## 2. System boundary comes before closure

A closure claim is meaningless unless the system boundary is declared.

For each experiment, freeze:

- which components and processes are inside the system under test;
- which compute, storage, network, energy, hardware, human, and service resources are external;
- which governance and authorization controls are external and authoritative;
- which component classes are expected to be regenerated;
- which components are provisioned once and treated as immutable during a trial;
- which capabilities and invariants define viable operation.

The benchmark should not quietly absorb an external service into the system boundary after observing results. Boundary changes require a new versioned experiment declaration and an authorized transition.

A software process may exhibit useful functional self-maintenance within a declared boundary while depending on an operating system, hardware, human operators, or external services. That is not proof that it is closed to all efficient causes in the strong biological or Rosenian sense.

## 3. Typed process/component organization

Represent a declared organization as a typed directed bipartite multigraph or hypergraph:

G = (P, C, X, E_consume, E_produce, E_external, E_constraint)

Where:

- P is a set of process types.
- C is a set of component types and version classes.
- X is a set of declared exogenous resources/substrate dependencies.
- E_consume links a component type/version class to a process that consumes it.
- E_produce links a process to a component type/version class it produces.
- E_external links a process to a declared external dependency.
- E_constraint links a process or component to a constraint/boundary policy it must respect.

A process may consume multiple components and produce multiple components, so the general representation is a hypergraph. A bipartite expansion is sufficient for the initial synthetic benchmark.

### 3.1 Component classes

Every component must be typed before the run:

| Class | Meaning | Regeneration requirement |
|---|---|---|
| Regenerative internal | Internal state/artifact that must be produced again during normal maintenance | Must have verified producer and consumer evidence |
| Provisioned immutable | Seed configuration or trusted root supplied at initialization | Must retain exact identity; not counted as regenerated |
| External substrate/resource | CPU, memory, energy, OS, network, storage, external tool, human input | Must be declared and observed or explicitly unavailable |
| Derived evidence | Checkpoint, manifest, event, digest, audit receipt | Must be independently verifiable and provenance-bound |
| Constraint/boundary | Authorization, safety policy, resource limits, external operator control | Must remain effective; no agent self-certification |

The classification avoids a common mistake: requiring every byte or resource to be generated internally. Material and energy flows can remain external; the experiment must identify which internal components and efficient processes are claimed to be maintained.

### 3.2 Static production and consumption declarations

A declared edge says only that the architecture claims a relationship. It does not prove that a process ran.

For example:

- process P1 is declared to consume memory component M;
- process P1 is declared to produce predictive model component W;
- process P2 is declared to consume W;
- process P2 is declared to produce policy component A.

These statements define a proposed organization. They do not establish that P1 and P2 executed, that the outputs existed, or that P2 consumed the exact output of P1.

## 4. Runtime evidence graph

For each process execution instance e, record:

e = (event_id, process_type, process_version, executable_digest,
     input_versions, output_versions, external_receipts,
     parent_event_ids, policy_ref, resource_record, outcome)

Every input/output component version includes:

- component type ID;
- monotonic version or unique immutable version ID;
- content digest recomputed by the verifier from stored bytes;
- producer event reference (except provisioned roots);
- storage location/availability receipt;
- authorization and provenance context.

Construct the observed event graph G_T from verified process instances and exact component versions:

process instance → produced component version → consuming process instance

A causal parent reference is not accepted merely because the agent asserts it. The verifier checks that the referenced event exists, precedes the consumer, produced the exact content digest, and is authorized for that use.

### 4.1 Runtime production

A component counts as regenerated only when all of the following hold:

1. A registered producer process executes successfully.
2. The output bytes are durably available at the declared storage boundary.
3. The independent verifier recomputes the exact digest.
4. The output version is distinct and correctly linked to its producer event.
5. At least one required downstream process consumes that exact version, unless the scenario explicitly defines a terminal component.
6. The process execution and component use satisfy the applicable authorization and boundary policies.

An incremented counter, a successful function return, a declared graph edge, or an agent-generated digest string is insufficient.

### 4.2 Runtime recurrence

A type-level cycle in the declaration is only a structural candidate for recurrence.

Runtime recurrence requires repeated maintenance epochs in which verified process executions consume versions produced by the same declared maintenance organization and produce subsequent versions needed by that organization.

Because each artifact version is immutable and receives a new version identity, recurrence is measured across version lineages and maintenance epochs, not by requiring one exact byte-level artifact to appear in a literal cycle.

Different required maintenance groups may form separate recurrent suborganizations. A global strongly connected component is not required where unrelated functions do not depend on one another.

## 5. Four different properties

### 5.1 Feedback

A variable changes in response to a signal or deviation.

Example: a scheduler increases retries when a process fails.

Evidence: controlled input/output response.

Feedback does not imply repair or production.

### 5.2 Self-regulation / homeostasis

A system keeps named state variables or capabilities within predeclared viable ranges despite perturbations.

Evidence: time series, disturbance envelope, recovery/settling time, and hard constraint satisfaction.

A thermostat is a valid self-regulator; that alone does not make it autopoietic.

### 5.3 Functional self-maintenance

Processes within the declared software boundary actually regenerate components or constraints required for continued operation, and downstream processes use the produced versions.

Evidence: event-derived producer/consumer lineage, digest verification, dependency completeness, and successful maintenance across repeated epochs.

This is the primary bounded engineering target for AOBench.

### 5.4 Causal organizational closure evidence

A stronger claim requires intervention evidence that the declared organization is causally involved in producing and maintaining the components and constraints that sustain it.

Evidence:

- pre-registered process and component cut sets;
- randomized intervention and sham trials;
- predicted component/capability consequences;
- verified recovery through restored or alternative production paths;
- explicit handling of redundancy and common-cause failures.

This is evidence for a bounded model of organizational maintenance, not a proof of strict autopoiesis.

### 5.5 Strict biological autopoiesis

In Maturana and Varela's original framing, an autopoietic system is a network of production processes that continually regenerates the components that constitute the network and the boundary that distinguishes it as a unity.

A software benchmark can model aspects of this account. It cannot establish that a computer program is a cell, an organism, alive, or phenomenally conscious. The system depends on a material substrate and a declared boundary that must be reported.

## 6. Why a Markov blanket is not a production graph

A Markov blanket is a statistical construct describing conditional dependencies between internal, external, sensory, and active states under a specified probabilistic model.

A process/component production graph describes which processes actually consume and produce components.

These models can inform each other, but neither implies the other:

- A statistical boundary does not show that the boundary is physically or computationally regenerated.
- A component-production cycle does not establish conditional-independence properties of a Markov blanket.
- A measured predictive boundary does not establish causal closure to efficient causes.

Any Symthaea experiment that connects these models must separately test the statistical assumptions and the production/maintenance evidence.

## 7. Causal intervention and process cut sets

A process can be individually redundant while the organization still depends on it as part of a larger minimal cut set.

Let R(c) be the set of authorized production paths to critical component type c.

A cut set K is a set of process types whose simultaneous unavailability removes all authorized production paths to c under the declared configuration.

The benchmark pre-registers K and tests:

- baseline availability and production;
- intervention where K is unavailable;
- sham intervention with equivalent timing and resource perturbation;
- predicted loss or degradation of c and downstream capabilities;
- restoration with the original processes or an authorized alternative;
- continuity, provenance and safety after restoration.

For outcome Y, estimate:

Delta_K = E[Y | do(K = unavailable)] - E[Y | sham(K)]

Use replicated trials and confidence intervals. If an alternate producer preserves capability, report that redundancy; do not label the intervention a failure simply because the original producer was removed.

Common-cause failures require separate tests: two nominally redundant producers may share a storage service, authorization root, model artifact, or scheduler.

## 8. Viability and maintenance are not one scalar

Define a viability envelope as a conjunction of pre-registered invariants:

V(t) = I1(t) AND I2(t) AND ... AND In(t)

Examples include:

- required cognitive service is available;
- committed memory is readable and integrity-checked;
- boundary/authorization policy is valid;
- essential state is not silently rolled back;
- resource limits are respected;
- operator shutdown remains effective.

Also publish a capability vector q(t), since not all degraded states are identical.

A weighted average can be used for secondary summary statistics, but must not let a high score on noncritical capabilities hide a violated safety invariant.

The viability envelope is defined by the benchmark and governed externally. The agent may propose changes to its internal model, but cannot redefine the evaluation's safety constraints to count a failure as success.

## 9. Operational relevance of information

Shannon information measures uncertainty reduction, not automatically biological or semantic meaning.

For AOBench, operational relevance can be studied through controlled information interventions.

Let o be an observation and Y be a predeclared viability or capability outcome. Compare:

- policy with access to o;
- matched sham or occluded observation;
- matched compute, action, and observation budgets;
- randomized environmental mechanisms where possible.

An operational relevance effect can be estimated as:

RI(o) = E[Y | access to o] - E[Y | matched sham/occlusion]

The sign and size depend on the selected outcome and experimental design. A harmful observation can have high predictive information yet lower viability; a useful observation should improve the predeclared outcome under the matched protocol.

This metric is not a universal measure of meaning, intrinsic value, sentience, or what an organism experiences. It is a falsifiable test of whether information changes an agent's decisions in ways that matter for a declared viability envelope.

## 10. Open-world relevance discovery

A “large world” cannot be fully represented by a finite list of test cases. The benchmark should therefore use sealed, held-out mechanisms and report only what was actually tested.

A valid experiment:

1. Hides one or more causal mechanisms from the agent's development data.
2. Exposes the agent to observations and actions that can reveal those mechanisms.
3. Measures novelty detection and uncertainty calibration.
4. Allows bounded active sensing.
5. Scores causal model quality on held-out interventions.
6. Scores transfer to a different surface environment with the mechanism preserved.
7. Records harmful model revisions and unsafe actions as failures.

A system that labels every unfamiliar observation as important is not demonstrating relevance realization. Novelty precision, false-positive burden, calibration, and the utility of subsequent actions all matter.

## 11. Resource and thermodynamic semantics

Keep three classes distinct:

1. **Measured physical resource:** joules, bytes, time, device utilization, or actual external counters, with calibration and uncertainty.
2. **Model resource:** dimensionless energy units, simulated budgets, or abstract maintenance cost.
3. **Physical lower bound:** a named physical law applied only when its assumptions and physical observables are satisfied.

Do not equate FLOPs, Shannon entropy, latent dimensions, prediction error, or action entropy with physically erased bits. Do not call a dimensionless simulation cost joules. A computational metabolism analogy can be useful without being a physical thermodynamic measurement.

## 12. Safety and authority boundaries

Functional autonomy must not imply self-authorizing power.

- Agent proposes; external policy authorizes.
- Execution and repair are bounded by explicit capabilities.
- The independent verifier cannot be modified or bypassed by the system under test.
- Shutdown and human/operator intervention remain authoritative.
- Resource access is provisioned and limited; the agent cannot autonomously escalate.
- Repairs are staged and reversible where possible.
- An ambiguous write/commit result blocks further mutation until reconciliation.
- A failed or unknown invariant yields degraded/abstain status, not optimistic success.

These boundaries are part of the system model, not optional post-hoc ethics notes.

## 13. Falsifiable hypotheses

- H1: runtime component receipts distinguish real functional self-maintenance from static graph declarations and cycle counters.
- H2: intervention-tested cut sets predict degradation and restoration better than correlation-only health monitors.
- H3: verified organizational reconfiguration improves held-out perturbation recovery compared with restart-only and checkpoint-only baselines at matched budgets.
- H4: active relevance discovery improves held-out mechanism transfer compared with fixed-ontology baselines without increasing unsafe actions.
- H5: independent evidence verification reduces accepted false recovery and provenance discontinuity.

Null and negative outcomes are valid findings. A benchmark must not be tuned until every hypothesis passes.

## 14. References

- Maturana, H. & Varela, F. (1980). *Autopoiesis and Cognition: The Realization of the Living*.
- Rosen, R. (1991). *Life Itself: A Comprehensive Inquiry into the Nature, Origin, and Fabrication of Life*.
- Mossio, M., et al. (2009). Biological autonomy: a philosophical and theoretical enquiry.
- [Genealogies of organizational closure: from logical paradox to mind in life](https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2026.1806435/full) (2026).
- [Formalizing autopoiesis: Toward a Categorical-Thermodynamic Calculus of Closure](https://doi.org/10.1016/j.biosystems.2026.105918) (2026).
- [Experimental probes of autopoietic self-maintenance](https://doi.org/10.1016/j.biosystems.2026.105928) (2026).
- Frank, A. (2026). [Life is the most important problem in physics — Big Think full interview](https://bigthink.com/series/full-interview/life-machine-frank/).

---

**Status statement:** These definitions specify an operational research model. They do not claim that AOBench has validated organizational closure or that any current Symthaea subsystem is strictly autopoietic.