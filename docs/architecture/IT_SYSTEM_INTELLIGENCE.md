# IT Systems Intelligence Architecture

## Purpose

Symthaea already contains substantial IT-support, log-analysis, observability, NixOS, security, causal-reasoning and infrastructure-continuity work. The goal of this architecture is to connect those capabilities without creating parallel domain silos.

The central distinction is:

```text
KnowledgeGraph
    what generally tends to be true

SystemStateGraph
    what appears to be true about this environment now

Evidence
    what observations justify those state beliefs

Causal hypotheses
    what may explain the observations

Authority
    what changes are permitted
```

These are not interchangeable.

```text
knowledge != current state
observation != truth
confidence != currentness
diagnosis != authority
proposed change != execution permission
command accepted != desired outcome verified
```

## Existing ownership to preserve

| Concern | Existing owner / direction | Rule |
|---|---|---|
| IT support workflow | `crates/domains/symthaea-support` | Extend rather than replace |
| Windows/syslog learning wedge | `crates/core/symthaea-logparse` | Feed normalized evidence/state |
| Symthaea self-observability | `crates/domains/symthaea-observability` and existing observability modules | Do not turn SystemStateGraph into a second tracing framework |
| Nix/NixOS intelligence and enforcement | Nixward and existing Nix modules | Reuse typed Nix evidence/actions; do not duplicate Nix policy |
| Long-lived semantic knowledge | existing KnowledgeManager / knowledge graph | Keep general truth distinct from current environment state |
| Network workload intent/enforcement | Nixward Network Covenant | Do not duplicate packet/capability policy |
| Network-device/fabric continuity | issue #1100 program | SystemStateGraph may consume/serve observations; continuity qualification remains separate |
| Physical/actuation authorization pattern | issue #669 and related authority work | Reuse bounded/revocable capability semantics for future IT remediation |

## SystemStateGraph V1 boundary

`SystemStateGraphV1` lives initially in `symthaea-support` because `symthaea-logparse` already identifies a typed SystemStateGraph as the intended Phase-2 integration point.

Do not extract a generic crate until at least two independent domains need the stable same abstraction.

V1 owns:

- canonical entity identity inside a modeled environment;
- typed dependency/topology relations;
- normalized current attributes;
- immutable observation identities;
- provenance for every observation;
- explicit observation/currentness timing;
- graph-local revisions;
- evidence references for projected state;
- referential/evidence integrity checks.

V1 does **not** own:

- semantic truth or encyclopedia knowledge;
- arbitrary execution authority;
- vendor configuration translation;
- packet enforcement;
- network continuity qualification;
- an incident ticketing UI;
- a new telemetry transport;
- a new distributed database.

## Initial entity vocabulary

```text
Organization
Site
Network
Device
Host
VirtualMachine
Container
Process
Service
Storage
Identity
Policy
Certificate
Software
Configuration
TelemetrySource
Change
Incident
Custom
```

The vocabulary is intentionally provider-neutral. Adapters may preserve provider-specific fields as attributes or `Custom` kinds until stable cross-provider semantics are demonstrated.

## Initial relation vocabulary

```text
runs_on
depends_on
connects_to
routes_via
resolves_via
authenticates_via
authorized_by
mounted_from
replicates_to
deployed_from
configured_by
observed_by
changed_by
member_of
protected_by
custom
```

These edges are the basis for future dependency, blast-radius and causal reasoning.

## Evidence/currentness model

An observation records source identity, source kind, collector identity/version, schema/model version, optional immutable artifact digest, confidence, source-declared event time, collector observation time, optional ingestion time, maximum allowed age and clock uncertainty.

Important distributed-systems rule:

```text
source event time
    != collector observation time
    != ingestion time
```

Remote event clocks are not assumed to be strictly ordered with the collector clock.

If a source does not define a currentness horizon, the graph returns `Indeterminate`; it must not silently treat the observation as fresh.

## Projection rule

Adapters should follow:

```text
raw source
    -> source-specific parser
    -> normalized immutable observation
    -> record evidence
    -> project entity/relation state
```

The observation is retained separately from the mutable projection so later reasoning can reconstruct why a state belief existed.

## Near-term integration sequence

### ITK-0 — ownership map

This document. Prevent architectural duplication before adding adapters.

### ITK-1 — SystemStateGraphV1

Introduce typed entities, relations, graph revisions and traversal primitives.

### ITK-2 — evidence/provenance/currentness

Bind graph projection to immutable normalized observations. Keep confidence separate from freshness and clock uncertainty.

### ITK-3 — support knowledge convergence

Refactor `symthaea-support::knowledge` toward a domain view over the main knowledge substrate rather than a growing independent knowledge engine. Preserve privacy/local-only semantics.

### ITK-4 — technology/version/applicability identity

Add canonical technology identities and claim applicability so a fact can distinguish product, edition, version, build, platform, feature/profile and configuration context.

### ITK-5 — change/event timeline

Model deployments, configuration changes, package updates, policy changes, certificate rotations, routing changes, service restarts and hardware/topology changes as first-class evidence/state.

Primary query:

```text
what changed before the failure?
```

### ITK-6 — causal hypothesis graph

Represent competing root-cause hypotheses separately from state. A hypothesis is an inference over evidence, never an observation itself.

### ITK-7 — diagnostic information gain

Upgrade fixed diagnostic heuristics into posterior/EIG-driven test selection where evidence supports it.

### ITK-8 — external telemetry normalization

Prefer adapters into existing ecosystems, beginning with OpenTelemetry resource/log/metric/trace semantics where applicable.

### ITK-9 — standards/source registry

Version RFC/IANA/OpenConfig/vendor/NIST/OWASP/etc. knowledge and transformations explicitly.

### ITK-10 — defensive ontology bridges

Map MITRE ATT&CK/D3FEND concepts into canonical Symthaea identities rather than inventing an isolated cyber ontology.

### ITK-11 — packet-capture modality

Normalize packet captures into flows, protocol-state observations and typed anomalies rather than using packet text as the reasoning substrate.

### ITK-12 — network continuity bridge

Connect live topology/state evidence to the network continuity program from #1100 without moving continuity or execution authority into `symthaea-support`.

### ITK-13 — dependency/blast-radius reasoning

Traverse typed dependencies to estimate directly and transitively affected entities before remediation.

### ITK-14 — bounded IT remediation capability

Reuse the authority theorem already established elsewhere:

```text
feasible change
    != low-risk change
    != selected change
    != authorized change
    != executed change
    != verified outcome
```

No persisted/deserialized record should recreate live execution authority.

### ITK-15 — golden incident corpus

Build reproducible incident fixtures with topology, config, logs, metrics, traces, packet captures, change history, distractors, ground truth, correct diagnostics, unsafe diagnostics, remediation, rollback and verification.

## First integration targets

1. `symthaea-logparse` -> normalized observations -> SystemStateGraph.
2. `symthaea-support::diagnostics` consumes typed current state instead of only symptom strings.
3. `symthaea-support::predictive` consumes graph-derived telemetry/dependency context.
4. Nix/Nixward adapters publish exact configuration/service/package observations.
5. Network continuity adapters consume/emit state and evidence without granting authority.

## Qualification invariants

At minimum, future tests should prove:

1. identical observation replay is idempotent;
2. the same observation ID cannot be rebound to different evidence;
3. projected state cannot cite nonexistent evidence;
4. relations cannot reference nonexistent endpoints;
5. one entity ID cannot silently change semantic kind;
6. one relation ID cannot silently change endpoints or kind;
7. undefined currentness is `Indeterminate`, not `Fresh`;
8. stale evidence cannot satisfy consumers that explicitly require fresh state;
9. source clock disorder does not get rewritten into false causal ordering;
10. a diagnosis/hypothesis cannot mint execution authority;
11. simulation evidence cannot be relabeled hardware/production evidence;
12. adapter-specific unknowns remain unknown rather than being inferred as absence.

## Extraction criterion

Keep the implementation in `symthaea-support` until another domain needs the same stable types. Extract a generic `symthaea-system-model` only when reuse is concrete and the abstraction has survived at least two independent adapters.

That avoids premature generalization while preserving an intentional path to a shared world-model substrate.
