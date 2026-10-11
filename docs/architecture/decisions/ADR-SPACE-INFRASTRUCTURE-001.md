# ADR-SPACE-INFRASTRUCTURE-001: Thin Solar Infrastructure Layer

**Status:** Accepted for v0.1 implementation
**Date:** 2026-09-10

## Decision

Implement space infrastructure as a thin, typed composition layer over existing Symthaea engineering, orbital, digital-twin, safety, HAL, fabrication, and simulation-bridge capabilities.

Do not create a monolithic space subsystem and do not duplicate domain truth models already owned by existing crates.

## Rationale

The repository already has dedicated owners for orbital mechanics, engineering composition, digital twins, formal safety, hardware abstraction, fabrication authority, and external simulation backends. A new infrastructure layer should therefore model cross-domain assets, resources, services, observations, reservations, proposed actions, authorizations, and receipts, while delegating physics and machine execution to those owners.

The layer must also remain usable when Symthaea is absent. Intelligence may recommend actions, but it cannot become the implicit authority root for physical execution.

## Consequences

Positive:

- preserves the workspace DAG and feature-gated composition model;
- minimizes duplicated physics and control logic;
- allows conventional software and human operators to use the same infrastructure interfaces;
- supports evidence-bearing interoperability claims;
- keeps simulation evidence distinct from real-world qualification.

Trade-offs:

- more adapters are required between domains;
- cross-domain optimization must tolerate heterogeneous fidelity;
- authority promotion is intentionally explicit and therefore more verbose than direct command dispatch.

## Hard constraints

1. `CommandProposal` is descriptive and non-executable.
2. Missing or stale authority fails closed.
3. External-standard compatibility is represented as evidence-bearing metadata, never assumed from a string match alone.
4. Simulation-only results cannot be promoted to qualified real-world capability.
5. Local controllers remain responsible for immediate hardware safety and final command acceptance.
