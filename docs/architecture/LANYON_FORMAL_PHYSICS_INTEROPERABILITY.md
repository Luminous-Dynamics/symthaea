# Lanyon Formal Physics Interoperability

## Status

Experimental interoperability target. This document describes a **loss-aware
adapter to the public Lanyon specification surface**. It does not claim
compatibility with Lanyon's private compiler, internal DSL grammar, or
proprietary neural/symbolic implementation.

Lanyon publicly describes a compact, scientifically-aware formal specification
language from which it deterministically expands solver implementations and
machine-checkable proofs. Its public solver repositories expose generated
Racket/Lisp specifications, Lean proofs, and C implementations.

## Why the fit is strong

Symthaea and Lanyon solve different halves of the same reliability problem.

Symthaea currently provides:

- physical quantity kinds and SI dimensions;
- explicit unit transforms, including affine temperature semantics;
- scalar-domain and refinement judgments;
- typed multi-physics topology;
- provenance and evidence identities;
- Ramanujan -> EUREKA hypothesis/challenge separation.

Lanyon provides a compact formal solver specification and a deterministic
verification/compilation pipeline. Its published examples cover coordinates,
state variables, parameter assumptions, fluxes, wave speeds, and diffusive
fluxes.

The intended composition is therefore:

```
Ramanujan / HDC discovery
        |
        v
Symthaea PhysicalType + provenance
        |
        v
typed physical solver specification
        |
        +---- semantic envelope (units/kinds/refinements)
        |
        v
Lanyon public formal specification
        |
        v
Lanyon /verify
        |
        v
Lanyon /compile
        |
        +---- Lean proof artifact
        +---- generated C implementation
        |
        v
external execution / simulation
        |
        v
independent evidence receipt
```

This keeps mathematical/physical semantic identity distinct from numerical
implementation correctness and from empirical agreement with the world.

## The crucial boundary

The public Lanyon specifications observed in July-October 2026 do not expose
the full Symthaea quantity ontology. In particular, the public examples are
compact system specifications rather than a complete SI quantity-kind and unit
ontology.

Therefore **do not flatten PhysicalType into a Lanyon expression string**.

Instead, the adapter emits two linked artifacts:

1. a deterministic Racket specification matching the public system shape;
2. a semantic envelope containing the exact Symthaea PhysicalType and its
   digest for each bound coordinate/state/parameter.

The bundle digest covers both artifacts.

A Lanyon verification result can then be attached to the bundle by an external
evidence layer without changing the original physical semantic identity.

## Mapping

| Symthaea | Lanyon public surface | Treatment |
|---|---|---|
| expression AST | Lisp/Racket S-expression | structural conversion |
| coordinate names | `coordinates` | direct |
| state variables | `state` | direct |
| positivity/domain constraints | assumptions | explicit formal expressions |
| model parameters | `parameters` | direct |
| dimensional/quantity semantics | no complete public equivalent observed | semantic sidecar |
| fluxes | `fluxes` | direct |
| characteristic speeds | `wavespeeds` | direct |
| diffusive terms | `diffusive-fluxes` | direct |
| solver implementation | generated C | external artifact |
| formal correctness | generated Lean | external artifact |
| physical-world agreement | empirical execution/evidence | never implied by proof |

## Loss rules

The adapter fails closed when a mapping is ambiguous.

For example, the current Symthaea `Expr::Sum` variant is rejected rather than
inventing a Lanyon syntax that has not been established from the public
artifacts.

Likewise, physical semantic bindings must reference a declared system symbol,
must pass `PhysicalType::validate()`, and must carry a digest matching the
exact serialized physical type.

## Recommended future integration

The most valuable collaboration point is not "make Symthaea speak Lanyon
internally." It is a **verified interchange contract**:

```
Canonical Physics Spec
  = equations
  + physical semantics
  + assumptions
  + topology
  + provenance
  + exact solver intent
```

Symthaea should own the canonical semantic/provenance envelope.

Lanyon should consume a deterministic solver projection of that envelope,
perform its formal specification -> proof -> implementation pipeline, and
return proof/implementation artifacts whose identities are bound back to the
same canonical specification digest.

This gives us a two-way evidence boundary without requiring either project to
adopt the other's internal architecture.

## Important qualification distinction

A successful Lanyon `/verify` result would establish a property of the formal
specification and the generated implementation under the formal assumptions.

It would **not** by itself establish that:

- the specification expresses the intended real-world problem;
- the chosen physical model is empirically true;
- measured parameters are correct;
- boundary/initial conditions match reality;
- the numerical model is adequate for the intended engineering use.

That distinction aligns with Lanyon's own public discussion of semantic
correctness as a separate problem from syntactic/formal correctness.

## Public references

- Lanyon: https://www.lanyon.ai/
- Lanyon Formulary: https://www.lanyon.ai/research/formulary/
- Lanyon Advection-Diffusion: https://www.lanyon.ai/research/advection-diffusion/
- Lanyon public solver repositories: https://github.com/lanyonai
