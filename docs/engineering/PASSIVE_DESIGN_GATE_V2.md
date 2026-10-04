# Passive Design Gate v2

## What changed

An additive workspace crate, `symthaea-passive-design-gate`, now provides an executable
evidence-coverage layer for no-moving-parts design claims.

The gate deliberately does not infer kinematics from a mesh. It first asks whether the
five safety-critical fields were supplied at all:

- moving solid components
- mechanical joints
- active power
- commanded actuators
- external control requirement

Explicit zero/false values count as evidence. Missing values remain missing.

## Why this boundary matters

The fabrication kernel's lower-level passive evidence API can already validate structured
facts and contradictions. The new crate makes the missing-field rule explicit without
forcing a large in-place kernel refactor while the API is still settling.

This also gives us an isolated CI target for the epistemic gate before it is promoted into
the main candidate evaluator.

## Intended integration

```text
GeometricThought / CSG
        |
        v
structured passive facts
        |
        v
evidence coverage gate
        |
        v
passive contract
        |
        v
physics + manufacturability + material objectives
        |
        v
candidate observation / Pareto archive
        |
        v
Mycelix provenance record
```

The current crate is intentionally an adapter rather than a second physics engine.
Physics remains in the fabrication kernel and solver bridges.

## Search-space direction

Current inverse-design research strongly supports graph/topology-aware representations for
large discrete design spaces, especially when nonlinear responses and manufacturing
constraints matter. A physics-guided diffusion line of work published in September 2026
also reinforces a useful design principle: the governing physics should guide generation
directly rather than being treated as an after-the-fact label.

For Symthaea, the compatible path is not to replace HDC with a neural latent model. Instead:

1. represent CSG/topology as a typed graph;
2. bind graph role, geometry, material, and boundary-condition symbols into HDCs;
3. retrieve diverse prior candidates and known failures;
4. mutate/generate candidates under hard geometry and passive constraints;
5. evaluate with exact or explicitly bounded solver envelopes;
6. preserve the Pareto frontier rather than collapsing everything to one score;
7. write the surviving evidence chain to Mycelix.

## Scientific boundary

The gate proves only that the evidence and contract are internally eligible for evaluation.
It does not prove useful function, safety, durability, manufacturability under every process,
or physical performance outside the tested operating envelope.

That distinction is intentional.