# symthaea-earth-causal

Evidence-preserving bridge from `symthaea-earth-observation` into `symthaea-causal-reasoning`.

## Load-bearing rule

**Association is not causation.**

Registering observations, measurements, derived features, or hypotheses creates causal variables only. Recording an `EvidenceAssociation` leaves the causal DAG unchanged. A directed edge can be added only through an explicit `StructuralEdgeClaim` with a declared basis:

- domain structural assumption;
- controlled intervention with supporting evidence;
- separately validated structural model with supporting evidence.

An identified total causal effect is not automatically evidence of a direct edge.

## Why this bridge exists

Planetary Perception needs to feed real Earth evidence into Symthaea's existing causal and counterfactual machinery without collapsing the epistemic distinctions established by `symthaea-earth-observation`.

The intended flow is:

```text
Earth observation
      |
      v
measured / derived evidence
      |
      v
symthaea-earth-causal
      |
      +--> descriptive associations (no DAG mutation)
      |
      +--> reviewed structural claims
                |
                v
       symthaea-causal-reasoning
                |
                v
       identified / unidentified /
        assumption-required query
```

## Deliberate non-scope

This crate does not:

- discover causal structure from raw observational correlations;
- infer causal edges from HDC similarity;
- treat a hypothesis as an observed variable;
- estimate causal effects;
- run do-calculus itself;
- choose policy interventions;
- convert simulation output into real-world authority.

Those responsibilities remain with the causal engine, model-specific adapters, and later bounded decision-support layers.
