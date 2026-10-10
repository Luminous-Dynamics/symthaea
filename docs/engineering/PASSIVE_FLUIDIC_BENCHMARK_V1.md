# Passive Fluidic Benchmark v1

`symthaea-passive-fluidic-benchmark` is the first concrete benchmark adapter for
fixed-geometry fluidic rectification.

## What it evaluates

The benchmark accepts paired forward/reverse results from an actual backend:

`pressure_drop_forward`, `flow_rate_forward`
`pressure_drop_reverse`, `flow_rate_reverse`

It computes:

`diodicity = Δp_reverse / Δp_forward`

and separately validates that the paired flow-rate magnitudes agree within an explicit
relative tolerance.

## Why pairing is mandatory

A pressure-drop ratio is not meaningful as a directional performance comparison when
the two simulations or measurements use substantially different flow rates. The
benchmark therefore rejects mismatched operating points instead of silently normalizing
them away.

## Physics boundary

This crate deliberately does not pretend to be a CFD solver. It consumes outputs from
the existing solver ecosystem and makes the comparison contract deterministic.
Reynolds number calculation delegates to the existing `symthaea-thermofluids` crate.

## First research target

The natural first serious case is a fixed-geometry Tesla-style or topology-optimized
fluidic rectifier. Recent 2026 work reports substantial numerical diodicity improvements
from topology optimization of fixed-geometry thermofluidic diodes, including gravity-aware
formulations. The research result is a candidate-generator/solver benchmark target, not
a value that this library assumes.

## Required evidence before promotion

Before a candidate can be treated as a real engineering result, record:

1. geometry digest;
2. solver backend and version;
3. mesh/tessellation configuration;
4. boundary conditions and fluid properties;
5. matched forward/reverse operating points;
6. convergence evidence;
7. manufacturability evidence;
8. structural/durability evidence where relevant;
9. independent reproduction or physical measurement.

The benchmark metric is therefore an observation in the evidence chain, not the evidence
chain itself.