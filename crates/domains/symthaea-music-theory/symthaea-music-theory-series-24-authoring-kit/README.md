# Symthaea Music Theory — Patch Series 24 Authoring Kit

**Theme:** Bounded verification and malicious-artifact resistance.

Series 22 makes independent verification possible, and Series 23 proves the implementation and release chain. Series 24 ensures that a hostile public artifact cannot force unbounded memory, CPU, filesystem, subprocess, or signature-verification work before it is rejected.

## Core rule

All verification limits are supplied by the verifier or deployment. An artifact may declare its size, but it cannot authorize itself to consume more resources. Cheap structural and bound checks occur before canonicalization, hashing, external signature calls, lineage expansion, or archive extraction wherever semantics permit.

## Expected base

The exact Series 23 final tree with its demonstrated cumulative-integration evidence bundle.
