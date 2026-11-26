# Kinetic Intervention Rate–Distortion point schema v0.1

This wire contract describes one candidate checkpoint measurement for #13420.

It records:
- physics cohort and initial-state identity;
- task contract, intervention-family digest, QoI registry, horizon and split;
- representation identity;
- serialized checkpoint payload and required decoding metadata;
- model/decoder assets, amortization rollout count and peak working-set memory separately;
- encode/decode/advance/observe/transition/fallback/training time and optional energy;
- separate microscopic, weak-observable, macro-QoI, intervention-response and sensitivity dispositions;
- physical gates, finite response-probe coverage / optional global-bound evidence and provenance.

The schema is a wire format only. A semantic auditor must verify byte sums, source and receipt identities, physical-reference convergence, branch association and split leakage.

A finite probe library is empirical coverage, not a certified global response bound. A global-bound claim requires a separately justified method and digested evidence artifact.

The fixture is synthetic-only. It has placeholder all-zero digests, no physical simulation, no measured distortion and physical_execution_authorized=false. It must not be included in a scientific scorecard.
