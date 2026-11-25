# Intervention-equivalent kinetic state fixture v0.1

This synthetic fixture provides three serializable response signatures under one frozen intervention / QoI / horizon / structure contract.

State A has response 0 at both times, B has response 0.75, and C has response 1.5. The task-fixed normalization scale is 1 and pairwise threshold is 1:

- d(A,B)=0.75: pairwise compatible
- d(B,C)=0.75: pairwise compatible
- d(A,C)=1.5: not pairwise compatible

The fixture is a minimal counterexample to treating approximate pairwise similarity as a transitive equivalence relation. A clustering algorithm that unions compatible edges would put A, B and C in the same class despite the A–C diameter exceeding tolerance.

The signature's structure class, intervention ID, QoI ID, time grid, branch IDs and complex responses are all explicit. The contract must match exactly before comparison. The fixture is analytic only; it contains no Vlasov reference output or physical state pair.

See [#13503](https://github.com/Luminous-Dynamics/symthaea/issues/13503).

## Per-QoI normalization

The response-signature schema includes a positive, finite scale for each registered QoI. The comparison divides each complex residual by the scale of its QoI before taking the RMS across the fixed intervention/QoI/time grid. Signatures with different scale vectors cannot be compared. This prevents unit mismatches (for example field amplitude versus energy) and prevents a caller from changing the comparison normalization after signatures are created.

The Rust tests include a two-QoI known-answer case with scales [1, 10] and a negative control rejecting a pair whose scale vectors differ.


## Finite-class diameter audit

Pairwise approximate compatibility remains useful as a diagnostic, but it is not transitive. The Rust API now reports both:
- normalized RMS distance across the registered response grid (useful as a summary);
- maximum normalized complex-response error at any individual intervention/QoI/time entry (the admission criterion).

`pairwise_compatible` uses the maximum per-entry error, not RMS, so a single large future-response miss cannot be diluted by many small entries.

`audit_response_signature_cluster` checks every pair in a proposed finite class. It reports the number of compatible/incompatible pairs, maximum RMS and per-entry errors, completeness, branch stability and structure-class compatibility. A class is admitted only when every pair satisfies the frozen tolerance and all identity / observation gates pass. The audit has bounded member and comparison-work budgets.

For A/B/C, A–B and B–C pass at tolerance 1, but A–C fails at 1.5. The complete three-member class is therefore rejected even though connected-component union-find would merge all three. This is a finite empirical diameter check, not a proof of equivalence for untested kinetic states or interventions.
