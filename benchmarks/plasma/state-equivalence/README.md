# Intervention-equivalent kinetic state fixture v0.1

This synthetic fixture provides three serializable response signatures under one frozen intervention / QoI / horizon / structure contract.

State A has response 0 at both times, B has response 0.75, and C has response 1.5. The task-fixed normalization scale is 1 and pairwise threshold is 1:

- d(A,B)=0.75: pairwise compatible
- d(B,C)=0.75: pairwise compatible
- d(A,C)=1.5: not pairwise compatible

The fixture is a minimal counterexample to treating approximate pairwise similarity as a transitive equivalence relation. A clustering algorithm that unions compatible edges would put A, B and C in the same class despite the A–C diameter exceeding tolerance.

The signature's structure class, intervention ID, QoI ID, time grid, branch IDs and complex responses are all explicit. The contract must match exactly before comparison. The fixture is analytic only; it contains no Vlasov reference output or physical state pair.

See [#13503](https://github.com/Luminous-Dynamics/symthaea/issues/13503).