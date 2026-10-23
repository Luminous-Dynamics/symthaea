# LQCD-016B independent subgroup overrelaxation oracle

## Executed subject

`scripts/lqcd-su3-overrelaxation-oracle.py --self-test`

The standard-library Python oracle was executed independently before commit. It imports no Symthaea/Rust code.

Local executed copy SHA-256:

`40d748a47341db1af1a1801fca31799879aec6e774fac56003334f16ec4bebd8`

The checked-in source is the normative subject; the digest records the exact local copy that was executed before commit and must not be silently substituted for a later revision.

## Qualification fixture

- lattice: `2 x 2 x 2 x 2`
- Wilson beta: `5.7`
- target link: `(0,0,0,0), mu=0`
- four deterministic nontrivial fixture links are installed before the test
- all three Cabibbo-Marinari subgroup pairs `(01)`, `(02)`, `(12)` are tested

Observed output:

```text
ok
fixture_action=2.9199835510572014
pair_01_constant=5.9637708963658911
pair_01_force=11.366866478675853,-0.95765431664647327,-1.9596989138847434,-3.0299334156546145
pair_01_reflection=0.80525833990558371,-0.15209235061534931,-0.31123465861336203,-0.48120672291087679
pair_01_action_error=3.9968028886505635e-15
pair_01_trace_error=0
pair_01_involution_link_error=3.3422138886441676e-16
pair_01_involution_action_error=7.9936057773011271e-15
pair_02_constant=5.7140014193142807
pair_02_force=11.616635955727462,-0.39855539573736998,0.26586311875638025,-1.6712797681200611
pair_02_reflection=0.95618280840072045,-0.067114715164366678,0.044770006074143331,-0.28143506974689153
pair_02_action_error=7.1054273576010019e-15
pair_02_trace_error=7.1054273576010019e-15
pair_02_involution_link_error=6.6207375370311646e-16
pair_02_involution_action_error=7.5495165674510645e-15
pair_12_constant=5.6528650593615719
pair_12_force=11.677772315680171,0.04439028059179595,-8.8817841970012523e-16,1.3586536475345525
pair_12_reflection=0.97326098576938058,0.0075008834280431676,-1.5008066407008718e-16,0.22957959475313186
pair_12_action_error=0
pair_12_trace_error=0
pair_12_involution_link_error=2.2377260456559048e-16
pair_12_involution_action_error=3.5527136788005009e-15
```

## Construction

For one SU(2) subgroup, the contribution of the touched plaquettes is affine in the unit quaternion `a=(a0,a1,a2,a3)` representing the subgroup rotation:

`T(a) = c + q . a`.

The oracle reconstructs `c` and `q` without assuming a staple orientation convention, using the five probes `+I`, `-I`, `i sigma1`, `i sigma2`, and `i sigma3`. The equal-action reflection farthest from the identity is

`a' = 2 (q0 / |q|^2) q - e0`.

It then verifies both the touched-plaquette trace and the full Wilson action are unchanged and that applying the reflection twice returns the original link.

## Semantics established

- the subgroup overrelaxation reflection is an SU(2) element embedded in SU(3);
- all three canonical SU(2) subgroups preserve the Wilson action on the nontrivial fixture to floating-point precision;
- the update is an involution on the fixture to floating-point precision;
- the reference construction is independent of hand-coded staple orientation and is therefore suitable as a parity oracle for a later optimized staple implementation.

## Non-claims

This is not an ergodic sampler by itself. Pure overrelaxation does not replace a stochastic ergodic transition kernel in the current Symthaea program. This evidence does not establish equilibrium, thermalization, adequate ESS, topological tunneling, continuum scaling, string tension, or glueball masses.

## References

The design follows the standard microcanonical overrelaxation idea and Cabibbo-Marinari SU(2)-subgroup composition. In production lattice gauge calculations, heat-bath updates are commonly interleaved with multiple overrelaxation sweeps; this oracle qualifies only the deterministic equal-action subgroup reflection.
