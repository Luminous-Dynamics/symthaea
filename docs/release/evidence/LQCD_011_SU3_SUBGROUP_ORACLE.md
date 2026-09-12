# LQCD-011 SU(3) subgroup proposal oracle

## Executed subject

`scripts/lqcd-su3-subgroup-oracle.py --self-test`

The standard-library oracle was executed independently before commit. It imports no Symthaea code.

## Qualification fixture

- lattice: `2 x 2 x 1 x 1`
- Wilson `beta = 6.0`
- updated link: `(0,0,0,0), mu=0`
- embedded subgroup: color pair `(0,1)`
- axis: `(1,2,3)` normalized internally
- angle: `0.2`

Observed self-test output:

```text
ok
max_unitarity_error=0
max_determinant_error=0
max_inverse_error=0
delta_action_full=0.15946737727006699
delta_action_local=0.15946737727006699
local_delta_parity_error=0
forward_acceptance=0.85259778100989414
restoration_action_error=0
restoration_link_error=0
```

Pre-commit oracle SHA-256 from the executed local copy was:

`08af9a4aec054c7219e71cab507bd2dd7eac44e4dfa2cdae30a0b0a2244ad2f7`

The checked-in file should be treated as the normative source; the digest above records the locally executed pre-commit copy and must not be silently substituted for a later changed file.

## Semantics established

- all three canonical SU(2) color-pair embeddings `(01)`, `(02)`, `(12)` remain in SU(3);
- changing the proposal angle sign returns the exact dagger/inverse for the deterministic fixture;
- left multiplication preserves SU(3) membership;
- the six plaquettes touching one link are sufficient for the local Wilson-action difference;
- local and full-action differences agree on the qualification fixture;
- Metropolis acceptance is `min(1, exp(-Delta S))`;
- applying the inverse proposal restores both the link and the full Wilson action.

## Non-claims

This does **not** establish a production random proposal law, SU(3) ergodicity, detailed balance for an implemented sweep scheduler, thermalization, autocorrelation control, an equilibrium gauge ensemble, continuum physics, string tension, or a glueball mass. A production updater must separately bind an RNG/proposal distribution whose forward and reverse densities are demonstrably symmetric (or account for a Hastings ratio).

## References

The architecture follows the Cabibbo-Marinari strategy of updating SU(3) through embedded SU(2) subgroups. Later heat-bath work should be qualified separately rather than inheriting Metropolis evidence.
