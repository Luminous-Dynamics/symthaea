# LQCD-020T — independent cubic-orbit enumeration oracle

Exact executed standard-library subject SHA-256:

`0fc173257a6a189bd99c9ed7f141ea71bf08c00784c1fcf83ef554feef0716b7`

Canonical result SHA-256:

`0bd498266d54b7014426ee23c5713215c40e56af3b2cf1bc0cb2298ae12904ae`

The oracle qualifies the deterministic unique signed-permutation orbit used to average already-qualified off-axis Wilson-loop measurements on an isotropic cubic spatial lattice. It does **not** reimplement SU(3) transport; the gauge/operator authority remains LQCD-020L/020M.

Frozen orbit sizes / per-orientation shortest-path counts include:

- `(1,0,0)`: 6 orientations, 1 shortest path each;
- `(1,1,0)`: 12 orientations, 2 paths each;
- `(1,1,1)`: 8 orientations, 6 paths each;
- `(2,1,0)`: 24 orientations, 3 paths each;
- `(3,3,3)`: 8 orientations, 1680 paths each;
- `(6,3,0)`: 24 orientations, 84 paths each.

Alias tests prove signed/permuted representatives canonicalize to the same orbit. Zero displacement fails closed. Within one orbit, every orientation has the same shortest-path multiplicity, so the existing 020M `max_paths` budget has a consistent interpretation across the orbit.

## Authority boundary

This establishes cubic-orbit combinatorics only. It does not prove equilibrium, rotational restoration, continuum symmetry, a static potential, or a string tension. Orbit averaging is an estimator construction layered over the independently qualified off-axis Wilson operator.
