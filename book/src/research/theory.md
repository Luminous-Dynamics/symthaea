# Theoretical Foundations

Symthaea integrates four theoretical frameworks into a single architecture.

## Integrated Information Theory (IIT)

IIT proposes that consciousness *is* integrated information (Phi). A system is conscious to the degree that it generates more information as a whole than the sum of its parts.

Symthaea computes Phi at every cognitive cycle via the Spectral MIP algorithm:
- Pairwise mutual information from rolling covariance
- Fiedler vector ordering (spectral relaxation of normalized cut)
- Bordered Cholesky sweep for O(n^3) total complexity
- At n=128 dimensions: ~5.5 ms on a single CPU core

**Validation**: Sampled partition Phi vs exact computation: r = 0.9998. The previously used algebraic connectivity (lambda_2) was found to be *anti-correlated* with true Phi (r = -0.14) — a critical correction.

## Hyperdimensional Computing (HDC)

All information is encoded as 16,384-dimensional vectors. The quasi-orthogonality of random high-dimensional vectors (cosine similarity ~0.0 with SD ~0.008) provides:

- **Exponential capacity**: ~1,170 items (Theta(D/log D))
- **Noise robustness**: corrupting 6% of dimensions reduces similarity by only ~0.06
- **Compositional structure**: binding (associations) + bundling (superpositions) + similarity

## Liquid Time-Constant Networks (LTC)

CfC neurons evolve through continuous-time ODEs with state-dependent time constants. The closed-form solution enables O(1) temporal jumps: evolving from t to t+dt costs the same regardless of dt.

This enables a single architecture to handle 500 Hz motor reflexes and 0.1 Hz deliberation without separate "fast" and "slow" systems.

## Active Inference (Free Energy Principle)

The system minimizes variational free energy through both perception (updating beliefs) and action (changing the world). This naturally balances exploration and exploitation and provides a principled account of attention allocation.
