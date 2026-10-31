# PITB v3 support-landscape hardening (2026-10-31)

**Status:** v3 contract and synthetic regression controls added to the research branch. No real Lean rollout ledger, Lean metaprogram build, or Actions result is claimed.

## Two defects found by source review

The v2 analyzer had two issues that its not-yet-executed workflow self-test should have exposed:

1. **Synthetic fixture construction mixed fixed declarations into the mutable universe.** The fixture helper built the powerset over the union of mutable and fixed IDs, then labelled that union as mutable. The production validator correctly rejects this because the mutable and fixed sets overlap. v3 now enumerates only the mutable declarations, retains the fixed declarations in every context, and keeps the full telescope order over both sets.
2. **Replicate seed reuse was not rejected.** v2 required the same seed digest at a given replicate index across contexts (the pairing requirement), but did not require distinct seeds across replicate indices. A ledger could therefore reuse one seed for multiple trial numbers and inflate the apparent sample size. v3 now requires a one-to-one mapping between replicate indices and seed digests, while retaining the same seed at each index across the different contexts.

Distinct seed digests are a necessary integrity condition, not proof of statistical independence. The runner must also use a frozen, documented seed-generation protocol and independent random streams; the Clopper-Pearson interpretation continues to assume independent Bernoulli outcomes within each context.

## v3 artifacts

- Analyzer: scripts/research/pitb_support_landscape_v3.py
- Trial schema: data/benchmarks/proof-insight-transfer-v0/support-landscape-trial-v3.schema.json
- Focused workflow: .github/workflows/pitb-support-landscape-v3.yml

The v3 self-test includes the existing alternatives, conjunction, dependency closure, fixed-declaration closure, stochastic reliability, incomplete-lattice, paired-seed, budget, and semantic-gate controls, plus a negative control that reuses one seed across two replicate indices. The workflow compiles the exact committed script before running self-tests and validates the exact v3 Draft 2020-12 schema.

## Important trust boundary

The analyzer verifies digest syntax, group binding, budgets, schedule completeness, context closure, and statistics. It does not authenticate the referenced receipt bytes or prove that a digest came from a trusted runner, Lean kernel, semantic gate, or cost meter. A SHA-256 string is an integrity identifier, not a signer identity. That limitation remains explicit and is the subject of the next independent decision-certificate track.
