# Cover Letter — Ramanujan Protocol

For workshop/venue submissions that request a short narrative pitch.
arXiv itself does not require one.

---

Dear editors,

I am submitting "Ramanujan Protocol: Autonomous Conservation-Law Discovery"
for consideration. The paper describes a reproducible pipeline that takes a
dynamical system as input and returns a symbolically verified conservation
law as output, using genetic symbolic regression, chain-rule differentiation,
and Z3 formal proofs in combination.

The paper's distinguishing feature is not that it solves any single
problem — several of the six formally proved invariants are textbook
results — but that a single deterministic pipeline, under a single random
seed, produces machine-checkable proofs across a heterogeneous set of
dynamical systems (harmonic oscillator, Lotka–Volterra, Kepler two-body,
Hénon–Heiles chaos, anisotropic coupled oscillator, triangular-number
sequence). The seventh problem in our test set (the circular restricted
three-body problem's Jacobi integral) is honestly reported as a failure:
the pipeline returns a numerically-conserved-but-semantically-wrong
expression, and the paper documents the twelve-session research arc that
traced the failure back to the fitness function itself rather than
papering over it.

This honest failure analysis is, in my view, the paper's single most
important contribution to the AI-for-science literature. Most autonomous-
discovery papers present curated successes; this paper presents six
successes, one instructive failure, and the full debugging trace in
between. Readers who want to extend the work know exactly where the
ceiling is.

Reproducibility is a first-class concern. A single Docker container
regenerates the paper's entire results table in under thirty minutes.
Nine Z3 witness files are supplied as ancillary material and can be
verified independently using any SMT-LIB2-compliant solver. The full
reproduction harness, including a deterministic seed (= 42), is
documented in VERIFY.md.

I believe the paper fits naturally into the scientific-discovery workshop
lineage (NeurIPS / ICML AI4Science, ML4PS) and complements recent work on
symbolic regression (SRBench, PySR) by demonstrating how SMT-based
formalisation changes what a "discovery" pipeline can claim.

I would be grateful for your consideration.

Sincerely,
Tristan Stoltz
Luminous Dynamics
tristan.stoltz@evolvingresonantcocreationism.com
