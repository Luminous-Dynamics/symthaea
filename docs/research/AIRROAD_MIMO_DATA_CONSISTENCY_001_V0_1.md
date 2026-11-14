# AIRROAD-MIMO-DATA-CONSISTENCY-001 — bounded input/output data to MIMO model-error sets

**Status:** V0.1 exact-rational single-frequency data-consistency reference  
**Tracking issue:** https://github.com/Luminous-Dynamics/symthaea/issues/11891  
**Composes with:** #11882 continuous rational certificate, #11885 Lipschitz envelope, #11858 shared calibration, #11801 robust MIMO stability, ENG-MEAS #4919 and ROB-SYSID #4801

## 1. Research basis

Hindi, Seong and Boyd show that data-consistent MIMO frequency-domain uncertainty models can be derived through LMI feasibility/SDP while accounting for bounded noise and fitting error: https://web.stanford.edu/~boyd/papers/unc_fdd.html

Frequency-domain MIMO identification literature also treats bounded unstructured model error using input/output disturbance bounds and H-infinity/structured uncertainty methods: https://pure.tue.nl/ws/portalfiles/portal/3525408/388453.pdf

AIRROAD does not claim to invent these techniques. This work defines an exact small-instance evidence adapter: input excitation, observed output, hard noise set, LMI, full-rank coverage, model identity and the scope of the resulting uncertainty claim.

## 2. Single-frequency data-consistency contract

At one frequency knot, nominal model G0 and experiment input u_i produce a predicted output G0*u_i. The measured output y_i gives nominal residual:

    r_i = y_i - G0*u_i

For deterministic hard output-noise radius epsilon_i:

    ||r_i,true||_2 <= ||r_i||_2 + epsilon_i

For additive uncertainty, a sufficient squared gain gamma_i satisfies:

    gamma_i >= (||r_i||_2 + epsilon_i)^2 / ||u_i||_2^2

The corresponding bounded-data LMI is:

    [ gamma_i I_p       r_i,worst ]
    [ r_i,worst*        u_i* u_i ] >= 0

The oracle checks the exact symmetric block matrix with rational arithmetic and verifies positive semidefiniteness via all principal minors. The contract keeps gamma as a squared gain bound, not an amplitude bound.

If the input vectors span the complete input space, per-column error bounds can yield a conservative full-matrix outer bound. For U=I₂:

    ||E||_2^2 <= ||E||_F^2 <= sum_i gamma_i

This uses no independence assumption between experiment errors: it follows by summing the separate hard column bounds. It is conservative; tighter spectral bounds need their own certificate.

## 3. Exact 2x2 fixture

Nominal model:

    G0 = [1 0]
         [0 1]

Inputs are the two basis vectors e1 and e2, so U=I₂ and det(U)=1. Observed outputs are:

    y1 = [1, 1/10]^T
    y2 = [1/10, 1]^T

Both output observations have deterministic Euclidean hard-error radius epsilon=1/100. Nominal residuals are:

    r1 = [0, 1/10]^T
    r2 = [1/10, 0]^T

The worst-case residual magnitude is 11/100 and each input has squared norm 1. Therefore:

    gamma_1 = gamma_2 = (11/100)^2 = 121/10000

Each exact 3x3 LMI is positive semidefinite; one Schur-complement eigenvalue is exactly zero. The total column-wise bound is:

    ||E||_F^2 <= 242/10000 = 121/5000
    ||E||_2^2 <= 121/5000

This is a conservative full-matrix spectral-norm-squared bound at one frequency knot.

## 4. What this does not prove

- One frequency knot does not establish a continuous-frequency uncertainty envelope.
- One input direction does not bound a full MIMO operator over every input direction.
- Statistical confidence intervals are not deterministic hard noise balls.
- A nominal residual without calibration, model and topology identity cannot enter the data-consistency result.
- The Frobenius-derived spectral outer bound is not the exact spectral norm unless separately established.

The pointwise output must be combined with an independently justified continuous-domain certificate (#11882, #11885 or equivalent) before #11801 consumes it for robust stability.

## 5. Negative controls and benchmark targets

The exact oracle rejects gamma below the required bound, zero input with nonzero residual, rank-deficient excitation for a full-matrix claim, missing/stale noise or model identity, statistical-only noise semantics, and promotion of a pointwise certificate to a continuous-band result.

Targets, not achieved results:
- 100% exact Fraction agreement with the two LMI fixtures;
- zero PSD acceptance below gamma=121/10000;
- zero full-matrix bounds from rank-deficient input ensembles;
- zero statistical-to-hard-bound promotion;
- zero pointwise-to-continuous promotion without a separate full-band method;
- report gamma, noise radius, input rank, Frobenius/spectral norm semantics, proof runtime and unsupported cases separately.

## 6. Claim ceiling

V0.1 is one exact synthetic frequency-knot data-consistency reference. It does not implement general LMI/SDP optimization, infer physical noise bounds, qualify real measurements, prove continuous-frequency model error, establish robust aircraft stability, or create aircraft safety/airworthiness/certification evidence.
