# SDCB robust composite-null test induction M0

**Status:** analytic exact-rational finite-tree fixture; Rust reference authored but not compiled or CI-qualified.

Machine-readable fixture: `docs/research/SDCB_COMPOSITE_NULL_TEST_INDUCTION_V1.json`.

## The key construction

For a fixed horizon N and terminal randomized test probability \(\phi_N\in[0,1]\), define one conditional-expectation process per null member:

\[
g_n^{(j)}=\mathbb E_{P_j^\pi}[\phi_N\mid\mathcal F_n].
\]

Then define the robust process by the pointwise lower envelope:

\[
g_n=\min_j g_n^{(j)}.
\]

For any true null member k and any bounded public stopping time \(\tau\leq N\),

\[
g_\tau\leq g_\tau^{(k)}
\quad\Longrightarrow\quad
\mathbb E_{P_k^\pi}[g_\tau]
\leq
\mathbb E_{P_k^\pi}[g_\tau^{(k)}]
=
\mathbb E_{P_k^\pi}[\phi_N]
\leq\alpha.
\]

At the terminal horizon each member-specific conditional expectation equals the terminal test (on its defined support), so \(g_N=\phi_N\).

The lower envelope is not asserted to be a supermartingale under each member. The guarantee is an all-bounded-stopping-times expectation inequality from pointwise domination of the true member's martingale.

## Frozen exact challenge

Null laws:
- B1: iid Bernoulli(1/5)
- B2: iid Bernoulli(4/5)

Terminal test:
- Observe X1 and X2.
- If X1 differs from X2, terminal rejection probability is 5/32.
- Otherwise it is zero.
- alpha = 1/20.

Under either null, \(P(X_1\ne X_2)=8/25\), giving terminal size:

\[
\frac{8}{25}\cdot\frac{5}{32}=\frac1{20}.
\]

### Member-specific conditional levels

| Public first observation | B1 conditional level | B2 conditional level | Robust lower envelope |
|---|---:|---:|---:|
| X1 = 0 | 1/32 | 1/8 | **1/32** |
| X1 = 1 | 1/8 | 1/32 | **1/32** |

At n=0, the robust level is 1/20. At n=2, it is exactly the original terminal phi.

### Every deterministic stopping rule

| Stopping rule | Expected level under B1 | Expected level under B2 |
|---|---:|---:|
| Stop at n=0 | 1/20 | 1/20 |
| Stop at n=1 on both branches | 1/32 | 1/32 |
| Stop at n=1 only when X1=0 | 1/20 | 1/32 |
| Stop at n=1 only when X1=1 | 1/32 | 1/20 |
| Continue to n=2 | 1/20 | 1/20 |

All five deterministic stopping rules are exactly valid under both null members. Randomized stopping follows by convexity.

## Strict negative controls

- Reusing only B1's conditional process under true B2 gives \(17/160\), exceeding alpha=1/20.
- Taking the maximum of the two member-specific processes gives 1/8 under either null, also exceeding alpha.
- Under B1 after X1=1, the conditional expectation of the terminal test is 1/8 while the lower envelope is 1/32. Thus the lower envelope is not generally a B1-supermartingale.

These controls protect against two tempting but incorrect claims: one member's conditional process is not automatically robust to the other nulls, and the maximum of valid member processes is not valid by construction.

## Implementation

- `scripts/research/sdcb_robust_test_induction.rs` — exact rational fixture, stopping-policy evaluation and adversarial controls.
- `scripts/research/run_sdcb_robust_test_induction_tests.sh` — standalone Rust test/executable harness.
- Eight Rust tests cover uniform terminal size, member-specific conditional values, the lower envelope, terminal matching, every deterministic stopping rule, single-member misuse, max aggregation, and the non-supermartingale witness.

## Validation boundary

- Independent Python `Fraction` calculations confirm every conditional value, all five stopping policies, and both negative controls.
- Rust tests are authored but **not compiled or executed** because `rustc` is unavailable in the current authoring environment.
- This is a finite two-step Bernoulli fixture. Continuous null families need measurable infima/conditional versions and separate proof; adaptive experiment policies must be common and non-anticipating across null members.
- This is not an e-process, a production stopping policy, a physical experiment, or an SOTA claim.

Literature comparator: *Anytime validity is free: inducing sequential tests* (JRSS B, 2026), https://doi.org/10.1093/jrsssb/qkag050.
