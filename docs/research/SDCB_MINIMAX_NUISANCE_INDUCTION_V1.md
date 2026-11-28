# SDCB maximin + nuisance-trajectory + induced-test integration M0

**Status:** exact rational finite-tree known answer; Rust implementation authored but not compiled/CI-qualified.

Machine-readable artifact: `docs/research/SDCB_MINIMAX_NUISANCE_INDUCTION_V1.json`.

## What this integrates

This fixture composes three existing authorities:
- #14185/#14218: terminal finite-family maximin test and primal/dual certificate;
- #14230: complete persistent/rectangular/history-adaptive nuisance trajectory laws;
- #14222/#14223: composite-null lower-envelope induction (published prior art, not a new theorem).

It does not introduce a new test engine, sequential theorem, scheduler or evidence ledger. #13567 remains the policy/terminal-regret oracle.

## Eight complete nuisance trajectory laws

X1,X2 are binary. Select p1 from {1/5,4/5}. The trajectory member also freezes the public-history transition map p2(0),p2(1), each in {1/5,4/5}. Conditional trajectory probabilities are:

\[
\begin{aligned}
P(00)&=(1-p_1)(1-p_2(0)),\\
P(01)&=(1-p_1)p_2(0),\\
P(10)&=p_1(1-p_2(1)),\\
P(11)&=p_1p_2(1).
\end{aligned}
\]

These eight complete path laws include the persistent shared and precommitted rectangular subfamilies. A history-dependent p2 map is selected as a fixed member before evaluation; it does not observe hidden evaluator truth.

## Exact finite-family maximin terminal test

Alternative:
\[
A=(0,1/2,1/2,0).
\]

Null family: all eight trajectory laws. The largest null discordance probability is 4/5, so with alpha=1/10 the exact randomized test is:

\[
\phi_2=(0,1/8,1/8,0).
\]

It has maximum null size 1/10 and alternative power 1/8.

### Matching exact dual certificate

Trajectory members 2 and 6 have joint laws:
\[
Q_2=(4/25,16/25,4/25,1/25),\quad
Q_6=(1/25,4/25,16/25,4/25).
\]

Their equal mixture is:
\[
Q_0=(1/10,2/5,2/5,1/10).
\]

Use alternative weight lambda=1 and unnormalized null weights w2=w6=5/8, total w=5/4. Then pointwise:

\[
A(y)\leq\frac54 Q_0(y),\quad y\in\{00,01,10,11\}.
\]

The positive-part slack is zero. The dual objective is:
\[
\alpha w=(1/10)(5/4)=1/8.
\]

This matches the primal power 1/8 exactly. Thus the terminal test is a finite-family maximin solution, not merely a size-valid positive control.

## Member-specific induced processes

For every null trajectory j:
\[
g_n^j=E_j[\phi_2\mid\mathcal F_n],\qquad
g_n=\min_j g_n^j.
\]

At n=0, member terminal sizes range from 1/40 to 1/10, hence:
\[
g_0=1/40.
\]

At n=1:
- for X1=0, member-specific conditional tests range from 1/40 to 1/10;
- for X1=1, they also range from 1/40 to 1/10.

The pointwise lower envelope is:
\[
g_1(0)=g_1(1)=1/40.
\]

At n=2, \(g_2=\phi_2\) exactly on all four histories.

For any true member j and any public bounded stopping time tau, pointwise domination by its own conditional-expectation martingale gives:
\[
E_j[g_\tau]\leq E_j[g_\tau^j]=E_j[\phi_2]\leq1/10.
\]

The lower envelope is **not** asserted to be a memberwise supermartingale.

## Exact stopping-time results

| Public stopping rule | Worst null expected level | Alternative induced power |
|---|---:|---:|
| Stop at n=0 | 1/40 | 1/40 |
| Continue through n=1 to n=2 | 1/10 | 1/8 |
| Stop at n=1 only if X1=0 | 17/200 | 3/40 |
| Stop at n=1 only if X1=1 | 17/200 | 3/40 |
| Stop at n=1 on both histories | 1/40 | 1/40 |

This exposes the validity/utility distinction: earlier stopping remains null-valid but reduces alternative power. The experiment policy must trade stopping cost against terminal decision loss, using #13567 rather than a duplicate planner.

## Implementation

- `scripts/research/sdcb_minimax_nuisance_induction.rs` — exact rational trajectory, primal/dual certificate and conditional-process verifier.
- `scripts/research/run_sdcb_minimax_nuisance_induction_tests.sh` — standalone Rust test/executable harness.
- 11 authored tests cover normalized trajectory laws, nested subfamilies, null size, alternative power, dual certificate, member-specific conditional values, lower envelope, terminal match, every deterministic stopping rule, non-supermartingale caution and early-stopping power.

## Validation boundary

Independent Python `Fraction` arithmetic checked all eight trajectory laws, the exact mixture dual, all conditional values, all stopping-rule levels and the alternative power loss under early stopping. Rust tests and the standalone harness are authored but **not compiled or executed** because rustc is unavailable here.

This is a finite two-step integration benchmark for published composite-null theory. It is not a new anytime-valid theorem, continuous nuisance-family proof, physical sensor result, production adaptive policy or SOTA claim.
