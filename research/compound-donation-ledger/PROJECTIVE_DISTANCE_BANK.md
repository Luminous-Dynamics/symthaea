# Projective multi-recipient overshoot routing bank

## Executive idea

The fixed-destination ledger has one recipient per donor. That is safe and easy to audit, but it creates a concentration bottleneck: a donor's capped reserve can only help one later hypothesis.

A projective distance prior distributes that reserve across future candidate positions using globally fixed weights. The key is to distinguish two constructions:

- **Full-reserve routing:** split the existing capped-overshoot reserve among recipients. Aggregate donor outflow is unchanged.
- **Threshold-sliced bank:** give each recipient slice its own overshoot threshold. This is more conservative and may delay reserve utilization, but it has a tighter expected-null-debit upper bound.

The two constructions have different objectives and must be evaluated separately.

## 1. Full-reserve projective router

For donor i, retain the existing cumulative capped-overshoot reserve

\[
D_{i,t}=\min\{(\gamma_iX_{i,t}-1)_+,\gamma_i\}.
\]

Let \((\rho_d)_{d\ge1}\) be a fixed probability distribution over positive integer offsets. Set

\[
\rho_d=\frac{1}{d(d+1)},\qquad
\sum_{d=1}^{\infty}\rho_d=1.
\]

At each update, split the newly earned reserve increment as

\[
\Delta O_{i,d,t}=\rho_d\Delta D_{i,t}
\]

and bind that slice to destination \(i+d\). Cumulative outflow to distance d is

\[
O_{i,d,t}=\rho_dD_{i,t}.
\]

The sum is exactly the existing reserve:

\[
\sum_{d\ge1} O_{i,d,t}=D_{i,t}.
\]

Therefore the donor's aggregate pathwise budget is unchanged:

\[
\sum_d O_{i,d,t}
\le\gamma_i\min(X_{i,t},1),
\]

and

\[
\sum_d\Delta O_{i,d,t}
=\Delta D_{i,t}
\le\gamma_i\Delta X_{i,t}.
\]

No null-debit improvement is claimed for this router: it withdraws exactly the same aggregate reserve as the single-recipient policy. Its potential advantage is broader recipient coverage and reduced concentration, not increased total evidence or automatically higher power.

Every destination is later than its source. Thus the same prefix-budget argument applies. A finite horizon only affects whether a share is internal or beyond the current cut; it does not change the source's reserve or the destination assigned to an offset.

### Exact finite-horizon tail

For any integer K,

\[
\sum_{d=1}^{K}\frac{1}{d(d+1)}
=\sum_{d=1}^{K}\left(\frac1d-\frac1{d+1}\right)
=\frac{K}{K+1}.
\]

The unmaterialized tail has exact mass \(1/(K+1)\). A donor whose first K future offsets are represented can therefore hold

\[
\text{tail escrow}_{i,K,t}=\frac{D_{i,t}}{K+1}.
\]

This is compressed escrow for destinations beyond the K visible offsets, not destroyed capital. When the horizon grows, new destination shares can be materialized from the tail without changing the old-cut total. If candidate IDs are missing or dormant, their shares remain in per-destination escrow.

The source can thus maintain a countable routing policy with finite state per finite horizon. The policy digest fixes the prior and offset semantics. The independent verifier must recompute the exact visible shares and tail balance.

## 2. Threshold-sliced reserve bank

A second construction assigns a separate capped-overshoot expert to each offset. With

\[
\lambda_{i,d}=\gamma_i\rho_d,
\]

define

\[
D_{i,d,t}=\min\{(\lambda_{i,d}X_{i,t}-1)_+,\lambda_{i,d}\}.
\]

For every \(\lambda\in[0,1]\),

\[
0\le D_\lambda(x)\le\lambda\min(x,1),
\]

and \(D_\lambda\) is nondecreasing and \(\lambda\)-Lipschitz. Therefore

\[
\sum_dD_{i,d,t}
\le\gamma_i\min(X_{i,t},1),
\qquad
\Delta\sum_dD_{i,d,t}
\le\gamma_i\Delta X_{i,t}.
\]

This bank generally withdraws **less** than the full-reserve router at finite evidence values: each slice must independently cross its own overshoot threshold. It is a conservative alternative, not a free diversification gain.

Under a valid nonnegative e-process, Ville's inequality gives
\(\Pr(\sup_tX_{i,t}\ge u)\le1/u\). Tail integration yields

\[
\mathbb E[D_\lambda(\sup_tX)]
\le\lambda\log(1+\lambda)
\le\lambda^2.
\]

For the telescoping distance prior,

\[
\sum_{d=1}^{\infty}\rho_d^2
=\sum_{d=1}^{\infty}\frac{1}{d^2(d+1)^2}
=\frac{\pi^2}{3}-3
\approx0.289868.
\]

Hence the threshold-sliced construction satisfies

\[
\mathbb E\!\left[\sum_dD_{i,d,\infty}\right]
\le\gamma_i^2\left(\frac{\pi^2}{3}-3\right).
\]

This is a stronger expected-null-withdrawal upper bound than the coarse \(\gamma_i^2\) bound, but it applies only to the threshold-sliced construction. It does not apply to the full-reserve router in Section 1.

## 3. Projectivity and cut accounting

Use a fixed global candidate order and immutable offset destinations \(i+d\). Do not derive offsets from the current active-family size.

Use source-scoped semantic identities such as

\[
(\text{source id},\text{distance},\text{source-local ordinal}),
\]

not the writer's existing single global sequential token counter. A high-ID donor must not shift a low-ID donor's semantic token identity.

For old prefix \([m]\), a share with destination \(i+d\le m\) is internal. A share with destination \(i+d>m\) lies beyond the cut. When extending the horizon, some such shares may be materialized or credited to newly visible candidates. The correct comparison is the cut projection from ROUTING_PROJECTIVITY_REFINEMENT.md:

- compare full token state for internal edges;
- compare immutable identity, source, destination, creation round and face amount for crossing edges;
- collapse all escrow and credits beyond the cut into one outside balance.

Literal remaining-balance equality for crossing tokens is not required and is generally false.

The pathwise null-mass accounting bound is unchanged because the total outgoing amount is unchanged and all edges point forward. This does not authenticate the global filtration, source completeness, or real arrival timestamps. It also does not, by itself, prove a new general SupFDR theorem.

## 4. Exact arithmetic and representation

For the full-reserve router, all visible recipient shares plus the compressed tail must equal the parent reserve exactly. With fixed-point integer balances, use one of:

1. rational remainder carry per source/prior component; or
2. exact wide multiply/divide with the division remainder retained in the source/tail escrow.

Never round each child independently and silently discard the residual. Never allow rounding to borrow against a future reserve increment.

For a fixed K implementation, the normalized K-offset prior can be precommitted globally, but K must not depend on the active horizon. If the unnormalized first K telescoping weights are used, retain the exact tail \(1/(K+1)\).

For the countable bank, only materialize offsets within the current finite candidate prefix and keep the remainder in compressed tail escrow. Do not create infinitely many physical token rows.

## 5. Exact-rational screens

The committed Python harness checks both constructions with fractions.Fraction:

- threshold-sliced donor cap and own-increment Lipschitz bounds;
- telescoping partial-sum identity and unallocated tail;
- full-reserve split equality;
- per-share donor and increment bounds;
- cut-projected outside-balance equality across horizon extensions;
- prefix budgets under randomly dormant/active recipients;
- exact escrow conservation;
- geometric-prior comparison as an alternative.

The recorded default run uses 100 deterministic traces, a 16-candidate global universe, 20 rounds, and 15 maximum offsets. It records 810,043 threshold-slice/property assertions, 92,000 exact split/cut assertions, and 15,200 prefix/conservation assertions. The maximum exact cut-projection error is zero.

A deterministic comparison at \(\gamma=0.2\) shows the full-reserve single-recipient policy reserves 0.2 at \(X=10\), while the threshold-sliced bank reserves zero at that evidence level; at \(X=1000\), the telescope threshold bank reserves about 0.1857 across 13 destinations. This illustrates its delayed utilization tradeoff, not improved power. The full-reserve router, by contrast, splits the same 0.2 total across destinations.

These are exact bookkeeping screens, not e-process validity tests, FDR/SupFDR simulations, Rust builds, or statistical power comparisons.

## 6. Benchmark plan

Compare:

1. single next-recipient capped overshoot;
2. full-reserve telescoping-distance routing;
3. full-reserve geometric routing with the same total prior mass;
4. fixed-K routing, with K fixed independently of the active horizon;
5. threshold-sliced telescoping bank;
6. fixed-rate donation and no-donation baselines.

Use paired evidence streams, precommitted global candidate weights, and common random numbers. Sweep sparse, dense, heterogeneous, late and switching alternatives. Separate independent, shared-control, block-positive, balanced-negative, time-varying and adversarial conditionally valid null dependence.

Report FDR, SupFDR, stopped-FDR, FDX and P(any false); power and time-to-discovery; total reserve volume/utilization; recipient coverage and concentration; stranded and virtual escrow; worst-regime regret; p50/p95/p99 event latency; memory; and independent-verifier cost.

The 2026 online compound-e-value / donation e-LOND paper reports O(log t) decision-time algorithms. That is an algorithmic reference, not an apples-to-apples benchmark for a provenance-preserving transfer ledger. Measure decision latency separately from escrow accounting and receipt verification. The 2026 dynamic e-closure work makes horizon projectivity and globally summable weights explicit requirements; the distance prior is designed to satisfy a fixed, projective prior contract. UAI 2026 paper: https://arxiv.org/abs/2603.24792. Dynamic e-closure preprint: https://arxiv.org/abs/2608.09927.

Confirmatory error-rate cells should use at least 50,000 frozen paired streams, with more for low-tail targets, and simultaneous uncertainty intervals across the prespecified grid. Do not call a small Monte Carlo power crossing SOTA.

## Claim boundary

The full-reserve router preserves the current pathwise donor budget while changing recipient allocation. The threshold-sliced bank preserves the same cap but trades reserve utilization for a smaller expected-null-withdrawal upper bound. Neither is yet a Rust implementation, an authenticated adaptive router, a new general SupFDR theorem, a repeated-SCORE spending algorithm, or evidence of power superiority.
