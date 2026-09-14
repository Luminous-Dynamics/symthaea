# LQCD-021F — independent covariance-rank admission oracle

Independent standard-library execution for the covariance-rank / fail-closed primary-fit authority added to #2715.

Exact executed subject SHA-256:

`1e1bcd844c5c509ddb5d6ec45e4116109c27413fd0deab819c2b814574501c37`

Canonical result SHA-256:

`35ee9fd0192771120773b30036d3566d12de57484ace07200d29c77bf380f8e8`

## Rank theorem

For `B=8` leave-one-block-out jackknife replicates, the centered empirical covariance rank is at most:

`B - 1 = 7`.

That rank budget is an admission condition, not an implementation detail discovered after an inversion routine fails.

The executed fixtures establish:

- six active points with rank 6 -> `Admissible`;
- seven active points with rank 7 -> `Admissible`;
- eight requested points -> `NotAdmissible(RankBudgetExceeded)` before inversion;
- six requested points with one duplicated covariance mode -> `NotAdmissible(RankDeficient)` with observed rank 5.

## No post-hoc rescue

A diagnostic ridge lane may exist only as `DiagnosticOnly`. It cannot convert a failed primary covariance gate into an authorized primary result.

The oracle freezes these final-data rescue attempts as forbidden primary mutations:

- dropping a point;
- switching to diagonal covariance;
- changing block size;
- truncating a singular mode;
- adding a primary ridge;
- switching fit family.

A scientifically justified change requires a new frozen analysis/campaign subject.

## Scientific boundary

This subject qualifies covariance-rank and primary-admission semantics only. It does not establish that a real beta=6.0 covariance matrix is well-conditioned, does not qualify a production Rust fit implementation, and does not establish EHK agreement.
