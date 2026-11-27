# MON-040 — Replay Reproducibility Qualification Contract v1

Profile: mon040-replay-reproducibility-input-v1. Status: offline repeated-replay evaluator with an exact one-sided Clopper–Pearson bound; live replay adapter pending.

## Research question

Does the exact recovery failure signature persist under a preregistered number of repeated replay trials for the same fault set, scenario, recovery boundary and source revision?

## Input identity

The trial binds:
- explicit non-empty fault ID list and canonical fault-set digest;
- scenario and recovery-boundary SHA-256;
- exact lowercase code revision;
- target blocked verdict and failure-signature SHA-256;
- preregistered run count, target reliability and one-sided confidence level;
- individual replay observations, each with unique replay ID and seed, outcome, timestamp, scope digests and a full-record outcome receipt.

Each replay receipt is recomputed over the observation with its own receipt field blank. An altered field with an old receipt digest blocks. Missing or null identity is invalid.

## Statistical decision

For (k) exact target-signature matches in (n) preregistered observations, MON-040 computes the one-sided Clopper–Pearson lower confidence bound (L) satisfying:

[
P_{p=L}(X ge k mid X sim mathrm{Binomial}(n,p)) = alpha
]

where (alpha=1-	ext{confidence level}). The result qualifies reproducibility only if all (n) observations are present and complete and (L) is at least the minimum target reliability.

For 29/29 matches at 95% one-sided confidence, the exact lower bound is approximately 0.901855. For 28/29 it is approximately 0.846608. Therefore the 29/29 case clears a 0.90 target, while 28/29 does not.

No missing replay is removed from the denominator. An incomplete observation or an indeterminate replay outcome holds the decision. A different blocked signature is not counted as the target failure.

## Independence assumption and common causes

The binomial interval assumes the replay outcomes are independent and identically distributed within the preregistered trial population. Different random seeds do not prove independent execution. Before reporting the bound, the benchmark operator must describe shared failure domains, including executor, hypervisor, host, clock, network, storage, credentials/identity, orchestration and broker/database control plane.

Where these domains are shared or correlated, the nominal interval is not a valid universal risk bound. The report must either justify the independence model, cluster the analysis by independent execution unit, or describe the bound as conditional/exploratory.

## Verdicts and limits

- `qualified`: complete preregistered sample and one-sided lower confidence bound meets the reliability target.
- `blocked`: malformed input, scope mismatch, duplicate identities/seeds, stale receipts, or a complete sample whose lower bound is below target.
- `indeterminate`: missing/incomplete observations or an indeterminate replay outcome.

Even a reproducibility-qualified offline result does not prove production reliability or root cause. Both offline and runtime-candidate modes remain campaign-indeterminate until a trusted replay scheduler, independent verifier and MON-026 custody adapter authenticate execution and receipts. Production use is always false.
