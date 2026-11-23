# SEMI-PRIV-001 — VM update-inversion red-team fixture

## Implemented attack surfaces

The Rust prototype currently covers two sharply defined synthetic logistic-regression leakage channels.

### Per-record update inversion

For one record, binary logistic regression produces:

- grad_w = (sigmoid(w·x + b) - y) x
- grad_b = sigmoid(w·x + b) - y

When grad_b is not numerically degenerate, the attacker obtains x = grad_w / grad_b and infers y from its sign. Tests demonstrate exact reconstruction under direct per-record update visibility. Per-example norm clipping multiplies the entire gradient vector by one scalar and therefore preserves the feature/bias ratio in this particular attack. This is a concrete example of why clipping is not a universal privacy guarantee.

### Known-auxiliary inversion of aggregates

The second test assumes the attacker knows exact features and labels for all but one contributing record and observes the aggregate together with cohort size. It subtracts the known per-record gradients from the visible sum and attempts inversion only when exactly one unknown remains.

The current fixtures demonstrate exact recovery from both a per-client batch aggregate and a secure/global aggregate when three of four synthetic records are known. When two unknown records remain in a global aggregate, the current one-record inversion routine explicitly reports that the attack is not applicable; it does not claim a reconstruction.

This is a strong auxiliary-information threat model. It does not establish that every secure-aggregation deployment leaks under ordinary attacker capabilities. It demonstrates that secure aggregation alone is not a universal privacy proof when external auxiliary knowledge can nearly determine the contributing population.

## Visibility semantics

The per-record mode attempts the single-record inversion. Per-client and secure aggregate modes are tested through the known-auxiliary attack. An attack not applicable to a visibility mode is not reported as “private”; it is reported as untested by that specific attack.

## Defence semantics

- Norm clipping is tested as a transformation; it is not treated as a privacy proof.
- Gaussian perturbation is a deterministic synthetic experiment with fixed seed. It reports reconstruction error; it does not output epsilon/delta.
- formal_differential_privacy_claim remains false in every report.

A formal DP comparison still requires an adjacency definition (wafer/record, lot or fab/client), per-example contribution bound, sampling mechanism, noise calibration, accountant, release cadence, multi-round composition and explicit trust assumptions.

## Next attack families

The next iteration should add:
- multi-record deconvolution with partial rather than all-but-one auxiliary data;
- colluding clients and malicious coordinator selection of initialization/update cadence;
- membership inference and property inference;
- temporal linkage and cross-round differencing;
- model extraction from checkpoints/query access;
- source attribution across clients/fabs;
- privacy/utility Pareto curves on matched cross-fab and few-shot VM tasks.

## Claim ceiling

Synthetic recovery demonstrates a leakage channel under the declared model and knowledge. A failed attack does not prove privacy; an aggregate-only mode that this attack cannot invert is not thereby certified private. No industrial data or private fab records are used.