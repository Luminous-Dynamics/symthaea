# HARDSEC-009 — Empirical Trigger and Payload Coverage Campaigns

**Status:** Rust audit utility and preregistered metric contract  
**Purpose:** turn recorded stimulus/replay campaigns into reproducible, bounded coverage curves  
**No claim:** this utility does not prove trigger unreachability, estimate prevalence across arbitrary inputs, or certify a design.

## 1. Why another metric layer is needed

The HARDSEC-007 trigger-budget calculator answers a hypothetical planning question: if a per-window activation probability (p) were known, how many independent trials (N) would be required to make a miss probability small?

A real campaign poses the inverse question: given the actual stimulus sequence log and observed outcomes, what did the campaign exercise, and what uncertainty is supported by its sampling design?

SeqCube-HT provides a concrete methodological reference. Its 9 October 2026 paper evaluates random detectability using 100,000 independent normal-mode random windows of length 12 cycles. It explicitly describes this as a fixed-budget, fixed-window screening metric, not an intrinsic trigger probability: triggers whose temporal span exceeds 12 cycles cannot complete, and observed coverage jointly reflects effective trigger literals and temporal-span distribution. The paper also separates trigger activation from payload observability. (Xu et al., DOI 10.1186/s42400-026-00676-2.)

Published SeqCube-HT results are useful comparison targets:
- 3,240 witness-validated instances from 4,500 attempts (72.0% yield);
- 20.9 seconds per witness-validated instance versus 39.5 seconds for its best matched comparator;
- graph-detector F1 69.2% under instance-random testing versus 47.8% when the insertion source is held out;
- localization recall 55.7% versus 31.4% under those same two boundaries;
- after covariate matching, the paper reports 77.2% F1 on its fixed four-circuit corpus;
- ISCAS-to-Extended-IP cross-scale F1 is 56.8% without threshold retuning.

These are paper-reported values on the authors' internal corpus, not reproduced Symthaea results. The article says the complete generation corpus and run records are available from the authors on reasonable request. Its public witness-validation artifact is one reference fixture, not a member of the 3,240-instance corpus. Do not equate fixture replay with reproduction of the generation or detector experiments.

## 2. Paradigm shift: publish coverage curves, not one scalar

For each fixed design, reset model, observation boundary, stimulus mode and window length, publish cumulative curves at predeclared trial budgets—for example 10, 100, 1,000, 10,000 and 100,000 windows.

Each budget point reports these outcomes separately:

1. **Trigger activation:** did the complete declared trigger path activate from the stated reset/initial-state model?
2. **Payload observation:** did the effect reach the declared primary output, state boundary or other observation point?
3. **Payload effect conditional on activation:** among trials with a trigger activation, how often was an effect observed?
4. **Cycle evidence:** first trigger cycle, first payload-observation cycle and trigger-to-effect delay where recorded.
5. **Inactive controls:** did matched inactive sequences suppress trigger activation and payload effect?

A trigger firing is not proof that its payload reached an observation boundary. A finite trace in which no trigger fired is not proof that the trigger is unreachable.

## 3. Statistical contract

### 3.1 Empirical rates

Always report observed counts and denominators. Unknown outcomes are counted separately and excluded from the corresponding known-outcome denominator. Do not silently reclassify unknown as not observed.

### 3.2 When confidence intervals are allowed

The auditor emits a Wilson 95% two-sided interval only when:
- the campaign declares `sampling_model = "iid"`;
- the stimulus mode is `random_simulation` or `constrained_random`;
- the campaign provides a non-empty independence rationale;
- every trial contributing to the metric has a known outcome.

The independence rationale is caller-declared. The auditor does not prove statistical independence. Directed, recorded, clustered, deterministic, ATPG and formal-cover campaigns receive empirical counts/rates only—not IID intervals.

### 3.3 Zero observed activations

For zero activations in (N) complete IID Bernoulli trials, the one-sided exact upper confidence bound at confidence (1-alpha) is:

[
p_{mathrm{upper}} = 1 - alpha^{1/N}.
]

At 95% confidence, (alpha=0.05). For example, zero activations in 1,000 IID windows gives an upper bound of approximately 0.002991 (0.2991%) per window under that specific sampling model.

This bound is conditional on the named design, reset model, stimulus distribution, window length, observation conditions, simulator and IID assumption. It is not a probability for all possible input sequences, all states, another window length, or real silicon. If the campaign contains unknown outcomes, this upper bound is omitted.

### 3.4 Payload rates

Report:
- payload effects divided by all trials;
- payload effects divided by trials where trigger activation was observed.

These answer different questions. Do not impute payload effects from trigger activation. If a trigger activates but no output/state effect is observed, that remains a payload-observation failure for that trial.

### 3.5 Formal cover and deterministic campaigns

A formal `cover_reached` is an existential witness under the encoded model, assumptions and bound. A bounded `bounded_no_counterexample` is only a bounded no-witness result. Neither supplies a probability estimate.

The auditor enforces:
- `cover_reached` requires an observed trigger and a trigger witness digest;
- `bounded_no_counterexample` requires trigger status `not_observed`;
- `timeout` / `unknown` requires trigger status `unknown`;
- formal results are not accepted on simulation campaigns.

## 4. Input schema

The current JSON contract is `hardsec-activation-campaign-v1`. It binds:

- campaign ID, dataset ID and dataset role;
- design ID, exact netlist SHA-256;
- reset-model SHA-256 and declared reset semantics;
- stimulus mode, sampling model, caller independence rationale, tool/version, configuration digest and random-distribution digest;
- window length and strictly increasing predeclared budget points;
- observation-boundary identity/type/description;
- ordered trial records containing unique trial ID, per-condition trial index, active/inactive-control condition, explicit `pair_id` when matched controls are supplied, sequence digest, trigger outcome/cycle/witness, payload outcome/cycle/witness, outcome-trace digest, and formal result/bound only for formal-cover campaigns.

Trials within each condition must have contiguous ordered indices. If an inactive-control set is included, it must have the same number of trials and every `pair_id` must map to exactly one active and one inactive-control trial. Pairing is never inferred from row order alone. Sequence duplicates are reported, not silently deduplicated; the receipt preserves the total trial count and unique-sequence count separately.

The campaign file is read from one open file handle, with before/after file metadata checks and a no-symlink path check, then hashed exactly. Trial sequence hashes remain caller-supplied references; this utility does not dereference or re-hash raw stimulus files. The output receipt is unsigned, created with no-overwrite semantics, and must be outside the campaign directory.

## 5. Rust implementation

The opt-in example is:
- `examples/hardsec_activation_campaign_audit.rs`
- Cargo feature: `hardsec-activation-audit`
- example: `hardsec_activation_campaign_audit`

Run with:

```bash
cargo test --locked --manifest-path tools/hardsec-audit-harness/Cargo.toml \
  --features hardsec-activation-audit --example hardsec_activation_campaign_audit

cargo run --locked --manifest-path tools/hardsec-audit-harness/Cargo.toml \
  --features hardsec-activation-audit --example hardsec_activation_campaign_audit -- \
  --campaign /path/to/campaign.json \
  --receipt /path/outside/campaign/activation-coverage.json
```

The receipt includes campaign digest, subject/reset/configuration provenance, tool and stimulus metadata, all normalized trial records, per-budget curves for active and inactive controls, Wilson intervals only where justified, zero-event bounds only for complete IID data, cycle histograms, paired-control discordance, and explicit claim ceilings.

## 6. Negative controls and current tests

The unit tests cover:
- separate trigger and payload rates;
- Wilson intervals and exact zero-event upper bounds;
- non-IID interval suppression;
- IID claims rejected for directed sequences or missing rationale;
- payload effect without trigger, impossible cycle ordering and out-of-window cycles;
- non-contiguous/duplicate trial identity;
- missing distribution hashes and invalid budget points;
- unpaired active/inactive counts;
- formal-cover typed outcomes and witness requirements;
- unknown outcomes suppressing confidence intervals;
- duplicate sequence reporting without trial deduplication;
- digest validation and formal metadata misuse;
- active/inactive descriptive comparison without causal wording.

The suite currently contains 31 authored tests, including explicit matched-pair identity and discordance tests. Compilation and CI execution remain pending for this newly created branch; no test is described as passed until the focused workflow confirms it.

## 7. Claim ceiling

A pass means only that the supplied campaign JSON is internally consistent and that the receipt's calculations follow the declared records and sampling model. It does not validate the simulator, the actual input trace bytes, the truth of the trigger labels, reset semantics, or the independence of random draws. It does not prove absence/unreachability, detect a Trojan, estimate design prevalence, or certify physical-silicon behavior.