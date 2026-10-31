# SCI-TARGET-DELTA-VERIFICATION-001 — Intended claim transition and collateral-scope audit

**Status:** Rust plan/receipt/report analyzer, strict schemas, deterministic synthetic fixture, hostile unit tests and CI controls under development. This is a second-stage audit composed with SCI-INTERVENTION-QUALITY-001; it does not replace the global semantic-relation panel.

**Tracking:** [Issue #9398](https://github.com/Luminous-Dynamics/symthaea/issues/9398).  
**Parent:** [PR #9274 — SCI-INTERVENTION-QUALITY-001](https://github.com/Luminous-Dynamics/symthaea/pull/9274).

## 1. Why the second semantic layer is necessary

The parent intervention-quality audit asks whether the candidate pair is globally meaning-preserving, meaning-changing or ambiguous. That is necessary but insufficient.

Suppose a test generator is asked to withdraw a decisive source supporting claim (C). It changes a date string elsewhere, and blinded reviewers correctly judge the pair as meaning-changing. The generator has not implemented the intended evidence-withdrawal intervention. The global semantic label is true, but the specific test is invalid for its declared purpose.

The opposite failure is also possible: the intended target claim changes correctly, but an unrelated material claim changes as collateral damage. A binary meaning-changing label hides that scope violation.

The paradigm shift is to evaluate two separate questions:

1. **Global relation:** did the candidate preserve or change meaning as declared?
2. **Target delta:** did the exact intended claim/scope transition occur, in the declared direction, without material collateral changes?

A mutation is a valid instance of its intended intervention only when both checks pass.

## 2. Two-layer adjudication

The target-delta plan is bound to the exact parent quality-plan bytes and commitment, as well as the analyzer source revision, study ID and purpose. It must contain exactly the same candidate census as the parent.

For every candidate it preregisters:
- target claim/entity/scope identity digest;
- baseline and variant target-state digests;
- expected target transition;
- exact sorted set of allowed changed claim/scope identity digests;
- exact sorted set of preserved control claim/scope identity digests.

The plan enforces consistency:
- an unchanged target must have identical target-state digests, no allowed changed IDs, and the target identity must be in the preserved set;
- a changed target must have distinct target-state digests, and the target identity must be in the allowed changed set and not the preserved set;
- changed and preserved identity sets must be disjoint;
- the expected target transition must agree with the parent's semantic-invariant / semantic-change declaration.

The transition vocabulary is deliberately typed: unchanged, support increased/decreased, contradiction introduced, scope changed, temporal validity changed, unit changed, identity changed, source withdrawn/superseded, and other declared change.

## 3. Blinded reviewer protocol

Reviewers receive baseline and variant artifacts with the target identity, but do not see:
- generator identity;
- intended global relation;
- pipeline accepted/rejected/abstained disposition;
- downstream evaluator outcome;
- expected target transition;
- allowed changed-claim list.

Each reviewer records:
- observed target transition;
- observed changed claim/scope identity digests;
- whether a material collateral change outside the permitted scope occurred;
- whether the collateral assessment is ambiguous.

The reviewer panel uses a fixed preregistered 3–7 member size and consensus threshold. The receipt must contain every parent candidate exactly once and one unique, blinded reviewer record per panel member. Ambiguity is retained; it is never promoted to success.

This is an audit of reviewer declarations, not proof that reviewers were independent, competent, or correct. Hashes bind declared artifacts; they do not prove raw artifact custody.

## 4. Final validity rule

For candidate (i), retain three independent statuses:

- (G_i): global semantic-relation validity from the parent audit;
- (T_i): target-delta and exact-scope validity;
- (F_i): final intended-intervention validity.

Combination rule:

[
F_i =
egin{cases}
	ext{invalid}, & G_i=	ext{invalid} lor T_i=	ext{invalid} \
	ext{ambiguous}, & G_i=	ext{ambiguous} lor T_i=	ext{ambiguous} \
	ext{valid}, & G_i=T_i=	ext{valid}
end{cases}
]

A known invalid result cannot be hidden by an ambiguous result on the other axis. Conversely, two valid subchecks are required for a final valid mutation.

A target-transition consensus requires the same complete tuple (direction, changed-scope set, collateral status) to reach the preregistered panel threshold. The report separately measures pairwise agreement on the full tuple and on the coarse transition label alone.

## 5. Metrics

The report includes:
- global-relation valid / invalid / ambiguous counts;
- target-delta valid / invalid / ambiguous counts;
- final valid / invalid / ambiguous counts;
- accepted / rejected / abstained dispositions cross-tabulated by final validity;
- exact changed-scope match count and material collateral-change count;
- reviewer pairwise full-tuple agreement and transition-only agreement;
- transformation-family strata, including unique semantic-cluster counts per family;
- generation tokens, micro-USD cost and elapsed time;
- unique final-valid accepted semantic-cluster yield per 1,000 generation tokens and per USD.

The unit of success is a **unique, accepted, globally valid, target-valid intervention**, not merely a generated candidate or a changed string.

All candidates in the parent full census remain in the denominator. The report does not silently drop rejected, abstained or ambiguous candidates.

## 6. Executable v1

Files:
- `schema/intervention-target-delta-plan.schema.json`
- `schema/intervention-target-delta-receipt.schema.json`
- `schema/intervention-target-delta-report.schema.json`
- `src/intervention_target_delta.rs`
- `tools/intervention_target_delta_fixture.py`

The analyzer first validates the parent plan and receipt using SCI-INTERVENTION-QUALITY-001, then validates the target-delta plan and receipt against those exact parent bytes.

CLI:
```sh
python3 research/science-bench-refresh/tools/intervention_target_delta_fixture.py /tmp/sbr-target-delta
cargo run --locked --quiet --manifest-path research/science-bench-refresh/Cargo.toml -- \
  analyze-intervention-target-delta \
  /tmp/sbr-target-delta/intervention-quality-plan.example.json \
  /tmp/sbr-target-delta/intervention-quality-receipt.example.json \
  /tmp/sbr-target-delta/intervention-target-delta-plan.example.json \
  /tmp/sbr-target-delta/intervention-target-delta-receipt.example.json
```

The synthetic fixture composes with the existing seven-candidate quality fixture. It includes invariant and meaning-changing interventions, accepted/rejected/abstained dispositions, a deliberately invalid global semantic label, and a deliberately ambiguous panel. The expected final counts are asserted in CI. This is protocol QA, not a human study or empirical mutation-generator result.

## 7. Hostile controls

Rust tests cover:
- exact parent quality-plan and receipt binding;
- wrong target direction;
- exact changed-scope mismatch;
- collateral material changes;
- ambiguous target panels;
- known target failure not masked by global ambiguity;
- reviewer blinding violations;
- missing and duplicate candidate adjudications;
- target-plan / parent-study binding;
- deterministic reports and permanent claim-boundary flags.

Additional qualification targets:
- 100% clean synthetic fixture acceptance;
- 100% rejection of malformed, duplicate, missing, unblinded and tampered artifacts;
- no candidate omitted from the full census;
- ambiguous target observations never counted as valid;
- no semantic truth, reviewer competence/independence, raw custody or execution-authenticity claim.

## 8. Claim boundary and next stage

The report can show a structurally valid receipt for a poor target-delta design. It does not establish semantic truth, human reviewer competence or independence, raw artifact custody, execution authenticity, causal effects, or downstream judge failure induction. All such flags remain false; external anchoring remains required and scoring eligibility remains false.

The next stage should test whether final-valid mutations actually reveal evaluator faults: blinded mutation-effectiveness trials, held-out evaluator versions, cost-matched heuristic versus explanation-guided generation, and verified unique failure discovery per unit budget.
