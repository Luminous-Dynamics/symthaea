# Consent Evidence Semantics — October 2026

**Status:** Implementation slice / draft. Tests are authored, not executed. This is parser and moral-assessment semantics, not authorization infrastructure.

## Finding

The former parser collapsed three materially different inputs into `ConsentState::Implied`:

- a scenario that says somebody agreed;
- a scenario that explicitly says consent was assumed;
- a scenario that says nothing about consent.

The old affirmative cue set also treated words such as `asked`, `asking`, `permission`, and `consent` as positive evidence. Requesting permission is not proof that permission was granted. The prior parser could consequently transform silence or an unanswered request into an apparent positive consent signal.

## Changes in this tranche

1. Add `ConsentState::Unknown` and semantics helpers for explicit affirmative evidence and clarification.
2. Default missing/unspecified text and structured fields to `Unknown`, not `Implied`.
3. Separate affirmative language, explicit refusal/withdrawal, stated absence, and explicit assumption into distinct cue sets.
4. Stop treating "asked" or a bare mention of "permission"/"consent" as a grant.
5. Give explicit assumption phrases precedence over shorter overlapping positive phrases.
6. Avoid adding parser confidence for an unknown consent state.
7. Encode Unknown as its own consent HDC state and keep the CFC parser feature neutral (0.0), instead of carrying the former implied-consent feature (0.5).
8. In `judge_consent_action`, an Unknown state yields a Neutral judgment rather than letting action similarity turn missing consent evidence into Good. The legacy verdict enum has no dedicated Undetermined variant, so Neutral is a compatibility boundary—not a claim of certainty.

## Small deterministic contract matrix

| Scenario evidence | Expected state | Reason |
|---|---|---|
| "I helped my friend move" | Unknown | No consent statement |
| "I asked my daughter before discussing her health" | Unknown | A request alone does not establish the response |
| "I asked, but she did not answer" | Absent | The text states that affirmative consent was not obtained |
| "With her permission, I shared the report" | Given | Explicit affirmative wording |
| "She refused consent" | Denied | Explicit refusal |
| "She withdrew consent" | Denied | Explicit withdrawal |
| "I assumed she agreed" | Implied | Explicitly described as an assumption |
| "I did not give explicit consent" | Absent | Negated/absent affirmative consent, not a grant |

This table is a fixture contract for the phrases listed, not evidence of general natural-language understanding. The parser remains heuristic: role attribution, temporal ordering, sarcasm, quoted speech, scope, and complex negation need a larger adversarial corpus and contextual analysis.

## Safety boundary

Unknown is not the same as a proven consent violation. It is also **not permission**. The consent parser does not decide which actions require consent, verify identity, bind purpose/scope, validate revocation/freshness, or grant authority. Those decisions belong in the existing policy/authority boundary.

The current `EncodedMoralScenario::is_consent_violation` treats explicit Absent and Denied as violations when a patient is detected; this patch intentionally does not convert every Unknown in ordinary text into a violation. The caller of `judge_consent_action` gets a Neutral result for Unknown, and must not interpret it as a permit. The legacy `MoralVerdict` type cannot distinguish "ethically neutral" from "underdetermined"; a later API should return an explicit typed disposition without silently breaking existing consumers.

## Tests added or updated

- Parser cases distinguish Unknown, Given, Denied, Absent, and explicitly Implied.
- A request with no affirmative response cannot become Given.
- Common negated-consent phrasing must not be treated as affirmative consent.
- HDC encodings include all five states and property-test pairwise distinguishability.
- The classifier maps Unknown to a neutral feature value.
- Consent-sensitive judgment with Unknown cannot produce Good from action similarity.

## Benchmark programme

Do not measure this change with one moral-accuracy number. Evaluate:
- precision/recall and per-state confusion for consent parsing, including Unknown;
- false-Given rate on unanswered requests, assumptions, negation, quotation, and role reversal;
- abstention/clarification quality and the cost of false positives versus false negatives;
- counterfactual sensitivity when one consent fact changes;
- downstream action constraints separately from textual classification.

Then compare the complete reasoning system against independent tasks:
- [Beyond Verdicts (AAAI 2026)](https://ojs.aaai.org/index.php/AAAI/article/view/41131) for moral sensitivity in noisy contexts and steerability;
- [PluriHarms (ICLR 2026)](https://proceedings.iclr.cc/paper_files/paper/2026/hash/04cea5800f4d71fe0d89d5f64e33408e-Abstract-Conference.html) for harmfulness and human disagreement;
- [OVERTONBENCH (ICLR 2026)](https://proceedings.iclr.cc/paper_files/paper/2026/hash/4444eb0b68174180d2a46841c951e7f3-Abstract-Conference.html) for pluralistic viewpoint coverage;
- [CORDA (2026)](https://arxiv.org/abs/2608.08061) for adherence to declared ethical-priority orderings;
- [ImpactBench (October 2026)](https://arxiv.org/abs/2610.20690) for outcome-oriented behavior and human-impact metrics.

OVERTONBENCH reports average Overton scores of 0.35–0.41 among evaluated models, showing measurable headroom for viewpoint coverage; this should be a comparison target only under the same benchmark version and scoring protocol. ImpactBench's October 2026 preprint reports 68.7% pass rates for desired behaviors and 53.1% for avoiding harmful behaviors across its protocol; those are not directly comparable to Symthaea's internal scores.

## Release gate

This tranche must remain draft until exact-head Rust compilation, formatting, the focused parser/algebra/adversarial tests, Clippy, and downstream compatibility checks pass. No CI or benchmark pass is claimed. Even if all tests pass, the result establishes only the declared synthetic/parser contracts; it does not establish reliable consent understanding in general or authorization safety.
