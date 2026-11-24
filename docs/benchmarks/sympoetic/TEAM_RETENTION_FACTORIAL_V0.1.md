# DELIB-TEAM-RETENTION-001: factorial evaluator contract v0.1

## Purpose and ownership

This evaluator is the first executable, deterministic summary path for Mycelix issue [#6597](https://github.com/Luminous-Dynamics/mycelix/issues/6597). It extends SympoeticBench rather than introducing a second benchmark framework.

The experiment asks whether AI-assisted collaboration leaves a human team more capable after AI access is removed, and whether a voluntary transactive-memory scaffold changes that outcome.

Mycelix owns the group-level experiment, conditions, task governance and interpretation. SympoeticBench owns this reusable input validator and estimator. Information elicitation and provenance mechanics remain with Mycelix #6089; persistent-memory storage and trajectory recording remain with Symthaea #12359.

## Four randomized conditions

| Arm | Collaboration phase | Evaluation phase |
|---|---|---|
| H0 — human only | Human collaboration, usual process | Immediate, delayed and transfer tasks without AI |
| H1 — human + TMS | Human collaboration with voluntary transactive-memory scaffolding | Same AI-off assessments |
| A0 — human + AI | Human collaboration with AI assistance, usual process | Same AI-off assessments |
| A1 — human + AI + TMS | Human collaboration with AI and voluntary transactive-memory scaffolding | Same AI-off assessments |

TMS scaffolding may help participants record who has relevant expertise, surface unique facts, route questions, preserve unresolved claims and invite independent contributions. It must not become a compulsory expertise registry or a surveillance mechanism.

All four arms must be assessed on comparable task families, with task order, time budgets, information budgets and scoring rules specified before confirmatory collection. The AI model/runtime configuration is pinned and must match across the two AI-exposure arms. The transactive-memory intervention also has a pinned configuration digest that must match across both TMS arms; non-TMS arms carry NA. These are independent factors and must not change together.

## Measurement contract

The TSV contains one row per randomized team, after a preregistered task-level aggregation step. Rows are not conversation turns or repeated tasks. The exact header is enforced by the Rust parser.

Required measures, each bounded to [0, 1]:

- Baseline group task accuracy.
- Assisted-phase accuracy for AI-exposure arms only; this is descriptive and is not substituted for the AI-off endpoint.
- Immediate AI-off accuracy.
- Delayed AI-off accuracy at the preregistered follow-up interval.
- Transfer accuracy on a held-out task family or changed context.
- Unique member-held fact retention and correct use.
- Expertise-routing accuracy.
- Independently defined agency score from a predeclared rubric.

Immediate, delayed and transfer gains are each calculated as the corresponding AI-off accuracy minus baseline accuracy. Retention, routing and agency remain separate outcomes; they are not collapsed into a single “team intelligence” score.

The input also carries a task-bundle digest, pinned AI-configuration digest for AI arms, pinned TMS-configuration digest for TMS arms, a declared AI-off gate status and a per-team AI-off receipt digest. The current v0.1 code validates digest shape, uniqueness and declared status. It does **not** authenticate the recorder, prove endpoint isolation, or independently establish that cached model outputs and agent processes were removed. A real confirmatory runner must provide that independent isolation evidence.

## Factorial estimands

For an endpoint with arm means H0, H1, A0 and A1:

- AI main effect = ((A0 + A1) - (H0 + H1)) / 2.
- TMS main effect = ((H1 + A1) - (H0 + A0)) / 2.
- AI × TMS interaction = (A1 - A0) - (H1 - H0).

Positive values mean a higher value on the endpoint. For adverse outcomes, define and document the direction before interpreting the contrast. These estimates describe the specified task, population, model and intervention; they are not universal effects of AI or memory scaffolding.

The implementation computes percentile intervals by independently resampling teams with replacement within each randomized arm. The team is the resampling unit. It never treats messages, turns or repeated tasks as independent teams. Fewer than two teams in any arm yields point estimates only. The code's two-team threshold is a software guard, **not** an adequate study sample-size recommendation. Confirmatory sample size must come from a pilot variance estimate and a preregistered power analysis at the group randomization level.

The v0.1 estimator is intentionally descriptive: it does not fit a hierarchical repeated-task model, calculate p-values, adjust for multiple endpoints, solve missing-data problems, or replace a preregistered primary-outcome analysis.

## Synthetic known-answer fixture

The checked-in fixture contains 16 fictional teams, four per arm. It is designed to exercise the estimator, not to represent observed people or an empirical intervention.
Its repeated-character SHA-256-shaped values are placeholders for parser testing, not hashes of real task, AI, TMS or receipt artifacts; the v0.1 validator checks format and consistency only.

Its immediate AI-off endpoint has the following known contrasts:

- AI main effect: -0.0350.
- TMS main effect: +0.1100.
- AI × TMS interaction: +0.0600.

The fixture also has an AI-assisted mean of 0.905 across AI-exposure arms. This intentionally illustrates why assisted output and retained independent capability must be reported separately. None of these synthetic values is a human-subject result, SOTA result, or evidence that TMS is effective.

## Confirmatory advancement gates

Before making a human-benefit or superiority claim:

1. Freeze task families, held-out variants, rubrics, model snapshot, prompts, budgets, randomization, endpoint definitions, exclusion rules, missing-data policy, multiplicity handling and analysis code.
2. Independently score outcomes blind to arm; report rater reliability and adjudication.
3. Prove the AI-off boundary in the runner, including removal of model/API access, agent processes, cached responses, model-generated artifacts and evaluator-only answers from the candidate path.
4. Preserve unique and minority-held information, consent restrictions, dissent and abstention as first-class observations.
5. Report each endpoint, each arm, negative effects, uncertainty, burden, costs and preregistered strata. Do not let an aggregate mean conceal teams that are harmed.
6. Use pilot data to estimate group-level variance, task reliability, attrition and the smallest practically important effect; preregister the confirmatory power plan before collecting confirmatory data.
7. Replicate on held-out task structures and an independent setting before claiming generalization.

Synthetic fixtures establish only that this bounded input/estimator path behaves as specified. Human trials require independent ethics review, informed and revocable consent, fair compensation, privacy and retention controls, and a participant withdrawal path.
