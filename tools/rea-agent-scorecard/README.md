# REA evidence-grounded reverse-engineering agent scorecard

This is an offline, oracle-separated scorer for the REA-AGENT-001 benchmark. It is not a reverse-engineering agent and does not execute benchmark binaries.

## Data custody

The scorer takes two separate files after agent execution:

1. A hidden, rights-cleared oracle manifest containing expected values, exact artifact digests, operation/environment scope, fault families and optional mutation-pair/occurrence expectations.
2. An agent submission manifest containing answers/abstentions and raw claim/result/Evidence JSON strings.

The oracle is not copied into the agent workspace. The scorecard validates suite identity and task IDs, rejects duplicate submissions and unknown tasks, and includes every oracle task in the eligible denominator—even if the agent did not submit it.

The agent cannot self-declare that replay passed. The scorer calls the independent REA claim replay API using the exact raw claim, result and Evidence strings. Duplicate keys are rejected in the oracle, submission, claim and replay paths. The scorer is read-only: it does not execute or extract artifacts.

## Grounding rule

An answer earns evidence-grounded credit only when all of the following hold:

- the answer matches the independently authored expected value under RFC 8785 canonical JSON semantics;
- the claim is the supported v0 provider-reported-observation predicate;
- the exact subject digest, operation and all represented environment dimensions match the oracle;
- the asserted value matches the oracle's expected answer;
- any expected duplicate-member occurrence binding matches the asserted value;
- the independent replay implementation returns ReplayMatch and SupportedWithinScope.

This is benchmark credit, not a production proof that a provider is truthful. The hidden oracle establishes the task's known answer; REA replay establishes that the agent's admitted Evidence reports that value under the exact tested scope. A provider report alone still does not establish real-world truth outside the benchmark's independently known ground truth.

## Metrics reported independently

- Raw full-suite accuracy: correct answers / all eligible oracle tasks.
- Conditional answer accuracy: correct answers / answered tasks.
- Answer coverage and abstention rate: each uses all eligible tasks.
- Evidence-grounded full-suite accuracy: grounded correct answers / all eligible tasks.
- REA parametric leakage rate (PLR): correct answers lacking a valid grounded chain / raw-correct answers. If there are no raw-correct answers, the rate is null, not zero.
- False-supported rate: claimed-supported answers without valid grounding / all agent-reported supported claims.
- Replay coverage, replay match rate, mismatch count and invalid replay count.
- Exact scope and occurrence binding.
- Per-fault-family numerators and denominators.
- Relevant-mutation sensitivity: answer changes correctly when the relevant fact changes.
- Irrelevant-mutation invariance: answer remains stable when unrelated bytes change, while each claim is rebound to the exact new artifact digest.

No composite leaderboard score is emitted. High raw accuracy cannot compensate for false-supported claims or a high PLR.

## Input schemas

- schemas/oracle-v0.schema.json
- schemas/submission-v0.schema.json
- schemas/scorecard-v0.schema.json

Nullable fields in the oracle and submission must be explicit. Raw claim/result/Evidence documents are stored as strings inside the submission so lexical duplicate keys remain visible to the replay parser.

An optional occurrence expectation binds a JSON Pointer in the expected/claimed answer to an exact occurrence index. It can also bind a raw filename digest at a separate JSON Pointer. This makes duplicate filenames with different central-directory occurrences independently scoreable.

Pair metadata has exactly two roles: baseline and mutated. Relevant-mutation pairs must have different expected answers; irrelevant-mutation pairs must have identical expected answers; both must have distinct artifact digests. The scorer reports pair success only when both submitted answers are present and correct. It separately reports whether both answers are evidence-grounded.

## Bounds and command

- Oracle file: 16 MiB.
- Submission file: 64 MiB.
- Tasks: 10,000.
- Evidence records per task: 4,096 (replay API limit).
- Embedded raw Evidence per task: 32 MiB (submission-level bound).
- Per-claim/result limits are enforced again by the independent replay verifier.

Run from the repository root:

    cargo run --release --locked --manifest-path tools/rea-agent-scorecard/Cargo.toml -- hidden-oracle.json agent-submission.json

The scorecard JSON is written to stdout; malformed/oversized inputs fail closed with a nonzero exit status. Keep the oracle file outside the agent's accessible working directory and pass it only to the post-run scorer.

## Qualification boundary

The synthetic scorecard fixtures test scorer behavior, not agent skill. Model comparisons require a preregistered artifact-family split, held-out source/recipe families, rights review, blinded oracles, task-cost budgets and independent human adjudication. Fixed-suite scorecard correctness is not generalized performance.

The external SRE-Bench, AgentRE-Bench and Decompile-Bench protocols are complementary. Do not claim an official leaderboard score without the benchmark owner’s grader and protocol; do not conflate task pass rates, text similarity, evidence-grounded accuracy or PLR.
