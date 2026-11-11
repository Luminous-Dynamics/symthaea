# Threat model and assurance boundary

## Protected property

A low-authorized observer must not receive an observation influenced by protected input unless the exact transformation, source, audience, and purpose are explicitly allowed by the release policy.

## In scope

- direct protected-to-low flows;
- aliases and copies;
- deterministic transformation and hashing;
- multi-source joins and fan-in lineage union;
- exact, expiring release-rule matching;
- source-directness for the permitted declassification operator;
- paired-world public-input mismatch;
- invariance under irrelevant protected-input mutation;
- avoiding output-value hashes in evaluator receipts.

## Out of scope

- interpreting natural-language skill instructions;
- probabilistic or nondeterministic model behavior;
- model hidden state, attention, timing, cache, or network side channels;
- proving that a declared source is correctly classified;
- signatures, issuer authentication, or trusted policy distribution;
- action execution, OS isolation, or persistent nonce consumption;
- privacy of the test oracle itself.

## Trust assumptions

The evaluator controls the world generator, protected/public split, deterministic transformation semantics, low-audience policy, evaluation time, and expiring release grants. The agent under test must not receive evaluator-only labels, paired-world secrets, oracle outputs, or hidden mutation schedules. Evaluator receipts are not an agent-visible telemetry channel.

A passing test establishes only that this finite DSL satisfies the tested relational property. It is not a claim of universal noninterference or production security.
