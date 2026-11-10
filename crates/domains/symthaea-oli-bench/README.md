# symthaea-oli-bench

Source-aware benchmark adapters for the Open-World Life & Intelligence programme.

## First pilot: great-ape hidden-object displacement

Dataset: EVApeCognition v1.0.4, CC BY 4.0, immutable version DOI 10.5281/zenodo.18849382.
Study: Albiach-Serrano, Call & Barth (2010), DOI 10.1002/ajp.20790.

The checked-in fixture is a minimized derivative of the source CSV:
- 375 trials, 25 pseudonymous subjects, 5 conditions;
- 3 trials per subject / condition;
- retains species as a coarse descriptor;
- removes participant names, age, sex and all separate roster fields.

Source data file Git blob SHA: 92499810019fb631fffb508dbacc1de2e550797b.
Exact source tag: v1.0.4.

## Run

\`\`\`bash
cargo test -p symthaea-oli-bench --locked
cargo run -p symthaea-oli-bench --locked
cargo run -p symthaea-oli-bench --locked -- path/to/minimized.csv
\`\`\`

The audit validates task rows and computes condition-level descriptive statistics, deterministic subject-cluster bootstrap intervals, and three leave-one-subject-out baselines.

## Claim ceiling

This is one study and one spatial/object-tracking task. It does not produce an animal intelligence ranking, a universal intelligence score, or a consciousness/sentience inference. It does not establish cross-study or cross-laboratory generalization.
