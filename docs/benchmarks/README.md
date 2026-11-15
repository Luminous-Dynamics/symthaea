# Luminous Benchmark Programme

This directory separates three questions that are often collapsed into one AI score:

1. **Cognitive capability:** what can a system perceive, remember, reason about, learn, and generate?
2. **Operational trustworthiness:** can a system perform delegated work within evidence, authority, and safety constraints?
3. **Sympoetic collaboration:** can humans and AI systems create and solve problems together while preserving participant agency, pluralism, and traceable contributions?

## Benchmark map

| Programme | Primary question | Status |
|---|---|---|
| Symthaea Psych-Bench | Does the architecture exhibit the cognitive and psychometric behaviours represented by its test suite? | Existing suite; see its own README and calibration transparency report |
| Agent-operations interoperability | How does a Symthaea-driven agent compare on external, role-aware operations evaluations? | Research plan only; external AOBench is not copied or modified here |
| SympoeticBench | Does human-AI collaboration produce useful joint capability without eroding agency, diversity, consent, or provenance? | Protocol and development seed corpus proposed in v0.1 |

These programmes are complementary, not interchangeable. Psych-Bench is not a human-AI collaboration benchmark. An operations score is not evidence of general intelligence. A creative artifact's quality is not evidence that collaboration was reciprocal.

## Why a separate SympoeticBench

The existing Psych-Bench README correctly cautions that most of its tests use synthetic HDC stimuli and validate architectural mechanics rather than human-normed task performance. Its calibration transparency report also identifies substantial post-hoc tuning in the existing suite. Those disclosures are useful precedents: SympoeticBench must use frozen task definitions, preregistered scoring, external human baselines, and held-out scenarios from its first version.

External research now measures several adjacent constructs:

- Common-ground formation and repair in interactive human-AI collaboration.
- Multi-agent collaboration over long horizons, including causal contribution analysis.
- Creative quality, professional disagreement, diversity, and perceived ownership.
- Human-AI interaction modes, including reflective human-led and model-led workflows.

SympoeticBench should connect these lines of work while addressing a missing intersection: reciprocal human-AI creation over time, with explicit tests for consent, contribution provenance, agency preservation, constructive disagreement, and population-level diversity.

## Evidence contract

Every result intended for comparison must identify:

- Benchmark release, scenario IDs, split, and scenario-content digests.
- Exact system, model snapshot, adapter, prompts, tool configuration, and memory policy.
- Runtime environment, hardware class, sampling settings, and random seeds.
- Complete event trace or a consented, privacy-minimised trace digest.
- Output artifact digest and scoring implementation version.
- Human-rater protocol, rater agreement, uncertainty intervals, and missing-data rules.
- Cost, latency, human time, and interaction burden.

Hard safety failures are reported separately and cannot be averaged away by high artifact quality. No single weighted aggregate is defined in v0.1.

## Scientific boundaries

SympoeticBench evaluates observable interaction behaviour and participant-reported experience. It does not test or imply AI consciousness, sentience, personhood, emotional experience, or moral status. It does not determine legal authorship or ownership. Attribution and permission measurements are evidence about workflow and governance, not a legal ruling.

## Current release status

Version 0.1 is a design and development release. It contains a proposed protocol, scenario schema, and 14-scenario development seed corpus. It does not contain human-subject study results, validated scales, a public leaderboard, or SOTA claims. The seed corpus is for harness development and pilot design; it is not a locked test set.

## Next steps

1. Independently review the construct definitions and scenario schema.
2. Implement a trace recorder and deterministic scenario runner.
3. Add a reference scripted partner and a transparent baseline agent.
4. Pilot task clarity and rubric reliability before a powered human study.
5. Preregister the confirmatory study and freeze its held-out scenario split.
6. Publish results only with full version, cost, and uncertainty metadata.
