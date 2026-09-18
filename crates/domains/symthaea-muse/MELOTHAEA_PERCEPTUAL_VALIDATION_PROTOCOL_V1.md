# Melothaea Perceptual Validation Protocol v1

## Status

**Pre-outcome protocol design. Not executed. Not a positive result.**

This protocol is frozen while the MEL-003C acoustic subjects remain runner-queued.
Its purpose is to prevent the human-evaluation question, stimuli, exclusions, or
analysis from being redesigned around whichever acoustic result eventually
appears.

If predecessor evidence later reveals a defect that invalidates this protocol,
cancel v1 and create a new version with a new lineage. Do not silently amend v1
and present the amended protocol as preregistered.

## Scope

The first Melothaea perceptual study concerns one narrow intervention:

> bounded Sonata accompaniment re-articulation in `RecapitulationPrimary`

The intervention splits qualifying `Harmony`/`Bass` events into consecutive
same-pitch halves while preserving the returning melody, pitch-duration mass,
formal return obligation, and tested theory constraints.

The study does **not** evaluate Melothaea as a complete composer, Symthaea as a
general cognitive system, consciousness, artistic superiority, genre quality,
or listener preference in general.

## Required predecessor line

Execution is blocked until the relevant exact-head subjects have actually run
and their evidence has been reviewed:

1. symbolic density/effect evidence;
2. rendered waveform survival;
3. performed-event localization;
4. independently calibrated matched-window gate;
5. fixed eight-seed introduced-event acoustic panel;
6. score/audio provenance binding;
7. real-music control panel;
8. introduced-vs-control descriptive contrast.

A queued workflow is not evidence. A green structural run does not imply that
the acoustic gate generalized successfully. Acoustic success, if any, still
does not imply human audibility.

## Evidence questions remain separate

The protocol has three human questions. No answer may substitute for another.

### P1 — discriminability

**Question:** can listeners distinguish the baseline and intervention renders
above chance under blinded presentation?

This is an audibility/discrimination question only.

A positive P1 result does not establish that listeners heard the intended
musical dimension, preferred the intervention, or considered it better music.

### P2 — intended-effect attribution

**Question:** when the baseline and intervention are compared, do listeners
identify the intervention as having more distinct accompaniment attacks / greater
local rhythmic-event density?

This is the first human endpoint directly connected to the symbolic
`Density` target.

A positive P2 result does not establish preference or artistic improvement.

### P3 — preference

**Question:** which version does the listener prefer for the excerpt?

P3 is secondary/exploratory in v1. Preference cannot validate P1 or P2.
A listener may correctly hear increased density and dislike it, or prefer a
version without correctly identifying the intended effect.

## Fixed musical subjects

Use the existing fixed Sonata seed panel in full:

```text
3
11
23
41
59
79
97
127
```

Do not select only seeds whose acoustic gate passed.
Do not remove acoustically awkward seeds after inspecting listener responses.
Do not replace a seed because its intervention is hard to hear.

The study population is the fixed set of eight musical subjects unless a new
protocol version is created before human collection begins.

## Stimulus identity and provenance

Every human stimulus must descend from the same score/audio identities bound by
the MEL-003C provenance line.

For every seed preserve:

- baseline symbolic score SHA-256;
- intervention symbolic score SHA-256;
- baseline native-render SHA-256;
- intervention native-render SHA-256;
- renderer identity;
- sample rate;
- code revision;
- render configuration;
- performance-dialect identity;
- excerpt transformation identity;
- final presented-audio SHA-256.

The presented clip manifest must itself be canonicalized and hashed before any
listener schedule is generated.

## Excerpt policy

The intervention concerns the return region, so a generic head excerpt is not
valid evidence.

For each seed, derive the excerpt from the **symbolically declared
`RecapitulationPrimary` region**, with a fixed context rule established before
responses are collected. v1 uses:

```text
one notated bar before RecapitulationPrimary
through
one notated bar after RecapitulationPrimary
```

where available. At piece boundaries, truncate rather than substituting a
different context length.

Map the symbolic boundaries through the same performed timeline used by the
renderer. Do not locate excerpts by searching the waveform for a favorable
transient or by centering on whichever attack produced the largest acoustic
metric.

The exact performed-time boundaries and excerpt hashes must be stored in the
artifact manifest.

## Loudness policy

Loudness can become an unintended cue in a discrimination or density task.
Melothaea already contains the pairwise verified level-matching path used by
`ab_listening_pack`, which:

- uses BS.1770-style K-weighted integrated loudness measurement;
- computes one pair-reachable target;
- applies scalar gain only;
- remeasures the result;
- rejects a pair if the post-match level difference exceeds the declared limit;
- does not run corrective EQ/compression mastering.

Use that verified scalar-gain path for the **primary P1/P2 pack** so a trivial
global-loudness difference is not allowed to stand in for temporal-density
perception.

Also retain the untouched native excerpts as a **secondary sensitivity pack**.
The native pack may show whether the intervention's complete natural realization
is easier to discriminate, but it does not replace the level-controlled primary
pack.

Do not independently master the two variants.

## P1 task — blinded discrimination

Use a double-blind triple-stimulus / ABX-style task.

For each seed:

- `A` and `B` contain the baseline and intervention in concealed randomized
  order;
- `X` is bit-identical to either presented `A` or presented `B` after all fixed
  excerpt/level transforms;
- the listener answers `X = A` or `X = B`;
- chance accuracy is 0.5.

The listener may replay A, B, and X within the trial before committing the
response.

The answer key must not be available to the participant, operator interacting
with the participant, or analyst compiling blinded raw responses.

### P1 primary endpoint

Per-listener correctness across the eight fixed seeds, retained as the full
participant × seed response matrix.

Do not reduce raw evidence to one pooled binomial count before preserving the
participant and musical-subject identities.

## P2 task — intended density attribution

Use a separate blinded two-alternative forced-choice block.

For each seed, present the same level-controlled pair under opaque randomized
left/right labels and ask:

> Which excerpt has more distinct note attacks or re-articulations in the
> accompaniment — in other words, which accompaniment feels locally more
> event-dense?

The target answer is the intervention artifact.

Use the wording above unchanged in the confirmatory study. Do not replace
"event-dense" with emotionally loaded or quality-oriented terms such as
"richer", "more exciting", or "better".

### P2 primary endpoint

Per-listener correctness across the same eight fixed seeds, preserving the
participant × seed response matrix.

## P3 task — preference

After the P2 density judgment for a pair is committed, ask:

> Which version do you prefer in this excerpt?

Allowed responses:

```text
left
right
no preference
```

Preference is analyzed separately. `No preference` is not coerced into either
arm.

Do not reveal whether the listener's P2 judgment was correct before P3.

## Block order

Use this fixed order:

1. P1 discrimination block;
2. P2 density-attribution + P3 preference block.

This prevents the explicit density wording in P2 from priming the earlier P1
question about what feature to listen for.

Within each block, seed/trial order and A/B or left/right assignment are
participant-specific and generated from the existing secret-seeded study
scheduler.

## Familiarization

Before the scored blocks, provide practice trials that are not among the eight
confirmatory Sonata subjects.

Practice may teach:

- how ABX controls work;
- how to replay/submit;
- what "distinct note attacks or re-articulations" means operationally.

Practice responses are never included in confirmatory estimates.

Do not train listeners on the exact confirmatory baseline/intervention pairs.

## Listener population

Record, without using as post-hoc exclusion criteria:

- age band;
- self-reported hearing difficulty;
- headphone/speaker category;
- musical training band;
- audio-production experience band;
- familiarity with Sonata form.

The first study does not claim population-wide generality. Report the actual
sample composition.

Do not infer protected traits that were not voluntarily collected for the
study.

## Listening environment

Prefer controlled headphone listening.

At minimum require:

- stereo playback;
- no mono downmix;
- no active system EQ/spatial enhancement known to the participant;
- a quiet environment;
- successful playback/attention checks;
- completion of the fixed excerpt before response is accepted where supported
  by the existing study runner.

Record device/environment metadata as context rather than silently excluding a
participant after observing outcomes.

## Exclusion policy

Use only frozen operational exclusions already supported by the cognition-study
framework:

- failed attention check;
- technical playback failure;
- incomplete block;
- duplicate participation.

Retain excluded blocks and exclusion reasons.

Do not add exclusions for:

- low accuracy;
- surprising preferences;
- lack of musical training;
- an unfavorable seed;
- disagreement with the acoustic gate.

## Sample-size and stopping discipline

Minimum confirmatory target: **24 included listeners**, consistent with the
existing confirmatory fixture discipline.

Before enrolling the first confirmatory listener, produce and seal a separate
simulation-based power artifact over plausible discrimination/attribution rates
without using human outcomes from this experiment.

That artifact may increase the final fixed sample size above 24.
It may not reduce the minimum below 24.

Once the first confirmatory response is collected:

- do not alter the target sample size from observed effects;
- do not stop early for a favorable result;
- do not recruit replacement participants based on unblinded arm-level results.

## Analysis hierarchy

### Raw/descriptive views

Always report:

- participant × seed response matrix;
- per-participant accuracy;
- per-seed accuracy;
- overall response count;
- missing/excluded blocks;
- device/training composition;
- P3 no-preference rate.

A naive pooled exact-binomial statistic may be shown only if explicitly labeled
as ignoring repeated-listener and repeated-seed dependence. It is not the
primary inferential result.

### Cross-classified dependence

Responses are clustered both by listener and by musical subject. v1 therefore
must not treat all participant × seed trials as independent Bernoulli samples.

Before confirmatory collection, freeze one of the following analysis paths in a
separate analysis-plan artifact:

1. a participant- and musical-subject random-effects logistic model; or
2. a deterministic two-way cluster/bootstrap procedure that resamples listener
   and seed identities and retains the complete crossed response matrix.

The implementation and simulation checks for the chosen path must be frozen
before unblinded confirmatory analysis.

Do not choose between analysis methods based on which gives the smaller
p-value after collection.

## Practical effect reporting

For P1 and P2 separately report:

- estimated accuracy;
- uncertainty interval;
- distance above 0.5 chance;
- participant-level distribution;
- seed-level distribution.

Do not create one combined "perception score" from P1 and P2.

A future analysis plan may preregister a practical-support margin above chance,
but **this protocol document does not choose that margin from unseen acoustic
or future listener outcomes**. The margin must be fixed in the sealed analysis
plan before confirmatory collection.

## Multiplicity

P1 discrimination and P2 density attribution are separate primary claims.
If both receive null-hypothesis tests, the sealed analysis plan must declare the
familywise-error procedure before data collection (for example Holm correction)
or explicitly use an intersection-union claim whose logic is fixed in advance.

P3 preference is not allowed into the P1/P2 multiplicity family as a substitute
endpoint.

## Randomization and blinding

Reuse the existing cognition-study infrastructure:

- public schedule with anonymous codes and artifact digests;
- private codebook;
- commitment to the private randomization key;
- participant-specific counterbalancing;
- canonical JSON identities;
- raw evidence without arm labels;
- forward hash-chained session logs where the study runner is used;
- sealed evidence before private compilation/unblinding.

The operator should not hand-edit A/B assignments.

## Artifact hierarchy

Freeze, in order:

1. source score/audio provenance;
2. excerpt specification;
3. primary level-controlled clips;
4. secondary native clips;
5. artifact manifest and digests;
6. protocol digest;
7. analysis-plan digest;
8. participant schedule and private codebook;
9. raw responses;
10. sealed raw-evidence digest;
11. compiled blinded/unblinded dataset according to the established workflow;
12. final report.

Environment drift after evidence collection begins requires a new evidence
lineage rather than silently regenerating stimuli.

## Relationship to existing tools

### `ab_listening_pack`

Useful for:

- verified pairwise loudness matching;
- opaque clip names;
- fast operator/listener pilot checks;
- source/output audio hashes.

It is not, by itself, sufficient for the confirmatory P1/P2 claims because its
own documentation correctly notes listener/case dependence and the absence of
full sealing/attention/cohort controls.

### `cognitive_study` / `cognitive_study_runner`

Use for the confirmatory workflow because the existing apparatus already
supports:

- frozen manifests and methodology;
- secret-seeded schedules;
- participant assignment validation;
- raw evidence and exclusions;
- reproducibility attestations;
- confirmatory amendment controls;
- sealed collection and release discipline.

Extend these tools for ABX and density-attribution response types rather than
creating an unrelated web survey whose provenance cannot join the current
evidence chain.

## External methodological references

The study is not presented as a formally compliant ITU test, but its design is
informed by established subjective-audio principles:

- **ITU-R BS.1116-3**, _Methods for the subjective assessment of small
  impairments in audio systems_: double-blind reference/object presentation,
  controlled listening, repeatable access to stimuli, and careful subject/test
  methodology.
  <https://www.itu.int/rec/R-REC-BS.1116>
- **ITU-R BS.1534-3 (MUSHRA)**, _Method for the subjective assessment of
  intermediate quality level of audio systems_: explicit attention to test
  material, subject selection, listening conditions, hidden controls, and
  statistical/reporting discipline.
  <https://www.itu.int/rec/R-REC-BS.1534>

Melothaea's task is not audio-codec impairment scoring, so do not describe this
protocol as BS.1116- or MUSHRA-compliant. The relevant principles are borrowed;
the endpoint is different.

## Claims allowed after execution

Depending on the actual evidence, the strongest narrow claims are:

### If P1 is supported

> Under the frozen tested stimuli, renderer, listening protocol, and listener
> sample, baseline and accompaniment-rearticulated Sonata returns were human-
> discriminable above the preregistered criterion.

### If P2 is supported

> Under the frozen tested conditions, listeners identified the re-articulated
> version as having greater accompaniment attack/event density above the
> preregistered criterion.

### If P3 favors one condition

> In this tested listener sample, that condition received more stated
> preferences under the frozen protocol.

Do not transform any of those into:

- "Melothaea makes better music";
- "humans prefer cognitive composition";
- "the listener model is validated";
- "Symthaea understands music";
- a consciousness claim;
- a general claim about all styles, renderers, populations, or interventions.

## Null and mixed outcomes

Null and mixed results are valid outcomes.

Examples:

- P1 positive, P2 null: the intervention is audible but not reliably heard as
  the intended density increase.
- P1 positive, P2 positive, P3 baseline-preferred: control works perceptually
  but the tested listeners prefer less re-articulation.
- P1 null: do not use P3 preference noise to claim perceptual success.
- strong acoustic localization with P1 null: machine-side acoustic evidence did
  not establish human discriminability.
- P1/P2 positive with poor C5 specificity: investigate whether the acoustic gate
  is a bad mechanistic proxy even if humans hear the intervention.

The evidence architecture should make these disagreements visible rather than
forcing them into one success bit.

## Execution gate

**Do not execute confirmatory listener collection yet.**

First:

1. obtain exact-head execution for the MEL-003C acoustic stack;
2. review introduced and real-music control distributions without changing this
   protocol;
3. implement and freeze the ABX/density response schemas in the existing study
   framework;
4. implement/freeze the selected cross-classified analysis method;
5. generate and seal the power artifact;
6. generate and validate the perceptual artifact pack;
7. freeze the final manifest/analysis plan;
8. only then enroll confirmatory listeners.
