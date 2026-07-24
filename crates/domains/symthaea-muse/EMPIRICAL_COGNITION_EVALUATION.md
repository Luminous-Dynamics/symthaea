# Empirical Cognition Evaluation

## Status

V8 freezes the evidence path for the first confirmatory Symthaea–Muse Sonata
study. It does not supply positive results. It makes the eventual result harder
to bias, easier to reproduce, and narrower to report.

## Questions remain separate

The program evaluates three independent questions:

1. **Mechanism:** does HDC/CfC temporal state influence the committed FEP
   trajectory when the raw symbolic stream is held fixed?
2. **Prediction:** does the adaptive world model improve held-out symbolic
   outcome prediction over the unchanged hand-authored prior?
3. **Usefulness:** does the resulting policy improve theory-valid Sonata
   outcomes, blinded listener judgment, or artist workflow over fixed,
   random-valid, and hand-authored policies?

No result may be substituted for another. Mechanistic influence is not musical
quality. Prediction calibration is not preference. A listener preference result
is not evidence of consciousness.

## Frozen manifest

`experiment_manifest` is the authority for the study population. It binds:

- preregistration and analysis-plan digests;
- the public commitment to the private randomization key;
- exact policy versions for all four arms;
- primary endpoints and alpha;
- pilot and confirmatory fixtures;
- subject, renderer, soundfont, theory-constraint, and complete input digests;
- tonic, meter, orchestration, seed, and family identity.

At least four pilot fixtures and twenty-four confirmatory fixtures are required.
Related themes, transformations, or orchestrations share a `family_id`. A family
may occur in only one split, preventing a near-duplicate from appearing in both
pilot and confirmatory analysis. Confirmatory seeds and frozen-input digests
must be unique.

## Blinding and randomization

`blinded_study` produces two files:

- a public schedule containing anonymous codes, positions, and artifact
  digests, but no policy labels;
- a private codebook mapping those anonymous presentations to policy arms.

The four arms rotate through presentation positions using a secret-seeded Latin
schedule, so every complete block of four fixtures balances each arm across
each position. Anonymous codes are generated from the same private 256-bit key.

The manifest stores only `SHA256(private_key)`. Schedule generation refuses a
private key that does not open that commitment. The key and codebook must remain
separate from public schedules, listener responses, Studio logs, and analysts
performing blinded data collection.

The deterministic generator supports exact replay. It is not an authorization
or encryption system.

Manifest, schedule, codebook, and raw-evidence bindings use canonical JSON with
lexicographically sorted object keys, so whitespace and object-key order cannot
change the committed identity.

## Raw listener evidence

Listeners submit complete four-presentation blocks. Each included block records:

- an anonymous participant identifier;
- one response for every presentation in the frozen fixture;
- return recognition;
- development-instability and earned-recapitulation ratings;
- a unique preference rank from one through four;
- playback completion, attention-check status, and elapsed time.

Raw listener files contain no arm labels. Included blocks must contain all four
presentations, unique ranks, complete playback, a passed attention check, and
bounded finite ratings.

Excluded blocks are retained with one of the preregistered reasons:

- failed attention check;
- technical playback failure;
- incomplete block;
- duplicate participation.

The compiler ignores excluded blocks but preserves their counts. It does not
permit post-hoc free-text exclusions.

## Artist workflow evidence

Artist workflow blocks are also blinded and complete across all four
presentations. Each arm receives an explicit kept, edited, or rejected outcome
and a non-zero time to commitment. V8 allows one included workflow block per
fixture in the first study, avoiding an aggregate boolean that would conceal
multiple conflicting artist decisions.

## Private compilation

`study_evidence::compile_study_dataset` is the only intended path from raw
anonymous evidence to arm-labelled `CognitiveTrialRecord` values.

Compilation joins:

- the public schedule;
- the private codebook;
- independently generated structural outcomes;
- included listener blocks;
- included artist workflow blocks;
- the frozen manifest and policy versions.

Preference is compiled as the normalized proportion of pairwise wins implied by
the within-block rank. Structural, perceptual, and workflow channels remain
separate.

## Confirmatory analysis

`confirmatory_analysis` analyzes only fixtures declared confirmatory in the
manifest. The analysis plan must exactly match the frozen manifest endpoints,
alpha, manifest digest, schedule digest, and complete confirmatory fixture
count.

The first frozen stop rule requires at least twelve included listeners for every
confirmatory fixture on each listener endpoint. All frozen confirmatory fixtures
must be present; selecting a favorable subset is invalid.

For each primary endpoint, Symthaea is compared with:

- fixed policy for superiority;
- random-valid policy for superiority;
- the hand-authored heuristic for non-inferiority.

Positive effects always favor Symthaea. For time to commitment, the effect is
comparator time minus Symthaea time.

The frozen practical margins remain:

- at least `0.05` over fixed and random-valid on bounded rates;
- no more than `0.02` below the heuristic on bounded rates;
- at least ten seconds faster than fixed and random-valid;
- no more than five seconds slower than the heuristic.

Inference operates on paired fixture-level differences:

- deterministic paired bootstrap confidence intervals;
- one-sided paired sign-randomization tests;
- Holm correction across every preregistered primary comparison.

A comparison passes only when its corrected p-value clears alpha and the lower
confidence bound clears the practical superiority or non-inferiority margin.
An endpoint passes only when all three comparator gates pass. The number of
primary endpoints required for study success is frozen in the analysis plan.

## Temporal confirmatory analysis

`temporal_confirmatory` extends the V7 mechanistic ablation with group-safe
pilot/confirmatory splits and paired bootstrap intervals. At least twenty-four
confirmatory pairs are required.

The gate requires:

- valid V7 paired traces;
- no family crossing between pilot and confirmatory data;
- mean sensory influence at least the frozen `0.005` threshold with a lower
  bootstrap bound above zero;
- mean paired action divergence of at least ten percent with a lower bootstrap
  bound above zero.

A pass remains a mechanism claim only.

## Claim-safe report

`cognitive_evidence_report` has no overall cognition score. It emits separate
statuses for:

- temporal state influencing FEP;
- adaptive prediction improvement;
- theory-valid Sonata operation;
- superiority to simple baselines;
- non-inferiority to the hand-authored heuristic;
- blinded listener benefit;
- artist workflow benefit.

Missing evidence is `NotEvaluated`, invalid evidence is `InvalidEvidence`, and a
valid null result is `NotSupported`. These states must not be collapsed.

## Command-line workflow

Build the helper with the theory feature:

```sh
cargo build -p symthaea-muse --features theory --bin cognitive_study
```

Validate the frozen manifest:

```sh
cognitive_study validate-manifest manifest.json
```

Generate a public schedule and separately protected private codebook. The key
file must contain exactly sixty-four hexadecimal characters and should not be
committed:

```sh
cognitive_study build-schedule \
  manifest.json artifacts.json randomization.key \
  public-schedule.json private-codebook.json
```

Validate the schedule and codebook before data collection:

```sh
cognitive_study validate-schedule \
  manifest.json public-schedule.json private-codebook.json
```

Seal the canonical raw evidence payload after collection is frozen. The digest
covers every field except the digest field itself:

```sh
cognitive_study seal-evidence \
  unsealed-raw-evidence.json raw-evidence.json
```

Then compile the anonymous evidence with the private codebook:

```sh
cognitive_study compile-evidence \
  manifest.json public-schedule.json private-codebook.json \
  raw-evidence.json compiled-dataset.json
```

Run the confirmatory study analysis:

```sh
cognitive_study analyze \
  manifest.json compiled-dataset.json analysis-plan.json report.json
```

Run the separate temporal-mechanism analysis:

```sh
cognitive_study analyze-temporal \
  temporal-records.json temporal-plan.json temporal-report.json
```

## Required operational discipline

- Freeze and hash the manifest, analysis plan, render stack, and artifact set
  before collecting responses.
- Store the randomization key and codebook outside the public study directory.
- Do not inspect pilot outcomes while altering confirmatory fixtures.
- Do not stop confirmatory listener collection early for favorable results.
- Do not replace excluded participants after inspecting arm-level outcomes;
  follow only the frozen stop rule.
- Preserve raw responses, exclusions, compiled records, recipes, audio digests,
  code revision, runtime identity, and every generated report.
- Publish null and negative results without changing endpoints, margins,
  exclusions, split membership, or multiplicity correction.

## Explicit limits

V8 does not implement a mixed-effects model and does not claim population-wide
listener generalization. Its paired fixture bootstrap respects the frozen
musical-input pairing and provides a transparent first confirmatory analysis.
A later multi-site study should preregister listener- and composition-level
mixed effects before collection.

The first study is narrow: Sonata return interventions, the frozen renderer and
soundfont, the declared listener population, and the tested artist workflow.

## V8.2 methodological hardening

V8.2 adds a separately frozen methodology plan, participant-specific Williams
counterbalancing, evidence-to-assignment validation, full theory-verifier
provenance, shared candidate-budget evidence, family-clustered inference, and a
direct complete-ranking analysis. See `METHODOLOGICAL_HARDENING.md`.

The V8 fixture-level analysis remains useful as a transparent descriptive and
robustness view. It is no longer sufficient by itself for the primary
confirmatory preference claim.
