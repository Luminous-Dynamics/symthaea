# JUNG-001 — Evidence-Bounded Jungian Symbolic Interpretation

**Status:** Research contract / design proposal  
**Issue:** [#7365](https://github.com/Luminous-Dynamics/symthaea/issues/7365)  
**Scope:** Reuse the existing Glyph Codex, HDC representations, and epistemic/provenance contracts. No new production cognitive authority is introduced here.

## 1. Research question

Can Symthaea use Jung's analytical psychology to improve contextual interpretation of symbols, narratives, conflicting motives, and personal meaning—without turning interpretive similarity into unsupported psychological facts?

The target is not to make Symthaea a Jungian analyst, to establish that it has a human unconscious, or to encode Jung's whole ontology as literal machine state. The target is an inspectable interpretation capability whose output can be compared against clear baselines and adversarial controls.

## 2. What exists today

The current source anchor is `src/hdc/glyph_basis.rs`. It constructs 11 field-modality vectors from keyword bundles using `TextHdcEncoder`, then describes a learned interaction matrix over modality co-activation. The companion paper is `papers/consciousness-theory/glyph-codex/glyph_codex.tex`.

The paper maps Jung's four psychological functions crossed with introversion/extraversion to eight axes, then adds Shadow Integration, Anima/Animus Balance, and Self-Actualization. It also combines this with Graves/Spiral Dynamics and other consciousness-oriented concepts.

These are useful symbolic-encoding choices to investigate, but their present representation and interaction scores do **not** establish:
- detection of an archetype in the Jungian or clinical sense;
- access to unconscious content;
- validated personality or developmental classification;
- a measured change in consciousness;
- universal cultural meaning for a symbol; or
- evidence that a symbolic reading is true for a particular person.

A keyword-derived vector can show what the encoder associates with its own seed vocabulary. It cannot independently validate the psychological theory used to choose that vocabulary.

## 3. Concept and evidence ontology

Use typed, non-interchangeable records:

| Type | Meaning | What it does not imply |
|---|---|---|
| `PrimaryTextClaim` | A claim attributable to a specific Jung text/edition/translation and location | That Jung's claim is empirically correct |
| `ScholarlyInterpretation` | An interpretation by a named secondary source | Consensus or experimental confirmation |
| `EmpiricalFinding` | A result from a described empirical study | Generality beyond its sample/design |
| `JungianConstruct` | A theory-defined concept such as archetype, complex, persona, shadow, individuation, or the Self | A directly observed mechanism or validated diagnostic label |
| `SymbolicObservation` | An explicit symbol/image/narrative in a source or user-provided text | A universal interpretation |
| `InterpretationHypothesis` | One context-qualified possible reading with supporting and contradicting evidence | A factual claim about the author's/user's hidden motives |
| `OpenQuestion` | An unresolved or underdetermined issue | Permission to fill a gap with an invented answer |

Every source record should bind author, title, edition/translation, publication year, page/section, URL or archive identifier, license/rights status, and a content digest when the underlying text can lawfully be retained. Store short evidence spans only where lawful; do not vendor copyrighted translations or whole books without permission.

## 4. Useful ideas to operationalize

### 4.1 Archetypes and archetypal images are not the same type

Represent an archetype as a theory-level construct and a concrete image/story as an observation or cultural representation. A hero narrative may be compared with a hero-archetype interpretation, but image similarity alone cannot establish an underlying universal archetype.

### 4.2 Complexes as hypotheses about clustered meaning

A cluster of recurring themes, affective language, memories, or narratives may be represented as a **candidate thematic cluster**. Calling it a Jungian complex remains an interpretation, not a direct state measurement. No diagnosis or personal hidden-motive assertion follows automatically.

### 4.3 Shadow as a reflective question, not an accusation

A useful system may ask: “Could there be another perspective or disowned concern worth considering?” It must not assert: “This proves you are projecting” or infer that a user secretly possesses a trait from a symbol match.

### 4.4 Individuation as a narrative lens, not a universal stage score

Symthaea may compare a story to themes of integration, differentiation, conflict, and revision. It should not rank people on a universal individuation scale without a separately validated measure and clear consent.

### 4.5 Anima/animus requires cultural and historical qualification

Jung's historic anima/animus framework is tied to a gender-binary formulation. The system should preserve that context when explaining the original theory, while not imposing the binary as a factual model of every person's identity or psychology. It should allow alternative contemporary interpretations and mark disagreement.

### 4.6 The four functions are a historical typology, not an established cognitive partition

Thinking, feeling, sensation, and intuition can organize comparative discussion. Treating their introverted/extraverted combinations as eight independent computational dimensions is a modeling choice, not a proof that cognition has exactly those dimensions.

## 5. Architecture proposal

```text
lawfully sourced text / user-provided context
              |
              v
  citation + culture + context extraction
              |
              v
 typed symbols, relations, narrative events
              |
              v
 HDC/glyph association proposes candidates
              |
              v
 evidence and contradiction retrieval
              |
              v
 multiple interpretation hypotheses
              |
              v
 uncertainty / context sufficiency / abstention
              |
              v
 readable explanation + sources + alternatives
```

Constraints:

- HDC similarity is a retrieval signal, not a truth score.
- Preserve source context, language, author/culture, and provenance throughout the graph.
- Return several candidate interpretations where appropriate; explain why each was proposed.
- Explicitly separate `association`, `textual support`, `cross-source corroboration`, and `empirical support`.
- Ask a clarifying question if personal or cultural context changes the reading materially.
- Allow “insufficient context,” “traditions disagree,” and “this source does not support that inference.”
- Never convert this subsystem into clinical assessment, user personality typing, action authority, or a consciousness/Φ upgrade.
- Generated interpretations may propose questions or hypotheses. Belief revision and action continue to use their existing independent evidence and authority gates.

## 6. Deterministic evaluation plan

### Development corpus

Use versioned, lawfully usable myths, literature, and scholarly descriptions, plus clearly labeled authored synthetic cases. Freeze each corpus manifest and split before scoring. Do not mix public development material with held-out confirmation.

### Comparators

1. Keyword / bag-of-words baseline.
2. Existing Glyph Codex/HDC projection.
3. Context-aware typed-relation baseline that preserves source, speaker, culture, and narrative roles.
4. Ablations removing cultural context, contradiction retrieval, or alternative-generation steps.

No method is assumed to win.

### Adversarial and negative controls

- Same symbol in two contexts requiring different interpretations.
- Lexically similar images from distinct traditions.
- A culturally specific symbol presented without provenance.
- Symbol keywords injected into an unrelated passage.
- Contradictory source interpretations.
- A passage that contains no support for the tempting interpretation.
- Missing or ambiguous context where abstention is the correct response.
- Gendered archetype vocabulary applied to a context that does not license that inference.

### Metrics

Keep each metric separate:
- source attribution correctness;
- context-sensitive interpretation accuracy against frozen labels;
- unsupported inference rate;
- ability to preserve legitimate competing interpretations;
- contradiction detection;
- abstention precision/coverage;
- explanation traceability to exact evidence spans;
- deterministic replay;
- compute, memory, and latency.

An aggregate score must not hide unsafe or epistemically weak components. Initial public corpus results are development probes only. Freeze unseen cases before tuning and preserve negative results.

## 7. Learning contract

Any update to a symbolic relation, modality interaction, or interpretation preference must record:

- exact prior and resulting artifact identities;
- observations and source lineage used;
- whether the update was supervised, self-generated, or externally observed;
- the update rule and parameters;
- competing explanations and known counterexamples;
- regression results on frozen controls;
- rollback/version semantics.

Self-generated dreams, simulated narratives, and model-produced interpretations may generate hypotheses, but may not validate themselves. A generated interpretation does not become stronger evidence merely because the same system repeats it or because two correlated models agree.

Prevent cultural overgeneralization by retaining tradition-specific relations and measuring transfer separately. Do not collapse culturally distinct meanings into one universal symbolic dictionary for convenience.

## 8. Evidence and scholarly framing

The literature is not settled. The field includes competing definitions of archetype and different claims about what is innate, learned, embodied, narrative, or clinically useful. Keep primary texts, Jungian scholarship, empirical findings, and computational hypotheses separate.

Starter references:

- International Association of Analytical Psychology, [Analytical Psychology overview](https://iaap.org/jung-analytical-psychology/analytical-psychology/): an overview of Jung's concepts and their traditional meanings.
- Roesler (2025), [The Process of Transformation](https://pmc.ncbi.nlm.nih.gov/articles/PMC12121546/): reviews multiple theoretical strands, identifies contested/refuted biological and anthropological arguments, and calls for stronger research on transformation processes.
- McGovern et al. (2025), [Eigenmodes of the deep unconscious](https://academic.oup.com/nc/article/2025/1/niaf039/8293123): offers a predictive-processing/neuroscience synthesis and testable hypotheses; it should be treated as a proposal, not consensus proof.
- Goodwyn (2024), [Demystifying Jung's “Archetypes” with Embodied Cognition](https://guilfordjournals.com/doi/10.1521/pdps.2024.52.3.283): argues for an embodied-cognition reinterpretation.
- Sullivan (2024), [Revisiting Jung's Theory of Archetypes](https://guilfordjournals.com/doi/10.1521/pdps.2024.52.3.253): critiques the arguments advanced for contemporary empirical relevance.
- Rößler (2026), [A Late Confirmation? Review of C. G. Jung's Understanding of Dreams in the Light of Modern Dream Research](https://pmc.ncbi.nlm.nih.gov/articles/PMC13381552/): argues for connections between Jung's dream framework and contemporary hypotheses about dreaming. Treat this as a scholarly synthesis, not proof of universal symbol meanings or a validated computational inference method.

Read and compare these rather than treating any single paper as the verdict.

## 9. Acceptance and claim ceiling

This research track is ready to move forward only when:

- included source rights and provenance are explicit;
- concept types cannot be confused by the serialized schema;
- every interpretation links to supporting and contradicting evidence;
- negative controls catch lexical and cultural-context failures;
- user-specific psychological inferences are not made from symbol scores;
- exposed and held-out data identities are immutable;
- replay and independent scoring can reconstruct the result;
- the system may abstain without penalty when context is insufficient.

Passing the software tests would establish only that the implementation follows this contract. It would not establish the validity of Jung's theory, a universal collective unconscious, clinical effectiveness, subjective experience, or that Symthaea has a human-like psyche.
