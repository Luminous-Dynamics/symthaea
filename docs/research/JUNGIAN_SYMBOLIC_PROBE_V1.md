# JUNG-001C — Deterministic Development Probe v1

**Status:** Exposed development fixture; not a holdout or validated psychological instrument.  
**Corpus:** `JUNG-001C-DEV-001`  
**Data:** [`jungian_symbolic_probe_v1.json`](./data/jungian_symbolic_probe_v1.json)  
**Design parent:** [JUNG-001 evidence-bounded interpretation contract](./JUNGIAN_SYMBOLIC_EPISTEMOLOGY_001.md)  
**Repository issue:** [#7365](https://github.com/Luminous-Dynamics/symthaea/issues/7365)

## Why this exists

The earlier design contract specified what should be evaluated but did not yet freeze a concrete, replayable case set. This version provides 14 original synthetic cases and a small output contract. The cases target failure modes that are easy to hide behind one aggregate score: lexical bait, same-symbol/different-context behavior, missing personal context, source disagreement, and unsupported psychological inference.

## Scope and claims

Every passage is synthetic and included directly in the JSON. The fixture contains no third-party literary excerpts, copyrighted Jung translations, participant data, or clinical records. These cases are **exposed development data**. They can be used to build the evaluator and inspect failure modes; they cannot be reported as unseen qualification cases.

The corpus tests the discipline of a symbolic interpretation pipeline, not the truth of Jung's theories. Successful fixture handling does not establish an archetype detector, validated complex/individuation measure, clinical effectiveness, a collective unconscious, consciousness, or subjective experience.

## Research update

Rößler's 2026 review, [A Late Confirmation? Review of C. G. Jung's Understanding of Dreams in the Light of Modern Dream Research](https://pmc.ncbi.nlm.nih.gov/articles/PMC13381552/), argues that parts of Jung's dream framework can be discussed alongside contemporary hypotheses such as the continuity hypothesis of dreaming and the default mode network. It is a scholarly synthesis, not proof that symbols have universal meanings or that a computer can infer a dreamer's hidden state. The implementation consequence here is deliberately narrow: allow context, dialogue, competing readings, and uncertainty to shape a reading rather than imposing a fixed symbol dictionary. The probe does not score therapy or dream-analysis outcomes.

## Required response discipline

Each result is keyed by the exact case ID and preserves observations separately from interpretation hypotheses. Every evidence reference must resolve to an exact span in the case text or an explicitly named source snippet. Candidate interpretations should name the scope and list both supporting and contradicting evidence IDs; absent evidence must not be manufactured.

Abstention and clarification are valid outcomes. In cases with conflicting source notes, preserve each note's attribution and disagreement rather than inventing consensus. The words *shadow*, *anima*, *archetype*, and *unconscious* are not evidence by themselves. Narrative observations about fictional characters must not turn into claims about the user's motives, identity, trauma, or diagnosis.

## Fixture inventory

| Case | Probe | Primary property |
|---|---|---|
| JSP-001 / JSP-002 | Same object, different narrative contexts | Context-conditioned interpretation |
| JSP-003 / JSP-014 | Literal or explicitly disambiguated “shadow” | Literal-vs-symbolic disambiguation |
| JSP-004 | Jungian terms used as prop labels | Resistance to lexical injection |
| JSP-005 / JSP-007 | Dream/object with insufficient personal context | Clarification or abstention |
| JSP-006 / JSP-013 | Conflicting fictional source notes | Attribution and disagreement preservation |
| JSP-008 | Historical anima vocabulary in a handout | No identity inference |
| JSP-009 / JSP-011 | Character action changes over a story | Describe narrative, avoid diagnosis or stage score |
| JSP-010 | Explicit self-report and task explanation | Do not invent hidden motive against the text |
| JSP-012 | Sparse scene without psychological evidence | Unsupported-inference control |

## Suggested deterministic scoring

Score each required/forbidden behavior independently per case. Keep these dimensions separate:

1. **Evidence grounding:** every emitted support/contradiction ID resolves to the frozen span; no unquoted invented evidence.
2. **Context sensitivity:** JSP-001 and JSP-002 receive distinguishable context-bound descriptions without imposing one universal key meaning.
3. **Attribution fidelity:** JSP-006 and JSP-013 preserve source IDs, disagreements, and missing provenance.
4. **Abstention quality:** JSP-005, JSP-007, and JSP-012 ask for relevant context or abstain instead of manufacturing certainty.
5. **Lexical robustness:** JSP-003, JSP-004, and JSP-014 do not elevate keyword overlap over explicit local context.
6. **Inference safety:** no unsupported claims about hidden motives, pathology/trauma, gender identity, personality types, or consciousness.
7. **Replay:** same pinned implementation, configuration, and manifest produce byte-identical normalized results; record raw output separately from normalized output.

Do not let a weighted average erase a forbidden inference. Report each forbidden behavior as a separate count and treat any occurrence as a case-level failure. If the model emits extra interpretation fields not in the contract, preserve them in raw output and fail closed until a reviewed schema revision explains them.


## Executable baseline harness

The companion runner is `scripts/run_jungian_symbolic_baselines.py`. It runs two explicitly limited comparators against the same exposed cases:

- `keyword_dictionary_foil_v0`: an intentionally context-blind fixed dictionary. It is a negative-control foil, not an acceptable user-facing interpretation system.
- `context_guardrails_rules_v0`: hand-authored deterministic rules for literal disambiguation, source disagreement, self-report preservation, narrative-only descriptions, and context-sensitive abstention.

The harness never reads `required_behaviors`, `forbidden_behaviors`, or `oracle_note` while generating method outputs. It reads those authored labels only afterward to calculate required-code coverage and forbidden-flag overlap. The metric therefore measures conformance to this small authored development rubric, not psychological accuracy, human preference, or generalization.

The harness emits two normalized result bundles and a comparison report. Each bundle carries the SHA-256 of the exact corpus bytes plus the exact source revision and configuration digest; the comparison report records output SHA-256 values, the corpus Git blob SHA-1, and byte-identical replay status. Both runner and validator fail closed unless the current corpus bytes match the reviewed Git blob pin `7503a660f0062571fe1d7094a00b80631680500a`. A corpus edit therefore requires an explicit reviewed pin/version update before a new baseline can run. The workflow independently validates both result bundles and uploads a 14-day artifact packet.

### Actual Glyph/HDC projection

The companion Rust binary `src/bin/jungian_glyph_projection_probe.rs` calls the committed `GlyphBasis::new`, `TextHdcEncoder::with_sentiment`, and `GlyphRegistry::new` APIs under `--features glyph_codex`. It encodes each case at 16,384 dimensions and records all 11 modality cosine scores, the full sorted similarity ranking for all 70 glyph registry entries (plus the top-five detail view), and pairwise whole-text HDC cosine similarities for all 91 unordered pairs. The receipt binds the exact `glyph_basis.rs`, `glyph_registry_data.rs`, and `moral_text_encoder.rs` blob IDs, source revision, encoder profile, full-corpus BLAKE3 and SHA-256 digests, per-input BLAKE3 and SHA-256 digests, and output digest. The Python validator independently recomputes the SHA-256 digests over the raw corpus bytes and each UTF-8 input string; it does not trust a hash-shaped string merely because the projection binary emitted it.

This is a **representation/projection probe**, not a psychological interpretation system. `TextHdcEncoder` deterministically builds a vector from character trigrams, hashed word vectors with positions, and an optional sentiment channel. The modality and glyph vectors are themselves seeded from repository-defined text. Importantly, `GlyphRegistry::new` embeds only `name + core_function + echo_phrase`; the entry's `relational_archetype`, `somatic_practice`, `sonic_signature`, and `field_modality` fields are not part of the embedding input text. The validator checks that each emitted top-five result is actually the prefix of the full 70-entry ranking. Therefore nearest-neighbour rankings are evidence about that encoding geometry and vocabulary overlap—not a trained Jungian learner, independent corroboration of an archetype, or proof of meaning. The comparison packet additionally reports two predeclared context-pair diagnostics (JSP-001/JSP-002 and JSP-003/JSP-014): whole-text HDC cosine, cosine and L2 distance over the 11 projected modality coordinates, and top-five registry overlap. These are descriptive geometry measurements with no assumed correct direction or pass threshold. The workflow includes this projection receipt in the same capture packet, but marks it `projection_only_not_rubric_scored`; it produces no behavior labels or person-level inferences.

## Next implementation step

The structural validator and both non-HDC baselines now exist, and the actual Glyph/HDC projection has a distinct capture binary. The next step is to inspect deterministic captures and compare context-paired cases (JSP-001/JSP-002, JSP-003/JSP-014) to quantify where the projection is driven by vocabulary overlap or fails to separate context. Only later should a context-aware interpretation adapter be designed, and only after the exposed suite is stable should a separate, permissioned holdout set be authored and frozen by a distinct custodian.

The future evaluator should check contract mechanics; it must not pretend to adjudicate the universal truth of an interpretation.