# Proof-Guided Mathematical Discovery and Reuse

**Status:** Research RFC; proposed, not implemented or qualified.  
**Date:** 2026-10-16  
**Repository:** `Luminous-Dynamics/symthaea`

## Thesis

Move Symthaea from a proof-script generator with hand-tuned tactic cascades toward a **proof-guided discovery loop**: retrieve relevant verified mathematics, propose a conjecture or proof strategy, let Lean independently check the resulting proof, and preserve enough provenance to reproduce the result later.

The model is a search and proposal mechanism—not a source of proof authority. Accepted mathematical claims must pass the pinned verifier and the repository's existing evidence policy.

## Why now

OpenAI's public [math repository](https://github.com/openai/math) contains mathematical manuscripts and supporting artifacts, including a growing Lean library. Its README explicitly warns that the collection is at mixed verification stages and that not all results have Lean formalizations. The [Lean subtree](https://github.com/openai/math/tree/main/lean) also documents a dedicated formalization catalogue and comparator challenges.

Symthaea already has:
- `crates/bridges/symthaea-lean-bridge` for emitting/checking Lean proof scripts;
- symbolic-regression and conjecture/invariant-discovery machinery;
- an existing cognition-to-Lean research plan at `docs/phase6-scoping/cognition-to-lean-bridge.md`;
- the diagonal-Gaussian entropy theorem work in [issue #6220](https://github.com/Luminous-Dynamics/symthaea/issues/6220).

This proposal complements those efforts; it does not replace them or create a competing Lean/Mathlib environment.

## Paradigm shift: from a proof dump to a provenance graph

Index *mathematical claims and their dependencies*, not just source files or tactic strings. Each indexed item should carry:

- upstream repository and immutable commit SHA;
- source path and blob digest;
- paper/family identifier and declared theorem name;
- exact formal statement, explicit assumptions, types, and domain restrictions;
- proof status: manuscript-only, formalization present, accepted in a pinned environment, or independently adapted and accepted;
- Lean toolchain, Mathlib/package-manifest digests, and required imports;
- license/attribution metadata and any local modifications;
- negative controls and known limitations.

Never label a result “formally verified” solely because it appears in OpenAI's repository. A source proof is a candidate until the actual declared theorem is checked in an admitted, pinned environment. If a theorem is adapted, the local statement and proof receive a distinct identity.

Initially store **metadata and narrow dependency references**, not a wholesale vendored copy of OpenAI's Lean library. Avoid importing large unrelated transitive dependency graphs. Reuse Mathlib's established lemmas where practical. The initial candidate-only index is [`proof-corpus-manifest.yaml`](proof-corpus-manifest.yaml); its pinned entries are not Symthaea-qualified and must not be treated as an accepted proof cache.

## Proposed loop

1. **Normalize the goal.** Extract its AST, types, assumptions, domains, and semantic family. Preserve the original statement.
2. **Retrieve candidates.** Combine exact-symbol/type matching, operator and hypothesis features, dependency proximity, and optional HDC similarity. HDC scores may rank results but must never establish mathematical equivalence.
3. **Generate alternatives.** Propose lemma applications, small proof plans, or tactic-cascade variants. Keep each attempt bounded and independently attributable.
4. **Check independently.** Run the repository-approved Lean checker in a bounded subprocess. Reject `sorry`, unexpected axioms, altered theorem statements, unpinned dependency drift, timeouts, and incomplete output according to the admitted proof policy.
5. **Test sensitivity.** Run intentionally false or weakened/strengthened variants: flipped signs, removed positivity/domain premises, reversed orientation, boundary values, and unauthorized generalizations. The verifier harness itself must demonstrate that it rejects these controls.
6. **Record evidence.** Bind theorem source, dependency/environment identity, exact repository head, command, output digest, and exit status in a machine-readable receipt.
7. **Learn only from labelled outcomes.** A successful Lean check is the positive label for a proof attempt. A timeout, elaboration failure, or missing environment is not a mathematical counterexample; preserve those categories separately.

Treat Lean source as untrusted input: no network access, bounded wall time/memory/output, isolated workspace, and no implicit execution of arbitrary upstream build scripts.

## First pilot: entropy/logarithm theorem retrieval

Use [#6220](https://github.com/Luminous-Dynamics/symthaea/issues/6220) as the first evaluation target because its theorem families and hostile controls are already explicit. Candidate references include:

- [OpenAI: logarithmic and entropy inequalities](https://github.com/openai/math/blob/1581d4cbdd69a0faf34b2089614fe2a814316a6a/lean/OAI/Analysis/LpDimension/Entropy.lean)
- [OpenAI: relative entropy](https://github.com/openai/math/blob/1581d4cbdd69a0faf34b2089614fe2a814316a6a/lean/OAI/Probability/SKRatio/Entropy/Entropy.lean)

These files are **references to inspect**, not yet approved runtime dependencies. Extract only relevant lemma names, precise assumptions, and proof ideas. Prove the Symthaea statements under its canonical environment. In particular, keep separate:
- exact theorems over real numbers;
- correctness/refinement of the Rust model and its `f64` computations;
- scientific claims about free-energy or real-world information gain.

Do not acquire Mathlib ad hoc or create a FEP-local Lake root. The exact admitted Lean project, Mathlib revision, toolchain bytes, manifest digest and qualification workflow must be used as specified by the existing formal-verification lineage.

## Executable first slice

The branch adds a Rust `xtask` preflight validator for the candidate manifest:

```sh
cargo run -p xtask -- proof-corpus-validate
```

The command is intentionally metadata-only. It validates the schema version, required negative-control policy, unique entry IDs, lowercase 40-hex commit/blob identities, exact source-path-to-pinned-URL correspondence, and candidate-only status. It does not fetch upstream files, compile Lean, or qualify any theorem. Unit tests cover a valid fixture, floating branch URL rejection, accidental status promotion, omitted hostile controls, and duplicate identities.

This provides a deterministic low-cost gate before any proof retrieval/qualification work. Its own compile/test result is still unverified until executed against this exact branch head.

## Reuse existing Symthaea provenance work

Symthaea main now also contains an explicit [evidence provenance derivation graph](https://github.com/Luminous-Dynamics/symthaea/blob/main/src/knowledge/evidence_provenance_graph.rs) and its [benchmark RFC](https://github.com/Luminous-Dynamics/symthaea/blob/main/docs/research/PROVENANCE_DERIVATION_GRAPH_BENCHMARK_2026-10-16.md). The proof corpus should reuse that work where the abstraction fits rather than build a competing generic origin/derivation graph.

The connection needs a typed adapter, not a semantic shortcut:

- A proof-corpus entry can be a derived evidence artifact with pinned repository revision, source path/blob SHA, declared theorem, formalization status, environment identity, and license/attribution provenance.
- A proof-check receipt is a separate qualification event bound to that exact entry and exact source/checker environment.
- A derivative article, repository README, formalization source, successful compile, and verified theorem are different evidence kinds; ancestry must not silently promote one to another.
- Multiple sources in a derivation graph do not establish mathematical truth or independent corroboration. The Lean checker still owns proof acceptance, and the provenance graph only records/validates lineage.
- Unknown source lineage, stale check receipts, missing exact-head identity, or ambiguous theorem mapping must remain explicit blockers rather than being filled by similarity scores.

This gives us a coherent platform-level design: Symthaea's existing provenance graph tracks **where evidence came from and how it was derived**; the proof corpus adds **what exact mathematical statement the evidence concerns and whether the pinned checker accepted it**. Do not duplicate the provenance graph implementation in this RFC branch or claim integration before a reviewed adapter exists.

## External retrieval baseline: LeanSearch v2

The pinned candidate manifest also records [LeanSearch v2](https://github.com/frenzymath/LeanSearch-v2/tree/94f4888cbaf9f4322535755f86cbac690ec18080) as an **evaluation reference**, not a Symthaea production dependency. Its authors distinguish standard semantic retrieval from a reasoning loop for **global premise retrieval**—recovering a set of lemmas needed by a complete theorem rather than selecting only the next tactic's premise.

The paper reports (these are upstream reported results, not independently reproduced by this RFC) 46.1% recovery of ground-truth premise groups within 10 retrieved candidates on a 69-query research-level Mathlib benchmark, versus 38.0% for the next-best compared retrieval system and 9.3% for premise-selection baselines. A fixed downstream prover loop reported 20% proof success with its retrieval versus 16% for the next-best compared retriever and 4% without retrieval. See the [paper](https://arxiv.org/abs/2605.13137) and its [benchmark repository](https://github.com/frenzymath/LeanSearch-v2/tree/94f4888cbaf9f4322535755f86cbac690ec18080/benchmark).

This motivates a key design change: evaluate **whole-premise-set recall and downstream proof closure together**. A system that ranks one individually relevant lemma highly but misses the other essential premises may still fail to prove the goal. Compare, on the same frozen queries:

- B0: current hand-coded tactic cascade (no learned retrieval);
- B1: deterministic lexical/type/operator retrieval;
- B2: LeanSearch v2 reference/retrieval outputs where the exact benchmark split and environment are comparable;
- B3: HDC similarity reranking alone;
- B4: hybrid retrieval combining structural filters, global-premise candidates, and optional HDC reranking.

Do not claim a fair head-to-head result from published leaderboard figures alone. LeanSearch v2 documents a Mathlib v4.28.0-rc1 corpus and GPU/embedding/reranker dependencies for local reproduction, while Symthaea's canonical environment is separately pinned. Do not introduce its Python/GPU stack into production Rust. First run the deterministic local baseline; use external results as context until a compatible, reproducible retrieval run is available. Retrieval quality is not proof validity: every generated proof remains subject to Symthaea's independent checker and hostile controls.

## Evaluation design

Freeze a baseline for the current hand-coded cascade before introducing retrieval. Partition by **goal family/source**, not random individual goals, to limit train/test leakage. Include curated, auto-ingested, and generated hostile examples. Publish the task manifest, exclusions and exact verifier environment.

| Dimension | First acceptance target |
|---|---|
| Proof soundness gate | Zero accepted results with `sorry`, unauthorized axioms, unverified placeholders, or altered theorem statements |
| Provenance completeness | 100% of accepted results bind source/theorem identity and pinned checker environment |
| Reproducibility | Same pinned head/environment reproduces the same acceptance classification; changes invalidate old receipts |
| Hostile sensitivity | Every required negative control is rejected; missing premises and sign/orientation mutations cannot pass |
| Retrieval usefulness | Measure Recall@1/5/10 against a human-reviewed candidate-lemma set before claiming retrieval quality |
| Held-out utility | Establish baseline first; target at least +10 percentage points absolute proof closure on held-out goal families versus the frozen baseline, with no weakening of hard gates |
| Cost | Report median/p95 time, timeout rate, memory, and imported dependency count; no unbounded runs |

These are **proposed targets**, not measured results. Report every denominator and distinguish the curated from auto-ingested split. If HDC retrieval does not improve held-out results, keep it advisory or remove it; do not tune on the test set.

## Phased implementation

### P0 — inventory and reproducible manifest
Create a small versioned manifest with immutable upstream references, license metadata, theorem status and domain tags. No generated proof is accepted in this phase.

### P1 — minimal retrieval baseline
Implement a deterministic lexical/type/operator retrieval baseline and reviewer-labelled candidate set. Record top-k results. HDC ranking is an ablation, not a dependency of correctness.

### P2 — entropy pilot
Apply a small number of relevant lemmas to #6220 using the approved environment. Add hostile controls and exact-head receipts.

### P3 — learned strategy selection
Compare existing cascade, deterministic retrieval-ranked cascades, and optional HDC-ranked cascades on family-held-out tasks. Retain the simplest strategy that earns a measurable improvement.

### P4 — downstream implementation contracts
For any theorem intended to constrain Rust behavior, specify a separate bridge/refinement obligation and floating-point error policy. Never infer implementation correctness from a Lean theorem alone.

## Stop conditions

Stop or narrow the work if upstream licensing/provenance is unclear; the approved Mathlib environment cannot load the required result; formal statements do not match the intended claim; the corpus encourages statement leakage; hostile controls pass; or retrieval does not beat the baseline. Publish negative results.

## Definition of done

The pilot is complete only when an independent reviewer can reproduce the accepted theorem from pinned source and environment, inspect the complete assumptions, verify the negative controls, and see a held-out comparison against the frozen baseline. Runner status alone is not qualification.

## References

- [OpenAI math repository](https://github.com/openai/math)
- [OpenAI Lean formalizations](https://github.com/openai/math/tree/main/lean)
- [Symthaea cognition-to-Lean Phase 6 scoping](https://github.com/Luminous-Dynamics/symthaea/blob/main/docs/phase6-scoping/cognition-to-lean-bridge.md)
- [Symthaea issue #6220: diagonal-Gaussian entropy invariants](https://github.com/Luminous-Dynamics/symthaea/issues/6220)
