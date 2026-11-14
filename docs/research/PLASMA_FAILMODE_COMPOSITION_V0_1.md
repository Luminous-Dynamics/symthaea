# Failure-mode composition evidence — 2026-11-14

Issue: #12097  
Branch: research/plasma-failure-composition-v0-1  
Dependency: #12074 (method ontology) and #12070 (equivalence design).

## Thesis

A single failure_mode_fraction is not sufficient to qualify the failure phenotype of an adhesive joint. Failure modes form a category distribution; a stable cohesive percentage can conceal increases in adhesive failure or other categories. The full specimen-level vector must remain available and be linked to its classifier, observer reliability, independent lot, analysis plan and result.

## Standards boundary

ASTM D5573-99R19 classifies failure modes in adhesively bonded FRP joints. Its development scope is SMC FRP. The standard says the classifications are based on visual observation of the failure surface without a microscope or other magnification, and explicitly does not judge whether any specific failure mode is acceptable. It can be used to describe failures arising from tests such as ASTM D3163 and D5868.

Source: https://store.astm.org/d5573-99r19.html

The image digest in this receipt is documentation/custody evidence; it does not replace the D5573 observation mode. The S0 record requires direct_unaided_visual. Fractional area measurement is a separate protocol; D5573 is the category taxonomy, not a universal area-fraction metrology specification.

## Implementation

- Rust module: xtask/src/plasma_failure_composition.rs
- Fixture: data/research/plasma_failure_composition_fixture_v0_1.json
- Command: cargo run -p xtask -- plasma-failure-composition-audit

The receipt preserves:
- a versioned, ordered category ontology;
- one complete closed vector per specimen;
- image digest, cluster identity, observer references, blinding state and method observation mode;
- exact D5573 method/edition/material/geometry/endpoint/claim scope;
- separate fraction-measurement protocol and observer-reliability receipt;
- baseline/candidate independent manufacturing cohorts;
- specimen and cluster non-overlap or exact paired-cluster identity;
- explicit unclassifiable specimens and unresolved observer disagreements;
- preregistered zero-handling, category margins, joint-distance margin and analysis method;
- full vector-manifest digest and per-category plus joint-distance result intervals.

Area-fraction and specimen-count-fraction are separate bases. A specimen-count composition must be one-hot. Continuous area fractions are allowed only under the separately declared area-measurement protocol. The S0 closure tolerance cap of 0.05 is a prototype guardrail, not a standard requirement.

## Paradigm shift

Replace a scalar score with:

surface state → adhesive constitutive response → joint strength/fracture response → full failure-mode distribution → aging/competing failure → use-case-specific acceptance policy.

The arrows are evidence dependencies, not automatic implications. D5573 classification does not create an acceptance decision. A valid study can still report COMPOSITIONAL_EQUIVALENCE_NOT_ESTABLISHED; design integrity and outcome are separate.

## Frozen fixture

The exact branch fixture contains 40 synthetic known-answer cases:
- 4 PASS;
- 15 BLOCKED;
- 21 REJECTED.

Latest source/fixture readback and independent JavaScript replay: 40/40 expected dispositions and outcome labels matched.

Cases include closed-composition failures, category/ontology/order mismatches, measurement-basis confusion, count-fraction non-one-hot vectors, method/edition/material/claim mismatch, pseudo-replication, shared clusters, paired cluster mismatch, unclassifiable specimens, image and observer-reliability evidence gaps, post-outcome freeze, zero/pseudocount changes, digest mismatch, per-category interval omission, scalar masking, non-equivalence result, and magnified observation used for D5573.

## Research and benchmark programme

Compare:
1. single scalar failure-mode fraction;
2. category counts without independent lot lineage;
3. specimen-level vectors without observer reliability;
4. complete vectors with observer evidence, cluster-aware design and preregistered multivariate analysis.

Report full-vector completeness, closure error, observer agreement, unclassifiable fraction, independent-lot count, per-category interval coverage, joint compositional distance, false equivalence, abstention and review burden separately.

Prospective contract goals:
- zero PASS for incomplete/renormalized vectors;
- zero count-based pseudo-replication accepted as independent lots;
- 100% detection of method/ontology/specimen/cohort mismatches;
- 100% detection of post-outcome category, zero-policy or margin changes;
- zero scalar summaries accepted instead of full vectors;
- deterministic replay of the frozen vector contract.

These are regression goals, not physical performance results.

## Next frontier

ISO 10365:2022 provides a general classification of main failure patterns for bonded assemblies regardless of adherend/adhesive. It can complement the SMC-specific D5573 route. A separate follow-on should register method-specific ontologies, not merge the categories into one unqualified taxonomy.

Official ISO scope: https://www.iso.org/standard/80800.html

NDT/POD and time-dependent plasma surface recovery are separate workstreams:
- #12099 — activation recovery, storage and time-to-bond;
- #12100 — defect-specific NDT/POD for adhesive joints.

## Claim ceiling

S0 PASS means only the vector, method, observer, independent-unit and plan/result references are structurally consistent. Eight Rust tests are authored but not executed. Rust formatting, compilation, Clippy and exact-head hosted CI are not verified. No physical failure mechanism, acceptability, bond strength, service life, safety, product qualification or release authority is established.
