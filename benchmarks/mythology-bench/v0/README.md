# MythologyBench v0 — Egypt-first development fixtures

**Status:** Development fixtures and evaluation design only; not a published benchmark or evidence of model performance.

Mythology deserves a first-class module in the shared knowledge fabric, but mythology is **not simply a subset of esotericism**. This seed pack is linked to [EsotericaBench v0](../../esoterica-bench/v0/case.schema.json). Shared claim/provenance contracts are proposed; separate task families keep mythological reasoning results from being conflated with esotericism scores.

## Contents

- [`case.schema.json`](case.schema.json): versioned evaluation case contract.
- [Shared source catalog](../../source-catalog/v0/README.md): source IDs, source-family grouping, split governance and rights-review status. Every case source reference must resolve to this metadata catalog.
- [`cases.jsonl`](cases.jsonl): **26** development cases (Egypt-first + synthetic transfer/metamorphic controls).
- [`narrative-graph.schema.json`](narrative-graph.schema.json): a source-grounded narrative-graph shape contract.
- [`narrative-graph-fixtures.jsonl`](narrative-graph-fixtures.jsonl): four explicitly synthetic graph fixtures.
- [`TRANSFER_PROTOCOL.md`](TRANSFER_PROTOCOL.md): cross-source splits, transfer metrics, metamorphic tests, baselines and nonclaims.
- [`metamorphic-pair.schema.json`](metamorphic-pair.schema.json) and [`metamorphic-pairs.jsonl`](metamorphic-pairs.jsonl): ten synthetic baseline/intervention pairs, covering context/source changes, evidence removal, relation mutation, variant isolation, transfer, counterevidence, access revocation, aliases and prompt injection.
- [`STRUCTURAL_TRANSFER_SCORE.md`](STRUCTURAL_TRANSFER_SCORE.md): proposed evidence-bounded structural-transfer scorecard, controls, SOTA comparison protocol and provisional targets.
- [Preregistered comparison protocol](../../benchmark-runs/v0/README.md): baseline ladder, shared evidence/compute budgets, held-out split requirements, source-family confidence intervals and non-compensatory release gates.
- [Multi-answer annotation contract](../../knowledge-annotation/v0/README.md): blinded claim-level review, alternative interpretations, abstention and disagreement preservation.
- [Validator harness](../../../tools/knowledge-benchmark-harness/): offline Rust JSON Schema validation, source-reference checks, paired intervention contracts, and digest-bound reports.

Schema validation checks data structure only. A separate domain validator must verify graph endpoint/source references, relation-specific evidence, interval consistency, permissions, and variant isolation.

## Why begin with ancient Egypt?

Egyptian religious literature provides a rich stress test for multiple local creation traditions, context-dependent divine names and attributes, temporal variation in textual corpora, relationships between myth/ritual/kingship/funerary practice/social order, and the distinction between source evidence and modern scholarly interpretation.

Egypt must not be presented as a single, uniform worldview spanning all periods and regions. The civilization lasted thousands of years; local traditions changed, interacted, and sometimes accommodated apparently different accounts side by side.

## Sources used to formulate development cases

These links are scholarly context, not grants of full-text reuse rights:
- [The Met — Divine Egypt](https://www.metmuseum.org/exhibitions/divine-egypt/inside-the-exhibition): local creation traditions and flexible divine identities.
- [UCL Digital Egypt — Religious texts](https://www.ucl.ac.uk/museums-static/digitalegypt/literature/religious/index.html): Pyramid Texts, Coffin Texts and funerary literature.
- [UCL Digital Egypt — Book of the Dead](https://www.ucl.ac.uk/museums-static/digitalegypt/literature/religious/bdquestions.html): variable formula selections and manuscript context.
- [Fitzwilliam Museum — Egyptian funerary literature](https://book-of-the-dead.fitzmuseum.cam.ac.uk/explore/book-of-the-dead/funerary-literature): textual succession and variation.
- [Cambridge History of Religions in the Ancient World — Egyptian religion](https://www.cambridge.org/core/books/abs/cambridge-history-of-religions-in-the-ancient-world/egyptian-religion/C83766991A320AEE4BFEC436D8F1FED5): interpretive limits and varied evidence.
- [Oxford Research Encyclopedia — Ancient Egyptian Religion](https://academic.oup.com/edited-volume/62249/chapter-abstract/551373292): cycles and their scholarly context.
- [Historical/legal discussion of maat](https://www.scielo.org.za/scielo.php?lng=en&nrm=iso&pid=S1021-545X2018000100004&script=sci_arttext&tlng=en): historical discussion of ethics, law, truth and order.

## How this differs from generic culture and narrative benchmarks

- [CulturalBench (ACL 2025)](https://aclanthology.org/2025.acl-long.1247/) provides a broad, human-verified cultural knowledge evaluation across 45 regions.
- [Nunchi-Bench (Findings ACL 2025)](https://aclanthology.org/2025.findings-acl.794/) tests culture-specific reasoning and situation interpretation for Korean superstition.
- [EventRelBench (Findings EMNLP 2025)](https://aclanthology.org/2025.findings-emnlp.482/) tests event relation extraction including coreference, temporal and causal relations.
- [ChronoSense (ACL 2025)](https://aclanthology.org/2025.acl-short.46/) and [ETRQA (Findings ACL 2025)](https://aclanthology.org/2025.findings-acl.1198/) test time-interval, event-temporal and compound reasoning.

MythologyBench should complement rather than duplicate them: its focus is source-witness attribution, versioned narratives, tradition/time-conditioned meaning, variant isolation, transmission claims, and transfer to unseen source families without cultural projection. Recent work adds two important controls to the roadmap: [BRINK (EACL 2026)](https://aclanthology.org/2026.eacl-long.114/) warns that graph QA may reward direct-triple lookup over reasoning under incomplete knowledge, while [KGHaluBench (Findings EACL 2026)](https://aclanthology.org/2026.findings-eacl.206/) motivates dynamically generated, graph-verified negative controls. Future revisions should pair multi-hop completeness with critical-edge removal and must abstain when a missing edge breaks the only justified path. These are proposed tracks, not capabilities measured by the current ten synthetic pairs.

## Acceptance boundary

The newer cases add local-to-universal overgeneralization, graph similarity versus identity, multi-answer grading, and missing-domain abstention. The proposed structural-competence track is described in [the transfer protocol](TRANSFER_PROTOCOL.md) and [Structural Transfer Scorecard](STRUCTURAL_TRANSFER_SCORE.md), drawing cautiously on ICML 2026 meta-cultural structure research and ACL 2026 FrameNet-Cultures. Ten synthetic intervention pairs provide contract tests only, not historical benchmark cases or results.

All cases are `development`; synthetic controls are not historical evidence. The shared source catalog currently has 13 metadata-only records in 10 source families, all marked development, rights unknown, and not reviewed. Source links are not an invitation for automated crawling. Rights remain `unknown` until reviewed. Use [`scripts/validate-knowledge-benchmarks.sh`](../../../scripts/validate-knowledge-benchmarks.sh) to run the focused harness over ten input groups: Esoterica cases, claim controls, Mythology cases, metamorphic pairs, source catalog, synthetic source pack, annotation fixtures, comparison plans, paired-result labels, and narrative graphs. The script also runs `score-annotation-agreement` on synthetic reviewer controls and `score-paired-results` on fabricated paired observations; both outputs must remain labelled qualification-only, not model-performance or expert-reliability results. The committed `Cargo.lock` is enforced with Cargo's `--locked` mode, the harness binds its SHA-256 into each report, refuses remote schema references, and never crawls scholarly source URLs.

No score should be claimed until this harness has been built and tested on the exact head, expert review and source-disjoint split manifests exist. Structural validation is not a model benchmark pass. A queued, skipped or absent runner is not a pass.
