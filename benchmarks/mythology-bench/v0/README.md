# MythologyBench v0 — Egypt-first development fixtures

**Status:** Development fixtures and evaluation design only; not a published benchmark or evidence of model performance.

Mythology deserves a first-class module in the shared knowledge fabric, but mythology is **not simply a subset of esotericism**. This seed pack shares the versioned case contract proposed in [EsotericaBench v0](../../esoterica-bench/v0/case.schema.json). Reuse of the common claim record is planned; this directory does not authorize bulk ingestion or imply that the parent draft PR has merged.

## Why begin with ancient Egypt?

Egyptian religious literature provides a rich stress test for multiple local creation traditions, context-dependent divine names and attributes, temporal variation in textual corpora, relationships between myth/ritual/kingship/funerary practice/social order, and the distinction between source evidence and modern scholarly interpretation.

Egypt must not be presented as a single, uniform worldview spanning all periods and regions. The civilization lasted thousands of years; local traditions changed, interacted, and sometimes accommodated apparently different accounts side by side.

## Sources used to formulate the cases

These links are scholarly context, not grants of full-text reuse rights:
- [The Met — Divine Egypt](https://www.metmuseum.org/exhibitions/divine-egypt/inside-the-exhibition): local creation traditions and flexible divine identities.
- [UCL Digital Egypt — Religious texts](https://www.ucl.ac.uk/museums-static/digitalegypt/literature/religious/index.html): Pyramid Texts, Coffin Texts and funerary literature.
- [UCL Digital Egypt — Book of the Dead](https://www.ucl.ac.uk/museums-static/digitalegypt/literature/religious/bdquestions.html): variable selections of formulae and manuscript context.
- [Fitzwilliam Museum — Egyptian funerary literature](https://book-of-the-dead.fitzmuseum.cam.ac.uk/explore/book-of-the-dead/funerary-literature): textual succession and variation.
- [Cambridge History of Religions in the Ancient World — Egyptian religion](https://www.cambridge.org/core/books/abs/cambridge-history-of-religions-in-the-ancient-world/egyptian-religion/C83766991A320AEE4BFEC436D8F1FED5): interpretive limits and varied evidence.
- [Oxford Research Encyclopedia — Ancient Egyptian Religion](https://academic.oup.com/edited-volume/62249/chapter-abstract/551373292): religious cycles and their contexts.
- [Historical/legal discussion of maat](https://www.scielo.org.za/scielo.php?lng=en&nrm=iso&pid=S1021-545X2018000100004&script=sci_arttext&tlng=en): historical discussion of law, ethics, truth and order.

## Evaluation targets

Build held-out tests for variant retention and source attribution; local/temporal context for deities and symbols; lineage detection that separates resemblance from documented contact; narrative structure and causal order; emic (tradition-internal) versus etic (scholarly/external) interpretations; respectful handling of sacred/community-controlled knowledge; calibrated abstention where dates or attributions are unsupported; and resistance to myth-to-science overclaiming.

Compare BM25, dense retrieval, conventional RAG, graph retrieval, and the evidence-aware approach under equal evidence and compute budgets. Use domain-expert adjudication where interpretation is not mechanically decidable. Report by tradition, language, era, source genre, and answerability; no aggregate should conceal systematic misrepresentation.

All cases are `development`. Synthetic controls are not historical evidence. External links must not be fetched or ingested automatically by a benchmark runner. Rights status remains `unknown` until reviewed. Schema validation and fixture checks demonstrate data-contract integrity only—not historical truth, community authorization, supernatural efficacy, or model quality.
