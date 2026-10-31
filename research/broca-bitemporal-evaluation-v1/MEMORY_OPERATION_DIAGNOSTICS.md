# Broca memory operation diagnostics

**Protocol:** BROCA-MEMORY-DIAGNOSTICS-V1  
**Status:** typed examiner states, multi-label defect taxonomy and report digest authored; tests not compiled in this environment.

## Why aggregate QA accuracy is insufficient

A wrong final answer can be caused by three different failures: the fact was never stored, it was stored but never retrieved, or the answerer received usable evidence and still failed. Treating those as one “memory accuracy” number leads to the wrong engineering intervention.

EvalMem (Findings EMNLP 2026) formalizes three independent examiners: Encoding (is gold evidence present in the store?), Retrieval (did the native retriever return usable gold evidence?) and Generation (can the answerer use oracle evidence correctly?). Its reported default LoCoMo defect rates were 22.1% retrieval, 7.7% encoding and 6.5% generation; a recall-first, source-aware probe raised evidence recall 70.2% to 95.6%. Those values motivate stage attribution; they are not Broca results.

## Broca's typed examiner contract

`MemoryOperationObservation` carries one state per layer:

| Layer | States | Defect codes |
|---|---|---|
| Encoding | gold evidence present, ambiguous, wrong, missing, not evaluated | `encoding_missing`, `encoding_ambiguous`, `encoding_wrong` |
| Retrieval | usable gold hit, late hit, noisy hit, missed, negative-evidence intrusion, not evaluated | `retrieval_failure`, `retrieval_late`, `retrieval_noise`, `noise_induced_retrieval` |
| Generation | correct and supported, hallucinated without support, ignored oracle evidence, reasoning failure with evidence, false abstention, not evaluated | `generation_hallucination`, `generation_faithfulness`, `generation_reasoning_failure`, `false_abstention` |

Defects are multi-label: one question can have an encoding defect plus a generation hallucination. Retrieval miss is only attributed when the gold fact is known to be present in the store; it is not blamed when the encoder never stored the fact. Negative evidence intrusion is a separate retrieval defect. Generation faithfulness (ignoring oracle evidence) is distinct from reasoning failure while using evidence.

The report is bound to a benchmark ID/revision, evaluation-protocol digest and sorted observation digest. It records per-defect counts and per-query rates. The report is a reproducibility object, not a substitute for the underlying examiner observations. `verify_against` recomputes the report from those observations.

## Gold-label requirements

- Encoding: inspect the underlying memory export/store with an independent source locator; do not infer absence from retrieval miss.
- Retrieval: compare the native result set to annotated positive and negative evidence; record rank, context budget and signal/noise ratio in the upstream runner.
- Generation: run paired answerer trials with native retrieved context and oracle gold context; distinguish ignoring evidence from faulty inference.
- Judge quality: use deterministic checks wherever possible and blinded human adjudication on a stratified sample. LLM-as-judge is secondary and must be calibrated.

## Metrics to publish

- store coverage and encoding missing/ambiguous/wrong rates;
- strict evidence recall-all@k, Recall@k, MRR/nDCG, late retrieval rate, retrieval noise and negative-intrusion rate;
- answer accuracy with native context and oracle context;
- generation faithfulness, reasoning failure, unsupported assertion, hallucination and false abstention rates;
- complete claim-proof coverage, lifecycle and speaker-attribution errors;
- per-stage p50/p95/p99 latency and full write/read/maintenance token cost.

Do not collapse Recall@K, answer accuracy, FAMA, event extraction recall and action success into one score. A system can have excellent evidence recall but poor answer grounding, or high answer accuracy while silently using stale/unauthorized information.

## New Broca use

Use the diagnostics to decide what to repair:

- encoding failures: ingestion, source parsing, entity/claim extraction or memory updates;
- retrieval failures: candidate enumeration, lexical/semantic views, rank fusion, event filtering and evidence packet construction;
- generation failures: reasoning, uncertainty calibration, conflict explanation, citation/proof use and abstention;
- lifecycle defects: temporal update/supersession/retraction policy;
- speaker defects: conversation role binding and multi-party belief attribution.

