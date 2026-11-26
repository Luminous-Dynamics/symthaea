# DNA Codec and Soft-Decision Benchmark — v0.1

**Programme:** DNA-CODEC-SOFT-DECODING-001  
**Tracking:** https://github.com/Luminous-Dynamics/symthaea/issues/13729  
**Biochemical observation owner:** https://github.com/Luminous-Dynamics/symtropy/issues/7567

## 1. New 2026 benchmark anchors

### Gimpel et al., Nature Communications, 14 March 2026

“Comparison of state-of-the-art error-correction coding for sequence-based DNA data storage,” Nature Communications 17, 3963 (2026): https://doi.org/10.1038/s41467-026-70548-3

The study compares six representative codecs using standardized in-silico and in-vitro settings, real sequencing-read nonidealities, read clustering and a declared benchmark workflow. It reports demonstrations at 43 EB/g for material-deposition synthesis and 13 EB/g for electrochemical synthesis. It is now a required baseline rather than an optional related citation.

The six codec families are:
- DNA-Aeon: arithmetic inner coding + fountain outer coding.
- DNA Fountain: Reed–Solomon inner coding + fountain outer coding.
- DNA-RS: Reed–Solomon inner/outer.
- Goldman: parity + repetition.
- HEDGES: convolutional inner coding + Reed–Solomon outer coding.
- Yin–Yang: constrained bit-to-base encoding without ECC in the evaluated implementation.

The source paper reports isolated-code scenario maxima such as 14% error and 65% sequence loss. These are not general physical channel guarantees. We must reproduce the authors' exact scenario definitions, payload sizes, coverage and sequence constraints before interpreting them.

Benchmark artifacts:
- dt4dds benchmark: https://github.com/fml-ethz/dt4dds-benchmark
- notebooks: https://github.com/fml-ethz/dt4dds-benchmark_notebooks
- archived software version: https://doi.org/10.5281/zenodo.17391326
- archived notebooks: https://doi.org/10.5281/zenodo.17391328
- experiment reads: ENA PRJEB90546

Observed repository refs at research intake:
- dt4dds-benchmark main latest commit observed 2025-08-28: 0928fd7f26709ce63fd4d0765c50db1be8af0f7d.
- The article's Zenodo archive is the publication-associated reference; do not assume GitHub main is byte-identical to the paper's archived release.

### Ding et al., Bioinformatics, 3 August 2026 — Polus

“Polus: a context-aware enhancement framework for DNA storage via transformer-based soft-decision decoding,” Bioinformatics 42(8), btag563 (2026): https://doi.org/10.1093/bioinformatics/btag563

Polus combines multi-read sequence context and Phred quality inputs to estimate per-base confidence, then ranks low-confidence positions as candidate erasures. The hard-decision decoder runs first; soft-decision erasure trials are triggered after decoder/CRC failure. The article reports in-silico results: 38.9% lower DNA Fountain coverage requirement for the evaluated full-recovery scenario, approximately 80% higher effective physical density in that setup, and recovery of certain YYC indel cases.

Pinned Polus source at intake:
- repository: https://github.com/dinglulu/Polus
- observed latest commit: 656b58008e196dbad42c9b09f8ec1bf350ee8614 (15 July 2026)
- SeqFormer repository: https://github.com/dinglulu/SeqFormer
- SeqFormer README blob captured at intake, but repository commit pin remains pending.

The paper's gains are in silico; do not present them as in-vitro measurements or generalize them across synthesis/sequencing platforms. Its SeqFormer quality calibration is platform-specific; cross-platform transfer must be tested, not assumed.

## 2. Research thesis

A useful next step is not inventing a seventh hand-designed codec before reproducing the best existing ones. It is composing:
1. multi-read clustering and consensus;
2. calibrated per-base posterior error probabilities;
3. sequence constraints;
4. inner/outer error correction;
5. adaptive erasure and resequencing;
6. whole-file exact verification;
7. cost/coverage selection.

The learned model may propose confidence-ranked erasures. The pinned decoder and an independent payload scorer decide whether recovery actually succeeded. The model must never author its own labels or acceptance authority.

## 3. Baseline tournament

Separate the stages instead of comparing opaque end-to-end labels:
- consensus: iterative majority, POA, BSAlign, BMALA, DivBMA, SeqFormer;
- hard-decision codec: DNA-Aeon, DNA Fountain, DNA-RS, Goldman, HEDGES, Yin–Yang;
- soft-decision enhancement: Polus/SeqFormer-guided erasure ranking versus hard-decision-only decoding;
- clustering: unique-read deduplication, alignment clustering and consensus-capable clustering;
- adaptive read allocation: uniform deeper sequencing versus confidence-targeted resequencing.

Pin source commit, codec parameters, read clustering method, dataset accession/version, platform and model weights. The actual codecs may run in isolated Python/C++ environments; Symthaea's trusted scoring and evidence receipt remain independent. No old Python/PyTorch dependency is added to the production Rust runtime.

## 4. Dataset and split protocol

Use the publication-associated DT4DDS benchmark and ENA PRJEB90546 as the first reproduction target. Keep the original archived versions and licenses. Additional datasets may be added only with source accession, platform, chemistry, batch and read quality metadata.

Never randomly split reads from the same oligo between train and test. Split by source dataset, synthesis batch/library and sequencing run. Illumina and ONT are separate cohorts. A model trained on Illumina must not be treated as calibrated for ONT.

Reference sequences and original file bytes are scorer-only labels. Inference-time reconstruction receives reads and quality values, not expected sequences.

## 5. Metrics and acceptance

Report per-file and per-library:
- byte-exact full-file recovery and independent digest match;
- bit recovery and block/strand recovery as diagnostic metrics, not substitutes for full-file success;
- coverage/depth at first exact full recovery;
- physical density incorporating strand length, logical code rate and read coverage;
- GC deviation, max homopolymer and motif constraints;
- substitution, insertion, deletion, dropout and duplicate/abundance distributions;
- per-base calibration (Brier, ECE/reliability diagrams when sample size supports them);
- precision/recall of flagged errors, fraction of bases erased and erasure trials to success;
- adaptive resequencing overhead relative to uniform deepening;
- encode/consensus/decode runtime, memory, synthesis cost and sequencing cost separately.

A benchmark pass requires the exact original file bytes and independent expected digest. A 100% bit recovery diagnostic on a different partial file is not an archival pass.

## 6. Luminous hypotheses and stretch target

- Use calibrated soft-erasure ranking after hard-decision failure, but require platform-held-out calibration.
- Attribute likely sequencing errors versus synthesis-derived fixed errors; do not waste reads on errors that additional sequencing cannot resolve.
- Allocate sequencing adaptively to oligos with highest expected reduction in residual file uncertainty per marginal cost.
- Optimize code rate, sequence constraints, oligo count/length, read coverage and synthesis/sequencing cost jointly.
- Preserve provenance and separate learned confidence from decode proof.

Stretch target: after reproducing Polus, seek at least 10% fewer reads or lower sequencing cost than Polus on a genuinely held-out library/platform cohort, while preserving exact whole-file recovery, identical constraints and no synthesis-cost regression. This is a target only, not an achieved result.

## 7. Small Rust reference scope

This branch adds a zero-dependency Rust sequence-channel utility with:
- deterministic global edit-distance alignment and a documented tie-break for substitution/insertion/deletion counts;
- GC and maximum-homopolymer checks;
- confidence-ranked soft-erasure candidates from externally calibrated probabilities;
- progressive erasure-set generation;
- read-record versus molecule multiplicity accounting;
- byte-exact payload equality.

It does not implement DNA-Aeon, DNA Fountain, HEDGES, Polus/SeqFormer, synthesis, basecalling, clustering or physical sequencing. The reference exists to test metrics and acceptance contracts. Its inputs are synthetic and not physical error probabilities.

## 8. Run commands

cargo fmt --manifest-path research/physical-storage/dna-channel-benchmark/Cargo.toml -- --check

cargo clippy --manifest-path research/physical-storage/dna-channel-benchmark/Cargo.toml --all-targets -- -D warnings

cargo test --manifest-path research/physical-storage/dna-channel-benchmark/Cargo.toml --all-targets

cargo run --manifest-path research/physical-storage/dna-channel-benchmark/Cargo.toml

python3 research/physical-storage/dna-channel-benchmark/oracle/verify_dna_metrics.py

No Rust toolchain pass is claimed until exact-head commands run. The independent Python oracle validates only deterministic metrics and ranking logic.
