# DNA Whole-System Codec Tournament — v0.1

**Programme:** DNA-WHOLE-SYSTEM-CODEC-TOURNAMENT-001  
**Tracking:** [Symthaea #13772](https://github.com/Luminous-Dynamics/symthaea/issues/13772)  
**Parent:** [DNA-CODEC-SOFT-DECODING-001 #13729](https://github.com/Luminous-Dynamics/symthaea/issues/13729)  
**Physical channel owner:** [Symtropy #7567](https://github.com/Luminous-Dynamics/symtropy/issues/7567)

## 1. Purpose

Compare complete DNA-storage architectures under a common source, sequence-channel, copy/read budget, cost boundary and independent exact-file scorer. This extends the earlier ECC/soft-decision-only view to include codebook construction, biochemical sequence constraints, indexing, strand redundancy, motif removal and distributed recovery.

This is a source-review and reproduction protocol. No new DNA synthesis, sequencing or physical storage measurement was performed in this increment.

## 2. SOTA intake

The machine-readable register is `dna_whole_system_baselines_v0_1.json`.

### DNA-SaM

Wang et al., *Nature Communications*, 19 October 2026: https://doi.org/10.1038/s41467-026-78759-4

Observed source:
- Repository: https://github.com/DNA-intelligence/DNA-SaM
- Observed repository commit: `fd5aa708775782952938e08fcfe22d19bcc2cc8a`
- Publication archive: https://doi.org/10.5281/zenodo.21615403
- License: MIT.

Architecture:
- 15 bits mapped to 8 nt, before the other strand-level overheads;
- suffix-automaton search/replacement for forbidden sequence patterns;
- data/file hierarchical segmentation and address strands;
- file-attribute strands and XOR metadata recovery;
- single-base edit correction and XOR recovery;
- index and primer overhead;
- in-vitro synthesis/sequencing and an in-vivo plasmid passaging demonstration.

**Important source inconsistency:** the paper says 42,560 eligible 8-nt words, partitioned into 32,768 data words plus 9,792 replacement words. The repository README blob `6df38c3bf416705b518be0698f64f6584f865fc8` says 37,768 data words plus 9,792 replacement words, which totals 47,560. Do not compute an accepted payload rate from this inconsistent source. First diff the DOI archive against repository HEAD, inspect codebook generation, reproduce all counts and add a regression test.

The reported 20× coverage was sufficient for complete recovery for most tested datasets under the paper's particular experiment; it is not a universal coverage guarantee. The in-vivo 100-generation observation is specific to its plasmid/host and experimental protocol. It is not equivalent to 100 generations of ordinary non-biological archive retention. The paper's large image was segmented into units and describes one randomly selected unit for wet validation; do not imply that every segment of the complete large dataset was in-vivo tested.

### Siyuan Code

Li et al., *Nature Communications*, 27 August 2026: https://doi.org/10.1038/s41467-026-77024-y

Observed source:
- Repository: https://github.com/DNASiyuanCode/SiyuanCode
- Observed repository HEAD: `a037d04dbee4afe22c685a1e0803f599b516ae6f`
- Paper claims: 1.66 net bits/nt, 41 EB/g in-vitro, approximately 1 MB/s encode/decode and perfect retrieval with six copies per strand in the tested conditions.

Architecture:
- enumerate suitable strands and assign binary values bijectively;
- use Trie/automaton and dynamic programming/enumerative coding to avoid high-error patterns;
- add RS-based slice redundancy/checksums and indexes;
- use multiple copies and sequence constraints tuned to the target chemistry.

**Rights gate:** the observed repository has no root LICENSE file. Its README says the modified MESA simulator is public but unlicensed. The paper text is CC BY-NC-ND 4.0. Treat the public source as source-inspectable, not automatically licensed for redistribution, derivative bundling or training. Resolve source, simulator, sample and raw-read rights before any reuse that requires permission.

### Polus / SeqFormer

Ding et al., *Bioinformatics*, 3 August 2026: https://doi.org/10.1093/bioinformatics/btag563

Observed pins:
- Polus: `656b58008e196dbad42c9b09f8ec1bf350ee8614`
- SeqFormer: `f95fb2f866c13033874a505bf5effe7195e246e6`

The SeqFormer SHA is an observed repository-head pin, not yet proven to be the exact paper training artifact/model-weight version. Resolve weights, preprocessing, training split and environment before reproduction. Reported 38.9% fewer DNA Fountain sequencing reads and approximately 80% higher effective density are paper-reported in-silico results for that benchmark setup.

### DT4DDS and six-codec comparison

Gimpel et al., *Nature Communications*, 14 March 2026: https://doi.org/10.1038/s41467-026-70548-3

Pins:
- DT4DDS commit: `0928fd7f26709ce63fd4d0765c50db1be8af0f7d`
- Software archive: https://doi.org/10.5281/zenodo.17391326
- Notebooks: https://doi.org/10.5281/zenodo.17391328
- Data: ENA PRJEB90546

Codec set: DNA-Aeon, DNA Fountain, DNA-RS, Goldman, HEDGES and Yin–Yang. The paper reports 43 EB/g for material-deposition synthesis and 13 EB/g for electrochemical synthesis in two separate in-vitro demonstrations; 117 EB/g is a realistic-workflow in-silico feasibility result. The isolated 14% error and 65% sequence-loss limits must be reported as separate test-envelope results, not universal operational tolerances.

## 3. Fundamental benchmarking correction

Never compare one headline number without reconstructing its denominator.

Keep these distinct:
- codebook coding potential;
- net bits/nt after all per-strand metadata and parity;
- bits per oligo and number of oligos per payload;
- physical mass/density at declared synthesis and sample boundaries;
- molecular copies per strand;
- unique sequencing reads per strand;
- total read depth and sequencing cost;
- exact file recovery.

In particular, **molecule copy number is not sequencing coverage**. PCR amplification can produce many reads derived from one molecule, and duplicate read count cannot be used as independent physical-copy evidence. Record UMI or other molecule-provenance where available; otherwise physical molecule count remains unknown.

## 4. Whole-system benchmark contract

Every baseline must be run over the same file corpus and file-type mix, with the same declared target payload, sequence policy, primer/index budget, physical platform/chemistry where possible, copy-number/read-budget strata and byte-exact scorer.

Where platform, chemistry or budget differs, use separate strata and explicitly state non-comparability. Do not normalize incomparable in-vitro and in-silico results by an invented factor.

### Workload families

1. **Single-file exact reconstruction:** tiny, medium and large files; exact whole-file digest.
2. **Mixed-file corpus:** binary executables, compressed archives, text, images and random/high-entropy input to expose file-type assumptions.
3. **Low-copy recovery:** sweep molecule-copy count independently of read-depth.
4. **Sequencing-depth sweep:** sweep unique reads/strand, total reads, PCR duplicate fraction and abundance skew.
5. **Index and metadata damage:** lose/corrupt address, file-attribute, checksum, primer and parity/XOR strands.
6. **Biochemical constraint stress:** GC fraction, homopolymer, tandem repeats, promoter motifs and platform-specific forbidden patterns.
7. **Indel/substitution/dropout:** matched channels with substitutions, insertions, deletions, strand dropout, chimeras, contamination and clustered errors.
8. **Storage and replication:** declared environmental storage, extraction and repeated biological passaging where applicable; keep these as distinct modalities.

### Independent scorer

- Candidate code receives public sequence/read observations and permitted public descriptors only.
- Reference file bytes and expected digests remain scorer-only.
- Scorer verifies exact final bytes, object count, ordering, file length, manifest and all declared metadata relations.
- A syntactically valid sequence, successful inner decoder, high model confidence or non-empty output is not a file-recovery pass.
- All incorrect, truncated, reordered and duplicated output cases fail closed.

## 5. Required measures

### Coding efficiency
- net logical bits/nt after all indexing, primer, ECC, constraint replacement, XOR/parity and spare-codebook overhead;
- accepted useful sequence fraction and replacement frequency;
- strand count/length and effective payload bits per strand;
- physical density with exact mass denominator, synthesis yield and sample loading.

### Reliability and read economics
- exact-file recovery versus copy count and unique read coverage;
- synthesis yield, raw read depth, unique molecule count, abundance variance and PCR duplicate rate;
- substitution/indel/dropout/chimera/contamination by sequence context and platform;
- minimum read coverage at a preregistered exact-file recovery level;
- recovery cost/latency and uncorrectable fraction.

### Compute and economic boundary
- encoding/decoding/clustering runtime, throughput and peak memory;
- synthesis cost, sequencing cost and total cost per exactly restored byte, separately;
- energy and operator/laboratory time only when metered with a declared boundary;
- adaptive resequencing savings minus decision/latency overhead.

No composite score should hide a trade-off between payload density, recovery probability, synthesis cost and sequencing cost.

## 6. Research hypotheses worth testing

1. **Adaptive sampling / resequencing:** allocate the next read to the oligo whose measurement most reduces residual whole-file uncertainty per unit cost. Validate on held-out libraries and against uniform extra coverage.
2. **Synthesis-vs-sequencing fault attribution:** infer whether a missing or erroneous molecule is fixed at synthesis/storage or merely misread; do not purchase repeated reads for irreparable source defects.
3. **Joint constrained-channel compiler:** optimize codebook, forbidden motifs, strand length, local checksum, outer ECC, index and copy/read budget together.
4. **Hierarchical local recovery:** compare per-unit XOR and address/attribute recovery against global fountain or Reed–Solomon recovery under clustered unit loss.
5. **Independent molecular audit:** package original input, encoded strands, primer/index definitions, codebooks, version, raw reads, sequencing calibration, decoder, scorer and exact file digest in the Physical Recovery Capsule.

These are research hypotheses, not current achieved results.

## 7. Negative controls and metamorphic tests

- codebook count mismatch or duplicate codeword;
- inconsistent codeword-to-bit mapping;
- changed forbidden motif policy after encoding;
- lost index/address/attribute/XOR record;
- same reads duplicated to inflate apparent molecular coverage;
- abundance skew with high total reads but low unique molecules;
- sample-ID/primer collision and read cross-talk;
- single-base substitution, insertion, deletion and mixed edit;
- corrupted parity/checksum and valid but wrong sequence block;
- stale/mismatched model or base-quality calibration;
- file truncation, chunk reorder, chunk duplication and wrong original length;
- claimed in-vivo or retention status without matching experiment class.

## 8. Statistical protocol

- Split by independent synthesis batch, library, sample, storage cohort, reader/sequencer platform and time—not random reads from one pool.
- Use paired comparisons on identical physical cohorts where possible.
- Keep synthetic, in-silico, paper reproduction, in-vitro and in-vivo evidence labels separate.
- Publish sample size, uncertainty intervals, missing observations, failed runs and right-censoring.
- Never exclude unsuccessful synthesis, dropout, failed sequencing or decoder abstention from the denominator.
- A copy/read sweep is not a storage lifetime experiment.

## 9. Qualification ladder

- D0 — source, licensing, codebook and metric-definition audit.
- D1 — deterministic sequence/codec known-answer tests and independent scorer.
- D2 — reproduce published in-silico results with exact source and environment pins.
- D3 — reproduce published in-vitro data from authorized read/source artifacts.
- D4 — prospectively preregistered, matched lab campaign with independent batches and raw observations.
- D5 — repeat storage/extraction/migration and reader replacement under a defined archive service contract.
- D6 — biological/in-vivo passaging only as a separately governed, independently reviewed modality and scope.

## 10. Prospective targets

- 100% exact-file recovery for finite known-answer positive cases inside declared correction conditions.
- Zero false acceptance on frozen wrong-file, wrong-address, duplicate-molecule, corruption and stale-calibration controls.
- Full reconciliation of all input bits, strand bases, indexes, primers, checksums and redundancy.
- Reproduce the chosen primary paper baselines before attempting an optimization claim.
- Target >=10% lower combined synthesis+sequencing cost or unique read count at the same net bits/nt and exact-file recovery on independently held-out batches/platforms. Research target only.
- No claim of SOTA superiority from theoretical sequence capacity, in-silico channel models or a single successful in-vitro sample.

## 11. Evidence ceiling

DNA-SaM's >2^70 PB is a theoretical architecture-scale statement, not a demonstrated synthesized capacity. Its 100-generation passaging result is a specific biological experiment, not general glass/tape-equivalent retention. Siyuan's reported 41 EB/g and six-copy retrieval are paper results that must be reproduced with the same sample/copy/read protocol. No Symthaea DNA physical measurement or SOTA improvement is claimed by this protocol.
