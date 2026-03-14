# Cover Letter - PLoS Computational Biology Submission

**Date**: February 23, 2026

**To**: Editorial Board
PLoS Computational Biology

**Re**: Submission of Research Article - "Hyperdimensional Active Inference: Free Energy Principle in Vector Symbolic Architectures"

---

Dear Editors,

We are pleased to submit our manuscript, **"Hyperdimensional Active Inference: Free Energy Principle in Vector Symbolic Architectures"**, for consideration as a Research Article in *PLoS Computational Biology*.

## Summary

This paper presents Hyperdimensional Active Inference (HAI), the first integration of the Free Energy Principle with Hyperdimensional Computing (HDC). By reformulating variational free energy in 16,384-dimensional hypervector space, we achieve O(d) complexity where existing active inference implementations require O(n²)–O(n³) matrix operations. The work introduces precision-weighted binding — a novel HDC operation for confidence-modulated feature composition — and derives eight interpretable motor commands directly from expected free energy minimization.

## Key Contributions

1. **Computational efficiency**: 7.9× total speedup over pymdp (1.9× belief inference, 15.8× action selection) on standard benchmarks (T-Maze, Grid World), with success rates improving from 10–16% to 88–100%.

2. **Cross-domain validation**: 17 benchmarks spanning neuroscience (*C. elegans* connectome analysis, EEG seizure detection), ethics (92.9% on ETHICS benchmark via compositional moral algebra — no pretrained language models), signal processing (94.5% speaker identification on synthetic MFCC features), and federated learning (Byzantine tolerance validated to 34%).

3. **Mathematical foundations**: 14 rigorous mathematics modules grounding HDC in established theory (information theory, IIT 3.0, Riemannian geometry, tensor algebra, causal calculus).

4. **Open-source implementation**: Complete Rust codebase (~343K LOC, 6,575+ tests across 30 workspace crates), all benchmarks reproducible via standalone examples.

## Relevance to PLoS Computational Biology

This work addresses a core challenge in computational neuroscience: making the Free Energy Principle computationally tractable at scale. The biological validation (*C. elegans* connectome, EEG analysis) and the brain-inspired HDC substrate align directly with the journal's scope. The open-source codebase and reproducible benchmarks align with PLoS's commitment to open science.

## Novelty

While Liquid AI (LFM2/LFM2.5) validates liquid neural networks commercially and FedHDC demonstrates federated HDC, no existing system integrates active inference with hyperdimensional computing. Our HDC moral algebra achieves 92.9% on ETHICS using only vector operations — a fundamentally different approach from large language models.

## Suggested Reviewers

We respectfully suggest the following expert reviewers (detailed qualifications in attached file):

1. **Dr. Larissa Albantakis** (University of Wisconsin-Madison) — IIT 4.0 co-author, integrated information expertise
2. **Dr. Anil Seth** (University of Sussex) — Consciousness measures, practical Φ approximations
3. **Dr. Olaf Sporns** (Indiana University) — Brain network topology, connectome analysis
4. **Dr. William Marshall** (Bard College) — IIT theory, network information theory
5. **Dr. Pentti Kanerva** (UC Berkeley / SETI Institute) — Hyperdimensional computing pioneer
6. **Dr. Rafael Yuste** (Columbia University) — Neural networks, consciousness neuroscience
7. **Dr. Danielle Bassett** (University of Pennsylvania) — Network neuroscience, brain dynamics

We request that the following individuals **not review** our manuscript due to potential conflicts of interest:

- Dr. Giulio Tononi (personal communication regarding unpublished IIT extensions)
- Dr. Christof Koch (co-founder of Allen Institute, institutional conflicts)

## Reproducibility and Data Availability

All benchmarks are reproducible via standalone Rust examples with deterministic seeds. The complete codebase, raw timing data, and supplementary materials are available at https://github.com/Luminous-Dynamics/symthaea. External datasets used (Sleep-EDF, LibriSpeech, ETHICS, Social Chemistry 101) are publicly available under open licenses.

## Manuscript Details

- **Abstract**: 141 words
- **Figures**: 8 (PNG + PDF, 300 DPI)
- **References**: 48 citations (Vancouver numbered style via plos2015.bst)
- **Supplementary Materials**: Benchmark reproduction commands, implementation details

## Author Contributions and AI Disclosure

**Primary Author**: Tristan Stoltz (conceived study, designed experiments, implemented framework, analyzed data, wrote manuscript, takes full scientific responsibility)

**AI Assistance**: Claude Code (Anthropic) contributed to manuscript drafting, figure generation, statistical analysis, and literature review under human direction and supervision. All AI-generated content was critically reviewed, edited, and validated by the human author.

## Competing Interests

The authors declare no competing interests. This work received no external funding.

---

We thank you for your consideration and look forward to your response.

Sincerely,

**Tristan Stoltz**
Founder & Principal Investigator
Luminous Dynamics
Richardson, TX, USA
Email: tristan.stoltz@evolvingresonantcocreationism.com

---

**Attachments**:
1. Manuscript PDF (hai_paper.pdf)
2. Figures (8 files, PNG + PDF formats)
3. Supplementary Materials
4. Suggested Reviewers (detailed qualifications)
