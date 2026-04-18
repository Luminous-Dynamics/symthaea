# Psych-Bench BRM Submission Checklist

## Manuscript
- [x] Title page with author name, affiliation, email
- [x] Abstract (<250 words for BRM)
- [x] Keywords
- [x] All sections complete (Introduction through Conclusion)
- [x] CRediT Author Contributions statement
- [x] Declaration of Competing Interests
- [x] Open Practices Statement (data + code availability)
- [ ] Word count statement in cover letter (target: 8,000-12,000)
- [ ] ORCID for corresponding author
- [ ] Suggested reviewers (3-5 names with expertise in computational cognitive science, HDC, or benchmark design)

## Figures & Tables
- [x] Figure 1: Architecture diagram
- [x] Figure 2: Encoding pipeline
- [x] Figure 3: Neuromod framework
- [x] Figure 4: 27-domain signature bar chart
- [x] Figure 5: Forest plot (normative z-scores)
- [x] Figure 6: 8-domain radar chart
- [x] Figure 7: Ablation grouped bars
- [x] Figure 8: SAT curves
- [x] Table 1: Benchmark inventory summary
- [x] Table 2: Psychometric validation (ICC)
- [x] Table 3: Cross-domain correlations
- [x] Table 4: Consciousness indicators
- [x] Table 5: Framework comparison

## Supplement
- [x] Table S1: Complete 141-benchmark inventory
- [x] Section S2: Cross-seed stability (5 seeds)
- [x] Section S3: ICC reliability distribution (Figure S1)
- [x] Section S4: Neuromod dissociation details + API usage

## Data
- [x] CSV files regenerated via `cargo run -p symthaea-psych-bench --example psych_bench_paper_data`
- [x] CSVs copied to `papers/psych-bench/arxiv-submission/`
- [ ] Verify commit hash in CSV headers matches submission version

## Pre-submission
- [ ] Compile paper with pdflatex (no errors)
- [ ] Compile supplement with pdflatex (no errors)
- [ ] Check all citations resolve (no "?" in PDF)
- [ ] Verify figure/table numbering is sequential
- [ ] Check cross-references (\ref) all resolve
- [ ] Proofread abstract one final time
- [ ] Write cover letter

## Submission
- [ ] Upload manuscript PDF
- [ ] Upload supplementary PDF
- [ ] Upload CSV data files as supplementary
- [ ] Enter metadata (title, abstract, keywords)
- [ ] Select article type
- [ ] Submit
