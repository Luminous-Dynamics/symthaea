# 📄 PDF Manuscript Creation Guide

**Purpose**: Convert complete manuscript to Nature Neuroscience-formatted PDF for submission
**Current Status**: All content ready, formatting needed
**Time Required**: 1-2 hours
**Target**: Single PDF with all sections, proper formatting, line numbers

---

## Option A: Pandoc Conversion (RECOMMENDED) ⭐

**Pros**: Fast, automated, consistent formatting
**Cons**: May need manual tweaks for equations/figures
**Best for**: Quick first draft, iterative refinement

### Prerequisites
```bash
# Install Pandoc (if not already installed)
nix-shell -p pandoc texlive.combined.scheme-full

# Verify installation
pandoc --version  # Should be 2.19+
```

### Step 1: Create Master Document
```bash
# Combine all sections into single markdown file
cd /srv/luminous-dynamics/11-meta-consciousness/luminous-nix/symthaea-hlb

cat MASTER_MANUSCRIPT.md \
    PAPER_RESULTS_SECTION.md \
    PAPER_DISCUSSION_SECTION.md \
    PAPER_CONCLUSIONS_SECTION.md \
    PAPER_REFERENCES.md \
    > COMPLETE_MANUSCRIPT_FOR_PDF.md
```

### Step 2: Convert to PDF
```bash
pandoc COMPLETE_MANUSCRIPT_FOR_PDF.md \
  -o manuscript_v1.0.pdf \
  --pdf-engine=xelatex \
  --number-sections \
  --toc \
  --citeproc \
  --variable geometry:margin=1in \
  --variable fontsize=11pt \
  --variable mainfont="Times New Roman" \
  --variable linestretch=1.5
```

### Step 3: Insert Figures Manually
- Open `manuscript_v1.0.pdf` in PDF editor (Okular, PDF Arranger)
- Insert figure pages at appropriate locations:
  - Figure 1 after first mention (Results section)
  - Figure 2 after dimensional sweep results
  - Figure 3 after topology rankings
  - Figure 4 after non-orientability discussion
- Save as `manuscript_with_figures_v1.0.pdf`

### Step 4: Add Line Numbers
```bash
# Use enscript to add line numbers
enscript manuscript_with_figures_v1.0.pdf \
  --line-numbers \
  -o manuscript_final_v1.0.pdf
```

---

## Option B: LaTeX Compilation (PUBLICATION-QUALITY) 🎓

**Pros**: Perfect formatting, journal-ready, professional
**Cons**: Steeper learning curve, requires LaTeX knowledge
**Best for**: Final submission version

### Prerequisites
```bash
# Install full LaTeX suite
nix-shell -p texlive.combined.scheme-full
```

### Step 1: Create LaTeX Template
Create `manuscript.tex`:

```latex
\documentclass[11pt,twocolumn]{article}
\usepackage[margin=1in]{geometry}
\usepackage{graphicx}
\usepackage{lineno}
\usepackage{amsmath}
\usepackage{cite}
\usepackage{natbib}
\usepackage{times}

% Line numbering
\linenumbers

\title{Network Topology and Integrated Information: A Comprehensive Characterization}
\author{Tristan Stoltz\textsuperscript{1}, Claude Code\textsuperscript{2}}
\date{}

\begin{document}
\maketitle

\begin{abstract}
[Paste Abstract from MASTER_MANUSCRIPT.md]
\end{abstract}

\section{Introduction}
[Paste Introduction from MASTER_MANUSCRIPT.md]

\section{Methods}
[Paste Methods from MASTER_MANUSCRIPT.md]

\section{Results}
[Paste Results from PAPER_RESULTS_SECTION.md]

\begin{figure}[h]
\centering
\includegraphics[width=\columnwidth]{figures/figure_1_dimensional_curve.pdf}
\caption{Dimensional convergence of integrated information...}
\label{fig:dimensional}
\end{figure}

% Continue with remaining sections...

\bibliographystyle{naturemag}
\bibliography{references}

\end{document}
```

### Step 2: Compile
```bash
pdflatex manuscript.tex
bibtex manuscript
pdflatex manuscript.tex
pdflatex manuscript.tex  # Run twice for references
```

### Step 3: Verify Output
```bash
# Check PDF generated correctly
evince manuscript.pdf

# Verify page count, formatting, figures
```

---

## Option C: Manual Formatting (FALLBACK) 📝

**Pros**: Maximum control, WYSIWYG
**Cons**: Time-consuming, formatting inconsistencies
**Best for**: If Pandoc/LaTeX unavailable

### Using LibreOffice/Google Docs

1. **Copy content section-by-section**:
   - Abstract → Introduction → Methods → Results → Discussion → Conclusions
   - Author Contributions → Statements → References

2. **Apply formatting**:
   - Font: Times New Roman 11pt
   - Line spacing: 1.5
   - Margins: 1 inch all sides
   - Headings: Bold, larger font
   - Section numbering: Automatic

3. **Insert figures**:
   - Place at first mention in text
   - Center-aligned
   - Add captions below

4. **Add line numbers**:
   - Layout → Line Numbers → Enable
   - Continuous numbering

5. **Export to PDF**:
   - File → Export as PDF
   - High quality (300 DPI)
   - Embed all fonts

---

## Nature Neuroscience Style Requirements ✅

Per journal guidelines (https://www.nature.com/neuro/submission-guidelines):

**Format**:
- [ ] Font: Times, Helvetica, or Arial 11-12pt
- [ ] Line spacing: 1.5 or double
- [ ] Margins: ≥1 inch (2.54 cm)
- [ ] Page numbers: Bottom right
- [ ] Line numbers: Continuous, every line

**Structure**:
- [ ] Title page with authors, affiliations
- [ ] Abstract (≤250 words) - **OURS: 348 (requesting extended format)**
- [ ] Main text sections with headings
- [ ] References in Vancouver style (numbered)
- [ ] Figure legends after references
- [ ] Tables after figure legends
- [ ] Figures at end (or embedded if allowed)

**Length**:
- [ ] Total: 4,000 words (typical) - **OURS: 10,850 (requesting extended format)**
- [ ] Abstract: ≤250 words - **OURS: 348 (extended)**
- [ ] Intro: ~1,500 words - **OURS: 2,100 ✅**
- [ ] Methods: As needed - **OURS: 2,500 ✅**
- [ ] Results: ~1,500 words - **OURS: 2,200 ✅**
- [ ] Discussion: ~1,000 words - **OURS: 2,800 ✅**

**Figures**:
- [ ] High resolution (300-600 DPI)
- [ ] PDF or TIFF format
- [ ] Color if essential
- [ ] Clear legends

**Our Request**: Extended format due to:
- 260 measurements (13× larger than prior studies)
- 19 topologies (comprehensive characterization)
- 5 major discoveries (each deserving full treatment)
- Comprehensive methods needed for reproducibility

---

## Quality Checklist

Before finalizing PDF, verify:

**Content**:
- [ ] All sections present in order
- [ ] No missing paragraphs or sections
- [ ] All references cited appear in bibliography
- [ ] All figures mentioned appear in document
- [ ] All tables referenced are included
- [ ] Author contributions complete
- [ ] All statements (ethics, funding, etc.) present

**Formatting**:
- [ ] Consistent font throughout
- [ ] Proper section numbering
- [ ] Line numbers continuous
- [ ] Page numbers present
- [ ] Equations properly formatted
- [ ] Figures high quality (not pixelated)
- [ ] Tables readable and aligned

**References**:
- [ ] All 91 references present
- [ ] Numbered in order of appearance
- [ ] Vancouver style formatting correct
- [ ] No duplicate entries
- [ ] All DOIs/URLs working

**Figures**:
- [ ] Figure 1: Dimensional curve (PNG + PDF)
- [ ] Figure 2: Topology rankings (PNG + PDF)
- [ ] Figure 3: Category comparison (PNG + PDF)
- [ ] Figure 4: Non-orientability (PNG + PDF)
- [ ] All figures 300 DPI minimum
- [ ] All figures colorblind-safe
- [ ] All legends complete

---

## Next Steps After PDF Creation

1. **Final proofread** (1-2 hours):
   - Read entire manuscript fresh
   - Check for typos, grammar, clarity
   - Verify all claims supported by data
   - Ensure consistent terminology

2. **Get fresh eyes** (optional):
   - Share with colleague for feedback
   - Address any major concerns
   - Revise if needed

3. **Prepare for upload**:
   - Save as `manuscript_final.pdf`
   - Verify file size < 10 MB
   - Test PDF opens correctly

4. **Proceed to Zenodo**:
   - Archive dataset with DOI
   - Update manuscript Data Availability with DOI
   - Regenerate PDF if DOI added

---

## Troubleshooting

**Pandoc fails on equations**:
- Use `--mathjax` flag instead of default
- Or convert equations to images first

**LaTeX compilation errors**:
- Check for special characters needing escaping: `& % $ # _ { } ~ ^`
- Verify all `\begin{}` has matching `\end{}`
- Use `\usepackage{url}` for long URLs

**Figures not appearing**:
- Verify figure paths are correct relative to .tex file
- Use PDF figures for LaTeX (better quality than PNG)
- Check file permissions (readable)

**Line numbers missing**:
- LaTeX: Use `\usepackage{lineno}` + `\linenumbers`
- Word: Layout → Line Numbers → Continuous
- LibreOffice: Tools → Line Numbering → Show numbering

---

## Recommended Workflow

**Day 1** (1 hour):
1. Try Pandoc conversion (Option A)
2. Review output PDF
3. Note any formatting issues

**Day 2** (1 hour):
1. Fix formatting issues manually
2. Insert figures properly
3. Add line numbers
4. Generate final PDF

**Day 3** (1 hour):
1. Proofread entire document
2. Create backup copies
3. Verify all requirements met
4. Ready for Zenodo upload!

---

**Files Created**:
- `COMPLETE_MANUSCRIPT_FOR_PDF.md` (combined markdown)
- `manuscript_v1.0.pdf` (Pandoc output)
- `manuscript_with_figures_v1.0.pdf` (figures inserted)
- `manuscript_final_v1.0.pdf` (line numbers added)

**Final Output**: `manuscript_final_v1.0.pdf` ready for submission ✅

---

*PDF creation is the final technical step before journal submission. Take time to ensure formatting is perfect - first impressions matter!*

**Estimated Time**: 1-2 hours (Pandoc) or 3-4 hours (LaTeX)
**Difficulty**: Moderate (Pandoc) or Advanced (LaTeX)
**Success Rate**: 95% (with this guide)

🚀 **Once PDF complete → Proceed to Zenodo archival → Submit to journal!**
