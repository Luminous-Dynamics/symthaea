# Final Pre-Submission Checks - Zero-TrustML Paper

**Date**: November 5, 2025
**Status**: Execute before submission
**Estimated Time**: 1 hour
**Target**: IEEE S&P or USENIX Security

---

## 🎯 Purpose

This document provides the **final checklist** to execute immediately before submitting the paper. All items must pass before clicking "Submit" on the venue portal.

---

## ✅ Phase 1: PDF Quality Verification (15 minutes)

### 1.1 Generate Final PDF

```bash
cd latex-submission/

# Clean build
rm -f main.pdf main.aux main.log main.bbl main.blg *.aux

# Full compilation
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex

# Verify output
ls -lh main.pdf
```

**Expected**: `main.pdf` exists, size 1-5 MB

### 1.2 Check PDF Properties

```bash
# File size (must be <10 MB for most venues)
ls -lh main.pdf

# PDF info
pdfinfo main.pdf

# Expected output:
# - Title: "Byzantine-Robust Federated Learning Beyond the 33% Barrier..."
# - Author: (EMPTY or "Anonymous")
# - Creator: LaTeX
# - PDF version: 1.4 or higher
```

**Checklist**:
- [ ] File size <10 MB
- [ ] PDF version ≥ 1.4
- [ ] Title present
- [ ] Author field empty or "Anonymous"
- [ ] No identifying metadata

### 1.3 Verify Fonts Embedded

```bash
# Check all fonts embedded
pdffonts main.pdf

# Expected: All fonts show "yes" in "emb" column
```

**Checklist**:
- [ ] All fonts embedded (no "no" in "emb" column)
- [ ] No Type 3 fonts (bitmap fonts - cause rendering issues)
- [ ] Common fonts: Times, Helvetica, Computer Modern

### 1.4 Check for Author Metadata

```bash
# Check for identifying information in metadata
exiftool main.pdf | grep -i author
exiftool main.pdf | grep -i creator
exiftool main.pdf | grep -i producer

# Expected: Author field empty or "Anonymous"
```

**Checklist**:
- [ ] Author metadata empty or "Anonymous"
- [ ] No institution names in metadata
- [ ] No file paths revealing identity

### 1.5 Anonymization Verification (CRITICAL)

```bash
# Search PDF text for common identifying terms
pdftotext main.pdf - | grep -i "university"
pdftotext main.pdf - | grep -i "laboratory"
pdftotext main.pdf - | grep -i "our previous work"
pdftotext main.pdf - | grep -i "@"  # Email addresses

# Expected: No matches (or only in references)
```

**Checklist**:
- [ ] No university names
- [ ] No lab/institution names
- [ ] No "our previous work" phrases
- [ ] No email addresses
- [ ] No grant numbers
- [ ] No acknowledgments section

---

## ✅ Phase 2: Content Verification (20 minutes)

### 2.1 Open PDF and Visually Verify

**Open**: `main.pdf` in a PDF reader

**Checklist**:
- [ ] **Abstract**: Appears on first page, 189 words, readable
- [ ] **Sections**: All 7 sections present (1. Introduction through 7. Conclusion)
- [ ] **Figures**: All 3 figures (Figure 1, 2, 3) appear correctly
- [ ] **Tables**: All 9 tables (Table 1-9) render correctly
- [ ] **References**: Bibliography appears and is formatted consistently
- [ ] **Appendix**: Hyperparameters and implementation details present
- [ ] **Page Numbers**: Consecutive and correct

### 2.2 Page Count Verification

```bash
# Count pages
pdfinfo main.pdf | grep "Pages:"

# IEEE S&P: ≤13 pages (excluding references)
# USENIX Security: ≤15 pages (excluding references)
```

**Checklist**:
- [ ] Page count within venue limit
- [ ] References don't count toward limit
- [ ] Appendix counts toward limit (unless venue specifies otherwise)

### 2.3 Figure Quality Check

Open each figure and verify:

**Figure 1: Mode 1 Performance**
- [ ] Axes labeled clearly
- [ ] Legend present and readable
- [ ] 33% barrier line visible
- [ ] Data points match Table 1
- [ ] High resolution (no pixelation when zoomed)

**Figure 2: Confusion Matrix Grid**
- [ ] 4 matrices (20%, 35%, 45%, 50% BFT)
- [ ] Values match Table 1
- [ ] Color-coded appropriately
- [ ] Text readable at print size

**Figure 3: Mode 0 vs Mode 1 Comparison**
- [ ] Side-by-side layout clear
- [ ] Mode 0: 100% FPR visible
- [ ] Mode 1: 0% FPR visible
- [ ] Caption explains identical setup

### 2.4 Citation Verification

**Check for [?]** - Indicates missing citations:

```bash
# Search for broken citations
pdftotext main.pdf - | grep "\[?\]"

# Expected: No matches
```

**Checklist**:
- [ ] No [?] in PDF (all citations resolved)
- [ ] All citations compile correctly
- [ ] References section complete
- [ ] In-text citations match bibliography

### 2.5 Cross-Reference Verification

**Open PDF and spot-check**:

- [ ] "Figure 1" reference → Jumps to Figure 1 (if hyperref enabled)
- [ ] "Table 1" reference → Jumps to Table 1
- [ ] "Section 5.2" reference → Jumps to Section 5.2
- [ ] All cross-references accurate

---

## ✅ Phase 3: Numerical Accuracy Spot-Check (10 minutes)

### 3.1 Key Metrics Verification

**Open PDF and verify these critical numbers**:

- [ ] **Abstract**: "100% Byzantine detection with 7.7% FPR"
- [ ] **Table 1, 35% BFT Row**: Detection=100%, FPR=7.7%, τ=0.497104
- [ ] **Section 5.2** (Mode 0 vs Mode 1):
  - Mode 0: 100% detection, 100% FPR
  - Mode 1: 100% detection, 0% FPR (seed 42) or 7.7% FPR (multi-seed)
- [ ] **Table 9** (Adaptive Threshold):
  - Fixed: 84.6% FPR
  - Adaptive: 7.7% FPR
  - Improvement: 91% (correctly calculated as (84.6-7.7)/84.6)

### 3.2 Consistency Check

**Verify 7.7% FPR at 35% BFT appears in**:
- [ ] Abstract
- [ ] Table 1
- [ ] Section 5.1.3 (Confusion Matrix Analysis)
- [ ] Section 5.2 (Mode 0 vs Mode 1)
- [ ] Section 5.7 (Summary of Results)
- [ ] Introduction (Contribution statement)
- [ ] Discussion section

**All locations must show consistent value: 7.7%**

---

## ✅ Phase 4: LaTeX Compilation Verification (5 minutes)

### 4.1 Clean Compilation

```bash
cd latex-submission/

# Delete all auxiliary files
rm -f *.aux *.log *.bbl *.blg *.out *.toc *.lof *.lot

# Recompile from scratch
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex

# Check for errors
echo $?  # Should be 0 (success)
```

**Checklist**:
- [ ] `pdflatex` exits with code 0 (no errors)
- [ ] `bibtex` exits with code 0 (no errors)
- [ ] No "Undefined reference" warnings
- [ ] No "Citation undefined" warnings
- [ ] No "Overfull hbox" warnings exceeding 10pt

### 4.2 Log File Review

```bash
# Check for warnings in log
grep -i "warning" main.log | head -20
grep -i "error" main.log | head -20

# Common acceptable warnings:
# - "Underfull \hbox" (minor spacing issue)
# - "Font shape undefined" (if substituted correctly)

# Unacceptable errors:
# - "Undefined control sequence"
# - "Missing $ inserted"
# - "File not found"
```

**Checklist**:
- [ ] No critical errors in log
- [ ] All figures found and included
- [ ] All cross-references resolved
- [ ] Bibliography compiled correctly

---

## ✅ Phase 5: Supplementary Materials Preparation (10 minutes)

### 5.1 Code Repository (Optional but Recommended)

If including code:

**Create anonymous repository**:
1. Create new GitHub account (anonymous, e.g., `zerotrustml-anon`)
2. Upload code:
   - `tests/test_mode1_boundaries.py`
   - `tests/test_mode0_vs_mode1.py`
   - `tests/test_mode1_adaptive_threshold.py`
   - README with reproduction instructions
3. Add `.gitignore` to exclude identifying information
4. Double-check no commits reveal identity

**Checklist**:
- [ ] Repository URL anonymous
- [ ] README includes reproduction steps
- [ ] Dependencies documented (`requirements.txt` or `environment.yml`)
- [ ] Test execution instructions clear
- [ ] No commit messages revealing identity

### 5.2 Data Availability Statement

**Include in paper (Appendix or Section 4)**:

> **Data Availability**: Experiments use synthetic MNIST-like data generated with publicly documented parameters (Dirichlet α=0.1, seeds 42, 123, 456). Full data generation code is provided in the supplementary repository. No private or proprietary data was used.

**Checklist**:
- [ ] Data availability statement present
- [ ] Dataset clearly identified (MNIST)
- [ ] Data generation parameters documented
- [ ] Synthetic data creation reproducible

---

## ✅ Phase 6: Submission Portal Preparation (5 minutes)

### 6.1 Required Information

**Prepare in advance** (have ready before starting submission):

- [ ] **Paper Title**: "Byzantine-Robust Federated Learning Beyond the 33% Barrier: Ground Truth Validation with Decentralized Infrastructure"
- [ ] **Abstract** (189 words): [Copy from paper]
- [ ] **Keywords**: Byzantine Fault Tolerance, Federated Learning, Ground Truth Validation, Detector Inversion, Adaptive Thresholds, Holochain DHT, Decentralized Infrastructure
- [ ] **Paper PDF**: `main.pdf` (final version)
- [ ] **Supplementary Materials** (if any):
  - Code repository URL (anonymous)
  - Data availability statement
- [ ] **Conflicts of Interest**: List any reviewers with conflicts

### 6.2 Venue-Specific Requirements

**IEEE S&P**:
- [ ] Submission site: https://www.ieee-security.org/TC/SP2025/
- [ ] Format: IEEE conference template (2-column)
- [ ] Page limit: 13 pages (excluding references)
- [ ] Deadline: [Check current cycle - typically August or November]
- [ ] Double-blind: Yes (anonymization required)

**USENIX Security**:
- [ ] Submission site: https://www.usenix.org/conference/usenixsecurity25
- [ ] Format: USENIX LaTeX template
- [ ] Page limit: 15 pages (excluding references)
- [ ] Deadline: [Check current cycle - typically February or August]
- [ ] Double-blind: Yes (anonymization required)

---

## ✅ Phase 7: Final Sanity Checks (5 minutes)

### 7.1 The Last-Minute Checklist

**Before clicking "Submit", answer these questions**:

1. **Did you read the abstract one more time?**
   - [ ] Yes, and it's clear, concise, and compelling

2. **Are all figure numbers correct?**
   - [ ] Searched "Figure" in PDF and verified all references

3. **Are all table numbers correct?**
   - [ ] Searched "Table" in PDF and verified all references

4. **Is the PDF anonymized?**
   - [ ] Ran `pdftotext main.pdf - | grep -i "university"` → No matches
   - [ ] Ran `exiftool main.pdf | grep -i author` → Empty or "Anonymous"

5. **Is the page count within limits?**
   - [ ] IEEE S&P: ≤13 pages (excluding references)
   - [ ] USENIX Security: ≤15 pages (excluding references)

6. **Did you save a backup?**
   - [ ] Copied `main.pdf` to `zerotrustml-submission-YYYY-MM-DD.pdf`

---

## ✅ Phase 8: Submission Execution

### 8.1 IEEE S&P Submission Process

1. **Create Account**
   - Go to https://www.ieee-security.org/TC/SP2025/
   - Create account if needed
   - Verify email

2. **Start Submission**
   - Click "Submit New Paper"
   - Fill required fields:
     - Title
     - Abstract
     - Keywords
     - Authors (mark as "Anonymous" for review)

3. **Upload PDF**
   - Select `main.pdf`
   - Wait for upload confirmation
   - Verify file size and name

4. **Upload Supplementary Materials** (if any)
   - Code repository link
   - Additional appendices

5. **Declare Conflicts**
   - List reviewers with conflicts of interest
   - Be comprehensive to avoid ethical issues

6. **Review Submission**
   - Check all fields correct
   - Verify PDF displays correctly in portal
   - Confirm anonymization

7. **Submit**
   - Click "Submit Paper"
   - **SAVE CONFIRMATION EMAIL**
   - **NOTE PAPER ID NUMBER**

### 8.2 USENIX Security Submission Process

1. **Create Account**
   - Go to https://www.usenix.org/conference/usenixsecurity25
   - Create account if needed
   - Verify email

2. **Start Submission**
   - Click "Submit Paper"
   - Fill required fields:
     - Title
     - Abstract
     - Keywords
     - Optional artifacts (code, data)

3. **Upload PDF**
   - Select `main.pdf`
   - Wait for upload confirmation

4. **Optional: Request Artifact Evaluation**
   - If code is production-ready
   - Provides additional credibility

5. **Submit**
   - Click "Submit"
   - **SAVE CONFIRMATION EMAIL**
   - **NOTE PAPER ID NUMBER**

---

## ✅ Phase 9: Post-Submission Actions

### 9.1 Immediate Actions (Within 1 Hour)

- [ ] **Save confirmation email** to dedicated folder
- [ ] **Note paper ID** in project documentation
- [ ] **Archive submitted PDF**: `zerotrustml-submitted-YYYY-MM-DD-HH-MM.pdf`
- [ ] **Archive LaTeX source**: `tar -czf zerotrustml-latex-source.tar.gz latex-submission/`
- [ ] **Update project status**: Mark as "Submitted, awaiting review"

### 9.2 Calendar Reminders

- [ ] **Review Period Start**: [Venue-specific, typically 2-3 months]
- [ ] **Review Period End**: [Venue-specific]
- [ ] **Decision Notification**: [Venue-specific]
- [ ] **Camera-Ready Deadline** (if accepted): [Venue-specific, typically 2-4 weeks after acceptance]

### 9.3 Prepare for Revisions

- [ ] **Document**: Keep all test outputs, logs, and results
- [ ] **Backup**: Ensure all code and data backed up
- [ ] **Availability**: Be ready to respond to reviewer questions
- [ ] **Revisions**: Prepare to make revisions if needed (shepherd process)

---

## 📊 Final Status Verification

### Completion Checklist

| Phase | Status | Time | Notes |
|-------|--------|------|-------|
| 1. PDF Quality | ☐ | 15 min | File size, fonts, metadata |
| 2. Content Verification | ☐ | 20 min | Figures, tables, citations |
| 3. Numerical Accuracy | ☐ | 10 min | Key metrics spot-check |
| 4. LaTeX Compilation | ☐ | 5 min | Clean build verification |
| 5. Supplementary Materials | ☐ | 10 min | Code, data statements |
| 6. Portal Preparation | ☐ | 5 min | Gather required info |
| 7. Final Sanity Checks | ☐ | 5 min | Last-minute review |
| 8. Submission Execution | ☐ | - | Actually submit |
| 9. Post-Submission | ☐ | - | Archive and track |

**Total Time**: ~1 hour

**Status**: ☐ Ready to Submit  ☐ Issues Found (need fixing)

---

## 🎉 Submission Confidence Assessment

Answer honestly:

1. **Are you confident all results are correct?**
   - [ ] Yes, verified against test outputs

2. **Are you confident the paper is anonymous?**
   - [ ] Yes, no identifying information

3. **Are you confident the writing is clear?**
   - [ ] Yes, had team review

4. **Are you confident it will compile correctly?**
   - [ ] Yes, tested clean compilation

5. **Are you proud of this work?**
   - [ ] Yes, it represents our best effort

**If all YES**: **Submit with confidence!** 🚀

**If any NO**: **Fix before submitting.** Don't rush.

---

## 💡 Final Wisdom

**From the 3-Day Action Plan Success**:

> "Reviewers will trust '7.7% FPR' more than '0% FPR' because it shows realistic testing with heterogeneous data. **Honest results are more credible than perfect results.**"

**Submission Mantra**:

> "We have done the work. We have fixed the issues. We have been honest and thorough. This paper is ready."

**Expected Outcome**: **ACCEPT** with high confidence
- Addresses important problem (Byzantine FL beyond 33%)
- Novel contribution (ground truth validation)
- Strong empirical validation (direct A/B comparison)
- Practical impact (Holochain integration)
- Publication-quality presentation

---

**Date**: November 5, 2025
**Status**: Ready for final pre-submission checks
**Paper**: Zero-TrustML v1 (100% Submission-Ready)
**Target Venue**: IEEE S&P or USENIX Security
**Confidence**: HIGH 🎯

---

*"The final checks are the safety net. Execute them diligently, then submit with confidence."*
