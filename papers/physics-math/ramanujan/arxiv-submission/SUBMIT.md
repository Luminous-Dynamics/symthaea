# Ramanujan Protocol — arXiv Submission Runbook

Step-by-step how to push this paper to arXiv. Everything arXiv asks for
is already staged in this directory.

## Prerequisites

- An arXiv account with submission privileges. If you're submitting your
  first cs.SC paper, you may need an endorser. Apply for endorsement at
  https://arxiv.org/auth/need-endorsement if prompted.
- Your ORCID identifier (recommended but optional).
- A current MSC 2020 code reference handy (we've picked three in METADATA.md).

## One-time prep: build the source tarball

From this directory:

```bash
cd /srv/luminous-dynamics/symthaea/papers/physics-math/ramanujan
# Make sure the paper compiles clean first
./reproduce.sh          # regenerates results_table.tex from live code
nix-shell -p texliveFull --run "pdflatex -interaction=nonstopmode main.tex && pdflatex -interaction=nonstopmode main.tex"

# Source-only tarball for arXiv (main.tex + inputs, NO pdfs/aux/log)
cd arxiv-submission
tar czf ramanujan_arxiv_source.tar.gz -C .. main.tex results_table.tex

# Ancillary files (proofs + reproduction harness)
tar czf ramanujan_ancillary.tar.gz \
  -C .. proofs reproduce.sh Dockerfile VERIFY.md \
  showcase_stdout.txt showcase_stderr.txt
```

Sanity check: the source tarball must contain `main.tex` at its root and
compile with `pdflatex` in a vanilla texlive environment. Test in a fresh
directory:

```bash
mkdir -p /tmp/arxiv-test && cd /tmp/arxiv-test
tar xzf /srv/luminous-dynamics/symthaea/papers/physics-math/ramanujan/arxiv-submission/ramanujan_arxiv_source.tar.gz
nix-shell -p texliveFull --run "pdflatex -interaction=nonstopmode main.tex && pdflatex -interaction=nonstopmode main.tex"
# Expect: main.pdf, 12 pages, no errors
```

## Step-by-step arXiv submission

1. Go to https://arxiv.org/submit

2. **Start submission** → choose license `arXiv.org perpetual, non-exclusive`

3. **Files**:
   - Upload `ramanujan_arxiv_source.tar.gz`
   - arXiv will auto-compile. Inspect the generated PDF in the preview.
   - If the compile fails, fix in the source tree locally, re-tar, re-upload.

4. **Metadata** (copy from METADATA.md in this directory):
   - Title: `Ramanujan Protocol: Autonomous Conservation-Law Discovery with Z3 Formal Proofs`
   - Authors: `Tristan Stoltz (Luminous Dynamics)`
   - Abstract: full text from METADATA.md
   - Comments: from METADATA.md
   - Primary subject: `cs.SC`
   - Cross-list: `cs.LG`, `math.DS`, `cs.AI`
   - MSC codes: `68W30, 37J05, 68T20`

5. **Ancillary files** (optional but highly recommended for this paper):
   - Upload `ramanujan_ancillary.tar.gz` in the ancillary-files section.
   - These ship alongside the paper but are not compiled. Reviewers can
     download them to verify the Z3 proofs directly.

6. **Review** → **Submit**.

7. arXiv assigns an identifier (e.g., `2604.XXXXX`). Save it.

## After submission

- Email the arXiv ID + PDF link to tristan.stoltz@evolvingresonantcocreationism.com
  and any collaborators you want to notify.
- Consider posting to relevant venues (cs.SC mailing list, Twitter, etc.).
- If you plan conference submission (ICML / NeurIPS scientific-discovery
  workshop), the workshop form usually wants the arXiv link plus a
  cover-letter-style pitch — see `COVER_LETTER.md` in this directory.

## If arXiv rejects / flags the submission

Most common issues for pipeline papers:
- **Missing endorsement**: submit endorsement request with a paragraph
  explaining the paper's topic. Takes 1-7 days.
- **Auto-compile failure**: usually a missing TeX package. The file
  `main.tex` uses only `amsmath`, `amssymb`, `booktabs`, `hyperref`,
  `geometry` — all standard. If arXiv fails, check the log for the
  missing package and add `\usepackage{PKG}` or the class to use.
- **Subject reclassification**: arXiv may move cs.SC → cs.LG or similar.
  Accept unless you feel strongly; the paper will reach the right audience
  either way.

## Not submitted from here — why not

This runbook prepares everything. It does not submit for you because:
1. Your arXiv credentials are not available to this toolchain.
2. Submission is a permanent public action; you should be the one
   clicking the final Submit button.

The tarballs are ready; you upload them when you're ready.
