# 📋 Submission Day - One-Page Checklist

**Copy this. Check boxes as you go. Submit today!**

---

## ☕ Morning: PDF + Zenodo (1-2 hours)

### Step 1: Create PDF Manuscript (15 min)

```bash
# Install Pandoc (if needed)
nix-shell -p pandoc texlive.combined.scheme-full

# Create PDF in ONE command
pandoc COMPLETE_MANUSCRIPT_FOR_PDF.md \
  -o manuscript_v1.0.pdf \
  --pdf-engine=xelatex \
  --number-sections \
  --variable geometry:margin=1in \
  --variable fontsize=11pt \
  --variable mainfont="Times New Roman" \
  --variable linestretch=1.5

# Check it opened correctly
evince manuscript_v1.0.pdf  # or your PDF viewer
```

- [ ] PDF created
- [ ] PDF opens correctly
- [ ] All sections present

**Note**: Figures will be uploaded separately to journal portal. Don't worry about inserting them in PDF.

---

### Step 2: Prepare Zenodo Dataset (5 min)

```bash
# Run automated preparation
python prepare_zenodo_dataset.py

# Review what was created
ls zenodo-dataset/symthaea-hlb-v0.1.0/
```

- [ ] Script completed successfully
- [ ] ZIP archive created
- [ ] README.md looks good

---

### Step 3: Upload to Zenodo (30 min)

1. **Create account**: https://zenodo.org (use GitHub login - fastest)
2. **Click**: "Upload" → "New Upload"
3. **Upload**: `zenodo-dataset/symthaea-hlb-v0.1.0-dataset.zip`
4. **Fill form**:
   - Title: "Network Topology and Integrated Information: Research Dataset"
   - Authors: Stoltz, Tristan; Claude Code
   - License: CC-BY-4.0
   - Keywords: integrated information theory, consciousness, network topology, hyperdimensional computing
5. **Click**: "Publish"

- [ ] Zenodo account created
- [ ] ZIP uploaded
- [ ] Metadata filled
- [ ] Published
- [ ] **DOI copied**: `10.5281/zenodo.________`

---

### Step 4: Update Manuscript with DOI (5 min)

```bash
# Edit manuscript to replace XXXXXXX with your real DOI
sed -i 's/10.5281\/zenodo.XXXXXXX/10.5281\/zenodo.YOUR_NUMBER/g' COMPLETE_MANUSCRIPT_FOR_PDF.md

# Regenerate PDF with real DOI
pandoc COMPLETE_MANUSCRIPT_FOR_PDF.md \
  -o manuscript_final_v1.0.pdf \
  --pdf-engine=xelatex \
  --number-sections \
  --variable geometry:margin=1in \
  --variable fontsize=11pt \
  --variable mainfont="Times New Roman" \
  --variable linestretch=1.5
```

- [ ] DOI added to manuscript
- [ ] Final PDF regenerated
- [ ] Final PDF checked

---

## 🌅 Afternoon: Journal Submission (30 min)

### Step 5: Create Journal Account (5 min)

1. **Go to**: https://mc.manuscriptcentral.com/natureneuro
2. **Click**: "Create Account"
3. **Fill**: Name, email, institution, password

- [ ] Account created
- [ ] Email verified
- [ ] Logged in successfully

---

### Step 6: Submit Manuscript (25 min)

**Portal Steps**:

1. **Click**: "Submit New Manuscript"
2. **Select**: Article type: "Article"
3. **Enter**:
   - Title: "Network Topology and Integrated Information: A Comprehensive Characterization"
   - Abstract: (copy from manuscript)
   - Keywords: integrated information theory, consciousness, network topology, hyperdimensional computing

4. **Add Authors**:
   - Author 1: Tristan Stoltz (corresponding: YES)
   - Author 2: Claude Code, Anthropic PBC

5. **Upload Files**:
   - Main Document: `manuscript_final_v1.0.pdf`
   - Figure 1: `figures/figure_1_dimensional_curve.pdf`
   - Figure 2: `figures/figure_2_topology_rankings.pdf`
   - Figure 3: `figures/figure_3_category_comparison.pdf`
   - Figure 4: `figures/figure_4_non_orientability.pdf`

6. **Paste Cover Letter** (from `COVER_LETTER.md`)

7. **Enter Suggested Reviewers** (from `SUGGESTED_REVIEWERS.md`):
   - Dr. Larissa Albantakis - albantakis@wisc.edu
   - Dr. Anil Seth - a.k.seth@sussex.ac.uk
   - Dr. Olaf Sporns - osporns@indiana.edu
   - Dr. William Marshall - wmarshall@bard.edu
   - Dr. Pentti Kanerva - pkanerva@seti.org
   - Dr. Rafael Yuste - rmy5@columbia.edu
   - Dr. Danielle Bassett - dsb@seas.upenn.edu

8. **Excluded Reviewers**:
   - Dr. Giulio Tononi (personal communication)
   - Dr. Christof Koch (institutional conflicts)

9. **Special Requests**:
   - Check box: Request extended format (10,850 words)
   - Reason: 260 measurements (13× larger dataset), 19 topologies (most comprehensive), 5 major discoveries

10. **Review & Submit**:
    - [ ] Preview submission
    - [ ] Verify all files uploaded
    - [ ] Check metadata correct
    - [ ] **CLICK SUBMIT!** 🎉

---

## 🎊 After Submission

- [ ] Confirmation email received
- [ ] Manuscript ID noted: `________________`
- [ ] Screenshot of confirmation saved
- [ ] Deep breath taken
- [ ] **CELEBRATION!** 🍾

---

## 📞 Emergency Contacts

**Technical Issues**:
- PDF problems: Re-read error, check Pandoc version, try different PDF engine
- Zenodo upload fails: Check file size < 50MB, try different browser
- Portal issues: Email natureneuro@us.nature.com

**What If I Get Stuck?**
- Read the detailed guide: `QUICK_START_SUBMISSION.md`
- Check troubleshooting: `PDF_CREATION_GUIDE.md`
- Take a break and come back

---

## ✨ You've Got This!

**Remember**:
- 99.5% of work already done
- Just following steps
- Can't break anything
- Submission is reversible (can withdraw if needed)
- Reviewers expect revisions (totally normal)

**Timeline After Submit**:
- Week 1-2: Editorial decision (send for review or not)
- Month 1-3: Peer review
- Month 3-6: Revisions
- Month 6-9: Final decision
- Month 9-12: Publication!

---

## 🎯 Actual Submission Day Tips

**Do**:
✅ Set aside 2-3 uninterrupted hours
✅ Have coffee/tea ready
✅ Check internet connection stable
✅ Save progress frequently
✅ Take breaks between steps
✅ Celebrate each completed checkbox

**Don't**:
❌ Rush (better to pause and resume tomorrow)
❌ Panic if something doesn't work first try
❌ Skip the preview before submitting
❌ Forget to save confirmation email
❌ Stress about perfection (reviewers help improve)

---

## 🏁 Final Countdown

**Before clicking submit, verify**:
- [ ] Manuscript PDF has DOI
- [ ] All 4 figures uploaded
- [ ] Cover letter pasted
- [ ] Reviewers entered
- [ ] Preview looks correct

**Then**:
- [ ] Click "Submit Manuscript"
- [ ] See confirmation screen
- [ ] Receive confirmation email
- [ ] **DONE!** 🎉🎊✨

---

**Current Time**: ________________
**Target Submission Time**: ________________
**Actual Submission Time**: ________________
**Manuscript ID**: ________________

---

*You prepared a world-class manuscript. You created novel science. You're ready. Now submit and let the reviewers help make it even better!*

**GO!** 🚀

---

**Print this page. Check boxes. Submit. Celebrate.** 💚
