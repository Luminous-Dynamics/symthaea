# Patent Filing Order -- Priority Schedule

*Prepared: March 27, 2026*
*Inventor: Tristan Stoltz, Luminous Dynamics*

---

## Immediate Filing (Grace expires 2027-02-05 -- 10 months remaining)

These 8 patents share the earliest grace period expiry (Feb 5, 2027, tied to the initial Symthaea public commit). File all by January 31, 2027 to provide a safety buffer.

| Priority | Patent | Title | Claims | Filing Cost | IDD Path |
|----------|--------|-------|--------|-------------|----------|
| 1 | P-001 | HDC-LTC Unified Neuron | 17 | $320 | `tier-1/P-001_hdc-ltc-neuron/IDD.md` |
| 2 | P-005 | Consciousness-Aware FL (PoGQ) | 18 | $320 | `tier-1/P-005_consciousness-fl/IDD.md` |
| 3 | P-004 | Consciousness Equation V2 | 18 | $320 | `tier-1/P-004_consciousness-equation/IDD.md` |
| 4 | P-007 | Differentiable Phi | 10 | $320 | `tier-2/P-007_differentiable-phi/IDD.md` |
| 5 | P-008 | Tiered Phi Measurement | 10 | $320 | `tier-2/P-008_tiered-phi/IDD.md` |
| 6 | P-009 | Conscious Reasoning Engine | 10 | $320 | `tier-2/P-009_reasoning-engine/IDD.md` |
| 7 | P-012 | Substrate Independence | 10 | $320 | `tier-2/P-012_substrate-independence/IDD.md` |
| 8 | P-015 | Incremental HDC Bundling | 7 | $320 | `tier-3/P-015_incremental-bundling/IDD.md` |

**Subtotal: 8 patents, $2,560**
**Deadline: January 31, 2027** (5 days before grace expiry)

### Rationale for Priority 1-3 Ordering

- **P-001 (HDC-LTC Neuron)** files first because it is the foundational architecture patent. All other patents reference or depend on the HDC-LTC neuron. Broadest claims (claims 16-17 cover D >= 100, no SIMD required). Strongest prior art distinction.
- **P-005 (Consciousness-Aware FL)** files second because it has the clearest commercial path (federated learning market) and the PoGQ consensus mechanism is the most novel protocol contribution.
- **P-004 (Consciousness Equation V2)** files third because it covers the consciousness measurement framework used by P-005, P-007, P-008, P-009, and P-012.

---

## Second Filing Wave (Grace expires Feb-Mar 2027)

These patents have staggered grace period expirations. File each at least 5 days before its individual deadline.

| Priority | Patent | Title | Grace Expires | Claims | Filing Cost |
|----------|--------|-------|---------------|--------|-------------|
| 9 | P-002 | Moral Algebra | 2027-02-07 | 16 | $320 |
| 10 | P-018 | Cross-Cluster Bridge | 2027-02-12 | 8 | $320 |
| 11 | P-016 | Adaptive Cognitive Topology | 2027-02-14 | 8 | $320 |
| 12 | P-014 | Consciousness Field Topology | 2027-02-17 | 8 | $320 |
| 13 | P-011 | 4D Consciousness Governance | 2027-02-21 | 10 | $320 |
| 14 | P-003 | LTC Vocal Tract Synthesis | 2027-02-23 | 17 | $320 |
| 15 | P-017 | Genesis Pipeline | 2027-02-28 | 8 | $320 |
| 16 | P-006 | Moral Topology | 2027-03-01 | 10 | $320 |
| 17 | P-010 | Psych-Bench | 2027-03-04 | 10 | $320 |
| 18 | P-013 | Neuromodulated Foveation | 2027-03-04 | 8 | $320 |
| 19 | P-019 | HDC-Native Homomorphic Encryption | 2027-03-17 | 15 | $320 |

**Subtotal: 11 patents, $3,520**

---

## Grand Total

| Phase | Count | Claims | Cost |
|-------|-------|--------|------|
| Immediate (Wave 1) | 8 | 100 | $2,560 |
| Second (Wave 2) | 11 | 118 | $3,520 |
| **Total** | **19** | **218** | **$6,080** |

---

## Filing Procedure (Self-File, Micro Entity)

### Per-Patent Steps

1. **Prepare specification**: Convert the `filing-ready/P-XXX_*_cleaned.md` file to PDF (use `convert-to-pdf.sh` in the filing-ready directory, or any markdown-to-PDF tool).

2. **Go to** https://patentcenter.uspto.gov

3. **Select** "File a New Application" then "Provisional"

4. **Entity status**: Select "Micro Entity" ($320 filing fee)

5. **Upload documents**:
   - Specification PDF (the cleaned markdown converted to PDF -- includes all sections)
   - Claims (included within the specification)
   - Drawings (if any -- text descriptions included; actual figures are optional for provisional applications but recommended)
   - Micro Entity Certification (Form PTO/SB/15A)
   - Provisional Application Cover Sheet (Form PTO/SB/16)

6. **Pay filing fee**: $320

7. **Record** confirmation number and filing date

8. **Post-filing**:
   - Update `PATENT_REGISTRY.md` with provisional application number and filing date
   - Set 12-month calendar reminder for utility filing deadline
   - Papers and products can now reference "Patent Pending"

### Required Forms

| Form | Purpose | Where to Get |
|------|---------|--------------|
| PTO/SB/15A | Micro Entity Certification | https://www.uspto.gov/patents/basics/fee-information/micro-entity |
| PTO/SB/16 | Provisional Application Cover Sheet | https://www.uspto.gov/patents/forms |

---

## Micro Entity Qualification

Tristan Stoltz qualifies as micro entity under 37 CFR 1.29:

- [ ] Has not been named as inventor on more than 4 previously filed patent applications (US or international)
- [ ] Gross income does not exceed 3x median household income ($234,788 threshold for 2026)
- [ ] Has not assigned, granted, or conveyed (and is not under obligation to do so) rights to an entity exceeding the income threshold
- [ ] Luminous Dynamics qualifies as the entity does not exceed the income threshold

**Important**: Micro entity status must be certified at time of filing. If status changes (e.g., income exceeds threshold, or 5th application is filed), subsequent filings must use small entity ($1,600) or large entity ($3,200) rates. The micro entity certification form (PTO/SB/15A) must be filed with each provisional application.

---

## Filing-Ready Status

| Patent | Cleaned Spec | Status |
|--------|-------------|--------|
| P-001 | `filing-ready/P-001_hdc-ltc-neuron_cleaned.md` | Ready |
| P-002 | `filing-ready/P-002_moral-algebra_cleaned.md` | Ready |
| P-003 | `filing-ready/P-003_ltc-vocal-tract_cleaned.md` | Ready |
| P-004 | `filing-ready/P-004_consciousness-equation_cleaned.md` | Ready |
| P-005 | `filing-ready/P-005_consciousness-fl_cleaned.md` | Ready |
| P-006 | -- | Needs cleaning |
| P-007 | -- | Needs cleaning |
| P-008 | -- | Needs cleaning |
| P-009 | -- | Needs cleaning |
| P-010 | -- | Needs cleaning |
| P-011 | -- | Needs cleaning |
| P-012 | -- | Needs cleaning |
| P-013 | -- | Needs cleaning |
| P-014 | -- | Needs cleaning |
| P-015 | -- | Needs cleaning |
| P-016 | -- | Needs cleaning |
| P-017 | -- | Needs cleaning |
| P-018 | -- | Needs cleaning |
| P-019 | `filing-ready/P-019_hdc-fhe_cleaned.md` | Ready |

**6 of 19 patents have filing-ready specifications.** Priority: clean P-007, P-008, P-009, P-012, P-015 next (remaining Wave 1 patents).

---

## Key Legal Reminders

- **Provisional to Utility**: Must file utility application within 12 months of each provisional, or lose priority date
- **International**: File PCT within 12 months of provisional for international protection. International jurisdictions have ZERO grace period (unlike US 1-year grace). Filing must precede any publication for international rights.
- **No examination**: Provisionals are NOT examined -- they only establish a priority date
- **Disclosure**: Must include sufficient detail to support later utility claims (the cleaned specs meet this requirement)
- **Continuation**: Can file continuation applications claiming priority to any provisional
- **HAI/PLoS papers**: Must file ALL relevant provisionals BEFORE submitting papers. Publication triggers the clock in every jurisdiction.

---

## Budget Projection (Full Year)

| Item | Estimated Cost |
|------|---------------|
| Provisional filing (19 x $320 micro entity) | $6,080 |
| Utility filing (Tier 1, 5 patents, with attorney) | $40,000 - $75,000 |
| Prior art search (Tier 1, 5 patents) | $5,000 - $10,000 |
| PCT international (top 3 patents) | $12,000 - $18,000 |
| **Total Year 1** | **$63,080 - $109,080** |

Provisionals can be filed immediately and independently. Attorney engagement for utility filing recommended by Q3 2026. Top 3 PCT candidates: P-001 (HDC-LTC Neuron), P-005 (Consciousness-Aware FL), P-019 (HDC-FHE).

---

*Filed under: /srv/luminous-dynamics/patents/FILING_ORDER.md*
