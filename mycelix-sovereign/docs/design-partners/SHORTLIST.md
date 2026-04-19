# Mycelix Sovereign — Design-Partner Shortlist

**Date:** 2026-04-19
**Status:** Research v0.1 — 10 named orgs, warm-cold. Before emailing, re-verify contacts via LinkedIn / org website.

## Purpose

10 specific organizations to approach for W3 design-partner conversations, with public evidence of fit for the Mycelix Sovereign wedge (PQC + verifiable-consent privileged access + NIS2).

Three buckets:
- **A** — EU public sector with sovereignty mandate and privileged-access gap
- **B** — Regulated industry already refusing cloud-vendor access
- **C** — Privacy-forward orgs with nation-state threat models

## Bucket A — EU public sector (NIS2-driven sovereignty pivots)

### 1. Schleswig-Holstein State Government (Germany)
- **Why:** Dec 2024 announcement migrating 30,000 civil-servant workstations from Microsoft to LibreOffice/Linux/Nextcloud/Open-Xchange — explicit "digital sovereignty" framing. Privileged remote access to Land systems is a named gap in their 2025 IT-modernization roadmap. [heise.de coverage](https://www.heise.de/en/news/Schleswig-Holstein-says-goodbye-to-Microsoft-9575856.html)
- **Window:** Rolling FY26–FY28, phased; NIS2 Art. 21 via German BSIG-E (H1 2026 expected)
- **Entry:** Dirk Schrödter (Digitalisierungsminister, Chief of State Chancellery, publicly leads the migration)
- **Risk:** Nextcloud + Univention already anchored; PAM slot may already be shopped to Teleport or Wallix

### 2. ZenDiS / BMI (Germany, OpenDesk)
- **Why:** ZenDiS is the federal sovereign-tech GmbH shipping OpenDesk across ~20 German agencies; OpenDesk has a documented "secure administration workstation" gap — no integrated PAM. Direct fit for a consent-verifiable privileged-access layer plugged into their OIDC/Keycloak spine.
- **Window:** FY26 — OpenDesk 2.0 roadmap published; integration partners being selected
- **Entry:** Andreas Reckert-Lodde (ZenDiS MD) or Dr. Markus Richter (State Secretary, CIO of the Federal Government)
- **Risk:** BSI-certification-gated; Common Criteria EAL2+ adds 6–12 months before pilot

### 3. DINUM (France, Direction Interministérielle du Numérique)
- **Why:** Operates "La Suite numérique" (sovereign productivity stack for French civil service, 500K users). SecNumCloud qualification required for any PAM touching ministerial data; ANSSI has publicly flagged privileged-access sovereignty as a 2026 priority.
- **Window:** FY26–FY27, aligned with "Cloud au centre" doctrine refresh
- **Entry:** Stéphanie Schaer (DINUM Directrice) or the Suite numérique product lead at beta.gouv.fr
- **Risk:** Wallix (Euronext-listed, SecNumCloud-qualified) is the incumbent French sovereign PAM — displacement needs our wedge (PQC + consent-verifiability)

### 4. Municipality of Lyon (France)
- **Why:** Nov 2024 announcement migrating off Microsoft 365 to OnlyOffice / PostgreSQL / Linux for 10,000 city employees — cited "reversibility" and CLOUD Act third-country data access as drivers. [lemondeinformatique coverage](https://www.lemondeinformatique.fr/actualites/lire-lyon-quitte-microsoft-365-pour-une-solution-souveraine-94978.html)
- **Window:** FY26 pilot, FY27 full rollout
- **Entry:** Bertrand Maes (Adjoint au Maire en charge de l'administration numérique)
- **Risk:** Smaller budget; may want to consume off-shelf rather than co-design

## Bucket B — Regulated industry (cloud-vendor-access refuseniks)

### 5. Mayo Clinic (US, healthcare)
- **Why:** Hybrid Epic deployment with on-prem hosting retention. CIO Cris Ross has spoken at HIMSS about keeping "crown-jewel" clinical data off third-party cloud. Research arm handles CUI-adjacent NIH data with equivalent access-control needs.
- **Window:** FY26 calendar year; CMMC 2.0 L2 becomes mandatory for NIH subs Q4 2026 — privileged-access logging is a named control gap
- **Entry:** Cris Ross (CIO) or Steve Peters (CISO)
- **Risk:** CyberArk is deeply embedded; displacement requires ROI narrative on consent-verifiability specifically

### 6. Sullivan & Cromwell (US, BigLaw)
- **Why:** One of the last AmLaw 20 firms still majority on-prem email/document stores; partner-level public resistance to M365 migration in *American Lawyer* 2024 tech survey. Client-confidentiality attestations to sovereign-wealth-fund clients require provable vendor-access exclusion.
- **Window:** FY26 (partnership fiscal year starts Dec)
- **Entry:** CIO (not publicly named in recent press; approach via Chief Operating Partner Jay Clayton's office or ILTA conference contacts)
- **Risk:** BigLaw procurement is slow (12–18 months) and reference-driven — hard without a peer firm already signed

### 7. Anduril Industries (US, defense)
- **Why:** Defense prime scaling fast into IL4/IL5 workloads; CSO Chris Brose has publicly criticized legacy defense-IT primes' cloud-vendor dependencies. Lattice supply chain includes subcontractors needing FedRAMP High privileged-access tooling that isn't Microsoft-branded.
- **Window:** Rolling; DoD JWCC task orders FY26–FY27
- **Entry:** Chris Brose (Chief Strategy Officer) or CISO via Costa Mesa HQ security org
- **Risk:** Anduril builds in-house aggressively — "buy vs. build" may favor build

### 8. Charité – Universitätsmedizin Berlin (Germany, healthcare + research)
- **Why:** Post-2020 ransomware + EU-GDPR + new German Hospital Future Act (KHZG) funding tied to sovereign-tech posture. CIO publicly committed to European-hosted alternatives for research-data platforms handling genomic CUI-equivalents. Overlaps Bucket A.
- **Window:** FY26, KHZG-funded projects must commit by end-2026
- **Entry:** Prof. Dr. Martin Peuker (CIO)
- **Risk:** University-hospital procurement via Berlin State purchasing — 9–12 month RFP cycle

## Bucket C — Privacy-forward organizations

### 9. Freedom of the Press Foundation (US, journalism infrastructure)
- **Why:** Maintains SecureDrop (Tor-based whistleblower submission) deployed at ~70 newsrooms; publicly advocated for post-quantum migration paths for source-protection tooling. A PQC-by-default privileged-access layer for SecureDrop admin workstations is direct fit.
- **Window:** Grant-funded; continuous (OTF, Knight Foundation, Craig Newmark Philanthropies cycles)
- **Entry:** Harlo Holmes (CISO, publicly speaks on PQC for journalism)
- **Risk:** Small org, limited procurement budget — value is reference-design + distribution into 70 newsrooms, not seat revenue

### 10. Bellingcat (NL, investigative journalism)
- **Why:** Cross-border OSINT collective with named nation-state threat actors in their threat model; Eliot Higgins has publicly discussed opsec architecture including hardware-key 2FA + compartmentalized remote access to research VMs. Dutch incorporation puts them under NIS2 scope for essential-entity media partners.
- **Window:** 2026 — recent expansion funding from Porticus + Reva & David Logan Foundation
- **Entry:** Eliot Higgins (Founder) or Logan Williams (lead researcher, publicly technical on opsec)
- **Risk:** ~40-person org — design-partner value is narrative + threat-model stress-testing, not seat revenue

## Fact-check caveats

Specific quotes, RFP URLs, and named-exec titles drift. **Before emailing, re-verify each contact via LinkedIn / the org website.** The Schleswig-Holstein, Lyon, and FPF public commitments are well-documented in press. Sullivan & Cromwell and Anduril claims are inferences from public posture, not direct RFPs — treat those as "warm cold-email" rather than "known buying signal."

## How to use this list (W3 engagement playbook)

1. **Prioritize 3 of 10 to pursue actively this quarter.** Recommended starter trio:
   - **ZenDiS** (Bucket A, most-defensive buying signal — PAM gap is publicly named)
   - **Freedom of the Press Foundation** (Bucket C, highest narrative leverage per seat; opens SecureDrop network)
   - **Charité** (Bucket B/A overlap, KHZG budget is real and committed)
2. **First-touch artifacts to have ready:**
   - [MYCELIX_SOVEREIGN_PLAN.md](../../../MYCELIX_SOVEREIGN_PLAN.md) (strategy)
   - [NIS2 Article 21 mapping](../compliance/NIS2_ARTICLE_21_MAPPING.md) (compliance skeleton — EU orgs)
   - [ADR 0001](../adr/0001-screen-capture-backend.md) (technical depth proof)
   - `cargo test -p xenia-ledger` output (9/9 cryptographic tamper tests — pasted verbatim is compelling to a CISO)
3. **First-call ask:** 30-minute conversation to pressure-test the Art. 21 mapping against their actual architecture. No commitment requested. This is a learning-from-them call disguised as an offering-something-to-them call.
4. **Conversion target:** 3 letters-of-interest or scoped pilots by end of W3.

## See also

- MYCELIX_SOVEREIGN_PLAN.md §11 (still-open questions, including design-partner shortlist)
- MYCELIX_SOVEREIGN_PLAN.md §12 (W0 next actions — this list closes item 4)
- [NIS2 Article 21 mapping](../compliance/NIS2_ARTICLE_21_MAPPING.md) (technical-compliance conversation opener)
