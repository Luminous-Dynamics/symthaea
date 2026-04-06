# Mycelix Health — Go-to-Market Strategy

## One-Line Pitch

Patient-controlled health records on Holochain — HIPAA/FHIR compliant, no corporate data silos.

## The Problem

- Patients don't own their health data — Epic, Cerner, and insurers do
- Data portability is a legal right (21st Century Cures Act) but practically impossible
- Interoperability mandates (USCDI, FHIR R4) remain unenforced
- Data breaches expose millions annually — centralized databases are honeypots

## The Solution

**Mycelix Health**: A decentralized health information system where every patient's data lives on their own Holochain source chain.

### Technical Differentiators

| Feature | Mycelix Health | Epic/Cerner | Apple Health |
|---------|---------------|-------------|--------------|
| Data ownership | Patient (source chain) | Hospital | Limited export |
| Interoperability | FHIR R4 native | FHIR adapter | HealthKit only |
| Consent management | On-chain, granular | Per-system | Binary |
| Privacy | Agent-centric DHT | Central DB | Device-only |
| Cost per patient | ~$0 (P2P) | $30K-$60K/bed | Free (limited) |

### Cluster Readiness

- **15 zomes** across 3 tiers (7 MVP + 8 Tier 2)
- **4,228 tests** passing
- **FHIR R4 compliance**: Native data model
- **Clinical Decision Support** (CDS): Evidence-based alerts
- **Telehealth**: Encrypted video/messaging
- **Nutrition tracking**: WHO dietary guidelines
- **Insurance module**: Claims, eligibility, pre-authorization
- **Clinical trials**: Consent, enrollment, data collection

## Target Customer Segments

### Tier 1: Community Health Centers / FQHCs (Fastest path)
- **Why**: Underserved, price-sensitive, mission-aligned
- **Size**: 1,400+ FQHCs in the US serving 30M patients
- **Pain**: Can't afford Epic, stuck on outdated EHRs
- **Entry**: Replace or supplement existing EHR with patient-controlled records
- **Revenue**: $5-15/patient/year SaaS

### Tier 2: Concierge/DPC Practices
- **Why**: Already patient-centric, tech-forward, small enough to adopt
- **Size**: ~25,000 DPC practices in the US
- **Pain**: Need portable records that follow patients
- **Entry**: Patient data vault + FHIR bridge to existing systems
- **Revenue**: $20-50/patient/year premium

### Tier 3: Health Cooperatives / Mutual Aid
- **Why**: Ideologically aligned, cooperative governance matches Mycelix architecture
- **Size**: Growing movement post-COVID
- **Pain**: No infrastructure for health data sharing within cooperatives
- **Entry**: Full stack — records, consent, sharing, group analytics
- **Revenue**: Per-cooperative licensing

## Pilot Program Design

### Phase 1: Single FQHC Pilot (Months 1-6)
- **Goal**: 100 patients using Mycelix Health for records + consent management
- **Scope**: identity-vault (personal cluster) + health-vault + FHIR bridge
- **Success metrics**:
  - Patient records created and portable
  - FHIR R4 compliance verified by ONC-certified testing
  - Patient satisfaction > 80%
  - Zero data breaches (agent-centric = no central target)
- **Resources needed**: 1 engineer (Holochain/Rust), 1 clinical liaison
- **Cost**: ~$50K for 6 months (eng salary + FQHC coordination)

### Phase 2: Multi-Site Expansion (Months 6-12)
- **Goal**: 3-5 FQHCs, 1,000+ patients
- **Add**: CDS alerts, telehealth, insurance pre-auth
- **Revenue target**: $50K ARR (proof of willingness-to-pay)

### Phase 3: Platform Play (Months 12-24)
- **Goal**: White-label Mycelix Health for EHR vendors
- **Add**: Clinical trials module, nutrition, data dividends
- **Revenue target**: $500K ARR

## Regulatory Path

| Requirement | Status | Notes |
|-------------|--------|-------|
| HIPAA BAA | Needed | Agent-centric model simplifies — no central processor |
| ONC Certification | Needed | FHIR R4 compliance already built |
| FDA 21 CFR Part 11 | If clinical trials | Electronic signatures module ready |
| State-specific | Varies | Patient consent on-chain provides audit trail |

**Key advantage**: Holochain's agent-centric model means there's no central "covered entity" holding all patient data. Each patient IS their own data controller. This fundamentally simplifies HIPAA compliance.

## Competitive Landscape

| Competitor | Approach | Weakness |
|------------|----------|----------|
| **Epic MyChart** | Patient portal on Epic's servers | Hospital-owned, not portable |
| **CommonHealth** | Android health data app | No clinical integration |
| **Patientory** | Blockchain health records | Gas fees, blockchain scalability |
| **MedRec** | Ethereum health records | Same blockchain problems |
| **Solid (Inrupt)** | Pod-based data ownership | No health-specific features |

**Our moat**: Agent-centric DHT (no gas, no mining) + 15 health-specific zomes + FHIR native + consciousness gating for access control.

## Funding Sources

### Grants (Non-dilutive)
- **HRSA FQHC grants**: Community health innovation ($50K-$500K)
- **ONC Health IT grants**: Interoperability advancement ($100K-$2M)
- **Mozilla Responsible AI**: Open-source health tech ($50K-$250K)
- **Shuttleworth Fellowship**: Open infrastructure ($300K over 3 years)
- **NSF SBIR Phase I**: Health informatics ($275K)

### Angel/Seed
- Target: $500K-$2M seed round after pilot results
- Pitch: "Patient-owned health records" + pilot data + FHIR compliance
- Target investors: Health tech angels, cooperative economy funds, impact investors

## Key Risks

1. **Holochain maturity**: Still pre-1.0 — mitigation: FHIR bridge allows hybrid mode
2. **Clinical adoption**: Doctors resist workflow changes — mitigation: supplement, don't replace
3. **Regulatory uncertainty**: Novel architecture — mitigation: HIPAA counsel from day 1
4. **Competition from Apple/Google**: Big tech health platforms — mitigation: they don't serve FQHCs

## Next Steps

1. [ ] Identify 3 potential FQHC pilot partners (TX/CA/NY)
2. [ ] Prepare ONC FHIR compliance test suite
3. [ ] Draft HIPAA analysis memo (agent-centric model)
4. [ ] Apply for HRSA innovation grant (next cycle)
5. [ ] Build FHIR bridge demo (Mycelix Health → Epic FHIR R4 endpoint)
