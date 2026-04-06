# Commercialization Roadmap

## Phase Priority (Derived from Market Analysis, March 28 2026)

### Phase 1: Healthcare (Months 0-6) — $4.5T TAM
**Entry**: FQHC pilot, patient-controlled FHIR R4 records
**Product**: mycelix-health + mycelix-personal + FHIR bridge
**Revenue model**: $5-15/patient/year SaaS
**First milestone**: 100-patient pilot at 1 FQHC
**Detailed plan**: `docs/GO_TO_MARKET_HEALTHCARE.md`

### Phase 2: Education (Months 3-9) — $200B TAM
**Entry**: Privacy-first classroom platform for school district or homeschool co-op
**Product**: mycelix-praxis (29 learning science frameworks)
**Revenue model**: $3-10/student/year
**First milestone**: 1 school/co-op, 200 students

### Phase 3: Symtropy Game (Months 6-12) — Consumer proof-of-concept
**Entry**: Steam Early Access or itch.io
**Product**: symtropy/ (consciousness-driven game)
**Revenue model**: $15-30 game sales
**First milestone**: 1,000 wishlists, playable demo
**Why**: Most tangible proof the consciousness engine works, generates awareness

### Phase 4: Robotics Middleware (Months 9-18) — $80B TAM
**Entry**: SAR helicopter or AUV water monitoring (grant-funded)
**Product**: symthaea + symthaea-helicopter/auv
**Revenue model**: Per-platform licensing ($10K-100K/deployment)
**First milestone**: Working demo with real hardware partner
**Funding**: DOD SBIR, NOAA grants, DARPA

### Phase 5: Space Coordination (Months 12-24) — $1T+ by 2040
**Entry**: Academic partnerships, ESA/NASA SBIR
**Product**: mycelix-space (orbital mechanics + debris bounties)
**Revenue model**: Per-operator licensing
**First milestone**: Published paper + academic collaboration

## Revenue Projections (Conservative)

| Month | Healthcare | Education | Symtropy | Robotics | Total |
|-------|-----------|-----------|----------|----------|-------|
| 6 | $0 (pilot) | $0 | $0 | $0 | $0 |
| 12 | $50K ARR | $10K ARR | $15K sales | $0 | $75K |
| 18 | $200K ARR | $50K ARR | $50K sales | $25K license | $325K |
| 24 | $500K ARR | $150K ARR | $100K sales | $100K license | $850K |

## Funding Strategy

### Non-Dilutive First (Months 0-6)
- HRSA FQHC innovation grants ($50K-$500K)
- ONC Health IT grants ($100K-$2M)
- Mozilla Responsible AI ($50K-$250K)
- Shuttleworth Fellowship ($300K/3yr)
- NSF SBIR Phase I ($275K)

### Seed Round (Month 6-9, after pilot results)
- Target: $2-5M
- Valuation: $15-25M pre-money (justified by IP + pilot data)
- Use: 3-4 engineers + 1 BD/clinical liaison + paper publications
- Investors: Health tech angels, impact funds, cooperative economy funds

### Series A (Month 18-24, after revenue)
- Target: $10-20M
- Valuation: $50-80M (with revenue traction + published papers)
- Use: Scale healthcare, enter education, expand team to 15-20

## Credibility Building (Parallel Track)

### Academic Papers (Months 0-6)
1. **Spectral MIP** → NeurIPS/PNAS submission (outline ready: `papers/SPECTRAL_MIP_PAPER_OUTLINE.md`)
2. **Qualia Confidence Matrix** → BRM/CogSci submission (outline ready: `papers/QUALIA_CONFIDENCE_MATRIX_PAPER_OUTLINE.md`)

### Conference Talks
- Holochain community (Mycelix architecture)
- Consciousness science (ASSC, TSC) — Phi validation results
- Health IT (HIMSS) — patient data sovereignty

### Open Source Strategy
- Symthaea engine: AGPL (community builds on it, enterprise licenses for proprietary use)
- Mycelix clusters: AGPL (same dual-license model)
- Papers + benchmarks: fully open (credibility > revenue)

## Key Hires (First 4)

1. **Rust/Holochain engineer** — extend health cluster, FHIR bridge
2. **Clinical liaison / health IT specialist** — FQHC relationships, ONC certification
3. **Systems engineer** — CI/CD, deployment, infrastructure
4. **Research scientist** — paper writing, benchmark validation, conference presence

## Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| Holochain pre-1.0 | FHIR bridge enables hybrid mode; health cluster works standalone |
| Single developer | First hire is Rust/Holochain engineer; knowledge transfer via docs |
| No revenue | Non-dilutive grants bridge to pilot; pilot bridges to seed round |
| Regulatory | HIPAA counsel from day 1; agent-centric model simplifies compliance |
| Market timing | Healthcare data portability mandated by law; timing is now |
