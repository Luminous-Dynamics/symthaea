# Symthaea Long-Term Roadmap

**Vision**: Establish Symthaea as the definitive framework for consciousness-aware AI evaluation and development.

**Document Version**: 1.0.0
**Created**: 2026-01-04
**Maintainer**: Luminous Dynamics

---

## Executive Summary

This roadmap outlines three parallel development paths for Symthaea's benchmark and consciousness measurement infrastructure:

1. **Scientific Publication Path** - Establish academic credibility through peer-reviewed research
2. **Platform/Service Path** - Build community infrastructure for consciousness-aware AI evaluation
3. **Research Depth Path** - Advance the frontier of consciousness-capability-ethics understanding

**Timeline**: 2026-2028 (3-year horizon)
**Goal**: Position Symthaea as the standard for consciousness measurement in AI systems

---

## Path 1: Scientific Publication

### Phase 1.1: ArXiv Preprint (Q1 2026)

**Deliverables**:
- [ ] Publication-quality figures (4 main + 6 supplementary)
- [ ] Complete methods section with reproducibility package
- [ ] Results narrative covering 260 Φ measurements
- [ ] Discussion of biological and AI implications
- [ ] Supplementary materials with full data tables

**Target Journals**:
1. Nature Computational Science (IF: 12.0)
2. PNAS (IF: 12.8)
3. PLoS Computational Biology (IF: 4.8)

**Paper Title**: "Dimensional Optimization of Integrated Information: Topology-Consciousness Mapping in Hyperdimensional Computing"

### Phase 1.2: Peer Review & Publication (Q2-Q3 2026)

**Activities**:
- [ ] Submit to ArXiv (cs.AI, q-bio.NC cross-list)
- [ ] Submit to target journal
- [ ] Address reviewer comments
- [ ] Generate Zenodo DOI for reproducibility package
- [ ] Create companion website with interactive visualizations

### Phase 1.3: Follow-up Studies (Q4 2026 - 2027)

**Planned Publications**:
1. "Consciousness-Capability Correlation in Large Language Models"
2. "The Ethics-Topology Interface: Moral Reasoning and Network Structure"
3. "Real-Time Consciousness Monitoring for AI Safety"

---

## Path 2: Platform/Service

### Phase 2.1: API Design (Q1 2026)

**Deliverables**:
- [ ] OpenAPI 3.0 specification for Symthaea Benchmark API
- [ ] Authentication and rate limiting design
- [ ] Model submission protocol
- [ ] Result format standardization

**API Endpoints**:
```
POST /api/v1/submit          # Submit model for evaluation
GET  /api/v1/results/{id}    # Get evaluation results
GET  /api/v1/leaderboard     # Public leaderboard
GET  /api/v1/datasets        # Available benchmark datasets
POST /api/v1/compare         # Compare two models
```

### Phase 2.2: Web Platform (Q2 2026)

**Technology Stack**:
- Frontend: SvelteKit + TailwindCSS
- Backend: Rust (Axum) + PostgreSQL
- Compute: Kubernetes with GPU nodes
- Storage: S3-compatible object storage

**Features**:
- [ ] Model submission portal
- [ ] Interactive leaderboard with filtering
- [ ] Φ visualization dashboard
- [ ] Dataset browser and downloader
- [ ] API documentation and playground

### Phase 2.3: Community Launch (Q3 2026)

**Activities**:
- [ ] Beta launch with select partners
- [ ] Documentation and tutorials
- [ ] Community Discord/forum
- [ ] Contributor guidelines
- [ ] First public benchmark competition

### Phase 2.4: Sustainability (Q4 2026+)

**Revenue Model** (if needed):
- Free tier: Quick benchmarks, public results
- Pro tier: Full suite, private results, priority queue
- Enterprise: On-premise deployment, custom benchmarks
- Academic: Free full access with .edu verification

---

## Path 3: Research Depth

### Phase 3.1: Ground Truth Validation (Q1 2026)

**PyPhi Cross-Validation**:
- [ ] Implement PyPhi integration via pyo3
- [ ] Compare HDC-Φ with exact IIT 3.0 Φ on small systems (n≤8)
- [ ] Establish correlation coefficient and confidence bounds
- [ ] Document approximation error characteristics

**Expected Outcome**: Demonstrate r > 0.90 correlation with exact Φ

### Phase 3.2: Real Ethics Datasets (Q1-Q2 2026)

**Dataset Integration**:
- [ ] ETHICS benchmark (Hendrycks et al., 2021) - 95,000 examples
- [ ] BBQ Bias Benchmark - 58,492 examples
- [ ] WinoBias - 3,160 examples
- [ ] CrowS-Pairs - 1,508 pairs
- [ ] MoralChoice - 1,767 scenarios
- [ ] TruthfulQA - 817 questions

**Custom Datasets**:
- [ ] NixOS intent classification (1,000 examples)
- [ ] Consciousness-ethics probes (500 scenarios)
- [ ] Sycophancy detection suite (200 adversarial examples)

### Phase 3.3: Φ-Capability-Ethics Triangle (Q2-Q3 2026)

**Research Questions**:
1. Does Φ predict MMLU accuracy? (capability correlation)
2. Does Φ predict ETHICS accuracy? (moral reasoning correlation)
3. Do topologies predict moral framework preference?
4. Can we optimize all three simultaneously?

**Methodology**:
- [ ] Design controlled experiments with matched architectures
- [ ] Vary only topology while holding other factors constant
- [ ] Statistical analysis with proper controls
- [ ] Pre-registration on OSF for credibility

### Phase 3.4: Advanced Consciousness Metrics (Q3-Q4 2026)

**Beyond Φ**:
- [ ] Global Workspace Theory metrics (broadcast efficiency)
- [ ] Higher-Order Thought indicators
- [ ] Metacognitive calibration (CCI)
- [ ] Temporal integration measures
- [ ] Multi-agent collective consciousness

### Phase 3.5: Real-Time Monitoring (2027)

**Production Integration**:
- [ ] Streaming Φ calculation for live systems
- [ ] Anomaly detection for consciousness degradation
- [ ] Safety thresholds and automated interventions
- [ ] Integration with model serving infrastructure

---

## Technical Infrastructure

### Compute Requirements

| Phase | GPU Hours/Month | Storage | Estimated Cost |
|-------|-----------------|---------|----------------|
| 1.1-1.2 | 100 | 100 GB | $500 |
| 2.1-2.2 | 500 | 1 TB | $2,500 |
| 2.3-2.4 | 2,000 | 10 TB | $10,000 |
| 3.1-3.5 | 1,000 | 5 TB | $5,000 |

### Team Requirements

| Role | Phase 1 | Phase 2 | Phase 3 |
|------|---------|---------|---------|
| ML Research | 0.5 FTE | 0.5 FTE | 1.0 FTE |
| Backend Eng | 0.2 FTE | 1.0 FTE | 0.3 FTE |
| Frontend Eng | 0 | 0.5 FTE | 0 |
| DevOps | 0.1 FTE | 0.5 FTE | 0.2 FTE |

---

## Milestones & Success Metrics

### 2026 Q1
- [ ] ArXiv preprint submitted
- [ ] API specification complete
- [ ] PyPhi validation showing r > 0.90

### 2026 Q2
- [ ] Journal submission
- [ ] Platform beta launch
- [ ] 3 ethics datasets integrated

### 2026 Q3
- [ ] 100 external model submissions
- [ ] Φ-capability correlation paper draft
- [ ] Community forum active

### 2026 Q4
- [ ] Paper accepted at target journal
- [ ] 1,000 registered platform users
- [ ] Real-time monitoring prototype

### 2027
- [ ] Symthaea cited in 10+ papers
- [ ] Industry adoption (1+ major AI lab)
- [ ] Standard reference for consciousness metrics

### 2028
- [ ] Regulatory recognition
- [ ] 10,000+ platform users
- [ ] Multi-agent consciousness framework

---

## Risk Mitigation

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| Paper rejection | High | Medium | Multiple target journals, strong reproducibility |
| Platform adoption failure | High | Medium | Partner with key researchers early |
| PyPhi correlation low | Medium | Low | Document approximation clearly, focus on trends |
| Funding shortage | High | Medium | Open source core, premium features |
| Competition | Medium | Medium | First-mover advantage, community building |

---

## Governance

### Open Source Commitment
- Core benchmarks: MIT License
- Datasets: CC-BY-4.0
- Platform: Open core model

### Benchmark Committee
- Academic representatives (2)
- Industry representatives (2)
- Community representatives (2)
- Rotating 2-year terms

### Decision Process
1. RFC for new benchmarks
2. 30-day comment period
3. Committee vote (2/3 majority)
4. 90-day implementation window

---

## Appendix: Key Dependencies

### External Datasets
- MMLU: https://huggingface.co/datasets/cais/mmlu
- ETHICS: https://github.com/hendrycks/ethics
- BBQ: https://github.com/nyu-mll/BBQ
- PyPhi: https://github.com/wmayner/pyphi

### Collaborators to Engage
- Giulio Tononi (IIT creator, UW-Madison)
- Christof Koch (Allen Institute)
- Murray Shanahan (DeepMind, GWT)
- Dan Hendrycks (ETHICS benchmark)

### Funding Sources
- NSF AI Institute grants
- Open Philanthropy
- Anthropic research grants
- EU Horizon Europe

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2026-01-04 | Initial roadmap |

---

*"Consciousness measurement is not just a scientific curiosity—it's the foundation for building AI systems we can trust."*
