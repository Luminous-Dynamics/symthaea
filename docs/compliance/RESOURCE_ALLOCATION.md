# Resource Allocation Plan

**ISO 42001 A.2.4 Compliance** | Version: 1.0 | Date: 2026-03-08
Owner: Tristan Stoltz, Luminous Dynamics

---

## 1. Current Resource Profile

### Human Resources

| Role | Person | Allocation | Responsibilities |
|------|--------|-----------|-----------------|
| Lead Developer / System Architect | Tristan Stoltz | Full-time | Architecture, implementation, testing, compliance, operations |
| AI Pair Programming | Claude (Anthropic) | On-demand | Code review, implementation assistance, compliance documentation |

### Computational Resources

| Resource | Specification | Purpose |
|----------|--------------|---------|
| Development machine | NixOS 25.11, multi-core | Development, testing, CI |
| Ollama local inference | gemma3:4b, mistral:7b | Symthaea neural bridge integration |
| Holochain conductors | hc 0.6.x | Mycelix cluster testing |
| sccache | Shared compilation cache | Build acceleration across workspaces |

### Infrastructure Resources

| Resource | Provider | Purpose |
|----------|----------|---------|
| Git hosting | GitHub | Source control, CI/CD, issue tracking |
| Domain hosting | Various | luminousdynamics.org, atlas.luminousdynamics.io, nixforhumanity.org, mycelix.net |
| Supabase | Cloud | Terra Atlas production database |
| BWS | Bitwarden | Credential management |

## 2. Resource Adequacy Assessment

### Sufficient Resources

- **Computational**: Development hardware exceeds requirements for Rust compilation, HDC operations, and Holochain testing
- **Tooling**: NixOS flakes provide reproducible environments; sccache eliminates redundant compilation
- **AI assistance**: Claude provides effective force multiplication for code review, testing, and documentation
- **Testing infrastructure**: 12,000+ automated tests provide comprehensive coverage without dedicated QA

### Resource Constraints

| Constraint | Impact | Mitigation |
|-----------|--------|-----------|
| Single developer | Bus factor = 1; limited review diversity | Comprehensive documentation; automated CI; AI-assisted review |
| No dedicated security reviewer | Security review is self-performed | Automated clippy/audit in CI; OWASP-aware development practices |
| No dedicated compliance officer | Compliance is developer-maintained | Compliance dashboard in CI; quarterly self-review schedule |
| No external stakeholder panel | Limited value validation diversity | Planned: external feedback mechanism (see `EXTERNAL_FEEDBACK_PROTOCOL.md`) |

## 3. Resource Scaling Plan

### Phase 1: Current (Solo + AI)

- Maintain comprehensive test suites as primary quality gate
- Use CI compliance dashboard for continuous monitoring
- Quarterly self-assessment against compliance matrix
- Leverage AI pair programming for review diversity

### Phase 2: Small Team (2-4 developers)

Triggers: External funding, partnership, or community growth

- Assign dedicated compliance champion (rotating quarterly)
- Implement mandatory code review (no self-merge for safety-critical changes)
- Establish external advisory board for IEEE 7000 value validation
- Add dedicated security review for Class A changes (per `GOVERNANCE_CHARTER.md`)

### Phase 3: Organization (5+ developers)

- Dedicated compliance officer role
- Formal change advisory board for Class A decisions
- External security audit annually
- Stakeholder engagement program for value validation
- Dedicated QA with adversarial testing focus

## 4. Competency Requirements

| Competency | Current Level | Required Level | Gap |
|-----------|--------------|---------------|-----|
| Rust systems programming | Expert | Expert | None |
| Holochain/DHT architecture | Advanced | Advanced | None |
| Consciousness science (IIT/FEP) | Advanced | Advanced | None |
| AI compliance frameworks | Intermediate | Intermediate | None |
| Security engineering | Intermediate | Advanced | Partial — mitigated by automated tooling |
| Formal verification | Basic | Intermediate | Planned for Phase 3 |

## 5. Resource Review Schedule

| Activity | Frequency | Owner |
|----------|----------|-------|
| Resource adequacy self-assessment | Quarterly | Lead Developer |
| Competency gap analysis | Semi-annually | Lead Developer |
| Infrastructure capacity review | Quarterly | Lead Developer |
| Scaling trigger evaluation | Quarterly | Lead Developer |

---

*This plan addresses ISO 42001 A.2.4 by formalizing resource identification, adequacy assessment, constraint acknowledgment, and scaling strategy. Review quarterly or when team composition changes.*
