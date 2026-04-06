# Governance

This document describes the governance model for Mycelix Praxis, including decision-making processes, roles, and responsibilities.

## Principles

1. **Transparency**: All decisions are documented and publicly visible
2. **Inclusivity**: Community input is valued and considered
3. **Meritocracy**: Contributions and expertise guide decision-making
4. **Accountability**: Decision-makers are accountable to the community
5. **Agility**: Fast decision paths for urgent issues, deliberate paths for complex changes

## Roles

### Contributors

Anyone who contributes to the project through code, documentation, design, testing, or community support.

**Responsibilities**:
- Follow the [Code of Conduct](CODE_OF_CONDUCT.md)
- Follow [contribution guidelines](CONTRIBUTING.md)
- Engage constructively in discussions

**Privileges**:
- Submit issues and pull requests
- Participate in discussions
- Propose features and improvements

### Core Developers

Active contributors who have demonstrated sustained commitment and expertise.

**Responsibilities**:
- Review pull requests
- Triage issues
- Mentor new contributors
- Maintain code quality
- Uphold project standards

**Privileges**:
- Write access to repository
- Vote on core developer nominations
- Participate in technical decision-making

**Nomination**: Proposed by current core developers, requires majority approval.

### Maintainers

Long-term contributors with deep project knowledge and commitment to the project's vision.

**Responsibilities**:
- All core developer responsibilities
- Release management
- Security response
- Governance decisions
- Protocol and architecture direction
- Community stewardship

**Privileges**:
- All core developer privileges
- Final approval on major changes
- Veto power (sparingly used, must be justified)
- Access to sensitive resources (keys, credentials, accounts)

**Nomination**: Proposed by current maintainers, requires 2/3 approval.

**Current Maintainers**:
- @Luminous-Dynamics/maintainers

## Decision-Making Process

We use a **tiered decision framework** based on impact and urgency.

### Fast Path (0-48 hours)

**When to use**: Security fixes, critical bugs, operational emergencies

**Process**:
1. Any maintainer can trigger fast path
2. Post issue with `priority: p0` label
3. Implement fix
4. Notify other maintainers (async)
5. Merge and deploy
6. Retrospective within 7 days

**Example**: Emergency patch for actively exploited vulnerability

### Normal Path (3-14 days)

**When to use**: Features, non-critical bugs, refactors, dependency updates

**Process**:
1. Open issue or PR
2. Discussion and design review
3. At least 1 maintainer approval
4. CI passes
5. Merge

**Example**: Adding new aggregation algorithm

### Slow Path (14+ days)

**When to use**: Protocol changes, governance changes, major architectural decisions, breaking changes

**Process**:
1. Create Architecture Decision Record (ADR) in `docs/adr/`
2. Open discussion issue or GitHub Discussion
3. Minimum 14-day comment period
4. Address feedback and iterate
5. Final decision by maintainers (majority or consensus)
6. Update ADR with decision and rationale
7. Implement

**Example**: Changing FL protocol message format, adding new VC schema

### Emergency Halt

Any maintainer can **halt a problematic change** if:
- Security risk discovered post-merge
- Critical bug affecting data integrity
- Violation of governance or code of conduct

**Process**:
1. Revert or disable feature
2. Post incident report
3. Follow appropriate decision path to resolve

## Decision Authority

| Decision Type | Authority | Approval Required |
|--------------|-----------|-------------------|
| Bug fix (minor) | Core dev | 1 maintainer |
| Feature (non-breaking) | Core dev | 1 maintainer |
| Feature (breaking) | Maintainer | Majority of maintainers |
| Protocol change | Maintainer | 2/3 of maintainers |
| Governance change | Maintainer | 2/3 of maintainers |
| Security response | Any maintainer | Notify others async |
| Core dev nomination | Core devs | Majority of maintainers |
| Maintainer nomination | Maintainers | 2/3 of maintainers |
| Emergency halt | Any maintainer | Retrospective required |

## Architecture Decision Records (ADRs)

For significant decisions, we use ADRs to document:
- **Context**: What is the issue we're facing?
- **Decision**: What did we decide?
- **Consequences**: What are the implications?
- **Alternatives**: What other options did we consider?
- **Status**: Proposed, Accepted, Deprecated, Superseded

See `docs/adr/` for templates and examples.

## Conflict Resolution

1. **Discussion**: Raise concern respectfully
2. **Mediation**: Request neutral maintainer to mediate
3. **Vote**: If consensus fails, maintainers vote (majority wins)
4. **Appeal**: Can appeal to full maintainer group (2/3 to overturn)
5. **Code of Conduct**: Behavioral issues follow [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)

## DAO Alignment

This governance model is **temporary** and focused on the code repository. As the project matures:

- **On-chain governance** will be introduced via the `dao_zome`
- **Community voting** on protocol changes and treasury allocation
- **Off-chain coordination** will remain for operational decisions (CI, releases)

This document will evolve to reflect the hybrid on-chain/off-chain model.

## Governance Scope

**This governance applies to**:
- Code in this repository
- Documentation and specifications
- Release process
- Maintainer and core dev roles
- Security response

**This governance does NOT apply to**:
- Course content and curricula (governed by DAOs)
- Credential standards (governed by communities)
- Individual node operation
- User data and privacy choices

## Amendments

This governance document can be amended via the **slow path**:
1. Propose changes in `docs/governance-proposals/`
2. Open discussion with 21-day comment period
3. Requires 2/3 maintainer approval
4. Update this document and announce changes

## Transparency

- **Decisions**: Documented in issues, PRs, and ADRs
- **Meetings**: Notes posted in Discussions (if/when we hold them)
- **Roadmap**: Public and tracked in GitHub Projects
- **Metrics**: Contributor stats, release cadence (public)

## Accountability

- **Maintainers**: Expected to respond to pings within 7 days
- **Core devs**: Expected to respond to review requests within 14 days
- **Inactivity**: Role removed after 6 months of inactivity (gracefully, with notice)
- **Removal**: Violation of Code of Conduct or abuse of privileges results in immediate removal

## Community Input

We welcome community feedback on governance:
- Open a [Discussion](https://github.com/Luminous-Dynamics/mycelix-praxis/discussions)
- Propose governance improvements via PR to this document
- Participate in roadmap planning

---

**Version**: 0.1.0
**Last Updated**: 2025-11-15
**Next Review**: 2026-05-15 (6 months)
