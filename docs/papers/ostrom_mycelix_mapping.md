# Ostrom's 8 Design Principles: Mapping to Mycelix

**Author**: Tristan Stoltz, Luminous Dynamics
**Date**: March 2026
**Version**: 1.0

## Abstract

Elinor Ostrom's eight design principles for governing the commons (1990) are the most empirically validated framework for evaluating community resource management systems. This document provides a rigorous code-level mapping of each principle to specific Mycelix zomes, data structures, and functions. We score each principle 0–5 on implementation completeness and identify gaps requiring future work.

## 1. Clearly Defined Boundaries

**Ostrom**: "Individuals or households who have rights to withdraw resource units from the common-pool resource must be clearly defined, as must the boundaries of the resource itself."

### Mycelix Implementation

**Identity boundaries** are defined by the `ConsciousnessTier` enum (`crates/mycelix-bridge-common/src/consciousness_profile.rs`):
- `Observer` (score < 0.25): Read-only access, zero governance weight
- `Participant` (0.25–0.45): Basic participation, 50% vote weight
- `Citizen` (0.45–0.60): Full participation, 75% vote weight
- `Steward` (0.60–0.80): Enhanced governance, 100% vote weight
- `Guardian` (0.80+): Full authority, 100% vote weight

Each tier creates an explicit membership boundary with clear inclusion/exclusion criteria.

**Resource pool boundaries** are enforced by DNA-level isolation. The commons cluster is split into `commons_land` (property, housing, water, transport) and `commons_care` (food, care, mutual aid) DNAs. Cross-DNA communication requires explicit bridge dispatch (`mycelix-commons/zomes/commons-bridge/coordinator/src/lib.rs`).

**Bootstrap boundaries**: Communities under 5 members receive temporary bootstrap credentials (`BOOTSTRAP_COMMUNITY_THRESHOLD = 5` in `consciousness_thresholds.rs`) with 1-hour TTL, preventing premature governance participation.

**Dispatch allowlists**: The `ALLOWED_ZOMES` constant in each bridge coordinator hard-codes which zome names can be dispatched to, creating a compile-time security boundary.

**Gap**: Boundaries are identity-based, not resource-based. No explicit mechanism defines "this water system serves these 50 households." Resource-level boundaries are implicit in DNA membership rather than explicitly modeled.

**Score: 4/5**

## 2. Proportional Equivalence Between Benefits and Costs

**Ostrom**: "Rules specifying the amount of resource units that a user is allocated are related to local conditions and to rules requiring labor, material, or money."

### Mycelix Implementation

**Progressive vote weighting**: `ConsciousnessTier::vote_weight_bp()` returns 0 (Observer), 5000 (Participant), 7500 (Citizen), 10000 (Steward/Guardian) basis points. Higher engagement earns more governance influence.

**Profile weighting**: `ConsciousnessProfile::combined_score()` weights: identity 25%, reputation 25%, community 30%, engagement 20%. Contribution tracks with influence.

**Quadratic voting**: The voting zome (`mycelix-governance/zomes/voting/`) supports `QuadraticVoteCast` with `credits_spent` and `vote_strength`, implementing Weyl (2017) cost-based proportionality where voice is proportional to the square root of credits spent.

**Care timebanking**: The `care_timebank` zome in the commons cluster tracks hours given and received, creating a direct time-for-time reciprocity mechanism.

**FL gradient weighting**: Consciousness-gated federated learning (`consciousness_thresholds.rs`) dampens gradient contributions below 0.3 consciousness (0.3x), boosts above 0.6 (1.5x), and vetoes below 0.1. Higher-quality contributions receive more influence over shared model updates.

**Gap**: No explicit mechanism tracks resource extraction vs. provision at the individual level. Proportionality is governance-weighted, not resource-use weighted. Ostrom emphasizes rules about resource *appropriation* proportional to *provision*, which is more granular than tier-based vote weighting.

**Score: 3/5**

## 3. Collective-Choice Arrangements

**Ostrom**: "Most individuals affected by the operational rules can participate in modifying the operational rules."

### Mycelix Implementation

**Proposal system**: `mycelix-governance/zomes/proposals/` supports 10 proposal types (AddRule, ModifyRule, Emergency, GeneralDecision, BudgetAllocation, MembershipChange, etc.) with a full lifecycle: Draft → Discussion → Voting → Passed/Failed → Implemented.

**Voting methods**: Both `VotingMethod::Majority` and `VotingMethod::Consensus` in mutual aid governance, plus Phi-weighted and quadratic voting in the main voting zome.

**Constitutional amendment process**: `mycelix-governance/zomes/constitution/` requires elevated consciousness gates (0.6 for constitutional changes). Charter versioning prevents silent overwrites — amendments create new versions with full audit trail.

**Consciousness gating for participation**: `GovernanceActionType::ProposalSubmission` requires consciousness score ≥ 0.3, `Voting` requires ≥ 0.4, `Constitutional` requires ≥ 0.6. This ensures those most engaged can participate proportionally.

**Council system**: Holonic councils (`mycelix-governance/zomes/councils/`) with configurable quorum and supermajority thresholds. `require_council_member()` ensures only designated members participate in council decisions.

**Score: 5/5**

## 4. Monitoring

**Ostrom**: "Monitors, who actively audit common-pool resource conditions and appropriator behavior, are accountable to the appropriators or are the appropriators."

### Mycelix Implementation

**Audit trails**: `log_governance_gate()` and `get_governance_audit_trail()` in bridge coordinators log every governance gate decision. `should_audit()` always logs rejections and high-tier actions, samples 10% of basic actions.

**Consciousness snapshots**: `record_consciousness_snapshot()` creates timestamped records of each agent's consciousness state, linked via DHT. This provides a historical record of who held what governance capacity and when.

**Rate limiting**: `enforce_rate_limit()` in both commons and civic bridges tracks dispatch frequency per agent (100 calls/60 seconds), creating auditable link records. Exceeded limits are logged and rejected.

**Query auditability**: Both `query_commons()` and `query_civic()` store queries on the DHT with domain/type/timestamp, then link to resolution results. Every information request is recorded.

**Evidence chain of custody**: `CustodyEvent` in justice-cases integrity tracks Submitted, Accessed, Copied, Sealed, and Unsealed actions on evidence, maintaining forensic-quality audit trails.

**Gap**: Monitoring is primarily system-level (governance actions, dispatches). No explicit resource-use monitoring (e.g., water extraction meters, harvest tracking). Monitors are the system itself (DHT validators), not designated human monitors — this differs from Ostrom's emphasis on accountable human monitors.

**Score: 4/5**

## 5. Graduated Sanctions

**Ostrom**: "Appropriators who violate operational rules are likely to be assessed graduated sanctions (depending on the seriousness and context of the offense) by other appropriators, by officials accountable to the appropriators, or by both."

### Mycelix Implementation

**Consciousness tier demotion**: Users whose reputation, community trust, or engagement drops will automatically lose tier privileges. This is implicitly graduated: first they lose Constitutional access (requires 0.6), then Voting (0.4), then Proposal submission (0.3), then all participation (Observer tier).

**Enforcement action types**: `EnforcementActionType` in justice-enforcement integrity provides explicit graduated options: `Notification` (mildest), `ReputationUpdate`, `AccessRevocation`, `AssetFreeze`, `FundsTransfer` (most severe), plus `ManualRequired` and `CrossHappAction`.

**Case severity levels**: `CaseSeverity` enum (Minor, Moderate, Serious, Critical) allows proportionate responses to different violation types.

**FL contribution dampening**: Gradient contributions are dampened (0.3x weight) below 0.3 consciousness, boosted (1.5x) above 0.6, and vetoed entirely below 0.1. This creates graduated influence based on behavioral history.

**Gap**: No explicit "graduated sanctions schedule" mapping specific violations to specific penalty levels. The enforcement types exist as building blocks, but there is no codified escalation ladder (e.g., "first offense = warning, second = temporary suspension, third = expulsion"). The system provides tools but not a mandatory graduated policy.

**Score: 3/5**

## 6. Conflict-Resolution Mechanisms

**Ostrom**: "Appropriators and their officials have rapid access to low-cost local arenas to resolve conflicts among appropriators or between appropriators and officials."

### Mycelix Implementation

**Three-tier justice system**: `CasePhase` enum implements Filed → Negotiation → Mediation → Arbitration → Appeal → Enforcement → Closed. This is a textbook graduated conflict resolution pathway matching Ostrom's emphasis on escalation from low-cost to formal mechanisms.

**Mediation**: `Mediation` struct with session tracking, settlement proposals, and status lifecycle. Mediation is the default first-response mechanism.

**Arbitration**: `Arbitration` struct with panel formation and four selection methods: `Random`, `ReputationWeighted`, `MutualAgreement`, `CommunityElected`. Formal `Decision` records include remedies.

**Restorative justice**: `RestorativeCircle` with five role types: facilitator, harm-doer, harm-receiver, community-member, and support-person. Session tracking and agreement records enable community-centered resolution.

**Appeals**: `Appeal` struct with five grounds: ProceduralError, NewEvidence, LegalError, ExcessiveRemedy, BiasOrConflict.

**Remedies**: `RemedyType` enum includes Compensation, Restitution, Injunction, CommunityService, RestorativeProcess, PublicApology, and BehavioralRestriction.

**Score: 5/5**

## 7. Minimal Recognition of Rights to Organize

**Ostrom**: "The rights of appropriators to devise their own institutions are not challenged by external governmental authorities."

### Mycelix Implementation

**Self-governance via constitution**: The charter system allows communities to define their own rules via `create_charter()` without any external authority check. Communities are sovereign over their own governance rules.

**Council spawning**: The `can_spawn_children` flag on councils allows holonic governance — sub-communities can self-organize without top-level permission.

**Bootstrap credentials**: Communities under 5 members get bootstrap credentials with minimal requirements (`BOOTSTRAP_MIN_IDENTITY = 0.25`), enabling self-organization before full governance infrastructure is established.

**Holochain sovereignty**: As a Holochain application, Mycelix is inherently autonomous — there is no centralized server that could enforce external authority over community governance.

**Gap**: Since Mycelix operates on Holochain, there is no "external authority" by construction. However, there is also no explicit mechanism for interfacing with external legal jurisdictions or asserting self-governance rights vis-à-vis state actors. The `CaseContext.jurisdiction` field exists but is optional and unstructured.

**Score: 3/5**

## 8. Nested Enterprises

**Ostrom**: "Appropriation, provision, monitoring, enforcement, conflict resolution, and governance activities are organized in multiple layers of nested enterprises."

### Mycelix Implementation

**Cluster architecture**: 7 clusters (commons, civic, hearth, governance, identity, personal, attribution) with cross-cluster bridges via `CallTargetCell::OtherRole`. Each cluster manages its own domain while coordinating through typed bridge dispatch.

**Sub-cluster nesting**: Commons is split into `commons_land` and `commons_care` DNAs, each with its own bridge instance, nested within the unified hApp. The civic cluster similarly contains separate justice, emergency, and media domains.

**Holonic councils**: Council hierarchy with `parent_council_id`, `can_spawn_children`, and `max_delegation_depth` — explicit fractal governance where councils can spawn sub-councils with delegated authority.

**Domain-specific governance**: `mutualaid-governance` runs its own proposals/voting/rules within the mutual aid domain, nested under the broader governance cluster. Each domain has autonomy over its operational rules while participating in cluster-wide governance.

**Consciousness gating across tiers**: The 5-tier system (Observer → Guardian) creates nested authority levels, each containing the permissions of the level below.

**Cross-cluster dispatch**: The bridge pattern (commons-bridge, civic-bridge, hearth-bridge, etc.) enables coordination between nested enterprises without tight coupling, matching Ostrom's emphasis on coordination across governance levels.

**Score: 5/5**

## Summary

| Principle | Score | Strength | Key Gap |
|-----------|-------|----------|---------|
| 1. Clearly Defined Boundaries | 4/5 | Multi-layer identity + DNA boundaries | Resource-level boundaries not modeled |
| 2. Proportional Equivalence | 3/5 | Quadratic voting, tier weighting | No per-resource extraction tracking |
| 3. Collective-Choice Arrangements | 5/5 | 10 proposal types, constitutional amendments | — |
| 4. Monitoring | 4/5 | Audit trails, rate limits, custody chains | No resource-use monitoring |
| 5. Graduated Sanctions | 3/5 | Enforcement types, tier demotion | No codified escalation schedule |
| 6. Conflict Resolution | 5/5 | Three-tier justice + restorative circles | — |
| 7. Rights to Organize | 3/5 | Inherently autonomous (Holochain) | No external jurisdiction interface |
| 8. Nested Enterprises | 5/5 | Fractal clusters, holonic councils | — |
| **Overall** | **32/40 (80%)** | | |

## Discussion

Mycelix achieves strong implementation on 4 of 8 principles (3, 6, 8 scoring 5/5; 1, 4 scoring 4/5). The architecture's fractal cluster design (P8), comprehensive justice system (P6), and rich governance tooling (P3) are standout strengths.

The three weaker principles (P2, P5, P7 at 3/5) share a common root cause: Mycelix was designed as governance infrastructure, not as a resource management system. Ostrom's framework emphasizes *resource*-level proportionality, monitoring, and sanctions — tracking who takes how much water, how much fish, how many trees. Mycelix tracks governance *participation* and *behavior*, not resource *extraction*.

### Recommendations

1. **P2 (Proportionality)**: Add resource-use tracking to domain zomes (water meter readings, harvest records, time contributions) and weight governance influence by provision as well as by tier.
2. **P5 (Graduated Sanctions)**: Create an explicit `SanctionSchedule` entry type mapping violation categories to mandatory minimum/maximum penalties, with community override provisions.
3. **P7 (Rights to Organize)**: Add a `Jurisdiction` struct modeling the interface between Mycelix communities and external legal systems, including dispute escalation to external arbitration.

## References

- Ostrom, E. (1990). *Governing the Commons: The Evolution of Institutions for Collective Action*. Cambridge University Press.
- Ostrom, E. (2005). *Understanding Institutional Diversity*. Princeton University Press.
- Weyl, E.G. (2017). The robustness of quadratic voting. *Public Choice*, 172(1-2), 75-107.
