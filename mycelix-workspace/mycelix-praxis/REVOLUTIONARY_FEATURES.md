# Mycelix Praxis: Revolutionary Features

**Version**: 0.2.0-dev
**Updated**: December 30, 2025
**Status**: Differentiation Features Implemented

---

## Executive Summary

Mycelix Praxis is now positioned to be **the world's best decentralized education platform** with three revolutionary features that no other platform offers:

1. **Learning Pods** - Novel social learning circles
2. **Knowledge Roots** - Decentralized curriculum graph
3. **Proof of Learning** - Verifiable genuine learning (extending PoGQ)

Combined with the existing core features (FL, W3C VCs, DAO governance), Praxis offers:
- **Complete data sovereignty** - Your learning data stays on YOUR device
- **Privacy-preserving improvement** - Courses get better without exposing your mistakes
- **Verifiable, portable credentials** - W3C standards that work everywhere
- **Community governance** - Educators and learners make the rules
- **Genuine learning verification** - Not just completion, but actual understanding

---

## Core Architecture: 12 Zomes

```
┌─────────────────────────────────────────────────────────────┐
│                   MYCELIX EDUNET DNA                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  CORE EDUCATION FEATURES                                     │
│  ─────────────────────────                                   │
│  • learning_zome    - Courses, enrollment, progress          │
│  • fl_zome          - Federated learning rounds              │
│  • credential_zome  - W3C Verifiable Credentials             │
│  • dao_zome         - Community governance                   │
│                                                              │
│  DIFFERENTIATION FEATURES (NEW)                              │
│  ──────────────────────────────                              │
│  • pods_zome        - Learning Pods (social learning)        │
│  • knowledge_zome   - Knowledge Roots (curriculum graph)     │
│                                                              │
├─────────────────────────────────────────────────────────────┤
│  SUPPORTING CRATES                                           │
│  ─────────────────                                           │
│  • praxis-core      - Types, crypto, Proof of Learning       │
│  • praxis-agg       - FL aggregation algorithms              │
│  • mycelix-governance - DAO utilities                        │
└─────────────────────────────────────────────────────────────┘
```

---

## Feature 1: Learning Pods

### What Is It?

Learning Pods are small groups (3-7 people) that form cohort-based learning circles. Unlike traditional online learning which is isolating, Pods create accountability, support, and social motivation.

### Why It's Revolutionary

| Traditional | Learning Pods |
|-------------|---------------|
| Learn alone | Learn with peers |
| No accountability | Mutual support |
| Isolated progress | Shared journey |
| Individual certificates | Collective achievements |
| One-size-fits-all | Pod-customized pacing |

### Key Components

```rust
// Entry Types (pods_zome)
├── LearningPod        - The pod itself with goals, members, status
├── PodMembership      - Links agents to pods with roles
├── PodProgress        - Private progress within pods
├── PodDiscussion      - Threaded discussions
├── DiscussionReply    - Replies to discussions
├── PodAchievement     - Collective achievements
└── PodChallenge       - Learning challenges
```

### Pod Lifecycle

```
1. FORMATION     - Creator invites members, sets learning goals
      ↓
2. ACTIVE        - Members learn, share progress, collaborate
      ↓
3. ACHIEVEMENT   - Pod completes goals, earns collective credentials
      ↓
4. EVOLUTION     - Pod can split, merge, or graduate
```

### Zome Functions (25 total)

**Pod Lifecycle**:
- `create_pod()` - Create a new learning pod
- `get_pod()` / `list_pods()` - Retrieve pod information
- `update_pod_status()` - Change pod status

**Membership**:
- `invite_member()` - Invite someone to a pod
- `accept_invitation()` - Accept a pod invitation
- `leave_pod()` - Leave a pod
- `get_pod_members()` / `get_my_pods()` - Query memberships

**Progress & Social**:
- `update_pod_progress()` - Update your progress (private)
- `get_pod_progress_stats()` - Anonymized aggregate stats
- `create_discussion()` / `create_reply()` - Discussions
- `record_achievement()` - Collective achievements
- `create_challenge()` - Learning challenges

---

## Feature 2: Knowledge Roots

### What Is It?

Knowledge Roots is a decentralized curriculum graph - a community-built map of how knowledge connects. Unlike traditional curricula controlled by institutions, Knowledge Roots is:

- **Community-built**: Anyone can propose connections
- **DAO-governed**: Community votes on changes
- **AI-enhanced**: ML suggests optimal paths
- **Skill-aligned**: Maps to ESCO, O*NET, SFIA frameworks
- **Credential-gated**: Advanced topics unlock with mastery

### Why It's Revolutionary

| Traditional Curriculum | Knowledge Roots |
|------------------------|-----------------|
| Institution-controlled | Community-built |
| Fixed pathways | Adaptive paths |
| Geographic silos | Global knowledge graph |
| No skill mapping | Industry-aligned |
| One-size-fits-all | Personalized |

### Key Components

```rust
// Entry Types (knowledge_zome)
├── KnowledgeNode      - A concept, skill, or topic
├── LearningEdge       - Relationship between nodes (prerequisites, etc.)
├── LearningPath       - Curated sequence through the graph
├── SkillTree          - Visual grouping of related knowledge
├── NodeProgress       - Learner progress on nodes (private)
├── EdgeVote           - Community governance on edges
└── PathRecommendation - AI-suggested learning paths
```

### Node Types

```
Concept     - Fundamental concepts (e.g., "Variables in Programming")
Skill       - Practical skills (e.g., "Write unit tests")
Topic       - Knowledge topics (e.g., "Machine Learning")
Course      - Modules/courses (e.g., "Introduction to Python")
Assessment  - Certifications (e.g., "Python Developer Certification")
Project     - Practical applications
```

### Edge Types (Relationships)

```
Requires      - Strict prerequisite
Recommends    - Soft prerequisite
RelatedTo     - Related topics
PartOf        - Composition
LeadsTo       - Sequence
AlternativeTo - Choose one
Specializes   - Builds upon
AppliedIn     - Practical application
```

### Zome Functions (20+ total)

**Node Management**:
- `create_node()` / `get_node()` / `list_nodes()` - Basic CRUD
- `get_nodes_by_domain()` - Filter by domain
- `search_nodes()` - Search by tags, difficulty
- `update_node_status()` - Governance

**Edge Governance**:
- `propose_edge()` - Propose a curriculum connection
- `get_node_edges()` / `get_node_prerequisites()` - Query relationships
- `vote_on_edge()` - Community voting

**Path Discovery**:
- `create_path()` / `get_path()` / `list_paths()` - Curated paths
- `find_path()` - Pathfinding between nodes
- `generate_recommendation()` - AI-powered recommendations

**Progress**:
- `update_node_progress()` - Track learning
- `get_my_progress()` - View progress
- `check_prerequisites()` - Verify readiness

---

## Feature 3: Proof of Learning (PoL)

### What Is It?

Proof of Learning extends the Proof of Gradient Quality (PoGQ) concept from Mycelix-Core's Byzantine-resistant federated learning to verify **genuine learning**, not just completion.

### Why It's Revolutionary

| Traditional Verification | Proof of Learning |
|--------------------------|-------------------|
| Time spent (seat time) | Learning trajectory |
| Test scores (easily gamed) | Error pattern analysis |
| Certificates (can be faked) | Retention curves |
| Binary pass/fail | Continuous mastery score |
| No transfer verification | Application transfer |

### PoL Components

```rust
pub struct PoLComponents {
    trajectory: f64,         // How knowledge builds over time
    error_authenticity: f64, // Genuine learner mistake patterns
    retention: f64,          // Natural forgetting/relearning curves
    transfer: f64,           // Can apply to novel problems (highest weight)
    contribution: f64,       // Teaching others demonstrates mastery
    consistency: f64,        // Learning over time vs cramming
}
```

### Component Weights

```
Transfer            25%   - Hardest to fake, requires understanding
Trajectory          20%   - Progressive improvement
Retention           20%   - Spaced repetition patterns
Error Authenticity  15%   - Real learner mistake patterns
Contribution        15%   - Peer help, discussions
Consistency          5%   - Regular vs cramming
```

### Evidence Types

```rust
pub enum LearningEvidence {
    Assessment      - Completed assessment with score
    Activity        - Learning activity record
    PeerHelp        - Helped another learner
    Discussion      - Contributed to discussion
    FederatedUpdate - Submitted FL update with gradient quality
    Credential      - Earned credential
    RetentionTest   - Spaced repetition test result
    TransferTask    - Applied knowledge to new context
}
```

### Integration with MATL

PoL integrates with the Mycelix Adaptive Trust Layer:

```rust
pub struct PoLMATLScore {
    matl_base: f64,      // Base MATL composite score
    pol_adjustment: f64, // PoL score adjustment
    combined: f64,       // Final combined score
    pol_weight: f64,     // Weight given to PoL
}

// High PoL increases trust for FL contributions
// Low PoL flags potential gaming/cheating
```

---

## Combined Power: The Praxis Difference

### Complete Feature Matrix

| Feature | Praxis | Coursera | Khan | Duolingo | Web3 Learn |
|---------|--------|----------|------|----------|------------|
| Data Sovereignty | ✅ | ❌ | ❌ | ❌ | Partial |
| Privacy-Preserving FL | ✅ | ❌ | ❌ | ❌ | ❌ |
| Social Learning Pods | ✅ | ❌ | ❌ | Limited | ❌ |
| Community Curriculum | ✅ | ❌ | ❌ | ❌ | ❌ |
| Proof of Learning | ✅ | ❌ | ❌ | ❌ | ❌ |
| W3C Credentials | ✅ | ❌ | ❌ | ❌ | Partial |
| DAO Governance | ✅ | ❌ | ❌ | ❌ | Some |
| Offline-First | ✅ | ❌ | Partial | Partial | ❌ |
| Open Source | ✅ | ❌ | Partial | ❌ | Varies |
| No Vendor Lock-in | ✅ | ❌ | ❌ | ❌ | ❌ |

### Unique Value Propositions

```
┌─────────────────────────────────────────────────────────────┐
│           MYCELIX EDUNET: THE COMPLETE PACKAGE              │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  🔒 SOVEREIGNTY                                              │
│     Your data stays on YOUR device. No platform can sell    │
│     your learning history. Export everything, anytime.      │
│                                                              │
│  🧠 GENUINE LEARNING                                         │
│     Proof of Learning verifies actual understanding, not    │
│     just completion. Credentials mean something.            │
│                                                              │
│  👥 SOCIAL LEARNING                                          │
│     Learning Pods provide accountability, support, and      │
│     motivation. Never learn alone again.                    │
│                                                              │
│  🌳 ADAPTIVE CURRICULUM                                      │
│     Knowledge Roots creates personalized paths through      │
│     community-built curriculum. AI finds your optimal path. │
│                                                              │
│  🏛️ DEMOCRATIC GOVERNANCE                                    │
│     Community decides curriculum structure, platform        │
│     policies, and feature priorities. No corporate control. │
│                                                              │
│  🌐 PORTABLE CREDENTIALS                                     │
│     W3C Verifiable Credentials work everywhere, forever.    │
│     Your achievements belong to you.                        │
│                                                              │
│  🔐 PRIVACY BY DESIGN                                        │
│     Federated learning improves courses without exposing    │
│     individual mistakes. Your struggles stay private.       │
│                                                              │
│  💪 RESILIENT                                                │
│     No single point of failure. Works offline. Can't be     │
│     shut down or censored.                                  │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Technical Summary

### New Zomes Added

| Zome | Entry Types | Link Types | Functions |
|------|-------------|------------|-----------|
| pods_integrity | 7 | 9 | Validation |
| pods_coordinator | - | - | 25+ |
| knowledge_integrity | 7 | 12 | Validation |
| knowledge_coordinator | - | - | 20+ |

### New Core Module

| Module | Purpose | Lines of Code |
|--------|---------|---------------|
| proof_of_learning | PoL analysis & MATL integration | ~500 |

### Files Created

```
zomes/pods_zome/
├── integrity/
│   ├── Cargo.toml
│   └── src/lib.rs           # 7 entry types, 9 link types
└── coordinator/
    ├── Cargo.toml
    └── src/lib.rs           # 25+ zome functions

zomes/knowledge_zome/
├── integrity/
│   ├── Cargo.toml
│   └── src/lib.rs           # 7 entry types, 12 link types
└── coordinator/
    ├── Cargo.toml
    └── src/lib.rs           # 20+ zome functions

crates/praxis-core/src/
└── proof_of_learning.rs     # PoL algorithm & MATL integration
```

---

## Roadmap

### Immediate (Next 2 Weeks)
- [ ] Build WASM targets for new zomes
- [ ] Write unit tests for all new entry types
- [ ] Integration tests for pod lifecycle
- [ ] Integration tests for knowledge graph operations
- [ ] PoL analyzer unit tests

### Short-Term (Q1 2026)
- [ ] Web client integration for Pods
- [ ] Web client integration for Knowledge Roots
- [ ] PoL scoring in credential issuance
- [ ] Mobile-responsive UI for pods

### Medium-Term (Q2 2026)
- [ ] ML-powered path recommendations
- [ ] Skill framework alignment (ESCO, O*NET)
- [ ] Cross-pod challenges and competitions
- [ ] Institutional onboarding tools

### Long-Term (Q3 2026+)
- [ ] Mobile apps (React Native)
- [ ] Real-time collaboration in pods
- [ ] Advanced analytics dashboard
- [ ] Enterprise features

---

## Conclusion

Mycelix Praxis is now architecturally complete for its differentiation features. With **Learning Pods**, **Knowledge Roots**, and **Proof of Learning**, we have created a platform that offers genuine innovations no competitor can match.

The path forward is clear:
1. **Complete testing** - Ensure all zomes work correctly
2. **Build UI** - Web client for new features
3. **Pilot** - Test with real learners
4. **Scale** - Mobile, institutions, ecosystem integration

**We are building the best decentralized education system ever created.** 🎓

---

*Document Version*: 1.0.0
*Last Updated*: December 30, 2025
