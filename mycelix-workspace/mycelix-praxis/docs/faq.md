# Frequently Asked Questions (FAQ)

Common questions about Mycelix Praxis, federated learning, and contributing.

## Table of Contents

- [General](#general)
- [Architecture & Technology](#architecture--technology)
- [Privacy & Security](#privacy--security)
- [Federated Learning](#federated-learning)
- [Credentials & Governance](#credentials--governance)
- [Development & Contributing](#development--contributing)
- [Roadmap & Future](#roadmap--future)

---

## General

### What is Praxis?

Praxis is a **decentralized education platform** that combines Holochain's agent-centric architecture with federated learning (FL) to enable privacy-preserving, personalized learning experiences. Think of it as:
- **Wikipedia** (community-driven content) +
- **Khan Academy** (personalized learning) +
- **Blockchain** (verifiable credentials) −
- **Your data on their servers** (everything stays local)

### Why does this exist?

Traditional online learning platforms have problems:
1. **Data silos**: Your learning data is locked in proprietary systems
2. **No portability**: You can't take your credentials elsewhere
3. **Privacy concerns**: Platforms profit from your learning behavior
4. **Centralized control**: Platforms decide what you can learn

Praxis solves this by:
- Keeping your data **on your device**
- Using **W3C Verifiable Credentials** for portable achievements
- Enabling **collaborative AI** without sharing raw data
- Giving **communities** control over curricula via DAOs

### Who is this for?

**Learners** who want:
- Privacy-preserving personalized education
- Portable, verifiable credentials
- Community-driven content

**Educators** who want:
- Tools to create and share courses
- Fair compensation models
- No platform lock-in

**Researchers** who want:
- Privacy-preserving ML on education data
- Decentralized infrastructure
- Open protocols

**Developers** who want:
- To build on Holochain
- To work on federated learning
- To contribute to open education

### Is this production-ready?

**No.** We're currently at **v0.1.0-alpha**:
- ✅ Core libraries work
- ✅ Architecture documented
- ✅ Examples available
- ⏳ Holochain zomes not implemented yet
- ⏳ Web client doesn't connect to conductor yet
- ⏳ No security audit

**Timeline**: v1.0.0 targeted for Q3 2026 (see [ROADMAP.md](../ROADMAP.md))

### What's the license?

**Apache-2.0** - permissive open source license.

You can:
- ✅ Use commercially
- ✅ Modify and distribute
- ✅ Use for private projects
- ✅ Grant patent rights

You must:
- Include the license
- State changes made
- Include copyright notice

See [LICENSE](../LICENSE) for details.

---

## Architecture & Technology

### Why Holochain instead of blockchain?

**Blockchain** (like Ethereum):
- ❌ Global consensus required (slow, expensive)
- ❌ All data public by default
- ❌ Energy-intensive (proof of work)
- ❌ Scalability limits

**Holochain**:
- ✅ Agent-centric (no global consensus needed)
- ✅ Privacy by default (data stays local unless shared)
- ✅ Energy-efficient (no mining)
- ✅ Horizontally scalable (more agents = more capacity)

For education, we need:
- **Privacy**: Learning data shouldn't be public
- **Scalability**: Millions of learners
- **Speed**: Real-time interactions
- **Offline support**: Learn anywhere

Holochain enables all of this.

### What's the difference between `praxis-core` and `praxis-agg`?

**`praxis-core`** (`crates/praxis-core/`):
- **Purpose**: Core types, crypto, provenance
- **What it provides**:
  - Types: `RoundId`, `ModelId`, `PrivacyParams`
  - Crypto: BLAKE3 hashing, commitments
  - Provenance: `ModelProvenance`, `ProvenanceChain`
- **Used by**: All zomes, aggregation crate

**`praxis-agg`** (`crates/praxis-agg/`):
- **Purpose**: Robust aggregation algorithms for FL
- **What it provides**:
  - Trimmed mean (default)
  - Median (max robustness)
  - Weighted mean
  - L2 norm clipping
- **Used by**: FL zome for gradient aggregation

**Analogy**: `praxis-core` is the foundation, `praxis-agg` is a specialized tool built on it.

### What are "zomes"?

**Zome** = **Z**ygote h**ome**ostasis (Holochain's term for modules)

Think of zomes as:
- **Microservices** for Holochain apps
- **Smart contracts** (but agent-centric, not blockchain)
- **Modules** that define data types and validation rules

Praxis has 4 zomes:
1. **`learning_zome`**: Courses, progress, activities
2. **`fl_zome`**: Federated learning rounds
3. **`credential_zome`**: W3C Verifiable Credentials
4. **`dao_zome`**: Governance proposals and votes

Each zome:
- Defines entry types (like database schemas)
- Validates entries (ensures data integrity)
- Exposes functions (like API endpoints)

### Where's the actual Holochain DNA?

**Not implemented yet.** v0.1.0 focuses on:
- ✅ Core Rust libraries (`praxis-core`, `praxis-agg`)
- ✅ Zome data structures (types, not HDK implementation)
- ✅ Web client UI
- ✅ Documentation and examples

**v0.2.0** (Q1 2026) will add:
- Holochain DNA manifest
- HDK (Holochain Development Kit) integration
- Actual zome functions (create, read, update)
- Conductor integration

**Why this order?** Get the architecture right first, then implement.

### Why doesn't the web client connect to Holochain yet?

Same reason - we're in v0.1.0-alpha. The web client currently:
- ✅ Has UI components
- ✅ Shows mock data
- ⏳ Will connect to conductor in v0.2.0

This lets us:
1. Design the UX first
2. Get feedback on the interface
3. Implement zomes in parallel
4. Integrate when both sides are ready

---

## Privacy & Security

### Is my learning data private?

**Yes.** Your data stays on your device:

**Private (never leaves your device)**:
- ❌ Quiz answers
- ❌ Exercise submissions
- ❌ Full learning history
- ❌ Raw training data

**Shared (only what you choose)**:
- ✅ Course enrollments (public by default)
- ✅ Gradient updates for FL (only commitments public)
- ✅ Credentials (you control disclosure)

**Never shared**:
- Raw learning content interactions
- Specific answers or mistakes
- Time spent on each lesson

See [Privacy Model](privacy.md) for details.

### How does federated learning protect my privacy?

Federated learning (FL) trains AI models **without seeing your data**:

**Traditional ML**:
```
Your device → Upload data → Central server → Train model
              ❌ Your data exposed!
```

**Federated Learning**:
```
Your device → Train locally → Upload gradients → Aggregate
              ✅ Only gradients shared, not data!
```

Plus, we add:
1. **Gradient clipping**: Bounds what any one person can contribute
2. **Commitments**: Only hashes published initially
3. **Robust aggregation**: Removes outliers (trimmed mean/median)
4. **Optional DP**: Add noise for formal privacy

See [Protocol](protocol.md) for technical details.

### Can the coordinator see my gradients?

**Kind of.** The coordinator:
- ✅ Sees clipped gradients (bounded contribution)
- ✅ Sees aggregated statistics (median loss, etc.)
- ❌ Doesn't see your raw training data
- ❌ Doesn't see which data points you have

**Future improvement** (v2.0): Secure aggregation using multi-party computation (MPC) so the coordinator can't see individual gradients at all.

### What if someone tries to poison the model?

We have multiple defenses:

1. **Gradient clipping**: Attacker can't submit extreme values
2. **Trimmed mean**: Top/bottom 10% removed (tolerates up to 10% malicious)
3. **Median**: Even more robust (50% breakdown point)
4. **Validation**: Coordinator tests model on holdout set
5. **Reputation** (future): Weight by historical accuracy

See [Threat Model](threat-model.md) for full analysis.

### How do I report a security vulnerability?

**Do NOT** open a public issue. Instead:

1. Email **security@mycelix.org**
2. Or use [GitHub Security Advisories](https://github.com/Luminous-Dynamics/mycelix-praxis/security/advisories)

We'll:
- Acknowledge within 48 hours
- Provide updates every 14 days
- Credit you in release notes (unless you prefer anonymity)
- Follow a 90-day embargo

See [SECURITY.md](../SECURITY.md) for full policy.

---

## Federated Learning

### How does a federated learning round work?

**6 phases** (see [Protocol](protocol.md) for details):

1. **DISCOVER** (24h): Coordinator announces round
2. **JOIN** (variable): Participants signal intent
3. **ASSIGN** (6h): Coordinator selects participants
4. **UPDATE** (48h): Participants train locally, submit gradients
5. **AGGREGATE** (2h): Coordinator combines gradients
6. **RELEASE** (instant): New model published

**Total**: ~72-80 hours per round

### What's "clipping" and why is it important?

**Clipping** = Bounding the L2 norm of gradients

**Without clipping**:
```
Gradient from honest user: [0.1, 0.2, 0.3]  (norm ≈ 0.37)
Gradient from attacker:    [100, 200, 300] (norm ≈ 374) ❌ Huge!
```

**With clipping** (max norm = 1.0):
```
Gradient from honest user: [0.1, 0.2, 0.3]  (norm ≈ 0.37) ✅ Unchanged
Gradient from attacker:    [0.27, 0.53, 0.80] (scaled to norm 1.0) ✅ Bounded
```

This:
- Limits attacker influence
- Protects privacy (bounds information leakage)
- Enables differential privacy (prerequisite for DP)

### What aggregation methods are supported?

1. **Trimmed Mean** (default):
   - Remove top/bottom 10% per dimension
   - Average the rest
   - Tolerates up to 10% malicious participants

2. **Median**:
   - Take median per dimension
   - Tolerates up to 50% malicious participants
   - Slower convergence than mean

3. **Weighted Mean**:
   - Weight by sample count or validation loss
   - Good for heterogeneous data
   - Vulnerable to weight manipulation

See `crates/praxis-agg/src/methods.rs` for implementations.

### Can I add my own aggregation method?

**Yes!** It's a great first contribution:

1. Add function to `crates/praxis-agg/src/methods.rs`:
   ```rust
   pub fn my_method(updates: &[Vec<f32>]) -> Result<Vec<f32>> {
       // Your implementation
   }
   ```

2. Add tests:
   ```rust
   #[test]
   fn test_my_method() {
       // Test with valid inputs
       // Test with malicious inputs
       // Test edge cases
   }
   ```

3. Document when to use it

4. Open a PR!

See [CONTRIBUTING.md](../CONTRIBUTING.md) for details.

---

## Credentials & Governance

### What are Verifiable Credentials?

**W3C Verifiable Credentials** (VCs) are like digital diplomas:

**Traditional certificates**:
- PDF or paper
- Hard to verify
-易 to forge
- Not machine-readable

**Verifiable Credentials**:
- JSON with cryptographic signature
- Instantly verifiable
- Tamper-proof
- Machine-readable
- Portable across platforms

**Example use cases**:
- Prove you completed a course
- Prove you have a skill (without revealing your score)
- Prove you're qualified (without revealing institution)

See `examples/credentials/` for examples.

### Who can issue credentials?

In v0.1:
- **Any agent** can issue
- **Verifiers** check:
  - Signature is valid
  - Issuer is trusted (manually curated list)
  - Credential not revoked

In v1.0:
- **DAO-curated issuer registry**
- **Reputation-based trust**
- **Community vetting process**

### What's the DAO for?

**DAO** = Decentralized Autonomous Organization

The DAO governs:
- **Curricula**: Which courses are featured/recommended
- **Standards**: Rubrics for assessment
- **Issuers**: Who can issue credentials
- **Protocol**: Changes to FL parameters
- **Treasury**: Funding for development, audits, grants

**Decision paths**:
- **Fast** (24-48h): Emergency fixes
- **Normal** (3-14 days): Features, updates
- **Slow** (14+ days): Protocol changes, governance changes

See [GOVERNANCE.md](../GOVERNANCE.md) for details.

### How do I vote on proposals?

(v1.0 feature - not implemented yet)

**Process**:
1. Browse active proposals (DAO zome)
2. Read proposal details
3. Vote: For / Against / Abstain
4. Optionally justify your vote

**Voting power**: One agent = one vote (v1.0)
**Future**: Reputation-weighted, quadratic voting, etc.

---

## Development & Contributing

### I'm new to Rust. Can I still contribute?

**Absolutely!** We have tasks for all skill levels:

**No Rust required**:
- ✅ Improve documentation
- ✅ Add examples
- ✅ Test web UI
- ✅ Report bugs
- ✅ Design graphics/logos
- ✅ Write tutorials

**Beginner Rust**:
- ✅ Add tests
- ✅ Fix clippy warnings
- ✅ Add error messages
- ✅ Improve comments

**Intermediate Rust**:
- ✅ Add aggregation methods
- ✅ Implement zome functions
- ✅ Optimize performance

**Advanced Rust**:
- ✅ Design new protocols
- ✅ Security audits
- ✅ Unsafe code review

See [good first issues](https://github.com/Luminous-Dynamics/mycelix-praxis/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22) to start!

### How do I run the tests?

```bash
# All tests
make test

# Rust only
cargo test --workspace

# Specific crate
cargo test -p praxis-core

# Web only
cd apps/web && npm test

# With output
cargo test -- --nocapture
```

See [Getting Started](getting-started.md#testing) for more.

### Why do the zomes use `serde` instead of HDK?

**v0.1.0 strategy**: Build libraries first, HDK integration later.

**Current**:
```rust
// zomes/fl_zome/src/lib.rs
use serde::{Serialize, Deserialize};  // ✅ Works now

pub struct FlRound { ... }
```

**v0.2.0**:
```rust
// zomes/fl_zome/src/lib.rs
use hdk::prelude::*;  // ✅ HDK integration

#[hdk_entry_helper]
pub struct FlRound { ... }
```

This lets us:
1. Test business logic independently
2. Iterate on types quickly
3. Keep CI green
4. Integrate HDK when Holochain zomes are ready

### How do I add a new example?

1. Create JSON file in appropriate directory:
   ```bash
   # For a new course
   touch examples/courses/my-course.json
   ```

2. Follow the schema (see existing examples)

3. Validate against schema:
   ```bash
   ajv validate -s schemas/vc/*.schema.json -d examples/credentials/mine.json
   ```

4. Add entry to `examples/README.md`

5. Commit and open PR!

See [examples/README.md](../examples/README.md) for guidelines.

### Can I use this for my own project?

**Yes!** Apache-2.0 license allows:
- Commercial use
- Modification
- Distribution
- Private use

**Requirements**:
- Include license and copyright notice
- State changes made

**We'd love to hear about it!** Open a [Discussion](https://github.com/Luminous-Dynamics/mycelix-praxis/discussions) to share your project.

---

## Roadmap & Future

### When will v1.0 be ready?

**Target**: Q3 2026 (~21 months from now)

**Milestones**:
- **v0.2** (Q1 2026): DAO governance, DP support
- **v0.3** (Q2 2026): Pilot with 50-100 users
- **v1.0** (Q3 2026): Production-ready, security audit

See [ROADMAP.md](../ROADMAP.md) for detailed timeline.

### What's blocking v1.0?

**Technical**:
- [ ] Holochain DNA implementation
- [ ] End-to-end FL rounds
- [ ] DAO governance implementation
- [ ] Security audit
- [ ] Performance testing at scale

**Community**:
- [ ] 10+ regular contributors
- [ ] 100+ GitHub stars
- [ ] 5+ production courses

**Legal/Compliance**:
- [ ] GDPR compliance review
- [ ] FERPA compliance review
- [ ] Terms of service
- [ ] Privacy policy

### Can this scale to 1 million users?

**Theoretically yes**, but needs validation:

**Holochain scalability**:
- ✅ Horizontally scalable (more agents = more capacity)
- ✅ No global consensus bottleneck
- ⏳ Needs performance testing at scale

**FL scalability**:
- ✅ 100 participants tested in simulation
- ⏳ Need to test 1,000+ participants
- ⏳ Need efficient gradient transfer protocol

**DHT scalability**:
- ✅ Sharding distributes load
- ⏳ Need optimization for high-traffic entries

**Roadmap**: Performance testing in v0.3 with 100-500 users.

### Will there be a mobile app?

**Yes!** Planned for v0.3 (Q2 2026).

**Platform**: React Native (shares code with web app)

**Features**:
- Offline-first design
- Push notifications for FL rounds
- Credential wallet
- Course downloads

### Is there a business model?

**Praxis core is free and open source** (always will be).

**Potential revenue streams** (future):
- Premium course hosting
- Enterprise deployments
- Credential verification API
- FL-as-a-Service for organizations
- Grants and donations

**Philosophy**: Public goods funding + sustainable revenue, not extractive platform model.

### Is there a whitepaper?

**Not yet.** We have:
- ✅ [Protocol specification](protocol.md)
- ✅ [Architecture docs](architecture.md)
- ✅ [Threat model](threat-model.md)
- ⏳ Academic paper (planned for v1.0)

**If you're interested in writing one**: Open a [Discussion](https://github.com/Luminous-Dynamics/mycelix-praxis/discussions)!

---

## Still have questions?

- **Search**: Use GitHub search (often faster than asking)
- **Discussions**: [GitHub Discussions](https://github.com/Luminous-Dynamics/mycelix-praxis/discussions)
- **Issues**: [GitHub Issues](https://github.com/Luminous-Dynamics/mycelix-praxis/issues)
- **Email**: info@mycelix.org (for press, partnerships, etc.)

**Pro tip**: Before asking, check:
1. This FAQ
2. [Getting Started](getting-started.md)
3. [CONTRIBUTING.md](../CONTRIBUTING.md)
4. Existing GitHub issues/discussions

Happy learning! 🎓
