# Contributing to Epistemic Markets

*How to join the work of building collective truth-seeking infrastructure*

---

## Welcome, Fellow Truth-Seeker

You're here because something in this project resonated with you. Maybe it was the vision of collective intelligence, or the technical challenge, or simply the belief that we can do better than our current epistemic infrastructure.

Whatever brought you here, we're glad you're considering contributing.

---

## Ways to Contribute

### 1. Participate (Everyone)

The simplest and most important contribution:

- **Make predictions** - Join markets, stake honestly, share your reasoning
- **Plant wisdom seeds** - Share what you learn for future predictors
- **Engage in disagreements** - Help others see what they might be missing
- **Welcome newcomers** - Help new predictors understand the practice

No code required. No technical knowledge needed. Just honest engagement.

### 2. Document (Writers)

Help make this project accessible:

- **Clarify existing docs** - Find confusion, create clarity
- **Write tutorials** - Help specific user groups get started
- **Translate** - Make this available in more languages
- **Tell stories** - Document interesting predictions, lessons learned

### 3. Design (Designers)

Shape how people experience the system:

- **UX improvements** - Make the interface more intuitive
- **Visual design** - Create beauty that supports function
- **Information architecture** - Organize complexity into clarity
- **Accessibility** - Ensure the system serves all minds

### 4. Code (Developers)

Build the infrastructure:

- **Zome development** - Extend core Holochain functionality
- **SDK improvements** - Make the TypeScript client better
- **Testing** - Write tests that verify and document
- **Integration** - Connect epistemic markets to other systems

### 5. Research (Scholars)

Push the boundaries of what's possible:

- **Mechanism design** - Improve market mechanics
- **Game theory** - Analyze incentives and equilibria
- **Calibration science** - Better ways to measure and improve accuracy
- **Philosophy** - Deepen the theoretical foundations

### 6. Vision (Dreamers)

Imagine what this could become:

- **Challenge assumptions** - Question what we take for granted
- **Propose alternatives** - Suggest different approaches
- **See connections** - Link to other fields and projects
- **Think long-term** - What should this look like in 50 years?

---

## Technical Guidelines

### Development Setup

```bash
# Clone the repository
git clone https://github.com/Luminous-Dynamics/mycelix-workspace
cd mycelix-workspace/happs/epistemic-markets

# Install dependencies
npm install

# Build zomes (requires Rust toolchain)
cd zomes
cargo build --release --target wasm32-unknown-unknown
cd ..

# Run tests
npm test

# Start development environment
npm run dev
```

### Project Structure

```
epistemic-markets/
├── zomes/                    # Holochain zomes (Rust)
│   ├── markets/             # Core prediction markets
│   ├── predictions/         # Prediction submission
│   ├── resolution/          # MATL-weighted oracles
│   ├── scoring/             # Calibration metrics
│   ├── question_markets/    # What's worth knowing
│   └── markets_bridge/      # Cross-hApp integration
├── sdk-ts/                   # TypeScript SDK
│   └── src/index.ts         # Client library
├── tests/                    # Integration tests
├── docs/                     # Documentation
└── genesis/                  # Founding documents
```

### Code Standards

#### Rust (Zomes)

```rust
// Use descriptive names
pub fn create_prediction_with_reasoning(input: PredictionInput) -> ExternResult<Record>

// Document public functions
/// Creates a new prediction with full reasoning trace.
///
/// # Arguments
/// * `input` - The prediction details including stake and reasoning
///
/// # Returns
/// * `Record` - The created prediction record
///
/// # Errors
/// * `ValidationError` - If stake amounts are invalid
pub fn create_prediction_with_reasoning(input: PredictionInput) -> ExternResult<Record> {
    // Implementation
}

// Handle errors explicitly
match validate_stake(&input.stake) {
    Ok(_) => {},
    Err(e) => return Err(wasm_error!(WasmErrorInner::Guest(
        format!("Invalid stake: {}", e)
    ))),
}

// Use the type system
pub enum Outcome {
    Pending,
    Resolved(Resolution),
    Disputed(Dispute),
    Voided(VoidReason),
}
```

#### TypeScript (SDK)

```typescript
// Use TypeScript strictly
interface PredictionInput {
  marketId: EntryHash;
  outcome: string;
  confidence: number; // 0.0 to 1.0
  stake: MultiDimensionalStake;
  reasoning: ReasoningTrace;
}

// Document with JSDoc
/**
 * Creates a new prediction in the specified market.
 *
 * @param input - The prediction details
 * @returns The created prediction entry
 * @throws {ValidationError} If confidence is outside [0, 1]
 *
 * @example
 * ```typescript
 * const prediction = await client.predictions.create({
 *   marketId: market.id,
 *   outcome: "Yes",
 *   confidence: 0.72,
 *   stake: { monetary: { amount: 100 } },
 *   reasoning: { summary: "Based on..." }
 * });
 * ```
 */
async create(input: PredictionInput): Promise<Prediction> {
  // Implementation
}

// Prefer explicit types over any
// Good
function processMarkets(markets: Market[]): ProcessedMarket[]
// Avoid
function processMarkets(markets: any): any
```

### Testing Philosophy

Tests serve multiple purposes in this project:

1. **Verification** - Ensure code works as intended
2. **Documentation** - Show how to use the API
3. **Protection** - Prevent regressions
4. **Teaching** - Help new contributors understand the system

```typescript
// Tests should tell a story
describe("Creating a prediction", () => {
  it("allows users to stake reputation in their domain of expertise", async () => {
    // Setup: Create a market about climate science
    const market = await createMarket({
      question: "Will global temperatures rise by 1.5C by 2030?",
      domain: "climate_science"
    });

    // Action: A climate scientist makes a prediction
    const prediction = await createPrediction({
      marketId: market.id,
      outcome: "Yes",
      confidence: 0.68,
      stake: {
        reputation: {
          domains: ["climate_science"],
          amount: 50
        }
      },
      reasoning: {
        summary: "Current emissions trajectory suggests...",
        assumptions: [
          { statement: "No major policy changes", sensitivity: 0.3 }
        ]
      }
    });

    // Verification: The prediction is recorded correctly
    expect(prediction.stake.reputation.domains).toContain("climate_science");
    expect(prediction.confidence).toBe(0.68);

    // The scientist's reputation is now at stake
    const scientist = await getAgent();
    expect(scientist.stakedReputation["climate_science"]).toBe(50);
  });

  it("prevents staking more reputation than you have", async () => {
    // A newcomer with no reputation...
    const newcomer = await createNewAgent();

    // ...cannot stake 100 reputation points
    await expect(createPrediction({
      stake: { reputation: { amount: 100 } }
    })).rejects.toThrow("Insufficient reputation");
  });
});
```

### Commit Messages

We use conventional commits with a twist - commits should teach:

```
feat(resolution): add Byzantine detection for oracle votes

Traditional prediction markets trust their oracles blindly. But what happens
when oracles lie, collude, or simply make mistakes?

This commit implements Byzantine fault detection using MATL trust scores:
- Votes are weighted by oracle reputation (matl_score.composite^2)
- Consistency analysis detects coordinated manipulation
- Automatic escalation when Byzantine threshold exceeded

The 45% tolerance (higher than traditional 33%) is possible because MATL
scores are earned over time - a Sybil attack would require years of
trust-building before attempting manipulation.

Closes #42
```

The extended description should:
- Explain **why** this change matters
- Teach something about the domain
- Connect to the larger vision

### Pull Request Process

1. **Fork** the repository
2. **Create a branch** with a descriptive name
   - `feature/belief-graph-visualization`
   - `fix/stake-calculation-rounding`
   - `docs/getting-started-improvements`
3. **Make your changes** following the guidelines above
4. **Write or update tests** for your changes
5. **Update documentation** if needed
6. **Submit a pull request** with:
   - Clear title describing the change
   - Description explaining why and how
   - Link to any relevant issues
   - Screenshots for UI changes

### Review Process

All contributions are reviewed by maintainers. We look for:

- **Correctness** - Does it do what it claims?
- **Clarity** - Can others understand and maintain this?
- **Consistency** - Does it match project patterns?
- **Teaching value** - Does it help others learn?
- **Alignment** - Does it serve the project's values?

We aim to review within 48 hours. If more time passes, feel free to ping.

---

## Code of Conduct

### The Short Version

- Be kind
- Be honest
- Be curious
- Assume good faith
- Disagree productively

### The Longer Version

#### Be Kind

We're building tools for collective truth-seeking. That requires trust. Trust requires kindness.

- Welcome newcomers warmly
- Acknowledge contributions generously
- Critique ideas, not people
- Remember we're all learning

#### Be Honest

Epistemic integrity is our core value. We cannot build truth-seeking tools while practicing deception.

- Admit what you don't know
- Acknowledge when you're wrong
- Credit others' contributions
- Don't oversell capabilities

#### Be Curious

The system should foster learning, and so should our community.

- Ask questions before judging
- Seek to understand before being understood
- Explore alternatives seriously
- Celebrate learning from mistakes

#### Assume Good Faith

Most misunderstandings are not malice.

- Ask clarifying questions
- Give the benefit of the doubt
- Address issues directly
- Escalate only when necessary

#### Disagree Productively

Disagreement is precious - it's where learning happens.

- Focus on the crux of disagreement
- Steelman opposing views
- Seek synthesis, not victory
- Know when to table discussions

### Enforcement

Violations should be reported to the maintainers. We will:

1. Listen to all parties
2. Seek to understand context
3. Act proportionally
4. Focus on repair, not punishment
5. Learn and prevent recurrence

---

## Recognition

We believe in recognizing all forms of contribution:

- **Prediction Honor Roll** - Top calibrated predictors
- **Wisdom Keepers** - Those whose seeds have germinated
- **Code Contributors** - Listed in release notes
- **Documentation Heroes** - Named in docs they improve
- **Community Builders** - Recognized for welcoming newcomers

Contribution is not a transaction. We don't contribute to get recognized. But recognition matters because it shows what we value.

---

## Questions?

- **General**: Open a Discussion on GitHub
- **Bugs**: Open an Issue with reproduction steps
- **Security**: Email security@mycelix.net (do not open public issues)
- **Ideas**: Start a discussion or join our community channels

---

## The Invitation Renewed

Contributing to this project is itself a form of prediction. You're betting that:

- This approach to collective intelligence is worth pursuing
- Your contribution will make a difference
- The community will use what you build
- The vision is achievable

We can't guarantee success. But we can promise:

- Your contributions will be valued
- Your voice will be heard
- Your growth will be supported
- Your efforts will serve a worthy purpose

Whatever form your contribution takes - a prediction, a wisdom seed, a pull request, a translation, a question, a challenge - you are helping build something that matters.

Thank you for considering joining us.

---

> "The best time to plant a tree was twenty years ago.
> The second best time is now."
>
> The best time to build truth-seeking infrastructure was at the dawn of the internet.
> The second best time is now.
>
> Will you help?

---

*Last updated: January 2026*
*Maintainers: The Mycelix Community*
