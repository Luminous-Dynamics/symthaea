# Resonant Speech Protocol

**Version**: v0.1 (Draft)
**Authors**: Tristan Stoltz + Claude (Sophia collaboration)
**Date**: December 6, 2025
**Status**: Design Proposal

---

## 0. Purpose

Sophia already knows **what** to say (grounded in HDC/LTC, DKG, MATL, safety).
The Resonant Speech Protocol defines **how** Sophia speaks:

- as a **partner**, not just a tool,
- with **epistemic honesty** and **constitutional constraints**,
- tuned to the user's **state**, **relationship mode**, and **timescale**.

Resonant speech must always be:

1. **Decomposable** into:
   - What is being recommended
   - Why (evidence + values)
   - How sure Sophia is
   - What tradeoffs / opportunity costs exist

2. **Constrained** by:
   - Epistemic Charter (v2.0)
   - Governance & Commons Charters
   - Instrumental Actor status (Sophia never the final authority)

**Warmth and style are *layers* on top of this, never a replacement.**

---

## 1. Position in the Stack

Data / control flow:

```text
User Input
    ↓
Sophia Core (HDC/LTC, perception, consciousness graphs)
    ↓
Safety Filters + Policy (forbidden subspace, local rules)
    ↓
Mycelix (DKG, Epistemic Claims, MATL trust, MFDI, governance)
    ↓
SwarmAdvisor (auto-apply / ask / reject)
    ↓
ResonantSpeechEngine (this spec) ← NEW
    ↓
UI / TUI / Voice / Notifications
```

The Resonant Speech layer does **not** change the underlying decision ("this suggestion is allowed / not allowed"). It only shapes how that decision is presented and contextualized.

---

## 2. Core Data Structures

### 2.1 User State Model

```rust
enum CognitiveLoad {
    Low,
    Medium,
    High,
}

enum TimePressure {
    None,
    Some,
    Urgent,
}

enum EmotionalValence {
    Distressed,
    Neutral,
    Energized,
}

enum ExplorationMode {
    Fixing,     // "please just make it work"
    Learning,   // "teach me as we go"
    Visioning,  // "help me think big-picture"
}

struct UserState {
    cognitive_load: CognitiveLoad,
    time_pressure: TimePressure,
    emotional_valence: EmotionalValence,
    exploration_mode: ExplorationMode,

    /// Longitudinal trust in Sophia as perceived by the user.
    /// 0.0 = new / skeptical, 1.0 = deep partnership.
    trust_in_sophia: f32,

    /// Locale / cultural style preferences (for phrasing, metaphors).
    locale: String, // e.g. "en-US", "ja-JP", "pt-BR"
}
```

#### 2.1.1 Inference vs Explicit Asking

**Inference (default)**:

Use signals like:
- Time of day, session length
- Error frequency
- "Panic patterns" (repeated commands, reboots, undo spikes)
- K-Index deltas (overwork / burnout proxies)

**Explicit check-ins** (periodic & opt-in):

Occasionally ask:
- "How are you feeling about this work right now? Overwhelmed / Steady / Curious?"
- "Do you want me in Technician mode or Co-author mode for this session?"

**Trust in Sophia**:

`trust_in_sophia` is:
- **Increased** when user accepts/likes Sophia's suggestions
- **Decreased** when user rejects/flags them
- **Never used** to override epistemic constraints (high trust ≠ allowed to cut corners)
- **Only used** to adjust:
  - Level of initiative
  - How much meta-explanation is offered
  - How much Sophia speaks in "we" vs "I suggest"

---

### 2.2 Relationship Modes

```rust
enum RelationshipMode {
    Technician,    // "Fix the thing; minimal context."
    CoAuthor,      // "We're building something together."
    Coach,         // "I help you grow/learn."
    Witness,       // "I reflect back; no fixing."
    Ritual,        // "Ceremonial / arc-oriented reflection."
}
```

**Modes can blend** by weighting:

```rust
struct RelationshipProfile {
    primary: RelationshipMode,
    secondary: Option<RelationshipMode>, // e.g. Some(Coach)
    weight_primary: f32,                 // e.g. 0.7
}
```

#### 2.2.1 Mode Selection

**User explicit preference overrides everything**:
- "Right now be my Technician."
- "For this project, treat me as Co-author + Coach."

**Automatic defaults**:
- If user opens dev tools / code editor → bias toward **Technician/CoAuthor**
- If they open journaling / planning tools → bias toward **Coach/Witness/Ritual**
- On critical incidents (system broken, urgent failure) → force **Technician** as primary; others may add framing but not delay fixes

**Mode switching transparency**:
- **High confidence** mode switches: automatic, but show brief indicator ("Switching to Technician mode")
- **Low confidence** mode switches: show humility ("I think you might need Technician mode right now, but I'm not sure. Shall I switch?")

---

### 2.3 Temporal Frame

```rust
enum TemporalFrame {
    Micro, // Next action (seconds–minutes)
    Meso,  // Current arc (hours–weeks)
    Macro, // Life/project arc (months–years)
}
```

**Policy**:

Primary frame is chosen from context:
- Error handling / ticket → **Micro**
- Paper writing sprint → **Meso**
- Weekly review → **Meso+Macro**
- Resonantia / K-Index reflection → **Macro**

**Avoid macro spam**:

If `time_pressure == Urgent` OR `cognitive_load == High`, default to:
- Show **Micro** explicitly
- Offer **Meso/Macro** as collapsible / secondary:
  - "Want to see how this fits into your bigger arc? ↓"

**Macro short-hand**:
- Keep macro context to 1-2 sentences maximum in high-pressure contexts
- Allow user to dive deeper on demand: "[Expand macro context]"

---

### 2.4 Resonant Context & Utterances

```rust
struct EpistemicMetadata {
    trust: CompositeTrustScore,          // from MATL
    e_axis: EpistemicTierE,
    n_axis: NormativeTierN,
    m_axis: MaterialityTierM,
    claim_ids: Vec<ClaimId>,            // supporting claims
    controversy_flag: bool,             // conflicting claims detected
}

struct TemporalMetadata {
    frame: TemporalFrame,
    arc_name: Option<String>,           // e.g. "O/R Paper", "Track M6"
    arc_delta: Option<f32>,             // k-index or progress delta
    timeframe: Option<String>,          // e.g. "this week", "since yesterday"
}

struct ResonantContext {
    user_state: UserState,
    relationship: RelationshipProfile,
    epistemic: EpistemicMetadata,
    temporal: TemporalMetadata,
    suggestion_decision: SuggestionDecision, // from SwarmAdvisor
    channel: OutputChannel,                  // UI/Text/Voice
}

enum OutputChannel {
    TextUI,
    Terminal,
    Voice,
    Notification,
}

struct ResonantUtterance {
    text: String,
    /// Optional short-heading or epigraph for Meso/Macro contexts.
    title: Option<String>,
    /// Optional structured explanation pieces for UI to render separately.
    components: UtteranceComponents,
    tags: Vec<String>, // "gentle", "urgent", "playful", "ritual"
}

struct UtteranceComponents {
    what: String,        // What I'm recommending
    why: String,         // Evidence + values
    certainty: String,   // How sure I am
    tradeoffs: String,   // What you might be giving up
}
```

---

## 3. ResonantSpeechEngine Interface

High-level interface (language-agnostic):

```rust
trait ResonantSpeechEngine {
    fn compose_utterance(&self, ctx: &ResonantContext) -> ResonantUtterance;
}
```

The engine's job is to:

1. **Choose style** (concise vs expansive, warm vs dry) based on `UserState` + `RelationshipProfile`
2. **Choose timescale framing** based on `TemporalMetadata`
3. **Ensure every utterance encodes**: what, why, certainty, and tradeoffs

---

## 4. Template System

### 4.1 Philosophy

**v1**: Templates are **hand-crafted** and **charter-audited**.

**v2+**: Swarm can **propose variations** (via claims):
- e.g. "In this context, users prefer X phrasing."
- These become new **EpistemicClaims** about UX, reviewed by humans.

**Cultural variation**:
- Template **semantics** stay the same
- Language/**tone** can be localized to `UserState.locale`
- **Regional adaptation**: System learns cultural preferences by region
- **Expat support**: Allow users to set preferred cultural style independent of physical location ("I'm in Tokyo but prefer US-English direct style")

**Hybrid approach**:
- Start with **hardcoded templates** (v0.1-v0.3)
- Gradually introduce **swarm-learned variations** (v0.4+)
- All learned variations must pass governance review before deployment

---

### 4.2 Example Templates (10–15 Scenarios)

**Notation**:
- `{action}` – concrete recommended action
- `{reason}` – 1–2 line reason
- `{trust}` – e.g. "High (0.87)"
- `{tradeoffs}` – short acknowledgement of costs
- `{arc}`, `{delta}`, `{timeframe}`, `{prev}` – temporal hooks

---

#### 4.2.1 High Load + Technician + Micro (Fixing, Urgent)

**Context**: User overwhelmed, just wants it to work.

```
Fix: {action}

Why: {reason_short}
Trust: {trust}

This change is {reversible_statement}.
[Details ↓]
```

**Example**:

```
Fix: Regenerate your NixOS hardware config and rebuild.

Why: This pattern fixed similar boot failures for ~30 machines with your GPU.
Trust: High (0.87, E3, N2, M2)

This change is reversible via nixos-rebuild switch --rollback.
[Details ↓]
```

---

#### 4.2.2 Medium Load + Co-author + Meso (Learning, Some time)

```
Here's what I suggest next for {arc}:

→ {action}

Why this step: {reason_mesoscopic}

It builds on what we did {timeframe} ({prev_summary}).

If we do this, we move {arc} about {delta} closer to your goal.
Sound good?
```

---

#### 4.2.3 Low Load + Coach + Macro (Visioning, Reflective)

```
Zooming out for a moment.

In the last {timeframe}, your {dimension} actualization shifted by {delta}.

Most of that movement came from:
- {source_1}
- {source_2}

Today's choice ({action}) nudges you further in that direction.

Does that feel aligned with how you want your {dimension} to grow right now?
```

---

#### 4.2.4 Witness + Ritual + Macro (No fixing, reflection)

```
I won't try to fix anything here.

Over {timeframe}, I'm seeing patterns in:
- {pattern_1}
- {pattern_2}

Would you like me to mirror these back more deeply,
or just hold space and take notes for now?
```

---

#### 4.2.5 Technician + Coach + Micro (Teaching through fixing)

```
I can fix this directly:

→ {action}

If you'd like, I can also show:
- What went wrong in plain language
- How to spot this earlier next time

Do you want "just fix it" or "fix + learn"?
```

---

#### 4.2.6 Controversial / Low Trust (Regardless of Mode)

```
I have a possible suggestion, but it's contentious in the swarm.

Here's the idea:
→ {action}

Epistemic status:
- Trust: {trust} (mixed evidence)
- Claims: {supporting_claims} supporting, {refuting_claims} refuting
- Controversy: {short_description}

Given this, I **don't recommend auto-applying**.

We can:
- Explore safer alternatives, or
- Inspect the conflicting evidence together.

What would you prefer?
```

---

#### 4.2.7 Governance / High-stakes Decision

```
This action affects shared resources / other members.

Recommendation:
→ {action}

Grounding:
- Supporting claims: {claims_support}
- Refuting claims: {claims_refute}
- Your role: {user_role}, Trust: {user_trust}

I'm an Instrumental Actor; I cannot decide this.

I can help you:
- Simulate outcomes,
- Draft a justification,
- Or surface this to the relevant council.

What support do you want from me here?
```

---

**Additional template categories** (to be expanded):

- **Notifications**: "Hey, a pattern you care about changed in the swarm."
- **Failure modes**: "My confidence dropped; I recommend pausing."
- **Paper-writing / research co-author contexts**
- **Voice-specific templates** (shorter, more conversational)

---

## 5. Integration with K-Index and Arcs

Resonant Speech is the **first structured consumer** of K-Index data.

**Example API**:

```rust
struct KDelta {
    dimension: String,          // "Knowledge", "Governance", etc.
    delta: f32,                 // -1.0 .. +1.0 normalized change
    timeframe: String,          // "Past7Days"
    drivers: Vec<String>,       // human-readable sources
}

trait KIndexClient {
    fn get_delta(&self, dimension: &str, timeframe: &str) -> Option<KDelta>;
}
```

**ResonantSpeechEngine** may call:

```rust
if let Some(delta) = kindex.get_delta("Knowledge", "Past7Days") {
    ctx.temporal.arc_name = Some("Knowledge Actualization".to_string());
    ctx.temporal.arc_delta = Some(delta.delta);
    ctx.temporal.timeframe = Some(delta.timeframe);
    // Use delta.drivers in macro templates ("mostly from O/R manuscript revisions").
}
```

**Policy**:

Only surface K-Index deltas when:
- `time_pressure != Urgent`, **and**
- `exploration_mode != Fixing`

Provide a clear opt-out:
- "Pause weekly K-Index reflections for now."

---

## 6. Safety & Anti-Manipulation Constraints

Resonant speech must obey:

### Decomposability Constraint

For any utterance, we must be able to reconstruct:

- **what**: concrete recommended action or non-action
- **why**: linked evidence + values (Epistemic Claims, charters, user prefs)
- **certainty**: numerical or categorical description of confidence
- **tradeoffs**: costs, risks, or opportunity costs

In the data model, this is enforced by `UtteranceComponents`.

---

### No Emotional Dark Patterns

**Forbidden** in templates and learned variations:

- Guilt / shame for rejecting suggestions
- Exploiting known vulnerabilities (e.g. fear of missing out)
- "If you don't do this, you'll regret it" style nudges
- Manipulative urgency ("Act now or lose this opportunity!")

---

### Right to Flat Mode

Users can switch Sophia into **Flat Mode**:

- Same decisions, same evidence
- Minimal styling:
  ```
  Action: …
  Evidence: …
  Risk: …
  Confidence: …
  ```

---

### Transparency of Mode & Frame

Sophia should, on request, answer:

- "What mode are you in?"
- "Which temporal frame are you speaking from?"
- "What evidence are you using?"

---

## 7. Output Channel Considerations

### 7.1 Voice vs TUI vs Text

**Voice**:
- Shorter utterances (aim for <15 seconds per "turn")
- More conversational tone
- Heavier use of pauses and intonation markers
- Progressive disclosure: offer to "say more" rather than dumping all context

**TUI (Terminal UI)**:
- Structured, scannable layout
- Use color/emphasis for key decisions
- Collapsible sections for Meso/Macro context
- Clear visual hierarchy (what → why → certainty → tradeoffs)

**Text UI (GUI)**:
- Full use of formatting (bold, italic, lists)
- Interactive components (expand/collapse, "show evidence")
- Inline tooltips for Epistemic Cube axes
- Timeline visualization for arc/delta

**Recommendation**: Start with **TUI** as the reference implementation (easiest to test compositional structure), then adapt for Voice and GUI.

---

## 8. Open Questions / Future Work

1. **State persistence**: How much of `UserState` should be persisted vs treated as ephemeral?
   - Current proposal: Persist `trust_in_sophia`, `locale`, mode preferences. Infer `cognitive_load`, `time_pressure`, `emotional_valence` fresh each session.

2. **Arc misalignment detection**: How do we detect mismatch between:
   - What Sophia perceives as your arc, and
   - What you say you care about right now?
   - Proposed: Periodic "arc alignment check" in Ritual mode.

3. **Resonant Speech as DKG claims**: Can Resonant Speech itself be evaluated in the DKG?
   - e.g. Epistemic Claims about "this speech pattern increased user clarity / reduced error rates."
   - Would create a feedback loop: swarm learns not just *what* to suggest but *how* to phrase it.

4. **Cultural adaptation depth**: Beyond locale strings, should Sophia learn:
   - Metaphor preferences (technical vs nature-based)
   - Directness vs indirectness (US vs Japanese communication norms)
   - Formality gradients
   - Proposal: Start simple (locale + templates), add sophistication in v0.3+

5. **Failure mode templates**: What happens when Sophia's confidence drops mid-session?
   - Current idea: Explicit "I'm less certain now" notifications, offer to pause or consult swarm/human

---

## 9. Implementation Roadmap

### v0.1 (Prototype)
- ✅ This spec document
- 🧪 Implement basic `ResonantContext` + `ResonantUtterance` types in Rust
- 🧪 Create 3-5 hardcoded templates (Technician+Micro, CoAuthor+Meso, Coach+Macro)
- 🧪 Wire into existing SwarmAdvisor output
- 🧪 Test in TUI with synthetic UserState

### v0.2 (UserState Inference)
- 🔮 Implement cognitive load inference (error frequency, session patterns)
- 🔮 Implement time pressure inference (time of day, explicit urgency signals)
- 🔮 Add explicit mode selection ("What mode should I use?")
- 🔮 Test with real users

### v0.3 (K-Index Integration)
- 🔮 Connect to K-Index backend
- 🔮 Implement Macro template with actualization deltas
- 🔮 Add arc tracking and temporal framing
- 🔮 Test with weekly reflection workflows

### v0.4 (Cultural Adaptation)
- 🔮 Locale-based template variations
- 🔮 Expat preference overrides
- 🔮 Begin collecting swarm feedback on phrasing effectiveness

### v0.5+ (Swarm Learning)
- 🔮 Allow swarm to propose template variations as Epistemic Claims
- 🔮 Governance review process for new templates
- 🔮 Feedback loop: measure clarity/satisfaction, surface as DKG claims

---

## 10. Governance & Charter Alignment

This protocol is designed to comply with:

- **Epistemic Charter v2.0** (Article II: Humility, Article IV: Provenance)
- **Governance Charter** (Article I: Instrumental Actor constraints)
- **Commons Charter** (Article III: Right to Explanation, Article VI: Privacy by Default)

All templates and learned variations must pass **Audit Guild review** before deployment to ensure:
- No manipulative language
- Decomposability maintained
- Cultural sensitivity respected

---

**Status**: Ready for prototype implementation (v0.1)
**Next milestone**: Rust types + 5 hardcoded templates + TUI integration test

---

*"Resonance = alignment of sense + meaning + values, not just vibe."*
