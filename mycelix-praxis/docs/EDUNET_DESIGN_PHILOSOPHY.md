# Praxis Design Philosophy: Emotionally Responsive Educational UI

## Core Principle
The app should feel like a calm, competent study companion — not a game, not a test, not a tool. It should meet the student where they are emotionally and adapt its presence accordingly.

## Theoretical Foundations

### Ken Wilber's AQAL (All Quadrants, All Levels)
- **I (Interior-Individual)**: How the student *feels* — anxiety, curiosity, confidence. The consciousness engine detects this.
- **It (Exterior-Individual)**: What the student *can do* — mastery scores, accuracy. The learning engine tracks this.
- **We (Interior-Collective)**: Does the student feel *connected* — social proof, community, purpose.
- **Its (Exterior-Collective)**: What *systems* support the student — CAPS curriculum, exam structure, Holochain.

Most EdTech only builds "It" and "Its." We build all four.

### Self-Determination Theory (Deci & Ryan)
- **Autonomy**: Student chooses what to study. Sovereignty model. Never commanding, always inviting.
- **Competence**: Clear, honest progress. Celebrate growth, not just absolute level.
- **Relatedness**: Even solo, the student feels the tool understands them. Community layer when Holochain is live.

### Behavioral Economics
- **Implementation intentions**: "When it is 16:00, I will study Calculus for 25 minutes."
- **Gain framing**: "47 study sessions available" not "47 days until exams."
- **Social proof**: "2,847 students studied this topic this month."
- **Gentle loss aversion**: Streak protection, not streak guilt.
- **Micro-reviews**: "Just 5 minutes" option for low-motivation days.

## Theme System: Base + Modulation

Three user-selectable base themes: Dark, Light, High Contrast.
Five consciousness-driven CSS modulation variables:
- `--consciousness-warmth`: color temperature (-1 cool to +1 warm)
- `--consciousness-density`: spacing multiplier (0.7 spacious to 1.3 dense)
- `--consciousness-chrome-opacity`: navigation visibility (0 hidden to 1 full)
- `--consciousness-animation-speed`: UI responsiveness (0.5 slow to 2 fast)
- `--consciousness-accent-saturation`: color vibrancy (0.5 muted to 1.2 vivid)

Time-of-day modulation: morning (cool/energetic), afternoon (neutral), evening (warm/calming).

All transitions: 4 seconds minimum, ease-in-out. Subliminal — the student feels comfortable without knowing why.

## South African Context
- Mobile-first (360px Samsung Galaxy A series)
- Offline-after-first-load via Service Worker (~4.5MB cache, ~R0.67 one-time cost)
- SA currency, geography, sports in examples
- SA grade structure (Grade R, Grade 1-12), not US terms
- "Your data stays on your device" is a feature, not fine print

## Typography
- Body: Inter (self-hosted woff2)
- Math: KaTeX for complex notation
- Line height: 1.7 body, 2.0 math explanations
- Paragraph spacing: 1.25em

## Emotional Adaptation Rules
1. Stress detected: warm colors, more spacing, slower animations
2. Flow detected: fade navigation, maximize content
3. Low motivation: slightly more vivid accents, curiosity prompts
4. High social readiness: surface community features
5. Cardinal rule: **Never make the student feel watched.**

## Implementation Phases
1. CSS foundation (variables, typography, mobile nav)
2. Consciousness-to-UI bridge (emotional_ui.rs)
3. Onboarding + student profile (conversational, 30 seconds)
4. Behavioral nudges (session planning, exam countdown, micro-reviews)
5. Mathematical rendering (KaTeX)
6. Offline support (Service Worker)
7. Community layer (Holochain, AQAL "We" quadrant)
