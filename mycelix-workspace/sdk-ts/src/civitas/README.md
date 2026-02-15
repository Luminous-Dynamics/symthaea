# Civitas - Civic AI for Citizen Sovereignty

> *"Civitas"* (Latin): The community of citizens; the body of citizens who together constitute a state.

## What Civitas Is

Civitas is **citizen-owned AI**. It's not government surveillance or control—it's a tool that belongs to the people and advocates for their interests.

## What Civitas Does

| Agent | Purpose | Example |
|-------|---------|---------|
| **Benefits-Civitas** | Navigate social programs | "What benefits might I qualify for?" |
| **Justice-Civitas** | Understand and exercise rights | "Can I appeal this decision?" |
| **Participation-Civitas** | Engage in democracy | "What's up for vote this week?" |

## Key Principles

### 1. Transparency
Every response includes:
- **Epistemic status**: How confident is Civitas? (known/uncertain/unknown)
- **Sources**: Where does this information come from?
- **Limitations**: What Civitas doesn't know

### 2. Sovereignty
- Citizens own their data
- Civitas can't share information without consent
- Citizens can always reach a human
- No dark patterns or manipulation

### 3. Accessibility
- Plain language (no bureaucratic jargon)
- Multi-language support
- SMS access (no smartphone required)
- Voice interface option

### 4. Consciousness-Gating
Civitas uses Symthaea's consciousness architecture:
- Won't hallucinate or guess when uncertain
- Says "I don't know" when appropriate
- Escalates to humans for complex matters

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                    CIVITAS                          │
│  (Citizen-facing AI for Benefits/Justice/Civic)    │
├─────────────────────────────────────────────────────┤
│                    SYMTHAEA                         │
│  (Consciousness framework - epistemic governance)  │
├─────────────────────────────────────────────────────┤
│                    MYCELIX                          │
│  (Trust, identity, governance, finance)            │
├─────────────────────────────────────────────────────┤
│                   HOLOCHAIN                         │
│  (Distributed, citizen-owned infrastructure)       │
└─────────────────────────────────────────────────────┘
```

## Usage

### TypeScript
```typescript
import { createCivitas } from '@mycelix/sdk/civitas';

const benefits = createCivitas('benefits', {
  symthaeaUrl: 'http://localhost:5491',
  language: 'en',
  accessibilityMode: true,
  voiceEnabled: false,
});

const context = await benefits.startConversation(citizenId);
const response = await benefits.chat("What programs might I qualify for?");

console.log(response.content);
console.log(response.epistemicStatus); // { confidence: 'uncertain', ... }
console.log(response.actions); // [{ label: 'Apply for SNAP', smsCommand: 'APPLY SNAP' }]
```

### SMS
```
YOU: HELP benefits
CIVITAS: I can help you find programs you qualify for...

YOU: What can I get?
CIVITAS: Based on your profile, you may be eligible for:
• SNAP (90% match)
• Medicaid (85% match)
Reply APPLY SNAP to start, or LEARN SNAP for details.
```

## Domains

### Benefits-Civitas
- Eligibility screening
- Application assistance
- Status tracking
- Renewal reminders
- Document upload guidance

### Justice-Civitas
- Rights education
- Appeal process guidance
- Mediation referrals
- Legal aid connections
- Restorative justice options

### Participation-Civitas
- Active proposal summaries
- Voting assistance
- Delegation setup (liquid democracy)
- HEARTH budget participation
- Civic impact tracking

## Configuration

```typescript
interface CivitasConfig {
  symthaeaUrl: string;        // Symthaea backend
  conductorUrl?: string;      // Holochain conductor
  language: 'en' | 'es' | 'zh' | 'vi' | 'ko' | 'tl';
  accessibilityMode: boolean; // Plain language mode
  voiceEnabled: boolean;      // Text-to-speech
  domain: 'benefits' | 'justice' | 'participation';
}
```

## The Civitas Promise

1. **We work for you**, not the government
2. **We tell you what we know** and admit what we don't
3. **We protect your privacy** by default
4. **We help you exercise your rights**, not just comply with rules
5. **We connect you to humans** when AI isn't enough

---

*Civitas: AI that serves the community of citizens.*
