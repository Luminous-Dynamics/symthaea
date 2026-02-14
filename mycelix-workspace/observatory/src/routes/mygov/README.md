# MY GOV Dashboard - Demo Guide

## Quick Start

```bash
cd /srv/luminous-dynamics/mycelix-workspace/observatory
npm run dev
# Open http://localhost:5173/mygov (or whatever port is shown)
```

## What You're Looking At

**MY GOV** is a citizen-facing government dashboard powered by **Civitas**, a civic AI assistant. It demonstrates:

1. **Algorithmic Transparency** - Every decision shows its reasoning
2. **Trust Score Breakdown** - See exactly how trust is calculated (MATL)
3. **Civitas Chat** - AI assistant that admits uncertainty
4. **Appeal Rights** - Clear path to challenge decisions
5. **Multi-Channel Access** - Same info via web or SMS

## Demo Flow

### 1. First Visit
- Welcome modal explains the concept
- Click "Get Started" to dismiss

### 2. Explore the Dashboard
- **Stats bar**: Pending, Approved, Appealable, Trust Score
- **Pending Decisions**: Track applications in progress
- **Recent Decisions**: See outcomes with explanations
- **Trust Score**: Breakdown of how it's calculated
- **Appeal Rights**: Decisions you can challenge

### 3. Try Civitas Chat (Right Side)
Ask natural language questions:

**Benefits Tab:**
- "What benefits can I get?"
- "What's the status of my applications?"
- "When do I need to renew?"

**Rights Tab:**
- "Can I appeal this decision?"
- "What are my tenant rights?"
- "How do I find a lawyer?"

**Civic Tab:**
- "What's up for vote?"
- "How does delegation work?"
- "What is HEARTH?"

### 4. Notice the Details
- **Confidence indicators**: Civitas marks uncertain responses
- **Action buttons**: Suggested next steps
- **SMS commands**: Every action has an SMS equivalent
- **Sources**: Claims cite regulations when available

## Key Concepts

### Civitas
Latin for "community of citizens." It's AI that works FOR citizens, not government. Key principles:
- Admits uncertainty
- Cites sources
- Advocates for citizen interests
- Connects to humans when needed

### MATL (Trust Score)
The trust score uses Mycelix Adaptive Trust Layer:
- 34% Byzantine fault tolerance
- Multi-factor assessment
- Transparent breakdown
- Affects processing priority, not eligibility

### SMS Equivalence
Everything you can do on web, you can do via SMS:
- `STATUS permit-123` - Check status
- `VOTE YES 1` - Cast vote
- `BALANCE` - Check civic credits
- `HELP 75080` - Find resources

## Architecture

```
┌─────────────────────────────────────┐
│         MY GOV Dashboard            │
│  (SvelteKit, this directory)        │
└─────────────────┬───────────────────┘
                  │
┌─────────────────▼───────────────────┐
│         Civitas Agents              │
│  (sdk-ts/src/civitas/)              │
│  Benefits | Justice | Participation │
└─────────────────┬───────────────────┘
                  │
┌─────────────────▼───────────────────┐
│         Symthaea (Rust)             │
│  Consciousness-gated AI engine      │
└─────────────────┬───────────────────┘
                  │
┌─────────────────▼───────────────────┐
│         Mycelix (Holochain)         │
│  Identity, Governance, Finance      │
└─────────────────────────────────────┘
```

## Demo Mode vs Production

**Demo Mode (current):**
- Simulated data
- Mock Civitas responses
- No Holochain connection

**Production (future):**
- Real Holochain conductor
- Symthaea consciousness engine
- Live civic data

## Files

| File | Purpose |
|------|---------|
| `+page.svelte` | Main dashboard |
| `components/PendingDecisions.svelte` | Track applications |
| `components/RecentDecisions.svelte` | Decision history |
| `components/TrustScoreExplainer.svelte` | MATL breakdown |
| `components/AppealRights.svelte` | Appeal eligible decisions |
| `components/CivitasChat.svelte` | AI assistant widget |
| `components/WelcomeModal.svelte` | First-visit intro |

## SMS Gateway

The same AI is available via SMS:
```
services/sms-gateway/
├── src/
│   ├── index.ts           # Express server
│   ├── civitas-adapter.ts # Routes to Civitas
│   └── commands/
│       ├── vote.ts        # Liquid democracy
│       └── balance.ts     # CGC/HEARTH
```

## Reset Welcome Modal

To see the welcome modal again:
```javascript
localStorage.removeItem('civitas-welcome-seen')
location.reload()
```

---

*Civitas: AI that serves the community of citizens.*
