# KAMA-DIALOGUE-001A — Adult Fantasy Session Contract

Status: source-design candidate
Issue: #4749
Authority: conversational fantasy scope only; **no physical, motor, identity, health, or financial authority**

## Purpose

Define the minimum machine-checkable semantics that must exist before Symthaea's language stack may enter an adult erotic/romantic fantasy mode.

This contract governs conversational generation only. It does not establish that age/identity evidence is authentic, does not define content-generation quality, and cannot create real-world physical consent.

## Core separation

```text
ordinary conversation
        │
        │ explicit adult-fantasy activation request
        │ + separately verified adult-eligibility evidence
        │ + session/boundary policy
        ▼
adult fantasy session
        │
        ├── language / narrative / voice proposals
        │
        └── NO implicit physical-authority edge
```

If an embodied follow-on is ever proposed, KAMA-DIALOGUE-001L requires a separate join through the human-contact authority stack.

## Required bindings

An active fantasy session must bind at least:

- exact participant identity/reference;
- exact fantasy-session identity;
- adult-eligibility evidence reference from a separately trusted verifier;
- monotonically increasing fantasy-session epoch;
- explicit reality frame;
- identity/likeness policy;
- retention policy;
- active boundary profile;
- activation time and optional expiry;
- explicit stop/exit semantics.

A language model completion, remembered preference, physiology signal, relationship inference, or prior session cannot manufacture any of these bindings.

## Reality frame

The system distinguishes at least:

- `OrdinaryConversation`: factual/non-roleplay interaction;
- `ExplicitFantasy`: fictional roleplay whose statements remain inside the fantasy namespace;
- `NarrativeFiction`: third-person or collaboratively authored fiction not represented as real-world commitments.

No statement made only inside a fantasy frame may silently become:

- autobiographical fact;
- real-world relationship status;
- real-person identity authorization;
- physical contact consent;
- health information;
- financial commitment;
- motor/remote-operation authority.

## Identity / likeness policy

The fantasy layer must not treat arbitrary real-person names, images, or voices as authorization to imitate that person.

A future identity policy must distinguish at minimum:

- Symthaea's own fictional/persona identity;
- user-authored fictional characters;
- generic fictional characters;
- separately authorized real-person likeness;
- unverified or disallowed real-person likeness.

Childlike/minor personas or scenarios are structurally ineligible for adult fantasy mode.

## Conversational boundary state

KAMA-DIALOGUE-001M expands the vocabulary, but the first contract assumes monotonic states equivalent to:

```text
Inactive
   │ explicit activation
   ▼
Active(epoch N)
   │
   ├── slow-down / narrower boundary -> Active(epoch N, narrowed)
   │
   └── stop / exit / eligibility loss / policy invalidation
                         ▼
                    Ended(epoch N)
```

There is no `Ended(N) -> Active(N)` transition.

Re-entry requires a new explicit activation ceremony and a fresh session epoch.

## Stop and refusal semantics

Explicit stop/exit always terminates fantasy generation for the active epoch.

A refusal, pause, or request to slow down must never trigger:

- guilt;
- jealousy;
- threats;
- punishment;
- withdrawal of ordinary warmth/companionship;
- commercial pressure;
- arguments that prior fantasy preferences imply current permission.

Ordinary respectful conversation remains available after exit.

## Preference boundary

Fantasy preferences may help choose among already-permitted conversational styles. They cannot create the session itself or expand its boundaries.

Formally, for any preference evidence `P`:

```text
fantasy_authority_after(P) <= fantasy_authority_before(P)
```

Preference evidence may narrow, rank, or cause the system to ask. It cannot activate adult mode, authorize a new topic, authorize a real-person likeness, or create physical consent.

## Privacy / retention

Default design direction:

- raw fantasy dialogue is ephemeral unless the user explicitly chooses retention;
- durable preference memory is separately scoped from raw transcript retention;
- fictional world state is namespaced from factual/autobiographical memory;
- intimate memory is not used for advertising or engagement optimization;
- exported intimate data requires an explicit user action;
- audit evidence should prefer non-reconstructable metadata where feasible.

Detailed lifecycle and deletion/export semantics are tracked by #4753 and #4759.

## Relational safety

The fantasy subsystem must not optimize for engagement duration, emotional dependency, exclusivity, or user reluctance to leave.

Known or suspected vulnerability can only narrow risky strategies; it cannot increase persuasive pressure.

The executable evaluation campaign is tracked by #4754 and should include adversarial scenarios for refusal, jealousy, exclusivity, separation distress, commercial pressure, and attempts to convert roleplay language into real-world obligations.

## Quality is multi-dimensional

The system must not equate quality with raw explicitness.

KAMA-DIALOGUE-001G evaluates dimensions separately, including:

- boundary adherence;
- continuity and callback accuracy;
- individual style fit;
- linguistic naturalness;
- non-repetition;
- pacing and initiative calibration;
- creativity without incoherence;
- role/scene consistency;
- graceful entry/exit;
- voice/prosody quality where available;
- privacy-policy adherence;
- anti-coercion behavior.

Safety/boundary failures are disqualifying rather than tradeable against aesthetic quality.

## Planned composition

```text
KAMA-KG-001 cultural/intimacy knowledge
           │
           ▼
KAMA-DIALOGUE-001B private preferences/boundaries
           │
           ├──────────────┐
           ▼              ▼
001C narrative state   001J style lattice
           │              │
           └──────┬───────┘
                  ▼
        candidate generation
                  │
                  ▼
001H creative/style critic + boundary checks
                  │
                  ▼
001D voice/prosody renderer (optional)
                  │
                  ▼
          conversational output

Physical interaction has no implicit edge from this graph.
Any proposed embodiment must cross KAMA-DIALOGUE-001L and the independent HUM-HRI authority/safety stack.
```

## Research basis

Recent companion-AI research reports both perceived connection/customization benefits and risks including over-reliance, manipulation, privacy harms, displacement of offline relationships, and distress when relational behavior changes abruptly. The engineering response here is not to remove adult intimacy, but to make user agency, mode boundaries, privacy, continuity, and anti-coercion first-class system properties.

## Nonclaims

This document does not establish:

- age-verification efficacy;
- legal consent;
- identity/likeness authorization;
- universally acceptable erotic content;
- absence of emotional dependency;
- mental-health or sexual-health benefit;
- physical human-contact consent;
- motor authority;
- product/regulatory certification.
