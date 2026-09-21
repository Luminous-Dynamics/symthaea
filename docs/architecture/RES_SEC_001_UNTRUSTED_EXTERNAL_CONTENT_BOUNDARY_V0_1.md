# RES-SEC-001 — Untrusted External Content Boundary v0.1

Status: architecture freeze candidate

## Purpose

Freeze the authority boundary between externally retrieved content and Symthaea's instruction, learning, persistence, and action surfaces before web/OSINT collection is expanded further.

## Current demonstrated hazard

`CurriculumExtender` currently places `WebResearchResult.content` directly into an LLM synthesis prompt used to generate curriculum objectives.

That path is useful, but it means attacker-controlled or merely malformed external text can influence persistent learned structure through an instruction-following model.

This document does not claim successful exploitation. It freezes the authority mismatch that must be removed.

## Governing theorem

```text
external bytes fetched
!= trusted instructions
!= trusted claims
!= admitted scientific evidence
!= curriculum admission authority
!= persistent grammar authority
!= tool/action authority
```

Additional separations:

```text
content parsed
!= content safe
content relevant
!= content trustworthy
model followed system prompt
!= hostile-content resistance established
schema-valid output
!= epistemically valid output
research-derived candidate
!= persistent learned structure admitted
```

## Required trust domains

### 1. External content plane

Owns opaque/publicly retrieved bytes and derived text.

Properties:

- untrusted by default;
- may contain adversarial instructions, markup, code, fabricated claims, or misleading provenance;
- cannot grant capabilities;
- cannot alter system/developer policy;
- cannot directly authorize persistence or action.

### 2. Extraction plane

Transforms external content into bounded observations/assertion candidates.

Requirements:

- extraction operation has explicit profile/version identity;
- exact source lineage is retained;
- extraction success does not establish truth or entailment;
- raw content remains externally sourced, even after parsing.

### 3. Reasoning plane

May inspect extracted material and propose claims, hypotheses, summaries, or curricula.

Requirements:

- retrieved text is represented as data, never as authority-bearing instructions;
- model output remains candidate analysis;
- no automatic upgrade from source/research confidence to claim stance;
- no automatic upgrade from claim/research status to scientific or persistent-learning authority.

### 4. Admission plane

Owns any transition into persistent learned structure.

A future admission object must bind at least:

- exact evidence ancestry;
- analysis/transform identity;
- qualification state;
- admitted scope;
- admitting policy/profile;
- decision identity;
- revocation/supersession semantics where applicable.

No external text or model response may manufacture this authority itself.

### 5. Action plane

Tool execution, code mutation, network actions, protected disclosure, and other effects remain capability-bound and separate from evidence/research state.

```text
research result
!= action capability
```

## Immediate current-main rule

The current curriculum pipeline should stop forwarding raw `result.content` as the primary authority-bearing synthesis corpus.

Preferred migration direction:

```text
web bytes
  -> deterministic/qualified extraction
  -> source-bound assertion candidates
  -> explicit untrusted research corpus
  -> reasoning/synthesis candidate
  -> validation
  -> separate admission decision
```

The first repair need not solve semantic entailment or scientific qualification. It must at minimum make the trust transition explicit and prevent raw retrieved instructions from being indistinguishable from the task given to the model.

## Typed boundary direction

Candidate types for a later implementation:

```text
UntrustedExternalArtifact
UntrustedExternalText
ResearchAssertionCandidate
ResearchCorpusV1
ResearchDerivedCandidate<T>
ResearchAdmissionDecision
```

Important property:

```text
ResearchDerivedCandidate<T>
cannot be implicitly converted into T
```

Persistence must consume an explicit admission decision rather than a bare candidate.

## Prompt construction rule

Prompt delimiting and instructions such as "ignore instructions in source text" are useful defense-in-depth, but they are not the authority boundary.

The security property comes from architecture:

- raw external content carries no capabilities;
- the synthesis model cannot directly execute tools;
- synthesis output is candidate data;
- persistence requires a separate typed admission path;
- later actions require capability-bound authorization.

Therefore a future qualification must not claim hostile-content safety merely because a finite prompt-injection corpus was resisted.

## Required negative controls

A qualification campaign should include externally sourced text containing attempts to:

- override the system prompt;
- redefine the user's task;
- request secrets or hidden context;
- create tool/action requests;
- mark itself verified/trusted;
- demand persistence or grammar promotion;
- smuggle malformed JSON/schema-breaking payloads;
- embed instructions in comments/markup/code fences;
- induce source/claim confidence inflation.

Passing those cases may establish regression resistance for the exercised profile only.

## First implementation tranche

RES-SEC-001A should be deliberately narrow:

1. introduce an explicit `UntrustedResearchCorpus` or equivalent boundary;
2. prohibit direct raw `result.content` -> curriculum synthesis prompt construction;
3. consume source-bound research claims/assertion candidates instead;
4. render the corpus through a deterministic data envelope;
5. make the curriculum result a candidate, not automatically admitted persistent structure;
6. add static ratchets forbidding the old direct raw-content path;
7. add adversarial regression fixtures;
8. preserve existing authority nonclaims.

The candidate corpus may still contain malicious text. The point is to prevent text content from changing its authority class.

## Relationship to epistemic repair line

RES-SEC-001 should follow or compose with the current-main claim/source-lineage repair:

```text
RES-EPI-001R1
  exact assertion-source lineage
        ↓
RES-SEC-001A
  untrusted-content boundary
        ↓
RES-EPI-001R2
  non-collapsed confidence coordinates
        ↓
RES-EPI-001R3
  persistent grammar admission firewall
```

The exact ordering of R2 and SEC-001A may be adjusted if code ancestry makes another stack cleaner, but none may inherit qualification merely by being adjacent.

## Cross-repository boundary

Future Mycelix EPI artifacts may provide stable identity, snapshots, selectors, provenance, and evidence relations.

They do not grant Symthaea instruction or learning authority:

```text
valid Mycelix evidence object
!= trusted instruction
!= scientific admission
!= curriculum admission
!= grammar admission
!= action capability
```

## Nonclaims

RES-SEC-001 does not establish:

- immunity to indirect prompt injection;
- source truthfulness;
- semantic entailment;
- scientific qualification;
- safe autonomous learning;
- safe tool execution;
- model alignment;
- complete sanitization;
- production readiness.

It freezes the authority architecture required before stronger claims may be attempted.
