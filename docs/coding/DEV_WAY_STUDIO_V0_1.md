# DEV-WAY Studio Protocol v0.1

Parent: DEV-WAY-000 / #5552. Preregistration: DEV-WAY-001T / #5553.

## Purpose

Define an optional pre-verification creative-coding loop for Symthaea. Studio mode helps discover intent, materially different approaches, simplifications and experiential fit. It does not alter verification, source identity, security, authority, persistence or qualification rules.

## Governing law

- creative candidate != verified code
- human preference != correctness theorem
- elegant != safe
- interactive demo != production patch
- model confidence != design quality evidence
- Studio disposition != Forge/CI disposition
- vibe != PASS

## Pipeline

`Intent -> EssenceBrief -> Diverge -> InspectableCandidate -> Critique -> Subtract -> Compare -> HumanDisposition -> ExplicitSpec -> ExistingCodingAgent -> ExistingVerification`

Any stage may return to `Intent`. Studio mode may also be skipped entirely.

## EssenceBriefV1

Candidate-only fields:

- desired_experience
- invariants
- freedoms
- non_goals
- taste_terms
- examples
- anti_examples
- complexity_budget
- reversibility_expectation
- unresolved_questions

An essence brief is not a technical specification and cannot satisfy a compiler, test, security or authority gate.

## CreativeCandidateV1

Each candidate records:

- exact parent intent/brief reference
- candidate family/ancestry
- approach axes that materially differ from siblings
- concise design summary
- optional inspectable artifact/demo reference
- explicit tradeoffs
- assumptions
- subtraction opportunities
- unresolved risks/questions

Candidates derived from one common design family remain related; cosmetic rewrites do not count as independent design alternatives.

## Divergence axes

Where relevant, candidate families should differ on at least one meaningful axis such as:

- data model
- interaction model
- control flow
- architecture scale
- explicit vs implicit state
- local vs distributed assumptions
- dependency strategy
- API shape
- user-facing metaphor
- minimal vs expressive solution

No fixed number of candidates is required. Stop when additional candidates add no materially new design information.

## Inspectable-artifact rule

Prefer a small safe artifact when it can answer a design question faster than prose: local demo, fixture, visualization, example, mock UI, type sketch, API sample or executable sandbox.

Candidate artifacts inherit ordinary sandbox/effect restrictions. They do not receive network/filesystem/process/persistence authority merely because Studio mode requested them.

## Critique pass

Critique must separate at least:

- fit to desired experience
- invariant coverage
- conceptual complexity
- operational complexity
- dependency burden
- reversibility
- unknowns/risks
- likely verification burden

The model may explain these dimensions but does not produce an authoritative overall design score.

## Subtraction pass

For each promising candidate, explicitly test whether these can be removed while preserving the selected essence/invariants:

- types
- states
- abstractions
- dependencies
- configuration
- interactions
- persistent state
- external services

Fewer lines alone is not evidence of simpler semantics. A reduction is retained only when the later explicit specification remains satisfiable.

## Human dispositions

Candidate-only dispositions:

- KeepExploring
- PromisingCandidate
- TooComplex
- TechnicallyGoodWrongFeel
- ElegantButUnproven
- SelectedForSpecification
- RejectedForNow

None aliases Verified, Correct, Safe, Qualified or ApprovedForEffect.

## Specification handoff

`SelectedForSpecification` creates a new explicit technical-specification lineage. The specification records which creative candidate informed it, but verification operates on the specification/source—not on Studio preference.

Material reframing of desired experience/invariants creates a new intent/brief lineage rather than silently mutating the old one.

## Creative-memory boundary

Generated ideas do not become coding experience/rules automatically.

`exploration -> selected lesson candidate -> later implementation/outcome evidence -> existing coding-experience promotion`

Rejected experiments may be preserved as negative/contrast evidence. They are not positive coding rules.

## Paired benchmark

The first experiment compares:

A. current coding-agent flow;
B. Studio pre-pass followed by the same existing coding-agent and verification flow.

Report separately:

- downstream correctness/pass state
- human preference where measured
- materially distinct candidate families
- repair attempts/compiler invocations
- dependency/API/state complexity measures
- time/iterations to selection
- security/authority regressions
- reframe/abandonment rate
- qualitative notes

No universal creativity/elegance score is defined.

## Source-inspiration boundary

Rick Rubin's external *The Way of Code* motivates the idea of coding as a creative, iterative medium. This profile is original Symthaea process design and does not reproduce the project's meditations, wording, source code or interactive artifacts.

## Claim ceiling

A future PASS may establish only deterministic Studio workflow behavior and measured effects on an exact frozen benchmark. It does not establish objective creativity, elegance, correctness, production readiness, human preference in general, security, effect authority or qualification.