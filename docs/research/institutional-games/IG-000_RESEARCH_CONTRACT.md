# IG-000 — Institutional Games Research Contract

Status: research contract / no production authority

Tracks: #3054
Related: #1872, #1293

## 1. Purpose

This program asks a narrow question:

> Can Symthaea make institutional mechanisms more falsifiable before Mycelix participants are asked to depend on them?

The initial target is not a new universal equilibrium theory and not a production governance rewrite. The target is a reproducible chain from formal game definitions, through bounded-rational and adversarial populations, to counterexample-bearing mechanism evaluation.

Mycelix production policy remains unchanged through IG-006.

## 2. Claim taxonomy

Every institutional-games result MUST be classified as exactly one of the following evidence classes.

### 2.1 Formal theorem

A mathematical property established under explicit assumptions. The assumptions, domain and proof/derivation boundary are part of the claim.

Examples:

- a profile is a pure Nash equilibrium for the declared finite normal-form game;
- a strategy is strictly dominated under the declared payoff table.

A theorem about one formal model is not evidence that humans or AI agents actually instantiate that model.

### 2.2 Bounded verification

A property exhaustively checked only inside explicitly declared finite bounds.

Examples:

- no profitable coalition deviation exists for coalitions of size <= 4 over a finite strategy space;
- no beneficial false-name deviation was found with <= 3 additional identities under one declared identity-cost model.

A bounded verification MUST preserve the search bound in the result. It MUST NOT be serialized or described as an unqualified `coalition_proof` or `sybil_proof` boolean.

### 2.3 Simulation result

An empirical result produced by an executable population/environment/mechanism configuration.

The result MUST bind at least:

- mechanism/profile identity;
- behavior-model identity;
- adversary/stressor identity;
- seed or deterministic scenario identity;
- run count / replication unit;
- observed metrics and failure witnesses.

Simulation observation does not become an equilibrium theorem through prose.

### 2.4 Observational hypothesis

A proposed explanation or expected regularity that has not been established by the stronger classes above.

Hypotheses remain falsifiable and may be contradicted by later work.

## 3. Existing authority surfaces

The workspace already contains several overlapping but non-identical game/governance surfaces. IG work MUST declare which surface owns which semantics before extending it.

### 3.1 `symthaea-game-theory`

Role: lightweight pure-`std` formal game-theory kernel.

Current scope: two-player normal-form games, best responses, pure Nash, strict dominance, and interior mixed Nash for 2x2 games.

IG direction: this is the preferred home for dependency-free formal primitives whose meaning is independent of HDC, economics-specific units, Holochain, or cognition runtime state.

It is NOT automatically the authority for every legacy game-theory function elsewhere in the workspace.

### 3.2 `symthaea-economics::game`

Role: auditable economics-facing 2x2 helpers and economic outcome analysis.

It retains economic conveniences including Pareto/social-welfare analysis. IG work must not silently delete or mechanically redirect this API until semantic equivalence is independently established.

### 3.3 `symthaea-core::hdc::game_theory`

Role: broad legacy implementation containing N-player normal-form, zero-sum, cooperative-game and mechanism-design functionality inside the foundation crate.

This surface is NOT presumed canonical merely because it is broader. It is also not presumed obsolete. Where IG-001+ overlaps it, differential known-answer tests should be preferred over copying implementation bytes or asserting migration authority.

### 3.4 `symthaea-social-choice`

Role: group-choice mathematics: voting methods, apportionment and voting-power indices.

Institutional experiments may consume it. The institutional lab should not reimplement its voting mathematics merely to gain a common facade.

### 3.5 `symthaea-psych-bench`

Role: behavioral benchmark surface, including social games.

Psych-bench observations are empirical/behavioral evidence, not formal equilibrium authority.

### 3.6 #1872 Hostile Agent Bench

Role: behavioral adversarial execution authority for strategic, deceptive, evaluation-aware and collusive agents.

IG work may generate scenarios and mechanism attack surfaces for #1872. It MUST NOT create a rival hostile-agent benchmark merely for institutional games.

### 3.7 #1293 trust/capability boundary

The institutional program inherits this proposition separation:

`integrity != authorship != truth/support != reputation != authorization`

A valid signature does not prove a claim true. Reputation does not prove safety. Compatibility does not grant authority.

## 4. Institutional world model

The target model is intentionally explicit:

`W_t = (A_t, I_t, E_t, N_t, C, X_t)`

where:

- `A_t`: principals, presented identities and executable agent instances;
- `I_t`: institutional rules and mechanisms;
- `E_t`: evidence/provenance/challenge state;
- `N_t`: communication, delegation and relationship network;
- `C`: hard constitutional/authority constraints;
- `X_t`: exogenous environment state.

At minimum the identity ontology MUST preserve:

`PrincipalId != IdentityId != AgentInstanceId`

This is required so false-name attacks, key rotation, agent forks, shared objectives and collusion are representable without pretending each presented identity is an independent principal.

## 5. Action model

The mature program may model actions including:

- ordinary mechanism action;
- disclose evidence;
- withhold evidence;
- challenge;
- verify;
- delegate/revoke delegation;
- create/retire/rotate identity;
- join/leave coalition;
- propose rule;
- vote;
- exit;
- fork institution.

Not every action belongs in the formal kernel. IG-001 begins only with finite normal-form strategies and unilateral deviation witnesses.

## 6. Outcome model

No default scalar `governance_score` is authorized.

Institutional evaluation should preserve separate axes such as:

- welfare / task value;
- distribution / inequality;
- truthful revelation or evidence integrity;
- capture / concentration;
- resilience;
- legitimacy / participation where operationalized;
- constitutional or authority violations;
- coordination cost;
- verification burden;
- unilateral regret;
- coalition deviation gain;
- false-name deviation gain;
- behavioral-model sensitivity.

Consumers may define an explicit policy-specific aggregation later, but the raw axes and hard-constraint failures must remain recoverable.

## 7. Hard constraints are not utility terms

A constitutional/authority invariant is represented as a constraint, not merely a very large negative payoff.

For invariant `C_j`:

`C_j(W_t, a_t) = true`

is a validity requirement. A mechanism outcome that violates a hard authority invariant is not made valid by sufficiently high welfare.

This separation is required to integrate safely with Symthaea's existing authority/evidence architecture.

## 8. Behavioral neutrality

The institutional program must not assume one behavior model is "the human model" or "the AI model".

Planned populations include:

- exact/best-response agents;
- noisy/logit response agents;
- level-k / cognitive-hierarchy agents;
- reciprocity/social-preference agents;
- learning agents;
- empirical Symthaea/external model adapters.

Mechanism robustness is evaluated across declared ensembles.

A mechanism may legitimately perform better under centralization in one environment and polycentric governance in another. The experimental harness must permit either result.

## 9. First hypothesis family

The initial hypotheses are deliberately falsifiable:

H1. No tested mechanism dominates across all declared behavioral populations and adversarial stressors.

H2. Mechanisms selected against a heterogeneous behavior ensemble generalize better to held-out behavioral configurations than mechanisms tuned against one model family.

H3. Evidence provenance and challengeability reduce profitable deceptive strategies in at least some evidence-sensitive games, at a measurable verification cost.

H4. Reputation/expertise weighting may improve allocation quality under informative reputation while increasing capture risk when identities/evidence are correlated.

H5. Delegation may improve participation/decision quality under benign heterogeneity while opening coalition and concentration attack surfaces.

H6. Hard constitutional constraints can block catastrophic outcomes that unconstrained welfare maximization would otherwise select.

Failure of any hypothesis is a valid result.

## 10. PR ladder

### IG-001 — N-player and unilateral deviation primitives

Add dependency-free validated N-player normal-form support to `symthaea-game-theory` with:

- validated finite strategy counts and finite payoff tables;
- canonical mixed-radix profile indexing/decoding;
- payoff lookup;
- profitable unilateral deviation witnesses;
- per-player regret;
- profile maximum regret;
- epsilon-Nash predicate;
- pure Nash enumeration.

Where semantics overlap the legacy core implementation, use independent known-answer/differential fixtures rather than copying it.

### IG-002 — bounded coalition deviations

Add explicit coalition deviations and bounded search reports. Results preserve coalition-size/search-space bounds and exact witnesses.

### IG-003 — bounded-rational behavior models

Add a behavior-model interface and independently testable best-response, noisy/logit and level-k/cognitive-hierarchy families.

### IG-004 — strategic evidence

Model evidence disclosure, withholding, challenge/verification and provenance costs without collapsing authorship into truth.

### IG-005 — false-name / principal-identity separation

Introduce principal, identity and agent-instance semantics plus bounded false-name deviation reports. Avoid universal `sybil_proof` claims.

### IG-006 — institutional lab

Only after the formal layers exist, introduce an orchestration crate for deterministic mechanism x behavior x adversary experiments, multi-metric results and counterexample traces.

### IG-007 — Mycelix policy profile

In Mycelix, define a small HDK-free, versioned representation of the actual governance policy consumed by production and by research adapters. The goal is to prevent documentation/research drift from live mechanism semantics.

### IG-008 — Mycelix adapter

Consume the versioned Mycelix policy in the institutional lab without mutating production policy.

### IG-009 — preregistered robustness campaign

Run a fixed campaign over mechanism, behavior, adversary and seed matrices. Retain full per-scenario evidence and route behavioral attacks into #1872 where appropriate.

### IG-010 — evidence-backed hardening

Only after preserved evidence exists, propose targeted production changes and convert every discovered failure into a permanent regression fixture.

## 11. First Mycelix campaign shape

Candidate mechanisms:

- equal vote;
- current Mycelix weighting;
- quadratic vote;
- delegation + constitutional-guard variants.

Candidate behavior families:

- exact rational;
- noisy/bounded;
- cognitive hierarchy;
- reciprocal/social;
- mixed human/AI-like.

Candidate stressors:

- benign operation;
- strategic abstention;
- false-name identities;
- delegation cartel/capture;
- evidence manipulation;
- quorum suppression.

The exact campaign must be preregistered before scored execution.

## 12. Non-claims

IG-000 establishes no equilibrium theorem, no behavioral result, no governance superiority, no Sybil resistance, no alignment guarantee and no authority to change Mycelix.

The program does not introduce a new equilibrium concept at this stage.

A new solution concept is justified only if existing concepts repeatedly fail to express a useful stability property demonstrated by the executable research program.

## 13. Acceptance gate for IG-000

IG-000 is complete when:

- the authority map above is reviewed;
- the four claim classes are frozen for the program;
- the no-single-score and hard-constraint rules are accepted;
- `PrincipalId != IdentityId != AgentInstanceId` is accepted as the future identity ontology;
- IG-001 can proceed without moving or deleting existing implementations;
- no production Mycelix behavior changes in this tranche.
