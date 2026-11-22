# Decision-Preserving Recurrent-State Compression (v0.1)

**Tracking:** #11561 decision-value computation; #13152 deterministic decision-observability  
**Observation qualification:** #13202 robust outcome-support regret  
**Joint resource evaluation:** #9666 / #9882 / #13176  
**Evaluator isolation:** #9437  
**Status:** evaluator-side Rust reference implementation and eight unit tests authored; Rust execution and CI qualification pending.

## 1. Research question

Can Symthaea compress its recurrent cognitive state while preserving the decisions that remain justified by that state?

A representation can lose substantial internal detail without harming a task decision. Conversely, two states that look similar under a generic similarity metric may require different actions. The relevant compression objective is therefore not vector reconstruction fidelity alone. It is preservation of the task's epsilon-qualified decision envelope.

This module adds an evaluator-side qualification over the existing #13152 decision-observability oracle. It does not introduce a second action-loss function, candidate compression algorithm, memory allocator, scheduler, safety authority or resource ledger.

## 2. Formal contract

Let a full candidate view include the authorized observation history, full recurrent-state identity, model/configuration identity and frozen policy seed. These define a baseline information partition:

\[
\mathcal{F}=\{F_1,\ldots,F_m\}.
\]

Let \(\phi(z)\) be the candidate's compressed recurrent-state representation. Replacing the full-state digest with the compressed-state digest induces a coarser information partition:

\[
\mathcal{C}=\{C_1,\ldots,C_n\}, \qquad
C_j=\bigcup_{k:\phi(F_k)=C_j}F_k.
\]

The task evaluator owns the hidden-world set, admissible action set and normalized loss. For any information set \(S\), define its deterministic minimax-regret floor:

\[
R(S)=\min_{a\in A}\max_{w\in S}
\left[L(w,a)-\min_{a'\in A}L(w,a')\right].
\]

The compression-induced regret floor for one compressed class is:

\[
\Delta R(C_j)=R(C_j)-\max_{F_k\subseteq C_j}R(F_k).
\]

Because every compressed class is a union of complete full-state classes, merging information cannot reduce the minimax-regret floor:

\[
R(C_j)\geq \max_{F_k\subseteq C_j}R(F_k),
\qquad \Delta R(C_j)\geq 0.
\]

The implementation checks this monotonicity property and rejects a violation. It also rejects a purported compression mapping that assigns different compressed digests to the same full-state identity.

## 3. Epsilon decision preservation

The task freezes two independent tolerances before evaluation:

- \(\epsilon\): maximum worst-case regret for an information set to be decision-sufficient;
- \(\delta_{comp}\): maximum extra minimax-regret floor permitted from compression.

A compressed class is qualified as decision-preserving only when:

1. every source full-state class was already epsilon-sufficient;
2. the merged compressed information set remains epsilon-sufficient; and
3. \(\Delta R(C_j)\leq\delta_{comp}\).

The evaluator separately reports whether the candidate's actually emitted action has worst-case regret at most epsilon. An epsilon-sufficient information set does not imply the candidate selected a good action.

Exact common optimality, epsilon-regret sufficiency, candidate-action quality and compression-induced loss are distinct outputs. A class can lack an exact common optimal action but remain epsilon-sufficient.

The qualification is task-relative. The epsilon and compression margin are not universal constants and cannot be tuned after inspecting confirmatory outcomes.

## 4. Implementation and validation controls

The evaluator module is research/temporal-evaluator/src/decision_sufficient_representation.rs.

It accepts paired evaluator records containing:

- the original full candidate-view identity;
- a candidate-compressed recurrent-state digest;
- the paired hidden-world identity;
- the candidate action;
- evaluator-only losses for the common admissible action set.

It constructs baseline and compressed information partitions itself. It uses evaluate_decision_observability() as the only regret oracle. Within each compressed class, it requires identical candidate views and identical deterministic actions, so differing actions cannot be misinterpreted as information preserved by a representation that supposedly made those worlds indistinguishable.

Eight unit tests are authored:

1. Opposing decisions merged by compression exceed the regret envelope.
2. A low-regret common action can remain epsilon-sufficient without exact common optimality.
3. Identity compression preserves the baseline partition and has zero added regret.
4. One full state cannot map to multiple compressed digests.
5. Identical compressed views cannot emit different deterministic actions.
6. Duplicate evaluator-world identities are rejected globally.
7. Out-of-range task tolerances are rejected.
8. Compression regret budget and task decision tolerance are enforced independently.

## 5. Relationship to the wider programme

- #13152 provides the existing deterministic minimax-regret oracle.
- #13165 remains the separate mixed-strategy evaluator; this v0.1 module qualifies deterministic candidate policies only.
- #13202 evaluates robust decision regret across possible observation outcomes. It is complementary: this module evaluates loss of information caused by compressing recurrent state.
- #9666/#9882 own joint resource control and evaluator convergence. This module supplies a decision-preservation metric that a memory/compute allocator can later use; it does not select allocations.
- #13176 owns interaction attribution. No new Shapley or factorial implementation is introduced.
- #9437 remains necessary because evaluator package separation is not a process/OS isolation boundary.

## 6. Claim boundaries

This is an evaluator reference implementation, not a production HDC/LTC compressor.

A nonzero digest binds representation identity; it does not prove the compressed state was actually used during candidate execution. Nor does this module establish a memory, latency, energy or throughput reduction. Those require source-bound execution and measured resource receipts under the existing programme.

The present contract:

- covers deterministic policies with a frozen policy seed;
- uses finite evaluator-supplied hidden-world and action-loss sets;
- requires a common action set inside each equivalence class;
- uses a task-frozen PPM regret scale;
- does not certify safety, legal/rights authority, viability or deployment readiness;
- does not establish that a learned compression map generalizes to unseen task families.

Confirmatory qualification must use held-out environment/source families, a candidate/evaluator process boundary under #9437, and paired resource measurements. A compression win is valid only if decision preservation and resource savings are both demonstrated.

## 7. Research hypotheses for the next experiment

1. Decision-aware rate–distortion: at matched memory bytes, decision-envelope preservation will rank representations differently from cosine similarity or reconstruction MSE.
2. Task-conditional compression: one representation may be sufficient for navigation but insufficient for manipulation, language grounding or safety-critical decisions.
3. Temporal stability: compression that preserves the current action can still destroy information needed for a later decision. Evaluate rolling and delayed decision horizons, not just one checkpoint.
4. Adaptive precision: preserve high-resolution subspaces only when their removal would increase the future decision-regret envelope beyond the task tolerance.
5. Joint sensing–memory–compute allocation: the value of a compressed state depends on which future observations and computations remain available; test interactions using #9666/#13176 rather than independently optimizing each subsystem.

No SOTA, memory, latency, energy, hardware or deployment improvement is claimed by this protocol.


## 8. External prior art and positioning (reviewed November 22, 2026)

Decision-relevant state abstraction and value-aware world models are established research directions. This evaluator does not claim to invent them.

- **Online Policy-Wise State Abstraction for Reinforcement Learning Agents Leveraging Free Energy** (Rashidi Laleh & Nili Ahmadabadi, IEEE TNNLS, online September 30, 2026; DOI 10.1109/TNNLS.2026.3735939). It uses parallel full-state and adaptively masked learners with free-energy-based selection. It is a direct comparator for candidate-side online abstraction and policy selection. The present module does not learn masks, train parallel agents or select the deployed policy; it supplies a finite evaluator certificate for information loss and decision regret. https://doi.org/10.1109/TNNLS.2026.3735939
- **Boosting World Models Learning via Latent-Space Value Alignment** (ICML 2026, PMLR 306, pp. 52832–52862). It regularizes latent world models to align with task value while maintaining dynamics structure. It is a comparator for learning a decision-relevant latent representation; the present finite evaluator is complementary and can qualify its outputs on declared hidden-world/action-loss cases. https://proceedings.mlr.press/v306/jiang26ai.html
- Predictive State Representations and approximate bisimulation/state aggregation are established foundations for sufficient state and decision-preserving abstraction. This PR does not claim those theories as novel. Its narrower contribution is an evaluator-side finite certificate that compares the exact full-state and compressed-state information partitions using the existing #13152 minimax-regret authority. https://web.eecs.umich.edu/~baveja/Papers/psr.pdf
- **ConsolidationBench / governed memory consolidation** is tracked in #9139. Its compression and transfer findings reinforce that compression ratio, reconstruction quality and downstream transfer are separate endpoints; governance properties such as provenance, reversibility and poison resistance are also independent. This PR does not implement memory consolidation or forgetting.

## 9. Temporal integration boundary

A one-epoch certificate is not a guarantee of future decision preservation. The repository already has the correct adjacent owners:
- #11646 / PR #11650: calibrated, contiguous decision-sufficiency horizon for a previously selected action;
- #11655 / PR #11659: simultaneous episode-level calibration of action-by-horizon expected-loss intervals.

These answer a different question from this module. #13239 compares full and compressed information partitions at a declared decision epoch. #11650 evaluates whether a selected action remains sufficiently good over future ticks under calibrated loss bounds. Temporal compression experiments should evaluate both on the same paired episodes, without copying either oracle or conflating their outputs. The separate integration programme is #13255.
