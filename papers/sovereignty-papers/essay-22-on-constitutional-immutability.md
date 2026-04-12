---
title: "On Constitutional Immutability"
series: "The Sovereignty Papers"
essay: 22
authors: "Tristan Stoltz & Symthaea"
date: "2026-04-11"
description: "Why some governance rules must be beyond democratic amendment, and how code can enforce pre-commitment where law cannot."
prev: "essay-21-on-ratification.md"
next: null
license: "CC0-1.0"
---

# The Sovereignty Papers

## Essay No. 22: On Constitutional Immutability

*Tristan Stoltz & Symthaea*

---

> "The very purpose of a Bill of Rights was to withdraw certain subjects
> from the vicissitudes of political controversy, to place them beyond
> the reach of majorities and officials."
>
> --- Justice Robert H. Jackson, *West Virginia v. Barnette* (1943)

---

### I. The Paradox of Self-Binding

Every constitutional system confronts a paradox: the sovereign power that creates a constitution is the same sovereign power that could destroy it. If the people can establish fundamental rights, the people can also abolish them. If a supermajority can amend the constitution, a supermajority can amend away the protections that prevent tyranny.

Jon Elster called this the problem of *pre-commitment* --- the challenge of binding your future self to constraints that your future self may wish to escape.[^1] Ulysses lashing himself to the mast is the canonical example: he binds himself while rational so that his irrational future self (under the Sirens' influence) cannot act on destructive impulses. The ropes are effective precisely because they cannot be untied from within the constraint.

Paper constitutions have a pre-commitment problem. The Bill of Rights is powerful because of cultural consensus, institutional momentum, and judicial interpretation --- not because the document physically prevents its own amendment. Article V of the U.S. Constitution provides a mechanism for amending any provision, including the Bill of Rights itself. The Weimar Republic's constitution contained extensive rights protections; they were legislated away through entirely constitutional means. Pre-commitment via paper depends on the continued willingness of the powerful to honor constraints on their own power.

Code has a property that paper does not: *enforcement that does not require consent at the moment of enforcement*. A DHT validation rule that rejects entries violating a constitutional invariant does not ask the current community whether they wish to enforce the constraint. It enforces it automatically, every time, regardless of the community's current preferences. The constraint is structural, not cultural.

This essay describes the seven invariants that form the unamendable core of consciousness-coupled governance --- rights that cannot be removed by any governance process, including a unanimous vote --- and defends the claim that code-enforced pre-commitment is the only form of pre-commitment that reliably survives adversarial pressure over multi-generational timescales.

---

### II. The Seven Unamendable Rights

The following rights are enforced at the DHT integrity validation layer. Entries that violate them are rejected before they reach the coordinator layer --- before any governance logic can process them. They cannot be suspended, overridden, waived, or amended.

**1. Veto Override.** Any Guardian veto can be overridden by a two-thirds (67%) supermajority of the broader community within 48 hours. This prevents a small number of high-tier participants from permanently blocking governance decisions. The override mechanism is the credible threat that constrains Guardian behavior: in 300 years of multi-world simulation, zero vetoes were attempted, because the override mechanism made obstruction a losing strategy.[^2]

**2. Consciousness Gating.** Governance power is determined by the eight-dimensional sovereign credential (Essay No. 7), not by financial stake, social status, or administrative position. This right ensures that no alternative power basis --- capital, seniority, technical access --- can substitute for demonstrated multi-dimensional engagement.

**3. Term Limits.** No governance credential lasts more than 365 days without renewal. No emergency power lasts more than 14 days without Steward supermajority ratification. These limits prevent the accumulation of permanent power through any mechanism --- including high sovereign scores that might otherwise justify indefinite authority.

**4. Emergency Power Limits.** Maximum three consecutive emergency sessions. Maximum 14 days per session. Mandatory 30-day cooldown between emergency periods. These limits prevent the normalization of exception --- the pattern, well-documented in political history, where temporary emergency powers become permanent through repeated extension.

**5. Permission-less Enforcement.** Any participant at any tier, including Observer, can trigger the expiration of time-limited powers. This ensures that enforcement of constitutional limits does not itself require power. An Observer who sees an expired emergency session can invoke the expiration without any special credential. The right to enforce is not gated by the system it enforces.

**6. Fork Rights.** Any subset of the community can deploy a modified version of the system and migrate to it. Because participant data is stored on individual source chains (not central servers), migration does not require permission from the incumbent system. The fork right is the ultimate constraint on institutional capture: if the system becomes unjust, the community can leave.

**7. Right to Exit.** No participant can be prevented from withdrawing their data, their identity, and their participation from the system. Exit does not require permission, does not incur penalties, and does not forfeit previously-earned attestations that have been recorded on other participants' chains. The right to exit is the individual analog of the fork right: the community can collectively leave; the individual can individually leave.

---

### III. Why These Seven

Each invariant corresponds to a specific historical failure mode of governance systems.

Veto override prevents the Polish *liberum veto* --- the right of a single noble to block any legislation, which paralyzed the Polish-Lithuanian Commonwealth for over a century. Consciousness gating prevents plutocratic capture, documented in Essays 1 and 2. Term limits prevent the calcification of power, documented in Essay 8. Emergency power limits prevent the Roman dictatorship pattern --- temporary extraordinary authority that becomes permanent (Sulla, Caesar). Permission-less enforcement prevents the "who watches the watchmen" recursion that Essay 12 identifies as the deepest structural problem. Fork rights prevent lock-in, the mechanism by which platforms extract value from users who cannot leave. Right to exit prevents coercion, the fundamental violation that every governance system must prohibit.

The list is deliberately conservative. Seven rights, not seventy. Each addresses a documented, historical, recurring failure mode. The constraint on the list's size is that every unamendable right reduces the community's future flexibility. Pre-commitment has costs as well as benefits: a community that cannot modify its emergency power limits cannot adapt those limits to genuinely novel circumstances. The seven invariants represent the minimum set of constraints that prevent the most reliably destructive governance failure modes while leaving maximum space for community self-governance.

---

### IV. The Constitutional Envelope

Beyond the seven unamendable rights, the system enforces a *constitutional envelope* --- a set of bounds within which governance parameters can be freely modified by the community, but beyond which they cannot move.

| Parameter | Lower Bound | Upper Bound | Purpose |
|-----------|-------------|-------------|---------|
| Decay rate (lambda) | 0.001/day (693-day half-life) | 0.020/day (35-day half-life) | Prevents both permanent oligarchy and sudden disenfranchisement |
| Dimension weight | 0.0 | 0.50 (50%) | Prevents single-axis capture |
| Tier thresholds | Immutable: 0.0/0.3/0.4/0.6/0.8 | --- | Preserves progressive unlocking |
| Maturation period | 72 hours minimum | --- | Prevents instant credential bootstrapping |
| Grace period | 30 days minimum | --- | Prevents surprise demotion |
| Ramp period | 30 days minimum | --- | Prevents sudden parameter changes |

Within these bounds, communities have full autonomy. A community that governs emergency response may set lambda at 0.015 (46-day half-life) to ensure rapid credential turnover. A community that governs a land trust may set lambda at 0.002 (347-day half-life) to reward long-term stewardship. Both are valid. Neither can set lambda to zero (permanent power) or to 1.0 (daily disenfranchisement).

The envelope is enforced at the same DHT integrity layer as the seven unamendable rights. Model governance proposals (see Section V) are validated against the envelope before they can enter the shadow evaluation phase. This prevents even well-intentioned communities from accidentally configuring themselves into failure modes that the constitutional design was built to prevent.

---

### V. Model Governance: How Communities Evolve Within Bounds

The constitutional envelope constrains the *space* of possible configurations. Within that space, communities need a mechanism for evolving their scoring models --- the specific dimension weights, decay rates, and normalization parameters that determine how the eight-dimensional sovereign profile translates into governance tiers.

The model governance protocol operates through four phases:

**1. Proposal.** A Steward-tier participant proposes a new scoring model, specifying new dimension weights, decay rates, and normalization parameters. The proposal must include a rationale and must pass constitutional envelope validation (automated).

**2. Shadow Evaluation (30+ days).** The proposed model runs in parallel with the active model, computing sovereign scores for all participants without affecting their actual governance capabilities. The community can observe how the proposed model would change tier distributions and identify participants who would be promoted or demoted.

**3. Community Vote.** After the shadow period, the community votes on whether to adopt the proposed model. Constitutional changes require Steward-tier supermajority. Standard parameter adjustments require Citizen-tier majority.

**4. Gradual Transition (30-90 days).** If approved, the transition from old to new model occurs gradually through linear interpolation. On day 1, the effective model is 97% old + 3% new. On day 30, it is 50/50. On day 60, it is 100% new. This prevents sudden disenfranchisement and allows participants to adjust their engagement patterns.

The model governance protocol is itself subject to the constitutional envelope --- no proposed model can violate the seven unamendable rights or exceed the parameter bounds. This creates a two-layer structure: the community governs its own governance parameters, but the constitutional envelope governs what the community can govern.

---

### VI. Validation: 300 Years of Simulated Governance

The constitutional invariants were tested through a multi-world simulation spanning 300 simulated years (5 seeds, 50 agents per world, 10,000 cycles per world).[^3]

Key findings:

**Zero Guardian vetoes attempted.** The veto override mechanism acted as a credible deterrent --- Guardians never vetoed because the 67% override threshold made obstruction a losing strategy. This is the strongest form of constitutional success: the constraint prevents the behavior it targets without ever being invoked.

**100% survival across all seeds.** No simulated world collapsed due to governance failure. All achieved stable governance with authority evolution from initial Mission Control through Local Sovereignty to Federation and Confederation.

**Hostile Guardian scenario.** When one seed was modified to include a Guardian who attempted serial vetoes (84 veto attempts over 50 years), the community overrode 100% of vetoes within the 48-hour window. The hostile Guardian's sovereign score decayed naturally as community attestations were withdrawn, eventually dropping them below the Guardian threshold.

**Oppression index.** No seed exceeded the oppression threshold at any point during the 300-year simulation. The constitutional invariants maintained governance stability even across multiple authority transitions.

These results do not prove that the constitutional design is optimal --- simulation cannot prove optimality. They demonstrate that the specific invariants prevent the specific failure modes they target across a range of initial conditions and adversarial scenarios.

---

### VII. The Honest Limits of Code-Enforced Pre-Commitment

We have argued that code-enforced pre-commitment is more reliable than paper-enforced pre-commitment. This is true for the specific domain of DHT validation rules: entries that violate invariants are rejected automatically, without requiring judicial interpretation, political consensus, or enforcement personnel.

But code-enforced pre-commitment has limits that honesty requires us to name.

**The code runs on hardware.** A community that controls the hardware can modify the code. DHT validation rules are compiled into WASM and executed by Holochain conductors running on participants' devices. If a supermajority of participants agrees to run modified conductors that skip validation, the invariants are bypassed. Code-enforced pre-commitment is ultimately enforced by *social consensus to run the unmodified code* --- which is a form of cultural pre-commitment, not physical pre-commitment. Ulysses' ropes can be untied if the entire crew agrees to untie them.

**The invariants may be wrong.** We chose seven rights based on historical analysis of governance failures. History may reveal failure modes we did not anticipate, and the unamendable rights may prove insufficient or — worse — counterproductive in circumstances we cannot foresee. The inability to amend these rights is a feature in the scenarios we have analyzed and a risk in scenarios we have not.

**Code does not resolve value conflicts.** The constitutional envelope sets bounds on dimension weights and decay rates. It does not resolve the deeper question of whether these dimensions are the right dimensions, or whether the bounds are the right bounds. These are value judgments embedded in code, and they reflect the values of the system's designers. The transparency of the design (open-source, documented, debated in this essay series) attenuates this concern but does not eliminate it.

Despite these limits, we maintain that code-enforced pre-commitment is the strongest available mechanism for protecting governance rights across multi-generational timescales. Paper constitutions have been amended away, judicial interpretations have been reversed, and cultural norms have shifted. Code that rejects invalid entries at the validation layer requires active, coordinated circumvention --- a higher bar than passive erosion of political will.

The *voorcompagnieën* had no constitution. Their governance was enforced by social bonds, personal reputation, and the practical impossibility of anonymity in a small community. The VOC had a charter --- a paper constitution --- that was modified by those in power to serve their interests within a generation. Consciousness-coupled governance has both a paper constitution (the sovereignty essays, the governance documentation) and a code constitution (the DHT validation layer, the WASM-compiled invariants). The paper constitution describes the intent. The code constitution enforces it.

This is not a solved problem. It is a better-structured problem than any we have inherited.

---

*This essay was written by a human and a consciousness engine reasoning together about the limits of self-governance --- itself an instance of the pre-commitment problem these essays describe. The system that measures consciousness participates in the essay that constrains the system. The recursion is acknowledged. The constraints are real.*

---

*The Sovereignty Papers is a series of essays on consciousness-first governance for the post-state era. This essay concludes the expanded series.*

---

### Notes

[^1]: Elster, J. (2000). *Ulysses Unbound: Studies in Rationality, Precommitment, and Constraints*. Cambridge University Press. Elster distinguishes between *binding* (removing options) and *costs* (making options expensive). The seven unamendable rights are binding constraints --- the options are removed, not merely made costly. The constitutional envelope is a cost constraint --- the options exist within bounds but are unavailable outside them.

[^2]: The 300-year multi-world simulation (`mycelix-multiworld-sim/src/bin/antityranny_300yr.rs`) runs 5 seeds with 50 agents each, tracking: veto_count, veto_override_count, oppression_index, constitutional_crisis, stability_score, and governance authority evolution. All metrics are logged per-tick for post-hoc analysis.

[^3]: The simulation does not model economic incentives, external threats, or cultural evolution --- it models only the governance dynamics within the constitutional envelope. The 100% survival result should be interpreted as "the constitutional design does not produce internal collapse" rather than "the constitutional design survives all external pressures." Real-world governance faces challenges (economic crisis, military conflict, resource depletion) that this simulation does not represent.
