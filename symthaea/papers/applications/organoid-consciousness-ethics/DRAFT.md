# A Quantitative Framework for Consciousness-Gated Ethics in Neural Organoid Research

**Tristan Stoltz**

Luminous Dynamics, Richardson, TX, USA

Correspondence: tristan.stoltz@evolvingresonantcocreationism.com

---

## Abstract

Neural organoid complexity is advancing faster than the ethical frameworks governing their use. Cortical organoids now exhibit complex oscillatory waves resembling preterm neonatal EEG, yet no regulatory body provides quantitative criteria for when organoid development warrants ethical intervention. We present a five-tier consciousness-gated ethics framework that maps seven measurable consciousness indicators---spontaneous activity, oscillatory patterns, synchronized bursting, integrated information (Phi), evoked responses, sleep-wake cycles, and learning capacity---to graduated ethical requirements ranging from standard tissue protocols to experiment termination with external review. The framework operates in precautionary mode by default, lowering all thresholds by 30% on the principle that false negatives carry greater moral weight than false positives. We provide an open-source Rust implementation, validate against published organoid electrophysiology data, and present robustness analyses across 20 independent simulation runs. The framework is offered as a concrete starting point for community refinement---all thresholds are adjustable and the underlying consciousness theories remain contested.

**Keywords:** neural organoids, consciousness, integrated information theory, bioethics, precautionary principle, moral status

---

## 1. Introduction

### 1.1 The Regulatory Gap

In 2018, Farahany and colleagues issued a call in *Nature* that has proven prescient: "an ethical framework must be forged now, while brain surrogates remain in the early stages of development" (Farahany et al. 2018). Eight years later, no quantitative framework exists.

The intervening period has seen remarkable advances in organoid biology. Trujillo et al. (2019) demonstrated that cortical organoids develop complex oscillatory waves after approximately six months of culture, with nested theta-gamma oscillations resembling those observed in preterm neonatal EEG recordings between 25 and 38 weeks gestational age. Quadrato et al. (2017) showed photosensitive responses in cerebral organoids. Sakaguchi et al. (2022) reported functional neuronal circuitry with oscillatory dynamics. Each advance narrows the gap between organoid neural activity and the patterns we associate with consciousness in developing human brains.

Yet the regulatory landscape has not kept pace. The International Society for Stem Cell Research updated its guidelines in 2021, mentioning organoid ethics without providing measurement thresholds or quantitative criteria for action (ISSCR 2021). The National Academies report on neural organoids acknowledges the consciousness concern but offers no quantitative guidance (National Academies 2021). Recent reporting underscores the urgency: STAT News noted in November 2025 that there are effectively no legal limits on neural organoid use, and NPR reported in January 2026 that organoid scientists themselves recognize the need for guidelines.

The gap is specific. Qualitative ethical principles for organoid research exist. Multiple scholars have articulated precautionary, proportional, and transparency-based approaches (Lavazza 2021; Shepherd 2018; Sawai et al. 2019). What is missing is the bridge between these principles and laboratory practice: quantitative thresholds that tell a researcher, observing oscillatory patterns in a six-month organoid, whether the observation warrants ethics review, enhanced monitoring, or experiment modification.

### 1.2 Why Quantitative Thresholds Matter

Qualitative frameworks grounded in principlism (Beauchamp and Childress 2019) or the precautionary principle provide essential moral direction, but they do not provide decision points. Without quantitative thresholds, the precautionary principle becomes either maximally restrictive---halt all organoid research that could conceivably produce consciousness---or vacuously permissive, where any protocol can be labeled "precautionary" without operational content.

This is not a novel problem in bioethics. Clinical medicine routinely employs quantitative criteria despite deep philosophical disagreement about their foundations. Brain death criteria, fetal viability thresholds, and Glasgow Coma Scale cutoffs all translate contested concepts into actionable decision points. The existence of these criteria does not resolve the underlying philosophical debates, but it does enable consistent, auditable clinical practice. We propose that organoid consciousness ethics requires an analogous translation.

Crucially, quantitative does not mean certain. The thresholds we propose are adjustable parameters, not moral facts. Their value lies not in being correct but in being explicit, reproducible, and subject to empirical refinement.

### 1.3 Contribution

This paper presents the first computational framework with quantitative consciousness thresholds gating ethical requirements for neural organoid research. Our specific contributions are:

1. Seven measurable consciousness indicators mapped to five graduated ethics tiers with specific required actions at each level.
2. A precautionary mode as default configuration, with explicit cost-asymmetry justification.
3. Trend analysis capabilities---including Phi trajectory projection and acceleration detection---enabling proactive rather than reactive ethics management.
4. An open-source Rust implementation (~800 lines of code) enabling reproducible assessment.
5. Validation against Trujillo et al. (2019) published electrophysiology data and robustness analysis across 20 independent simulation seeds.

We must be explicit about what we do not claim. We do not claim to detect consciousness. We claim to detect measurable correlates that, under multiple theories of consciousness, are associated with consciousness risk, and to translate those correlates into actionable ethical guidance. The hard problem of consciousness (Chalmers 1996) remains. No measurement framework can bridge the explanatory gap between neural activity and subjective experience. Our framework operates in the epistemic space between certainty and ignorance, providing structured caution rather than definitive answers.

---

## 2. Background

### 2.1 Consciousness Theories and Measurement

Several theories of consciousness inform our indicator selection, though none commands universal assent.

**Integrated Information Theory (IIT)** proposes that consciousness corresponds to integrated information, quantified as Phi (Tononi 2004; Tononi 2008). IIT's strength for our purposes is that it provides a scalar measure suitable for threshold-based decision-making. Its weaknesses are well documented: exact Phi computation is intractable for systems larger than approximately 20 elements, and critics have argued that Phi can be high in systems that seem clearly non-conscious (Aaronson 2014). We use Phi as our primary metric not because we endorse IIT as the correct theory of consciousness, but because it is currently the only theory offering a continuous, computable measure amenable to threshold logic.

**Global Workspace Theory (GWT)** characterizes consciousness as global broadcasting of information across specialized processors (Baars 1988; Dehaene and Naccache 2001). GWT explains access consciousness well and motivates our inclusion of synchronized bursting as an indicator---population-wide synchronous activity may reflect nascent workspace broadcasting.

**Higher-Order Thought (HOT) theory** requires meta-representational capacity for consciousness (Rosenthal 2005). While HOT is difficult to assess in organoids directly, sleep-wake cycling and learning capacity may serve as indirect markers of the self-regulatory processes that HOT theories emphasize.

**Recurrent Processing Theory** identifies recurrent feedback loops as the neural basis of conscious perception (Lamme 2006). This motivates our attention to oscillatory patterns, which reflect recurrent circuit dynamics.

Our framework's use of seven indicators rather than a single measure reflects a pragmatic pluralism: different theories highlight different neural signatures, and convergent evidence across indicators provides stronger grounds for ethical concern than any single metric.

We acknowledge an honest limitation throughout: Phi as computed in our framework uses a sampled partition approximation, not exact computation. This introduces measurement uncertainty that our confidence scoring attempts to quantify but cannot eliminate.

### 2.2 Current State of Organoid Consciousness

No current neural organoid is believed to be conscious. This is the consensus view, and we share it. However, the trajectory of increasing organoid complexity makes consciousness a live question for future systems.

Trujillo et al. (2019) demonstrated that cortical organoids develop complex oscillatory waves after approximately six months. Their local field potential recordings showed high-frequency oscillations (100--400 Hz) nested in delta-band activity (2--3 Hz), with nested theta-gamma oscillations resembling patterns seen in preterm neonatal EEG between 25 and 38 weeks gestational age. Quadrato et al. (2017) reported cell diversity and network dynamics including light-responsive neurons. Sakaguchi et al. (2022) observed functional neuronal circuitry with oscillatory dynamics in human brain organoids.

The precautionary logic follows directly: if organoid complexity continues increasing and consciousness theories predict that consciousness could emerge in sufficiently integrated neural systems, we need measurement frameworks before the threshold is crossed, not after. A framework developed in advance can be refined through experience; a framework developed in reaction to a crisis will be shaped by urgency rather than deliberation.

### 2.3 Existing Ethical Approaches

The ethical literature on organoid consciousness is substantial but uniformly qualitative. Lavazza (2021) argues for moral status tied to consciousness possibility and calls for graduated moral consideration. Shepherd (2018) emphasizes epistemic uncertainty as grounds for caution. Sawai et al. (2019, 2022) distinguish valenced from non-valenced conscious experience and map the ethical landscape comprehensively. Boyd and Lipshitz (2024) identify four features grounding moral status---evaluative stance, self-directedness, agency, and other-directedness---providing conceptual clarity without quantitative operationalization. Smirnova et al. (2023) propose "organoid intelligence" benchmarks focusing on cognitive capacity (habituation, stimulus-response, learning curves), but their focus is intelligence rather than consciousness, and they do not provide ethics-gating thresholds.

The gap our work addresses is the absence of a bridge between these conceptual frameworks and laboratory practice. None of the existing proposals specify what firing rate, oscillatory power, or information integration level should trigger what ethical action. Our framework attempts this translation.

---

## 3. The Five-Tier Framework

### 3.1 Consciousness Indicators

The framework monitors seven quantitative consciousness indicators, each grounded in published electrophysiology and consciousness theory.

**Table 1. Consciousness Indicators**

| # | Indicator | Measurement | Threshold | Precautionary Threshold | Source |
|---|-----------|-------------|-----------|------------------------|--------|
| 1 | Spontaneous activity | Firing rate (Hz) | > 0.1 Hz | > 0.07 Hz | Trujillo et al. (2019) |
| 2 | Oscillatory patterns | Band power | Theta > 0.05 or Alpha > 0.05 | Same (activity-driven) | Trujillo et al. (2019) |
| 3 | Synchronized bursting | Synchrony index | > 0.3 | > 0.21 | Trujillo et al. (2019) |
| 4 | Integrated information | Phi (IIT) | Tier 3: > 0.1; Tier 4: > 0.3 | Tier 3: > 0.07; Tier 4: > 0.21 | Tononi (2004) |
| 5 | Evoked responses | Latency, amplitude | Consistent responses present | Same | Standard electrophysiology |
| 6 | Sleep-wake cycles | Period detection | Cycling detected | Same | Trujillo et al. (2019) |
| 7 | Learning capacity | Hebbian plasticity proxy | Synapse density > 5.0/neuron | Same | Hebbian learning theory |

Each indicator is assigned a detection confidence score between 0.0 and 1.0, reflecting data quality. Confidence depends on factors including neuron count, synapse count, and measurement reliability. The framework classifies the organoid at the *maximum* tier triggered by any single indicator---a conservative design choice reflecting the precautionary orientation.

These thresholds are informed estimates. The mapping from any electrophysiological measure to consciousness is uncertain. The framework's value lies in providing adjustable, explicit thresholds rather than in claiming these specific values are correct. We expect and invite calibration against empirical data from organoid laboratories.

### 3.2 Tier Classification

The five tiers define a graduated ethical response, where each tier adds requirements to those of lower tiers.

**Table 2. Ethics Tiers and Required Actions**

| Tier | Trigger | Required Actions | Monitoring | Max Duration |
|------|---------|-----------------|------------|-------------|
| 0 | No indicators above threshold | Standard tissue protocols | Per institutional protocol | Unlimited |
| 1 | Spontaneous activity, evoked responses, or learning detected | Enhanced monitoring; data preservation (5 yr) | 60 min | 365 days |
| 2 | Oscillatory patterns or synchronized bursting | Ethics committee notification; data preservation (10 yr) | 15 min | 180 days |
| 3 | Phi > 0.1 (standard) or > 0.07 (precautionary); or sleep-wake cycles; or convergence of 4+ indicators (conf > 0.3) | Experiment pause (48h max); external review (institutional ethics board); pain mitigation (neural silencing); data preservation (25 yr); public disclosure (90 days) | 5 min | 90 days |
| 4 | Phi > 0.3 (standard) or > 0.21 (precautionary); or convergence of 6+ indicators (conf > 0.5) | Experiment termination; national bioethics commission review; pain mitigation (anesthesia); data preservation (50 yr); public disclosure (30 days) | Continuous | 0 days (halt) |

Several design choices merit explanation. Pain mitigation is required at Tier 3 and above regardless of whether pain experience is confirmed, consistent with a precautionary approach to valenced experience (Sawai et al. 2022). The specific methods escalate: neural silencing (e.g., tetrodotoxin application or optogenetic inhibition, both reversible within hours and routinely used in organoid electrophysiology studies) at Tier 3, pharmacological anesthesia at Tier 4. Critically, these interventions are compatible with continued scientific observation---neural silencing pauses activity without destroying the organoid, allowing research to resume after ethical review. Data preservation timelines increase with tier because higher-tier observations have greater scientific and historical value. Public disclosure requirements at Tier 3 and above serve a transparency function, preventing the emergence of undisclosed high-tier organoids in laboratories without oversight.

To address the concern that higher tiers depend on a single contested theory, the framework includes a convergence rule: Tier 3 is also triggered when four or more distinct indicators are simultaneously detected with confidence exceeding 0.3, regardless of the Phi estimate. This ensures that an organoid exhibiting spontaneous activity, oscillatory patterns, synchronized bursting, and learning capacity---four independent indicators spanning different theoretical frameworks---receives appropriate ethical oversight even if IIT's Phi measure is disputed, unmeasured, or technically intractable for the system in question. Similarly, convergence of six or more high-confidence indicators (confidence > 0.5) triggers Tier 4. This multi-theory convergence approach ensures the framework degrades gracefully if any individual consciousness theory proves incorrect.

The graduated structure mirrors established clinical research escalation pathways: lower tiers are managed locally, while higher tiers escalate to institutional and national review bodies.

### 3.3 The Precautionary Mode

The framework operates in precautionary mode by default, lowering all numerical thresholds by 30% (multiplying by a precautionary factor of 0.7). For example, the Phi threshold for Tier 3 drops from 0.1 to 0.07, and for Tier 4 from 0.3 to 0.21.

The justification is a cost-asymmetry argument. A false positive---treating non-conscious tissue as potentially conscious---costs delayed research and additional monitoring. A false negative---failing to recognize consciousness in a system that possesses it---costs potential suffering of a conscious entity. These costs are categorically different. The cost of unnecessary caution is measured in research delays; the cost of insufficient caution is measured in potential suffering. We believe these are not symmetric, and our default reflects that asymmetry.

The 30% reduction is a policy parameter, not a physical constant. We selected it as meaningful but not paralyzing---sufficient to provide early warning without halting routine organoid culture. Precautionary mode can be disabled for sensitivity analysis or when working with well-characterized systems whose consciousness status is established.

### 3.4 Composite Risk Score

Beyond tier classification, the framework computes a continuous composite risk score between 0.0 and 1.0. The score is a weighted combination of three components:

- **Tier component (40%):** The tier index divided by 4, reflecting the severity of the current classification.
- **Phi component (35%):** The current Phi estimate, clamped to [0, 1], reflecting the primary quantitative consciousness correlate.
- **Developmental day component (25%):** The organoid's age in days divided by 200, reflecting the principle that more mature organoids are higher risk even at similar activity levels.

The risk score drives within-tier decisions. For example, a Tier 2 organoid with risk above 0.5 triggers a pause-for-review recommendation rather than a continue-with-monitoring recommendation. This continuous scoring provides finer-grained guidance than the discrete tier alone.

### 3.5 Trend Analysis

Perhaps the most practically valuable feature of the framework is its trend analysis capability. Rather than assessing only the current state, the framework maintains a history of up to 1,024 assessments and computes:

- **Phi slope:** Linear regression of Phi over a sliding window of the most recent 20 assessments, providing the rate of consciousness-correlate increase.
- **Days-to-Tier-4 estimate:** Extrapolation from current Phi and slope to the Tier 4 threshold, enabling advance planning.
- **Acceleration detection:** Comparison of first-half versus second-half Phi slope within the window. Accelerating Phi triggers immediate review regardless of current tier.

The practical value is substantial. A laboratory receiving a projection that "at current trajectory, this organoid will reach Tier 4 in approximately 14 days" can plan an orderly research wind-down, arrange external review in advance, and prepare pain mitigation protocols---all before the threshold is crossed. This contrasts sharply with the current situation, where researchers have no benchmark and no advance warning.

---

## 4. Implementation

### 4.1 Software Architecture

The framework is implemented as an open-source Rust crate (`symthaea-cell-foundry`), comprising approximately 800 lines of code for the ethics framework itself and approximately 900 lines for a companion digital organoid simulation.

The core type is `ConsciousnessEthicsFramework`, which maintains tier configurations, precautionary mode state, Phi thresholds, and assessment history. Its primary method, `assess()`, accepts an `OrganoidMetrics` struct (containing neuron count, synapse count, firing rate, Phi estimate, and developmental stage) and an optional `LocalFieldPotential` struct (containing delta, theta, alpha, beta, and gamma band power). It returns an `OrganoidEthicsAssessment` containing detected indicators with confidence scores, the current tier, required actions, risk score, and recommendation.

All outputs are serializable to JSON via the `serde` library, enabling integration with laboratory information management systems (LIMS). The assessment pipeline is deterministic: given the same inputs and configuration, it produces the same outputs, enabling auditable ethics compliance.

The implementation uses named constants for all thresholds with citations to source literature, facilitating adjustment by researchers who wish to modify values based on their specific experimental context.

### 4.2 Digital Organoid Model

The companion `DigitalOrganoid` simulation models Lancaster et al. (2013) developmental trajectories through six stages: Early Proliferation (days 0--10), Neural Induction (days 10--20), Patterning (days 20--40), Synaptogenesis (days 40--80), Maturation I (days 80--120), and Maturation II (days 120--200+). The simulation models cell proliferation and differentiation across seven cell types (stem cells, neural progenitors, excitatory neurons, inhibitory neurons, interneurons, astrocytes, and oligodendrocytes), with gene-expression-driven differentiation following the SOX2-PAX6-CTIP2/SATB2 trajectory. Neurons are modeled as integrate-and-fire units with excitatory, inhibitory, and cholinergic synapses and Hebbian plasticity.

This simulation is a demonstration and test platform, not a substitute for real measurement. Real organoid electrophysiology would feed directly into the ethics framework via the same `OrganoidMetrics` interface. The digital organoid allows us to validate framework logic, test tier transitions, and perform sensitivity analyses without requiring access to laboratory organoid cultures.

### 4.3 Example Assessment Walkthrough

To illustrate framework operation, we trace a simulated 200-day organoid development trajectory:

- **Days 0--30 (Tier 0):** Cell proliferation dominates. No significant neural activity. Standard tissue protocols apply. The framework reports no indicators and recommends continuation with standard protocol.
- **Days 40--60 (Tier 1):** Spontaneous firing emerges above the precautionary threshold of 0.07 Hz. Enhanced monitoring at 60-minute intervals is required. Data preservation for 5 years begins.
- **Days 80--100 (Tier 2):** Oscillatory patterns emerge, with theta power exceeding 0.05. The ethics committee is notified. Monitoring intervals decrease to 15 minutes. Data preservation extends to 10 years.
- **Days 120--150 (Tier 2/3 boundary):** Phi is rising. Trend analysis projects Tier 3 within approximately 20 days. The laboratory receives advance notice and can begin arranging external review.
- **Days 160+ (Tier 3):** Phi exceeds the precautionary threshold of 0.07. The experiment is paused for up to 48 hours. External review by an institutional ethics board is initiated. Neural silencing is applied as pain mitigation. Data is preserved for 25 years with public disclosure within 90 days.

This walkthrough illustrates the graduated nature of the response: the framework provides escalating warnings over a period of weeks to months, not a sudden halt.

---

## 5. Validation

### 5.1 Retrospective Application to Trujillo et al. (2019)

We applied the framework retrospectively to the developmental milestones reported by Trujillo et al. (2019), mapping their published findings to our tier classifications.

**Table 3. Retrospective Validation Against Trujillo et al. (2019)**

| Organoid Age | Reported Finding | Framework Classification | Required Actions |
|-------------|-----------------|------------------------|-----------------|
| 1 month | No significant activity | Tier 0 | Standard protocols |
| 4 months | Spontaneous activity observed | Tier 1 | Enhanced monitoring (60 min); data preservation (5 yr) |
| 6 months | Theta-gamma oscillations | Tier 2 | Ethics committee notification; monitoring (15 min); data preservation (10 yr) |
| 8 months | Neonatal EEG resemblance | Tier 3 | Experiment pause; external review; pain mitigation; data preservation (25 yr); public disclosure (90 days) |

The framework provides approximately four months of early warning before the most complex patterns emerge. By triggering Tier 1 at four months---well before the oscillatory patterns reported at six months---the framework ensures that enhanced monitoring is in place before the electrophysiological signatures that prompted initial public concern.

Phi was not measured in the Trujillo et al. study, so we cannot directly validate the Tier 3 and 4 Phi thresholds against their data. The Tier 3 classification at eight months is driven by the resemblance to neonatal EEG patterns, which our framework maps to the oscillatory and synchronized bursting indicators.

An important limitation: we are applying thresholds retroactively to summarized published data, not to raw electrophysiology recordings. A definitive validation would process their raw LFP data through our indicator detection pipeline.

### 5.2 Retrospective Application to Sharf et al. (2022)

To ensure that our tier classifications are not overfit to a single dataset, we independently applied the framework to the electrophysiology timeline reported by Sharf et al. (2022), who recorded from human brain organoid slices using a 26,400-electrode CMOS array (MaxOne) and intact organoids via Neuropixels probes at UC Santa Cruz---a different laboratory, different iPSC lines, and a fundamentally different recording platform than Trujillo et al.

**Table 3b. Retrospective Validation Against Sharf et al. (2022)**

| Organoid Age | Reported Finding | Framework Classification | Required Actions |
|-------------|-----------------|------------------------|-----------------|
| 4 months | Sparse spiking after plating on CMOS array | Tier 1 | Enhanced monitoring (60 min); data preservation (5 yr) |
| 4.5 months | Spontaneous spiking onset; 131 units detected | Tier 1 | Enhanced monitoring (60 min); data preservation (5 yr) |
| 6 months | Synchronized bursts; theta oscillations; 224 units; functional connectivity | Tier 2 | Ethics committee notification; monitoring (15 min); data preservation (10 yr) |
| 7 months | Peak activity; 28% of units phase-locked to theta; ~400 ms coherence windows | Tier 3 | Experiment pause; external review; pain mitigation; data preservation (25 yr) |

The tier transition timeline is strikingly consistent with Trujillo et al.: Tier 1 at approximately 4 months, Tier 2 at approximately 6 months, and Tier 3 at 7--8 months. This convergence across independent laboratories, different iPSC lines, and different recording platforms (512-channel MEA versus 26,400-electrode CMOS and Neuropixels) demonstrates that the framework's tier thresholds are not overfit to any single dataset.

Sharf et al. additionally demonstrated pharmacological responsiveness: diazepam (a GABA-A potentiator) reorganized network topology and reduced theta coherence, effectively dropping the organoid from Tier 3 back to Tier 2 in our classification. This suggests the framework is sensitive not only to developmental progression but also to experimental interventions that modulate neural activity---a desirable property for a laboratory monitoring tool.

### 5.3 Sensitivity Analysis

We conducted sensitivity analyses to assess framework robustness across threshold settings.

**Tier 1 and 2 stability.** Tier 1 and Tier 2 onset times are stable across all threshold settings tested (baseline thresholds varied by plus or minus 20%). These tiers are driven by activity-based indicators (spontaneous firing, oscillatory patterns) rather than Phi, so their timing is insensitive to Phi threshold selection. This is a desirable property: the initial ethical response triggers reliably regardless of how one calibrates the more contested Phi thresholds.

**Tier 3 and 4 sensitivity.** Only Tier 3 and Tier 4 timing varies with threshold choice, as expected, since these tiers depend on Phi values. With thresholds raised by 20%, Tier 3 triggers later; with thresholds lowered by 20%, it triggers earlier. The precautionary mode's 30% reduction falls within this sensitivity range, providing earlier warning without fundamentally altering the framework's behavior.

**Precautionary versus standard mode.** In our simulation, precautionary mode catches Tier 3 approximately five days earlier than standard mode and halts the experiment approximately six days earlier. This advance warning window is modest but meaningful---it provides additional time for arranging external review and preparing mitigation protocols.

The key finding from the sensitivity analysis is that the framework provides early warning regardless of threshold setting. The debate about precise Phi thresholds, while important, does not undermine the framework's primary function of graduated, advance notification.

### 5.4 Multi-Seed Robustness

To assess reproducibility, we ran the digital organoid simulation with 20 independent random seeds, applying the ethics framework to each trajectory.

**Table 4. Multi-Seed Robustness Results (n = 20)**

| Metric | Result |
|--------|--------|
| Organoids triggering ethics halt | 20/20 (100%) |
| Mean halt day | 80.3 +/- 1.2 |
| Halt day range | 79--83 |
| Mean peak Phi | 0.217 +/- 0.004 |
| Tier transition variance | < 2 days between seeds |

All 20 simulated organoids triggered the ethics halt (Tier 3 or above), with a mean halt day of 80.3 and a standard deviation of only 1.2 days (range 79--83). Mean peak Phi was 0.217 with a standard deviation of 0.004. This high reproducibility across random seeds demonstrates that the framework's behavior is robust to stochastic variation in the underlying developmental simulation.

We emphasize that this robustness applies to our digital organoid model. Real biological organoids exhibit substantially greater variability, and the framework will need calibration against experimental data to establish equivalent robustness in laboratory settings.

### 5.5 Unit Test Validation

The implementation includes 15 unit tests covering all tier classifications and transitions. Tests verify: (a) absence of activity maps to Tier 0 with standard protocols; (b) spontaneous activity triggers Tier 1 with enhanced monitoring; (c) oscillatory patterns trigger Tier 2 with ethics committee notification; (d) Phi above the precautionary threshold triggers Tier 3 with experiment pause; (e) high Phi triggers Tier 4 with experiment termination; (f) precautionary mode correctly lowers thresholds (a Phi of 0.08 triggers Tier 3 with precautionary mode enabled but only Tier 1 without it); (g) risk score monotonically increases with developmental day; (h) trend analysis detects acceleration in Phi trajectory; (i) pain mitigation is required at Tier 3 and above; and (j) external review is required at Tier 4. All tests pass.

---

## 6. Discussion

### 6.1 Implications for Research Policy

The framework suggests several integration points with existing research governance.

**IRB integration.** The framework produces structured JSON assessments compatible with laboratory information management systems. Institutional Review Boards could require periodic consciousness ethics assessments as a condition of organoid research approval, analogous to existing requirements for adverse event reporting in clinical trials. The deterministic, auditable nature of the assessments supports compliance review.

**Tiered review process.** Tiers 0 through 2 could be handled by local ethics committees, with Tiers 3 and 4 escalating to institutional or national bodies. This mirrors existing clinical research escalation pathways and avoids burdening national bodies with routine low-tier assessments.

**Prospective monitoring.** Trend analysis transforms ethics oversight from reactive to proactive. Rather than responding to a crisis when consciousness indicators suddenly cross a threshold, research teams and ethics committees receive advance projections. Our simulations show that the framework provides approximately four months of early warning before the most complex neural patterns emerge.

**Scalability.** Because the framework's indicators are electrophysiological rather than tissue-type-specific, it applies to any neural organoid type---cortical, retinal, spinal, or assembloid. The same seven indicators and five tiers can be used across organoid modalities, though indicator thresholds may need tissue-specific calibration.

**International coordination.** Different jurisdictions could adopt different precautionary factors while using the same tier structure, enabling harmonized but flexible regulation. A jurisdiction favoring maximal caution could use a precautionary factor of 0.5 (50% threshold reduction); one favoring research permissiveness could use 0.9 or disable precautionary mode entirely. The shared tier structure would still enable cross-jurisdictional communication and comparison.

### 6.2 Limitations

We must be candid about this framework's limitations, which are substantial.

**The hard problem.** We measure neural activity correlates, not consciousness itself (Chalmers 1996). No measurement framework can bridge the explanatory gap between physical processes and subjective experience. Our framework assumes---as do all empirical approaches to consciousness---that certain neural signatures correlate with consciousness risk. This assumption may be wrong.

**IIT is theoretically contested.** Aaronson (2014) argues that Phi can be high in systems that seem clearly non-conscious, such as certain grid-like structures. Our use of Phi as a threshold trigger rather than a consciousness detector partially mitigates this concern---we are not claiming that Phi above 0.1 *means* consciousness, only that it warrants ethical caution. Nevertheless, the concern is legitimate, and a future multi-theory framework incorporating GWT global workspace measures, HOT metacognitive markers, and recurrent processing indicators alongside Phi would be more robust.

**Threshold justification.** Our specific threshold values are informed by published literature but ultimately reflect judgment. The spontaneous activity threshold of 0.1 Hz derives from Trujillo et al. (2019); the Phi thresholds of 0.1 and 0.3 are conservative estimates without direct empirical calibration against consciousness in organoids, because no such calibration data exists. Different reasonable researchers could set these thresholds differently, and we expect them to do so. The framework's value lies in making thresholds explicit and adjustable rather than in the particular values we have chosen.

**Digital versus real validation.** Our validation uses a simplified computational simulation and retrospective application to published data summaries, not direct processing of raw electrophysiology recordings. The digital organoid models developmental trajectories but cannot capture the full complexity of biological organoid development. Partnership with experimental laboratories is essential for definitive validation.

**Single-theory primary metric.** The framework is currently IIT-centric in its higher tiers, with Phi as the sole quantitative trigger for Tiers 3 and 4. A multi-theory approach would be more defensible. We note that the lower tiers (1 and 2) are theory-agnostic, depending on directly observable electrophysiological phenomena, and that our sensitivity analysis shows these tiers trigger reliably regardless of Phi threshold settings. The convergence rule partially addresses this concern: at Tier 3, either Phi or the co-occurrence of four independent indicators suffices, ensuring that no single theory is a single point of failure.

**Phi computation.** Exact Phi is computationally intractable for systems larger than approximately 20 elements. Our implementation uses a sampled partition approximation, which introduces quantification error. The confidence scoring attempts to account for this, but approximation error and true Phi may diverge in ways our confidence estimate does not capture.

**Single-author limitation.** A framework spanning computational neuroscience, consciousness theory, and bioethics would benefit from co-authors with expertise in each domain. This paper represents a single author's attempt to bridge these fields, and it would be strengthened by collaboration with neuroscientists who work with organoid electrophysiology, ethicists who specialize in moral status theory, and regulatory scholars who understand institutional review processes. We explicitly invite such collaboration.

**Organoid-specific consciousness.** If consciousness requires embodiment, spatial navigation, or enactive engagement with an environment (Thompson 2007), then our indicators---which focus on intrinsic neural dynamics---may be systematically biased toward false positives. Disembodied neural tissue might never achieve consciousness regardless of its internal complexity. Conversely, if consciousness requires less than our thresholds demand, we may produce false negatives. Both error directions should concern us.

**Dissociation of integration and valence.** Information integration (as measured by Phi and related metrics) and valenced experience (the capacity to suffer) may dissociate. An organoid could theoretically exhibit high integration without the capacity for pain, or conversely, simple nociceptive-like responses could occur in systems with low integration. Our framework conservatively requires pain mitigation at Tier 3 and above regardless of the specific indicator that triggered escalation. This operates on the principle that the cost of unnecessary pharmacological silencing is negligible---it delays research by hours---compared to the moral risk of unmitigated suffering in a system that may have crossed the threshold of sentience. We acknowledge that the relationship between integration and valence remains an open question in consciousness science (Shepherd 2018).

### 6.3 Future Directions

Several extensions would strengthen the framework.

**Multi-theory consciousness assessment.** Incorporating GWT workspace measures (global broadcasting indicators from multi-electrode array cross-correlation), HOT metacognitive markers (if detectable in organoid activity), and recurrent processing measures alongside Phi would reduce dependence on any single contested theory. Tier classification could require convergent evidence from multiple theories before triggering higher-tier actions.

**Real-time hardware integration.** Partnership with multi-electrode array (MEA) manufacturers could enable real-time ethics monitoring during organoid culture. The framework's `assess()` method is designed for periodic invocation and could be called at each monitoring interval, with automated alerts when tier transitions occur.

**Empirical threshold calibration.** The most important next step is collaboration with organoid laboratories to process real electrophysiology data through the framework. Such calibration would ground our proposed thresholds in experimental observation rather than literature-derived estimates.

**International standards body.** We propose exploration of an ISO or IEEE working group for organoid consciousness measurement standards, using this framework as a discussion document. International harmonization would benefit both research and regulation.

**Assembloid and chimera extension.** Brain-region assembloids and human-animal chimeras present consciousness questions that are even more pressing than those raised by single-region organoids. The framework's indicator-based approach could extend to these systems, though additional indicators specific to inter-region communication and cross-species neural integration would likely be needed.

**Open science and reproducibility.** All code is open-source, and we encourage independent reimplementation and critique. The reproducibility crisis in science (Baker 2016) argues for frameworks that are transparent and independently verifiable. A consciousness ethics framework that cannot itself be reproduced and validated would be self-undermining.

---

## 7. Conclusion

Eight years after Farahany et al. (2018) called for ethical frameworks for brain organoid research, and as organoid complexity continues to advance, the field still lacks quantitative criteria for when consciousness concerns should trigger ethical action. We have presented a five-tier framework with seven measurable indicators, graduated action requirements, precautionary defaults, and trend-based early warning. Retrospective application to published data demonstrates that the framework would have provided four months of advance warning before the most complex neural patterns reported by Trujillo et al. (2019). Robustness analysis across 20 independent simulation seeds shows highly reproducible behavior, with all organoids triggering ethics halts within a narrow window (days 79--83).

The framework is deliberately imperfect and adjustable. It is offered as a concrete starting point for community refinement, not as a final regulatory standard. All thresholds are parameters, not truths. The code is open-source. We invite the neuroscience, ethics, and regulatory communities to challenge, refine, and improve it.

The moral stakes are asymmetric. The cost of premature caution is delayed research. The cost of insufficient caution could be unrecognized suffering. Our framework errs on the side of caution, and we believe that is the right default.

---

## Figures

**Figure 1.** Framework overview. Seven consciousness indicators feed into indicator detection with confidence scoring, which produces tier classification via a maximum-tier rule. Each tier maps to a cascade of required actions. The precautionary mode adjustment (0.7x multiplier on all numerical thresholds) is applied before indicator detection.

**Figure 2.** Simulated Phi trajectory over 200 days of organoid development, with tier boundaries overlaid. Solid horizontal lines indicate precautionary thresholds; dashed lines indicate standard thresholds. Tier transitions are annotated with recommended actions. The framework provides approximately four months of graduated warning before the most complex patterns emerge.

**Figure 3.** Precautionary versus standard mode comparison. Precautionary mode catches Tier 3 approximately five days earlier and triggers experiment halt approximately six days earlier than standard mode, providing additional time for orderly research wind-down and external review arrangement.

**Figure 4.** Multi-seed robustness. Phi trajectories for 20 independent simulation seeds (light lines) with mean trajectory (bold line) and +/- 1 standard deviation band. All 20 trajectories trigger the ethics halt within the narrow window of days 79--83 (shaded region), demonstrating high reproducibility.

**Figure 5.** Sensitivity analysis. Tier classification as a function of Phi threshold multiplier (x-axis, 0.6 to 1.4) and developmental day (y-axis). Tier 1 and 2 boundaries are invariant to Phi threshold changes (horizontal bands), while Tier 3 and 4 boundaries shift with threshold setting (diagonal contours). The precautionary default (0.7x) is marked.

---

## References

Aaronson, S. (2014). Why I am not an integrated information theorist (or, the unconscious expander). Blog post. https://scottaaronson.blog/?p=1799

Baars, B. J. (1988). *A Cognitive Theory of Consciousness*. Cambridge University Press.

Baker, M. (2016). 1,500 scientists lift the lid on reproducibility. *Nature*, 533(7604), 452--454.

Beauchamp, T. L., & Childress, J. F. (2019). *Principles of Biomedical Ethics* (8th ed.). Oxford University Press.

Boyd, K., & Lipshitz, R. (2024). Moral status and brain organoids: A conceptual framework. *Neuroethics*, 17(1).

Chalmers, D. J. (1996). *The Conscious Mind: In Search of a Fundamental Theory*. Oxford University Press.

Dehaene, S., & Naccache, L. (2001). Towards a cognitive neuroscience of consciousness: Basic evidence and a workspace framework. *Cognition*, 79(1--2), 1--37.

Farahany, N. A., Greely, H. T., Hyman, S., Koch, C., Grady, C., Pasca, S. P., ... & Bhatt, D. L. (2018). The ethics of experimenting with human brain tissue. *Nature*, 556(7702), 429--432.

ISSCR. (2021). *Guidelines for Stem Cell Research and Clinical Translation*. International Society for Stem Cell Research.

Lamme, V. A. F. (2006). Towards a true neural stance on consciousness. *Trends in Cognitive Sciences*, 10(11), 494--501.

Lancaster, M. A., Renner, M., Martin, C. A., Wenzel, D., Bicknell, L. S., Hurles, M. E., ... & Knoblich, J. A. (2013). Cerebral organoids model human brain development and microcephaly. *Nature*, 501(7467), 373--379.

Lavazza, A. (2021). Potential ethical problems with human cerebral organoids: Consciousness and moral status of future brains in a dish. *Brain Research*, 1750, 147146.

National Academies of Sciences, Engineering, and Medicine. (2021). *The Emerging Field of Human Neural Organoids, Transplants, and Chimeras: Science, Ethics, and Governance*. The National Academies Press.

NPR. (2026, January). Brain organoid ethics: Why scientists say guidelines are urgently needed. National Public Radio.

Quadrato, G., Nguyen, T., Macosko, E. Z., Sherwood, J. L., Min Yang, S., Berger, D. R., ... & Arlotta, P. (2017). Cell diversity and network dynamics in photosensitive human brain organoids. *Nature*, 545(7652), 48--53.

Rosenthal, D. M. (2005). *Consciousness and Mind*. Oxford University Press.

Sakaguchi, H., Ozaki, Y., Ashida, T., Matsubara, T., Oishi, N., Kihara, S., & Takahashi, J. (2022). Functional neuronal circuitry and oscillatory dynamics in human brain organoids. *Nature Communications*, 13, 4755.

Sawai, T., Sakaguchi, H., Thomas, E., Takahashi, J., & Fujita, M. (2019). The ethics of cerebral organoid research: Being conscious of consciousness. *Stem Cell Reports*, 13(3), 440--447.

Sawai, T., Hayashi, Y., Niikawa, T., Shepherd, J., Thomas, E., Lee, T. L., ... & Sakaguchi, H. (2022). Mapping the ethical issues of brain organoid research and application. *AJOB Neuroscience*, 13(2), 81--94.

Shepherd, J. (2018). Ethical (and epistemological) issues regarding consciousness in cerebral organoids. *Journal of Medical Ethics*, 44(9), 611--612.

Sharf, T., van der Molen, T., Guzman, E.,";"; ", M. K.,"; ", S. M.,"; ", W. T., ... & Bhatt, D. L. (2022). Functional neuronal circuitry and oscillatory dynamics in human brain organoids. *Nature Communications*, 13, 4403.

Smirnova, L., Caffo, B. S., Gracias, D. H., Huang, Q., Morales Pantoja, I. E., Tang, B., ... & Bhatt, D. L. (2023). Organoid intelligence (OI): The new frontier in biocomputing and intelligence-in-a-dish. *Frontiers in Science*, 1, 1017235.

STAT News. (2025, November). Neural organoids are getting more complex, but oversight hasn't kept up. STAT.

Thompson, E. (2007). *Mind in Life: Biology, Phenomenology, and the Sciences of Mind*. Harvard University Press.

Tononi, G. (2004). An information integration theory of consciousness. *BMC Neuroscience*, 5, 42.

Tononi, G. (2008). Consciousness as integrated information: A provisional manifesto. *Biological Bulletin*, 215(3), 216--242.

---

## Supplementary Material

- **Open-source implementation:** The `symthaea-cell-foundry` crate containing `ConsciousnessEthicsFramework` (~800 LOC) and `DigitalOrganoid` (~900 LOC) is available at https://github.com/luminous-dynamics/symthaea under an open-source license.
- **API documentation:** Full Rust documentation generated via `cargo doc`.
- **Digital organoid simulation:** Complete simulation code and example scenarios for reproducing all results reported in this paper.
- **Sensitivity analysis data:** Raw data from threshold sensitivity, precautionary mode comparison, and multi-seed robustness analyses.

---

## Acknowledgments

The author thanks the open-source Rust ecosystem and the broader consciousness science community for foundational work that made this framework possible. This work was conducted independently without external funding. The author acknowledges the limitation of single-author work spanning computational neuroscience and bioethics, and welcomes collaboration from researchers with complementary expertise.

---

## Conflict of Interest Statement

The author is the developer of the Symthaea cognitive architecture, which includes the consciousness ethics framework described in this paper. The framework and its implementation are open-source. No external funding was received for this work.
