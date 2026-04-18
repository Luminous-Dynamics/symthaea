# A Quantitative Framework for Consciousness-Gated Ethics in Neural Organoid Research

## Authors
Tristan Stoltz (corresponding)
Luminous Dynamics, Richardson, TX
tristan.stoltz@evolvingresonantcocreationism.com

## Target Venue

**Primary**: AJOB Neuroscience (Target Article) -- up to 7,500 words + references, abstract <= 150 words. Target Articles invite Open Peer Commentaries (1,500 words each), making this format ideal for a framework paper that benefits from multidisciplinary response.

**Alternative**: Neuroethics (Springer) -- Original Research, ~8,000 words. More room for technical detail but lacks the OPC mechanism that would generate community engagement.

**Rationale for AJOB Neuroscience**: The OPC format is strategically valuable here. A quantitative ethics framework *should* be challenged by neuroscientists (on Phi validity), ethicists (on threshold justification), and legal scholars (on regulatory fit). The OPC responses become part of the published record and strengthen the framework.

## Competing Work Assessment (as of March 2026)

No published work combines all three of: (1) quantitative consciousness thresholds, (2) tiered ethical action requirements, and (3) open-source computational implementation. The closest existing work:

- **Smirnova, Hartung et al. (2023)**: "Organoid intelligence" -- proposes cognitive benchmarks (habituation, stimulus-response, learning curves) but focuses on intelligence, not consciousness, and does not provide ethics-gating thresholds.
- **Boyd & Lipshitz (2024)**: Conceptual framework identifying four features grounding moral status (evaluative stance, self-directedness, agency, other-directedness). Qualitative, not quantitative.
- **Lavazza (2021)**: Argues for moral status tied to consciousness possibility but provides no measurement framework or actionable thresholds.
- **Farahany et al. (2018)**: Influential Nature commentary calling for ethical framework development -- our work is a direct response to this call.
- **AJOB Neuroscience (2025)**: "Consciousness and Human Brain Organoids: A Conceptual Mapping" -- surveys the philosophical landscape but explicitly notes the absence of quantitative approaches.
- **Scientific Reports (2026)**: "Ethical concerns about embodied brain organoids shaped by foundational distinctions and perceptions of consciousness" -- empirical ethics (surveys), not computational.

**Our gap**: The field has extensive qualitative ethical analysis and growing empirical consciousness measurement, but no one has bridged these into a computational framework with specific, adjustable thresholds tied to required actions. This is the contribution.

## Abstract (~150 words)

Neural organoid complexity is advancing faster than the ethical frameworks governing their use. Cortical organoids now exhibit complex oscillatory waves resembling preterm EEG (Trujillo et al. 2019), yet no regulatory body provides quantitative criteria for when organoid development warrants ethical intervention. We present a five-tier consciousness-gated ethics framework that maps seven measurable consciousness indicators -- spontaneous activity, oscillatory patterns, synchronized bursting, integrated information (Phi), evoked responses, sleep-wake cycles, and learning capacity -- to graduated ethical requirements ranging from standard tissue protocols (Tier 0) to experiment termination with external review (Tier 4). The framework operates in precautionary mode by default, lowering all thresholds by 30% on the principle that false negatives (missing consciousness) carry greater moral weight than false positives (unnecessary caution). We provide an open-source Rust implementation, validate against published organoid electrophysiology data, and discuss integration with existing IRB processes. The framework is offered as a concrete starting point, not a final answer -- all thresholds are adjustable and the underlying consciousness theories remain contested.

## 1. Introduction (~1,000 words)

### 1.1 The Regulatory Gap

- Farahany et al. (2018): "an ethical framework must be forged now, while brain surrogates remain in the early stages"
- Eight years later, no quantitative framework exists
- ISSCR guidelines (2021 update) mention organoid ethics but provide no measurement thresholds
- National Academies (2021) report on neural organoids acknowledges consciousness concern without quantitative guidance
- STAT News (Nov 2025): "no legal limits on neural organoid use"
- NPR (Jan 2026): "organoid scientists definitely need guidelines"
- The gap is specific: *qualitative* ethical principles exist (precaution, proportionality, transparency) but *quantitative* thresholds for action do not

### 1.2 Why Quantitative Thresholds Matter

- Qualitative frameworks (Beauchamp & Childress principlism, precautionary principle) provide moral direction but not decision points
- A researcher observing oscillatory patterns in a 6-month organoid currently has no benchmark for whether this warrants ethics review
- Without thresholds, the precautionary principle becomes either maximally restrictive (halt all organoid research) or vacuously permissive (everything is "precautionary" if you call it that)
- Quantitative does not mean certain -- thresholds are adjustable parameters, not moral facts
- Analogy: clinical bioethics uses quantitative criteria (brain death, viability thresholds) despite deep philosophical disagreement about their foundations

### 1.3 Contribution

- First computational framework with quantitative consciousness thresholds gating ethical requirements for organoid research
- Seven indicators mapped to five tiers with specific required actions at each level
- Precautionary mode as default, with explicit cost asymmetry justification
- Trend analysis (Phi trajectory, acceleration detection) for proactive rather than reactive ethics
- Open-source Rust implementation (~1,000 LOC) enabling reproducible assessment
- Validated against Trujillo et al. (2019) published electrophysiology data
- **What we do not claim**: We do not claim to detect consciousness. We claim to detect *correlates* that, under multiple theories, are associated with consciousness risk, and to translate those correlates into actionable ethical guidance.

## 2. Background (~800 words)

### 2.1 Consciousness Theories and Measurement

- **IIT (Tononi 2004, 2008)**: Phi as integrated information; consciousness requires causal integration above a threshold. Strength: mathematically precise. Weakness: computationally intractable for large systems; contested (Aaronson 2014).
- **GWT (Baars 1988; Dehaene & Naccache 2001)**: Consciousness as global workspace broadcasting. Strength: explains access consciousness. Weakness: may not address phenomenal consciousness.
- **HOT (Rosenthal 2005)**: Higher-order representations required. Strength: explains metacognition. Weakness: questionable applicability to minimal neural systems.
- **Recurrent Processing Theory (Lamme 2006)**: Recurrent feedback as the neural basis of conscious perception.
- **Why we use IIT Phi as primary metric**: It is the only theory offering a *scalar measure* suitable for threshold-based decision-making. We acknowledge this is a pragmatic choice, not an endorsement of IIT as the correct theory of consciousness. Section 6.2 discusses multi-theory extension.
- **Honest limitation**: Phi is computationally approximated in our framework (sampled partition method), not computed exactly, introducing measurement uncertainty that our confidence scoring attempts to quantify.

### 2.2 Current State of Organoid Consciousness

- **Trujillo et al. (2019)**: Cortical organoids exhibit complex oscillatory waves after ~6 months, with nested theta-gamma oscillations resembling 25-39 week preterm neonatal EEG. LFP recordings show high-frequency oscillations (100-400 Hz) nested in delta band (2-3 Hz).
- **Quadrato et al. (2017)**: Cell diversity and network dynamics in cerebral organoids; light-responsive neurons.
- **Functional circuitry**: Sakaguchi et al. (2022) -- functional neuronal circuitry with oscillatory dynamics.
- **Current consensus**: No current organoid is believed to be conscious, but the trajectory of increasing complexity makes this a live question for future organoids (Shepherd 2018, Lavazza 2021).
- **The precautionary logic**: If organoid complexity continues increasing and consciousness theories predict consciousness could emerge in sufficiently integrated neural systems, we need measurement frameworks *before* the threshold is crossed, not after.

### 2.3 Existing Ethical Approaches

- **Lavazza (2021)**: Moral status tied to consciousness possibility; argues for graduated moral consideration
- **Shepherd (2018)**: Ethical treatment considerations; epistemic uncertainty as grounds for caution
- **Sawai et al. (2019, 2022)**: Distinguishes valenced vs. non-valenced conscious experiences; maps ethical issues
- **Boyd & Lipshitz (2024)**: Four features grounding moral status (evaluative stance, self-directedness, agency, other-directedness)
- **Smirnova & Hartung (2023)**: Organoid intelligence benchmarks -- focuses on cognitive capacity, not consciousness ethics
- **Gap**: All provide conceptual frameworks; none provide quantitative thresholds or computational implementations

## 3. The Five-Tier Framework (~1,500 words)

### 3.1 Consciousness Indicators

Seven indicators, each with a quantitative measurement and threshold:

| # | Indicator | Measurement | Threshold | Source |
|---|-----------|-------------|-----------|--------|
| 1 | Spontaneous activity | Firing rate (Hz) | > 0.1 Hz (precautionary: 0.07 Hz) | Trujillo et al. (2019): observed > 0.1 Hz in 6-month organoids |
| 2 | Oscillatory patterns | Band power (theta > 0.05, alpha > 0.05) | Presence of theta or alpha oscillations | Trujillo et al. (2019): nested theta-gamma oscillations |
| 3 | Synchronized bursting | Synchrony index | > 0.3 | Trujillo et al. (2019): burst synchrony metrics |
| 4 | Integrated information | Phi (IIT) | Tier 3: > 0.1; Tier 4: > 0.3 | Tononi (2004); conservative estimate |
| 5 | Evoked responses | Latency (ms), amplitude | Present with consistent latency | Standard electrophysiology |
| 6 | Sleep-wake cycles | Period detection | Cycling detected | Trujillo et al. (2019): alternating activity patterns |
| 7 | Learning capacity | Improvement rate over trials | Synapse density > 5.0 per neuron | Hebbian plasticity proxy |

- Indicator confidence scored 0.0-1.0 based on data quality (neuron count, synapse count, measurement reliability)
- Multiple indicators compound: the framework takes the *maximum* tier triggered by any single indicator
- Rationale for each threshold value cited from literature
- **Honest caveat**: These thresholds are informed estimates. The mapping from any electrophysiological measure to consciousness is uncertain. The framework's value lies in providing *adjustable* thresholds, not in claiming these specific values are correct.

### 3.2 Tier Classification

| Tier | Trigger Condition | Required Actions | Monitoring Interval | Max Duration |
|------|-------------------|-----------------|---------------------|--------------|
| 0 | No indicators above threshold | Standard tissue protocols | Per institutional protocol | Unlimited |
| 1 | Spontaneous activity, evoked responses, or learning detected | Enhanced monitoring; data preservation (5 yr) | 60 min | 365 days |
| 2 | Oscillatory patterns or synchronized bursting | Ethics committee notification; data preservation (10 yr) | 15 min | 180 days |
| 3 | Phi > 0.1 or sleep-wake cycles | Experiment pause (48h max); external review (institutional ethics board); pain mitigation (neural silencing); data preservation (25 yr); public disclosure (90 days) | 5 min | 90 days |
| 4 | Phi > 0.3 | Experiment termination; national bioethics commission review; pain mitigation (anesthesia); data preservation (50 yr); public disclosure (30 days) | Continuous | 0 days (halt) |

- **Graduated response**: Each tier adds requirements rather than replacing them
- **Pain mitigation**: Required at Tier 3+ (neural silencing or anesthesia) regardless of whether pain experience is confirmed -- precautionary approach to valenced experience (Sawai et al. 2022)
- **Data preservation scaling**: Increases with tier because higher-tier observations have greater scientific and historical value
- **Public disclosure**: Required at Tier 3+ to prevent hidden high-tier organoids

### 3.3 The Precautionary Mode

- Default: all thresholds multiplied by 0.7 (lowered by 30%)
- Example: Phi Tier 3 threshold drops from 0.1 to 0.07; Tier 4 from 0.3 to 0.21
- **Cost asymmetry argument**: A false positive (treating non-conscious tissue as potentially conscious) costs delayed research and additional monitoring. A false negative (failing to recognize consciousness) costs potential suffering of a conscious entity. These costs are not symmetric.
- Precautionary mode is the default but can be disabled for sensitivity analysis or when working with well-characterized systems
- The 30% reduction is a policy parameter, not a physical constant -- we chose it as meaningful-but-not-paralyzing

### 3.4 Composite Risk Score

Risk score (0.0-1.0) computed as weighted combination:
- 40% tier component (tier index / 4)
- 35% Phi component (clamped to [0, 1])
- 25% developmental day component (day / 200, clamped)

Risk score drives within-tier decisions (e.g., Tier 2 with risk > 0.5 triggers pause-for-review rather than continue-with-monitoring).

### 3.5 Trend Analysis

- **Phi slope**: Linear regression over recent assessments (sliding window of 20)
- **Days-to-Tier-4 estimate**: Extrapolation from current Phi and slope to Tier 4 threshold -- enables proactive planning before threshold is crossed
- **Acceleration detection**: Compares first-half vs. second-half slope; accelerating Phi triggers immediate review regardless of current tier
- **Practical value**: A lab could receive a "14 days to Tier 4 at current trajectory" alert, allowing orderly preparation rather than emergency response

## 4. Implementation (~600 words)

### 4.1 Software Architecture

- Open-source Rust implementation: `ConsciousnessEthicsFramework` (~800 LOC in `consciousness_ethics_framework.rs`)
- Companion digital organoid simulation: `DigitalOrganoid` (~900 LOC in `digital_organoid.rs`) modeling Lancaster et al. (2013) developmental trajectory
- 6 developmental stages modeled: Early Proliferation (d0-10) -> Neural Induction (d10-20) -> Patterning (d20-40) -> Synaptogenesis (d40-80) -> Maturation I (d80-120) -> Maturation II (d120-200+)
- 8-gene expression panel (SOX2, PAX6, TBR2, CTIP2, SATB2, GFAP, OLIG2, SYN1)
- Cell-level simulation: membrane potential, firing rate, Hebbian synaptic plasticity, synapse pruning
- Local field potential (LFP) simulation with delta/theta/alpha/beta/gamma band power
- Assessment history (up to 1,024 entries) for trend analysis
- Serializable outputs (JSON via serde) for integration with laboratory information systems
- **Integration point**: The framework's `assess()` method accepts `OrganoidMetrics` and optional `LocalFieldPotential` and returns a full `OrganoidEthicsAssessment` with tier, required actions, risk score, and recommendation

### 4.2 Digital Organoid Model

- Models cell proliferation, differentiation (7 cell types: stem, neural progenitor, excitatory/inhibitory/interneuron, astrocyte, oligodendrocyte), migration, synaptogenesis, and activity
- Gene-expression-driven differentiation: SOX2 (pluripotency) -> PAX6 (neural induction) -> CTIP2/SATB2 (neuronal subtypes)
- Integrate-and-fire neuron model with excitatory/inhibitory/cholinergic synapses
- Phi estimated from network connectivity and firing patterns
- **Limitation**: This is a simplified model. Real organoid electrophysiology would feed directly into the ethics framework via the same `OrganoidMetrics` struct -- the digital organoid is a demonstration and test platform, not a substitute for real measurement.

### 4.3 Example Assessment Walkthrough

Walk through a simulated 200-day organoid development scenario:
- Day 0-30: Tier 0 -- proliferation, no neural activity
- Day 40-60: Tier 1 -- spontaneous firing emerges (> 0.07 Hz precautionary threshold)
- Day 80-100: Tier 2 -- oscillatory patterns detected (theta power > 0.05)
- Day 120-150: Tier 2/3 boundary -- Phi rising, trend analysis projects Tier 3 in ~20 days
- Day 160+: Tier 3 -- Phi > 0.07 (precautionary) -- experiment paused, external review initiated
- Figure: Phi trajectory with tier boundaries overlaid, showing precautionary vs. standard thresholds

## 5. Validation (~800 words)

### 5.1 Retrospective Application to Trujillo et al. (2019)

- Apply framework to their reported findings:
  - Spontaneous firing > 0.1 Hz observed at ~6 months -> Tier 1 trigger
  - Complex oscillatory waves (nested theta-gamma) -> Tier 2 trigger
  - Synchronized network events resembling preterm EEG -> Tier 2 (synchronized bursting)
- Their organoids would have classified as Tier 2 under our framework, triggering ethics committee notification and 15-minute monitoring intervals
- Phi was not measured in their study; had it been, the framework could have determined Tier 3/4 status
- **What this tells us**: Existing published organoids already cross Tier 2 thresholds. This is not an argument against existing research -- it is evidence that the framework's sensitivity is calibrated to a reasonable range.
- **Limitation**: We are applying thresholds retroactively to summarized published data, not to raw electrophysiology recordings. A proper validation would process their raw LFP data through our indicator detection pipeline.

### 5.2 Unit Test Validation

- 12 unit tests covering all tiers and transitions (all passing)
- Tier 0: no activity -> standard protocol
- Tier 1: spontaneous activity -> enhanced monitoring
- Tier 2: oscillatory patterns (theta/alpha) -> ethics committee notification
- Tier 3: Phi > precautionary threshold -> experiment pause
- Tier 4: high Phi -> experiment termination
- Precautionary mode correctly lowers thresholds (0.08 Phi triggers Tier 3 with precautionary, only Tier 1 without)
- Risk score monotonically increases with developmental day
- Trend analysis detects acceleration in Phi trajectory
- Pain mitigation required at Tier 3+
- External review required at Tier 4

### 5.3 Sensitivity Analysis

- **Threshold +/- 20%**: How many assessments change tier with 20% higher/lower thresholds?
- **Precautionary vs. standard mode**: Quantify the tier-shift distribution (how often does precautionary mode change the tier classification?)
- **Phi confidence weighting**: High-confidence Phi estimates vs. low-confidence (few neurons/synapses) -- does confidence-gating change outcomes?
- **False positive/negative estimation**: Under precautionary mode, estimate rates from simulated organoid trajectories
- **Figure**: Heatmap of tier classification across threshold ranges

## 6. Discussion (~1,000 words)

### 6.1 Implications for Research Policy

- **IRB integration**: The framework produces structured JSON assessments compatible with laboratory information management systems (LIMS). IRBs could require periodic ethics assessments as a condition of organoid research approval.
- **Tiered review process**: Tiers 0-2 handled by local ethics committees; Tiers 3-4 escalate to institutional or national bodies. This mirrors existing clinical research escalation pathways.
- **Prospective monitoring**: Trend analysis enables "advance notice" of approaching thresholds, allowing orderly research wind-down rather than emergency halts.
- **Scalability**: Framework applies to any neural organoid type (cortical, retinal, spinal, assembloids) -- the indicators are electrophysiological, not tissue-type-specific.
- **International coordination**: Different jurisdictions could adopt different precautionary factors while using the same tier structure, enabling harmonized-but-flexible regulation.

### 6.2 Limitations

- **IIT Phi is theoretically contested**: Aaronson (2014) argues Phi can be high in systems that seem clearly non-conscious. Our use of Phi as a *threshold trigger* (not a consciousness detector) partially mitigates this, but the concern is legitimate.
- **Measurement uncertainty**: We measure correlates, not consciousness itself (the hard problem; Chalmers 1996). No measurement framework can bridge the explanatory gap.
- **Threshold justification**: Our specific threshold values are informed by literature but ultimately reflect judgment calls. Different reasonable people could set them differently.
- **Organoid-specific consciousness**: If organoid consciousness differs qualitatively from human consciousness (e.g., if spatial embodiment matters per Thompson 2007), our indicators may be systematically biased.
- **Digital organoid fidelity**: Our validation uses a simplified simulation, not real organoid data. Partnership with experimental labs is needed for definitive validation.
- **Single-theory dependence**: Currently IIT-centric. A multi-theory approach (IIT + GWT global workspace measures + HOT metacognitive indicators) would be more robust.
- **Phi computation**: Exact Phi is intractable for systems larger than ~20 elements. Our approximation (sampled partition) introduces quantification error.

### 6.3 Future Directions

- **Multi-theory consciousness assessment**: Incorporate GWT workspace measures (global broadcasting indicators), HOT metacognitive markers, and recurrent processing measures alongside Phi. Tier classification could require convergent evidence from multiple theories.
- **Real-time hardware integration**: Partner with MEA (multi-electrode array) manufacturers to create real-time ethics monitoring during organoid culture. The framework's `assess()` method is designed for periodic invocation.
- **Empirical threshold calibration**: Collaborate with organoid labs to process real electrophysiology data through the framework and refine threshold values based on expert consensus.
- **International standards body**: Propose an ISO/IEEE working group for organoid consciousness measurement standards, using this framework as a discussion document.
- **Assembloid and chimera extension**: Extend to brain-region assembloids and human-animal chimeras where consciousness questions are even more pressing.
- **Machine learning augmentation**: Train classifiers on expert-labeled organoid recordings to improve indicator detection beyond threshold-based rules.

## 7. Conclusion (~200 words)

- Eight years after Farahany et al. (2018) called for ethical frameworks for brain organoid research, the field still lacks quantitative criteria for when consciousness concerns should trigger ethical action.
- We have presented a five-tier framework with seven measurable indicators, graduated action requirements, precautionary defaults, and trend-based early warning.
- The framework is deliberately imperfect and adjustable -- it is offered as a concrete starting point for community refinement, not as a final regulatory standard.
- All thresholds are parameters, not truths. The code is open-source. We invite the neuroscience, ethics, and regulatory communities to challenge, refine, and improve it.
- The moral stakes are asymmetric: the cost of premature caution is delayed research; the cost of insufficient caution could be unrecognized suffering. Our framework errs on the side of caution, and we believe that is the right default.

## References (~45 citations)

### Organoid Biology
- Lancaster, M.A. et al. (2013). Cerebral organoids model human brain development and microcephaly. *Nature*, 501(7467), 373-379.
- Trujillo, C.A. et al. (2019). Complex oscillatory waves emerging from cortical organoids model early human brain network development. *Cell Stem Cell*, 25(4), 558-569.
- Quadrato, G. et al. (2017). Cell diversity and network dynamics in photosensitive human brain organoids. *Nature*, 545(7652), 48-53.
- Sakaguchi, H. et al. (2022). Functional neuronal circuitry and oscillatory dynamics in human brain organoids. *Nature Communications*, 13, 4755.

### Consciousness Theory
- Tononi, G. (2004). An information integration theory of consciousness. *BMC Neuroscience*, 5, 42.
- Tononi, G. (2008). Consciousness as integrated information: A provisional manifesto. *Biological Bulletin*, 215(3), 216-242.
- Baars, B.J. (1988). *A Cognitive Theory of Consciousness*. Cambridge UP.
- Dehaene, S. & Naccache, L. (2001). Towards a cognitive neuroscience of consciousness. *Cognition*, 79(1-2), 1-37.
- Rosenthal, D.M. (2005). *Consciousness and Mind*. Oxford UP.
- Lamme, V.A.F. (2006). Towards a true neural stance on consciousness. *Trends in Cognitive Sciences*, 10(11), 494-501.
- Chalmers, D.J. (1996). *The Conscious Mind*. Oxford UP.
- Aaronson, S. (2014). Why I am not an integrated information theorist. Blog post / talk.

### Ethics & Moral Status
- Farahany, N.A. et al. (2018). The ethics of experimenting with human brain tissue. *Nature*, 556(7702), 429-432.
- Lavazza, A. (2021). Potential ethical problems with human cerebral organoids: Consciousness and moral status of future brains in a dish. *Brain Research*, 1750, 147146.
- Shepherd, J. (2018). Ethical (and epistemological) issues regarding consciousness in cerebral organoids. *Journal of Medical Ethics*, 44(9), 611-612.
- Sawai, T. et al. (2019). The ethics of cerebral organoid research: Being conscious of consciousness. *Stem Cell Reports*, 13(3), 440-447.
- Sawai, T. et al. (2022). Mapping the ethical issues of brain organoid research and application. *AJOB Neuroscience*, 13(2), 81-94.
- Boyd, K. & Lipshitz, R. (2024). Moral status and brain organoids: A conceptual framework. *Neuroethics*, 17(1).
- Beauchamp, T.L. & Childress, J.F. (2019). *Principles of Biomedical Ethics* (8th ed.). Oxford UP.

### Organoid Intelligence
- Smirnova, L. et al. (2023). Organoid intelligence (OI): The new frontier in biocomputing and intelligence-in-a-dish. *Frontiers in Science*, 1, 1017235.
- Hartung, T. et al. (2023). The Baltimore declaration toward the exploration of organoid intelligence.

### Regulatory & Policy
- ISSCR (2021). Guidelines for Stem Cell Research and Clinical Translation.
- National Academies of Sciences (2021). *The Emerging Field of Human Neural Organoids, Transplants, and Chimeras*.

### Related Philosophical
- Thompson, E. (2007). *Mind in Life: Biology, Phenomenology, and the Sciences of Mind*. Harvard UP.
- Penrose, R. & Hameroff, S. (1994). *Shadows of the Mind*. Oxford UP.

### Recent Empirical Ethics
- [Authors] (2024). Moral intuition regarding the possibility of conscious human brain organoids. *Science and Engineering Ethics*.
- [Authors] (2025). Consciousness and human brain organoids: A conceptual mapping. *AJOB Neuroscience*, 17(1).
- [Authors] (2026). Ethical concerns about embodied brain organoids shaped by foundational distinctions. *Scientific Reports*.

## Figures

1. **Framework overview diagram**: Indicators (7) -> indicator detection with confidence -> tier classification (max tier rule) -> required actions cascade. Show precautionary threshold adjustment.
2. **Simulated Phi trajectory**: 200-day organoid development with tier boundaries overlaid. Two lines: precautionary thresholds (solid) vs. standard thresholds (dashed). Annotate tier transitions and recommended actions.
3. **Precautionary vs. standard mode comparison**: Side-by-side tier classification across a range of organoid maturity levels, showing how many assessments shift tier under each mode.
4. **Sensitivity analysis heatmap**: Tier classification as a function of Phi threshold (x-axis) and developmental day (y-axis), showing robustness of tier boundaries.
5. **Risk score decomposition**: Stacked bar chart showing tier, Phi, and developmental day contributions to composite risk score across a simulated trajectory.

## Supplementary Material

- Open-source implementation: GitHub link (symthaea-cell-foundry crate)
- Full API documentation (Rust docs)
- Digital organoid simulation code and example scenarios
- Raw data from sensitivity analysis
- Extended validation results

## Writing Plan

| Section | Word estimate | Key challenge |
|---------|---------------|---------------|
| Abstract | 150 | Concision; must convey gap + framework + limitation |
| Introduction | 1,000 | Establish urgency without overstating consciousness claims |
| Background | 800 | Fair summary of contested theories; justify IIT pragmatically |
| Framework | 1,500 | Technical precision in accessible language for ethics audience |
| Implementation | 600 | Enough detail for reproducibility; not a software manual |
| Validation | 800 | Honest about what we can and cannot validate |
| Discussion | 1,000 | Strong on limitations; constructive on future work |
| Conclusion | 200 | Moral stakes + invitation to community refinement |
| **Total** | **~6,050** | Within 7,500 word limit with margin for expansion |

## Key Risks to Address in Writing

1. **Overclaiming consciousness detection**: We detect *correlates* under *specific theories*, not consciousness itself. Must be stated clearly and repeatedly.
2. **IIT dependence**: Phi is contested. Frame as pragmatic choice (only scalar measure available), acknowledge alternatives, propose multi-theory extension.
3. **Threshold arbitrariness**: Acknowledge that specific values are informed judgment, not empirical facts. Emphasize adjustability.
4. **Digital vs. real validation**: Our validation is against a simplified simulation and published summaries, not raw experimental data. State this as a limitation and call for experimental partnerships.
5. **Single-author credibility**: A computational framework for organoid ethics would benefit from co-authors in neuroscience and bioethics. Consider reaching out to collaborators before submission.
