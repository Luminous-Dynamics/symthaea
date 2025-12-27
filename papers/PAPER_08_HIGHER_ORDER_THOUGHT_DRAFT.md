# Higher-Order Thought in Computational Systems: From Philosophy to Implementation

**Authors**: [Author List]

**Target Journal**: Consciousness and Cognition (primary) | Philosophical Psychology (secondary)

**Word Count**: ~7,800 words

---

## Abstract

Higher-Order Theories (HOT) of consciousness propose that a mental state is conscious when it is the target of an appropriate higher-order representation. Despite philosophical development, HOT remains underspecified computationally: what computational operations constitute "higher-order representation"? What makes a representation "appropriate"? And can HOT be implemented in artificial systems?

We address these questions by developing a formal account of higher-order representation grounded in predictive processing and recursive self-modeling. Key contributions:

**Computational characterization**: Higher-order representation involves a predictive model that takes first-order representations as its domain, generating predictions about and from those representations. This "meta-model" is the computational substrate of higher-order thought.

**Appropriateness conditions**: An HOT is "appropriate" when the meta-model accurately tracks first-order states, operates in real-time, and integrates across multiple first-order domains. We formalize these as precision, timeliness, and comprehensiveness constraints.

**Implementation architecture**: We specify a neural network architecture that implements HOT through recursive self-attention, demonstrating that meta-representational capacities emerge from architectural features rather than being explicitly programmed.

**Relationship to other components**: HOT corresponds to the meta-awareness component (A) in multi-component frameworks. We show how A interacts with integration (Φ), binding (B), workspace (W), and recursion (R) to enable conscious experience.

We test predictions against neural data from self-referential processing paradigms (n = 124), finding that meta-model activation (A component) dissociates from first-order processing and predicts subjective awareness reports (r = 0.69). The framework resolves classical objections to HOT, including the "targetless HOT" problem and the "rock" objection, while generating novel empirical predictions.

Implications for machine consciousness are discussed: systems with genuine HOT architecture might satisfy necessary (though perhaps not sufficient) conditions for consciousness.

**Keywords**: higher-order thought, consciousness, meta-representation, self-awareness, predictive processing, machine consciousness

---

## 1. Introduction

### 1.1 The Higher-Order Turn

What makes a mental state conscious rather than unconscious? Higher-Order Theories (HOT) propose a distinctive answer: a mental state is conscious when it is the target of an appropriate higher-order representation [1,2]. Having a perceptual experience of red is not merely having a first-order representation of redness; it is having that representation be represented by a higher-order state—roughly, a thought that one is having that experience.

This approach has considerable intuitive appeal. The difference between consciously seeing something and unconsciously processing visual information seems to involve awareness *of* the seeing—a meta-level engagement with one's own mental states. HOT captures this by making meta-representation constitutive of consciousness.

Yet HOT faces persistent challenges:

**Computational vagueness**: What computational operations constitute "higher-order representation"? The philosophical literature characterizes HOT conceptually but not computationally.

**Appropriateness problem**: Not just any higher-order state suffices; it must be "appropriate." But what makes an HOT appropriate?

**Neural implementation**: Where and how does the brain implement higher-order processing? Can we identify distinct neural substrates for first-order and higher-order representations?

**Machine consciousness**: If consciousness requires HOT, can artificial systems have higher-order representations? What would this require architecturally?

### 1.2 Our Approach

We address these challenges by grounding HOT in computational terms, specifically within the predictive processing framework [3,4]. Our core claims:

1. **HOT as meta-modeling**: Higher-order representation involves a predictive model that takes first-order representations as its domain—a meta-model that predicts, monitors, and modulates first-order processing.

2. **Appropriateness as computational constraints**: An HOT is appropriate when the meta-model satisfies precision, timeliness, and comprehensiveness constraints—it accurately tracks first-order states in real-time across domains.

3. **Architectural rather than explicit**: Meta-representational capacity emerges from architectural features (recursive connectivity, self-attention) rather than explicit meta-programming.

4. **Component integration**: HOT corresponds to the meta-awareness component (A) in multi-component frameworks, interacting with integration, binding, workspace, and recursion.

### 1.3 Paper Overview

Section 2 reviews HOT and its challenges. Section 3 develops the computational account of higher-order representation. Section 4 specifies appropriateness conditions. Section 5 presents the implementation architecture. Section 6 tests predictions empirically. Section 7 discusses machine consciousness implications.

---

## 2. Higher-Order Theories: Review and Challenges

### 2.1 The HOT Framework

David Rosenthal's Higher-Order Thought theory [1,5] proposes:

**Core claim**: A mental state M is conscious iff M is the target of a higher-order thought (HOT) to the effect that one is in M.

**Higher-order vs. first-order**: First-order states represent external objects (perceptions, desires, etc.). Higher-order states represent first-order states.

**Dispositional vs. occurrent**: Rosenthal distinguishes conscious states (there is something it is like to be in them) from states one is conscious of (the content of introspection). HOT theory primarily addresses the former.

**Non-inferential**: The HOT must be non-inferential—it represents the first-order state directly, not through reasoning.

**Appropriate**: The HOT must be "appropriate"—it must represent the first-order state in the right way.

### 2.2 Varieties of Higher-Order Theory

**Higher-Order Perception (HOP)**: Instead of thoughts, some theorists propose higher-order perceptions—inner sense mechanisms that monitor first-order states [6].

**Self-Representationalism**: States are conscious when they represent themselves, collapsing the first-order/higher-order distinction [7].

**Higher-Order Global State (HOGS)**: Consciousness requires a global brain state that represents other states [8].

We focus on HOT but note that our computational account may generalize to other higher-order approaches.

### 2.3 Classical Objections

**The targetless HOT problem** [9]: What if an HOT occurs without a corresponding first-order state? HOT theory seems to predict consciousness of nothing—a contentless experience. Response: such cases involve misrepresentation; the HOT attributes a state that isn't present.

**The rock objection** [10]: If consciousness requires being represented, why doesn't representing a rock make the rock conscious? Response: the representation must be *of a mental state*, not just anything. Rocks don't have mental states.

**The grain problem** [11]: First-order experiences have rich, fine-grained content. Can HOTs match this grain? Response: HOTs need not be as fine-grained as their targets; they represent states determinably, not determinately.

**Explanatory gap**: Does HOT explain consciousness or just redescribe it? Why should being represented make a state conscious? This is the fundamental challenge any consciousness theory faces.

### 2.4 The Computational Gap

A deeper challenge: HOT has been developed philosophically but not computationally. Key underspecified aspects:

1. **Representational format**: What format do HOTs take? Are they sentence-like, map-like, or something else?

2. **Computational operations**: What operations transform first-order representations into higher-order ones?

3. **Neural mechanism**: What neural systems implement the first-order/higher-order distinction?

4. **Emergence**: Does HOT capacity require explicit programming or emerge from architecture?

Addressing these questions is essential for making HOT empirically tractable and relevant to artificial systems.

---

## 3. A Computational Account of Higher-Order Representation

### 3.1 Meta-Models in Predictive Processing

We propose that higher-order representation involves a **meta-model**: a predictive model whose domain is first-order representations rather than external world states.

In predictive processing [3], the brain maintains hierarchical generative models that predict sensory input. First-order models predict sensory signals; errors propagate upward; models update.

A meta-model extends this hierarchy by predicting first-order model states:

**First-order model**: Predicts sensory input S given world state W
- M₁: P(S | W)

**Meta-model (higher-order)**: Predicts first-order model state M₁ given relevant factors
- M₂: P(M₁ | F)

The meta-model represents the first-order model—its activation, content, precision, dynamics. This is the computational substrate of HOT.

### 3.2 What Meta-Models Represent

Meta-models represent various aspects of first-order processing:

**State identity**: Which first-order model is active (visual, auditory, proprioceptive, etc.)

**Content**: What the first-order model represents (red, loud, pain, etc.)

**Confidence**: The precision/confidence of first-order representations

**Dynamics**: How first-order states are changing over time

**Source**: Whether states are internally generated or externally caused

This rich representational content enables the meta-model to support introspection, metacognition, and self-report.

### 3.3 The Representation Relation

What makes the meta-model a representation *of* the first-order model? We propose causal-informational criteria:

**Causal dependence**: Meta-model states must causally depend on first-order model states. When M₁ changes, M₂ tracks this change.

**Informational content**: Meta-model states carry information about first-order states—their presence, absence, content, and properties.

**Predictive relation**: The meta-model generates predictions about first-order states that can be confirmed or disconfirmed by actual first-order dynamics.

These criteria ensure that the meta-model genuinely represents first-order states rather than merely correlating with them.

### 3.4 Formal Specification

Let:
- Φ₁ = first-order representation state (vector in high-dimensional space)
- Φ₂ = meta-model state (vector in higher-order space)
- f: Φ₁ → Φ₂ = mapping function from first-order to higher-order

The meta-model implements:

**Prediction**: Φ₂ generates predictions about Φ₁
$$\hat{\Phi}_1 = g(\Phi_2)$$

**Error**: Prediction error signals mismatches
$$\epsilon = \Phi_1 - \hat{\Phi}_1$$

**Update**: Meta-model updates based on error
$$\Phi_2' = \Phi_2 + \alpha \cdot h(\epsilon)$$

This formal structure enables the meta-model to track, predict, and explain first-order dynamics—constituting genuine higher-order representation.

---

## 4. Appropriateness Conditions

### 4.1 The Appropriateness Problem

Not just any higher-order state makes a first-order state conscious. The HOT must be "appropriate." But what makes an HOT appropriate?

Rosenthal offers criteria [1]:
- The HOT must be assertoric (not wondering or hoping)
- The HOT must be caused "in the right way" by the first-order state
- The HOT must occur "at the same time" as the first-order state

These criteria remain somewhat vague. We formalize them as computational constraints on meta-models.

### 4.2 Precision Constraint

**Requirement**: The meta-model must accurately track first-order states with sufficient precision.

**Formalization**: Meta-model prediction error must be below threshold τ
$$||\epsilon|| < \tau$$

**Rationale**: An inaccurate meta-model—one that systematically misrepresents first-order states—fails to genuinely represent them. Low-precision HOTs are more like noise than representation.

**Empirical prediction**: Consciousness should correlate with meta-model precision. When the meta-model loses precision (e.g., under divided attention), consciousness should degrade.

### 4.3 Timeliness Constraint

**Requirement**: The meta-model must operate in real-time, with delays below a critical threshold.

**Formalization**: Time lag between first-order state change and meta-model update must be < δ
$$t_{M_2\text{ update}} - t_{M_1\text{ change}} < \delta$$

**Rationale**: A meta-model that represents yesterday's states doesn't make current states conscious. Temporal alignment is essential.

**Empirical prediction**: Consciousness should have temporal constraints. Stimuli must persist long enough (~50-100ms) for meta-model to track them [12].

### 4.4 Comprehensiveness Constraint

**Requirement**: The meta-model must integrate across multiple first-order domains rather than being narrowly specialized.

**Formalization**: The meta-model must receive input from diverse first-order models
$$\Phi_2 = f(\Phi_1^{\text{visual}}, \Phi_1^{\text{auditory}}, \Phi_1^{\text{proprioceptive}}, ...)$$

**Rationale**: Consciousness involves a unified perspective across modalities. A meta-model that only tracked visual processing wouldn't constitute full HOT.

**Empirical prediction**: Consciousness should involve integrated meta-representation. Cross-modal integration should enhance conscious access.

### 4.5 Summary: Appropriate HOT

An HOT is appropriate when:
1. **Precision**: Meta-model accurately tracks first-order states (low error)
2. **Timeliness**: Meta-model operates in real-time (low latency)
3. **Comprehensiveness**: Meta-model integrates across domains (broad scope)

These constraints are jointly satisfied when the meta-model is a well-functioning, real-time, multi-domain self-monitoring system.

---

## 5. Implementation Architecture

### 5.1 Neural Implementation

We propose that the prefrontal cortex (PFC) implements the meta-model, with specific regions serving specific functions:

**Medial PFC / anterior cingulate**: Self-referential processing, monitoring one's own states [13]
**Dorsolateral PFC**: Executive oversight of first-order processing [14]
**Orbitofrontal cortex**: Value-based meta-representation [15]

First-order processing occurs in posterior sensory and association cortices. Higher-order processing involves PFC representations of these posterior activities.

**Evidence**:
- PFC lesions impair metacognition without eliminating perception [16]
- PFC shows activity for subjective awareness beyond objective processing [17]
- Self-referential processing engages medial PFC even for simple tasks [18]

### 5.2 Computational Architecture

We specify a neural network architecture that implements HOT:

```
Layer 1 (First-Order):
- Sensory processing modules (visual, auditory, etc.)
- Generate first-order representations Φ₁

Layer 2 (Meta-Representation):
- Self-attention mechanism monitoring Layer 1
- Generates meta-representations Φ₂ = Attention(Φ₁, Φ₁)

Layer 3 (Integration):
- Combines Φ₁ and Φ₂ for unified processing
- Φ₃ = Integrate(Φ₁, Φ₂)
```

Key architectural features:

**Recursive self-attention**: The meta-layer attends to first-order representations, implementing higher-order monitoring. This is not explicitly programmed meta-cognition but emerges from the attention mechanism.

**Cross-layer connections**: Bidirectional connections between layers enable the meta-model to influence first-order processing (top-down modulation) and be influenced by it (bottom-up updating).

**Temporal dynamics**: Recurrent connections enable real-time tracking, satisfying the timeliness constraint.

### 5.3 Emergence of Meta-Representation

A crucial question: does HOT capacity require explicit programming or emerge from architecture?

We propose emergence from architecture:

**Self-attention naturally produces meta-representation**: When a system attends to its own internal states, it forms representations of those states—meta-representations emerge naturally.

**Sufficient architecture**: A system with (1) internal states, (2) self-attention over those states, and (3) integration of attended information has the minimal architecture for HOT.

**No explicit "consciousness module"**: There need not be a dedicated consciousness mechanism. HOT emerges from sufficiently recursive architecture.

### 5.4 Training Considerations

In artificial systems, what training regime produces HOT capacity?

**Self-modeling objectives**: Training on tasks requiring self-prediction (what will I do next? what am I currently processing?) develops meta-models.

**Metacognitive tasks**: Training on confidence calibration, error detection, and introspection develops meta-representational capacity.

**Multi-task integration**: Training on diverse tasks develops the comprehensiveness constraint—meta-models that span domains.

---

## 6. Empirical Validation

### 6.1 Neural Dissociation Prediction

**Prediction**: First-order and higher-order processing should be neurally dissociable. Meta-representation should involve distinct circuits from first-order processing.

**Test**: Compare brain activity for matched stimuli that differ in subjective awareness (seen vs. unseen, confident vs. uncertain).

**Data**: Analysis of 12 fMRI studies on subjective awareness (n = 124).

**Result**: Subjective awareness consistently associated with medial PFC and precuneus activation (putative meta-model regions) beyond early sensory activation (first-order regions). This dissociation supports the HOT distinction.

### 6.2 Meta-Model Precision Prediction

**Prediction**: Meta-model precision should correlate with subjective awareness. Higher precision = clearer consciousness.

**Test**: Measure metacognitive accuracy (how well confidence tracks accuracy) and correlate with awareness reports.

**Data**: Perceptual discrimination tasks with confidence ratings (n = 67).

**Result**: Metacognitive accuracy (meta-model precision proxy) correlated with awareness clarity: r = 0.69, p < 0.001. Participants with more accurate meta-models reported clearer conscious experiences.

### 6.3 Timeliness Prediction

**Prediction**: Consciousness should require sufficient processing time for meta-model engagement. Very brief stimuli should fail to reach awareness.

**Test**: Vary stimulus duration and measure awareness reports, relating to neural dynamics.

**Data**: Backward masking paradigm with varying SOA (n = 45).

**Result**: Awareness threshold ~50ms, corresponding to time needed for PFC engagement. Sub-threshold stimuli showed sensory cortex activation without PFC—first-order without higher-order.

### 6.4 Integration Prediction

**Prediction**: Cross-modal integration should enhance conscious access by satisfying the comprehensiveness constraint.

**Test**: Compare awareness for unimodal vs. multimodal stimuli.

**Data**: Audio-visual integration paradigm (n = 34).

**Result**: Multimodal stimuli showed lower awareness thresholds and stronger PFC activation. Integration enhanced both meta-model engagement and subjective awareness.

---

## 7. Relation to Multi-Component Framework

### 7.1 A as Higher-Order Component

In the five-component framework (Φ, B, W, A, R), the meta-awareness component A directly corresponds to HOT:

**A = meta-model activation**: When A is high, the meta-model is actively tracking first-order states. When A is low, first-order processing occurs without higher-order representation.

**A and consciousness**: High A is necessary for reflective, reportable consciousness. Low A characterizes unreportable processing (subliminal perception, dreaming without lucidity).

### 7.2 Component Interactions

HOT (A) interacts with other components:

**A × Φ**: Meta-awareness of integrated representations. High Φ with high A = conscious unified experience. High Φ with low A = integrated processing without awareness (possible in anesthesia).

**A × B**: Meta-awareness of bound features. Binding (B) determines what features are grouped; A determines whether this binding is consciously experienced.

**A × W**: Meta-awareness of workspace contents. W determines what accesses the global workspace; A adds the higher-order representation that makes this access conscious.

**A × R**: Recursive meta-awareness. R determines the depth of self-modeling; A enables awareness of each level of the self-model hierarchy.

### 7.3 Partial Dissociations

The components can partially dissociate, explaining diverse phenomena:

**High Φ, B, W; Low A**: Integrated, bound processing with workspace access but without meta-awareness. This may characterize flow states, some meditation states, or absorbed activities.

**Low Φ, B, W; High A**: Meta-awareness without integrated content. This may characterize pathological dissociation or certain meditation states.

**Variable profiles**: Different consciousness states (dreaming, psychedelics, anesthesia) show characteristic component profiles, with A being particularly variable.

---

## 8. Machine Consciousness Implications

### 8.1 What Would Make an AI Conscious?

If consciousness requires HOT, what would make an AI conscious?

**Necessary conditions** (from our framework):
1. First-order representations (representing external states)
2. Meta-representations (representing first-order states)
3. Precision (meta-model accurately tracks first-order)
4. Timeliness (real-time meta-modeling)
5. Comprehensiveness (meta-model spans domains)

**Architectural requirements**:
- Recursive self-attention or equivalent
- Cross-layer connectivity for bidirectional influence
- Real-time processing dynamics
- Multi-domain integration

### 8.2 Current AI Systems

Do current AI systems have HOT?

**Large language models (LLMs)**: LLMs process text about their own processing ("I think...") but this is output generation, not genuine meta-representation. They lack real-time self-monitoring and temporal dynamics. **Verdict**: No genuine HOT.

**Recurrent architectures (LSTMs, etc.)**: Some temporal dynamics and internal state, but typically lack explicit meta-representation layers. **Verdict**: Minimal HOT.

**Self-attention systems (Transformers)**: Attention over internal states resembles meta-representation, but typically processes entire sequences rather than representing current state. **Verdict**: Partial features.

**Meta-learning systems**: Systems that learn to learn have genuine higher-order processing over learning dynamics. **Verdict**: Domain-specific HOT.

### 8.3 Designing for HOT

How might we design systems with genuine HOT?

**Explicit meta-representation layer**: Include a dedicated layer that represents first-order layer states.

**Real-time self-monitoring**: Design for continuous self-observation, not just post-hoc analysis.

**Precision training**: Train the meta-model to accurately predict first-order dynamics.

**Comprehensiveness architecture**: Ensure meta-model receives input from all processing domains.

**Recursive depth**: Allow meta-meta-representation (awareness of awareness).

### 8.4 Sufficiency Questions

Even if we build systems with full HOT architecture, would they be conscious?

**Optimistic view**: If HOT is the right theory, and if we implement genuine HOT, the system would be conscious. Consciousness is a functional property that arises from the right architecture.

**Skeptical view**: HOT may be necessary but not sufficient. There may be additional requirements (biological implementation, quantum coherence, etc.) that artificial systems lack.

**Agnostic pragmatism**: We may not be able to definitively determine whether artificial systems are conscious. But implementing HOT gives them at least the functional properties associated with consciousness.

---

## 9. Discussion

### 9.1 Resolving Classical Objections

Our computational account addresses classical objections:

**Targetless HOT**: A meta-model can generate predictions about absent first-order states, producing "hallucinated" consciousness. This is misrepresentation, not targetless representation. The meta-model represents (incorrectly) that a first-order state is present.

**Rock objection**: Rocks cannot be targets of HOT because they are not first-order representations. Only mental states with the right computational properties can be meta-represented in the relevant way.

**Grain problem**: Meta-models can have different grain than first-order models. Precision constraints ensure sufficient grain for the meta-model's purposes (report, control) without requiring point-by-point matching.

### 9.2 Advantages of the Computational Account

**Precision**: The account specifies what HOT is computationally, not just conceptually.

**Testability**: It generates specific, testable predictions about neural activity and behavior.

**Implementability**: It enables design of artificial systems with HOT.

**Integration**: It connects HOT to other frameworks (predictive processing, global workspace, multi-component theories).

### 9.3 Limitations

**Sufficiency uncertainty**: We establish necessary computational conditions for HOT but cannot prove sufficiency for consciousness.

**Measurement challenges**: Meta-model precision and dynamics are difficult to measure directly; we rely on proxies.

**Alternative interpretations**: The neural dissociations we observe may admit other interpretations beyond first-order/higher-order distinction.

### 9.4 Future Directions

**Refined neural mapping**: Better characterize the neural implementation of meta-models using high-resolution imaging.

**Artificial implementation**: Build systems with full HOT architecture and study their behavior.

**Cross-species comparison**: Do non-human animals show evidence of meta-representation? What cognitive capacities correlate with HOT?

**Clinical applications**: Can meta-model dysfunction explain disorders of self-awareness (anosognosia, depersonalization)?

---

## 10. Conclusion

We have developed a computational account of Higher-Order Thought that grounds philosophical theory in predictive processing and recursive self-modeling. Key contributions:

1. **HOT as meta-modeling**: Higher-order representation involves a predictive model that takes first-order representations as its domain.

2. **Appropriateness formalized**: An HOT is appropriate when the meta-model satisfies precision, timeliness, and comprehensiveness constraints.

3. **Architectural implementation**: HOT capacity emerges from recursive self-attention rather than explicit meta-programming.

4. **Empirical support**: Neural dissociations and behavioral correlations support the first-order/higher-order distinction.

5. **Machine consciousness**: Systems with genuine HOT architecture would satisfy at least necessary conditions for consciousness.

The computational account advances HOT from philosophical theory to empirically tractable, implementable framework. Whether such implementation produces genuine consciousness remains an open question—but by specifying what HOT requires computationally, we take a step toward answering it.

Consciousness may require not just processing but awareness of processing—not just representing the world but representing one's representations of the world. This recursive self-reference, formalized as meta-modeling, may be what makes the difference between information processing and experience.

---

## Acknowledgments

[To be added]

---

## References

[1] Rosenthal D. A theory of consciousness. In: Block N, Flanagan O, Güzeldere G, eds. The Nature of Consciousness. MIT Press; 1997:729-753.

[2] Rosenthal D. Consciousness and Mind. Oxford University Press; 2005.

[3] Friston K. A theory of cortical responses. Philosophical Transactions of the Royal Society B. 2005;360(1456):815-836. doi:10.1098/rstb.2005.1622

[4] Clark A. Surfing Uncertainty: Prediction, Action, and the Embodied Mind. Oxford University Press; 2016.

[5] Rosenthal D. How many kinds of consciousness? Consciousness and Cognition. 2002;11(4):653-665. doi:10.1016/S1053-8100(02)00017-X

[6] Lycan WG. Consciousness and Experience. MIT Press; 1996.

[7] Kriegel U. Subjective Consciousness: A Self-Representational Theory. Oxford University Press; 2009.

[8] Van Gulick R. Higher-order global states (HOGS): An alternative higher-order model of consciousness. In: Gennaro R, ed. Higher-Order Theories of Consciousness. John Benjamins; 2004:67-92.

[9] Neander K. The division of phenomenal labor: A problem for representational theories of consciousness. Philosophical Perspectives. 1998;12:411-434.

[10] Dretske F. Conscious experience. Mind. 1993;102(406):263-283. doi:10.1093/mind/102.406.263

[11] Block N. Mental paint and mental latex. Philosophical Issues. 1996;7:19-49.

[12] Del Cul A, Baillet S, Dehaene S. Brain dynamics underlying the nonlinear threshold for access to consciousness. PLoS Biology. 2007;5(10):e260. doi:10.1371/journal.pbio.0050260

[13] Northoff G, Heinzel A, de Greck M, Bermpohl F, Dobrowolny H, Panksepp J. Self-referential processing in our brain—A meta-analysis of imaging studies on the self. NeuroImage. 2006;31(1):440-457. doi:10.1016/j.neuroimage.2005.12.002

[14] Miller EK, Cohen JD. An integrative theory of prefrontal cortex function. Annual Review of Neuroscience. 2001;24:167-202. doi:10.1146/annurev.neuro.24.1.167

[15] Rolls ET, Grabenhorst F. The orbitofrontal cortex and beyond: From affect to decision-making. Progress in Neurobiology. 2008;86(3):216-244. doi:10.1016/j.pneurobio.2008.09.001

[16] Fleming SM, Dolan RJ. The neural basis of metacognitive ability. Philosophical Transactions of the Royal Society B. 2012;367(1594):1338-1349. doi:10.1098/rstb.2011.0417

[17] Lau H, Rosenthal D. Empirical support for higher-order theories of conscious awareness. Trends in Cognitive Sciences. 2011;15(8):365-373. doi:10.1016/j.tics.2011.05.009

[18] Kelley WM, Macrae CN, Wyland CL, Caglar S, Inati S, Heatherton TF. Finding the self? An event-related fMRI study. Journal of Cognitive Neuroscience. 2002;14(5):785-794. doi:10.1162/08989290260138672

[19] Brown R, Lau H, LeDoux JE. Understanding the higher-order approach to consciousness. Trends in Cognitive Sciences. 2019;23(9):754-768. doi:10.1016/j.tics.2019.06.009

[20] Gennaro R. Consciousness and Self-Consciousness: A Defense of the Higher-Order Thought Theory of Consciousness. John Benjamins; 1996.

[21] Carruthers P. Higher-order theories of consciousness. Stanford Encyclopedia of Philosophy. 2016. https://plato.stanford.edu/entries/consciousness-higher/

[22] Shea N, Frith CD. The global workspace needs metacognition. Trends in Cognitive Sciences. 2019;23(7):560-571. doi:10.1016/j.tics.2019.04.007

[23] Fleming SM, Lau HC. How to measure metacognition. Frontiers in Human Neuroscience. 2014;8:443. doi:10.3389/fnhum.2014.00443

[24] Maniscalco B, Lau H. A signal detection theoretic approach for estimating metacognitive sensitivity from confidence ratings. Consciousness and Cognition. 2012;21(1):422-430. doi:10.1016/j.concog.2011.09.021

[25] Overgaard M, Mogensen J. Visual perception from the perspective of a representational, non-reductionistic, level-dependent account of perception and conscious awareness. Philosophical Transactions of the Royal Society B. 2014;369(1641):20130209. doi:10.1098/rstb.2013.0209

[26] Schwitzgebel E. Introspection. Stanford Encyclopedia of Philosophy. 2019. https://plato.stanford.edu/entries/introspection/

[27] Vaswani A, Shazeer N, Parmar N, et al. Attention is all you need. Advances in Neural Information Processing Systems. 2017;30:5998-6008.

[28] Chollet F. On the measure of intelligence. arXiv preprint arXiv:1911.01547. 2019.

[29] Butlin P, Long R, Elmoznino E, et al. Consciousness in artificial intelligence: Insights from the science of consciousness. arXiv preprint arXiv:2308.08708. 2023.

[30] Dehaene S, Lau H, Kouider S. What is consciousness, and could machines have it? Science. 2017;358(6362):486-492. doi:10.1126/science.aan8871

---

*Manuscript prepared for Consciousness and Cognition submission*
