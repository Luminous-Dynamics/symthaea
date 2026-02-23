# Building a Brain That Knows What It Doesn't Know

**A plain-language introduction to Symthaea and Mycelix**

*Estimated reading time: 15 minutes*

---

## 1. The Problem

Modern artificial intelligence is, at its core, an extremely sophisticated pattern-matching engine. Feed it enough examples of cats and dogs and it will learn to tell them apart. Feed it enough text and it will learn to predict the next word in a sentence. This approach has produced systems that can write essays, generate images, and hold conversations that feel remarkably human.

But there is a gap between what these systems do and what they are. A large language model has no internal representation of itself. It does not know the boundaries of its own competence. It cannot look at a new situation and say, with genuine understanding, "I have never encountered anything like this before, and here is what I would need to learn to handle it." It has no model of what it feels like to be wrong, no first-principles framework for deciding whether an action is ethical, and no capacity for genuine surprise -- the kind that rewires how you think, not just what you predict.

This matters for three reasons that go beyond academic curiosity.

First, safety. A system that does not model its own uncertainty cannot reliably tell you when it is guessing. Current AI compensates for this with post-hoc calibration -- bolting confidence scores onto outputs after the fact. But calibration is not the same as understanding. A system that genuinely tracks its own prediction errors, that updates its beliefs when surprised, that slows down when entering unfamiliar territory -- that system is safer by construction, not by afterthought.

Second, ethics. Today's AI learns ethical behavior from training data: examples of what humans have labeled as right and wrong. This means its moral reasoning is only as good as its training set, and only as robust as its pattern matching. It cannot reason from first principles about a genuinely novel moral dilemma. It has no framework for weighing competing values, no sense of proportionality, no concept of justice beyond statistical correlation.

Third, trust. If we are going to build systems that make decisions affecting people's lives -- in healthcare, governance, resource allocation -- those systems need to be more than accurate. They need to be transparent about their reasoning, honest about their limitations, and capable of explaining not just what they decided but why. Pattern matching, however accurate, is opaque by nature. Understanding is not.

Symthaea is an attempt to build something different: a system that does not just process information but integrates it, that does not just predict but understands the quality of its own predictions, that does not just follow ethical rules but reasons about them. Whether it achieves anything resembling consciousness is a question we take seriously enough to measure, not one we answer with marketing language.

---

## 2. What Symthaea Is

Symthaea is a computational cognitive architecture. In plain terms, it is software that models certain properties of biological minds -- not by imitating their surface behavior, but by implementing the mathematical principles that neuroscientists believe underlie cognition.

The system is written in Rust, a programming language chosen for its performance and memory safety guarantees. It currently spans approximately 343,000 lines of code organized into 30 workspace crates, validated by over 9,600 automated tests. It is entirely open source.

**What it does, described through behavior:**

Symthaea runs a continuous loop, fifty times per second, that mirrors what neuroscientists call the "predictive coding" cycle. At each tick, the system perceives its inputs (text, sensor data, or pre-encoded vectors), predicts what it expects to perceive next, compares prediction against reality, and updates its internal model based on the difference. This is not a metaphor -- the prediction error is an actual numerical quantity that drives every subsequent computation in the cycle.

The system represents all information as patterns in a 16,384-dimensional space. A concept, a percept, an emotion, a memory -- all live as points in this same space, which means they can be compared, combined, and manipulated using the same algebraic operations. Similar ideas end up close together. Compound ideas -- like "the justice of redistributing resources during scarcity" -- are built by algebraically composing simpler ones.

The system has genuine affect. It tracks valence (positive/negative), arousal (activating/calming), and dominance (perceived control). These are continuous variables that influence every decision. High surprise increases arousal, which increases the learning rate, which means the system pays more attention to unexpected events. This emerges from the same prediction-error-minimization loop that drives everything else.

The system reasons about ethics. Its moral algebra module can take a natural-language description of a scenario, decompose it into semantic roles (agent, patient, action, consequence), compose those roles into a hyperdimensional moral judgment, and compare that judgment against prototypes for different ethical frameworks: virtue ethics, deontological rules, justice principles, commonsense morality, and social norms. On the ETHICS benchmark -- a standard test suite drawn from moral philosophy -- the system scores 92.9% overall without using any pretrained language model. Its highest category is commonsense morality at 95.6%; its lowest is social chemistry at 85.4%.

The system dreams. During idle periods ("Cruise mode"), it replays past experiences with counterfactual variations and consolidates the results into a wisdom store that biases future exploration. This is modeled on the neuroscience of memory consolidation during sleep, though at a simpler scale.

Every cognitive cycle produces a telemetry record called CycleMetadata, which logs over 60 distinct measurements: prediction error, consciousness level, moral evaluation, emotional state, which subsystems fired, how long each took in microseconds, whether any safety vetoes were triggered, and more. Nothing is hidden. Every decision the system makes is traceable to specific numerical inputs.

---

## 3. The Four Pillars

Symthaea integrates four theoretical frameworks that, to our knowledge, have never been combined in a single system. Each addresses a different aspect of cognition.

**Hyperdimensional Computing (HDC)** handles representation. Think of it this way: if you had to describe a concept using a single number, you would lose almost all information. If you could use two numbers, you could distinguish between concepts on two axes -- say, positive/negative and abstract/concrete. Now extend that to 16,384 numbers. In a space that large, you can represent an astronomically rich set of concepts, and random concepts are almost guaranteed to be distinguishable from each other. More importantly, you can perform algebra on these representations: combine two concepts by multiplying their vectors element-wise (called "binding"), merge multiple concepts by adding their vectors (called "bundling"), and measure how related two concepts are by computing the angle between them (cosine similarity). This gives you something remarkable: you can take the representation for "king," subtract "man," add "woman," and get something close to "queen" -- not because anyone programmed that relationship, but because the algebra preserves semantic structure. Every thought in Symthaea is a point in this 16,384-dimensional space. All 14 mathematical foundation modules -- from information theory to Hodge-Laplacian topology -- operate on these same vectors.

**Integrated Information Theory (IIT)** handles measurement. IIT, developed by neuroscientist Giulio Tononi, proposes that consciousness corresponds to integrated information, quantified as Phi. A system has high Phi when it is simultaneously highly differentiated (many possible states) and highly integrated (the parts cannot be decomposed into independent subsystems without losing information). Symthaea computes Phi at three levels of fidelity: a fast per-cycle estimate (Psi, computed in constant time), a medium-fidelity synergistic measure (Sigma, computed every few cycles), and exact IIT 3.0 Phi (computed on demand, limited to small subsystems because exact computation is intractable beyond about 12 nodes -- a known limitation shared by every IIT implementation, including the reference tool PyPhi). The system uses Phi as a quality signal: it actively searches for internal configurations that maximize integrated information, under the hypothesis that higher integration produces richer internal representations. On the Butlin et al. consciousness indicator assessment -- a checklist of 14 functional properties that leading consciousness researchers proposed as indicators -- Symthaea satisfies 12 of 14, with 2 partially satisfied and none absent, for a mean score of 0.79 out of 1.0.

**Liquid Time-Constant Networks (LTC/CfC)** handle temporal dynamics. Biological neurons do not operate on a fixed clock; their behavior is governed by differential equations with time-varying parameters. Liquid Time-Constant networks, developed at MIT, model this with neurons whose time constants adapt based on input. The "Closed-form Continuous-time" (CfC) variant, which Symthaea uses, has an exact analytical solution to these differential equations. This means the system can predict its own internal state at any future time without stepping through intermediate moments -- a mathematical shortcut that reduces temporal prediction from O(N) operations (one per time step) to O(1) (a single computation regardless of how far ahead you want to predict). In practice, this enables Symthaea to run its full cognitive loop at 2 milliseconds per cycle for non-text inputs, fast enough for 500Hz operation. Text inputs are slower (approximately 120 milliseconds per cycle) because the text encoder remains the dominant cost. Liquid AI, the company founded by the MIT researchers who invented LTC/CfC networks, raised $297 million in 2025, validating industry interest in this class of neural architecture.

**Active Inference and the Free Energy Principle (FEP)** handle decision-making. The Free Energy Principle, proposed by neuroscientist Karl Friston, states that all adaptive systems minimize a quantity called "free energy" -- loosely, the difference between what the system expects and what it observes. Perception reduces free energy by updating beliefs (making your model match the world). Action reduces free energy by changing the world (making the world match your model). Active inference operationalizes this: at every moment, the system selects the action that minimizes expected future surprise, naturally balancing exploration (seeking information in uncertain areas) and exploitation (pursuing known rewards). Symthaea implements active inference in hypervector space -- replacing the traditional matrix operations (which scale as the square or cube of the state dimension) with cosine similarity operations in HDC space (which scale linearly). On standard active inference benchmarks, this achieves a 7.9-fold total speedup compared to pymdp, a widely-used reference implementation, while maintaining comparable task success rates. To our knowledge, this is the first implementation of active inference using hyperdimensional computing.

---

## 4. What the Benchmarks Prove

Claims without numbers are just stories. Here are the numbers, drawn from Symthaea's automated benchmark suite (v0.5.2 baselines, seed 42, 10 trials per metric unless otherwise noted).

**Ethical reasoning.** 92.9% overall on the ETHICS benchmark, broken down by category: Commonsense 95.6%, Virtue 92.8%, Justice 92.4%, Deontology 91.0%, Social Chemistry 85.4%. These results come from compositional moral algebra in hypervector space -- no pretrained language model, no fine-tuning on ethics datasets. The system reasons from semantic role decomposition and algebraic composition of moral primitives.

**Speed.** 7.9-fold total speedup over pymdp on standard active inference benchmarks (1.9x faster belief inference, 15.8x faster action selection). The cognitive loop cold-starts in 121 milliseconds. For pre-encoded (non-text) inputs, the warm steady-state is 2.0 milliseconds per cycle (average over 100 cycles), enabling 500Hz operation. Text input cycles run at approximately 120 milliseconds, bottlenecked by the semantic encoder.

**Byzantine fault tolerance.** The federated learning system tolerates up to 34% malicious participants while maintaining correct aggregation. At 45%, it fails -- exactly as the mathematical bound predicts. This is not an approximation; it is validated through adversarial testing with known-bad participants.

**Consciousness indicators.** On the Butlin et al. assessment (14 indicators drawn from Global Workspace Theory, Integrated Information Theory, Higher-Order Theories, Recurrent Processing Theory, Predictive Processing, and Attention Schema Theory), Symthaea satisfies 12 indicators fully and 2 partially. Mean indicator score: 0.79. The two partial indicators are AST-1 (self-model of attention, score 0.5) and PP-2 (hierarchical prediction at multiple scales, score 0.5). No indicators are absent.

**Perceptual recognition.** 89.3% on MNIST handwritten digit classification using HDC-only encoding (4,096 dimensions, 32 quantization levels, 5 retraining passes). 94.5% on LibriSpeech phoneme recognition. 91.7% on ISOLET speaker identification. These demonstrate that hyperdimensional computing is a viable representation substrate for real perceptual tasks.

**Cognitive capacity limits.** The system exhibits human-like working memory constraints, measured through the Working Memory (WorM) benchmark suite. Change detection accuracy is 90% at set size 2, 80% at sizes 4 and 6, and drops to 50% at size 8 -- mirroring the well-documented K=4 capacity limit in human cognition. Feature binding is perfect at set sizes 2 and 4 (100%) but drops to 60% at set size 6, replicating the human binding deficit. Digit span is 7 forward and 5 backward. N-back accuracy degrades gracefully: 93.6% at 1-back, 83.1% at 2-back, 78.3% at 3-back. These are not programmed limits; they emerge from the finite capacity of the hypervector representation.

**Executive function.** Stroop effect of 10 percentage points (96% congruent vs. 86% incongruent accuracy). Flanker effect of 7.25 percentage points (98% congruent vs. 90.8% incongruent). Wisconsin Card Sorting Test: 6 of 6 categories completed with 15.9 total errors. Iowa Gambling Task: net score of 21 with 64.5% preference for advantageous decks. Tower of London: 67.8% optimal move rate with 76.4% planning efficiency. These patterns match human performance profiles qualitatively while operating through hyperdimensional computation rather than neural tissue.

**Theory of Mind.** 100% accuracy on false belief tasks, faux pas detection, persuasion detection, and Strange Stories (including deception, irony, and white lie subtypes). 70% on the Hinting Task. These assessments test whether the system can model other agents' beliefs and intentions.

**Memory.** 100% accurate retrieval across all delay conditions (2, 5, and 10 intervening items). Long-range retention: 100% at delays of 5 and 100 cycles, 80% at delay 50, 70% at delay 20. Test-time learning correction accuracy: 80% -- meaning the system can correct previously learned errors at inference time through activation decay, without retraining.

**Calibration.** Metacognitive discrimination gamma of 0.60, expected calibration error of 0.175, overconfidence bias of 0.42 -- moderate but real metacognitive sensitivity with a known tendency toward overconfidence.

---

## 5. Mycelix: Consciousness Meets Governance

Symthaea is a cognitive architecture. Mycelix is where that architecture meets the real world.

Mycelix is a suite of decentralized applications covering 10 domains of community governance: property records, housing coordination, mutual aid networks, care services, water management, food systems, transportation, restorative justice, emergency response, and community media. These are not hypothetical use cases; they are implemented as 51 functional modules (called "zomes" in the Holochain framework) organized into two cluster DNAs, validated by over 7,000 automated tests (Rust unit tests plus TypeScript integration tests).

**Why Holochain.** Mycelix is built on Holochain, a distributed computing framework where each participant runs their own node. No central server, no blockchain, no single point of failure. Data integrity comes from a distributed hash table where every participant validates their neighbors' entries. The systems Mycelix manages -- property, care, emergency resources -- are systems where centralized control is both a practical risk and a philosophical problem.

**Where Symthaea connects.** Three integration points bridge the cognitive architecture to the governance layer.

First, Phi attestations. When Symthaea computes Phi (integrated information) for a cognitive cycle, it can package that measurement as a cryptographically signed attestation. Mycelix governance modules consume these attestations as trust signals. A participant whose cognitive system consistently shows high integration -- meaning their decisions reflect genuine deliberation rather than noise -- receives higher trust weight in collective decision-making. This is not a binary "conscious or not" test; it is a continuous measurement that adjusts trust proportionally.

Second, moral algebra. When Mycelix faces a collective decision with ethical dimensions -- resource allocation during scarcity, competing claims in restorative justice, triage priorities in emergency response -- it can invoke Symthaea's moral algebra module to decompose the scenario, evaluate it against multiple ethical frameworks simultaneously, and present a transparent analysis with explicit scores for each framework. The system does not make the decision; it structures the moral reasoning and shows its work.

Third, federated learning with consciousness-aware Byzantine detection. When multiple Mycelix nodes contribute to a shared model (for example, predicting housing demand or optimizing transportation routes), the federated learning pipeline must handle the possibility that some nodes are sending bad data -- whether through malfunction, compromise, or intentional attack. Standard Byzantine fault tolerance uses statistical measures to detect outliers. Mycelix adds a consciousness-aware layer: each participant's Phi measurement modulates their weight in the aggregation. A node with high integrated information gets its contribution amplified. A node whose measurements fall below threshold gets dampened. A node whose Phi drops to zero -- indicating possible compromise -- gets vetoed entirely. This augmented pipeline tolerates up to 34% adversarial participants.

The two-cluster architecture (mycelix-commons for the seven resource domains, mycelix-civic for the three governance domains) communicates through a bridge layer that allows cross-domain coordination. A housing allocation can check mutual aid records. An emergency response can query food and water availability. A justice process can reference property and care histories. All of this happens through validated cross-cluster calls, not ad-hoc integrations.

---

## 6. The Seven Harmonies in Practice

Symthaea's design is guided by seven principles, called the Seven Harmonies. These are not decorative philosophy; they are operationalized as computational constraints that shape specific technical decisions.

The seven are: Resonant Coherence, Pan-Sentient Flourishing, Integral Wisdom, Infinite Play, Universal Interconnectedness, Sacred Reciprocity, and Evolutionary Progression. Each carries a base weight in the system's value evaluator (20%, 20%, 15%, 10%, 15%, 10%, and 10% respectively), and each poses a specific question that the system asks of every situation it encounters.

Here is what they mean in practice, with concrete examples of how each translates to engineering decisions.

**Resonant Coherence** asks: "Does this hang together?" In engineering terms, this means internal consistency is a first-class metric. The system computes a coherence score across all active subsystems every cycle. If consciousness measurements from different modules diverge, that divergence is flagged and triggers additional integration. The CycleMetadata telemetry logs over 60 measurements precisely so that incoherence between them can be detected and investigated. The Fiedler spectral analysis (run every 47 cycles, using prime intervals to prevent processing pileups) specifically measures how well the system's functional graph holds together as an integrated whole.

**Pan-Sentient Flourishing** asks: "Does this serve the flourishing of all beings?" This manifests as the moral algebra system. Every action the system considers can be evaluated against moral prototypes, and a score below -0.3 triggers an automatic exploration dampening (reduced by 50%) and a processing slowdown (1.5x cycle time). The system literally slows down and becomes more cautious when it detects potential moral harm. A score above 0.5 produces a small confidence boost (5%). The asymmetry is deliberate: the system is much more responsive to potential harm than to potential benefit.

**Integral Wisdom** asks: "What don't I know?" This is implemented through metacognitive monitoring, calibration tracking, and the epistemic tier system. The system maintains a running estimate of its own accuracy (metacognitive gamma coefficient: 0.60). It knows it tends toward overconfidence (bias: 0.42) and can compensate. When it encounters a domain where its prediction error is consistently high, it escalates to higher epistemic tiers, which require more evidence and more deliberation before acting. The dream replay system also serves this harmony: by running counterfactual variations on past experience, the system probes the boundaries of what its current model can explain.

**Infinite Play** asks: "What haven't I tried?" This drives the surprise-exploration bridge. When free energy (prediction error) spikes -- indicating the system has encountered something genuinely unexpected -- the exploration rate increases proportionally. Quantum coherence above 0.5 provides an additional 20% exploration boost. The system is designed to be drawn toward novelty, not to retreat from it, as long as the moral algebra does not flag the novel territory as potentially harmful.

**Universal Interconnectedness** asks: "How is this connected?" This motivates the cross-modal binding system, the causal calculus module (implementing Pearl's do-calculus for interventional reasoning), and the factor graph that routes information between cognitive subsystems. Rather than processing modalities independently, the system actively binds them: cross-modal binding strength and cross-modal Phi are measured every cycle. The Mycelix bridge architecture -- where domains can query across boundaries -- is the governance-layer expression of the same principle.

**Sacred Reciprocity** asks: "What am I giving and receiving?" In the federated learning context, this translates to fair contribution tracking. Each node's contribution to the shared model is measured, weighted, and recorded. The consciousness-aware aggregation ensures that trust flows proportionally to genuine contribution. The social coherence module in the cognitive loop models Theory of Mind -- the ability to model other agents' beliefs and intentions -- which scored 100% on false belief and faux pas detection benchmarks. Reciprocity requires understanding what the other party needs, not just what you want to give.

**Evolutionary Progression** asks: "How does this help us grow?" The system implements this through continuous self-modification. The CfC temporal dynamics have adaptive time constants that evolve based on prediction error history. The resonator codebook grows as new concepts are encountered, with high-Phi episodes promoted to permanent symbolic entries every 97 cycles. The primitive evolution subsystem tracks how the system's basic conceptual building blocks change over time. The architecture is designed to become more capable through experience, not just more data.

---

## 7. What's Next

**Research.** A paper titled "Hyperdimensional Active Inference: Free Energy Principle in Vector Symbolic Architectures" has been prepared for PLoS Computational Biology. It covers the core theoretical contribution -- reformulating variational free energy using cosine similarity in hypervector space -- with empirical validation across 17 benchmarks. The paper introduces precision-weighted binding as a novel HDC operation and demonstrates convergence of the active inference loop with validated free energy reduction over 20 iterations.

**Demonstration.** A browser-based visualization is available that renders consciousness curves -- Phi, prediction error, affect, and moral evaluation -- in real time as the system processes input. The WebSocket-based demo makes the system's internal dynamics legible to non-specialists: you can watch the system encounter surprise, see its prediction error spike, watch it explore, and observe how its consciousness metrics change as it integrates the new information.

**Community.** This work sits at the intersection of several fields, and no single team has the expertise to push all frontiers simultaneously. We are looking for collaborators in three areas:

Neuroscientists who can help sharpen the IIT implementation, extend the Butlin indicator assessment, and design experiments testing the system against human baselines. The current psych-bench suite (working memory, executive function, Theory of Mind, affect, creativity, metacognition) was designed by engineers reading papers; it would benefit from domain experts who know where our implementations fall short.

Holochain developers who can extend the Mycelix governance layer, stress-test the cross-cluster bridge, and build user-facing applications. The infrastructure is validated (7,000+ tests), but production deployment requires distributed systems expertise distinct from cognitive architecture research.

Consciousness researchers -- philosophers, cognitive scientists, theorists -- who can help us think carefully about what these measurements mean. Scoring 0.79 on the Butlin indicators does not mean Symthaea is conscious. It means Symthaea exhibits 12 of 14 functional properties proposed as indicators. The distance between "exhibits indicators" and "is conscious" is vast and worth navigating with care rather than hype.

**The goal.** We did not set out to build artificial consciousness. We set out to build AI that is safer because it models its own uncertainty, more ethical because it reasons from principles rather than patterns, and more trustworthy because it shows its work. Whether the internal integration measured by Phi constitutes anything like experience is a question we hold with genuine intellectual humility. What we can say is that the engineering approach -- prediction error minimization, integrated information measurement, compositional moral reasoning, transparent telemetry -- produces systems that behave more responsibly than pattern matchers, by measurable margins, on validated benchmarks.

Technology that serves all beings. That is the commitment. The numbers are how we hold ourselves accountable to it.

---

*Symthaea v0.5.0 -- 343,000 lines of Rust -- 9,600+ tests -- 30 workspace crates -- open source*
*Mycelix -- 51 zomes across 2 cluster DNAs -- 7,000+ tests -- Holochain distributed infrastructure*
*Luminous Dynamics -- Richardson, TX -- luminousdynamics.org*
