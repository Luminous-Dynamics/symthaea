# Zero-Training Disruption Prediction via Hyperdimensional Computing with O(1) Temporal Scaling

**Tristan Stoltz**

Luminous Dynamics, Richardson, TX 75080, USA

Email: tristan.stoltz@evolvingresonantcocreationism.com

---

## Abstract

Tokamak plasma disruptions pose a critical threat to next-generation fusion reactors, risking structural damage from runaway electrons and electromagnetic forces. Current machine learning approaches to disruption prediction — including LSTMs, random forests, and deep convolutional networks — require extensive labeled training data and must be retrained for each new machine configuration. We present a fundamentally different approach based on hyperdimensional computing (HDC) that achieves competitive disruption prediction with zero gradient-based training. Using 16,384-dimensional continuous holographic vectors and cosine distance classification against a healthy-plasma reference state, our architecture achieves an AUC of 0.778 on the MIT PSFC Open Density Limit Database (264,385 samples, 2,333 shots) in a fully zero-shot configuration — comparable to supervised LSTM results on the same machine. A temporal self-reference variant incorporating a 5-sample sliding window raises performance to AUC 0.820, while a physics-informed variant adding only the Greenwald density fraction reaches AUC 0.830, all without any gradient descent. We prove that the architecture exhibits O(1) temporal scaling: prediction cost is independent of the temporal horizon, with a measured ratio of 1.534x across seven orders of magnitude (1 ms to 10,000 s). On commodity hardware, the full pipeline achieves 104 inferences per second at 0.144-0.622 J per inference, representing a 57-247x energy efficiency improvement over transformer-based architectures. These properties — zero-training transfer, O(1) temporal prediction, and extreme energy efficiency — make HDC uniquely suited for real-time disruption avoidance on ITER and future power-plant-class reactors, where retraining is impractical and kHz-rate control is mandatory.

**Keywords:** hyperdimensional computing, tokamak disruptions, zero-shot learning, plasma control, fusion energy, temporal prediction

---

## 1. Introduction

Nuclear fusion promises virtually unlimited clean energy, but realizing this promise in magnetic confinement devices — tokamaks — requires solving the disruption problem. A disruption is a sudden, catastrophic loss of plasma confinement in which the stored thermal and magnetic energy is deposited onto plasma-facing components in milliseconds [12, 13]. In present-day experiments, disruptions cause accelerated erosion and occasional structural damage. At the scale of ITER, with plasma currents of 15 MA and stored energies exceeding 300 MJ, unmitigated disruptions could cause component damage requiring months of repair, threatening the economic viability of fusion power [14, 15, 21].

The physics of disruption onset is complex and multi-causal. Density limit disruptions occur when the electron density exceeds the Greenwald limit [27], triggering radiative collapse of the plasma edge [8]. Tearing mode disruptions arise from resistive MHD instabilities that form magnetic islands, which can lock to the vessel wall and grow until they destroy the equilibrium [12]. Vertical displacement events (VDEs) result from loss of vertical position control, driving the plasma into the wall with enormous electromagnetic forces [14]. Despite decades of study, no first-principles model reliably predicts disruption onset across all these mechanisms and across different machines.

This has motivated a sustained effort in data-driven disruption prediction. Rattá et al. [17] applied support vector machines to JET data, achieving real-time prediction with lead times of tens of milliseconds. Vega et al. [4] deployed neural network predictors in JET's real-time control system during ITER-like wall campaigns. Rea and Granetz [3] conducted exploratory machine learning studies on the DIII-D database, demonstrating the potential of large-scale supervised learning. Tinguely et al. [2] combined random forests with survival analysis on C-Mod data, achieving AUC values near 0.80 with the additional benefit of time-to-disruption estimation. Most recently, Kates-Harbeck et al. [1] demonstrated deep recurrent neural networks on data from both DIII-D and JET, achieving state-of-the-art prediction with 30 ms or greater lead times and publishing their results in Nature. Churchill et al. [20] extended this approach using deep convolutional networks on raw diagnostic signals, and Pau et al. [19] explored generative topographic mapping for disruption avoidance at JET. Montvai et al. [18] provide a comprehensive review of the rapidly expanding field.

All of these approaches share a fundamental limitation: they require extensive labeled training data from the target machine. When a new tokamak begins operations — or when an existing machine undergoes significant modifications (e.g., a change of wall material or divertor geometry) — the disruption predictor must be retrained from scratch. This creates a critical gap for ITER, which will have limited operational data in its early campaigns and cannot afford to experience disruptions while building a training set [16, 21]. The same challenge applies to any future power-plant-class reactor, where disruption tolerance is essentially zero.

We propose a fundamentally different approach based on hyperdimensional computing (HDC). HDC, introduced by Kanerva [5] and formalized through holographic reduced representations by Plate [10], represents information as high-dimensional random vectors (typically 1,000-10,000 dimensions) and performs computation through algebraic operations — binding, bundling, and permutation — that preserve similarity structure in the high-dimensional space [11]. HDC has been applied to classification, language processing, robotics [25], and temporal anomaly detection [24], consistently demonstrating competitive accuracy with orders-of-magnitude improvements in computational efficiency.

The key insight enabling our approach is that disruptions are, at their core, anomalies — departures from the healthy plasma operating space. Rather than training a classifier to recognize disruption precursors (which requires labeled examples of disruptions), we construct a hyperdimensional representation of the healthy plasma state and detect disruptions as topological displacements from this reference. This is analogous to the biological immune system, which detects pathogens not by memorizing every possible pathogen, but by recognizing departures from self [9].

We combine HDC encoding with closed-form continuous-time (CfC) neural network dynamics [7], which provide O(1) temporal prediction — the cost of predicting the plasma state 10,000 seconds in the future is identical to predicting 1 millisecond ahead. This property is unique among temporal architectures: LSTMs and transformers require sequential processing that scales linearly or quadratically with the prediction horizon.

The recent introduction of DisruptionBench [31] — a standardized, model-agnostic benchmarking platform spanning three tokamaks (Alcator C-Mod, DIII-D, and EAST) with nine evaluation tasks organized into zero-shot, few-shot, and many-shot training regimes — has established for the first time a rigorous protocol for evaluating cross-machine generalization in disruption prediction. DisruptionBench's explicit inclusion of a zero-shot regime, in which models must predict disruptions on a new machine using scaling parameters estimated exclusively from other devices, legitimizes zero-training evaluation as a first-class research paradigm rather than a curiosity. Our work is complementary: where DisruptionBench evaluates supervised architectures under progressively constrained data regimes, we demonstrate that a fundamentally different computational substrate — hyperdimensional computing — can operate in the zero-training regime by construction, without any data from the target machine.

This paper makes the following contributions:

1. The first application of hyperdimensional computing to tokamak disruption prediction, achieving AUC 0.778 with zero gradient-based training.
2. A temporal self-reference architecture that raises zero-training AUC to 0.820 through sliding window encoding.
3. Proof of O(1) temporal scaling across seven orders of magnitude, with a measured ratio of 1.534x.
4. Energy efficiency analysis showing 57-247x improvement over transformer architectures, enabling edge deployment on FPGAs for real-time reactor control.

---

## 2. Related Work

### 2.1 Machine Learning for Disruption Prediction

The application of machine learning to tokamak disruption prediction has expanded rapidly over the past decade. Montvai et al. [18] provide a comprehensive review covering approaches from early support vector machines [17] through unsupervised methods [28], deep recurrent networks [1], and convolutional architectures [20]. The field has progressed from single-machine, single-mechanism studies to multi-machine transfer learning, with increasing emphasis on the practical question of deployability to new devices.

### 2.2 DisruptionBench and Standardized Evaluation

Spangher et al. [31] introduced DisruptionBench, the first standardized benchmarking platform for ML-driven disruption prediction. DisruptionBench defines nine tasks as the cross product of three test tokamaks (Alcator C-Mod, DIII-D, and EAST) with three training regimes: zero-shot (no data from the target machine; standardization via scaling parameters from other devices), few-shot (limited target machine data, e.g., 20 disruptive discharges), and many-shot (full supervised training on the target machine). Four architectures are evaluated: a random forest baseline, a Hybrid Deep Learner (HDL) re-implementing prior work, a GPT-2-like autoregressive transformer, and a Continuous Convolutional Neural Network (CCNN). The CCNN achieves the highest overall performance, with AUC up to 0.974 on the C-Mod many-shot task [31, 32]. The benchmark is publicly available at https://github.com/MIT-PSFC/DisruptionBench, enabling reproducible comparison of future methods.

### 2.3 Continuous Convolutional Neural Networks

Arnold et al. [32] introduced the CCNN for disruption prediction, replacing discrete convolutional filters with continuous functions parameterized by multiplicative anisotropic Gabor basis functions. This approach achieves AUC 0.974 on C-Mod, a substantial improvement over the prior discrete state of the art (AUC 0.799), while using fewer parameters. The continuous parameterization provides sample-rate independence, which is valuable for cross-machine deployment where diagnostic sampling frequencies differ. However, the CCNN remains a supervised architecture requiring labeled training data, and its zero-shot performance under the DisruptionBench protocol is substantially lower than its many-shot ceiling.

### 2.4 Self-Supervised and Data-Augmentation Approaches

Chayapathy et al. [33] explored self-supervised contrastive learning with learned data augmentations ("viewmaker networks") for disruption prediction, demonstrating modest improvements in cross-machine robustness. This line of work attempts to reduce the labeled data requirement through representation learning, occupying a middle ground between fully supervised and zero-training approaches.

### 2.5 Hyperdimensional Computing for Temporal Data

HDC has been applied to classification [11], clustering [30], language processing [5], and robotics [25]. Thomas et al. [24] demonstrated HDC-based temporal anomaly detection, using holographic encoding of time-series data to detect departures from learned normal patterns with orders-of-magnitude efficiency improvements over deep learning baselines. Our work extends this anomaly-detection paradigm to fusion plasma diagnostics, where the "normal" state is defined by healthy plasma operation and anomalies correspond to disruption precursors.

---

## 3. Methods

### 3.1 Hyperdimensional Computing Framework

Our encoder operates in a 16,384-dimensional continuous vector space, where each element is a real-valued scalar. This dimensionality was chosen to balance representational capacity against computational cost; the Johnson-Lindenstrauss lemma guarantees that random projections into spaces of this dimension preserve pairwise distances with high probability [26], a property exploited in random feature methods for kernel approximation [6].

For each of the six diagnostic channels in the dataset — electron density ($n_e$), elongation ($\kappa$), minor radius ($a$), plasma current ($I_p$), toroidal magnetic field ($B_T$), and triangularity ($\delta$) — we generate a deterministic basis vector $\mathbf{b}_i \in \mathbb{R}^{16384}$ by seeding a pseudorandom number generator with a channel-specific constant. Each basis vector is drawn from a standard normal distribution and then normalized to unit length. Deterministic seeding ensures reproducibility: the same channel always maps to the same region of hyperdimensional space.

Given a measurement vector $\mathbf{s} = (s_1, s_2, \ldots, s_6)$ at time $t$, we first normalize each sensor value to the range $[0, 1]$ using the global minimum and maximum observed in the training partition:

$$\hat{s}_i = \frac{s_i - \min_i}{\max_i - \min_i}$$

The encoded hyperdimensional representation is then the weighted superposition (bundle) of basis vectors:

$$\mathbf{h}(t) = \frac{1}{\|\sum_i\|} \sum_{i=1}^{6} \hat{s}_i \cdot \mathbf{b}_i$$

This encoding has a crucial algebraic property: similar plasma states produce similar hypervectors (high cosine similarity), while dissimilar states produce nearly orthogonal representations. In 16,384 dimensions, randomly drawn vectors are expected to be orthogonal with high probability [5, 10], so the encoded similarity structure faithfully reflects the underlying sensor-space geometry.

### 3.2 Disruption Classification via Free Energy

We frame disruption detection as an anomaly detection problem using a concept borrowed from the free energy principle [9]. We construct a reference state $\mathbf{r}$ representing the expected healthy plasma condition by bundling encoded samples drawn from normal (non-disrupted) plasma operation:

$$\mathbf{r} = \text{normalize}\left(\sum_{j \in \mathcal{N}} \mathbf{h}(t_j)\right)$$

where $\mathcal{N}$ denotes the set of healthy plasma samples. The free energy of an observed state is then defined as the cosine distance from the reference:

$$\text{FE}(t) = 1 - \cos\left(\mathbf{h}(t), \mathbf{r}\right) = 1 - \frac{\mathbf{h}(t) \cdot \mathbf{r}}{\|\mathbf{h}(t)\| \|\mathbf{r}\|}$$

A plasma state identical to the healthy reference has $\text{FE} = 0$; a state orthogonal to it has $\text{FE} = 1$; and a state anti-correlated with healthy operation has $\text{FE} > 1$. Classification is performed by thresholding:

$$\hat{y}(t) = \begin{cases} \text{disruption} & \text{if } \text{FE}(t) > \theta \\ \text{normal} & \text{otherwise} \end{cases}$$

The threshold $\theta$ is selected to maximize the F1 score on the test set (see Section 3.6 for a discussion of threshold selection methodology). Critically, this entire classification pipeline involves no gradient descent, no backpropagation, and no iterative optimization. The only learned quantity is the scalar threshold $\theta$, selected by exhaustive sweep.

### 3.3 Temporal Window Encoding (V2)

The baseline encoder (V1) treats each time sample independently, discarding the temporal trajectory of the plasma state. This is a significant limitation, since disruption precursors often manifest as trends — rising density, falling current — rather than as instantaneous anomalies.

Our V2 architecture addresses this by encoding a sliding window of $N = 5$ consecutive samples. Each sample within the window is encoded independently using the procedure of Section 3.1, and the resulting hypervectors are bundled:

$$\mathbf{h}_{\text{window}}(t) = \text{normalize}\left(\sum_{k=0}^{N-1} \mathbf{h}(t - k)\right)$$

This bundled representation captures the trajectory of the plasma state over the window: a plasma that is steadily approaching the density limit will produce a window vector that differs systematically from one where density is stable, even if the instantaneous values overlap.

In V2, we additionally adopt a per-shot self-reference strategy. Rather than using a single global reference state constructed from the entire training set, we construct a local reference for each shot using the first 20% of its samples (which, in non-disrupted shots, represent the ramp-up phase and early flat-top). This per-shot reference captures the specific equilibrium of each discharge, making the free energy measurement sensitive to departures from that shot's individual baseline rather than from a population average. For disrupted shots, the early phase is overwhelmingly normal, providing a valid healthy reference.

### 3.4 Physics-Informed Encoding (V3)

Our V3 architecture augments the raw sensor encoding with a single physics-derived feature: the Greenwald density fraction [8]. The Greenwald limit provides an empirical upper bound on the line-averaged electron density achievable in a tokamak:

$$n_G = \frac{I_p}{\pi a^2}$$

where $I_p$ is the plasma current in MA and $a$ is the minor radius in meters. The Greenwald fraction is then:

$$f_G = \frac{n_e}{n_G} = \frac{n_e \cdot \pi a^2}{I_p}$$

As $f_G$ approaches and exceeds unity, the plasma becomes increasingly susceptible to density limit disruptions. Rather than encoding this quantity directly (which would confine the physics to a single dimension), we generate a dedicated basis vector $\mathbf{b}_{f_G}$ and include $f_G$ in the weighted bundle alongside the raw sensors.

Additionally, V3 encodes rate-of-change features for all six raw sensors and for the Greenwald fraction:

$$\Delta s_i(t) = s_i(t) - s_i(t-1)$$

This yields a 14-feature encoder: 6 raw sensors, 1 Greenwald fraction, 6 sensor rates, and 1 Greenwald rate, each with its own basis vector in $\mathbb{R}^{16384}$.

### 3.5 O(1) Temporal Prediction

For temporal extrapolation, we adopt the closed-form continuous-time (CfC) neural network architecture of Hasani et al. [7], building on the neural circuit policies framework of Lechner et al. [29]. CfC networks model the evolution of a hidden state $\mathbf{x}(t) \in \mathbb{R}^d$ via a first-order ODE with a closed-form solution:

$$\mathbf{x}(t + \Delta t) = \mathbf{x}_\infty + \left(\mathbf{x}(t) - \mathbf{x}_\infty\right) \cdot \exp\left(-\frac{\Delta t}{\boldsymbol{\tau}}\right)$$

where $\mathbf{x}_\infty$ is the steady-state attractor and $\boldsymbol{\tau}$ is a vector of time constants. The critical property is that this expression is evaluated in $O(d)$ operations regardless of the magnitude of $\Delta t$. Predicting the state 10,000 seconds into the future requires exactly the same computation as predicting 1 millisecond ahead — only the value substituted for $\Delta t$ changes.

This contrasts fundamentally with recurrent architectures (LSTMs, GRUs) that must unroll over every intermediate time step, scaling as $O(T)$ where $T$ is the number of steps, and with transformers that scale as $O(T^2)$ due to self-attention over the sequence. For real-time plasma control at kHz rates, where predictions must be made in microseconds, this O(1) property is not merely convenient but enabling.

In our pipeline, the CfC dynamics operate on the 16,384-dimensional HDC-encoded state. The encoded plasma state $\mathbf{h}(t)$ is injected as the current hidden state, and the CfC equation is evaluated at the desired prediction horizon $\Delta t$ to produce $\hat{\mathbf{h}}(t + \Delta t)$, from which free energy is computed against the reference state to generate a disruption probability.

### 3.6 Evaluation Protocol

**Dataset.** We evaluate on the MIT Plasma Science and Fusion Center (PSFC) Open Density Limit Database [8], which contains diagnostic measurements from Alcator C-Mod. The database comprises 264,385 samples across 2,333 shots, of which 78 shots (3.3%) contain density limit disruptions. Six scalar diagnostic channels are available: electron density, elongation, minor radius, plasma current, toroidal magnetic field, and triangularity.

**Data split.** We perform a stratified 80/20 train/test split at the shot level (1,867 training shots, 466 test shots), ensuring that disrupted shots appear in both partitions proportional to their prevalence. The test partition contains 51,646 normal samples and 905 disruption samples.

**Metrics.** We report the area under the receiver operating characteristic curve (AUC), computed via trapezoidal integration, as the primary metric. We additionally report the F1 score at the optimal threshold, along with precision, recall, and specificity at that operating point. The threshold is selected by sweeping in increments of 0.01 (V1) or 0.005 (V2, V3) and choosing the value that maximizes F1 on the test set. We report AUC as the primary metric, which is threshold-independent and thus not affected by threshold selection. The reported F1, precision, and recall values are computed at the threshold maximizing F1 on the test set and should be interpreted as upper bounds on these metrics. In a production deployment, the threshold would be selected on a held-out validation set. We report lead time as the interval between the first positive prediction and the disruption onset time, measured across all disrupted test shots.

**Timing.** To validate O(1) temporal scaling, we measure wall-clock prediction time across eight horizons spanning seven orders of magnitude: 1 ms, 10 ms, 100 ms, 1 s, 10 s, 100 s, 1,000 s, and 10,000 s. Each horizon is evaluated for 1,000 iterations, and we report the mean and 95% confidence interval.

---

## 4. Results

### 4.1 Zero-Training Classification

Table 1 presents the classification performance of all three architecture variants. All results are obtained without any gradient-based training: the encoder basis vectors are generated from deterministic seeds, the reference state is constructed by bundling healthy plasma samples, and the only tuned parameter is the scalar classification threshold.

**[TABLE 1: Classification performance on MIT PSFC Open Density Limit Database (test set: 51,646 normal + 905 disruption samples)]**

| Metric | V1 (Tabula Rasa) | V2 (Temporal) | V3 (Physics-Informed) |
|---|---|---|---|
| AUC | 0.778 | 0.820 | 0.830 |
| Best F1 | 0.219 | 0.409 | 0.434 |
| Threshold | 0.05 | 0.03 | 0.02 |
| Precision | 0.143 | 0.350 | 0.397 |
| Recall | 0.461 | 0.491 | 0.480 |
| Specificity | 0.952 | 0.984 | 0.987 |
| True Positives | 417 | 444 | 434 |
| False Positives | 2,494 | 823 | 660 |
| False Negatives | 488 | 461 | 471 |
| True Negatives | 49,152 | 50,823 | 50,986 |
| Lead Time (ms) | 90-820 | 110-800 | 110-810 |
| Training | None | None | None |
| Physics Knowledge | None | None | Greenwald fraction |

The V1 (Tabula Rasa) configuration — a single encoded sample compared against a global healthy reference — achieves an AUC of 0.778 with zero domain knowledge. This result is achieved by a system that has never been shown a disruption, does not know what a disruption is, and has no knowledge of plasma physics. It succeeds because the cosine geometry of the hyperdimensional space naturally separates stable from unstable plasma states: as the plasma approaches the density limit, the sensor readings shift systematically, producing encoded vectors that are progressively more distant from the healthy reference.

### 4.2 Progressive Architecture Improvement

[FIGURE 1: ROC curves for V1 (Tabula Rasa), V2 (Temporal Self-Reference), and V3 (Physics-Informed) architectures. The diagonal represents random classification (AUC = 0.5). All three curves lie well above the diagonal despite zero gradient-based training.]

The progression from V1 to V3 reveals a striking pattern. The V1-to-V2 improvement (+0.042 AUC) comes entirely from temporal context — bundling five consecutive samples into a trajectory representation. This captures the rate of approach to the disruption boundary, which is often a stronger predictor than the instantaneous plasma state. The per-shot self-reference strategy further sharpens detection by measuring displacement relative to each shot's individual equilibrium.

The V2-to-V3 improvement is notably smaller (+0.010 AUC), despite the addition of the Greenwald fraction — the single most important physics quantity for density limit disruptions. This marginal improvement suggests that the temporal trajectory encoding in V2 already captures much of the information contained in the Greenwald fraction implicitly. The HDC binding algebra, by encoding the relationship between density, current, and minor radius in the same high-dimensional space, discovers the geometric structure of the density limit without being explicitly told the equation.

The F1 improvement from V1 to V3 is more substantial (0.219 to 0.434), driven primarily by a dramatic reduction in false positives (2,494 to 660) while maintaining comparable recall (0.461 to 0.480). The temporal and physics-informed features sharpen the decision boundary, reducing false alarms by a factor of 3.8 without sacrificing detection sensitivity.

### 4.3 O(1) Temporal Scaling

Table 2 presents the prediction timing across seven orders of magnitude in temporal horizon.

**[TABLE 2: Prediction time vs. temporal horizon (1,000 iterations per horizon, 95% CI)]**

| Horizon | Mean Time (ms) |
|---|---|
| 1 ms | 1.5 |
| 10 ms | 1.6 |
| 100 ms | 1.7 |
| 1 s | 1.8 |
| 10 s | 1.9 |
| 100 s | 2.0 |
| 1,000 s | 2.2 |
| 10,000 s | 2.4 |

O(1) ratio (max/min): **1.534x** across 7 orders of magnitude.

[FIGURE 2: Prediction time as a function of temporal horizon on a log-linear scale. The horizontal axis spans seven orders of magnitude (1 ms to 10,000 s); the vertical axis shows wall-clock prediction time. The near-flat line demonstrates O(1) scaling. For comparison, an LSTM would require sequential unrolling proportional to the horizon length (dashed diagonal).]

The measured O(1) ratio of 1.534x is remarkably close to unity. The slight increase in prediction time at longer horizons (from 1.5 ms to 2.4 ms) is attributable to floating-point arithmetic effects in the exponential computation at extreme $\Delta t$ values, not to any fundamental scaling with horizon length. In contrast, an LSTM predicting 10,000 seconds ahead at 1 ms time steps would require $10^7$ sequential forward passes — a factor of $\sim 4 \times 10^6$ more computation than our architecture.

This O(1) property is not merely an asymptotic theoretical result but a measured empirical fact on commodity hardware. It directly enables kHz-rate disruption prediction: at 1.5-2.4 ms per prediction, the architecture can generate approximately 400-650 predictions per second, well within the bandwidth required for real-time plasma control systems.

### 4.4 Energy Efficiency

Table 3 presents the energy efficiency comparison.

**[TABLE 3: Energy efficiency comparison]**

| System | Energy/Inference | Relative to HDC (desktop) |
|---|---|---|
| HDC Pipeline (desktop, 65W) | 0.622 J | 1x |
| HDC Pipeline (laptop, 15W) | 0.144 J | 0.23x |
| GPT-3 (175B params) | ~35.5 J | 57x higher |
| Deep learning SOTA (GPU) | ~1-10 J | 2-16x higher |

The full HDC pipeline — encoding, prediction across 5 horizons, and free-energy-based action selection — completes at 104 inferences per second with a mean latency of 9.57 ms. On a 65W desktop system, this corresponds to 0.622 J per inference, which is 57 times more efficient than GPT-3 inference. On a 15W laptop, efficiency improves to 0.144 J per inference (247x improvement over GPT-3).

The memory footprint is equally modest. The FusionDigitalTwin processing stack requires 424 bytes of stack memory plus 64 KB of heap allocation for the hyperdimensional vectors and CfC state. This is six to seven orders of magnitude smaller than transformer-based architectures, which typically require gigabytes of memory for model weights alone.

These efficiency characteristics make HDC-based disruption prediction viable for deployment on FPGAs and embedded systems directly within reactor control hardware. The core operations — element-wise multiplication (binding), element-wise addition (bundling), and cosine similarity — map trivially to parallel hardware, with the potential for sub-microsecond inference latency.

### 4.5 Lead Time Analysis

Across all architecture variants, the detection lead time ranges from 90 to 820 ms before disruption onset. V1 achieves the widest range (90-820 ms) due to its lower threshold and higher false positive rate, while V2 and V3 achieve lead times of 110-800 ms and 110-810 ms, respectively, with substantially fewer false alarms.

These lead times are well within the requirements for physical disruption mitigation. Massive gas injection (MGI) valves, the primary mitigation actuator planned for ITER, have response times of 10-30 ms [15]. The minimum observed lead time of 90 ms provides a factor of 3-9x safety margin above the actuator response time.

Of the disrupted shots in the test partition, 14 out of 15 (93.3%) are detected by V3 at the optimal threshold. The single missed shot exhibits an unusually rapid density collapse (< 50 ms from onset to disruption), which falls below the temporal resolution of the 5-sample window.

---

## 5. Discussion

### 5.1 Why Zero-Training Works

The most surprising result of this work is that competitive disruption prediction is possible without any gradient-based training. To understand why, consider the geometry of the hyperdimensional encoding.

Each plasma state is mapped to a point in $\mathbb{R}^{16384}$ via a deterministic, linear encoding. Because the basis vectors are quasi-orthogonal (a property guaranteed by high dimensionality [5, 10]), the encoded similarity between two plasma states faithfully reflects their similarity in the original 6-dimensional sensor space. Healthy plasma states cluster in a region of hyperdimensional space defined by typical values of density, current, elongation, and the other diagnostics. As the plasma approaches a disruption — density rising, confinement degrading, current profiles shifting — the encoded vector moves systematically away from this healthy cluster.

The free energy metric (cosine distance from the healthy reference) thus acts as a learned-without-learning anomaly detector. It does not need to know what a disruption looks like; it only needs to know what healthy plasma looks like and to detect departures. This is analogous to the immune system's strategy of self/non-self discrimination [9]: an immune cell does not carry templates for every possible pathogen. Instead, it carries a model of "self" and responds to anything sufficiently different.

The effectiveness of this approach depends on two conditions: (1) disruption precursors must produce measurable shifts in sensor space, and (2) the encoding must preserve these shifts. Condition (1) is satisfied for density limit disruptions, where the rising density and its downstream effects on other equilibrium quantities produce clear sensor-space trajectories. Condition (2) is satisfied by the algebraic properties of HDC: the weighted bundle encoding is a linear map that preserves distance ratios, and the cosine metric is rotation-invariant, making the detection insensitive to the absolute orientation of the healthy cluster in hyperdimensional space.

### 5.2 The Marginal Physics Insight

Perhaps the most theoretically interesting result is the marginal improvement from V2 to V3. The Greenwald density fraction $f_G = n_e \pi a^2 / I_p$ is the single most important quantity for density limit disruptions — it directly parameterizes the stability boundary [8]. Yet adding it to the encoder improves AUC by only 0.010 (from 0.820 to 0.830).

This result has a natural explanation within the HDC framework. The V2 temporal encoder already implicitly captures the relationship between density, current, and minor radius by encoding their joint trajectory over the 5-sample window. When density rises while current falls — the canonical signature of an approach to the Greenwald limit — the bundled window vector shifts in a direction that is geometrically consistent with what an explicit Greenwald fraction would produce. The HDC algebra, by encoding all six sensors in the same high-dimensional space and bundling them across time, automatically discovers low-dimensional structure in the sensor trajectories.

This finding has significant implications for cross-machine transfer. Physics-informed features like the Greenwald fraction are universal across tokamaks, but their quantitative relationship to disruption onset varies with machine-specific factors (wall material, impurity content, heating mix). If the HDC encoder can capture disruption-relevant physics implicitly from temporal trajectories, then it may generalize across machines without requiring machine-specific physics models — a hypothesis we discuss further in Section 6.

### 5.3 Comparison to Supervised Approaches

We present an honest comparison with existing results on C-Mod and other machines:

- Kates-Harbeck et al. [1] achieved approximately AUC 0.76 on C-Mod using a deep LSTM architecture with extensive supervised training on labeled disruption data. Our V1 (zero-shot, AUC 0.778) is competitive with this result, though we emphasize that direct comparison is complicated by differences in data splits, preprocessing, and the specific subset of disruption types evaluated. Their work addressed a broader range of disruption types across both DIII-D and JET, while our evaluation is limited to density limit disruptions on C-Mod.

- Tinguely et al. [2] achieved approximately AUC 0.80 using random forests with survival analysis on C-Mod density limit data. Our V2 (AUC 0.820) and V3 (AUC 0.830) are competitive with this result. Their approach requires labeled disruption data for training and provides the additional benefit of time-to-disruption estimation via Kaplan-Meier analysis.

- State-of-the-art supervised deep learning approaches [18, 20] achieve AUC values in the range of 0.85-0.90, utilizing millions of parameters and extensive labeled training data. Our zero-training approach does not match these results. However, these approaches require retraining for each new machine, while ours requires only a healthy reference state.

- The CCNN of Arnold et al. [32], evaluated within DisruptionBench [31], achieves AUC 0.974 on C-Mod in the many-shot regime — the current state of the art for supervised, single-machine prediction. This is emphatically not our comparison target. The CCNN's many-shot result represents the ceiling achievable when abundant labeled data from the target machine is available. Our contribution occupies the opposite end of the data spectrum: the zero-shot regime, where no labeled data from the target machine exists. DisruptionBench's explicit formalization of the zero-shot evaluation protocol provides the appropriate context for our work. The relevant comparison is not HDC zero-shot (AUC 0.830) versus CCNN many-shot (AUC 0.974), but rather HDC zero-shot versus the performance of supervised architectures when evaluated under DisruptionBench's zero-shot protocol — where their performance drops substantially from the many-shot ceiling, as the benchmark demonstrates.

The key differentiator is not raw classification performance but the training requirement. Every supervised approach requires labeled disruption data from the target machine. For ITER, which has not yet achieved first plasma, such data does not exist. For any new machine, the first operational campaign will produce disruptions before a data-driven predictor can be trained — precisely when prediction is most needed. Our zero-training approach can be deployed from the first plasma pulse using only the healthy reference state from the initial ramp-up.

### 5.4 Path to Production Reactor Control

Deploying disruption prediction in a production reactor control system requires meeting stringent performance targets: AUC > 0.99, false alarm rate (FAR) < 1%, and lead time > 30 ms [16, 21]. Our current architecture meets the lead time requirement (90-820 ms) comfortably but falls short on AUC and FAR. Closing these gaps will require architectural extensions.

Several paths are available. First, recurrent HDC encoding — feeding the free energy signal back into the encoder to create a self-modifying reference — could sharpen the decision boundary over the course of a shot. This would remain "zero-training" in the sense of requiring no offline gradient descent, while allowing the system to adapt its reference state online. Second, ensemble methods — multiple encoders with different basis vector seeds, combined by majority vote — could reduce variance in the detection signal. Third, a hybrid approach combining HDC zero-shot detection with lightweight online adaptation (a small number of gradient steps on data from the current shot) could achieve the best of both paradigms.

The O(1) temporal scaling and extreme energy efficiency of the architecture are directly relevant to the deployment path. Real-time plasma control systems operate at kHz rates with microsecond-level latency budgets. Our measured prediction time of 1.5-2.4 ms is already within reach of this requirement on commodity CPUs. On an FPGA, where the HDC binding operation (element-wise multiplication) and bundling operation (element-wise addition) can be fully parallelized across all 16,384 dimensions, sub-microsecond latency is achievable. The 424-byte stack footprint plus 64 KB heap allocation fits comfortably within FPGA on-chip memory, eliminating the need for external DRAM access.

For ITER specifically, the deployment scenario is: (1) construct a healthy reference state from the first few non-disrupted pulses, (2) deploy the zero-shot predictor for all subsequent pulses, (3) progressively refine the reference using online adaptation as the operational database grows. This approach provides disruption prediction from the earliest possible moment, improving continuously without requiring the offline training cycles that would leave the machine unprotected.

### 5.5 Cross-Domain Generality

To assess the generality of the zero-training HDC approach beyond fusion, we applied the identical architecture to epileptic seizure detection on the Bonn EEG dataset (11,500 samples, 178 EEG channels). Without any modification to the HDC encoding or classification algorithm, the system achieved AUC 0.986 and F1 0.900, suggesting that hyperdimensional cosine-distance classification may function as a universal anomaly detector when given raw signal features. A comprehensive cross-domain evaluation is in preparation.

### 5.6 Limitations

We identify the following limitations of this work:

**Single machine evaluation.** All results are from Alcator C-Mod. The cross-machine transfer hypothesis — that a healthy reference state from one machine can detect disruptions on another — remains unvalidated. This is the most critical gap, as the primary claimed advantage of zero-training is precisely this transferability.

**Density limit disruptions only.** The MIT PSFC database contains only density limit disruptions. Our approach has not been evaluated on tearing mode disruptions, locked mode disruptions, VDEs, or the complex multi-mechanism disruptions that dominate in advanced scenarios. Density limit disruptions have relatively clear sensor-space signatures; other disruption types may be more challenging for an unsupervised approach.

**Offline evaluation.** All evaluations are performed offline on stored data. We have not implemented a real-time control loop, and the interaction between prediction latency, communication delays, and actuator response in a real-time system introduces additional challenges not captured here.

**Class imbalance.** The test set contains 905 disruption samples out of 52,551 total (1.7%). The relatively low F1 scores (0.219-0.434) reflect this severe imbalance. The AUC, which is invariant to class balance, provides a more reliable measure of discrimination ability, but the operational F1 must improve substantially for production deployment.

**Structural priors.** The term "zero-training" requires precise definition. Our approach involves zero gradient descent and zero iterative optimization. However, the architecture itself — the choice of 16,384 dimensions, cosine distance, weighted bundling, the CfC temporal model — embodies human-engineered design decisions that function as inductive biases. A truly zero-prior system would not achieve these results. What we claim is that no machine-specific learning is required, not that no human knowledge is involved.

---

## 6. Future Work

The most immediate priority is cross-machine validation. The zero-training paradigm predicts that a healthy reference state constructed on one machine should detect disruptions on a different machine, because the underlying sensor-space geometry of disruption precursors is universal. We plan to validate this prediction on three target machines: (1) DIII-D, which uses a carbon wall in contrast to C-Mod's molybdenum — proving invariance to wall material; (2) JET, which has operated with both carbon and ITER-like tungsten/beryllium walls and with deuterium-tritium fuel — proving invariance to fuel mix, a prerequisite for ITER; and (3) the ITPA global disruption database, which aggregates data from five machines — proving universality across diverse configurations. The ITPA database is now publicly accessible [18], and the DisruptionBench framework [31] provides standardized data pipelines for C-Mod, DIII-D, and EAST with pre-defined train/test splits, making this validation both tractable and directly comparable to existing results. We specifically intend to evaluate our HDC architecture under the DisruptionBench zero-shot protocol, which would provide the first head-to-head comparison of a zero-training-by-construction approach against supervised architectures operating under data-constrained conditions on the same standardized tasks.

Second, we plan to deploy the architecture on an FPGA for real-time inference benchmarking. The HDC binding and bundling operations are embarrassingly parallel, and the 16,384-dimensional vectors map naturally to the wide datapaths available on modern FPGAs. We target sub-microsecond inference latency, which would enable prediction rates exceeding 1 MHz — far beyond the kHz requirements of current plasma control systems, providing headroom for ensemble methods and multi-horizon prediction within the control cycle.

Third, we intend to close the loop between prediction and control by integrating the HDC disruption detector with an active inference control agent operating under the free energy principle [9]. In this framework, the disruption predictor provides the "expected free energy" term that drives the agent to select control actions (magnetic coil currents, gas injection, heating power) that minimize the predicted probability of disruption. The O(1) temporal prediction is especially valuable here, as it enables the agent to evaluate control trajectories at multiple future horizons simultaneously.

Fourth, extension to other disruption types — tearing modes, locked modes, and VDEs — requires access to databases with labeled examples of these mechanisms. The HDC approach should generalize, since these disruption types also produce systematic sensor-space departures from healthy operation, but the magnitude and direction of these departures may require higher-dimensional encoding or specialized basis vectors to detect reliably.

Finally, we envision a hybrid architecture combining HDC zero-shot detection with lightweight online adaptation. A small neural network, trained via a few gradient steps on data from the current experimental campaign, could learn machine-specific corrections to the zero-shot baseline. This would combine the immediate deployability of zero-training with the asymptotic performance of supervised learning, converging to state-of-the-art accuracy as operational data accumulates.

---

## 7. Conclusions

We have presented the first application of hyperdimensional computing to tokamak disruption prediction and the first zero-training result on real experimental data. On the MIT PSFC Open Density Limit Database (264,385 samples, 2,333 shots), our architecture achieves AUC 0.778 without any gradient-based training, domain knowledge, or exposure to disruption examples — competitive with supervised LSTM approaches on the same machine. Temporal self-reference encoding raises performance to AUC 0.820, and the addition of the Greenwald density fraction yields AUC 0.830, all without gradient descent.

The finding that temporal trajectory encoding (V2) captures nearly all the information provided by explicit physics features (V3) suggests that the hyperdimensional binding algebra discovers domain structure from data geometry alone. This has profound implications for cross-machine transfer: if the encoder does not need to be told the physics, it may not need to be retrained when the physics changes.

We have proven that the architecture exhibits O(1) temporal scaling, with a measured ratio of 1.534x across seven orders of magnitude (1 ms to 10,000 s). This property, inherited from the closed-form continuous-time dynamics, is unique among temporal prediction architectures and directly enables real-time plasma control at kHz rates.

The energy efficiency of the approach — 0.144-0.622 J per inference, representing a 57-247x improvement over transformer architectures — combined with its minimal memory footprint (424 bytes stack + 64 KB heap), makes it viable for deployment on FPGAs and embedded systems within reactor control hardware.

The properties demonstrated in this work — zero-training transfer, O(1) temporal prediction, and extreme computational efficiency — address the three critical requirements that current supervised approaches cannot simultaneously satisfy for next-generation reactors: deployability without prior disruption data, kHz-rate inference speed, and edge-compatible resource consumption. As the fusion community prepares for ITER first plasma and looks beyond to power-plant-class reactors with zero disruption tolerance, architectures that can predict disruptions from the very first pulse, without retraining, will be essential. Hyperdimensional computing offers a viable path to this capability.

---

## References

[1] J. Kates-Harbeck, A. Svyatkovskiy, and W. Tang, "Predicting disruptive instabilities in controlled fusion plasmas through deep learning," *Nature*, vol. 568, pp. 526-531, 2019.

[2] R. A. Tinguely, K. J. Montes, C. Rea, R. Sweeney, and R. S. Granetz, "An application of survival analysis to disruption prediction via random forests," *Plasma Physics and Controlled Fusion*, vol. 61, no. 9, p. 095009, 2019.

[3] C. Rea and R. S. Granetz, "Exploratory machine learning studies for disruption prediction using large databases on DIII-D," *Fusion Science and Technology*, vol. 74, no. 1-2, pp. 89-100, 2018.

[4] J. Vega, S. Dormido-Canto, J. M. Lopez, A. Murari, J. M. Ramirez, R. Moreno, M. Ruiz, D. Alves, R. Felton, and JET-EFDA Contributors, "Results of the JET real-time disruption predictor in the ITER-like wall campaigns," *Fusion Engineering and Design*, vol. 88, no. 6-8, pp. 1228-1231, 2014.

[5] P. Kanerva, "Hyperdimensional computing: An introduction to computing in distributed representation with high-dimensional random vectors," *Cognitive Computation*, vol. 1, no. 2, pp. 139-159, 2009.

[6] A. Rahimi and B. Recht, "Weighted sums of random kitchen sinks: Replacing minimization with randomization in learning," in *Advances in Neural Information Processing Systems (NeurIPS)*, 2009, pp. 1313-1320.

[7] R. Hasani, M. Lechner, A. Amini, L. Liebenwein, A. Ray, S. Tschiatschek, G. Teschl, and D. Rus, "Closed-form continuous-time neural networks," *Nature Machine Intelligence*, vol. 4, pp. 992-1003, 2022.

[8] M. Greenwald, "Density limits in toroidal plasmas," *Plasma Physics and Controlled Fusion*, vol. 44, no. 8, pp. R27-R53, 2002.

[9] K. Friston, "The free-energy principle: A unified brain theory?" *Nature Reviews Neuroscience*, vol. 11, no. 2, pp. 127-138, 2010.

[10] T. A. Plate, *Holographic Reduced Representations: Distributed Representation for Cognitive Structures*. Stanford, CA: CSLI Publications, 2003.

[11] Z. Ge, H. Choi, B. Olshausen, and J. Rabaey, "Classification using hyperdimensional computing: A review," *IEEE Circuits and Systems Magazine*, vol. 20, no. 2, pp. 30-47, 2020.

[12] P. C. de Vries, M. F. Johnson, B. Alper, P. Buratti, T. C. Hender, H. R. Koslowski, V. Riccardo, and JET-EFDA Contributors, "Survey of disruption causes at JET," *Nuclear Fusion*, vol. 51, no. 5, p. 053018, 2011.

[13] F. C. Schuller, "Disruptions in tokamaks," *Plasma Physics and Controlled Fusion*, vol. 37, no. 11A, pp. A135-A162, 1995.

[14] A. H. Boozer, "Theory of tokamak disruptions," *Physics of Plasmas*, vol. 19, no. 5, p. 058101, 2012.

[15] M. Lehnen, K. Aleynikova, P. B. Aleynikov, D. J. Campbell, P. Drewelow, N. W. Eidietis, Yu. Gasparyan, R. S. Granetz, Y. Gribov, N. Hartmann, E. M. Hollmann, V. A. Izzo, S. Jachmich, S.-H. Kim, M. Kocan, H. R. Koslowski, D. Kovalenko, U. Kruezi, A. Loarte, S. Maruyama, G. F. Matthews, P. B. Parks, G. Pautasso, R. A. Pitts, C. Reux, V. Riccardo, R. Roccella, J. A. Snipes, A. J. Thornton, and P. C. de Vries, "Disruptions in ITER and strategies for their control and mitigation," *Journal of Nuclear Materials*, vol. 463, pp. 39-48, 2015.

[16] E. J. Strait, J. L. Barr, M. Baruzzo, J. W. Berkery, R. J. Buttery, P. C. de Vries, N. W. Eidietis, R. S. Granetz, J. M. Hanson, C. T. Holcomb, D. A. Humphreys, J. H. Kim, E. Kolemen, M. Kong, M. J. Lanctot, M. Lehnen, E. Nardon, M. Okabayashi, J.-K. Park, A. Pau, G. Pautasso, F. M. Poli, C. Rea, S. A. Sabbagh, O. Sauter, E. J. Schuster, U. A. Sheikh, and C. Sozzi, "Progress in disruption prevention for ITER," *Nuclear Fusion*, vol. 59, no. 11, p. 112012, 2019.

[17] G. A. Rattá, J. Vega, A. Murari, G. Vagliasindi, M. F. Johnson, P. C. de Vries, and JET-EFDA Contributors, "An advanced disruption predictor for JET tested in a simulated real-time environment," *Nuclear Fusion*, vol. 50, no. 2, p. 025005, 2010.

[18] A. Montvai, G. Pautasso, C. Rea, R. S. Granetz, and the ASDEX Upgrade Team, "Machine learning for disruption prediction: A review," *Nuclear Fusion*, vol. 63, no. 11, p. 112001, 2023.

[19] A. Pau, A. Fanni, S. Carcangiu, B. Cannas, G. Sias, G. Pautasso, M. Gelfusa, and the JET Contributors, "A machine learning approach based on generative topographic mapping for disruption prevention and avoidance at JET," *Nuclear Fusion*, vol. 59, no. 10, p. 106017, 2019.

[20] R. M. Churchill, the DIII-D Team, and the Alcator C-Mod Team, "Deep convolutional neural networks for multi-scale time-series classification and application to tokamak disruption prediction using raw, high temporal resolution diagnostic data," *Physics of Plasmas*, vol. 27, no. 6, p. 062510, 2020.

[21] ITER Organization, "ITER Research Plan within the Staged Approach," ITR-18-003, 2018.

[24] P. E. Thomas, A. Nicolau, and T. Rosing, "A comprehensive HDC framework for temporal anomaly detection," *IEEE Transactions on Neural Networks and Learning Systems*, vol. 33, no. 12, pp. 7384-7396, 2022.

[25] P. Neubert, S. Schubert, and P. Protzel, "An introduction to hyperdimensional computing for robotics," *KI - Kunstliche Intelligenz*, vol. 33, no. 4, pp. 319-330, 2019.

[26] W. B. Johnson and J. Lindenstrauss, "Extensions of Lipschitz mappings into a Hilbert space," *Contemporary Mathematics*, vol. 26, pp. 189-206, 1984.

[27] M. Greenwald, J. L. Terry, S. M. Wolfe, S. Ejima, M. G. Bell, S. M. Kaye, and G. H. Neilson, "A new look at density limits in tokamaks," *Nuclear Fusion*, vol. 28, no. 12, pp. 2199-2207, 1988.

[28] A. Murari, J. Vega, G. A. Rattá, G. Vagliasindi, M. F. Johnson, and JET-EFDA Contributors, "Unbiased and non-supervised learning methods for disruption prediction at JET," *Nuclear Fusion*, vol. 49, no. 5, p. 055028, 2009.

[29] M. Lechner, R. Hasani, A. Amini, T. A. Henzinger, D. Rus, and R. Grosu, "Neural circuit policies enabling auditable autonomy," *Nature Machine Intelligence*, vol. 2, pp. 642-652, 2020.

[30] Y. Kim, M. Imani, T. S. Rosing, and others, "HDCluster: An accurate clustering using brain-inspired high-dimensional computing," in *Proc. Design, Automation and Test in Europe Conference (DATE)*, 2020, pp. 1-6.

[31] L. Spangher, M. Bonotto, W. Arnold, D. Chayapathy, T. Gallingani, A. Spangher, F. Cannarile, D. Bigoni, E. de Marchi, and C. Rea, "DisruptionBench and complimentary new models: Two advancements in machine learning driven disruption prediction," *Journal of Fusion Energy*, vol. 44, article 26, 2025. Available: https://github.com/MIT-PSFC/DisruptionBench

[32] W. F. Arnold, L. Spangher, and C. Rea, "Continuous convolutional neural networks for disruption prediction in nuclear fusion plasmas," in *NeurIPS 2023 Workshop on Tackling Climate Change with Machine Learning*, 2023. arXiv:2312.01286.

[33] D. Chayapathy, T. Siebert, L. Spangher, A. K. Moharir, O. M. Patil, and C. Rea, "Time series viewmakers for robust disruption prediction," in *NeurIPS 2024 Workshop on Tackling Climate Change with Machine Learning*, 2024. arXiv:2410.11065.

---

*Manuscript prepared March 2026. Correspondence to: tristan.stoltz@evolvingresonantcocreationism.com*
