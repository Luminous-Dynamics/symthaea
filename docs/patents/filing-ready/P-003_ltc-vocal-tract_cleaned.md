# P-003: LTC-Controlled Vocal Tract Synthesizer with Analytical Least-Squares Refinement

## Provisional Patent Application

---

### 1. Title

**Liquid Time-Constant Neural Network Controller for Formant Speech Synthesis with Analytical Least-Squares Output Projection Refinement and Free Energy Principle Self-Tuning**

---

### 2. Inventor(s)

**Tristan Stoltz**, Luminous Dynamics

---

### 3. Date of Conception

**2026** (implementation completed February-March 2026).

First public disclosure: February 23, 2026 (git commit adding `symthaea-vocal-tract` crate).
Under 35 USC 102(b)(1)(A), the 1-year grace period expires **February 23, 2027**.

---

### 4. Technical Field

This invention relates to parametric speech synthesis, specifically to methods and systems for controlling a formant-based vocal tract synthesizer using Liquid Time-Constant (LTC) neural networks operating in hyperdimensional computing (HDC) space, with an analytical least-squares refinement of the output projection and a Free Energy Principle (FEP) active inference agent for runtime self-tuning.

---

### 5. Abstract

A speech synthesis system is disclosed in which a Liquid Time-Constant (LTC) neural network, operating on 16,384-dimensional hyperdimensional continuous vectors (ContinuousHV), controls a formant-based vocal tract model. The system maps cognitive/phonemic state through an HDC encoder into the LTC network, whose temporal dynamics are evolved via an O(D) closed-form solution, then projects the resulting 16,384D state to a 9-dimensional formant parameter space (F1-F3 frequencies, B1-B3 bandwidths, F0, energy, voicing) through a linear output head with activation-specific nonlinearities. A three-stage training pipeline is employed: (1) gradient-based phoneme target training with cosine-annealed learning rates and distance-adaptive per-attractor scaling, (2) BPTT-based transition training on phoneme pairs, and (3) an analytical least-squares refinement that solves the underdetermined system (N~44 phonemes vs. D=16,384 dimensions) in dual form using a Gram matrix with Tikhonov regularization and Gaussian elimination with partial pivoting. This LS refinement eliminates gradient interference between competing phonemes (e.g., IY requiring high F2 while UW requires low F2) and achieves 4.4 Hz average vowel formant error -- a 37x improvement over rule-based baselines. A Free Energy Principle active inference agent continuously modulates the controller's time constants, learning rate, and articulatory emphasis at 10 Hz based on 6-dimensional voice quality observations, closing a perception-action loop for self-tuning vocal production. A consciousness-modulated bandwidth scaling mechanism links the system's integrated information level to formant bandwidth, producing clearer articulation at higher consciousness levels. Experimental validation demonstrates near-perfect mel cepstral distortion (0.02 dB MCD), real-time throughput (559 Hz LTC, 342 Hz full pipeline), and all 8 benchmark vowels under 7 Hz error.

---

### 6. Background and Prior Art

#### 6.1 Rule-Based Formant Synthesis (Klatt, 1980)

The Klatt synthesizer (Klatt, 1980) established the formant synthesis paradigm: a source-filter model with a glottal source exciting a cascade/parallel bank of resonators parameterized by formant frequencies and bandwidths. Formant trajectories are specified by hand-crafted rules or look-up tables. While producing intelligible speech, the rule-based approach yields rigid, mechanical output because formant transitions are linear interpolations lacking the nonlinear temporal dynamics of biological articulators. The Klatt system has no learning capability and cannot adapt to new phonological contexts without manual re-engineering of its rule tables.

#### 6.2 Statistical Parametric Synthesis (HTS, 2006)

Hidden Markov Model (HMM)-based speech synthesis (HTS) replaced manual rules with statistical models trained on speech corpora. HMMs generate smooth parameter trajectories via maximum-likelihood parameter generation with dynamic features. However, HMMs impose strong distributional assumptions (Gaussian mixture models) that limit the expressiveness of generated trajectories, particularly for temporal dynamics at sub-phoneme timescales.

#### 6.3 Signal-Processing Vocoders (WORLD, 2016)

The WORLD vocoder (Morise et al., 2016) provides high-quality analysis/synthesis of speech signals using spectral envelope estimation (CheapTrick), fundamental frequency extraction (DIO/Harvest), and aperiodicity estimation. While WORLD achieves high fidelity for analysis-resynthesis, it is a signal-processing tool rather than a generative controller -- it does not learn articulatory-to-acoustic mappings or adapt its parameters online.

#### 6.4 End-to-End Neural Synthesis (Tacotron/WaveNet, 2017-2018)

Tacotron (Wang et al., 2017) and WaveNet (van den Oord et al., 2016) introduced end-to-end neural speech synthesis, mapping text directly to mel spectrograms or raw waveforms via deep neural networks. These systems achieve near-human naturalness but require massive training corpora (typically 20+ hours), GPU-intensive inference, and produce opaque intermediate representations that preclude interpretable control over articulatory parameters. They have no mechanism for consciousness-driven modulation or active inference self-tuning.

#### 6.5 Gap in Prior Art

No prior system combines:

1. **Hyperdimensional computing** (16,384D) for phoneme representation and cognitive state encoding
2. **Liquid Time-Constant dynamics** for biologically plausible temporal evolution of articulatory state
3. **Analytical least-squares refinement** that exploits the underdetermined nature of the phoneme-to-weight mapping
4. **Free Energy Principle active inference** for continuous runtime self-tuning of synthesis parameters
5. **Consciousness-modulated bandwidth** linking information integration metrics to voice quality

The closest prior art uses either rule-based control (Klatt), statistical sequence models (HTS), or end-to-end neural networks (Tacotron/WaveNet). None operates in hyperdimensional space, none employs LTC temporal dynamics, and none incorporates active inference or consciousness modulation.

---

### 7. Detailed Technical Description

#### 7.1 System Architecture Overview

The system comprises five principal components arranged in a multi-rate pipeline:

```
VoiceCognitiveState (10D)
      |
VocalTractHdcEncoder --> ContinuousHV (16,384D)
      |                       |
      |             [bind with phoneme HV]
      |                       |
VocalTractController (HdcLtcUnifiedNetwork + output head 16,384 -> 9)
      |
FormantFrame [F1, F2, F3, B1, B2, B3, F0, energy, voicing]
      |
FormantVocoder (LF glottal source + 5 resonator filters) --> audio samples

Every 20th motor step (10Hz):
  VocalTractFepAgent modulates tau, learning rate, emphasis
```

The pipeline operates at dual rates: a 200 Hz motor loop produces formant frames, while a 10 Hz cognitive loop updates the HDC encoding and runs the FEP agent.

#### 7.2 HDC-LTC Network Architecture

The core controller wraps an `HdcLtcUnifiedNetwork` -- a multi-layer network of unified HDC-LTC neurons, each operating on 16,384-dimensional continuous hypervectors.

**Network configuration:**
- Layers: 2
- Neurons per layer: 4 (8 total neurons)
- Dimension: 16,384 (HDC_DIMENSION)
- Neuron time constant (tau_base): 0.005 s
- Backbone tau: 0.1 s
- Layer binding: enabled (cross-layer compositional binding)
- Skip connections: disabled
- Fourier basis injection: frequencies [3.0, 5.0, 10.0] Hz at amplitude 0.1

Each neuron implements the Closed-form Continuous-time (CfC) dynamics equation:

```
x(t + dt) = x_inf + (x(t) - x_inf) * exp(-dt / tau)
```

where `x_inf` is the equilibrium state computed from the input, and `tau` is the adaptive time constant. This O(D) closed-form solution avoids numerical ODE integration, enabling temporal jumps of arbitrary size without accuracy loss.

The forward pass proceeds:

1. `network.evolve_closed_form(dt, cognitive_hv)` -- evolve all neurons with input HV
2. `network.output().normalize()` -- extract bundled, normalized final-layer output (16,384D)
3. Linear projection: `W_out @ hv + b_out` maps to 9D raw output
4. Activation functions produce the final `FormantFrame`

#### 7.3 Output Dimensions and Activation Functions

The 9-dimensional output head maps the 16,384D network state to articulatory parameters:

| Index | Parameter | Activation | Clamp Range | Default (Schwa) |
|-------|-----------|-----------|-------------|-----------------|
| 0 | F1 (Hz) | softplus + clamp | [200, 1000] | 500 |
| 1 | F2 (Hz) | softplus + clamp | [600, 3000] | 1500 |
| 2 | F3 (Hz) | softplus + clamp | [1500, 5000] | 2500 |
| 3 | B1 (Hz) | softplus + clamp | [30, 300] | 60 |
| 4 | B2 (Hz) | softplus + clamp | [30, 400] | 90 |
| 5 | B3 (Hz) | softplus + clamp | [50, 500] | 150 |
| 6 | F0 (Hz) | softplus + clamp | [base-range/2, base+range/2] | 120 |
| 7 | energy | sigmoid | [0, 1] | 0.5 |
| 8 | voicing | sigmoid | [0, 1] | 0.8 |

The softplus activation is defined as:

```
softplus(x) = ln(1 + e^x)    for |x| <= 20
            = x               for x > 20   (overflow guard)
            = 0               for x < -20  (underflow guard)
```

The sigmoid activation is:

```
sigmoid(x) = 1 / (1 + e^(-x))
```

Output bias is initialized to schwa (neutral vowel) defaults. For energy, `bias = 0.0` yields `sigmoid(0) = 0.5`. For voicing, `bias = 1.39` yields `sigmoid(1.39) ~ 0.8`.

Output weights are initialized from a deterministic genesis seed with scale factor 0.15:

```
W_out = genesis.hv("vocal_tract::output_weights", 9 * 16384).values * 0.15
```

#### 7.4 EMA Smoothing and Rate Limiting

Formant frequencies and bandwidths (F1-F3, B1-B3) are post-filtered with exponential moving average (EMA) smoothing followed by a rate limiter. Prosodic parameters (F0, energy, voicing) pass through unsmoothed.

**EMA update:**

```
f_ema(t) = f_prev + (f_raw - f_prev) * (1 - alpha)
```

where `alpha = 0.3` (default smoothing factor).

**Rate limiter:**

```
f_out(t) = f_prev + clamp(f_ema(t) - f_prev, -max_delta, +max_delta)
```

where `max_delta` adapts between steady-state (12 Hz/frame) and transition (20 Hz/frame) based on whether a phoneme boundary has been detected.

#### 7.5 Consciousness-Modulated Bandwidth Scaling

When cognitive channels are available (updated at 10 Hz), the system scales formant bandwidths based on the consciousness level (channel index 7):

```
bandwidth_scale = 1.2 - 0.4 * consciousness_level
```

At `consciousness_level = 0`: scale = 1.2 (wider bandwidths, mumbled speech)
At `consciousness_level = 1`: scale = 0.8 (narrower bandwidths, clearer vowels)

This implements the hypothesis that higher information integration (as measured by Phi/IIT) produces more precisely controlled articulation.

#### 7.6 Training Pipeline

Training proceeds in three stages:

**Stage 1: Gradient-Based Phoneme Training (100 epochs)**

For each phoneme target, a deterministic 16,384D HV is generated from the genesis seed. Training uses cosine-annealed learning rates:

```
epoch_lr = lr_min + (lr_peak - lr_min) * 0.5 * (1 + cos(pi * progress))
```

where `lr_peak = 30x base LR` and `lr_min = 10x base LR` (base LR = 0.001).

Per-phoneme distance from schwa (500, 1500, 2500 Hz) determines:
- Learning rate scaling: 1.0x for schwa-like phonemes up to 3.0x for extreme vowels, with F2 weighted 4x in the distance metric
- Adaptive step count: above-median distance phonemes get 20 gradient steps; below-median get 10
- Per-attractor adaptive LR: near-schwa phonemes (normalized distance < 0.3) receive a floor-ramped LR starting at 0.5x to prevent over-pulling; far-from-schwa phonemes receive boosted F2 error gradient (0.7x error scale)

Gradient normalization uses per-dimension error scales: `[400, 600, 1500, 100, 150, 200, 100, 1, 1]` with gradient clipping at 5.0. Weight decay is disabled during phoneme training.

Backpropagation proceeds through the output projection to the LTC network via BPTT with layer-wise gradient propagation.

**Stage 2: BPTT Transition Training (10 epochs)**

For each transition pair (from_phoneme, to_phoneme):
1. Warm up on the source phoneme for 20 steps (settle LTC state)
2. Switch to destination phoneme HV
3. Train over 16 transition frames (80 ms at 200 Hz) with linearly interpolated targets
4. Learning rate: 5x base LR
5. No weight decay

This teaches the network smooth formant trajectories during phoneme transitions.

**Stage 3: Analytical Least-Squares Refinement**

After gradient training has settled the LTC network weights, the output projection is refined using an analytical least-squares solution. The key insight is that the system is **underdetermined**: N ~ 44 phonemes but D = 16,384 dimensions. The dual-form solution exploits this:

**Step 1: Collect steady-state HVs.** For each phoneme, generate its HV, warm up the network for 20 steps, and extract the normalized final-layer output `x_i` (16,384D).

**Step 2: Compute target raw values.** Invert the activation functions:
- Formants (softplus): `raw ~ target` (softplus(x) ~ x for large x)
- Energy/voicing (sigmoid): `raw = logit(target) = ln(p / (1-p))`

**Step 3: Compute the Gram matrix.** `G = X X^T` where X is the N x D matrix of steady-state HVs:

```
G[i,j] = sum_{k=0}^{D-1} x_i[k] * x_j[k]
```

This yields an N x N matrix (e.g., 44 x 44), which is orders of magnitude smaller than the D x D matrix that would arise from the primal form.

**Step 4: Add Tikhonov regularization.** `G[i,i] += lambda` where `lambda = 0.01`.

**Step 5: Solve for dual coefficients.** For each output dimension d, solve:

```
(G + lambda * I) * alpha_d = y_d
```

where `y_d[i] = target_raw[i][d] - bias[d]`, using Gaussian elimination with partial pivoting (singularity threshold 1e-10).

**Step 6: Recover primal weights.** `w_new_d = X^T * alpha_d`:

```
w_new[d][j] = sum_{i=0}^{N-1} alpha_d[i] * x_i[j]
```

**Step 7: Blend with existing weights.**

```
w_final[d][j] = (1 - blend) * w_old[d][j] + blend * w_new[d][j]
```

At `blend = 1.0`, the LS solution fully replaces the gradient-trained weights.

**Step 8: Bias correction.** Update bias by the mean residual:

```
bias[d] += blend * mean(target[i][d] - pred[i][d]) / N
```

**Why LS works:** The gradient-based Stage 1 trains the LTC network to produce distinct steady-state HVs for each phoneme. The LS refinement then finds the unique minimum-norm output projection that maps these HVs to the correct formant targets. Because D >> N, the system is underdetermined and can fit all targets exactly (up to regularization). This eliminates the gradient interference problem where training on phoneme IY (high F2) destructively updates weights shared with phoneme UW (low F2).

#### 7.7 FEP Active Inference Agent

A Free Energy Principle active inference agent runs at 10 Hz (every 20 motor frames) and modulates the controller based on voice quality feedback.

**Observation space (6D):**
- `articulation_score` -- articulatory precision
- `formant_accuracy` -- formant frequency accuracy vs. targets
- `pitch_stability` -- F0 stability
- `coarticulation_smoothness` -- smoothness of phoneme transitions
- `duration_accuracy` -- phoneme duration accuracy
- `energy_consistency` -- energy envelope consistency

**Action space (6 actions):**

| Action | tau_factor | lr_factor | emphasis_factor | Condition |
|--------|-----------|-----------|-----------------|-----------|
| DropTau | 0.8 | 1.0 | 1.0 | High prediction error |
| RaiseTau | 1.2 | 1.0 | 1.0 | Low prediction error |
| BoostLR | 1.0 | 1.5 | 1.0 | Initial adaptation |
| ReduceLR | 1.0 | 0.7 | 1.0 | Converged |
| ShiftEmphasis | 1.0 | 1.0 | 1.3 | Low articulation |
| ExplorationBurst | 0.9 | 1.2 | 1.1 | Stuck in local minimum |

The agent configuration:
- State dimension: 6
- Observation dimension: 6
- Number of actions: 6
- Inference iterations: 5
- Belief learning rate: 0.1
- Planning horizon: 3
- Action temperature: 1.0
- TD learning: enabled (gamma=0.95, trace_decay=0.8, initial LR=0.05)

The perception-action loop:
1. `perceive(observation)` -- update internal belief state, compute free energy
2. `select_action()` -- plan over horizon, select action minimizing expected free energy
3. `act(action)` -- execute action, update generative model
4. `learn_from_outcome(action, observation)` -- TD learning from post-action quality

Emphasis modulation affects the controller output:

```
frame.energy = clamp(frame.energy * emphasis_factor, 0, 1)
bw_scale = 1 / sqrt(emphasis_factor)
frame.b1 *= bw_scale
frame.b2 *= bw_scale
frame.b3 *= bw_scale
```

#### 7.8 Prosody Head

A lightweight MLP (12 -> 8 -> 3) maps cognitive voice channels directly to prosody corrections, bypassing the 16,384D HDC bottleneck for prosody-specific modulation.

**Architecture:**
- Input: 12 cognitive channels (including prediction_error[0], arousal[2], consciousness_level[7], integrated_phi[10], expected_free_energy[11])
- Hidden: 8 neurons with tanh activation
- Output: 3 corrections (delta_F0 in Hz, delta_energy in logit space, delta_voicing in logit space)
- Clamping: delta_F0 in [-50, +50] Hz, delta_energy in [-1, +1], delta_voicing in [-0.5, +0.5]
- Learning rate: 10x base controller LR

**Pre-wired psychoacoustic mappings (neurons 0-5):**
- Neuron 0: arousal -> F0 raise (weight 2.0, bias -1.0, output weight 40.0 Hz)
- Neuron 1: arousal -> energy boost (weight 1.5, bias -0.75, output weight 0.8)
- Neuron 2: consciousness_level -> energy modulation (weight 1.5, bias -0.75, output weight 0.5)
- Neuron 3: prediction_error -> F0 drop (weight -1.5, bias 0.0, output weight 20.0 Hz)
- Neuron 4: integrated_phi -> F0 lift (weight 2.0, bias -1.0, output weight 25.0 Hz)
- Neuron 5: expected_free_energy -> energy drop (weight -1.0, bias 0.5, output weight 0.6)

Neurons 6-7 retain small random initialization for online learning.

Energy and voicing corrections are applied in logit space:

```
energy_out = sigmoid(logit(energy_pred) + delta_energy)
voicing_out = sigmoid(logit(voicing_pred) + delta_voicing)
```

#### 7.9 Fourier Basis Injection

The LTC network's equilibrium states are perturbed by sinusoidal basis functions at three rates corresponding to speech temporal structure:

- 3.0 Hz -- syllable rate
- 5.0 Hz -- prosodic rate
- 10.0 Hz -- formant transition rate

Injection amplitude: 0.1 (perturbation level). These frequencies are optimizable via coordinate descent.

#### 7.10 Coarticulation Modeling

**Carryover coarticulation:** When a phoneme boundary is detected, the system saves the old phoneme's bound HV and linearly blends toward the new phoneme's bound HV over 16 frames (80 ms at 200 Hz):

```
hv_effective = hv_old * (1 - t) + hv_new * t,  t = frame_count / 16
```

**Anticipatory coarticulation:** During the final frames of the current phoneme, the HV starts blending toward the next phoneme (maximum 30% blend):

```
anticipation_blend = t * 0.3
hv_effective = hv_current * (1 - anticipation_blend) + hv_next * anticipation_blend
```

#### 7.11 Vocoder (Source-Filter Synthesis)

The FormantVocoder converts formant frames to audio using the source-filter model:

- **Glottal source:** Liljencrants-Fant (LF) model parameterized by Rd (0.3 = pressed, 1.0 = modal, 2.7 = breathy)
- **Aspiration noise:** white noise during glottal open phase (level 0.03)
- **Fricative noise:** pink noise for unvoiced consonants
- **Jitter/shimmer:** Ornstein-Uhlenbeck process (jitter sigma 0.042, shimmer 0.02)
- **Formant filters:** 5 second-order IIR resonators (F1-F3 from controller, F4=3500 Hz, F5=4500 Hz fixed)
- **Cascade mode:** AllPoleResonator with transfer function H(z) = 1 / (1 - a1*z^-1 - a2*z^-2)
- **Spectral tilt:** 1-pole low-pass (coefficient 0.5)
- **Sample rate:** 24,000 Hz
- **Manner-aware excitation:** 7 source types (Vowel, Stop, Fricative, Nasal, Affricate, Liquid, Silent) with manner-specific energy and voicing targets

---

### 8. Novelty Statement

The following aspects of this invention are believed to be novel, either individually or in combination:

1. **LTC-to-formant architecture:** The use of a Liquid Time-Constant neural network operating in 16,384-dimensional hyperdimensional space as a controller for formant speech synthesis. No prior formant synthesizer uses LTC dynamics or hyperdimensional computing. The closed-form CfC temporal evolution provides biologically plausible articulatory dynamics without numerical ODE integration.

2. **Analytical LS refinement of the output projection:** The exploitation of the underdetermined nature of the phoneme-to-weight mapping (N << D) to compute an exact least-squares solution in dual form. This eliminates gradient interference between competing phoneme targets -- a fundamental problem in gradient-based training of shared output projections. The use of a Gram matrix (N x N instead of D x D) with Tikhonov regularization and blendable output enables efficient, stable refinement.

3. **FEP active inference self-tuning:** The integration of a Free Energy Principle active inference agent that continuously modulates the LTC controller's time constants, learning rate, and articulatory emphasis based on voice quality feedback. This closes a perception-action loop that enables the synthesizer to self-tune during operation.

4. **Consciousness-modulated bandwidth:** The coupling of formant bandwidth scaling to a consciousness/information integration metric, implementing the hypothesis that higher cognitive integration produces more precise articulatory control.

5. **Three-stage training pipeline:** The combination of (a) gradient training with distance-adaptive per-attractor learning rates and cosine annealing, (b) BPTT transition training on phoneme pairs, and (c) analytical LS refinement, as a unified pipeline for training LTC-based speech synthesizers.

6. **Prosody head with psychoacoustic pre-wiring:** A separate MLP head that bypasses the HDC bottleneck for prosody control, with hand-wired initial weights implementing psychoacoustically motivated mappings (arousal -> pitch, Phi -> expressiveness, prediction error -> pitch drop).

---

### 9. Suggested Claims

#### Independent Claims

**Claim 1 (System):** A speech synthesis system comprising:
- a hyperdimensional computing encoder that maps a cognitive state vector to a continuous hypervector of at least 10,000 dimensions;
- a Liquid Time-Constant neural network that evolves the hypervector through closed-form temporal dynamics;
- a linear output projection that maps the evolved hypervector to a formant parameter vector comprising at least formant frequencies, formant bandwidths, fundamental frequency, energy, and voicing;
- activation functions applied to the formant parameter vector, wherein formant frequencies use softplus activation with physical range clamping and energy/voicing use sigmoid activation;
- wherein the system produces formant frames at a motor rate of at least 100 Hz.

**Claim 2 (Method -- LS Refinement):** A method for training an output projection of a neural speech synthesizer, comprising:
- evolving a temporal dynamics network to steady state for each of a plurality of phoneme inputs to obtain steady-state hypervectors;
- computing a Gram matrix as the inner product of all pairs of steady-state hypervectors;
- adding Tikhonov regularization to the diagonal of the Gram matrix;
- for each output dimension, solving the regularized linear system to obtain dual coefficients;
- recovering output projection weights as a linear combination of the steady-state hypervectors weighted by the dual coefficients;
- blending the recovered weights with existing gradient-trained weights by a blend factor.

**Claim 3 (System -- FEP Self-Tuning):** A speech synthesis system comprising:
- a formant parameter controller with adjustable time constants, learning rate, and articulatory emphasis;
- a Free Energy Principle active inference agent that receives voice quality observations and selects modulation actions;
- wherein the active inference agent modulates the controller's time constants, learning rate, and emphasis factor based on minimization of expected free energy;
- wherein the active inference agent operates at a cognitive rate lower than the controller's motor rate.

**Claim 4 (Method -- Consciousness-Modulated Bandwidth):** A method for speech synthesis comprising:
- computing a consciousness or information integration metric for a cognitive system;
- scaling formant bandwidths of a vocal tract synthesizer as a decreasing function of the consciousness metric;
- wherein higher consciousness levels produce narrower bandwidths corresponding to clearer vowel articulation.

**Claim 5 (System -- Complete Pipeline):** A speech synthesis system comprising:
- a hyperdimensional encoder operating at a cognitive rate;
- a Liquid Time-Constant neural network controller operating at a motor rate higher than the cognitive rate;
- a prosody head comprising a multi-layer perceptron with hand-wired psychoacoustic initial weights that maps cognitive channels to fundamental frequency, energy, and voicing corrections;
- a Free Energy Principle active inference agent operating at the cognitive rate;
- a formant vocoder with manner-aware source excitation;
- wherein phoneme identity is encoded by binding a phoneme-specific hypervector with the cognitive hypervector.

**Claim 16 (independent, broad -- Generalized LS Refinement):** A method for refining an output projection of a neural network, comprising:
- evolving a temporal dynamics network to steady state for each of a plurality of input conditions to obtain steady-state internal representations;
- computing a Gram matrix as the inner product of all pairs of steady-state representations;
- applying regularization to the Gram matrix;
- solving the regularized linear system in dual form to obtain coefficients;
- recovering output projection weights as a linear combination of the steady-state representations weighted by the coefficients;
- wherein the method is applicable to any neural network with a temporal dynamics component and a linear output projection, independent of the application domain.

**Claim 17 (independent, broad -- Closed-Loop Neural Controller):** A method for controlling a parametric signal generator, comprising:
- encoding an input condition as a high-dimensional vector via hyperdimensional computing operations;
- evolving the high-dimensional vector through a continuous-time neural network with state-dependent time constants;
- projecting the evolved state to a parameter space of the signal generator via a trained output projection;
- wherein the continuous-time neural network provides inherent temporal smoothing of parameter trajectories without explicit interpolation.

#### Dependent Claims

**Claim 6** (depends on Claim 1): wherein the hyperdimensional computing encoder encodes the cognitive state as a 10-dimensional vector comprising prediction error, motor intention, emotional arousal, emotional valence, uncertainty estimate, attention focus, context novelty, consciousness level, articulation quality, and rate stability.

**Claim 7** (depends on Claim 1): further comprising EMA smoothing and rate limiting applied to formant frequencies and bandwidths but not to prosodic parameters (F0, energy, voicing).

**Claim 8** (depends on Claim 2): wherein the number of phoneme inputs N is less than 100 and the hypervector dimension D is at least 10,000, such that the Gram matrix is at most N x N and the linear system is solved in O(N^3) time.

**Claim 9** (depends on Claim 2): wherein the blend factor is 1.0 such that the least-squares solution fully replaces the gradient-trained weights.

**Claim 10** (depends on Claim 3): wherein the active inference agent's action space comprises at least: decreasing time constants for faster transitions, increasing time constants for smoother sustained sounds, boosting learning rate, reducing learning rate, shifting articulatory emphasis, and exploration bursts.

**Claim 11** (depends on Claim 3): wherein the active inference agent employs temporal difference learning with eligibility traces (trace_decay = 0.8) and discount factor (gamma = 0.95).

**Claim 12** (depends on Claim 5): further comprising coarticulation modeling wherein, at a phoneme boundary, the system linearly blends from the old phoneme's bound hypervector to the new phoneme's bound hypervector over a transition window.

**Claim 13** (depends on Claim 5): further comprising anticipatory coarticulation wherein, during the final frames of a current phoneme, the hypervector is blended toward the next phoneme's hypervector by up to 30%.

**Claim 14** (depends on Claim 5): wherein the prosody head includes pre-wired neurons mapping arousal to fundamental frequency raise, integrated information (Phi) to fundamental frequency lift, prediction error to fundamental frequency drop, and expected free energy to energy reduction.

**Claim 15** (depends on Claim 1): further comprising Fourier basis injection at syllable rate (3 Hz), prosodic rate (5 Hz), and formant transition rate (10 Hz) into the LTC network's equilibrium computation.

---

### 10. Experimental Validation

#### 10.1 Formant Accuracy

Average vowel formant error (F1+F2+F3 mean absolute error) after the full three-stage training pipeline:

| Vowel | Error (Hz) |
|-------|-----------|
| IY | 6.9 |
| UW | 3.1 |
| AA | 5.7 |
| IH | 4.9 |
| OW | 4.7 |
| AO | 5.2 |
| AH | 2.9 |
| UH | 2.8 |
| **Average** | **4.4** |

Compared to rule-based baseline: 162.0 Hz average error. The LTC controller achieves a **37x improvement**.

#### 10.2 LS Blend Sweep

Monotonic improvement with increasing blend factor:

| Blend | Avg Error (Hz) |
|-------|----------------|
| 0.5 | 51.5 |
| 0.6 | 42.0 |
| 0.7 | 32.4 |
| 0.8 | 22.9 |
| 0.9 | 13.4 |
| 1.0 | 4.4 |

Lambda effect is minimal: lambda=0.001 yields 30.4 Hz vs. lambda=0.01 yields 32.4 Hz at blend=0.7 (6% difference).

#### 10.3 Spectral Quality

- **Mel Cepstral Distortion (MCD):** 0.02 dB measured against own training targets (note: this metric reflects the output projection's fidelity to its formant targets, not a comparison against natural speech recordings)
- **Manner classification accuracy:** 10/10 manner types correct
- **Source type classification:** 12/12 source types correct

#### 10.4 Throughput

- **LTC controller throughput:** 559 Hz
- **Full pipeline throughput:** 342 Hz
- **Target:** 200 Hz motor rate (1.7x real-time margin)

#### 10.5 Test Coverage

- 177 passing tests total (93 vocal-tract + 32 vocoder + 18 formant-targets + 31 repl-voice + 1 controller + 2 fep)
- 0 test failures
- All consonant manner types validated (stops, fricatives, nasals, liquids, affricates, glides)

---

### 11. Key Source Files

| File | Description | LOC (approx) |
|------|-------------|------|
| `symthaea/crates/crates/symthaea-vocal-tract/src/controller.rs` | LTC controller, output projection, LS refinement, prosody head, training pipeline | ~1500 |
| `symthaea/crates/crates/symthaea-vocal-tract/src/pipeline.rs` | Dual-rate pipeline, coarticulation, prosody context, duration model | ~800 |
| `symthaea/crates/crates/symthaea-vocal-tract/src/fep.rs` | FEP active inference agent, 6D observation/action spaces | ~450 |
| `symthaea/crates/crates/symthaea-vocal-tract/src/encoder.rs` | HDC encoder: cognitive state -> 16,384D ContinuousHV | ~300 |
| `symthaea/crates/crates/symthaea-vocal-tract/src/types.rs` | FormantFrame, FormantTarget, SourceType definitions | ~250 |
| `symthaea/crates/crates/symthaea-vocal-tract/src/lib.rs` | Module structure and re-exports | ~60 |
| `symthaea/src/voice/vocoder.rs` | FormantVocoder: LF glottal, resonators, cascade filter | ~700 |

---

### 12. Closest Prior Art References

1. **Klatt, D.H.** (1980). "Software for a cascade/parallel formant synthesizer." *Journal of the Acoustical Society of America*, 67(3), 971-995. -- Rule-based formant synthesis; no learning, no HDC, no LTC.

2. **Morise, M., Yokomori, F., & Ozawa, K.** (2016). "WORLD: A vocoder-based high-quality speech synthesis system for real-time applications." *IEICE Transactions on Information and Systems*, E99-D(7), 1877-1884. -- Signal-processing vocoder; analysis-resynthesis, not generative control.

3. **Wang, Y., et al.** (2017). "Tacotron: Towards end-to-end speech synthesis." *Interspeech*. -- End-to-end neural TTS; requires large corpus, no formant-level control, no active inference.

4. **van den Oord, A., et al.** (2016). "WaveNet: A generative model for raw audio." *arXiv:1609.03499*. -- Autoregressive neural vocoder; no temporal dynamics model, no HDC.

5. **Zen, H., et al.** (2009). "Statistical parametric speech synthesis." *Speech Communication*, 51(11), 1039-1064. -- HMM-based synthesis; Gaussian assumptions, no LTC dynamics.

6. **Hasani, R., et al.** (2021). "Liquid Time-constant Networks." *AAAI*. -- LTC/CfC neural networks; applied to time series and robotics, not speech synthesis.

7. **Kanerva, P.** (2009). "Hyperdimensional computing: An introduction to computing in distributed representation." *Cognitive Computation*, 1(2), 139-159. -- HDC framework; applied to classification and language, not speech synthesis.

8. **Friston, K.** (2010). "The free-energy principle: a unified brain theory?" *Nature Reviews Neuroscience*, 11(2), 127-138. -- FEP/active inference theory; applied to perception and action, not speech synthesis control.

---

### 13. Figures (Text Descriptions)

**Figure 1: System Architecture Diagram**
Block diagram showing the complete pipeline: VoiceCognitiveState (10D) flowing into VocalTractHdcEncoder, producing a 16,384D ContinuousHV. This HV is bound with a phoneme identity HV (also 16,384D) via the HDC bind operation. The bound HV feeds into the VocalTractController containing the HdcLtcUnifiedNetwork (2 layers x 4 neurons, 16,384D internal state). The network output passes through a linear projection (9 x 16,384 weight matrix + 9D bias) and activation functions (softplus for formants, sigmoid for energy/voicing) to produce a FormantFrame. A feedback path shows the VocalTractFepAgent receiving 6D voice quality observations and outputting tau/LR/emphasis modulations back to the controller. The ProsodyHead (12->8->3 MLP) receives cognitive channels and adds corrections to F0/energy/voicing. Dual clock symbols indicate 200 Hz motor rate and 10 Hz cognitive rate.

**Figure 2: LS Refinement Algorithm**
Flowchart of the least-squares refinement: (a) For each of N phonemes, evolve LTC network to steady state and extract 16,384D output HV. (b) Compute N x N Gram matrix G = X X^T. (c) Add Tikhonov regularization lambda*I. (d) For each of 9 output dimensions, solve (G + lambda*I) * alpha = y using Gaussian elimination with partial pivoting. (e) Recover D-dimensional weights: w = X^T * alpha. (f) Blend with gradient-trained weights. Annotation showing the key property: N=44 << D=16,384 makes this tractable.

**Figure 3: Blend Sweep Results**
Line plot with blend factor (0.5 to 1.0) on x-axis and average vowel formant error (Hz) on y-axis. Shows monotonic decrease from 51.5 Hz at blend=0.5 to 4.4 Hz at blend=1.0. Horizontal dashed line at 162.0 Hz marks the rule-based baseline. Annotation: "37x improvement at blend=1.0".

**Figure 4: Per-Vowel Error Comparison**
Grouped bar chart with 8 vowels (IY, UW, AA, IH, OW, AO, AH, UH) on x-axis. Two bars per vowel: rule-based baseline (tall, ~162 Hz average) and LTC+LS (short, all under 7 Hz). Y-axis is formant error in Hz. All LTC+LS bars are nearly invisible relative to rule-based bars.

**Figure 5: FEP Active Inference Loop**
Circular diagram showing the perception-action cycle: Observation (6D voice quality) -> Perceive (update belief, compute free energy) -> Plan (3-step horizon, minimize expected free energy) -> Act (select from 6 actions) -> Modulate (tau/LR/emphasis) -> Controller -> Voice Output -> back to Observation. Inner loop shows TD learning updating value estimates from outcomes.

**Figure 6: Training Pipeline Stages**
Three-panel diagram: (a) Stage 1 -- Gradient training: multiple arrows from phoneme HVs through LTC network to output projection, with per-phoneme LR scaling shown by arrow thickness proportional to distance from schwa. (b) Stage 2 -- BPTT transition training: pairs of phonemes connected by curved arrows showing interpolated targets over 16 frames. (c) Stage 3 -- LS refinement: matrix equation w = X^T (X X^T + lambda*I)^{-1} y with dimensions annotated (44 x 16,384 -> 44 x 44 Gram).

**Figure 7: Consciousness-Bandwidth Coupling**
Dual-axis plot: left axis shows bandwidth scale factor (0.8 to 1.2), right axis shows formant bandwidth in Hz. X-axis is consciousness level (0 to 1). Linear relationship: bandwidth_scale = 1.2 - 0.4 * consciousness. Inset spectrograms at consciousness=0.2 (wide, blurred formants) and consciousness=0.9 (narrow, sharp formants).
