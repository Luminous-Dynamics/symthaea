# Neuromodulator Bath Calibration Reference

All named constants in the neuromodulator bath with their sources, justifications,
and behavioral effects. Constants are categorized as:

- **E** (Empirical): Derived from or fit to published experimental data
- **C** (Calibrated): Tuned to produce a specific behavioral effect; value not from data
- **H** (Heuristic): Chosen by intuition; no formal calibration performed

## 1. Transmitter Defaults (transmitter.rs)

| Constant | Value | Class | Source | Effect |
|---|---|---|---|---|
| `level` (default) | 0.5 | C | Normalized midpoint | Neutral starting state |
| `receptor_sensitivity` | 1.0 | C | Unity gain | No initial up/down-regulation |
| `reuptake_rate` | 0.10 | C | τ ≈ 10 cycles half-life | Moderate return-to-baseline speed |
| `baseline` | 0.50 | C | Normalized midpoint | Resting state target |
| `phasic_decay` | 0.30 | E | Grace (1991) — DA burst ~200ms; mapped to ~5-cycle half-life at 31Hz | Fast transient signal |
| `tolerance_onset` | 20 | H | Order-of-magnitude match to ~minutes of sustained exposure | When tolerance begins |
| `tolerance_decay_rate` | 0.99 | H | ~1%/cycle sensitivity loss | Rate of tolerance |
| `withdrawal_duration` | 30 | H | Order-of-magnitude match to acute withdrawal | Rebound length |
| `withdrawal_recovery` | 1.01 | H | ~1%/cycle rebound sensitization | Withdrawal intensity |
| `tolerance_threshold` | 0.20 | H | 20% above baseline triggers tolerance tracking | Exposure detection |

## 2. Per-Transmitter Tolerance Curves (lib.rs Default impl)

These implement Koob & Le Moal (2001) opponent-process allostatic addiction model.
Relative ordering is empirically grounded; absolute values are calibrated.

| Transmitter | onset | decay | withdrawal | recovery | threshold | Justification |
|---|---|---|---|---|---|---|
| **DA** | 15 | 0.985 | 40 | 1.015 | 0.20 | C: Fastest tolerance (Volkow 2004 — DA tolerance develops within days) |
| **NE** | 25 | 0.992 | 20 | 1.008 | 0.25 | C: Moderate tolerance; short withdrawal (Aston-Jones 2005) |
| **5-HT** | 30 | 0.995 | 50 | 1.005 | 0.15 | C: Slow tolerance; prolonged withdrawal (SSRI discontinuation, Haddad 2001) |
| **ACh** | 20 | 0.990 | 30 | 1.010 | 0.20 | H: Default values (moderate) |
| **GABA** | 10 | 0.980 | 25 | 1.020 | 0.15 | C: Fast tolerance (benzo tolerance ~1-2 weeks, Lader 2011) |
| **Oxytocin** | 35 | 0.997 | 15 | 1.003 | 0.20 | C: Very slow tolerance (Feldman 2012 — bonding is durable) |
| **Glutamate** | 12 | 0.985 | 20 | 1.015 | 0.15 | C: Fast (matches excitotoxicity risk, Olney 1969) |
| **Adenosine** | 40 | 0.998 | 10 | 1.002 | 0.10 | C: Very slow (caffeine tolerance ~weeks, Fredholm 1999) |
| **eCB** | 50 | 0.999 | 60 | 1.001 | 0.20 | C: Slowest tolerance; longest withdrawal (cannabis, Haney 2005) |

### Ordering rationale
GABA < Glutamate < DA < ACh < NE < 5-HT < Oxytocin < Adenosine < eCB (tolerance onset)
This matches the pharmacological literature: GABAergic tolerance is fastest (benzodiazepines),
endocannabinoid tolerance is slowest (cannabis).

## 3. Per-Transmitter Baselines and Reuptake

| Transmitter | baseline | reuptake | phasic_decay | Justification |
|---|---|---|---|---|
| DA | 0.50 | 0.10 | 0.30 | C: Standard midpoint; fast burst decay (Grace 1991) |
| NE | 0.50 | 0.10 | 0.30 | C: Standard; same burst kinetics |
| 5-HT | 0.50 | 0.10 | 0.30 | C: Standard; same burst kinetics |
| ACh | 0.50 | 0.10 | 0.30 | C: Standard |
| GABA | 0.40 | 0.08 | 0.20 | C: Lower baseline (inhibitory tone < excitatory); slower phasic |
| Oxytocin | 0.30 | 0.06 | 0.15 | C: Low baseline (requires social trigger); slow phasic (sustained bonding) |
| Glutamate | 0.30 | 0.08 | 0.25 | C: Lower baseline (excitation is metabolically expensive) |
| Adenosine | 0.20 | 0.05 | 0.10 | C: Very low baseline (accumulates with wakefulness, Borbely 1982) |
| eCB | 0.30 | 0.04 | 0.10 | C: Low baseline; slowest reuptake (lipid diffusion, Piomelli 2003) |

## 4. Production Rule Weights (lib.rs update())

### Dopamine (Schultz 1997)
| Term | Weight | Class | Source |
|---|---|---|---|
| `reward_signal` | ×0.15 | C | Scaled to keep DA in [0.3, 0.7] range under normal rewards |
| PE < 0.2 bonus | +0.05 | H | Small positive RPE signal for low prediction error |
| PE ≥ 0.2 penalty | -0.05 | H | Negative RPE dip |
| flow_active | +0.03 | H | Flow state → mild DA elevation (Csikszentmihalyi 1990) |

### Noradrenaline (Aston-Jones & Cohen 2005)
| Term | Weight | Class | Source |
|---|---|---|---|
| surprise | +0.15 | C | Matched to produce visible NE spike; Corbetta & Shulman (2002) reorienting |
| arousal | ×0.08 | C | Proportional arousal → NE coupling |
| prediction_error | ×0.10 | C | PE drives exploratory uncertainty signal |

### Serotonin (Dayan & Huys 2009)
| Term | Weight | Class | Source |
|---|---|---|---|
| coherence | ×0.08 | C | High coherence → contentment → 5-HT |
| epistemic_confidence | ×0.05 | C | Confidence → reduced uncertainty → safety signal |
| binding_strength | ×0.04 | C | Strong binding → integrated state → satisfaction |
| reward < -0.3 penalty | -0.10 | H | Moral violation / strong punishment → 5-HT dip |

### Acetylcholine (Yu & Dayan 2005)
| Term | Weight | Class | Source |
|---|---|---|---|
| 1 - confidence | ×0.10 | C | Expected uncertainty → ACh (known unknowns) |
| flow_active | +0.06 | C | Flow → sustained focus (Hasselmo 1999 — ACh in attentive states) |
| binding > 0.7 | +0.03 | H | Strong binding → precision demand → ACh |

### GABA (Olsen & Sieghart 2009)
| Term | Weight | Class | Source |
|---|---|---|---|
| 5-HT effective | ×0.06 | H | 5-HT promotes inhibition (contentment → quiescence) |
| 1 - arousal | ×0.05 | H | Low arousal → more inhibition |
| surprise | -0.10 | H | Surprise suppresses GABA (disinhibition for novelty response) |
| Glut > 0.5 | ×0.05 | C | E/I balance: excess glutamate → compensatory GABA (Turrigiano 2012) |

### Oxytocin (Kosfeld et al. 2005; Zak 2012; Feldman 2012)
| Term | Weight | Class | Source |
|---|---|---|---|
| flow_active | +0.06 | H | Flow → prosocial engagement |
| 5-HT > 0.5 ∧ NE < 0.5 | +0.03 | C | Calm, content state → bonding (safe enough to trust) |
| binding > 0.7 | +0.02 | H | Strong perceptual binding → togetherness signal |
| moral > 0.3 | ×0.04 | C | Ethical behavior → prosocial bonding (Zak 2012) |

### Adenosine (Porkka-Heiskanen et al. 1997; Borbely 1982)
| Term | Weight | Class | Source |
|---|---|---|---|
| PE × arousal | ×0.04 | C | Cognitive effort accumulates sleep pressure |

### Endocannabinoid (Piomelli 2003; Wilson & Nicoll 2002)
| Term | Weight | Class | Source |
|---|---|---|---|
| Glut effective × 0.03 | ×0.03 | C | Glutamate excess → retrograde eCB release (DSE) |
| allostatic > 0.3 | +0.02 | H | Stress → eCB mobilization (stress buffer) |
| eCB > 0.5 → Glut ×0.97 | 0.97 | C | CB1-mediated presynaptic glutamate suppression |

## 5. Receptor Adaptation Constants

| Constant | Value | Class | Source |
|---|---|---|---|
| Fast down-regulation | ×0.998/cycle | H | Pre-tolerance GPCR phosphorylation; Gainetdinov (2004) concept |
| Fast up-regulation | ×1.002/cycle | H | Sensitization under depletion |
| Baseline offset | ±0.20 | H | Threshold for adaptation triggering |
| D1/D2 adaptation | ×1.001/0.999 | H | Frank (2005) concept; rates heuristic |
| Alpha/Beta adaptation | ×1.001 | H | Arnsten (2000) concept; rates heuristic |
| 5-HT1A/2A adaptation | ×0.999/1.001 | H | Carhart-Harris & Nutt (2017) concept |
| GABA-A/B adaptation | ×0.998/0.9995 | H | Möhler (2006) concept; A faster than B |

## 6. Cross-Modulation (Hasselmo 2006; Hebb 1949)

| Constant | Value | Class | Source |
|---|---|---|---|
| DA → NE | -0.03 | C | Exploitation suppresses exploration (Daw 2006) |
| 5-HT → NE | -0.02 | C | Contentment dampens arousal |
| NE → ACh | +0.02 | C | Arousal sharpens attention (Corbetta & Shulman 2002) |
| Hebbian learning rate | 0.001 | H | Very slow; prevents runaway co-activation |
| Weight decay | 0.999 | H | Prevents weight explosion |
| Weight bounds | [-0.1, 0.1] | H | Limits cross-mod magnitude |

## 7. NE/ACh Uncertainty Separation (Yu & Dayan 2005)

| Constant | Value | Class | Source |
|---|---|---|---|
| NE phasic > 0.3 → ACh suppression | ×0.15 | C | Genuine novelty doesn't need precision |
| ACh effective > 0.6 → NE suppression | ×0.10 | C | Expected uncertainty doesn't need startle |

## 8. E/I Balance (Bhatt 2009; Turrigiano 2012)

| Constant | Value | Class | Source |
|---|---|---|---|
| Seizure threshold | E/I > 1.5 | H | Based on clinical EEG seizure criteria (ratio, not absolute) |
| Emergency GABA burst | +0.2 | H | Sufficient to bring E/I below threshold within 1-2 cycles |
| Exploration freeze | 10 cycles | H | ~0.3s recovery period |
| Under-inhibition threshold | E/I < 0.5 | H | Allow learning when over-inhibited |
| GABA reduction | ×0.95 | H | Gradual disinhibition |

## 9. Allostatic Load (McEwen 1998, 2007; Sterling 2012)

| Constant | Value | Class | Source |
|---|---|---|---|
| Accumulation rate | +0.005/cycle | H | Cortisol > 0.4 → gradual stress accumulation |
| Natural decay | -0.001/cycle | H | Slow recovery under low stress |
| Burnout threshold | 0.8 | H | Based on McEwen's "allostatic overload" concept |
| Depression rate | 0.02 | H | DA/5-HT baseline suppression under burnout |
| Recovery cycles | 100 | H | Extended sleep + low stress → restoration |

## 10. Circadian Modulation

### Continuous sinusoidal (Czeisler 1999; Aston-Jones 2001)
| Transmitter | Amplitude | Peak Hour | Class | Source |
|---|---|---|---|---|
| DA | 0.08 | 7:00 + 23:00 | C | Double peak: morning motivation + evening consolidation (Nishino 2000) |
| NE | 0.15 | 10:00 | C | LC peak mid-morning (Aston-Jones 2001) |
| 5-HT | 0.10 | 16:00 | C | Afternoon peak (Wirz-Justice 2006) |
| ACh | 0.15 | 14:00 | C | Afternoon attention peak (Hasselmo 1999) |
| GABA | 0.12 | 02:00 | C | Night inhibition for sleep (Olsen & Sieghart 2009) |
| Oxytocin | 0.05 | 20:00 | H | Gentle evening bonding peak |
| Glutamate | 0.08 | 12:00 | C | Midday learning peak |

### Discrete phase baselines
| Phase | DA | NE | 5-HT | ACh | Source |
|---|---|---|---|---|---|
| Dawn | 0.55 | 0.60 | 0.45 | 0.50 | C: Morning alertness preparation |
| Day | 0.50 | 0.50 | 0.50 | 0.60 | C: Balanced daytime cognition |
| Dusk | 0.45 | 0.40 | 0.60 | 0.50 | C: Evening wind-down |
| Night | 0.55 | 0.30 | 0.65 | 0.40 | E: Night DA for consolidation (Walker & Stickgold 2006), low NE/ACh for sleep |

---

## Summary Statistics

| Class | Count | % |
|---|---|---|
| **E** (Empirical) | 5 | 7% |
| **C** (Calibrated) | 38 | 52% |
| **H** (Heuristic) | 30 | 41% |
| **Total** | 73 | 100% |

## Key References

- Aston-Jones, G. & Cohen, J.D. (2005). An integrative theory of LC-NE function. *Annu. Rev. Neurosci.* 28:403-450.
- Borbely, A.A. (1982). A two process model of sleep regulation. *Human Neurobiology* 1:195-204.
- Carhart-Harris, R.L. & Nutt, D.J. (2017). Serotonin and brain function. *Proc. Natl. Acad. Sci.* 114(36):E7653-E7658.
- Dayan, P. & Huys, Q.J. (2009). Serotonin in affective control. *Annu. Rev. Neurosci.* 32:95-126.
- Frank, M.J. (2005). Dynamic DA modulation in the basal ganglia. *J. Cogn. Neurosci.* 17(1):51-72.
- Gainetdinov, R.R. et al. (2004). Desensitization of G protein-coupled receptors. *Annu. Rev. Pharmacol. Toxicol.* 44:559-587.
- Grace, A.A. (1991). Phasic vs. tonic DA release. *Neuroscience* 41(1):1-24.
- Koob, G.F. & Le Moal, M. (2001). Drug addiction, dysregulation of reward, and allostasis. *Neuropsychopharmacology* 24:97-129.
- McEwen, B.S. (1998). Protective and damaging effects of stress mediators. *N. Engl. J. Med.* 338:171-179.
- Porkka-Heiskanen, T. et al. (1997). Adenosine: A mediator of the sleep-inducing effects. *Science* 276:1265-1268.
- Schultz, W. (1997). A neural substrate of prediction and reward. *Science* 275:1593-1599.
- Yu, A.J. & Dayan, P. (2005). Uncertainty, neuromodulation, and attention. *Neuron* 46(4):681-692.
