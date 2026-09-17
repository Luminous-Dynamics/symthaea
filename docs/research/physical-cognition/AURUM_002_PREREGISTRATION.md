# AURUM-002 Preregistration — Gold Nanojunction Reservoir Characterization

Status: **PRE-EXECUTION / NO RESULT**  
Issue: #3627

## Frozen dependency set

```text
PHYS-001  5d90397cf04fafd5acf1835f0b16bb5941dc4f2b
PHYS-002  9ae4695ffaba9a94f9ad69a783724659036aa927
PHYS-003  eff60b7f418b632e7602d19b7d7006b6b24bb2f0
PHYS-004  3126a0bb63672b0647d783397cfc6b15164642b8
PHYS-005  5db7f62ef711a7fb9798e2c3d0e3e886c77b913a
PHYS-006  a4ccf536ab5df4f2c59775d355b25c3ee9805f42
WLOAD-001 65cd00d78e698b5e9189efdd5c3c91341b0fcfa4
WLOAD-002 65be4f5500df47287adf96264e30c7f49afd05c4
WLOAD-003 b44d0119c6578b3723d776377bb13cdaab8d446b
AURUM-001 86617dd7aa564846701ddff3d1adc948e0eec4c0
```

Every exact head above must independently qualify before a result-producing AURUM-002 runner is authorized. Branch existence, queued/pending CI, mergeability, merge-ref execution, or a result from any different head is not qualification evidence.

## 1. Purpose and claim boundary

AURUM-002 asks whether the **specific simulated dynamics** of the AURUM-001 gold-nanojunction-inspired model provide useful temporal/nonlinear computation beyond named substrate-neutral controls under a frozen evaluation protocol.

This is simulation characterization only. It cannot establish fabricated-gold energy efficiency, hardware latency, calibration to a particular Au film, a general gold computational advantage, consciousness relevance, or suitability for active Symthaea cognition.

## 2. Exact AURUM subject

Backend identity:

```text
family         = physical:gold-nanojunction
version        = v1
implementation = aurum-topological-network-sim
```

Frozen model configuration:

```text
junctions             = 128
topology               = SmallWorld { radius: 2, shortcuts_per_node: 1 }
baseline_conductance   = 0.15
max_conductance        = 1.00
switching_threshold    = 0.35
switching_gain         = 0.18
relaxation             = 0.025
threshold_disorder     = 0.08
stochasticity          = 0.01
input_coupling         = 1.00
recurrent_coupling     = 0.25
```

Every subject run must bind the exact implementation bytes and the exact behavior-affecting configuration through PHYS-002.

### 2.1 Configuration identity

`configuration_digest` MUST equal the locally derived value from `AurumConfig::configuration_digest()` for the runtime subject.

Canonical config V1 is domain-separated as:

```text
symthaea:aurum:config:v1\0
```

It binds junction count, topology tag and topology parameters, and exact little-endian IEEE-754 `f64::to_bits()` representations for every behavior-affecting continuous parameter. The per-run seed is excluded because PHYS-002 binds the seed plan separately.

### 2.2 Implementation identity through PHYS-005

A human-readable backend version is insufficient. The runner must use exact PHYS-005 head `5db7f62ef711a7fb9798e2c3d0e3e886c77b913a` to construct an `ImplementationCapsule`, derive the local PHYS-002 `BackendBinding`, and call `verify_exact_binding()` against the manifest before warmup.

AURUM implementation capsule V1 contains exactly:

```text
crates/domains/symthaea-aurum/Cargo.toml
crates/domains/symthaea-aurum/src/config_binding.rs
crates/domains/symthaea-aurum/src/lib.rs
crates/domains/symthaea-aurum/src/model.rs
```

PHYS-005 sorts paths bytewise and uses:

```text
"symthaea:physical:implementation-capsule:v1\0"
for each sorted file:
  u32_le(path_byte_length)
  path_utf8_bytes
  u64_le(content_byte_length)
  raw_file_bytes
```

`implementation_digest = BLAKE3(canonical_capsule_bytes)`.

A backend-identity, implementation-digest, or configuration-digest mismatch is a pre-execution refusal, not a warning. Git commit identity remains provenance but does not replace exact source/configuration binding.

## 3. Frozen controls

All controls come from exact PHYS-003 head `eff60b7f418b632e7602d19b7d7006b6b24bb2f0` and receive the same seed and exact PHYS-006 `drive_voltage` sequence as the paired AURUM arm.

Each comparator gets its own exact PHYS-002 `BackendBinding`, derived and verified through PHYS-005.

### C0 — memoryless nonlinear negative control

```text
units              = 128
input_gain         = 1.0
gain_spread        = 0.25
activity_threshold = 0.25
persistent_state   = 0
recurrent_edges    = 0
trainable_backend_parameters = 0
```

Purpose: determine whether instantaneous nonlinear features solve the task without memory.

### C1 — fading hysteretic ensemble

```text
units               = 128
baseline            = 0.0
switching_threshold = 0.35
threshold_spread    = 0.08
switching_gain      = 0.18
relaxation          = 0.025
activity_threshold  = 0.10
persistent_state    = 128
recurrent_edges     = 0
trainable_backend_parameters = 0
```

Purpose: determine whether threshold hysteresis plus fading memory explains an apparent AURUM effect.

### C2 — generic recurrent graph

```text
units                = 128
shortcuts_per_unit   = 2
input_gain           = 0.70
recurrent_gain       = 0.45
leak                 = 0.35
input_weight_spread  = 0.30
activity_threshold   = 0.10
persistent_state     = 128
trainable_backend_parameters = 0
```

Purpose: determine whether generic graph recurrence explains an apparent AURUM effect.

C2 edge count is retained but intentionally not required to equal AURUM's edge count in this first characterization. Exact topology/edge matching is a distinct future lineage.

## 4. Comparison matrix and commitment topology

PHYS-002 declares one comparator per manifest. Therefore every workload/comparator pair receives an independent manifest and comparison plan:

```text
M(W,C) = exact PHYS-002 manifest
P(W,C) = exact PHYS-004 BoundComparisonPlan bound to commitment(M(W,C))
```

Three primary workloads × three controls = **9 independently committed subject/comparator experiments** before any result exists.

Destroyed-dynamics and shuffled-target checks get additional commitments and cannot replace a primary matrix cell.

## 5. Frozen seed set

Exactly these 16 paired seeds are used in this order:

```text
1, 2, 3, 5, 8, 13, 21, 34,
55, 89, 144, 233, 377, 610, 987, 1597
```

The AURUM seed controls topology, threshold disorder, and stochastic trajectory. Seeds may not be searched, rerolled, replaced, or dropped because of unfavorable results. Missing/failed/invalid pairs remain explicitly missing/failed/invalid.

PHYS-006 requires the fixture record order to match this PHYS-002 `SeedPlan` exactly.

## 6. Frozen fixture identity through PHYS-006

All workload data must be serialized by exact PHYS-006 head `a4ccf536ab5df4f2c59775d355b25c3ee9805f42`.

PHYS-006 binds:

- workload family/version;
- ordered per-seed records;
- exact scalar/binary channel schema;
- exact IEEE-754 scalar values and binary labels;
- suite metadata;
- cross-seed schema invariance;
- exact ordered seed-plan agreement.

Metadata/channel insertion order is non-semantic; seed-record order and every channel value are semantic. The resulting `fixture_digest` is the PHYS-002 `WorkloadIdentity.fixture_digest`.

Subject and comparator must consume the same committed input record for each paired seed.

## 7. Frozen comparison projection

The downstream comparison vector has exactly four axes:

```text
state_mean
state_variance
activity_fraction
output
```

AURUM projection, with conductance span `s=0.85`:

```text
state_mean:        mean_conductance      * (20/17) + (-3/17)
state_variance:    conductance_variance  * (400/289)
activity_fraction: switched_fraction
output:            output_current
```

PHYS-003 control projection:

```text
state_mean:        state_mean      * 0.5 + 0.5
state_variance:    state_variance  * 0.25
activity_fraction: activity_fraction
output:            output
```

AURUM-only diagnostics such as `mean_recurrent_field` and `edge_count` remain raw evidence but are unavailable to the primary task learner. Any projection-key, scale, or offset change creates a new PHYS-004 lineage.

## 8. Structural budget policies

For C1/C2 use `BudgetMatchPolicy::state_and_readout_matched()`:

```text
units                    = required equal = 128
persistent_state_scalars = required equal = 128
recurrent_edges          = explicitly allowed to differ
trainable_parameters     = required equal = 0
readout_scalars          = required equal = 4
```

C0 intentionally removes persistent memory and uses:

```text
units                    = true
persistent_state_scalars = false
recurrent_edges          = false
trainable_parameters     = true
readout_scalars          = true
```

C0 is a negative control, not a state-matched comparator.

## 9. Exact workload subjects

### W1 — NARMA10 / WLOAD-001

Exact head: `65cd00d78e698b5e9189efdd5c3c91341b0fcfa4`

```text
family  = temporal:narma10
version = atiya-parlos-u05-v1
u[t]    ∈ [0, 0.5)
frame t input  = drive_voltage = u[t]
frame t target = y[t+1]
```

Exact recurrence:

```text
y[t+1] = 0.3*y[t]
       + 0.05*y[t]*sum(y[t-i], i=0..9)
       + 1.5*u[t]*u[t-9]
       + 0.1
```

Input uses the versioned, domain-separated SplitMix64 top-53-bit mapping defined by WLOAD-001. Output and delayed-input prehistory are zero. No divergence retry, clipping, tanh wrapper, normalization, sequence replacement, or seed replacement is permitted.

Metrics:

```text
nrmse  direction=Minimize role=Primary
mae    direction=Minimize role=Secondary
```

### W2 — delayed recall / WLOAD-002

Exact head: `65be4f5500df47287adf96264e30c7f49afd05c4`

```text
family  = temporal:delayed-recall
version = uniform-signed-d1-32-v1
u[t]    ∈ [-0.5, 0.5)
frame t input = drive_voltage = u[t]
delays = 1..32 inclusive
```

All delayed targets are fixture channels:

```text
target_delay_dd[t] = u[t-dd] if t >= dd else 0
```

The analysis layer may not regenerate or reinterpret delayed targets.

Metrics:

```text
memory_capacity           direction=Maximize role=Primary
r2_delay_01..r2_delay_32  direction=Maximize role=Diagnostic
```

Memory capacity is the sum of held-out `R²` over delays 1..32 with negative held-out `R²` contributing zero. The complete 32-delay curve is retained.

### W3 — causal change-point / WLOAD-003

Exact head: `b44d0119c6578b3723d776377bb13cdaab8d446b`

```text
family  = temporal:change-point
version = piecewise-four-level-causal-window-v1
levels  = {-0.35, -0.15, +0.15, +0.35}
segment length = exact uniform integer [64, 192]
noise   ∈ [-0.05, +0.05)
drive_voltage ∈ [-0.4, +0.4)
```

Discrete level/segment choices use WLOAD-003's deterministic unbiased rejection sampler (`splitmix64-unbiased-rejection-v1`), not biased modulo reduction.

Fixture channels:

```text
drive_voltage             scalar input
is_change                 1 only on first frame of new segment; frame 0 = 0
within_detection_window   1 on [change_frame, change_frame+3] only
```

The detection tolerance is causal; no pre-change frame receives positive-window credit.

Metrics:

```text
auroc                    direction=Maximize role=Primary
auprc                    direction=Maximize role=Secondary
f1_validation_threshold  direction=Maximize role=Secondary
```

The F1 threshold is selected on validation data only.

No workload or primary metric may be added after results are observed.

## 10. Frozen downstream learner budget

```text
projected features/frame = 4
primary temporal window  = 8 frames
primary feature count    = 32
readout                   = ridge linear regression/classification
backend trainable params = 0
```

Windows 1, 4, and 16 are secondary characterization only; window 8 remains primary regardless of outcome.

Ridge grid:

```text
1e-6, 1e-5, 1e-4, 1e-3, 1e-2,
1e-1, 1e0, 1e1, 1e2
```

Regularization is selected independently for each backend/seed/workload using validation data only. Every arm receives the same search grid and search budget.

Feature normalization is fitted on training data only. Zero-variance features map to zero after centering; no result-dependent epsilon is introduced.

## 11. Frozen temporal split

For the primary subjects above the exact fixture length is:

```text
reservoir warmup = 256 frames
training         = 2048 frames
validation       = 512 frames
test             = 1024 frames
total            = 3840 frames
```

Test data is unavailable to projection design, normalization, learner fitting, regularization selection, threshold selection, window selection, or metric selection.

## 12. Statistical reporting

All primary analysis is paired by seed.

Define canonical **benefit delta** so **positive always means the AURUM arm performed better**:

```text
Minimize metric: benefit_delta = comparator_value - aurum_value
Maximize metric: benefit_delta = aurum_value - comparator_value
```

For every primary metric retain/report:

- every subject value;
- every comparator value;
- every paired benefit delta;
- mean and median benefit delta;
- paired effect size where defined;
- exact two-sided sign-test result;
- valid, missing, failed, and invalid pair counts.

Do not flip metric direction, pool controls, discard unfavorable seeds, or promote secondary/diagnostic metrics after execution. Stronger inferential rules require a new preregistered lineage.

## 13. Energy, latency, and evidence boundary

```text
execution     = Simulation
energy_policy = UnmeasuredAllowed
```

Physical-device energy, whole-system energy, and hardware-latency claims are forbidden. Simulation wall-clock telemetry is engineering data only.

One simulation run produces `Simulated` evidence and cannot self-promote to `Measured` or `Replicated`.

## 14. Negative/destroyed controls

Additional non-primary commitments should include:

1. shuffled temporal targets/labels;
2. recurrence-destroyed C2 with `recurrent_gain=0`;
3. later, AURUM `recurrent_coupling=0`, which necessarily has a distinct configuration digest.

Failure of a negative control invalidates the corresponding cell; it does not become a positive discovery.

## 15. Permitted interpretation

Permitted conclusions remain simulator- and comparator-specific, for example:

- AURUM retained more or less temporal information than C1 under this exact protocol;
- generic recurrence explained or did not explain an observed simulation effect;
- evidence was inconclusive at 16 paired seeds.

AURUM-002 cannot by itself establish a real-gold hardware advantage or justify active-cognition integration.

## 16. Execution gate

**Do not implement or execute the result-producing AURUM-002 runner** until all conditions are simultaneously true:

1. PHYS-001 exact head `5d90397cf04fafd5acf1835f0b16bb5941dc4f2b` is qualified;
2. PHYS-002 exact head `9ae4695ffaba9a94f9ad69a783724659036aa927` is qualified;
3. PHYS-003 exact head `eff60b7f418b632e7602d19b7d7006b6b24bb2f0` is qualified;
4. PHYS-004 exact head `3126a0bb63672b0647d783397cfc6b15164642b8` is qualified;
5. PHYS-005 exact head `5db7f62ef711a7fb9798e2c3d0e3e886c77b913a` is qualified;
6. PHYS-006 exact head `a4ccf536ab5df4f2c59775d355b25c3ee9805f42` is qualified;
7. WLOAD-001 exact head `65cd00d78e698b5e9189efdd5c3c91341b0fcfa4` is qualified;
8. WLOAD-002 exact head `65be4f5500df47287adf96264e30c7f49afd05c4` is qualified;
9. WLOAD-003 exact head `b44d0119c6578b3723d776377bb13cdaab8d446b` is qualified;
10. AURUM-001 exact head `86617dd7aa564846701ddff3d1adc948e0eec4c0` is qualified;
11. exact W1/W2/W3 16-seed × 3840-frame suites are generated from those qualified workload heads and their PHYS-006 digests are retained;
12. all nine primary `M(W,C)` and `P(W,C)` commitments exist before result execution;
13. every runner instance derives its local binding through PHYS-005 and `verify_exact_binding()` succeeds before warmup;
14. exact checkout, fixtures, manifests, comparison plans, raw observations, and receipts are retained for replay.

Queued, pending, skipped, cancelled, stale-head, merge-ref-only, runner-mutated-only, or source-unqualified states are **not** execution authorization.
