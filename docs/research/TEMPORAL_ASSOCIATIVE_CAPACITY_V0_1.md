# Temporal associative-retrieval capacity (v0.1)

**Status:** research-only protocol and executable example  
**Issue:** [#13099](https://github.com/Luminous-Dynamics/symthaea/issues/13099)  
**Example:** `crates/core/symthaea-hdc-ltc/examples/temporal_associative_capacity.rs`  
**Protocol version:** `symthaea.temporal-associative-capacity/0.1`

## Research question

How much key–value information remains recoverable from an HDC bundle after the state is allowed to evolve over elapsed time and receive unrelated events?

This is a narrow measurement bridge between static VSA capacity and continuous-time temporal dynamics. It is deliberately not a learned episodic-memory claim: v0.1 performs **passive retention** of a static superposition. It does not train the neuron to store, consolidate, or recall episodes.

## Existing algebra and task

For each seed, dimension (D), and fan-in (K), generate independent bipolar key/value codebooks:

[
 k_i,v_iin{-1,+1}^D.
]

Store the superposition using the crate's existing operators:

[
 M=operatorname{bundle}_{i=1}^{K}(k_iodot v_i).
]

For query key (k_i), unbind and rank the frozen value codebook:

[
 q_i=k_iodot h,qquad
 hat i=argmax_joperatorname{cos}(q_i,v_j).
]

No new HDC binding or bundling primitive is introduced. The static HDC bundle is the incumbent reference.

## Experimental conditions

| Condition | Initial state / query | Interpretation |
|---|---|---|
| `static_hdc_bundle` | Original bundle, original keys | Static retrieval reference |
| `static_mismatched_pairing_control` | Bundle keys with the next value in a cyclic permutation; original expected labels | Deliberately wrong association; should not recover the original pairing |
| `static_random_query_control` | Original bundle, independent random query keys | Null control; top-1 accuracy should be near (1/K) over enough independent seeds |
| `hdc_ltc_zero_input` | Original bundle evolved under zero input | Passive state evolution without new sensory content |
| `hdc_ltc_distractor` | Original bundle evolved under deterministic unrelated bipolar inputs | Retention under interference |

For dynamic conditions, the same input-event count is run with regular and deterministic irregular positive time gaps summing approximately to the requested horizon. Each gap is supplied causally; no future input is visible. Static controls are emitted once per corpus, not copied into each temporal cohort.

The neuron is initialized with the exact same bundle via the existing public state setter and the same deterministic model seed across the two dynamic conditions. Its configuration, weights and state are therefore paired for each seed/dimension/load.

## Metrics

Every query produces a JSONL row containing the chosen value index, correctness, tie/validity flags, target similarity, best non-target similarity and target-vs-best-distractor margin.

Each cohort also emits:
- top-1 accuracy, with invalid/tied outcomes not silently counted as correct;
- chance accuracy (1/K);
- mean target/distractor similarity and retrieval margin;
- a **finite-sample plug-in Fano diagnostic**;
- neuron API update count and neuron-reported logical time;
- separate wall-time observations for model setup, input preparation, state evolution and query decoding.

For (K) balanced labels and top-1 error rate (e), the Fano expression is:

[
I(Y;hat Y)geq
max{0,log_2 K-h_2(e)-elog_2(K-1)}.
]

The JSON field is named `plug_in_fano_information_lower_bound_bits`, but the value is calculated from the finite cohort's observed error rate. It is a descriptive plug-in diagnostic—not a confidence interval, a distribution-free finite-sample guarantee, or a substitute for estimating mutual information under a specified sampling distribution. The diagnostic is not the same as retrieval accuracy: a deterministic but wrong permutation can still preserve information about the label while failing the original semantic lookup. The per-query rows are retained so that uncertainty estimation and alternative estimators can be applied later.

Wall-clock fields are observations from a single process run, not energy measurements. The model setup and input preparation costs are separated from the neuron-evolution loop; the latter includes the neuron method's own post-step statistics work. The API update counter counts public evolution calls, not floating-point operations or internal component operations.

## Running

From the repository root:

```bash
cargo run --release --manifest-path crates/core/symthaea-hdc-ltc/Cargo.toml \
  --example temporal_associative_capacity -- \
  --dims 1024,4096 --loads 4,8,16 --horizons 1,10 \
  --events 1,8 --seeds 5 --out temporal-capacity.jsonl
```

Arguments:
- `--dims`: vector dimensions (2–262144);
- `--loads`: memory fan-in (2–512);
- `--horizons`: finite non-negative horizons in seconds; zero is reserved for identity/static controls;
- `--events`: number of evolution calls for each dynamic cohort;
- `--seeds`: number of deterministic corpus/model seeds;
- `--out`: JSONL destination; omit it or pass `-` for stdout.

The JSONL stream contains one `query` record per item followed by one `summary` record for each cohort. A summary with (K) queries uses all (K) queries as its denominator; invalid and tied queries remain visible and cannot improve the correctness count.

## What v0.1 can and cannot establish

It can characterize, on synthetic bipolar key/value codebooks:
- static associative retrieval as fan-in and dimension vary;
- passive state retention under the selected HDC-LTC configuration;
- sensitivity to elapsed horizon and event-gap irregularity;
- the additional effect of unrelated event inputs;
- whether a simple random-query null control behaves around chance;
- basic update/setup/preparation/decode cost components.

It cannot establish:
- learned episodic memory, consolidation, online adaptation or continual learning;
- a formal information-retention law or certified mutual-information estimate;
- superiority to state-space, transformer, neural CDE, SNN or commercial retrieval systems;
- real-world data performance, hardware throughput or energy efficiency;
- a universal HDC capacity threshold.

## Next qualification stages

1. Verify exact retrieval and random-query null behavior over a larger independent seed set.
2. Add paired confidence intervals and explicit bootstrap/randomization tests at the seed/corpus level; queries within one bundle must not be treated as independent replicates.
3. Compose work accounting with the existing temporal ledger (#13060–#13062), without creating a second ledger.
4. Compare representation families using #11145's algebraic-conformance contract (MAP, HRR/FHRR, GHRR, sparse, BinaryHV and ContinuousHV).
5. Add matched dynamical controls: frozen state, liquid-only, HDC-only, bounded iterative evolution and an independent high-accuracy reference where an equivalent trajectory is defined.
6. Only then test online updates, retention under new learning, consolidation/replay, capacity-versus-forgetting, and task-level utility.

The benchmark should be judged by a quality/cost/robustness frontier with matched information, labels, splits, seeds and system boundaries—not by vector width, a single top-1 score, or one unpaired timing.

## Research context

- Static VSA capacity is already being studied in #7772 and #11145. IBM's 2026 JAIR work studies capacity bounds for MAP-I, MAP-B and sparse-binary VSAs on set-membership and set-intersection tasks; those results motivate task-specific empirical curves but are not directly reproduced by this key/value recall protocol.
- Irregular-time input integration is already owned by #13031 and the NIALIM/DFA-CfN source audit by #13042. This example deliberately does not claim to solve time-varying-input integration.
- Temporal work instrumentation and decision freshness are owned by #13060–#13062 and #13091. This example reports only the simple neuron-local counters available from its crate and should later consume the canonical ledger.
- The wider irregular-time comparison should map to BITS/TIME-style time-series evaluation and NeuroBench's separation of algorithm and deployed-system tracks where the tasks genuinely match.
