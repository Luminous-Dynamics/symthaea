//! Typed, decoder-synchronous byte predictors for the fixed SCM2 record layout.
//!
//! These are deliberately simple controls, not learned causal mechanisms:
//! - ScmFieldPredictor: one adaptive byte histogram per (variable, byte lane).
//! - ScmParentPredictor: conditions on declared same-timestep parent i16 values;
//!   this is an oracle-topology control, not learned graph discovery.
//! - ScmLagPredictor: conditions on the same field in the previous timestep.
//! - ScmHybridPredictor: combines coarse parent context with temporal lag.
//!
//! The 16-byte SCM2 schema/length header is modeled separately. The five i16 variables then
//! arrive in topological order, little-endian, two bytes per variable.

use crate::{ByteCdf, BytePredictor, BYTE_SYMBOLS};

const SCM_HEADER_BYTES: u64 = 16;
const SCM_VARIABLES: usize = 5;
const BYTE_LANES: usize = 2;
const CONTEXTS: usize = 16;
const MAX_CONTEXT_TOTAL: u32 = 1 << 14;

type BaseHistograms = [[[u16; BYTE_SYMBOLS]; BYTE_LANES]; SCM_VARIABLES];
type ContextHistograms = [[[[u16; BYTE_SYMBOLS]; CONTEXTS]; BYTE_LANES]; SCM_VARIABLES];

struct StructuredState {
    position: u64,
    header: [u16; BYTE_SYMBOLS],
    base: BaseHistograms,
    contexts: ContextHistograms,
    current_record: [u8; SCM_VARIABLES * BYTE_LANES],
    previous_record: [u8; SCM_VARIABLES * BYTE_LANES],
}

impl StructuredState {
    fn new() -> Self {
        Self {
            position: 0,
            header: [1; BYTE_SYMBOLS],
            base: [[[1; BYTE_SYMBOLS]; BYTE_LANES]; SCM_VARIABLES],
            contexts: [[[[1; BYTE_SYMBOLS]; CONTEXTS]; BYTE_LANES]; SCM_VARIABLES],
            current_record: [0; SCM_VARIABLES * BYTE_LANES],
            previous_record: [0; SCM_VARIABLES * BYTE_LANES],
        }
    }

    fn context_index(&self, variable: usize, lane: usize, conditioning: u8) -> usize {
        match conditioning {
            1 => self.parent_context(variable, lane),
            2 => Self::value_bucket(&self.previous_record, variable),
            3 => self.hybrid_context(variable, lane),
            4 => self.shuffled_parent_context(variable),
            _ => 0,
        }
    }

    fn value_bucket(record: &[u8; SCM_VARIABLES * BYTE_LANES], variable: usize) -> usize {
        let offset = variable * BYTE_LANES;
        let value = i16::from_le_bytes([record[offset], record[offset + 1]]) as i32;
        (((value + 32_768) * 4) / 65_536) as usize
    }

    fn parent_context(&self, variable: usize, _lane: usize) -> usize {
        let bucket = |parent: usize| Self::value_bucket(&self.current_record, parent);

        // Each full signed parent value is quantized to four coarse bins. Parents are
        // earlier in the serialized topological order, so both encoder and
        // decoder have already observed them when this CDF is requested.
        match variable {
            0 => 0,                         // x0: no same-step parent
            1 => bucket(0),                 // x1 <- x0
            2 => bucket(0) | (bucket(1) << 2), // x2 <- x0, x1
            3 => bucket(2),                 // x3 <- x2
            4 => bucket(1) | (bucket(3) << 2), // x4 <- x1, x3
            _ => 0,
        }
    }

    fn hybrid_context(&self, variable: usize, _lane: usize) -> usize {
        // Reduce the supplied parent context to one bit per parent, then
        // reserve two additional bits for the previous same-field value.
        // This caps the combined context space at 16 bins.
        let parent_bucket = |parent: usize| -> usize {
            (Self::value_bucket(&self.current_record, parent) >> 1) as usize
        };
        let parents = match variable {
            0 => 0,
            1 => parent_bucket(0),
            2 => parent_bucket(0) | (parent_bucket(1) << 1),
            3 => parent_bucket(2),
            4 => parent_bucket(1) | (parent_bucket(3) << 1),
            _ => 0,
        };
        let lag = Self::value_bucket(&self.previous_record, variable);
        (parents << 2) | lag
    }

    fn shuffled_parent_context(&self, variable: usize) -> usize {
        let bucket = |parent: usize| Self::value_bucket(&self.current_record, parent);

        // Deliberately mis-specified but topologically decodable mapping:
        // x1 <- x0 (only previous field available);
        // x2 <- x1,x0 (same parents in reversed positions);
        // x3 <- x1 (wrong: true parent is x2);
        // x4 <- x0,x2 (wrong: true parents are x1,x3).
        // This preserves the number of parent slots/context bins while
        // breaking the correct parent identities on downstream fields.
        match variable {
            0 => 0,
            1 => bucket(0),
            2 => bucket(1) | (bucket(0) << 2),
            3 => bucket(1),
            4 => bucket(0) | (bucket(2) << 2),
            _ => 0,
        }
    }

    fn make_cdf(histogram: &[u16; BYTE_SYMBOLS]) -> ByteCdf {
        let mut cdf = [0u32; BYTE_SYMBOLS + 1];
        for (index, frequency) in histogram.iter().enumerate() {
            cdf[index + 1] = cdf[index] + *frequency as u32;
        }
        cdf
    }

    fn update_histogram(histogram: &mut [u16; BYTE_SYMBOLS], symbol: u8) {
        let total: u32 = histogram.iter().map(|&count| count as u32).sum();
        if total >= MAX_CONTEXT_TOTAL {
            for count in histogram.iter_mut() {
                *count = ((*count as u32 + 1) / 2).max(1) as u16;
            }
        }
        histogram[symbol as usize] = histogram[symbol as usize].saturating_add(1);
    }

    fn observe(&mut self, symbol: u8, conditioning: u8) {
        if self.position < SCM_HEADER_BYTES {
            Self::update_histogram(&mut self.header, symbol);
        } else {
            let offset = self.position - SCM_HEADER_BYTES;
            let variable = ((offset / BYTE_LANES as u64) % SCM_VARIABLES as u64) as usize;
            let lane = (offset % BYTE_LANES as u64) as usize;
            let context = self.context_index(variable, lane, conditioning);

            // Global field/lane counts are shared across contexts. Conditional
            // counts add context-specific evidence when a context is enabled.
            Self::update_histogram(&mut self.base[variable][lane], symbol);
            if conditioning != 0 {
                Self::update_histogram(
                    &mut self.contexts[variable][lane][context],
                    symbol,
                );
            }

            // Store the observed byte only after the context is selected.
            self.current_record[variable * BYTE_LANES + lane] = symbol;
            if variable == SCM_VARIABLES - 1 && lane == BYTE_LANES - 1 {
                self.previous_record = self.current_record;
            }
        }
        self.position += 1;
    }

    fn cdf(&self, conditioning: u8) -> ByteCdf {
        if self.position < SCM_HEADER_BYTES {
            return Self::make_cdf(&self.header);
        }

        let offset = self.position - SCM_HEADER_BYTES;
        let variable = ((offset / BYTE_LANES as u64) % SCM_VARIABLES as u64) as usize;
        let lane = (offset % BYTE_LANES as u64) as usize;
        let base = &self.base[variable][lane];
        if conditioning == 0 {
            return Self::make_cdf(base);
        }

        // Additive interpolation combines the global field/lane evidence with
        // the selected context's evidence. It is decoder-synchronous and has
        // a deterministic integer CDF; both histograms are updated per symbol.
        let context = self.context_index(variable, lane, conditioning);
        let conditional = &self.contexts[variable][lane][context];
        let mut cdf = [0u32; BYTE_SYMBOLS + 1];
        for index in 0..BYTE_SYMBOLS {
            cdf[index + 1] =
                cdf[index] + base[index] as u32 + conditional[index] as u32;
        }
        cdf
    }
}

/// Adaptive field/lane model: preserves typed channel identity but ignores
/// cross-variable parent structure. Model ID 2 in the CDL custom-predictor space.
pub struct ScmFieldPredictor {
    state: StructuredState,
}

impl ScmFieldPredictor {
    pub fn new() -> Self {
        Self {
            state: StructuredState::new(),
        }
    }

    /// Approximate inline predictor state, excluding allocator/runtime overhead.
    pub fn state_bytes() -> usize {
        std::mem::size_of::<Self>()
    }
}

impl Default for ScmFieldPredictor {
    fn default() -> Self {
        Self::new()
    }
}

impl BytePredictor for ScmFieldPredictor {
    fn model_id(&self) -> u8 {
        2
    }

    fn cumulative_frequency(&self) -> ByteCdf {
        self.state.cdf(0)
    }

    fn observe(&mut self, symbol: u8) {
        self.state.observe(symbol, 0);
    }
}

/// Oracle-topology SCM model: conditions each field/lane distribution on coarse
/// same-timestep parent i16 values from the fixed SCM graph. Model ID 3.
///
/// This supplies the graph to the compressor. It must be reported as an
/// oracle-structure control, never as learned causal discovery or transfer.
pub struct ScmParentPredictor {
    state: StructuredState,
}

impl ScmParentPredictor {
    pub fn new() -> Self {
        Self {
            state: StructuredState::new(),
        }
    }

    /// Approximate inline predictor state, excluding allocator/runtime overhead.
    pub fn state_bytes() -> usize {
        std::mem::size_of::<Self>()
    }
}

impl Default for ScmParentPredictor {
    fn default() -> Self {
        Self::new()
    }
}

impl BytePredictor for ScmParentPredictor {
    fn model_id(&self) -> u8 {
        3
    }

    fn cumulative_frequency(&self) -> ByteCdf {
        self.state.cdf(1)
    }

    fn observe(&mut self, symbol: u8) {
        self.state.observe(symbol, 1);
    }
}

/// Noncausal temporal baseline: conditions each variable/lane on the previous
/// timestep's reconstructed full i16 value for that variable. Model ID 4.
/// This control has temporal context but does not inspect other variables or graph structure.
pub struct ScmLagPredictor {
    state: StructuredState,
}

impl ScmLagPredictor {
    pub fn new() -> Self {
        Self {
            state: StructuredState::new(),
        }
    }

    /// Approximate inline predictor state, excluding allocator/runtime overhead.
    pub fn state_bytes() -> usize {
        std::mem::size_of::<Self>()
    }
}

impl Default for ScmLagPredictor {
    fn default() -> Self {
        Self::new()
    }
}

impl BytePredictor for ScmLagPredictor {
    fn model_id(&self) -> u8 {
        4
    }

    fn cumulative_frequency(&self) -> ByteCdf {
        self.state.cdf(2)
    }

    fn observe(&mut self, symbol: u8) {
        self.state.observe(symbol, 2);
    }
}

/// Combined conditioning control: known same-step parents plus the
/// previous-timestep value of the target field. It is an oracle-graph model,
/// not learned causal structure. Parent buckets are reduced to one bit each
/// so parent and lag context fit within the fixed 16-context table. Model ID 5.
pub struct ScmHybridPredictor {
    state: StructuredState,
}

impl ScmHybridPredictor {
    pub fn new() -> Self {
        Self {
            state: StructuredState::new(),
        }
    }

    /// Approximate inline predictor state, excluding allocator/runtime overhead.
    pub fn state_bytes() -> usize {
        std::mem::size_of::<Self>()
    }
}

impl Default for ScmHybridPredictor {
    fn default() -> Self {
        Self::new()
    }
}

impl BytePredictor for ScmHybridPredictor {
    fn model_id(&self) -> u8 {
        5
    }

    fn cumulative_frequency(&self) -> ByteCdf {
        self.state.cdf(3)
    }

    fn observe(&mut self, symbol: u8) {
        self.state.observe(symbol, 3);
    }
}

const EXPERT_MIXTURE_TOTAL_WEIGHT: u32 = 1 << 24;
const EXPERT_COUNT: usize = 3;
// Fixed-share hazard: 1/256 of posterior mass is redistributed equally per byte.
const EXPERT_SHARE_TOTAL_WEIGHT: u32 = EXPERT_MIXTURE_TOTAL_WEIGHT / 256;

fn normalize_expert_cdf(cdf: &ByteCdf) -> [u32; BYTE_SYMBOLS] {
    let total = cdf[BYTE_SYMBOLS] as u64;
    let distributable = (1u64 << 15) - BYTE_SYMBOLS as u64;
    let mut frequencies = [1u32; BYTE_SYMBOLS];
    let mut remainders = [0u64; BYTE_SYMBOLS];
    let mut assigned = 0u64;

    for symbol in 0..BYTE_SYMBOLS {
        let raw = (cdf[symbol + 1] - cdf[symbol]) as u64;
        let scaled = raw * distributable;
        let extra = scaled / total;
        remainders[symbol] = scaled % total;
        frequencies[symbol] += extra as u32;
        assigned += extra;
    }

    let leftover = (distributable - assigned) as usize;
    let mut order: [usize; BYTE_SYMBOLS] = std::array::from_fn(|i| i);
    order.sort_unstable_by(|left, right| {
        remainders[*right]
            .cmp(&remainders[*left])
            .then_with(|| left.cmp(right))
    });
    for &symbol in order.iter().take(leftover) {
        frequencies[symbol] += 1;
    }
    frequencies
}

fn make_expert_mixture_cdf(
    weights: &[u32; EXPERT_COUNT],
    experts: &[[u32; BYTE_SYMBOLS]; EXPERT_COUNT],
) -> ByteCdf {
    let weight_total = weights.iter().map(|&w| w as u64).sum::<u64>();
    let target = 1u64 << 15;
    let distributable = target - BYTE_SYMBOLS as u64;
    let denominator = weight_total * target;
    let mut frequencies = [1u32; BYTE_SYMBOLS];
    let mut remainders = [0u64; BYTE_SYMBOLS];
    let mut assigned = 0u64;

    for symbol in 0..BYTE_SYMBOLS {
        let numerator = (0..EXPERT_COUNT)
            .map(|expert| weights[expert] as u64 * experts[expert][symbol] as u64)
            .sum::<u64>();
        let scaled = numerator * distributable;
        let extra = scaled / denominator;
        remainders[symbol] = scaled % denominator;
        frequencies[symbol] += extra as u32;
        assigned += extra;
    }

    let leftover = (distributable - assigned) as usize;
    let mut order: [usize; BYTE_SYMBOLS] = std::array::from_fn(|i| i);
    order.sort_unstable_by(|left, right| {
        remainders[*right]
            .cmp(&remainders[*left])
            .then_with(|| left.cmp(right))
    });
    for &symbol in order.iter().take(leftover) {
        frequencies[symbol] += 1;
    }

    let mut cdf = [0u32; BYTE_SYMBOLS + 1];
    for symbol in 0..BYTE_SYMBOLS {
        cdf[symbol + 1] = cdf[symbol] + frequencies[symbol];
    }
    cdf
}

fn update_expert_weights(
    prior: &[u32; EXPERT_COUNT],
    likelihoods: &[u32; EXPERT_COUNT],
) -> [u32; EXPERT_COUNT] {
    let evidence = (0..EXPERT_COUNT)
        .map(|i| prior[i] as u128 * likelihoods[i] as u128)
        .sum::<u128>();
    let distributable =
        EXPERT_MIXTURE_TOTAL_WEIGHT as u128 - EXPERT_COUNT as u128;
    let mut posterior = [1u32; EXPERT_COUNT];
    let mut remainders = [0u128; EXPERT_COUNT];
    let mut assigned = 0u128;

    for i in 0..EXPERT_COUNT {
        let scaled = prior[i] as u128 * likelihoods[i] as u128 * distributable;
        let extra = scaled / evidence;
        remainders[i] = scaled % evidence;
        posterior[i] += extra as u32;
        assigned += extra;
    }

    let leftover = (distributable - assigned) as usize;
    let mut order = [0usize, 1, 2];
    order.sort_unstable_by(|left, right| {
        remainders[*right]
            .cmp(&remainders[*left])
            .then_with(|| left.cmp(right))
    });
    for &i in order.iter().take(leftover) {
        posterior[i] += 1;
    }

    // Fixed-share switching gives every expert fresh probability mass at each
    // byte. This avoids permanent lock-in after a long run of one expert while
    // keeping all weights decoder-synchronous and integer deterministic.
    let retained_total = EXPERT_MIXTURE_TOTAL_WEIGHT - EXPERT_SHARE_TOTAL_WEIGHT;
    let share_base = EXPERT_SHARE_TOTAL_WEIGHT / EXPERT_COUNT as u32;
    let share_remainder = EXPERT_SHARE_TOTAL_WEIGHT % EXPERT_COUNT as u32;
    let mut shared = [share_base; EXPERT_COUNT];
    let mut share_remainders = [0u64; EXPERT_COUNT];
    let mut retained_assigned = 0u64;
    for i in 0..EXPERT_COUNT {
        let scaled = posterior[i] as u64 * retained_total as u64;
        let retained = scaled / EXPERT_MIXTURE_TOTAL_WEIGHT as u64;
        share_remainders[i] = scaled % EXPERT_MIXTURE_TOTAL_WEIGHT as u64;
        shared[i] += retained as u32;
        if i < share_remainder as usize {
            shared[i] += 1;
        }
        retained_assigned += retained;
    }
    let missing = (EXPERT_MIXTURE_TOTAL_WEIGHT as u64
        - shared.iter().map(|&weight| weight as u64).sum::<u64>()) as usize;
    let mut share_order = [0usize, 1, 2];
    share_order.sort_unstable_by(|left, right| {
        share_remainders[*right]
            .cmp(&share_remainders[*left])
            .then_with(|| left.cmp(right))
    });
    for &i in share_order.iter().take(missing) {
        shared[i] += 1;
    }
    debug_assert_eq!(
        shared.iter().map(|&weight| weight as u64).sum::<u64>(),
        EXPERT_MIXTURE_TOTAL_WEIGHT as u64
    );
    debug_assert!(retained_assigned <= retained_total as u64);
    shared
}

/// Decoder-synchronous Bayesian mixture of typed-field, temporal-lag and
/// known-parent SCM byte experts. The posterior weights are updated from the
/// likelihood of each decoded byte, with a positive integer floor so a weak
/// expert can recover after a regime change. No expert-selection symbols are
/// transmitted. The parent expert receives the known SCM topology and is an
/// oracle control, not a learned causal model. Model ID 7.
pub struct ScmExpertMixturePredictor {
    field: StructuredState,
    lag: StructuredState,
    parent: StructuredState,
    weights: [u32; EXPERT_COUNT],
}

impl ScmExpertMixturePredictor {
    pub fn new() -> Self {
        let third = EXPERT_MIXTURE_TOTAL_WEIGHT / 3;
        Self {
            field: StructuredState::new(),
            lag: StructuredState::new(),
            parent: StructuredState::new(),
            weights: [third, third, EXPERT_MIXTURE_TOTAL_WEIGHT - 2 * third],
        }
    }

    /// Posterior integer weights in expert order: field, lag, parent.
    pub fn expert_weights(&self) -> [u32; EXPERT_COUNT] {
        self.weights
    }

    /// Approximate inline predictor storage, excluding allocator/runtime overhead.
    pub fn state_bytes() -> usize {
        std::mem::size_of::<Self>()
    }

    fn expert_frequencies(&self) -> [[u32; BYTE_SYMBOLS]; EXPERT_COUNT] {
        [
            normalize_expert_cdf(&self.field.cdf(0)),
            normalize_expert_cdf(&self.lag.cdf(2)),
            normalize_expert_cdf(&self.parent.cdf(1)),
        ]
    }
}

impl Default for ScmExpertMixturePredictor {
    fn default() -> Self {
        Self::new()
    }
}

impl BytePredictor for ScmExpertMixturePredictor {
    fn model_id(&self) -> u8 {
        7
    }

    fn cumulative_frequency(&self) -> ByteCdf {
        make_expert_mixture_cdf(&self.weights, &self.expert_frequencies())
    }

    fn observe(&mut self, symbol: u8) {
        let experts = self.expert_frequencies();
        let likelihoods = [
            experts[0][symbol as usize],
            experts[1][symbol as usize],
            experts[2][symbol as usize],
        ];
        self.weights = update_expert_weights(&self.weights, &likelihoods);

        self.field.observe(symbol, 0);
        self.lag.observe(symbol, 2);
        self.parent.observe(symbol, 1);
    }
}

/// Negative-control topology: keeps parent-context arity but deliberately
/// misassigns downstream parent identities. It is not a valid SCM graph.
/// Model ID 6. This detects whether any coding gain is caused merely by
/// increased context capacity rather than alignment to the true parents.
pub struct ScmShuffledParentPredictor {
    state: StructuredState,
}

impl ScmShuffledParentPredictor {
    pub fn new() -> Self {
        Self {
            state: StructuredState::new(),
        }
    }

    /// Approximate inline predictor state, excluding allocator/runtime overhead.
    pub fn state_bytes() -> usize {
        std::mem::size_of::<Self>()
    }
}

impl Default for ScmShuffledParentPredictor {
    fn default() -> Self {
        Self::new()
    }
}

impl BytePredictor for ScmShuffledParentPredictor {
    fn model_id(&self) -> u8 {
        6
    }

    fn cumulative_frequency(&self) -> ByteCdf {
        self.state.cdf(4)
    }

    fn observe(&mut self, symbol: u8) {
        self.state.observe(symbol, 4);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        compress_with_predictor, decompress_with_predictor, generate_scm, ScmIntervention,
    };

    fn round_trip<P: BytePredictor + Default>(source: &[u8], expected_model_id: u8) {
        let mut encoder = P::default();
        let frame = compress_with_predictor(source, &mut encoder).unwrap();
        assert_eq!(frame[6], expected_model_id);

        let mut decoder = P::default();
        let decoded =
            decompress_with_predictor(&frame, source.len(), &mut decoder).unwrap();
        assert_eq!(decoded, source);
    }

    #[test]
    fn field_predictor_round_trips_observational_and_interventional_scm() {
        let observational = generate_scm(4096, 17, None).unwrap();
        let interventional = generate_scm(
            4096,
            17,
            Some(ScmIntervention {
                variable: 1,
                start_step: 1024,
                end_step: 3072,
                value: 20_000,
            }),
        )
        .unwrap();

        round_trip::<ScmFieldPredictor>(&observational.stream, 2);
        round_trip::<ScmFieldPredictor>(&interventional.stream, 2);
    }

    #[test]
    fn parent_predictor_round_trips_observational_and_interventional_scm() {
        let observational = generate_scm(4096, 23, None).unwrap();
        let interventional = generate_scm(
            4096,
            23,
            Some(ScmIntervention {
                variable: 1,
                start_step: 1024,
                end_step: 3072,
                value: 20_000,
            }),
        )
        .unwrap();

        round_trip::<ScmParentPredictor>(&observational.stream, 3);
        round_trip::<ScmParentPredictor>(&interventional.stream, 3);
    }

    #[test]
    fn lag_predictor_round_trips_observational_and_interventional_scm() {
        let observational = generate_scm(4096, 29, None).unwrap();
        let interventional = generate_scm(
            4096,
            29,
            Some(ScmIntervention {
                variable: 1,
                start_step: 1024,
                end_step: 3072,
                value: 20_000,
            }),
        )
        .unwrap();

        round_trip::<ScmLagPredictor>(&observational.stream, 4);
        round_trip::<ScmLagPredictor>(&interventional.stream, 4);
    }

    #[test]
    fn shuffled_parent_predictor_round_trips_observational_and_interventional_scm() {
        let observational = generate_scm(4096, 37, None).unwrap();
        let interventional = generate_scm(
            4096,
            37,
            Some(ScmIntervention {
                variable: 1,
                start_step: 1024,
                end_step: 3072,
                value: 20_000,
            }),
        )
        .unwrap();

        round_trip::<ScmShuffledParentPredictor>(&observational.stream, 6);
        round_trip::<ScmShuffledParentPredictor>(&interventional.stream, 6);
    }

    #[test]
    fn hybrid_predictor_round_trips_observational_and_interventional_scm() {
        let observational = generate_scm(4096, 31, None).unwrap();
        let interventional = generate_scm(
            4096,
            31,
            Some(ScmIntervention {
                variable: 1,
                start_step: 1024,
                end_step: 3072,
                value: 20_000,
            }),
        )
        .unwrap();

        round_trip::<ScmHybridPredictor>(&observational.stream, 5);
        round_trip::<ScmHybridPredictor>(&interventional.stream, 5);
    }

    #[test]
    fn expert_mixture_round_trips_observational_and_interventional_scm() {
        let observational = generate_scm(4096, 41, None).unwrap();
        let interventional = generate_scm(
            4096,
            41,
            Some(ScmIntervention {
                variable: 1,
                start_step: 1024,
                end_step: 3072,
                value: 20_000,
            }),
        )
        .unwrap();
        round_trip::<ScmExpertMixturePredictor>(&observational.stream, 7);
        round_trip::<ScmExpertMixturePredictor>(&interventional.stream, 7);
    }

    #[test]
    fn expert_mixture_weights_are_positive_normalized_and_deterministic() {
        let trace = generate_scm(4096, 43, None).unwrap();
        let mut first = ScmExpertMixturePredictor::default();
        let mut second = ScmExpertMixturePredictor::default();
        let initial_weights = first.expert_weights();
        for &symbol in &trace.stream {
            first.observe(symbol);
            second.observe(symbol);
            let weights = first.expert_weights();
            assert_eq!(weights, second.expert_weights());
            assert!(weights.iter().all(|&weight| weight > 0));
            assert_eq!(
                weights.iter().map(|&weight| weight as u64).sum::<u64>(),
                EXPERT_MIXTURE_TOTAL_WEIGHT as u64
            );
        }
        assert_ne!(first.expert_weights(), initial_weights);
    }

    #[test]
    fn expert_mixture_shifts_to_lag_during_intervention_without_boundary_labels() {
        let steps = 4096usize;
        let intervention_start = steps / 4;
        let intervention_end = steps * 3 / 4;
        let trace = generate_scm(
            steps,
            43,
            Some(ScmIntervention {
                variable: 1,
                start_step: intervention_start,
                end_step: intervention_end,
                value: 20_000,
            }),
        )
        .unwrap();

        let mut predictor = ScmExpertMixturePredictor::default();
        let mut before = None;
        let mut during = None;
        let mut after = None;
        for (index, &symbol) in trace.stream.iter().enumerate() {
            predictor.observe(symbol);
            let observed_bytes = index + 1;
            if observed_bytes == 16 + intervention_start * 10 {
                before = Some(predictor.expert_weights());
            } else if observed_bytes == 16 + ((intervention_start + intervention_end) / 2) * 10 {
                during = Some(predictor.expert_weights());
            } else if observed_bytes == 16 + (intervention_end + steps / 8) * 10 {
                after = Some(predictor.expert_weights());
            }
        }

        let before = before.expect("pre-intervention checkpoint");
        let during = during.expect("intervention checkpoint");
        let after = after.expect("post-intervention checkpoint");
        assert!(during[1] > before[1], "lag expert should gain weight under intervention");
        assert!(during[2] < before[2], "parent expert should lose weight under intervention");
        assert!(after[2] > during[2], "parent expert should begin recovering after a post-intervention observation window");
    }

    #[test]
    fn all_scm_predictors_have_distinct_stable_ids() {
        assert_eq!(ScmFieldPredictor::default().model_id(), 2);
        assert_eq!(ScmParentPredictor::default().model_id(), 3);
        assert_eq!(ScmLagPredictor::default().model_id(), 4);
        assert_eq!(ScmHybridPredictor::default().model_id(), 5);
        assert_eq!(ScmShuffledParentPredictor::default().model_id(), 6);
        assert_eq!(ScmExpertMixturePredictor::default().model_id(), 7);
        assert!(ScmFieldPredictor::state_bytes() > 0);
        assert!(ScmParentPredictor::state_bytes() > 0);
        assert!(ScmLagPredictor::state_bytes() > 0);
        assert!(ScmHybridPredictor::state_bytes() > 0);
        assert!(ScmShuffledParentPredictor::state_bytes() > 0);
        assert!(ScmExpertMixturePredictor::state_bytes() > ScmLagPredictor::state_bytes());
    }
}
