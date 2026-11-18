//! Typed, decoder-synchronous byte predictors for the fixed SCM1 record layout.
//!
//! These are deliberately simple controls, not learned causal mechanisms:
//! - ScmFieldPredictor: one adaptive byte histogram per (variable, byte lane).
//! - ScmParentPredictor: conditions on declared same-timestep parent bytes;
//!   this is an oracle-topology control, not learned graph discovery.
//! - ScmLagPredictor: conditions on the same field in the previous timestep.
//! - ScmHybridPredictor: combines coarse parent context with temporal lag.
//!
//! The 33-byte SCM1 header is modeled separately. The five i16 variables then
//! arrive in topological order, little-endian, two bytes per variable.

use crate::{ByteCdf, BytePredictor, BYTE_SYMBOLS};

const SCM_HEADER_BYTES: u64 = 33;
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

/// Oracle-topology SCM model: conditions each field/lane histogram on coarse
/// same-timestep parent bytes from the fixed SCM graph. Model ID 3.
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

/// Noncausal temporal baseline: conditions each variable/lane on the same
/// variable/lane in the preceding timestep. Model ID 4. This control has
/// temporal context but does not inspect other variables or use the SCM graph.
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
    fn field_parent_lag_and_hybrid_predictors_have_distinct_stable_ids() {
        assert_eq!(ScmFieldPredictor::default().model_id(), 2);
        assert_eq!(ScmParentPredictor::default().model_id(), 3);
        assert_eq!(ScmLagPredictor::default().model_id(), 4);
        assert_eq!(ScmHybridPredictor::default().model_id(), 5);
        assert_eq!(ScmShuffledParentPredictor::default().model_id(), 6);
        assert!(ScmFieldPredictor::state_bytes() > 0);
        assert!(ScmParentPredictor::state_bytes() > 0);
        assert!(ScmLagPredictor::state_bytes() > 0);
        assert!(ScmHybridPredictor::state_bytes() > 0);
        assert!(ScmShuffledParentPredictor::state_bytes() > 0);
    }
}
