// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Arm interface for the V2 temporal benchmark.
//!
//! # Why this trait is token-level
//!
//! `HdcLtcBridge` operates on `Array1<f32>`, not tokens. Every vector-based arm
//! therefore needs an encode/decode layer — **and that layer is a confound**. A
//! weak encoder can sink an otherwise capable mechanism; a leaky one can rescue
//! an incapable mechanism by smuggling the answer into the representation.
//! Either way the benchmark would be measuring embeddings while reporting a
//! verdict about temporal dynamics.
//!
//! The trait is token-level so encoding is a **declared, shared** stage rather
//! than each arm's private business. [`SharedCodec`] is the single encoder all
//! vector arms must use, so a difference between arms is a difference in
//! temporal mechanism.
//!
//! # The contract that makes `Static` a real control
//!
//! The pre-registration requires the negative control be mechanically
//! guaranteed, not hopefully weak. [`StaticArm`] satisfies that by
//! *construction*: it holds no state across [`TemporalArm::observe`] calls, so
//! it cannot condition on history even in principle.
//! [`static_arm_is_mechanically_memoryless`] verifies that property directly
//! rather than inferring it from a score — a low score could mean "memoryless"
//! or merely "badly tuned", and only the mechanical check distinguishes them.

use std::collections::HashMap;

use symthaea_evidence_plane::task_validator::TokenId;

/// An arm under test.
///
/// `observe` is called for each item in order; `predict` is called at a decision
/// point and must return the arm's prediction for the **next** token. `reset`
/// starts a fresh sequence.
pub trait TemporalArm {
    /// Human-readable arm name, recorded in results.
    fn name(&self) -> &'static str;
    /// Begin a new sequence, discarding any per-sequence state.
    fn reset(&mut self);
    /// Observe an item and the elapsed time since the previous one.
    fn observe(&mut self, token: TokenId, dt_since_prev: f64);
    /// Predict the next token.
    fn predict(&mut self) -> TokenId;
}

/// Shared token encoder for vector-based arms.
///
/// Deliberately minimal and deterministic. It exists to be **identical across
/// arms**, not to be good: any capability it adds or removes is added or removed
/// equally for everyone, so it cannot explain a difference between arms. An arm
/// that needs a different encoder to win has not demonstrated a temporal
/// capability, and that should be visible rather than absorbed.
#[derive(Debug, Clone)]
pub struct SharedCodec {
    vocab: Vec<TokenId>,
    index: HashMap<TokenId, usize>,
}

impl SharedCodec {
    /// Build from the corpus vocabulary, in sorted order so the encoding is
    /// reproducible across runs and machines.
    pub fn from_corpus(sequences: &[Vec<TokenId>]) -> Self {
        let mut vocab: Vec<TokenId> = sequences.iter().flatten().copied().collect();
        vocab.sort_unstable();
        vocab.dedup();
        let index = vocab.iter().enumerate().map(|(i, &t)| (t, i)).collect();
        Self { vocab, index }
    }

    pub fn len(&self) -> usize {
        self.vocab.len()
    }

    /// Vocabulary in canonical (sorted) order.
    pub fn vocab(&self) -> &[TokenId] {
        &self.vocab
    }

    pub fn is_empty(&self) -> bool {
        self.vocab.is_empty()
    }

    /// One-hot encode. Unknown tokens yield an all-zero vector rather than
    /// panicking, so an out-of-vocabulary item degrades an arm's input instead
    /// of aborting a run mid-sweep.
    pub fn encode(&self, token: TokenId) -> Vec<f32> {
        let mut v = vec![0.0; self.vocab.len()];
        if let Some(&i) = self.index.get(&token) {
            v[i] = 1.0;
        }
        v
    }

    /// Decode by argmax. Ties resolve to the lowest index, deterministically —
    /// a tie must not become a hidden source of run-to-run variance.
    pub fn decode(&self, v: &[f32]) -> Option<TokenId> {
        let mut best = None;
        let mut best_val = f32::NEG_INFINITY;
        for (i, &x) in v.iter().enumerate().take(self.vocab.len()) {
            if x > best_val {
                best_val = x;
                best = Some(self.vocab[i]);
            }
        }
        best
    }
}

/// The mechanically guaranteed negative control.
///
/// Holds **no state across `observe`**. Its prediction is a fixed function of
/// nothing at all, so at an aliased decision point — where the current token is
/// identical across branches — it must be at chance. Not by tuning, by
/// construction.
#[derive(Debug, Clone)]
pub struct StaticArm {
    guess: TokenId,
}

impl StaticArm {
    /// Commit to one answer. A memoryless arm facing an ambiguous point can do
    /// no better than committing, which is exactly why it lands at chance.
    pub fn new(guess: TokenId) -> Self {
        Self { guess }
    }
}

impl TemporalArm for StaticArm {
    fn name(&self) -> &'static str {
        "Static"
    }
    fn reset(&mut self) {}
    fn observe(&mut self, _token: TokenId, _dt_since_prev: f64) {
        // Intentionally empty. This is the whole contract: nothing observed can
        // reach `predict`. Adding state here would silently void the negative
        // control and every comparison that depends on it.
    }
    fn predict(&mut self) -> TokenId {
        self.guess
    }
}

/// An upper-bound reference: sees the true next token.
///
/// Not a competitor — it exists so a task with a ceiling below 1.0 is
/// detectable. If the oracle cannot score perfectly, the corpus is broken and no
/// arm's number means anything.
#[derive(Debug, Clone)]
pub struct OracleArm {
    answers: Vec<TokenId>,
    cursor: usize,
}

impl OracleArm {
    pub fn new(answers: Vec<TokenId>) -> Self {
        Self { answers, cursor: 0 }
    }
}

impl TemporalArm for OracleArm {
    fn name(&self) -> &'static str {
        "Oracle"
    }
    fn reset(&mut self) {
        self.cursor = 0;
    }
    fn observe(&mut self, _token: TokenId, _dt: f64) {
        self.cursor += 1;
    }
    fn predict(&mut self) -> TokenId {
        self.answers.get(self.cursor).copied().unwrap_or_default()
    }
}

/// Deterministic projection of tokens into a mechanism's working dimension,
/// plus a **parameter-free** readout.
///
/// # Why the readout must have no capacity
///
/// `HdcLtcBridge` returns a state vector, not token logits, so something must
/// map state to token. That mapping is capacity. If each arm brings its own
/// trained readout, an arm can win by having a better readout rather than
/// better temporal dynamics — the same confound as a private encoder, one layer
/// further on, and the plan explicitly requires "identical readout capacity
/// where architecturally possible."
///
/// Here there is no readout to be better at: a state decodes to whichever
/// token's embedding it is closest to by cosine similarity. Nothing is learned,
/// so nothing can differ between arms. Any difference that survives is a
/// difference in the state the mechanism produced.
#[derive(Debug, Clone)]
pub struct SharedProjection {
    dim: usize,
    embeddings: Vec<(TokenId, Vec<f32>)>,
}

impl SharedProjection {
    /// Build seeded, deterministic embeddings for every token in the corpus.
    ///
    /// Uses the high bits of an LCG: the low bits of an LCG are notoriously weak
    /// and taking them modulo a power of two would give visibly structured
    /// "random" vectors.
    pub fn new(codec: &SharedCodec, dim: usize, seed: u64) -> Self {
        let mut embeddings = Vec::with_capacity(codec.len());
        for (i, &token) in codec.vocab().iter().enumerate() {
            let mut state = seed
                .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                .wrapping_add(i as u64 + 1);
            let mut v = Vec::with_capacity(dim);
            for _ in 0..dim {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                // High 24 bits, mapped to [-1, 1].
                let u = ((state >> 40) as f32) / (((1u64 << 24) - 1) as f32);
                v.push(u * 2.0 - 1.0);
            }
            let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
            for x in v.iter_mut() {
                *x /= norm;
            }
            embeddings.push((token, v));
        }
        Self { dim, embeddings }
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Embedding for a token, or `None` if out of vocabulary.
    pub fn encode(&self, token: TokenId) -> Option<&[f32]> {
        self.embeddings
            .iter()
            .find(|(t, _)| *t == token)
            .map(|(_, v)| v.as_slice())
    }

    /// Parameter-free readout: nearest token by cosine similarity.
    ///
    /// Ties resolve to the earliest vocabulary entry, deterministically, so a
    /// tie cannot become hidden run-to-run variance.
    pub fn decode_nearest(&self, state: &[f32]) -> Option<TokenId> {
        let norm = state.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm <= 1e-8 {
            return None;
        }
        let mut best: Option<(TokenId, f32)> = None;
        for (token, emb) in &self.embeddings {
            let dot: f32 = emb
                .iter()
                .zip(state.iter())
                .map(|(a, b)| a * b)
                .sum::<f32>()
                / norm;
            match best {
                Some((_, b)) if dot <= b => {}
                _ => best = Some((*token, dot)),
            }
        }
        best.map(|(t, _)| t)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// THE CONTROL'S DEFINING PROPERTY, checked mechanically rather than
    /// inferred from a score. A low score could mean "memoryless" or merely
    /// "badly tuned"; only this distinguishes them.
    #[test]
    fn static_arm_is_mechanically_memoryless() {
        let mut a = StaticArm::new(42);
        let baseline = a.predict();

        // Feed wildly different histories; the prediction must not move.
        for hist in [
            vec![(1u32, 0.1), (2, 0.2), (3, 0.3)],
            vec![(9, 99.0), (8, 88.0)],
            vec![],
        ] {
            a.reset();
            for (t, dt) in hist {
                a.observe(t, dt);
            }
            assert_eq!(
                a.predict(),
                baseline,
                "Static conditioned on history — the negative control is void"
            );
        }
    }

    /// The encoder must be identical across arms, which means identical across
    /// constructions from the same corpus.
    #[test]
    fn shared_codec_is_deterministic_and_order_independent() {
        let a = SharedCodec::from_corpus(&[vec![5, 1, 3], vec![3, 1]]);
        let b = SharedCodec::from_corpus(&[vec![3, 1], vec![1, 3, 5]]);
        assert_eq!(a.len(), b.len());
        for t in [1u32, 3, 5] {
            assert_eq!(a.encode(t), b.encode(t), "encoding differs for {t}");
        }
    }

    #[test]
    fn codec_round_trips_known_tokens() {
        let c = SharedCodec::from_corpus(&[vec![10, 20, 30]]);
        for t in [10u32, 20, 30] {
            assert_eq!(c.decode(&c.encode(t)), Some(t));
        }
    }

    /// An unknown token must degrade input, not abort a sweep.
    #[test]
    fn unknown_token_encodes_to_zero_without_panicking() {
        let c = SharedCodec::from_corpus(&[vec![1, 2]]);
        assert!(c.encode(999).iter().all(|&x| x == 0.0));
    }

    /// Ties must not become hidden run-to-run variance.
    #[test]
    fn decode_ties_resolve_deterministically() {
        let c = SharedCodec::from_corpus(&[vec![7, 8, 9]]);
        let flat = vec![1.0, 1.0, 1.0];
        assert_eq!(c.decode(&flat), c.decode(&flat));
        assert_eq!(c.decode(&flat), Some(7), "lowest index, deterministically");
    }

    #[test]
    fn oracle_arm_tracks_the_answer_sequence() {
        let mut o = OracleArm::new(vec![11, 22, 33]);
        assert_eq!(o.predict(), 11);
        o.observe(0, 0.0);
        assert_eq!(o.predict(), 22);
        o.reset();
        assert_eq!(o.predict(), 11);
    }
}

#[cfg(test)]
mod projection_tests {
    use super::*;

    fn proj() -> SharedProjection {
        let codec = SharedCodec::from_corpus(&[vec![1, 2, 3, 4]]);
        SharedProjection::new(&codec, 64, 0xABCD)
    }

    /// Identical across arms means identical across constructions.
    #[test]
    fn projection_is_deterministic() {
        let a = proj();
        let b = proj();
        for t in [1u32, 2, 3, 4] {
            assert_eq!(a.encode(t), b.encode(t));
        }
    }

    /// A token's own embedding must decode back to it, or the readout is broken
    /// before any mechanism is involved.
    #[test]
    fn embedding_decodes_to_its_own_token() {
        let p = proj();
        for t in [1u32, 2, 3, 4] {
            let e = p.encode(t).expect("in vocab").to_vec();
            assert_eq!(p.decode_nearest(&e), Some(t));
        }
    }

    /// Distinct tokens must be distinguishable, otherwise the readout collapses
    /// the vocabulary and every arm scores identically for the wrong reason.
    #[test]
    fn distinct_tokens_are_not_collapsed() {
        let p = proj();
        let mut decoded: Vec<TokenId> = [1u32, 2, 3, 4]
            .iter()
            .map(|&t| p.decode_nearest(p.encode(t).unwrap()).unwrap())
            .collect();
        decoded.sort_unstable();
        assert_eq!(
            decoded,
            vec![1, 2, 3, 4],
            "readout collapsed the vocabulary"
        );
    }

    /// A degenerate state must not silently decode to an arbitrary token.
    #[test]
    fn zero_state_decodes_to_none() {
        let p = proj();
        assert_eq!(p.decode_nearest(&vec![0.0; 64]), None);
    }

    /// The readout has no learnable parameters, so it cannot differ between
    /// arms. This test exists to fail if someone adds state to it later.
    #[test]
    fn readout_is_stateless_across_calls() {
        let p = proj();
        let e = p.encode(3).unwrap().to_vec();
        let first = p.decode_nearest(&e);
        for _ in 0..10 {
            assert_eq!(p.decode_nearest(&e), first);
        }
    }
}

/// Multi-timescale exponential-trace baseline — the arm that must be beaten.
///
/// The pre-registration binds this to being tuned as hard as `HdcLtc`, because
/// a strawman EMA voids the comparison: EMA already showed *stronger* regime
/// separation than the HDC-LTC coupling in the predecessor ablation, at a 4-5x
/// lower compute cost.
///
/// # Why this is a credible competitor on §5.2, not a placeholder
///
/// With [`EmaBankArm::dt_aware`] set, each trace decays by `exp(-dt / half_life)`
/// — so elapsed time *itself* changes the state, which is precisely the claim
/// §5.2 tests and precisely what closed-form LTC dynamics are said to provide.
/// A bank of well-separated half-lives is a real answer to "a small number of
/// well-separated relevant timescales," which is the structurally-motivated
/// opportunity the plan requires EMA be given.
///
/// If this arm solves §5.2, the family does not require anything exotic. That
/// would be a result, not a failure of the benchmark.
#[derive(Debug, Clone)]
pub struct EmaBankArm {
    projection: SharedProjection,
    half_lives: Vec<f64>,
    traces: Vec<Vec<f32>>,
    dt_aware: bool,
}

impl EmaBankArm {
    /// `half_lives` are in the same time units as `dt_since_prev`.
    ///
    /// `dt_aware = false` gives the order-only ablation: traces decay a fixed
    /// amount per event regardless of elapsed time. Keeping both in one type is
    /// deliberate — it makes "does timing matter" a one-flag comparison within
    /// the same mechanism rather than a comparison across two implementations
    /// that might differ for unrelated reasons.
    pub fn new(projection: SharedProjection, half_lives: Vec<f64>, dt_aware: bool) -> Self {
        let dim = projection.dim();
        let traces = vec![vec![0.0; dim]; half_lives.len()];
        Self {
            projection,
            half_lives,
            traces,
            dt_aware,
        }
    }

    /// Sensible default bank: well-separated timescales spanning the intervals
    /// §5.2 uses. Not claimed to be tuned — the pre-registration requires tuning
    /// before any reported comparison, and this is a starting point for that,
    /// not a substitute for it.
    pub fn default_bank(projection: SharedProjection, dt_aware: bool) -> Self {
        Self::new(projection, vec![0.25, 1.0, 4.0, 16.0], dt_aware)
    }

    /// Combined state: the mean across traces.
    fn combined(&self) -> Vec<f32> {
        let dim = self.projection.dim();
        let mut out = vec![0.0; dim];
        for tr in &self.traces {
            for (o, t) in out.iter_mut().zip(tr.iter()) {
                *o += *t;
            }
        }
        let n = self.traces.len().max(1) as f32;
        for o in out.iter_mut() {
            *o /= n;
        }
        out
    }
}

impl TemporalArm for EmaBankArm {
    fn name(&self) -> &'static str {
        "EmaBank"
    }

    fn reset(&mut self) {
        for tr in self.traces.iter_mut() {
            tr.iter_mut().for_each(|x| *x = 0.0);
        }
    }

    fn observe(&mut self, token: TokenId, dt_since_prev: f64) {
        let Some(emb) = self.projection.encode(token) else {
            return; // Out of vocabulary: degrade, do not abort a sweep.
        };
        let emb = emb.to_vec();
        for (i, tr) in self.traces.iter_mut().enumerate() {
            let hl = self.half_lives[i].max(1e-9);
            // dt-aware: elapsed time itself sets the decay. Order-only: one
            // event is one unit of time regardless of the interval.
            let elapsed = if self.dt_aware { dt_since_prev } else { 1.0 };
            let decay = (-(elapsed.max(0.0)) / hl).exp() as f32;
            for (t, e) in tr.iter_mut().zip(emb.iter()) {
                *t = *t * decay + e * (1.0 - decay);
            }
        }
    }

    fn predict(&mut self) -> TokenId {
        self.projection
            .decode_nearest(&self.combined())
            .unwrap_or_default()
    }
}

#[cfg(test)]
mod ema_tests {
    use super::*;

    fn projection() -> SharedProjection {
        let codec = SharedCodec::from_corpus(&[vec![1, 2, 3, 4, 5]]);
        SharedProjection::new(&codec, 128, 0x51EED)
    }

    /// THE §5.2 CAPABILITY. Identical tokens in identical order, different
    /// elapsed interval — a dt-aware bank must reach a materially different
    /// state, because that is the entire claim the family tests.
    #[test]
    fn dt_aware_bank_distinguishes_intervals() {
        let mut fast = EmaBankArm::default_bank(projection(), true);
        let mut slow = EmaBankArm::default_bank(projection(), true);

        fast.observe(1, 0.0);
        fast.observe(2, 0.1);
        slow.observe(1, 0.0);
        slow.observe(2, 20.0);

        let (a, b) = (fast.combined(), slow.combined());
        let diff: f32 = a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).sum();
        assert!(
            diff > 1e-3,
            "dt-aware traces must diverge on different intervals, got {diff}"
        );
    }

    /// The order-only control must NOT distinguish them. This is what makes the
    /// previous test meaningful: it shows the divergence comes from timing
    /// rather than from anything else in the mechanism.
    #[test]
    fn order_only_bank_is_blind_to_intervals() {
        let mut fast = EmaBankArm::default_bank(projection(), false);
        let mut slow = EmaBankArm::default_bank(projection(), false);

        fast.observe(1, 0.0);
        fast.observe(2, 0.1);
        slow.observe(1, 0.0);
        slow.observe(2, 20.0);

        let (a, b) = (fast.combined(), slow.combined());
        let diff: f32 = a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).sum();
        assert!(
            diff < 1e-6,
            "order-only traces must be identical regardless of interval, got {diff}"
        );
    }

    /// §5.1 capability: different cues must leave different traces at an
    /// otherwise identical decision point.
    #[test]
    fn bank_retains_cue_identity_through_a_shared_token() {
        let mut a = EmaBankArm::default_bank(projection(), false);
        let mut b = EmaBankArm::default_bank(projection(), false);
        a.observe(1, 1.0);
        a.observe(3, 1.0); // shared alias token
        b.observe(2, 1.0);
        b.observe(3, 1.0);

        let (x, y) = (a.combined(), b.combined());
        let diff: f32 = x.iter().zip(y.iter()).map(|(p, q)| (p - q).abs()).sum();
        assert!(diff > 1e-3, "cue identity must survive the shared token");
    }

    /// Reset must fully clear state, or sequences leak into each other and every
    /// per-sequence result is contaminated.
    #[test]
    fn reset_fully_clears_state() {
        let mut arm = EmaBankArm::default_bank(projection(), true);
        arm.observe(4, 1.0);
        assert!(arm.combined().iter().any(|x| x.abs() > 1e-9));
        arm.reset();
        assert!(
            arm.combined().iter().all(|x| x.abs() < 1e-12),
            "state leaked across reset"
        );
    }

    /// An out-of-vocabulary token must degrade input, not abort a sweep.
    #[test]
    fn unknown_token_does_not_panic() {
        let mut arm = EmaBankArm::default_bank(projection(), true);
        arm.observe(9999, 1.0);
        let _ = arm.predict();
    }

    /// Determinism: identical input must give identical state.
    #[test]
    fn bank_is_deterministic() {
        let mut a = EmaBankArm::default_bank(projection(), true);
        let mut b = EmaBankArm::default_bank(projection(), true);
        for (t, dt) in [(1u32, 0.5), (2, 3.0), (5, 0.2)] {
            a.observe(t, dt);
            b.observe(t, dt);
        }
        assert_eq!(a.combined(), b.combined());
    }
}
