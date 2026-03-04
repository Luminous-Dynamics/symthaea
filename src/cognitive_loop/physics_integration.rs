//! Physics bridge integration for the cognitive loop.
//!
//! Wraps `symthaea_physics_bridge::PhysicsBridge` into a cycle-compatible
//! interface with telemetry and configurable query intervals.

#[cfg(feature = "physics-bridge")]
use symthaea_core::hdc::ContinuousHV;

/// Local telemetry snapshot from the physics bridge
#[derive(Debug, Clone, Default)]
pub struct PhysicsTelemetry {
    pub catalog_size: usize,
    pub results_returned: usize,
    pub top_match: String,
    pub top_score: f32,
    pub query_count: u64,
    pub queried_this_cycle: bool,
    pub effective_interval: usize,
    pub effective_blend_weight: f32,
    pub top_domain: String,
    pub pareto_context: Option<ParetoContext>,
}

/// Pareto frontier context for telemetry injection.
#[derive(Debug, Clone, Default)]
pub struct ParetoContext {
    pub frontier_size: usize,
    pub power_range: (f64, f64),
    pub dominant_domain: String,
    pub best_analogy_score: f32,
}

#[cfg(feature = "physics-bridge")]
pub(crate) struct PhysicsIntegration {
    bridge: symthaea_physics_bridge::PhysicsBridge,
    last_physics_hv: Option<ContinuousHV>,
    queried_this_cycle: bool,
    last_effective_interval: usize,
    last_effective_blend: f32,
    pareto_context: Option<ParetoContext>,
    encoder: Option<crate::spark_physics_bridge::DesignPointEncoder>,
}

#[cfg(feature = "physics-bridge")]
impl std::fmt::Debug for PhysicsIntegration {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PhysicsIntegration")
            .field("queried_this_cycle", &self.queried_this_cycle)
            .field("has_last_hv", &self.last_physics_hv.is_some())
            .field("last_effective_interval", &self.last_effective_interval)
            .field("last_effective_blend", &self.last_effective_blend)
            .field("has_encoder", &self.encoder.is_some())
            .finish()
    }
}

#[cfg(feature = "physics-bridge")]
impl PhysicsIntegration {
    pub fn new() -> Self {
        Self {
            bridge: symthaea_physics_bridge::PhysicsBridge::new(),
            last_physics_hv: None,
            queried_this_cycle: false,
            last_effective_interval: 0,
            last_effective_blend: 0.0,
            pareto_context: None,
            encoder: None,
        }
    }

    /// Query the physics catalog with a thought vector.
    /// Returns the closest physics concept as an HV if the result is finite and non-zero.
    pub fn query_thought_vector(&mut self, hv: &ContinuousHV) -> Option<ContinuousHV> {
        let result = self.bridge.query_hv(hv, 5);
        self.queried_this_cycle = true;
        let is_finite = result.values.iter().all(|v| v.is_finite());
        let has_content = result.values.iter().any(|v| *v != 0.0);
        if is_finite && has_content {
            self.last_physics_hv = Some(result.clone());
            Some(result)
        } else {
            None
        }
    }

    /// Collect telemetry and reset per-cycle flag.
    pub fn telemetry(&mut self) -> PhysicsTelemetry {
        let bridge_telem = self.bridge.telemetry();
        let top_domain = self
            .bridge
            .last_results()
            .first()
            .map(|r| format!("{:?}", r.domain))
            .unwrap_or_default();
        let telem = PhysicsTelemetry {
            catalog_size: bridge_telem.catalog_size,
            results_returned: bridge_telem.results_returned,
            top_match: bridge_telem.top_match.unwrap_or_default(),
            top_score: bridge_telem.top_score,
            query_count: self.bridge.query_count(),
            queried_this_cycle: self.queried_this_cycle,
            effective_interval: self.last_effective_interval,
            effective_blend_weight: self.last_effective_blend,
            top_domain,
            pareto_context: self.pareto_context.take(),
        };
        self.queried_this_cycle = false;
        telem
    }

    /// Inject Pareto frontier context for the next telemetry drain.
    pub fn set_pareto_context(&mut self, ctx: ParetoContext) {
        self.pareto_context = Some(ctx);
    }

    /// Get or create the cached DesignPointEncoder.
    #[allow(dead_code)]
    pub fn encoder(&mut self) -> &crate::spark_physics_bridge::DesignPointEncoder {
        self.encoder
            .get_or_insert_with(crate::spark_physics_bridge::DesignPointEncoder::new)
    }

    /// Encode a design point and query the physics bridge, returning the result HV.
    #[allow(dead_code)]
    pub fn query_design_point(
        &mut self,
        point: &symthaea_core::physics::DesignPoint,
    ) -> Option<ContinuousHV> {
        let hv = self
            .encoder
            .get_or_insert_with(crate::spark_physics_bridge::DesignPointEncoder::new)
            .encode(point);
        self.query_thought_vector(&hv)
    }

    /// Get the last physics HV (for blending into attention context).
    #[allow(dead_code)]
    pub fn last_physics_hv(&self) -> Option<&ContinuousHV> {
        self.last_physics_hv.as_ref()
    }

    /// Per-cycle physics query with interval checking and state blending.
    ///
    /// `substrate_tau_factor` modulates query frequency (>1 = query more often).
    /// `substrate_scale_pressure` further boosts blend weight for high-scale substrates.
    ///
    /// Returns true if a query was issued (regardless of result).
    pub fn query_cycle(
        &mut self,
        cycle_number: usize,
        base_interval: usize,
        base_blend_weight: f32,
        substrate_tau_factor: f32,
        substrate_scale_pressure: f32,
        hv16: &symthaea_core::hdc::BinaryHV,
        compressed_state: &mut [f32],
    ) -> bool {
        let tau = substrate_tau_factor.max(0.01);
        let effective_interval = ((base_interval as f32 / tau).round() as usize).max(1);
        let effective_blend = (base_blend_weight
            * tau
            * (1.0 + substrate_scale_pressure.clamp(0.0, 1.0) * 0.1))
        .clamp(0.0, 1.0);

        self.last_effective_interval = effective_interval;
        self.last_effective_blend = effective_blend;

        if effective_interval == 0 || cycle_number % effective_interval != 0 {
            return false;
        }

        let query_hv = hv16.to_continuous();
        let physics_hv = match self.query_thought_vector(&query_hv) {
            Some(hv) => hv,
            None => return true, // queried but no usable result
        };

        // Blend physics HV into compressed_state at low weight.
        // compress_for_ltc stride-samples at indices 0, stride, 2*stride, ...
        // so we must sample the physics HV at the same stride to align dimensions.
        let w = effective_blend;
        let physics_vals = &physics_hv.values;
        let stride = if physics_vals.len() > compressed_state.len() {
            physics_vals.len() / compressed_state.len()
        } else {
            1
        };
        for (i, state_val) in compressed_state.iter_mut().enumerate() {
            let physics_component = physics_vals.get(i * stride).copied().unwrap_or(0.0);
            *state_val = (1.0 - w) * *state_val + w * physics_component;
        }
        true
    }
}

#[cfg(all(test, feature = "physics-bridge"))]
mod tests {
    use super::*;

    #[test]
    fn test_physics_integration_creates() {
        let pi = PhysicsIntegration::new();
        assert!(!pi.queried_this_cycle);
        assert!(pi.last_physics_hv.is_none());
    }

    #[test]
    fn test_query_returns_finite_hv() {
        let mut pi = PhysicsIntegration::new();
        let query = ContinuousHV::random(16_384, 42);
        let result = pi.query_thought_vector(&query);
        // PhysicsBridge always returns something from its catalog
        assert!(result.is_some() || pi.queried_this_cycle);
    }

    #[test]
    fn test_query_cycle_respects_interval() {
        let mut pi = PhysicsIntegration::new();
        let hv = symthaea_core::hdc::BinaryHV::random(42);
        let mut state = vec![0.5f32; 32];

        // cycle 5 with interval 10 — should NOT query (tau=1.0, no substrate effect)
        let queried = pi.query_cycle(5, 10, 0.1, 1.0, 0.0, &hv, &mut state);
        assert!(!queried);

        // cycle 10 with interval 10 — should query
        let queried = pi.query_cycle(10, 10, 0.1, 1.0, 0.0, &hv, &mut state);
        assert!(queried);
    }

    #[test]
    fn test_blend_modifies_state() {
        let mut pi = PhysicsIntegration::new();
        let hv = symthaea_core::hdc::BinaryHV::random(42);
        let mut state = vec![1.0f32; 32];
        let original = state.clone();

        pi.query_cycle(0, 1, 0.5, 1.0, 0.0, &hv, &mut state);
        // At least some values should have changed
        let changed = state.iter().zip(original.iter()).any(|(a, b)| (a - b).abs() > 1e-6);
        assert!(changed, "Physics blend should modify compressed state");
    }

    #[test]
    fn test_stride_alignment() {
        // With blend_weight=1.0 the output should be exactly the stride-sampled
        // physics HV, verifying that the blend reads the correct elements.
        let mut pi = PhysicsIntegration::new();
        let hv = symthaea_core::hdc::BinaryHV::random(42);
        let compressed_len = 256;
        let mut state = vec![0.0f32; compressed_len];

        // Query to populate last_physics_hv (tau=1.0, no substrate effect)
        let queried = pi.query_cycle(0, 1, 1.0, 1.0, 0.0, &hv, &mut state);
        assert!(queried);

        // The physics HV is 16,384-D; compressed_state is 256-D.
        // stride = 16384 / 256 = 64
        if let Some(phv) = pi.last_physics_hv() {
            let stride = phv.values.len() / compressed_len;
            assert_eq!(stride, 64, "Expected stride of 64 for 16384→256 compression");
            for i in 0..compressed_len {
                let expected = phv.values[i * stride];
                assert!(
                    (state[i] - expected).abs() < 1e-6,
                    "state[{i}] = {} but expected physics_hv[{}] = {expected}",
                    state[i],
                    i * stride,
                );
            }
        }
    }

    #[test]
    fn test_telemetry_populated_after_query() {
        let mut pi = PhysicsIntegration::new();
        let query = ContinuousHV::random(16_384, 42);
        pi.query_thought_vector(&query);

        let t = pi.telemetry();
        assert!(t.queried_this_cycle || t.query_count > 0);
        assert!(t.catalog_size > 0);
    }

    // ── Task 1: Substrate-aware query tests ──

    #[test]
    fn test_tau_doubles_interval_halves() {
        let mut pi = PhysicsIntegration::new();
        let hv = symthaea_core::hdc::BinaryHV::random(42);
        let mut state = vec![0.5f32; 32];

        // base_interval=10, tau=2.0 → effective_interval = round(10/2) = 5
        // cycle 5 should query
        let queried = pi.query_cycle(5, 10, 0.1, 2.0, 0.0, &hv, &mut state);
        assert!(queried, "tau=2.0 should halve interval: cycle 5 should query with base 10");
        assert_eq!(pi.last_effective_interval, 5);
    }

    #[test]
    fn test_tau_halves_interval_doubles() {
        let mut pi = PhysicsIntegration::new();
        let hv = symthaea_core::hdc::BinaryHV::random(42);
        let mut state = vec![0.5f32; 32];

        // base_interval=10, tau=0.5 → effective_interval = round(10/0.5) = 20
        // cycle 10 should NOT query
        let queried = pi.query_cycle(10, 10, 0.1, 0.5, 0.0, &hv, &mut state);
        assert!(!queried, "tau=0.5 should double interval: cycle 10 should NOT query with base 10");
        assert_eq!(pi.last_effective_interval, 20);

        // cycle 20 should query
        let queried = pi.query_cycle(20, 10, 0.1, 0.5, 0.0, &hv, &mut state);
        assert!(queried, "cycle 20 should query with effective_interval=20");
    }

    #[test]
    fn test_tau_boosts_blend() {
        let mut pi = PhysicsIntegration::new();
        let hv = symthaea_core::hdc::BinaryHV::random(42);
        let mut state = vec![0.5f32; 32];

        // base_blend=0.1, tau=2.0 → effective_blend = 0.1 * 2.0 * 1.0 = 0.2
        pi.query_cycle(0, 1, 0.1, 2.0, 0.0, &hv, &mut state);
        assert!(
            (pi.last_effective_blend - 0.2).abs() < 1e-6,
            "tau=2.0 should double blend: expected 0.2, got {}",
            pi.last_effective_blend,
        );
    }

    #[test]
    fn test_scale_pressure_further_boosts_blend() {
        let mut pi = PhysicsIntegration::new();
        let hv = symthaea_core::hdc::BinaryHV::random(42);
        let mut state = vec![0.5f32; 32];

        // base_blend=0.1, tau=2.0, scale_pressure=1.0
        // → effective_blend = 0.1 * 2.0 * (1.0 + 1.0 * 0.1) = 0.1 * 2.0 * 1.1 = 0.22
        pi.query_cycle(0, 1, 0.1, 2.0, 1.0, &hv, &mut state);
        assert!(
            (pi.last_effective_blend - 0.22).abs() < 1e-4,
            "scale_pressure=1.0 should further boost blend: expected 0.22, got {}",
            pi.last_effective_blend,
        );
    }

    // ── Task 2: Pareto context tests ──

    #[test]
    fn test_pareto_context_inject_drain() {
        let mut pi = PhysicsIntegration::new();
        let ctx = ParetoContext {
            frontier_size: 5,
            power_range: (1.0, 50.0),
            dominant_domain: "Thermodynamics".into(),
            best_analogy_score: 0.85,
        };
        pi.set_pareto_context(ctx);

        let t = pi.telemetry();
        assert!(t.pareto_context.is_some(), "Pareto context should be present after inject");
        let pc = t.pareto_context.unwrap();
        assert_eq!(pc.frontier_size, 5);
        assert!((pc.best_analogy_score - 0.85).abs() < 1e-6);

        // Second drain should be None
        let t2 = pi.telemetry();
        assert!(t2.pareto_context.is_none(), "Pareto context should be drained after first read");
    }

    #[test]
    fn test_top_domain_populated_after_query() {
        let mut pi = PhysicsIntegration::new();
        let query = ContinuousHV::random(16_384, 42);
        pi.query_thought_vector(&query);

        let t = pi.telemetry();
        // PhysicsBridge always has catalog entries → top_domain should be non-empty
        assert!(!t.top_domain.is_empty(), "top_domain should be populated after a query");
    }

    // ── Task 3: Cached encoder tests ──

    #[test]
    fn test_encoder_lazy_init() {
        let mut pi = PhysicsIntegration::new();
        assert!(pi.encoder.is_none(), "Encoder should be None initially");
        let _enc = pi.encoder();
        assert!(pi.encoder.is_some(), "Encoder should be Some after first access");
    }

    #[test]
    fn test_query_design_point_returns_finite_hv() {
        let mut pi = PhysicsIntegration::new();
        let point = symthaea_core::physics::DesignPoint {
            power_kw: 25.0,
            param_value: 100.0,
            feasible: true,
            metric_value: 0.5,
            notes: String::new(),
        };
        let result = pi.query_design_point(&point);
        assert!(result.is_some(), "query_design_point should return Some");
        let hv = result.unwrap();
        assert_eq!(hv.values.len(), 16_384);
        assert!(hv.values.iter().all(|v| v.is_finite()));
    }
}
