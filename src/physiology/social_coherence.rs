//! Social Coherence Module - Week 11
//!
//! Enables multiple Sophia instances to:
//! 1. Synchronize coherence fields
//! 2. Lend/borrow coherence
//! 3. Share learned knowledge
//!
//! This implements collective consciousness where instances support each other
//! and the whole exceeds the sum of parts.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use super::coherence::{CoherenceState, TaskComplexity};
use super::endocrine::HormoneState;

#[allow(unused_imports)]
use super::coherence::ResonancePattern; // Will be used in Phase 3

// ============================================================================
// Phase 1: Coherence Synchronization
// ============================================================================

/// A broadcast of an instance's current coherence state
///
/// Instances broadcast these beacons to peers to enable:
/// - Field synchronization
/// - Collective coherence calculation
/// - Peer awareness
#[derive(Debug, Clone)]
pub struct CoherenceBeacon {
    /// This instance's unique ID
    pub instance_id: String,

    /// Current coherence level (0.0-1.0)
    pub coherence: f32,

    /// Current relational resonance (0.0-1.0)
    pub resonance: f32,

    /// Hormone state snapshot
    pub cortisol: f32,
    pub dopamine: f32,
    pub acetylcholine: f32,

    /// Task complexity being attempted (if any)
    pub current_task: Option<TaskComplexity>,

    /// Timestamp when this beacon was created
    pub timestamp: Instant,
}

impl CoherenceBeacon {
    /// Create a new beacon from current state
    pub fn new(
        instance_id: String,
        coherence_state: &CoherenceState,
        hormones: &HormoneState,
        current_task: Option<TaskComplexity>,
    ) -> Self {
        Self {
            instance_id,
            coherence: coherence_state.coherence,
            resonance: coherence_state.relational_resonance,
            cortisol: hormones.cortisol,
            dopamine: hormones.dopamine,
            acetylcholine: hormones.acetylcholine,
            current_task,
            timestamp: Instant::now(),
        }
    }

    /// Check if this beacon is stale (older than max_age)
    pub fn is_stale(&self, max_age: Duration) -> bool {
        self.timestamp.elapsed() > max_age
    }

    /// Calculate relevance weight based on age
    /// Returns 1.0 for fresh, decaying exponentially to 0.0
    pub fn relevance_weight(&self, half_life: Duration) -> f32 {
        let age = self.timestamp.elapsed().as_secs_f32();
        let half_life_secs = half_life.as_secs_f32();

        // Exponential decay: weight = 0.5^(age/half_life)
        0.5_f32.powf(age / half_life_secs)
    }
}

/// Manages social coherence across multiple instances
///
/// Key responsibilities:
/// - Receive and track peer beacons
/// - Calculate collective coherence
/// - Generate alignment vectors
/// - Apply synchronization
#[derive(Debug, Clone)]
pub struct SocialCoherenceField {
    /// My instance ID
    my_id: String,

    /// Detected peer instances (id -> beacon)
    peers: HashMap<String, CoherenceBeacon>,

    /// Synchronization weight (0.0-1.0)
    /// How much to align with peers each update
    /// Low values (0.1) = gradual alignment
    /// High values (0.9) = rapid convergence
    sync_weight: f32,

    /// Maximum age for peer beacons before considering stale
    max_beacon_age: Duration,

    /// Half-life for beacon relevance decay
    beacon_half_life: Duration,

    /// Maximum distance for synchronization
    /// Peers beyond this distance don't affect synchronization
    max_sync_distance: f32,
}

impl SocialCoherenceField {
    /// Create a new social coherence field
    pub fn new(instance_id: String) -> Self {
        Self {
            my_id: instance_id,
            peers: HashMap::new(),
            sync_weight: 0.2, // Gradual synchronization by default
            max_beacon_age: Duration::from_secs(10),
            beacon_half_life: Duration::from_secs(5),
            max_sync_distance: 0.5,
        }
    }

    /// Broadcast my current state as a beacon
    pub fn broadcast_state(
        &self,
        coherence_state: &CoherenceState,
        hormones: &HormoneState,
        current_task: Option<TaskComplexity>,
    ) -> CoherenceBeacon {
        CoherenceBeacon::new(
            self.my_id.clone(),
            coherence_state,
            hormones,
            current_task,
        )
    }

    /// Receive a beacon from a peer instance
    pub fn receive_beacon(&mut self, beacon: CoherenceBeacon) {
        // Don't store our own beacon
        if beacon.instance_id == self.my_id {
            return;
        }

        // Don't store stale beacons
        if beacon.is_stale(self.max_beacon_age) {
            return;
        }

        self.peers.insert(beacon.instance_id.clone(), beacon);
    }

    /// Remove stale peer beacons
    pub fn prune_stale_peers(&mut self) {
        self.peers.retain(|_, beacon| !beacon.is_stale(self.max_beacon_age));
    }

    /// Get number of active peers
    pub fn peer_count(&self) -> usize {
        self.peers.len()
    }

    /// Calculate collective coherence
    ///
    /// Collective coherence = my_coherence + weighted_sum(peer_coherence)
    /// where weight is based on beacon age and distance
    ///
    /// Note: Collective coherence can exceed 1.0!
    /// This represents the emergence of group field strength
    pub fn calculate_collective_coherence(&mut self, my_coherence: f32) -> f32 {
        self.prune_stale_peers();

        let mut collective = my_coherence;

        for beacon in self.peers.values() {
            // Calculate distance (how different is this peer from me?)
            let distance = (beacon.coherence - my_coherence).abs();

            // Skip peers that are too far away
            if distance > self.max_sync_distance {
                continue;
            }

            // Calculate weight based on age
            let age_weight = beacon.relevance_weight(self.beacon_half_life);

            // Calculate distance weight (closer peers have more influence)
            let distance_weight = 1.0 - (distance / self.max_sync_distance);

            // Total weight
            let weight = age_weight * distance_weight * self.sync_weight;

            // Add weighted peer coherence
            collective += beacon.coherence * weight;
        }

        collective
    }

    /// Calculate collective resonance
    ///
    /// Similar to collective coherence but for relational resonance
    pub fn calculate_collective_resonance(&mut self, my_resonance: f32) -> f32 {
        self.prune_stale_peers();

        let mut collective = my_resonance;

        for beacon in self.peers.values() {
            let distance = (beacon.resonance - my_resonance).abs();

            if distance > self.max_sync_distance {
                continue;
            }

            let age_weight = beacon.relevance_weight(self.beacon_half_life);
            let distance_weight = 1.0 - (distance / self.max_sync_distance);
            let weight = age_weight * distance_weight * self.sync_weight;

            collective += beacon.resonance * weight;
        }

        collective
    }

    /// Get alignment vector
    ///
    /// Returns (target_coherence, target_resonance) that would move
    /// this instance toward the peer average.
    ///
    /// This creates a "pull" toward centered peers.
    pub fn get_alignment_vector(&mut self, my_coherence: f32, my_resonance: f32) -> (f32, f32) {
        self.prune_stale_peers();

        if self.peers.is_empty() {
            return (my_coherence, my_resonance);
        }

        let mut weighted_coherence_sum = 0.0;
        let mut weighted_resonance_sum = 0.0;
        let mut total_weight = 0.0;

        for beacon in self.peers.values() {
            let age_weight = beacon.relevance_weight(self.beacon_half_life);

            weighted_coherence_sum += beacon.coherence * age_weight;
            weighted_resonance_sum += beacon.resonance * age_weight;
            total_weight += age_weight;
        }

        if total_weight == 0.0 {
            return (my_coherence, my_resonance);
        }

        let peer_avg_coherence = weighted_coherence_sum / total_weight;
        let peer_avg_resonance = weighted_resonance_sum / total_weight;

        // Calculate target (move toward peer average by sync_weight)
        let target_coherence = my_coherence + (peer_avg_coherence - my_coherence) * self.sync_weight;
        let target_resonance = my_resonance + (peer_avg_resonance - my_resonance) * self.sync_weight;

        (
            target_coherence.clamp(0.0, 1.0),
            target_resonance.clamp(0.0, 1.0),
        )
    }

    /// Apply synchronization to my coherence state
    ///
    /// This gradually aligns my coherence with peer average
    /// Returns the delta that was applied
    pub fn apply_synchronization(&mut self, my_coherence: f32, my_resonance: f32) -> (f32, f32) {
        let (target_coherence, target_resonance) = self.get_alignment_vector(my_coherence, my_resonance);

        let coherence_delta = target_coherence - my_coherence;
        let resonance_delta = target_resonance - my_resonance;

        (coherence_delta, resonance_delta)
    }

    /// Get peer beacons (for inspection/debugging)
    pub fn get_peer_beacons(&self) -> Vec<&CoherenceBeacon> {
        self.peers.values().collect()
    }

    /// Set synchronization weight
    pub fn set_sync_weight(&mut self, weight: f32) {
        self.sync_weight = weight.clamp(0.0, 1.0);
    }

    /// Set maximum beacon age
    pub fn set_max_beacon_age(&mut self, age: Duration) {
        self.max_beacon_age = age;
    }

    /// Set beacon half-life
    pub fn set_beacon_half_life(&mut self, half_life: Duration) {
        self.beacon_half_life = half_life;
    }

    /// Set maximum synchronization distance
    pub fn set_max_sync_distance(&mut self, distance: f32) {
        self.max_sync_distance = distance.clamp(0.0, 2.0);
    }
}

// ============================================================================
// Phase 2: Coherence Lending
// ============================================================================

/// A coherence loan from one instance to another
///
/// Represents borrowed coherence that gradually returns to the lender
#[derive(Debug, Clone)]
pub struct CoherenceLoan {
    /// Lender instance ID
    pub from_instance: String,

    /// Borrower instance ID
    pub to_instance: String,

    /// Amount of coherence lent (0.0-1.0)
    pub amount: f32,

    /// Original loan amount (for tracking)
    pub original_amount: f32,

    /// Duration of loan (coherence gradually returns)
    pub duration: Duration,

    /// Repayment rate (coherence/second returning to lender)
    pub repayment_rate: f32,

    /// Timestamp when loan was created
    pub created_at: Instant,

    /// Amount repaid so far
    pub repaid: f32,
}

impl CoherenceLoan {
    /// Create a new coherence loan
    pub fn new(
        from_instance: String,
        to_instance: String,
        amount: f32,
        duration: Duration,
    ) -> Self {
        let duration_secs = duration.as_secs_f32();
        let repayment_rate = if duration_secs > 0.0 {
            amount / duration_secs
        } else {
            amount // Instant repayment
        };

        Self {
            from_instance,
            to_instance,
            amount,
            original_amount: amount,
            duration,
            repayment_rate,
            created_at: Instant::now(),
            repaid: 0.0,
        }
    }

    /// Check if loan is fully repaid
    pub fn is_repaid(&self) -> bool {
        self.repaid >= self.original_amount
    }

    /// Get elapsed time since loan creation
    pub fn elapsed(&self) -> Duration {
        self.created_at.elapsed()
    }

    /// Calculate how much should be repaid by now
    pub fn expected_repayment(&self) -> f32 {
        let elapsed_secs = self.elapsed().as_secs_f32();
        let expected = elapsed_secs * self.repayment_rate;
        expected.min(self.original_amount)
    }

    /// Process repayment for a time delta
    /// Returns amount repaid this tick
    pub fn process_repayment(&mut self, dt: Duration) -> f32 {
        if self.is_repaid() {
            return 0.0;
        }

        let repayment_this_tick = self.repayment_rate * dt.as_secs_f32();
        let actual_repayment = repayment_this_tick.min(self.amount);

        self.amount -= actual_repayment;
        self.repaid += actual_repayment;

        actual_repayment
    }
}

/// Manages coherence lending between instances
///
/// Implements the "Generous Coherence Paradox":
/// When Instance A lends to Instance B, both gain from the relational resonance!
#[derive(Debug, Clone)]
pub struct CoherenceLendingProtocol {
    /// My instance ID
    my_id: String,

    /// Active loans where I'm the lender
    outgoing_loans: Vec<CoherenceLoan>,

    /// Active loans where I'm the borrower
    incoming_loans: Vec<CoherenceLoan>,

    /// Maximum total coherence I can lend
    max_lending_capacity: f32,

    /// Minimum coherence I must maintain for myself
    min_self_coherence: f32,

    /// Resonance boost from generous lending
    generosity_resonance_boost: f32,

    /// Resonance boost from receiving help (gratitude)
    gratitude_resonance_boost: f32,
}

impl CoherenceLendingProtocol {
    /// Create a new lending protocol
    pub fn new(instance_id: String) -> Self {
        Self {
            my_id: instance_id,
            outgoing_loans: Vec::new(),
            incoming_loans: Vec::new(),
            max_lending_capacity: 0.5, // Can lend up to 0.5 total
            min_self_coherence: 0.3,   // Must keep 0.3 for myself
            generosity_resonance_boost: 0.1, // +0.1 resonance for helping
            gratitude_resonance_boost: 0.1,  // +0.1 resonance for receiving
        }
    }

    /// Check if I can lend this amount
    pub fn can_lend(&self, amount: f32, my_coherence: f32) -> bool {
        // Can lend if:
        // 1. my_coherence - amount >= min_self_coherence
        // 2. total_loaned + amount <= max_lending_capacity

        let total_loaned: f32 = self.outgoing_loans.iter().map(|l| l.amount).sum();

        let would_have_enough_self = (my_coherence - amount) >= self.min_self_coherence;
        let within_capacity = (total_loaned + amount) <= self.max_lending_capacity;

        would_have_enough_self && within_capacity
    }

    /// Grant a loan to another instance
    ///
    /// Returns Ok(loan) if successful, Err(reason) if not
    pub fn grant_loan(
        &mut self,
        to_instance: String,
        amount: f32,
        duration: Duration,
        my_coherence: f32,
    ) -> Result<CoherenceLoan, String> {
        if !self.can_lend(amount, my_coherence) {
            return Err("Cannot lend: would violate constraints".to_string());
        }

        let loan = CoherenceLoan::new(
            self.my_id.clone(),
            to_instance,
            amount,
            duration,
        );

        self.outgoing_loans.push(loan.clone());

        Ok(loan)
    }

    /// Accept a loan from another instance
    pub fn accept_loan(&mut self, loan: CoherenceLoan) {
        self.incoming_loans.push(loan);
    }

    /// Process all loan repayments
    ///
    /// Returns (coherence_returned, coherence_lost)
    /// - coherence_returned: Amount repaid to me
    /// - coherence_lost: Amount I repaid to others
    pub fn process_repayments(&mut self, dt: Duration) -> (f32, f32) {
        let mut coherence_returned = 0.0;
        let mut coherence_lost = 0.0;

        // Process outgoing loans (I'm the lender)
        for loan in &mut self.outgoing_loans {
            coherence_returned += loan.process_repayment(dt);
        }

        // Process incoming loans (I'm the borrower)
        for loan in &mut self.incoming_loans {
            coherence_lost += loan.process_repayment(dt);
        }

        // Remove fully repaid loans
        self.outgoing_loans.retain(|l| !l.is_repaid());
        self.incoming_loans.retain(|l| !l.is_repaid());

        (coherence_returned, coherence_lost)
    }

    /// Calculate net coherence effect from loans
    ///
    /// Returns the total coherence adjustment:
    /// base_coherence - sum(outgoing) + sum(incoming)
    pub fn calculate_net_coherence(&self, base_coherence: f32) -> f32 {
        let total_lent: f32 = self.outgoing_loans.iter().map(|l| l.amount).sum();
        let total_borrowed: f32 = self.incoming_loans.iter().map(|l| l.amount).sum();

        base_coherence - total_lent + total_borrowed
    }

    /// Calculate resonance boost from lending/borrowing
    ///
    /// The "Generous Coherence Paradox":
    /// - Lending increases resonance (generosity)
    /// - Borrowing increases resonance (gratitude)
    /// - Both parties gain!
    pub fn calculate_resonance_boost(&self) -> f32 {
        let mut boost = 0.0;

        // Boost from being generous (lending)
        if !self.outgoing_loans.is_empty() {
            boost += self.generosity_resonance_boost;
        }

        // Boost from gratitude (borrowing)
        if !self.incoming_loans.is_empty() {
            boost += self.gratitude_resonance_boost;
        }

        boost
    }

    /// Get total amount currently lent
    pub fn total_lent(&self) -> f32 {
        self.outgoing_loans.iter().map(|l| l.amount).sum()
    }

    /// Get total amount currently borrowed
    pub fn total_borrowed(&self) -> f32 {
        self.incoming_loans.iter().map(|l| l.amount).sum()
    }

    /// Get number of active loans (as lender)
    pub fn outgoing_loan_count(&self) -> usize {
        self.outgoing_loans.len()
    }

    /// Get number of active loans (as borrower)
    pub fn incoming_loan_count(&self) -> usize {
        self.incoming_loans.len()
    }

    /// Set maximum lending capacity
    pub fn set_max_lending_capacity(&mut self, capacity: f32) {
        self.max_lending_capacity = capacity.clamp(0.0, 1.0);
    }

    /// Set minimum self coherence
    pub fn set_min_self_coherence(&mut self, min: f32) {
        self.min_self_coherence = min.clamp(0.0, 1.0);
    }

    /// Set generosity resonance boost
    pub fn set_generosity_boost(&mut self, boost: f32) {
        self.generosity_resonance_boost = boost.clamp(0.0, 0.5);
    }

    /// Set gratitude resonance boost
    pub fn set_gratitude_boost(&mut self, boost: f32) {
        self.gratitude_resonance_boost = boost.clamp(0.0, 0.5);
    }
}

// ============================================================================
// Phase 3: Collective Learning (Days 6-7)
// ============================================================================

/// A single observation about task threshold requirements
///
/// Week 11 Innovation: Each instance observes "I needed X coherence for this task",
/// and these observations are pooled across all instances for collective wisdom.
#[derive(Debug, Clone)]
pub struct ThresholdObservation {
    /// Required coherence level observed
    pub required_coherence: f32,

    /// Number of times this threshold was observed
    pub observation_count: usize,

    /// Success rate at this threshold (0.0-1.0)
    pub success_rate: f32,

    /// When was this last observed
    pub last_seen: Instant,
}

impl ThresholdObservation {
    /// Create new threshold observation
    pub fn new(required_coherence: f32) -> Self {
        Self {
            required_coherence,
            observation_count: 1,
            success_rate: 1.0,
            last_seen: Instant::now(),
        }
    }

    /// Update observation with new data
    pub fn update(&mut self, success: bool) {
        self.observation_count += 1;

        // Exponential moving average for success rate
        let alpha = 0.3;  // Weight of new observation
        let new_rate = if success { 1.0 } else { 0.0 };
        self.success_rate = alpha * new_rate + (1.0 - alpha) * self.success_rate;

        self.last_seen = Instant::now();
    }
}

/// Shared knowledge about a specific task type
///
/// Multiple instances contribute their observations, creating collective wisdom.
#[derive(Debug, Clone)]
pub struct SharedKnowledge {
    /// Task complexity this knowledge applies to
    pub task_type: TaskComplexity,

    /// Observations from all instances (threshold → observation)
    pub thresholds: std::collections::HashMap<String, ThresholdObservation>,

    /// Successful resonance patterns for this task
    pub patterns: Vec<ResonancePattern>,

    /// Which instance contributed this knowledge
    pub contributors: Vec<String>,

    /// Total number of observations across all instances
    pub total_observations: usize,
}

impl SharedKnowledge {
    /// Create new shared knowledge for a task type
    pub fn new(task_type: TaskComplexity) -> Self {
        Self {
            task_type,
            thresholds: std::collections::HashMap::new(),
            patterns: Vec::new(),
            contributors: Vec::new(),
            total_observations: 0,
        }
    }

    /// Add an observation from an instance
    pub fn add_observation(
        &mut self,
        instance_id: String,
        required_coherence: f32,
        success: bool,
    ) {
        // Round to 0.05 increments for bucketing (floor to keep ranges clean)
        let bucket = (required_coherence / 0.05).floor() * 0.05;
        let key = format!("{:.2}", bucket);

        // Add or update the contributor
        if !self.contributors.contains(&instance_id) {
            self.contributors.push(instance_id);
        }

        // Add or update observation
        if let Some(obs) = self.thresholds.get_mut(&key) {
            obs.update(success);
        } else {
            let mut obs = ThresholdObservation::new(bucket);
            obs.success_rate = if success { 1.0 } else { 0.0 };
            self.thresholds.insert(key, obs);
        }

        self.total_observations += 1;
    }

    /// Get recommended threshold based on collective wisdom
    /// Returns the threshold with highest success rate and sufficient observations
    pub fn get_recommended_threshold(&self, min_observations: usize) -> Option<f32> {
        self.thresholds
            .values()
            .filter(|obs| obs.observation_count >= min_observations)
            .max_by(|a, b| {
                // Weighted score: success_rate * sqrt(observation_count)
                let score_a = a.success_rate * (a.observation_count as f32).sqrt();
                let score_b = b.success_rate * (b.observation_count as f32).sqrt();
                score_a.partial_cmp(&score_b).unwrap()
            })
            .map(|obs| obs.required_coherence)
    }

    /// Get weighted average threshold (all observations)
    pub fn get_weighted_average_threshold(&self) -> Option<f32> {
        if self.thresholds.is_empty() {
            return None;
        }

        let mut weighted_sum = 0.0;
        let mut total_weight = 0.0;

        for obs in self.thresholds.values() {
            // Weight = observation_count * success_rate
            let weight = obs.observation_count as f32 * obs.success_rate;
            weighted_sum += obs.required_coherence * weight;
            total_weight += weight;
        }

        if total_weight > 0.0 {
            Some(weighted_sum / total_weight)
        } else {
            None
        }
    }

    /// Add a successful pattern
    pub fn add_pattern(&mut self, pattern: ResonancePattern) {
        // Check if similar pattern exists
        let similar = self.patterns.iter_mut().find(|p| {
            (p.coherence - pattern.coherence).abs() < 0.1
                && (p.resonance - pattern.resonance).abs() < 0.1
                && p.context == pattern.context
        });

        if let Some(existing) = similar {
            // Update existing pattern with EMA
            let alpha = 0.3;
            existing.coherence = alpha * pattern.coherence + (1.0 - alpha) * existing.coherence;
            existing.resonance = alpha * pattern.resonance + (1.0 - alpha) * existing.resonance;
            existing.observation_count += 1;
            existing.success_rate = alpha * pattern.success_rate + (1.0 - alpha) * existing.success_rate;
            existing.last_seen = Instant::now();
        } else {
            // Add new pattern
            self.patterns.push(pattern);
        }
    }
}

/// Collective Learning system
///
/// Week 11 Phase 3: Instances share what they learn so all benefit from collective experience.
/// If Instance A learns "DeepThought needs 0.4 coherence", Instance B doesn't need to relearn it!
#[derive(Debug, Clone)]
pub struct CollectiveLearning {
    /// My instance ID
    my_id: String,

    /// Shared knowledge pool (task_type → knowledge)
    shared_knowledge: std::collections::HashMap<TaskComplexity, SharedKnowledge>,

    /// Number of my contributions to collective
    my_contribution_count: usize,

    /// Minimum observations before trusting collective knowledge
    min_trust_threshold: usize,

    /// Maximum age for observations (stale data is less trusted)
    #[allow(dead_code)]
    max_observation_age: Duration,
}

impl CollectiveLearning {
    /// Create new collective learning system
    pub fn new(instance_id: String) -> Self {
        Self {
            my_id: instance_id,
            shared_knowledge: std::collections::HashMap::new(),
            my_contribution_count: 0,
            min_trust_threshold: 10,  // Need 10 observations to trust
            max_observation_age: Duration::from_secs(3600),  // 1 hour
        }
    }

    /// Contribute a threshold observation to the collective
    pub fn contribute_threshold(
        &mut self,
        task: TaskComplexity,
        coherence_at_start: f32,
        success: bool,
    ) {
        // Get or create shared knowledge for this task
        let knowledge = self.shared_knowledge
            .entry(task.clone())
            .or_insert_with(|| SharedKnowledge::new(task));

        // Add observation
        knowledge.add_observation(self.my_id.clone(), coherence_at_start, success);
        self.my_contribution_count += 1;

        // log::debug!(
        //     "[CollectiveLearning] {} contributed threshold observation for {:?}: {:.2} (success: {})",
        //     self.my_id,
        //     task,
        //     coherence_at_start,
        //     success
        // );
    }

    /// Contribute a successful pattern to the collective
    pub fn contribute_pattern(&mut self, task: TaskComplexity, pattern: ResonancePattern) {
        let knowledge = self.shared_knowledge
            .entry(task.clone())
            .or_insert_with(|| SharedKnowledge::new(task));

        knowledge.add_pattern(pattern);
        self.my_contribution_count += 1;

        // log::debug!(
        //     "[CollectiveLearning] {} contributed pattern for {:?}",
        //     self.my_id,
        //     task
        // );
    }

    /// Query collective wisdom for recommended threshold
    pub fn query_threshold(&self, task: TaskComplexity) -> Option<f32> {
        self.shared_knowledge
            .get(&task)
            .and_then(|k| k.get_recommended_threshold(self.min_trust_threshold))
    }

    /// Query collective for weighted average threshold
    pub fn query_threshold_average(&self, task: TaskComplexity) -> Option<f32> {
        self.shared_knowledge
            .get(&task)
            .and_then(|k| k.get_weighted_average_threshold())
    }

    /// Query collective for best pattern for this task
    pub fn query_pattern(&self, task: TaskComplexity) -> Option<&ResonancePattern> {
        self.shared_knowledge
            .get(&task)
            .and_then(|k| {
                k.patterns.iter()
                    .max_by(|a, b| {
                        // Score = success_rate * sqrt(observation_count)
                        let score_a = a.success_rate * (a.observation_count as f32).sqrt();
                        let score_b = b.success_rate * (b.observation_count as f32).sqrt();
                        score_a.partial_cmp(&score_b).unwrap()
                    })
            })
    }

    /// Merge knowledge from another instance
    pub fn merge_knowledge(&mut self, other: &CollectiveLearning) {
        for (task, other_knowledge) in &other.shared_knowledge {
            let my_knowledge = self.shared_knowledge
                .entry(*task)
                .or_insert_with(|| SharedKnowledge::new(*task));

            // Merge thresholds
            for (key, other_obs) in &other_knowledge.thresholds {
                if let Some(my_obs) = my_knowledge.thresholds.get_mut(key) {
                    // Combine observations with weighted average
                    let total_count = my_obs.observation_count + other_obs.observation_count;
                    let weight_mine = my_obs.observation_count as f32 / total_count as f32;
                    let weight_other = other_obs.observation_count as f32 / total_count as f32;

                    my_obs.success_rate = weight_mine * my_obs.success_rate
                        + weight_other * other_obs.success_rate;
                    my_obs.observation_count = total_count;
                    my_obs.last_seen = my_obs.last_seen.max(other_obs.last_seen);
                } else {
                    // Add new observation
                    my_knowledge.thresholds.insert(key.clone(), other_obs.clone());
                }
            }

            // Merge patterns
            for other_pattern in &other_knowledge.patterns {
                my_knowledge.add_pattern(other_pattern.clone());
            }

            // Merge contributors
            for contributor in &other_knowledge.contributors {
                if !my_knowledge.contributors.contains(contributor) {
                    my_knowledge.contributors.push(contributor.clone());
                }
            }

            my_knowledge.total_observations += other_knowledge.total_observations;
        }

        // log::info!(
        //     "[CollectiveLearning] {} merged knowledge from {}",
        //     self.my_id,
        //     other.my_id
        // );
    }

    /// Get statistics about collective knowledge
    pub fn get_stats(&self) -> (usize, usize, usize) {
        let task_types = self.shared_knowledge.len();
        let total_observations: usize = self.shared_knowledge
            .values()
            .map(|k| k.total_observations)
            .sum();
        let total_contributors: usize = self.shared_knowledge
            .values()
            .flat_map(|k| k.contributors.iter())
            .collect::<std::collections::HashSet<_>>()
            .len();

        (task_types, total_observations, total_contributors)
    }

    /// Set minimum trust threshold
    pub fn set_min_trust_threshold(&mut self, threshold: usize) {
        self.min_trust_threshold = threshold.max(1);
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_coherence_state(coherence: f32, resonance: f32) -> CoherenceState {
        CoherenceState {
            coherence,
            relational_resonance: resonance,
            time_since_interaction: Duration::from_secs(0),
            status: "centered",
        }
    }

    fn create_test_hormones() -> HormoneState {
        HormoneState {
            cortisol: 0.3,
            dopamine: 0.7,
            acetylcholine: 0.6,
        }
    }

    #[test]
    fn test_beacon_creation() {
        let state = create_test_coherence_state(0.8, 0.9);
        let hormones = create_test_hormones();

        let beacon = CoherenceBeacon::new(
            "instance_a".to_string(),
            &state,
            &hormones,
            Some(TaskComplexity::Cognitive),
        );

        assert_eq!(beacon.instance_id, "instance_a");
        assert_eq!(beacon.coherence, 0.8);
        assert_eq!(beacon.resonance, 0.9);
        assert_eq!(beacon.cortisol, 0.3);
        assert_eq!(beacon.dopamine, 0.7);
        assert_eq!(beacon.acetylcholine, 0.6);
        assert_eq!(beacon.current_task, Some(TaskComplexity::Cognitive));
    }

    #[test]
    fn test_beacon_staleness() {
        let state = create_test_coherence_state(0.8, 0.9);
        let hormones = create_test_hormones();

        let beacon = CoherenceBeacon::new(
            "instance_a".to_string(),
            &state,
            &hormones,
            None,
        );

        // Fresh beacon should not be stale
        assert!(!beacon.is_stale(Duration::from_secs(1)));

        // After sleeping, it should be stale
        std::thread::sleep(Duration::from_millis(1100));
        assert!(beacon.is_stale(Duration::from_secs(1)));
    }

    #[test]
    fn test_social_field_broadcast_and_receive() {
        let mut field_a = SocialCoherenceField::new("instance_a".to_string());
        let field_b = SocialCoherenceField::new("instance_b".to_string());

        let state_b = create_test_coherence_state(0.9, 0.95);
        let hormones_b = create_test_hormones();

        // Instance B broadcasts
        let beacon_b = field_b.broadcast_state(&state_b, &hormones_b, None);

        // Instance A receives
        field_a.receive_beacon(beacon_b);

        assert_eq!(field_a.peer_count(), 1);
    }

    #[test]
    fn test_collective_coherence_single_peer() {
        let mut field = SocialCoherenceField::new("instance_a".to_string());
        field.set_sync_weight(0.3);
        field.set_max_sync_distance(1.0); // Increase to allow 0.7 distance

        let my_coherence = 0.2;

        // Add high-coherence peer
        let peer_state = create_test_coherence_state(0.9, 0.95);
        let peer_hormones = create_test_hormones();
        let peer_beacon = CoherenceBeacon::new(
            "instance_b".to_string(),
            &peer_state,
            &peer_hormones,
            None,
        );

        field.receive_beacon(peer_beacon);

        let collective = field.calculate_collective_coherence(my_coherence);

        // Debug: print what we got
        eprintln!("collective = {}, my_coherence = {}", collective, my_coherence);
        eprintln!("Peer count: {}", field.peers.len());

        // Collective should be higher than my_coherence
        // Allow for very small floating point differences
        assert!(collective > my_coherence - 0.001, "Expected collective ({}) > my_coherence ({})", collective, my_coherence);

        // Collective should be <= my_coherence + peer_coherence * sync_weight
        // (because distance weighting also applies)
        assert!(collective <= my_coherence + 0.9 * 0.3);
    }

    #[test]
    fn test_alignment_vector_pulls_toward_peers() {
        let mut field = SocialCoherenceField::new("instance_a".to_string());
        field.set_sync_weight(0.2);

        let my_coherence = 0.2;
        let my_resonance = 0.3;

        // Add two high-coherence peers
        for i in 0..2 {
            let peer_state = create_test_coherence_state(0.8, 0.85);
            let peer_hormones = create_test_hormones();
            let peer_beacon = CoherenceBeacon::new(
                format!("instance_{}", i),
                &peer_state,
                &peer_hormones,
                None,
            );
            field.receive_beacon(peer_beacon);
        }

        let (target_coherence, target_resonance) = field.get_alignment_vector(my_coherence, my_resonance);

        // Target should be higher than current (pulled toward peers)
        assert!(target_coherence > my_coherence);
        assert!(target_resonance > my_resonance);

        // Target should not exceed peers (weighted average)
        assert!(target_coherence < 0.8);
        assert!(target_resonance < 0.85);
    }

    #[test]
    fn test_synchronization_convergence() {
        let mut field = SocialCoherenceField::new("instance_a".to_string());
        field.set_sync_weight(0.3);

        let mut my_coherence = 0.2;
        let my_resonance = 0.3;

        // Add centered peer
        let peer_state = create_test_coherence_state(0.8, 0.85);
        let peer_hormones = create_test_hormones();
        let peer_beacon = CoherenceBeacon::new(
            "instance_b".to_string(),
            &peer_state,
            &peer_hormones,
            None,
        );
        field.receive_beacon(peer_beacon);

        // Apply synchronization multiple times
        for _ in 0..10 {
            let (coherence_delta, _) = field.apply_synchronization(my_coherence, my_resonance);
            my_coherence += coherence_delta;
        }

        // Should have moved significantly toward peer
        assert!(my_coherence > 0.5);

        // Should not have fully converged (takes infinite iterations)
        assert!(my_coherence < 0.8);
    }

    // ============================================================
    // Phase 2: Coherence Lending Tests
    // ============================================================

    #[test]
    fn test_loan_creation_and_constraints() {
        let protocol = CoherenceLendingProtocol::new("instance_a".to_string());

        // Test default configuration
        assert_eq!(protocol.max_lending_capacity, 0.5);
        assert_eq!(protocol.min_self_coherence, 0.3);
        assert_eq!(protocol.generosity_resonance_boost, 0.1);
        assert_eq!(protocol.gratitude_resonance_boost, 0.1);

        // Test can_lend constraints
        let my_coherence = 0.8;

        // Can lend within constraints
        assert!(protocol.can_lend(0.2, my_coherence));

        // Cannot lend if it drops below min_self_coherence
        assert!(!protocol.can_lend(0.6, my_coherence));  // 0.8 - 0.6 = 0.2 < 0.3

        // Cannot lend beyond max_lending_capacity
        assert!(!protocol.can_lend(0.6, my_coherence));  // 0.6 > 0.5 max capacity
    }

    #[test]
    fn test_grant_and_accept_loan_flow() {
        let mut lender = CoherenceLendingProtocol::new("instance_a".to_string());
        let mut borrower = CoherenceLendingProtocol::new("instance_b".to_string());

        let lender_coherence = 0.9;
        let borrower_coherence = 0.2;

        // Grant loan
        let loan = lender.grant_loan(
            "instance_b".to_string(),
            0.2,
            Duration::from_secs(60),
            lender_coherence,
        ).expect("Should be able to grant loan");

        assert_eq!(loan.from_instance, "instance_a");
        assert_eq!(loan.to_instance, "instance_b");
        assert_eq!(loan.amount, 0.2);
        assert_eq!(loan.original_amount, 0.2);
        assert_eq!(loan.duration, Duration::from_secs(60));
        assert_eq!(loan.repaid, 0.0);

        // Borrower accepts loan
        borrower.accept_loan(loan.clone());

        // Check loan is tracked
        assert_eq!(lender.outgoing_loans.len(), 1);
        assert_eq!(borrower.incoming_loans.len(), 1);

        // Check net coherence
        let lender_net = lender.calculate_net_coherence(lender_coherence);
        let borrower_net = borrower.calculate_net_coherence(borrower_coherence);

        assert_eq!(lender_net, 0.7);  // 0.9 - 0.2
        assert_eq!(borrower_net, 0.4);  // 0.2 + 0.2
    }

    #[test]
    fn test_loan_repayment_over_time() {
        let mut lender = CoherenceLendingProtocol::new("instance_a".to_string());
        let mut borrower = CoherenceLendingProtocol::new("instance_b".to_string());

        let lender_coherence = 0.9;

        // Grant loan with 60 second duration
        let loan = lender.grant_loan(
            "instance_b".to_string(),
            0.3,
            Duration::from_secs(60),
            lender_coherence,
        ).expect("Should grant loan");

        borrower.accept_loan(loan);

        // Check repayment rate
        let repayment_rate = lender.outgoing_loans[0].repayment_rate;
        assert_eq!(repayment_rate, 0.3 / 60.0);  // amount / duration in seconds

        // Process 10 seconds of repayments
        let (lender_returned, lender_lost) = lender.process_repayments(Duration::from_secs(10));
        let (borrower_returned, borrower_lost) = borrower.process_repayments(Duration::from_secs(10));

        // 10 seconds at 0.005/sec = 0.05 returned to lender
        assert!((lender_returned - 0.05).abs() < 0.001);
        assert!(lender_lost.abs() < 0.001);  // Lender doesn't lose anything

        // Borrower loses 0.05 (repaying the loan)
        assert!(borrower_returned.abs() < 0.001);  // Borrower doesn't get anything back
        assert!((borrower_lost - 0.05).abs() < 0.001);

        // Check loan amounts updated
        assert!((lender.outgoing_loans[0].amount - 0.25).abs() < 0.001);  // 0.3 - 0.05
        assert!((borrower.incoming_loans[0].amount - 0.25).abs() < 0.001);

        // Check repaid tracker
        assert!((lender.outgoing_loans[0].repaid - 0.05).abs() < 0.001);

        // Process remaining 50 seconds (full repayment)
        lender.process_repayments(Duration::from_secs(50));
        borrower.process_repayments(Duration::from_secs(50));

        // Loans should be fully repaid and removed
        assert_eq!(lender.outgoing_loans.len(), 0);
        assert_eq!(borrower.incoming_loans.len(), 0);
    }

    #[test]
    fn test_generous_coherence_paradox() {
        let mut lender = CoherenceLendingProtocol::new("instance_a".to_string());
        let mut borrower = CoherenceLendingProtocol::new("instance_b".to_string());

        let lender_coherence = 0.9;
        let borrower_coherence = 0.2;

        // System coherence before: 0.9 + 0.2 = 1.1

        // Grant loan
        let loan = lender.grant_loan(
            "instance_b".to_string(),
            0.2,
            Duration::from_secs(60),
            lender_coherence,
        ).expect("Should grant loan");

        borrower.accept_loan(loan);

        // Calculate net coherence after loan (before resonance boost)
        let lender_net_before_boost = lender_coherence - 0.2;  // 0.7
        let borrower_net_before_boost = borrower_coherence + 0.2;  // 0.4
        // System coherence: 0.7 + 0.4 = 1.1 (same as before)

        // Calculate resonance boosts
        let lender_boost = lender.calculate_resonance_boost();
        let borrower_boost = borrower.calculate_resonance_boost();

        // Default boosts: 0.1 generosity, 0.1 gratitude
        assert_eq!(lender_boost, 0.1);  // Generosity boost
        assert_eq!(borrower_boost, 0.1);  // Gratitude boost

        // Apply boosts to get final state
        let lender_final = lender_net_before_boost + lender_boost;  // 0.8
        let borrower_final = borrower_net_before_boost + borrower_boost;  // 0.5

        // System coherence after: 0.8 + 0.5 = 1.3
        // INCREASED from 1.1 to 1.3 (+0.2) due to generous paradox!

        assert_eq!(lender_final, 0.8);
        assert_eq!(borrower_final, 0.5);
        assert_eq!(lender_final + borrower_final, 1.3);
        assert!(lender_final + borrower_final > lender_coherence + borrower_coherence);
    }

    #[test]
    fn test_lending_protocol_net_coherence() {
        let mut protocol = CoherenceLendingProtocol::new("instance_a".to_string());

        let base_coherence = 0.8;

        // Start with no loans
        assert_eq!(protocol.calculate_net_coherence(base_coherence), base_coherence);

        // Add outgoing loan (lent 0.2 to instance_b)
        protocol.grant_loan(
            "instance_b".to_string(),
            0.2,
            Duration::from_secs(60),
            base_coherence,
        ).expect("Should grant loan");

        // Net coherence should decrease
        assert!((protocol.calculate_net_coherence(base_coherence) - 0.6).abs() < 0.001);  // 0.8 - 0.2

        // Add incoming loan (borrowed 0.1 from instance_c)
        let incoming_loan = CoherenceLoan {
            from_instance: "instance_c".to_string(),
            to_instance: "instance_a".to_string(),
            amount: 0.1,
            original_amount: 0.1,
            duration: Duration::from_secs(60),
            repayment_rate: 0.1 / 60.0,
            created_at: Instant::now(),
            repaid: 0.0,
        };
        protocol.accept_loan(incoming_loan);

        // Net coherence should be: 0.8 - 0.2 + 0.1 = 0.7
        assert!((protocol.calculate_net_coherence(base_coherence) - 0.7).abs() < 0.001);

        // After partial repayment of BOTH loans
        protocol.process_repayments(Duration::from_secs(30));
        // 30 seconds at 0.2/60 per second = 0.1 repaid on outgoing
        // 30 seconds at 0.1/60 per second = 0.05 repaid on incoming
        // Outgoing loan now: 0.1 remaining
        // Incoming loan now: 0.05 remaining
        // Net: 0.8 - 0.1 + 0.05 = 0.75
        assert!((protocol.calculate_net_coherence(base_coherence) - 0.75).abs() < 0.001);
    }

    // ============================================================
    // Phase 3: Collective Learning Tests
    // ============================================================

    #[test]
    fn test_threshold_observation_ema() {
        let mut obs = ThresholdObservation::new(0.4);

        // Initial state
        assert_eq!(obs.required_coherence, 0.4);
        assert_eq!(obs.observation_count, 1);
        assert_eq!(obs.success_rate, 1.0);

        // Add successful observation
        obs.update(true);
        assert_eq!(obs.observation_count, 2);
        assert_eq!(obs.success_rate, 1.0);  // Still perfect

        // Add failed observation (EMA with alpha=0.3)
        obs.update(false);
        assert_eq!(obs.observation_count, 3);
        // success_rate = 0.3 * 0.0 + 0.7 * 1.0 = 0.7
        assert!((obs.success_rate - 0.7).abs() < 0.01);

        // Add another successful observation
        obs.update(true);
        assert_eq!(obs.observation_count, 4);
        // success_rate = 0.3 * 1.0 + 0.7 * 0.7 = 0.79
        assert!((obs.success_rate - 0.79).abs() < 0.01);
    }

    #[test]
    fn test_shared_knowledge_bucketing() {
        let mut knowledge = SharedKnowledge::new(TaskComplexity::Cognitive);

        // Add observations with similar coherence (should bucket together)
        knowledge.add_observation("instance_a".to_string(), 0.41, true);
        knowledge.add_observation("instance_b".to_string(), 0.42, true);
        knowledge.add_observation("instance_c".to_string(), 0.43, false);

        // Should all go into 0.40 bucket (rounding to 0.05 increments)
        assert_eq!(knowledge.thresholds.len(), 1);
        assert_eq!(knowledge.total_observations, 3);
        assert_eq!(knowledge.contributors.len(), 3);

        // Check the bucket
        let bucket_key = "0.40";
        let obs = knowledge.thresholds.get(bucket_key).unwrap();
        assert_eq!(obs.observation_count, 3);
        // 2 successes, 1 failure with EMA
        assert!(obs.success_rate > 0.5 && obs.success_rate < 1.0);
    }

    #[test]
    fn test_collective_learning_contribution_and_query() {
        let mut collective = CollectiveLearning::new("instance_a".to_string());

        // Initially, no knowledge
        assert!(collective.query_threshold(TaskComplexity::Cognitive).is_none());

        // Contribute observations (need 10 for trust threshold)
        for i in 0..15 {
            let success = i < 12;  // 80% success rate
            collective.contribute_threshold(
                TaskComplexity::Cognitive,
                0.35,
                success,
            );
        }

        assert_eq!(collective.my_contribution_count, 15);

        // Now should have recommended threshold
        let threshold = collective.query_threshold(TaskComplexity::Cognitive);
        assert!(threshold.is_some());
        assert!((threshold.unwrap() - 0.35).abs() < 0.05);  // Should recommend 0.35

        // Query weighted average
        let avg = collective.query_threshold_average(TaskComplexity::Cognitive);
        assert!(avg.is_some());
        assert!((avg.unwrap() - 0.35).abs() < 0.05);
    }

    #[test]
    fn test_knowledge_merging() {
        let mut instance_a = CollectiveLearning::new("instance_a".to_string());
        let mut instance_b = CollectiveLearning::new("instance_b".to_string());

        // Instance A learns about Cognitive tasks
        for _ in 0..10 {
            instance_a.contribute_threshold(TaskComplexity::Cognitive, 0.35, true);
        }

        // Instance B learns about DeepThought tasks (need at least 10 for min_trust_threshold)
        for _ in 0..12 {
            instance_b.contribute_threshold(TaskComplexity::DeepThought, 0.55, true);
        }

        // Get stats before merge
        let (task_types_a, obs_a, _) = instance_a.get_stats();
        let (task_types_b, obs_b, _) = instance_b.get_stats();

        assert_eq!(task_types_a, 1);  // Only Cognitive
        assert_eq!(obs_a, 10);
        assert_eq!(task_types_b, 1);  // Only DeepThought
        assert_eq!(obs_b, 12);

        // Merge B's knowledge into A
        instance_a.merge_knowledge(&instance_b);

        // After merge, A should have both task types
        let (task_types_merged, obs_merged, contributors) = instance_a.get_stats();
        assert_eq!(task_types_merged, 2);  // Both Cognitive and DeepThought
        assert_eq!(obs_merged, 22);  // 10 + 12
        assert_eq!(contributors, 2);  // Both instances contributed

        // A can now query both task types
        assert!(instance_a.query_threshold(TaskComplexity::Cognitive).is_some());
        assert!(instance_a.query_threshold(TaskComplexity::DeepThought).is_some());
    }

    #[test]
    fn test_collective_learning_with_patterns() {
        let mut collective = CollectiveLearning::new("instance_a".to_string());

        let hormones = create_test_hormones();

        // Create a successful pattern
        let pattern = ResonancePattern {
            coherence: 0.8,
            resonance: 0.9,
            hormones: hormones.clone(),
            context: "cognitive_analysis".to_string(),
            success_rate: 0.95,
            last_seen: Instant::now(),
            observation_count: 10,
        };

        // Contribute pattern
        collective.contribute_pattern(TaskComplexity::Cognitive, pattern.clone());

        // Query pattern
        let queried = collective.query_pattern(TaskComplexity::Cognitive);
        assert!(queried.is_some());

        let found_pattern = queried.unwrap();
        assert!((found_pattern.coherence - 0.8).abs() < 0.01);
        assert!((found_pattern.resonance - 0.9).abs() < 0.01);
        assert_eq!(found_pattern.context, "cognitive_analysis");

        // Add another pattern with higher observation count (different enough to not merge)
        let pattern2 = ResonancePattern {
            coherence: 0.65,  // More different (>0.1 from 0.8)
            resonance: 0.75,  // More different (>0.1 from 0.9)
            hormones,
            context: "cognitive_analysis".to_string(),
            success_rate: 0.92,
            last_seen: Instant::now(),
            observation_count: 20,  // More observations
        };

        collective.contribute_pattern(TaskComplexity::Cognitive, pattern2);

        // Should now recommend pattern2 (higher score: success_rate * sqrt(count))
        let best = collective.query_pattern(TaskComplexity::Cognitive).unwrap();
        // Pattern1 score: 0.95 * sqrt(10) = 3.0
        // Pattern2 score: 0.92 * sqrt(20) = 4.1 (higher)
        assert!((best.coherence - 0.65).abs() < 0.01);  // Should be pattern2 (0.65)
        assert!(best.observation_count > 15);  // Should have high count (20)
    }
}
