//! Credit system integration for Holochain.
//!
//! Manages credit issuance and tracking for nodes participating
//! in federated learning. Credits are earned through:
//! - Quality gradient submissions (based on PoGQ score)
//! - Byzantine behavior detection
//! - Peer validation activities
//! - Network uptime/contribution

use std::collections::HashMap;
use std::sync::Arc;

use tokio::sync::RwLock;
use tracing::{debug, info, warn};

use super::client::HolochainClient;
use super::types::*;

/// Reason for earning credits.
#[derive(Clone, Debug)]
pub enum CreditReason {
    /// Quality gradient submission.
    GradientQuality {
        pogq_score: f32,
        gradient_hash: String,
    },
    /// Detected Byzantine behavior in another node.
    ByzantineDetection {
        caught_node: String,
        evidence_hash: String,
    },
    /// Validated peer's gradient.
    PeerValidation {
        validated_node: String,
        gradient_hash: String,
    },
    /// Network contribution bonus.
    NetworkContribution { uptime_hours: u64 },
}

impl From<CreditReason> for EarnReason {
    fn from(reason: CreditReason) -> Self {
        match reason {
            CreditReason::GradientQuality {
                pogq_score,
                gradient_hash,
            } => EarnReason::QualityGradient {
                pogq_score,
                gradient_hash,
            },
            CreditReason::ByzantineDetection {
                caught_node,
                evidence_hash,
            } => EarnReason::ByzantineDetection {
                caught_node_id: caught_node,
                evidence_hash,
            },
            CreditReason::PeerValidation {
                validated_node,
                gradient_hash,
            } => EarnReason::PeerValidation {
                validated_node_id: validated_node,
                gradient_hash,
            },
            CreditReason::NetworkContribution { uptime_hours } => {
                EarnReason::NetworkContribution { uptime_hours }
            }
        }
    }
}

/// Credit configuration.
#[derive(Clone, Debug)]
pub struct CreditConfig {
    /// Base credits for gradient submission.
    pub base_gradient_credits: u64,
    /// Multiplier for PoGQ score (0-1 range).
    pub pogq_multiplier: f32,
    /// Credits for detecting Byzantine behavior.
    pub byzantine_detection_credits: u64,
    /// Credits for peer validation.
    pub peer_validation_credits: u64,
    /// Credits per hour of uptime.
    pub uptime_credits_per_hour: u64,
    /// Minimum PoGQ score to earn any credits.
    pub min_pogq_threshold: f32,
    /// PoGQ score threshold for bonus credits.
    pub bonus_pogq_threshold: f32,
    /// Bonus credits for high PoGQ scores.
    pub high_quality_bonus: u64,
}

impl Default for CreditConfig {
    fn default() -> Self {
        Self {
            base_gradient_credits: 10,
            pogq_multiplier: 2.0,
            byzantine_detection_credits: 50,
            peer_validation_credits: 5,
            uptime_credits_per_hour: 1,
            min_pogq_threshold: 0.3,
            bonus_pogq_threshold: 0.8,
            high_quality_bonus: 20,
        }
    }
}

/// Credit manager for tracking and issuing credits.
pub struct CreditManager {
    config: CreditConfig,
    client: Arc<HolochainClient>,
    /// Local cache of balances for quick lookups.
    balance_cache: Arc<RwLock<HashMap<String, u64>>>,
    /// Pending credits not yet committed to DHT.
    pending_credits: Arc<RwLock<Vec<PendingCredit>>>,
}

/// Pending credit to be issued.
#[derive(Clone, Debug)]
struct PendingCredit {
    holder: String,
    amount: u64,
    reason: EarnReason,
}

impl CreditManager {
    /// Create a new credit manager.
    pub fn new(client: Arc<HolochainClient>, config: CreditConfig) -> Self {
        Self {
            config,
            client,
            balance_cache: Arc::new(RwLock::new(HashMap::new())),
            pending_credits: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Create with default config.
    pub fn with_defaults(client: Arc<HolochainClient>) -> Self {
        Self::new(client, CreditConfig::default())
    }

    /// Calculate credits for a gradient submission based on PoGQ score.
    pub fn calculate_gradient_credits(&self, pogq_score: f32) -> u64 {
        if pogq_score < self.config.min_pogq_threshold {
            return 0;
        }

        let base = self.config.base_gradient_credits;
        let quality_bonus = (pogq_score * self.config.pogq_multiplier * base as f32) as u64;
        let high_quality_bonus = if pogq_score >= self.config.bonus_pogq_threshold {
            self.config.high_quality_bonus
        } else {
            0
        };

        base + quality_bonus + high_quality_bonus
    }

    /// Issue credits for a quality gradient submission.
    pub async fn credit_gradient_submission(
        &self,
        holder: &str,
        pogq_score: f32,
        gradient_hash: &str,
    ) -> HolochainResult<u64> {
        let amount = self.calculate_gradient_credits(pogq_score);

        if amount == 0 {
            debug!(
                holder = %holder,
                pogq = %pogq_score,
                "No credits issued - PoGQ below threshold"
            );
            return Ok(0);
        }

        let reason = CreditReason::GradientQuality {
            pogq_score,
            gradient_hash: gradient_hash.to_string(),
        };

        self.issue_credits(holder, amount, reason).await?;
        Ok(amount)
    }

    /// Issue credits for detecting Byzantine behavior.
    pub async fn credit_byzantine_detection(
        &self,
        holder: &str,
        caught_node: &str,
        evidence_hash: &str,
    ) -> HolochainResult<u64> {
        let amount = self.config.byzantine_detection_credits;
        let reason = CreditReason::ByzantineDetection {
            caught_node: caught_node.to_string(),
            evidence_hash: evidence_hash.to_string(),
        };

        self.issue_credits(holder, amount, reason).await?;
        Ok(amount)
    }

    /// Issue credits for peer validation.
    pub async fn credit_peer_validation(
        &self,
        holder: &str,
        validated_node: &str,
        gradient_hash: &str,
    ) -> HolochainResult<u64> {
        let amount = self.config.peer_validation_credits;
        let reason = CreditReason::PeerValidation {
            validated_node: validated_node.to_string(),
            gradient_hash: gradient_hash.to_string(),
        };

        self.issue_credits(holder, amount, reason).await?;
        Ok(amount)
    }

    /// Issue credits for network contribution.
    pub async fn credit_network_contribution(
        &self,
        holder: &str,
        uptime_hours: u64,
    ) -> HolochainResult<u64> {
        let amount = uptime_hours * self.config.uptime_credits_per_hour;
        let reason = CreditReason::NetworkContribution { uptime_hours };

        self.issue_credits(holder, amount, reason).await?;
        Ok(amount)
    }

    /// Issue credits to a holder.
    async fn issue_credits(
        &self,
        holder: &str,
        amount: u64,
        reason: CreditReason,
    ) -> HolochainResult<EntryHash> {
        info!(
            holder = %holder,
            amount = %amount,
            "Issuing credits"
        );

        let earn_reason: EarnReason = reason.into();
        let hash = self.client.issue_credit(holder, amount, earn_reason).await?;

        // Update local cache
        {
            let mut cache = self.balance_cache.write().await;
            let balance = cache.entry(holder.to_string()).or_insert(0);
            *balance += amount;
        }

        Ok(hash)
    }

    /// Queue credits for batch issuance.
    pub async fn queue_credits(&self, holder: &str, amount: u64, reason: CreditReason) {
        let pending = PendingCredit {
            holder: holder.to_string(),
            amount,
            reason: reason.into(),
        };

        self.pending_credits.write().await.push(pending);
    }

    /// Flush all pending credits to the DHT.
    pub async fn flush_pending(&self) -> HolochainResult<usize> {
        let pending: Vec<PendingCredit> = {
            let mut credits = self.pending_credits.write().await;
            std::mem::take(&mut *credits)
        };

        let count = pending.len();
        if count == 0 {
            return Ok(0);
        }

        info!(count = %count, "Flushing pending credits");

        for credit in pending {
            let credit_clone = credit.clone();
            if let Err(e) = self
                .client
                .issue_credit(&credit.holder, credit.amount, credit.reason)
                .await
            {
                warn!(
                    holder = %credit_clone.holder,
                    error = %e,
                    "Failed to issue credit"
                );
                // Re-queue failed credit
                self.pending_credits.write().await.push(credit_clone);
            }
        }

        Ok(count)
    }

    /// Get balance for a holder (cached or from DHT).
    pub async fn get_balance(&self, holder: &str) -> HolochainResult<u64> {
        // Check cache first
        {
            let cache = self.balance_cache.read().await;
            if let Some(&balance) = cache.get(holder) {
                return Ok(balance);
            }
        }

        // Fetch from DHT
        let balance = self.client.get_credit_balance(holder).await?;

        // Update cache
        {
            let mut cache = self.balance_cache.write().await;
            cache.insert(holder.to_string(), balance);
        }

        Ok(balance)
    }

    /// Refresh balance from DHT (bypass cache).
    pub async fn refresh_balance(&self, holder: &str) -> HolochainResult<u64> {
        let balance = self.client.get_credit_balance(holder).await?;

        {
            let mut cache = self.balance_cache.write().await;
            cache.insert(holder.to_string(), balance);
        }

        Ok(balance)
    }

    /// Get credit history for a holder.
    pub async fn get_history(&self, holder: &str) -> HolochainResult<Vec<CreditRecord>> {
        self.client.get_credit_history(holder).await
    }

    /// Get all cached balances.
    pub async fn get_all_cached_balances(&self) -> HashMap<String, u64> {
        self.balance_cache.read().await.clone()
    }

    /// Clear local cache.
    pub async fn clear_cache(&self) {
        self.balance_cache.write().await.clear();
    }

    /// Get number of pending credits.
    pub async fn pending_count(&self) -> usize {
        self.pending_credits.read().await.len()
    }
}

/// Credit leaderboard entry.
#[derive(Clone, Debug)]
pub struct LeaderboardEntry {
    pub holder: String,
    pub balance: u64,
    pub rank: usize,
}

/// Build a leaderboard from cached balances.
pub fn build_leaderboard(balances: &HashMap<String, u64>) -> Vec<LeaderboardEntry> {
    let mut entries: Vec<_> = balances
        .iter()
        .map(|(holder, &balance)| (holder.clone(), balance))
        .collect();

    entries.sort_by(|a, b| b.1.cmp(&a.1)); // Sort descending by balance

    entries
        .into_iter()
        .enumerate()
        .map(|(rank, (holder, balance))| LeaderboardEntry {
            holder,
            balance,
            rank: rank + 1,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_credit_calculation_below_threshold() {
        let config = CreditConfig::default();
        let client = Arc::new(HolochainClient::new(
            super::super::client::HolochainConfig::default(),
        ));
        let manager = CreditManager::new(client, config);

        // Below minimum threshold (0.3)
        assert_eq!(manager.calculate_gradient_credits(0.1), 0);
        assert_eq!(manager.calculate_gradient_credits(0.29), 0);
    }

    #[test]
    fn test_credit_calculation_normal() {
        let config = CreditConfig {
            base_gradient_credits: 10,
            pogq_multiplier: 2.0,
            min_pogq_threshold: 0.3,
            bonus_pogq_threshold: 0.8,
            high_quality_bonus: 20,
            ..Default::default()
        };
        let client = Arc::new(HolochainClient::new(
            super::super::client::HolochainConfig::default(),
        ));
        let manager = CreditManager::new(client, config);

        // At threshold (0.3): base + quality = 10 + (0.3 * 2.0 * 10) = 10 + 6 = 16
        let credits = manager.calculate_gradient_credits(0.3);
        assert_eq!(credits, 16);

        // Mid-range (0.5): 10 + (0.5 * 2.0 * 10) = 10 + 10 = 20
        let credits = manager.calculate_gradient_credits(0.5);
        assert_eq!(credits, 20);
    }

    #[test]
    fn test_credit_calculation_high_quality() {
        let config = CreditConfig {
            base_gradient_credits: 10,
            pogq_multiplier: 2.0,
            min_pogq_threshold: 0.3,
            bonus_pogq_threshold: 0.8,
            high_quality_bonus: 20,
            ..Default::default()
        };
        let client = Arc::new(HolochainClient::new(
            super::super::client::HolochainConfig::default(),
        ));
        let manager = CreditManager::new(client, config);

        // High quality (0.9): base + quality + bonus = 10 + 18 + 20 = 48
        let credits = manager.calculate_gradient_credits(0.9);
        assert_eq!(credits, 48);

        // Perfect (1.0): 10 + 20 + 20 = 50
        let credits = manager.calculate_gradient_credits(1.0);
        assert_eq!(credits, 50);
    }

    #[test]
    fn test_credit_reason_conversion() {
        let reason = CreditReason::GradientQuality {
            pogq_score: 0.9,
            gradient_hash: "hash123".to_string(),
        };

        let earn_reason: EarnReason = reason.into();
        match earn_reason {
            EarnReason::QualityGradient {
                pogq_score,
                gradient_hash,
            } => {
                assert!((pogq_score - 0.9).abs() < f32::EPSILON);
                assert_eq!(gradient_hash, "hash123");
            }
            _ => panic!("Wrong variant"),
        }
    }

    #[test]
    fn test_leaderboard() {
        let mut balances = HashMap::new();
        balances.insert("node_a".to_string(), 100);
        balances.insert("node_b".to_string(), 500);
        balances.insert("node_c".to_string(), 250);

        let leaderboard = build_leaderboard(&balances);

        assert_eq!(leaderboard.len(), 3);
        assert_eq!(leaderboard[0].holder, "node_b");
        assert_eq!(leaderboard[0].rank, 1);
        assert_eq!(leaderboard[0].balance, 500);

        assert_eq!(leaderboard[1].holder, "node_c");
        assert_eq!(leaderboard[1].rank, 2);
        assert_eq!(leaderboard[1].balance, 250);

        assert_eq!(leaderboard[2].holder, "node_a");
        assert_eq!(leaderboard[2].rank, 3);
        assert_eq!(leaderboard[2].balance, 100);
    }

    #[tokio::test]
    async fn test_pending_credits() {
        let client = Arc::new(HolochainClient::new(
            super::super::client::HolochainConfig::default(),
        ));
        let manager = CreditManager::with_defaults(client);

        assert_eq!(manager.pending_count().await, 0);

        manager
            .queue_credits(
                "node_1",
                100,
                CreditReason::GradientQuality {
                    pogq_score: 0.9,
                    gradient_hash: "h1".to_string(),
                },
            )
            .await;

        manager
            .queue_credits(
                "node_2",
                50,
                CreditReason::PeerValidation {
                    validated_node: "node_1".to_string(),
                    gradient_hash: "h1".to_string(),
                },
            )
            .await;

        assert_eq!(manager.pending_count().await, 2);
    }
}
