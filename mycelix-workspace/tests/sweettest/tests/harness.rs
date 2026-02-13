//! Shared test harness for Mycelix sweettest integration tests.
//!
//! Provides conductor setup, DNA loading, and helper utilities
//! that are common across all hApp test suites.
//!
//! Updated for Holochain 0.6 sweettest API.

use holochain::prelude::*;
use holochain::sweettest::*;
use std::path::PathBuf;

/// Known hApp DNA paths relative to the workspace root.
pub struct DnaPaths;

impl DnaPaths {
    /// Get the workspace root directory.
    pub fn workspace_root() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
    }

    pub fn identity() -> PathBuf {
        Self::workspace_root().join("../mycelix-identity/dna/mycelix_identity_dna.dna")
    }

    pub fn governance() -> PathBuf {
        Self::workspace_root().join("../mycelix-governance/dna/mycelix_governance.dna")
    }

    pub fn finance() -> PathBuf {
        Self::workspace_root().join("../mycelix-finance/dna/mycelix_finance.dna")
    }

    pub fn edunet() -> PathBuf {
        Self::workspace_root().join("../mycelix-edunet/dna/edunet.dna")
    }

    pub fn supplychain() -> PathBuf {
        Self::workspace_root().join("../mycelix-supplychain/holochain/dna/supplychain.dna")
    }

    pub fn health() -> PathBuf {
        Self::workspace_root().join("../mycelix-health/dna/health.dna")
    }

    pub fn marketplace() -> PathBuf {
        Self::workspace_root().join("../mycelix-marketplace/backend/mycelix_marketplace.dna")
    }

    pub fn climate() -> PathBuf {
        Self::workspace_root().join("../mycelix-climate/dnas/climate/workdir/climate.dna")
    }

    pub fn federated_learning() -> PathBuf {
        Self::workspace_root().join("../mycelix-core/zomes/federated_learning/workdir/dna/federated_learning.dna")
    }

    /// Commons cluster DNA (property + housing + care + mutualaid + water)
    pub fn commons() -> PathBuf {
        Self::workspace_root().join("../mycelix-commons/dna/mycelix_commons.dna")
    }

    /// Civic cluster DNA (justice + emergency + media)
    pub fn civic() -> PathBuf {
        Self::workspace_root().join("../mycelix-civic/dna/mycelix_civic.dna")
    }
}

/// A test agent with conductor and cell references for Holochain 0.6 API.
pub struct TestAgent {
    pub conductor: SweetConductor,
    pub cell: SweetCell,
    pub agent_pubkey: AgentPubKey,
}

impl TestAgent {
    /// Get a SweetZome reference for making calls.
    pub fn zome(&self, zome_name: &str) -> SweetZome {
        self.cell.zome(zome_name)
    }

    /// Make a zome call with the new Holochain 0.6 API.
    pub async fn call_zome_fn<I, O>(&self, zome_name: &str, fn_name: &str, input: I) -> O
    where
        I: serde::Serialize + std::fmt::Debug,
        O: serde::de::DeserializeOwned + std::fmt::Debug,
    {
        let zome = self.zome(zome_name);
        self.conductor.call(&zome, fn_name, input).await
    }

    /// Make a zome call that may fail, returning the ConductorApiResult.
    pub async fn call_zome_fn_fallible<I, O>(
        &self,
        zome_name: &str,
        fn_name: &str,
        input: I,
    ) -> Result<O, holochain::conductor::api::error::ConductorApiError>
    where
        I: serde::Serialize + std::fmt::Debug,
        O: serde::de::DeserializeOwned + std::fmt::Debug,
    {
        let zome = self.zome(zome_name);
        self.conductor.call_fallible(&zome, fn_name, input).await
    }
}

/// Set up N test agents sharing a DNA, with peer exchange for DHT sync.
/// Returns TestAgent structs with conductor, cell, and agent_pubkey.
pub async fn setup_test_agents(
    dna_path: &PathBuf,
    app_name: &str,
    n: usize,
) -> Vec<TestAgent> {
    let dna = SweetDnaFile::from_bundle(dna_path)
        .await
        .unwrap_or_else(|e| panic!("Failed to load DNA from {:?}: {:?}", dna_path, e));

    let mut agents = Vec::with_capacity(n);

    for _ in 0..n {
        let mut conductor = SweetConductor::from_standard_config().await;
        let app = conductor
            .setup_app(app_name, &[dna.clone()])
            .await
            .unwrap();

        let cell = app.cells()[0].clone();
        let agent_pubkey = cell.agent_pubkey().clone();

        agents.push(TestAgent {
            conductor,
            cell,
            agent_pubkey,
        });
    }

    // Connect all conductors for DHT gossip
    if n > 1 {
        let conductor_refs: Vec<&SweetConductor> = agents.iter().map(|a| &a.conductor).collect();
        SweetConductor::exchange_peer_info(conductor_refs).await;
    }

    agents
}

/// Wait for DHT propagation between agents.
/// Sweettest conductors gossip directly, but propagation still takes time,
/// especially with 3+ conductors.
pub async fn wait_for_dht_sync() {
    tokio::time::sleep(std::time::Duration::from_secs(2)).await;
}
