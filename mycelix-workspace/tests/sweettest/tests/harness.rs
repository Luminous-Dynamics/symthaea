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
    ///
    /// CARGO_MANIFEST_DIR = mycelix-workspace/tests/sweettest
    /// workspace_root     = mycelix-workspace (two levels up)
    pub fn workspace_root() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
    }

    pub fn identity() -> PathBuf {
        Self::workspace_root().join("../mycelix-identity/dna/mycelix_identity_dna.dna")
    }

    pub fn governance() -> PathBuf {
        Self::workspace_root().join("../mycelix-governance/dna/mycelix_governance_dna.dna")
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
        Self::workspace_root()
            .join("../mycelix-core/zomes/federated_learning/workdir/dna/federated_learning.dna")
    }

    /// Commons cluster DNA -- old unified (property + housing + care + mutualaid + water + food + transport).
    /// This is the pre-split DNA. Use `commons_land()` / `commons_care()` for the split sub-cluster DNAs.
    pub fn commons() -> PathBuf {
        Self::workspace_root().join("../mycelix-commons/dna/mycelix_commons.dna")
    }

    /// Commons-Land sub-cluster DNA (property + housing + water + food).
    pub fn commons_land() -> PathBuf {
        Self::workspace_root().join("../mycelix-commons/dna/mycelix_commons_land.dna")
    }

    /// Commons-Care sub-cluster DNA (care + mutualaid + transport + support + space).
    pub fn commons_care() -> PathBuf {
        Self::workspace_root().join("../mycelix-commons/dna/mycelix_commons_care.dna")
    }

    /// Civic cluster DNA (justice + emergency + media)
    pub fn civic() -> PathBuf {
        Self::workspace_root().join("../mycelix-civic/dna/mycelix_civic.dna")
    }

    /// Personal cluster DNA (identity vault + health vault + credential wallet + bridge)
    pub fn personal() -> PathBuf {
        Self::workspace_root().join("../mycelix-personal/dna/mycelix_personal.dna")
    }

    /// Hearth cluster DNA (kinship + gratitude + care + autonomy + stories + rhythms + bridge)
    pub fn hearth() -> PathBuf {
        Self::workspace_root().join("../mycelix-hearth/dna/mycelix_hearth.dna")
    }

    /// Attribution DNA (dependency registry + usage receipts + reciprocity pledges)
    pub fn attribution() -> PathBuf {
        Self::workspace_root().join("../mycelix-attribution/dna/mycelix_attribution_dna.dna")
    }

    /// LUCID hApp DNA (privacy/ZK attestation)
    pub fn lucid() -> PathBuf {
        Self::workspace_root().join("happs/lucid/lucid.dna")
    }

    /// Unified hApp manifest path (for reference; use `unified_happ_dnas()` for sweettest setup).
    pub fn unified_happ() -> PathBuf {
        Self::workspace_root().join("happs/mycelix-unified-happ.yaml")
    }

    /// All DNAs for the unified hApp, in role order.
    /// Returns (role_name, dna_path) pairs matching mycelix-unified-happ.yaml roles.
    pub fn unified_happ_dnas() -> Vec<(&'static str, PathBuf)> {
        vec![
            ("personal", Self::personal()),
            ("identity", Self::identity()),
            ("hearth", Self::hearth()),
            ("commons_land", Self::commons_land()),
            ("commons_care", Self::commons_care()),
            ("civic", Self::civic()),
            ("attribution", Self::attribution()),
            ("finance", Self::finance()),
            ("governance", Self::governance()),
            ("health", Self::health()),
        ]
    }
}

/// A test agent with conductor and cell references for Holochain 0.6 API.
pub struct TestAgent {
    pub conductor: SweetConductor,
    pub cell: SweetCell,
    pub agent_pubkey: AgentPubKey,
    /// Additional cells keyed by role name (for multi-role hApp tests).
    pub role_cells: std::collections::HashMap<String, SweetCell>,
}

impl TestAgent {
    /// Get a SweetZome reference for making calls (default/first cell).
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

    /// Make a zome call on a specific role (for multi-DNA hApp tests).
    ///
    /// Requires that the agent was set up with `setup_test_agents_from_happ()`
    /// which populates `role_cells`.
    pub async fn call_zome_fn_on_role<I, O>(
        &self,
        role_name: &str,
        zome_name: &str,
        fn_name: &str,
        input: I,
    ) -> O
    where
        I: serde::Serialize + std::fmt::Debug,
        O: serde::de::DeserializeOwned + std::fmt::Debug,
    {
        let cell = self.role_cells.get(role_name).unwrap_or_else(|| {
            panic!(
                "No cell found for role '{}'. Available roles: {:?}",
                role_name,
                self.role_cells.keys().collect::<Vec<_>>()
            )
        });
        let zome = cell.zome(zome_name);
        self.conductor.call(&zome, fn_name, input).await
    }

    /// Make a fallible zome call on a specific role.
    #[allow(dead_code)]
    pub async fn call_zome_fn_on_role_fallible<I, O>(
        &self,
        role_name: &str,
        zome_name: &str,
        fn_name: &str,
        input: I,
    ) -> Result<O, holochain::conductor::api::error::ConductorApiError>
    where
        I: serde::Serialize + std::fmt::Debug,
        O: serde::de::DeserializeOwned + std::fmt::Debug,
    {
        let cell = self.role_cells.get(role_name).unwrap_or_else(|| {
            panic!(
                "No cell found for role '{}'. Available roles: {:?}",
                role_name,
                self.role_cells.keys().collect::<Vec<_>>()
            )
        });
        let zome = cell.zome(zome_name);
        self.conductor.call_fallible(&zome, fn_name, input).await
    }
}

/// Set up N test agents sharing a DNA, with peer exchange for DHT sync.
/// Returns TestAgent structs with conductor, cell, and agent_pubkey.
pub async fn setup_test_agents(dna_path: &PathBuf, app_name: &str, n: usize) -> Vec<TestAgent> {
    let dna = SweetDnaFile::from_bundle(dna_path)
        .await
        .unwrap_or_else(|e| panic!("Failed to load DNA from {:?}: {:?}", dna_path, e));

    let mut agents = Vec::with_capacity(n);

    for _ in 0..n {
        let mut conductor = SweetConductor::from_standard_config().await;
        let app = conductor.setup_app(app_name, [&dna]).await.unwrap();

        let cell = app.cells()[0].clone();
        let agent_pubkey = cell.agent_pubkey().clone();

        agents.push(TestAgent {
            conductor,
            cell,
            agent_pubkey,
            role_cells: std::collections::HashMap::new(),
        });
    }

    // Connect all conductors for DHT gossip
    if n > 1 {
        let conductor_refs: Vec<&SweetConductor> = agents.iter().map(|a| &a.conductor).collect();
        SweetConductor::exchange_peer_info(conductor_refs).await;
    }

    agents
}

/// Set up N test agents from the unified hApp (multi-DNA).
///
/// Loads all available DNAs from the unified hApp role list and installs
/// them together in a single conductor so `CallTargetCell::OtherRole` works.
/// Each role's cell is accessible via `TestAgent::call_zome_fn_on_role()`.
///
/// Skips roles whose DNA file has not been built yet (with a warning), so
/// tests can run with a partial set of DNAs available.
///
/// # Arguments
///
/// * `_happ_path` - Path to the hApp YAML manifest (used only for existence check;
///   actual DNAs are loaded from `DnaPaths::unified_happ_dnas()`).
/// * `n` - Number of test agents to create.
#[allow(dead_code)]
pub async fn setup_test_agents_from_happ(_happ_path: &PathBuf, n: usize) -> Vec<TestAgent> {
    let role_dnas = DnaPaths::unified_happ_dnas();

    // Load all available DNAs
    let mut loaded: Vec<(String, DnaFile)> = Vec::new();
    for (role_name, dna_path) in &role_dnas {
        if !dna_path.exists() {
            eprintln!(
                "WARNING: DNA for role '{}' not found at {:?}, skipping",
                role_name, dna_path
            );
            continue;
        }
        let dna = SweetDnaFile::from_bundle(dna_path)
            .await
            .unwrap_or_else(|e| {
                panic!(
                    "Failed to load DNA for role '{}' from {:?}: {:?}",
                    role_name, dna_path, e
                )
            });
        loaded.push((role_name.to_string(), dna));
    }

    assert!(
        !loaded.is_empty(),
        "No DNAs could be loaded for unified hApp. Build at least one DNA first."
    );

    let role_names: Vec<String> = loaded.iter().map(|(name, _)| name.to_string()).collect();

    let mut agents = Vec::with_capacity(n);

    for i in 0..n {
        let mut conductor = SweetConductor::from_standard_config().await;
        let app_name = format!("mycelix-unified-{}", i);
        let app = conductor
            .setup_app(&app_name, &loaded)
            .await
            .unwrap_or_else(|e| panic!("Failed to setup unified hApp: {:?}", e));

        let cells = app.into_cells();
        assert_eq!(
            cells.len(),
            role_names.len(),
            "Cell count should match loaded DNA count"
        );

        let first_cell = cells[0].clone();
        let agent_pubkey = first_cell.agent_pubkey().clone();

        let mut role_cells = std::collections::HashMap::new();
        for (idx, role_name) in role_names.iter().enumerate() {
            role_cells.insert(role_name.clone(), cells[idx].clone());
        }

        agents.push(TestAgent {
            conductor,
            cell: first_cell,
            agent_pubkey,
            role_cells,
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
#[allow(dead_code)]
pub async fn wait_for_dht_sync() {
    tokio::time::sleep(std::time::Duration::from_secs(2)).await;
}
