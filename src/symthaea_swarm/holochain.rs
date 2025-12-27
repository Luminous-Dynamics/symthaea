// symthaea_swarm/holochain.rs - HTTP-based Holochain conductor client
//
// This module implements the Mycelix Protocol client traits using
// Holochain conductor's HTTP API.

use super::api::*;
use serde_json::json;
use std::sync::Arc;
use std::time::Duration;

/// HTTP-based Holochain Swarm Client
pub struct HolochainSwarmClient {
    /// Holochain conductor HTTP endpoint (e.g., "http://localhost:8888")
    conductor_url: String,

    /// DKG hApp cell ID
    dkg_cell_id: String,

    /// MATL service endpoint
    matl_url: String,

    /// MFDI service endpoint
    mfdi_url: String,

    /// This instance's DID
    my_did: Did,

    /// HTTP client
    http_client: reqwest::Client,
}

impl HolochainSwarmClient {
    /// Create new Holochain swarm client
    pub fn new(
        conductor_url: String,
        dkg_cell_id: String,
        matl_url: String,
        mfdi_url: String,
        my_did: Did,
    ) -> Result<Self, SwarmError> {
        let http_client = reqwest::Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .map_err(|e| SwarmError::Network(e.to_string()))?;

        Ok(Self {
            conductor_url,
            dkg_cell_id,
            matl_url,
            mfdi_url,
            my_did,
            http_client,
        })
    }

    /// Call a Holochain zome function
    async fn call_zome(
        &self,
        zome_name: &str,
        fn_name: &str,
        payload: serde_json::Value,
    ) -> Result<serde_json::Value, SwarmError> {
        let url = format!("{}/call_zome", self.conductor_url);

        let request_body = json!({
            "cell_id": self.dkg_cell_id,
            "zome_name": zome_name,
            "fn_name": fn_name,
            "payload": payload,
        });

        let response = self
            .http_client
            .post(&url)
            .json(&request_body)
            .send()
            .await
            .map_err(|e| SwarmError::Http(e.to_string()))?;

        if !response.status().is_success() {
            return Err(SwarmError::Http(format!(
                "HTTP error: {}",
                response.status()
            )));
        }

        let result: serde_json::Value = response
            .json()
            .await
            .map_err(|e| SwarmError::Serialization(e.into()))?;

        Ok(result)
    }

    /// Call MATL REST API
    async fn call_matl(&self, endpoint: &str) -> Result<serde_json::Value, SwarmError> {
        let url = format!("{}{}", self.matl_url, endpoint);

        let response = self
            .http_client
            .get(&url)
            .send()
            .await
            .map_err(|e| SwarmError::Http(e.to_string()))?;

        if !response.status().is_success() {
            return Err(SwarmError::Http(format!(
                "MATL HTTP error: {}",
                response.status()
            )));
        }

        let result: serde_json::Value = response
            .json()
            .await
            .map_err(|e| SwarmError::Serialization(e.into()))?;

        Ok(result)
    }

    /// Call MFDI REST API
    async fn call_mfdi(
        &self,
        method: reqwest::Method,
        endpoint: &str,
        body: Option<serde_json::Value>,
    ) -> Result<serde_json::Value, SwarmError> {
        let url = format!("{}{}", self.mfdi_url, endpoint);

        let mut request = self.http_client.request(method, &url);

        if let Some(body) = body {
            request = request.json(&body);
        }

        let response = request
            .send()
            .await
            .map_err(|e| SwarmError::Http(e.to_string()))?;

        if !response.status().is_success() {
            return Err(SwarmError::Http(format!(
                "MFDI HTTP error: {}",
                response.status()
            )));
        }

        let result: serde_json::Value = response
            .json()
            .await
            .map_err(|e| SwarmError::Serialization(e.into()))?;

        Ok(result)
    }
}

// ============================================================================
// DkgClient Implementation
// ============================================================================

#[async_trait::async_trait]
impl DkgClient for HolochainSwarmClient {
    async fn publish_pattern_claim(&self, pattern: SymthaeaPattern) -> Result<ClaimId, SwarmError> {
        let claim = pattern.to_epistemic_claim(&self.my_did);

        let payload = json!({
            "claim": claim,
        });

        let result = self
            .call_zome("dkg", "publish_claim", payload)
            .await?;

        let claim_id_str = result["claim_id"]
            .as_str()
            .ok_or_else(|| SwarmError::InvalidClaim("No claim_id in response".to_string()))?;

        let claim_id = Uuid::parse_str(claim_id_str)
            .map_err(|e| SwarmError::InvalidClaim(format!("Invalid UUID: {}", e)))?;

        Ok(claim_id)
    }

    async fn get_claim(&self, claim_id: ClaimId) -> Result<Option<EpistemicClaim>, SwarmError> {
        let payload = json!({
            "claim_id": claim_id.to_string(),
        });

        let result = self.call_zome("dkg", "get_claim", payload).await?;

        if result.is_null() {
            return Ok(None);
        }

        let claim: EpistemicClaim = serde_json::from_value(result)?;
        Ok(Some(claim))
    }

    async fn query_claims(
        &self,
        query: PatternQuery,
        limit: usize,
    ) -> Result<Vec<EvaluatedClaim>, SwarmError> {
        let payload = json!({
            "query": query,
            "limit": limit,
        });

        let result = self.call_zome("dkg", "query_claims", payload).await?;

        let claims: Vec<EvaluatedClaim> = serde_json::from_value(result)?;
        Ok(claims)
    }
}

// ============================================================================
// MatlClient Implementation
// ============================================================================

#[async_trait::async_trait]
impl MatlClient for HolochainSwarmClient {
    async fn trust_for_claim(&self, claim_id: ClaimId) -> Result<CompositeTrustScore, SwarmError> {
        let endpoint = format!("/trust/claim/{}", claim_id);
        let result = self.call_matl(&endpoint).await?;

        let trust_score: CompositeTrustScore = serde_json::from_value(result)?;
        Ok(trust_score)
    }

    async fn trust_for_agent(&self, did: &Did) -> Result<CompositeTrustScore, SwarmError> {
        let endpoint = format!("/trust/agent/{}", urlencoding::encode(did));
        let result = self.call_matl(&endpoint).await?;

        let trust_score: CompositeTrustScore = serde_json::from_value(result)?;
        Ok(trust_score)
    }

    async fn cartel_risk_for_agent(
        &self,
        did: &Did,
        window: Duration,
    ) -> Result<CartelRisk, SwarmError> {
        let window_secs = window.as_secs();
        let endpoint = format!(
            "/cartel_risk/agent/{}?window={}",
            urlencoding::encode(did),
            window_secs
        );
        let result = self.call_matl(&endpoint).await?;

        let cartel_risk: CartelRisk = serde_json::from_value(result)?;
        Ok(cartel_risk)
    }
}

// ============================================================================
// MfdiClient Implementation
// ============================================================================

#[async_trait::async_trait]
impl MfdiClient for HolochainSwarmClient {
    async fn ensure_instrumental_identity(
        &self,
        model_type: &str,
        model_version: &str,
        operator_did: &Did,
    ) -> Result<MycelixIdentity, SwarmError> {
        let endpoint = "/identity/instrumental_actor";

        let body = json!({
            "model_type": model_type,
            "model_version": model_version,
            "operator_did": operator_did,
            "instance_did": self.my_did,
        });

        let result = self
            .call_mfdi(reqwest::Method::POST, endpoint, Some(body))
            .await?;

        let identity: MycelixIdentity = serde_json::from_value(result)?;
        Ok(identity)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_holochain_client_creation() {
        let client = HolochainSwarmClient::new(
            "http://localhost:8888".to_string(),
            "dkg-cell-123".to_string(),
            "http://localhost:9000".to_string(),
            "http://localhost:9001".to_string(),
            "did:mycelix:symthaea:test".to_string(),
        );

        assert!(client.is_ok());
    }

    #[tokio::test]
    async fn test_pattern_to_claim_conversion() {
        let pattern = SymthaeaPattern {
            pattern_id: Uuid::new_v4(),
            problem_vector: vec![0.1, 0.2, 0.3],
            solution_vector: vec![0.4, 0.5, 0.6],
            success_rate: 0.95,
            context: "webcam not working".to_string(),
            tested_on_nixos: "25.11".to_string(),
            e_tier: EpistemicTierE::E2,
            n_tier: NormativeTierN::N1,
            m_tier: MaterialityTierM::M2,
        };

        let did = "did:mycelix:symthaea:test".to_string();
        let claim = pattern.to_epistemic_claim(&did);

        assert_eq!(claim.submitted_by_did, did);
        assert_eq!(claim.epistemic_tier_e, "E2");
        assert_eq!(claim.epistemic_tier_n, "N1");
        assert_eq!(claim.epistemic_tier_m, "M2");
    }
}
