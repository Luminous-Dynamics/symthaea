# Mycelix Interoperability Standards

## Overview

Mycelix does not exist in isolation. This document defines standards for interoperability with other Holochain applications, Web3 ecosystems, traditional Web2 systems, and legacy institutions. Our goal is to be a good citizen in the broader ecosystem while maintaining our core values.

**Core Principle**: Connected but sovereign—interoperate without dependency.

---

## Interoperability Layers

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    INTEROPERABILITY STACK                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    APPLICATION LAYER                             │   │
│  │  • External UI integrations                                      │   │
│  │  • Third-party app access                                        │   │
│  │  • Widget/embed support                                          │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                │                                        │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    API LAYER                                     │   │
│  │  • REST/GraphQL APIs                                             │   │
│  │  • WebSocket real-time                                           │   │
│  │  • Webhook notifications                                         │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                │                                        │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    PROTOCOL LAYER                                │   │
│  │  • Holochain inter-DNA                                           │   │
│  │  • Cross-chain bridges                                           │   │
│  │  • Federation protocols                                          │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                │                                        │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    IDENTITY LAYER                                │   │
│  │  • DID resolution                                                │   │
│  │  • Credential verification                                       │   │
│  │  • Cross-system identity mapping                                 │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                │                                        │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    DATA LAYER                                    │   │
│  │  • Data format standards                                         │   │
│  │  • Import/export formats                                         │   │
│  │  • Schema mappings                                               │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Identity Interoperability

### DID (Decentralized Identifier) Support

```rust
/// DID method for Mycelix identities
/// did:mycelix:<community>:<agent_hash>
pub struct MycelixDID {
    pub method: String,  // "mycelix"
    pub community: String,
    pub agent_hash: String,
}

impl MycelixDID {
    pub fn to_string(&self) -> String {
        format!("did:mycelix:{}:{}", self.community, self.agent_hash)
    }

    pub fn from_string(did: &str) -> Result<Self, DIDError> {
        let parts: Vec<&str> = did.split(':').collect();
        if parts.len() != 4 || parts[0] != "did" || parts[1] != "mycelix" {
            return Err(DIDError::InvalidFormat);
        }
        Ok(MycelixDID {
            method: "mycelix".into(),
            community: parts[2].into(),
            agent_hash: parts[3].into(),
        })
    }
}

/// DID Document for Mycelix identity
pub struct MycelixDIDDocument {
    pub context: Vec<String>,
    pub id: String,
    pub verification_method: Vec<VerificationMethod>,
    pub authentication: Vec<String>,
    pub assertion_method: Vec<String>,
    pub service: Vec<ServiceEndpoint>,
}

/// Resolve external DIDs
pub async fn resolve_did(did: &str) -> Result<DIDDocument, DIDError> {
    let method = extract_did_method(did)?;

    match method.as_str() {
        "mycelix" => resolve_mycelix_did(did).await,
        "key" => resolve_did_key(did).await,
        "web" => resolve_did_web(did).await,
        "ethr" => resolve_did_ethr(did).await,
        "ion" => resolve_did_ion(did).await,
        "pkh" => resolve_did_pkh(did).await,
        _ => Err(DIDError::UnsupportedMethod(method)),
    }
}
```

### Verifiable Credentials

```rust
/// W3C Verifiable Credential support
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VerifiableCredential {
    #[serde(rename = "@context")]
    pub context: Vec<String>,

    pub id: Option<String>,

    #[serde(rename = "type")]
    pub credential_type: Vec<String>,

    pub issuer: Issuer,

    #[serde(rename = "issuanceDate")]
    pub issuance_date: String,

    #[serde(rename = "expirationDate")]
    pub expiration_date: Option<String>,

    #[serde(rename = "credentialSubject")]
    pub credential_subject: CredentialSubject,

    pub proof: Proof,
}

/// Import external verifiable credential
pub async fn import_credential(
    vc: &VerifiableCredential,
    owner: &AgentPubKey,
) -> Result<ImportedCredential, CredentialError> {
    // Verify credential signature
    let verification = verify_credential_proof(vc).await?;
    if !verification.valid {
        return Err(CredentialError::InvalidProof);
    }

    // Check issuer trust (via Oracle or manual)
    let issuer_trust = check_issuer_trust(&vc.issuer).await?;

    // Store in Attest
    let imported = ImportedCredential {
        original: vc.clone(),
        owner: owner.clone(),
        verification_status: verification,
        issuer_trust,
        imported_at: Timestamp::now(),
    };

    store_imported_credential(&imported)?;

    Ok(imported)
}

/// Export Mycelix credential as W3C VC
pub async fn export_credential(
    mycelix_credential: &MycelixCredential,
) -> Result<VerifiableCredential, CredentialError> {
    let vc = VerifiableCredential {
        context: vec![
            "https://www.w3.org/2018/credentials/v1".into(),
            "https://mycelix.org/credentials/v1".into(),
        ],
        id: Some(format!("urn:mycelix:credential:{}", mycelix_credential.id)),
        credential_type: vec![
            "VerifiableCredential".into(),
            mycelix_credential.credential_type.to_vc_type(),
        ],
        issuer: Issuer {
            id: format!("did:mycelix:{}:{}",
                mycelix_credential.community,
                mycelix_credential.issuer
            ),
        },
        issuance_date: mycelix_credential.issued.to_rfc3339(),
        expiration_date: mycelix_credential.expires.map(|e| e.to_rfc3339()),
        credential_subject: mycelix_credential.to_vc_subject(),
        proof: generate_vc_proof(mycelix_credential).await?,
    };

    Ok(vc)
}
```

### Cross-System Identity Mapping

```rust
/// Link Mycelix identity to external identities
#[hdk_entry_helper]
pub struct IdentityLink {
    pub link_id: String,
    pub mycelix_agent: AgentPubKey,
    pub external_identity: ExternalIdentity,
    pub verification: LinkVerification,
    pub created: Timestamp,
    pub active: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ExternalIdentity {
    // Decentralized
    DID { did: String },
    Ethereum { address: String },
    Bitcoin { address: String },
    Solana { address: String },

    // Federated
    Email { email: String, verified: bool },
    OAuth { provider: String, id: String },

    // Government
    GovernmentID { country: String, id_type: String, hash: String },

    // Professional
    LinkedIn { profile_url: String },
    GitHub { username: String },

    // Domain
    DNSOwnership { domain: String },
}

/// Verify ownership of external identity
pub async fn verify_external_identity(
    identity: &ExternalIdentity,
    proof: &OwnershipProof,
) -> Result<LinkVerification, VerificationError> {
    match identity {
        ExternalIdentity::Ethereum { address } => {
            verify_ethereum_signature(address, proof).await
        }
        ExternalIdentity::DID { did } => {
            verify_did_control(did, proof).await
        }
        ExternalIdentity::Email { email, .. } => {
            verify_email_ownership(email, proof).await
        }
        ExternalIdentity::GitHub { username } => {
            verify_github_ownership(username, proof).await
        }
        ExternalIdentity::DNSOwnership { domain } => {
            verify_dns_txt_record(domain, proof).await
        }
        // ... other verifications
    }
}
```

---

## Holochain Ecosystem Integration

### Inter-DNA Communication

```rust
/// Standard interface for Holochain DNA interoperability
pub trait MycelixInteroperable {
    /// DNA hash for addressing
    fn dna_hash(&self) -> DnaHash;

    /// Supported interop operations
    fn supported_operations(&self) -> Vec<InteropOperation>;

    /// Handle incoming interop call
    fn handle_interop_call(
        &self,
        call: InteropCall,
    ) -> Result<InteropResponse, InteropError>;

    /// Make outgoing interop call
    fn make_interop_call(
        &self,
        target: DnaHash,
        call: InteropCall,
    ) -> Result<InteropResponse, InteropError>;
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct InteropCall {
    pub operation: String,
    pub payload: Vec<u8>,
    pub caller_proof: CallerProof,
    pub response_required: bool,
}

/// Standard operations other Holochain apps can call
pub enum StandardInteropOperation {
    /// Verify an identity claim
    VerifyIdentity { claim: IdentityClaim },

    /// Get trust score
    GetTrustScore { agent: AgentPubKey },

    /// Verify capability
    VerifyCapability { agent: AgentPubKey, capability: String },

    /// Query public data
    QueryPublicData { query: DataQuery },

    /// Submit attestation
    SubmitAttestation { attestation: Attestation },
}

/// Example: Other Holochain app verifying Mycelix identity
#[hdk_extern]
pub fn verify_mycelix_identity(
    mycelix_dna: DnaHash,
    agent: AgentPubKey,
    claim: IdentityClaim,
) -> ExternResult<VerificationResult> {
    let call = InteropCall {
        operation: "verify_identity".into(),
        payload: serialize(&VerifyIdentityRequest { agent, claim })?,
        caller_proof: generate_caller_proof()?,
        response_required: true,
    };

    let response = call_remote(
        mycelix_dna,
        ZomeName::from("attest"),
        FunctionName::from("handle_interop_call"),
        None,
        call,
    )?;

    deserialize(&response)
}
```

### Holochain App Registry

```rust
/// Registry of compatible Holochain applications
#[hdk_entry_helper]
pub struct HolochainAppRegistration {
    pub registration_id: String,
    pub app_name: String,
    pub dna_hash: DnaHash,
    pub version: String,
    pub description: String,

    /// Interop capabilities
    pub capabilities: AppCapabilities,

    /// Trust requirements
    pub trust_requirements: TrustRequirements,

    /// Verification status
    pub verified: bool,
    pub verifier: Option<AgentPubKey>,

    pub registered: Timestamp,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AppCapabilities {
    /// Can this app verify Mycelix identities?
    pub can_verify_identity: bool,

    /// Can this app contribute trust signals?
    pub can_contribute_trust: bool,

    /// Can this app issue credentials?
    pub can_issue_credentials: bool,

    /// Data types this app can share
    pub shareable_data_types: Vec<String>,

    /// Required Mycelix data
    pub required_data_types: Vec<String>,
}
```

---

## Web3/Blockchain Integration

### Ethereum Bridge

```rust
/// Bridge to Ethereum ecosystem
pub struct EthereumBridge {
    pub bridge_id: String,
    pub ethereum_rpc: String,
    pub bridge_contract: String,
    pub supported_operations: Vec<EthOperation>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum EthOperation {
    /// Read Ethereum state
    ReadState { contract: String, method: String },

    /// Verify token ownership
    VerifyTokenOwnership { token_contract: String },

    /// Verify NFT ownership
    VerifyNFTOwnership { nft_contract: String, token_id: String },

    /// Submit attestation to chain
    SubmitAttestation { data: AttestationData },

    /// Bridge assets
    BridgeAsset { asset: String, amount: String, direction: BridgeDirection },
}

/// Verify Ethereum token balance for access control
pub async fn verify_eth_token_balance(
    eth_address: &str,
    token_contract: &str,
    minimum_balance: u64,
) -> Result<bool, BridgeError> {
    let balance = query_token_balance(eth_address, token_contract).await?;
    Ok(balance >= minimum_balance)
}

/// Anchor Mycelix attestation to Ethereum
pub async fn anchor_attestation(
    attestation: &MycelixAttestation,
    eth_signer: &EthSigner,
) -> Result<TxHash, BridgeError> {
    let attestation_hash = hash_attestation(attestation);

    let tx = create_anchor_transaction(
        &attestation_hash,
        attestation.timestamp,
        eth_signer,
    )?;

    let tx_hash = submit_transaction(tx).await?;

    // Store anchor reference
    store_anchor_reference(attestation, &tx_hash)?;

    Ok(tx_hash)
}

/// Import Ethereum-based credentials (e.g., POAPs, badges)
pub async fn import_eth_credential(
    eth_address: &str,
    credential_type: EthCredentialType,
    mycelix_agent: &AgentPubKey,
) -> Result<ImportedCredential, BridgeError> {
    // Verify the Mycelix agent controls the Ethereum address
    verify_eth_address_control(eth_address, mycelix_agent).await?;

    match credential_type {
        EthCredentialType::POAP { event_id } => {
            import_poap(eth_address, event_id, mycelix_agent).await
        }
        EthCredentialType::NFTBadge { contract, token_id } => {
            import_nft_badge(eth_address, contract, token_id, mycelix_agent).await
        }
        EthCredentialType::SBT { contract, token_id } => {
            import_soulbound_token(eth_address, contract, token_id, mycelix_agent).await
        }
    }
}
```

### Multi-Chain Support

```rust
/// Generic multi-chain bridge
pub struct MultiChainBridge {
    pub supported_chains: HashMap<ChainId, ChainConfig>,
    pub message_relayers: Vec<RelayerConfig>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ChainId {
    Ethereum,
    Polygon,
    Arbitrum,
    Optimism,
    Solana,
    Cosmos,
    Polkadot,
    Near,
    Custom(String),
}

/// Cross-chain message
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CrossChainMessage {
    pub message_id: String,
    pub source_chain: ChainId,
    pub target_chain: ChainId,
    pub message_type: CrossChainMessageType,
    pub payload: Vec<u8>,
    pub proof: CrossChainProof,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum CrossChainMessageType {
    /// Identity attestation
    IdentityAttestation,

    /// Credential verification request
    CredentialVerification,

    /// Asset bridge
    AssetBridge,

    /// Governance vote anchoring
    GovernanceAnchor,

    /// Trust score sync
    TrustSync,
}
```

---

## Web2/Traditional System Integration

### REST API

```yaml
# OpenAPI specification for Mycelix external API
openapi: 3.0.0
info:
  title: Mycelix External API
  version: 1.0.0
  description: API for external systems to interact with Mycelix

paths:
  /api/v1/identity/verify:
    post:
      summary: Verify a Mycelix identity
      requestBody:
        content:
          application/json:
            schema:
              type: object
              properties:
                did:
                  type: string
                  description: Mycelix DID
                challenge:
                  type: string
                  description: Challenge to sign
                signature:
                  type: string
                  description: Signature proving control
      responses:
        '200':
          description: Verification result
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/VerificationResult'

  /api/v1/credentials/verify:
    post:
      summary: Verify a credential
      requestBody:
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/VerifiableCredential'
      responses:
        '200':
          description: Credential verification result

  /api/v1/trust/score:
    get:
      summary: Get trust score for an agent
      parameters:
        - name: did
          in: query
          required: true
          schema:
            type: string
      responses:
        '200':
          description: Trust score
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/TrustScore'

  /api/v1/data/export:
    post:
      summary: Export data in standard formats
      requestBody:
        content:
          application/json:
            schema:
              type: object
              properties:
                format:
                  type: string
                  enum: [json, csv, xml, ical, vcf]
                data_type:
                  type: string
                filters:
                  type: object
      responses:
        '200':
          description: Exported data

components:
  schemas:
    VerificationResult:
      type: object
      properties:
        valid:
          type: boolean
        did:
          type: string
        timestamp:
          type: string
          format: date-time

    TrustScore:
      type: object
      properties:
        agent_did:
          type: string
        composite_score:
          type: number
        dimensions:
          type: object
```

### Webhook System

```rust
/// Webhook configuration for external notifications
#[hdk_entry_helper]
pub struct WebhookSubscription {
    pub subscription_id: String,
    pub subscriber: AgentPubKey,
    pub endpoint_url: String,
    pub events: Vec<WebhookEvent>,
    pub secret: String,  // For HMAC signing
    pub active: bool,
    pub created: Timestamp,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum WebhookEvent {
    // Identity events
    IdentityCreated,
    IdentityUpdated,
    CredentialIssued,
    CredentialRevoked,

    // Governance events
    ProposalCreated,
    ProposalVoted,
    ProposalPassed,
    ProposalFailed,

    // Economic events
    TransactionCompleted,
    FundsAllocated,

    // Trust events
    TrustScoreChanged,
    EndorsementReceived,
}

/// Send webhook notification
pub async fn send_webhook(
    subscription: &WebhookSubscription,
    event: &WebhookEvent,
    payload: &serde_json::Value,
) -> Result<WebhookDelivery, WebhookError> {
    let delivery_id = generate_id();

    // Create signed payload
    let timestamp = Timestamp::now();
    let signature = sign_webhook_payload(
        &subscription.secret,
        &timestamp,
        payload,
    );

    let webhook_payload = WebhookPayload {
        delivery_id: delivery_id.clone(),
        event: event.clone(),
        timestamp,
        data: payload.clone(),
    };

    // Send with retry
    let response = send_with_retry(
        &subscription.endpoint_url,
        &webhook_payload,
        &signature,
        3, // max retries
    ).await;

    // Record delivery
    let delivery = WebhookDelivery {
        delivery_id,
        subscription_id: subscription.subscription_id.clone(),
        event: event.clone(),
        timestamp,
        response_status: response.status,
        success: response.success,
    };

    record_delivery(&delivery)?;

    Ok(delivery)
}
```

### OAuth Integration

```rust
/// OAuth provider for Mycelix authentication
pub struct MycelixOAuthProvider {
    pub client_registrations: HashMap<String, OAuthClient>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OAuthClient {
    pub client_id: String,
    pub client_secret_hash: String,
    pub redirect_uris: Vec<String>,
    pub scopes: Vec<OAuthScope>,
    pub name: String,
    pub trusted: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum OAuthScope {
    /// Basic profile (name, public info)
    Profile,

    /// Verify identity claims
    VerifyIdentity,

    /// Read trust score
    TrustScore,

    /// Read credentials
    Credentials,

    /// Read community memberships
    Memberships,

    /// Governance participation
    Governance,
}

/// OAuth authorization flow
pub async fn authorize(
    client_id: &str,
    redirect_uri: &str,
    scopes: &[OAuthScope],
    state: &str,
) -> Result<AuthorizationCode, OAuthError> {
    // Verify client
    let client = get_oauth_client(client_id)?;

    // Verify redirect URI
    if !client.redirect_uris.contains(&redirect_uri.to_string()) {
        return Err(OAuthError::InvalidRedirectUri);
    }

    // Verify scopes
    for scope in scopes {
        if !client.scopes.contains(scope) {
            return Err(OAuthError::InvalidScope);
        }
    }

    // Create authorization code
    let code = AuthorizationCode {
        code: generate_secure_code(),
        client_id: client_id.into(),
        scopes: scopes.to_vec(),
        redirect_uri: redirect_uri.into(),
        expires: Timestamp::now() + Duration::minutes(10),
        state: state.into(),
    };

    store_authorization_code(&code)?;

    Ok(code)
}

/// Exchange code for token
pub async fn token_exchange(
    code: &str,
    client_id: &str,
    client_secret: &str,
) -> Result<OAuthToken, OAuthError> {
    // Verify client
    let client = get_oauth_client(client_id)?;
    verify_client_secret(&client, client_secret)?;

    // Get and validate code
    let auth_code = get_authorization_code(code)?;
    if auth_code.client_id != client_id {
        return Err(OAuthError::InvalidCode);
    }
    if auth_code.expires < Timestamp::now() {
        return Err(OAuthError::ExpiredCode);
    }

    // Generate tokens
    let access_token = generate_access_token(&auth_code)?;
    let refresh_token = generate_refresh_token(&auth_code)?;

    // Invalidate code
    invalidate_authorization_code(code)?;

    Ok(OAuthToken {
        access_token,
        refresh_token,
        token_type: "Bearer".into(),
        expires_in: 3600,
        scopes: auth_code.scopes,
    })
}
```

---

## Data Format Standards

### Import/Export Formats

```rust
/// Standard data export formats
pub enum ExportFormat {
    /// JSON with JSON-LD context
    JsonLd,

    /// CSV for spreadsheet compatibility
    Csv,

    /// iCalendar for events
    ICal,

    /// vCard for contacts
    VCard,

    /// ActivityPub for social data
    ActivityPub,

    /// Custom community format
    Custom { schema_url: String },
}

/// Export community data
pub async fn export_data(
    community: &str,
    data_type: DataType,
    format: ExportFormat,
    filters: ExportFilters,
) -> Result<ExportedData, ExportError> {
    // Check permissions
    verify_export_permission(&filters.requester, community, &data_type)?;

    // Gather data
    let data = gather_export_data(community, &data_type, &filters).await?;

    // Transform to format
    let formatted = match format {
        ExportFormat::JsonLd => to_json_ld(&data)?,
        ExportFormat::Csv => to_csv(&data)?,
        ExportFormat::ICal => to_ical(&data)?,
        ExportFormat::VCard => to_vcard(&data)?,
        ExportFormat::ActivityPub => to_activity_pub(&data)?,
        ExportFormat::Custom { schema_url } => to_custom(&data, &schema_url)?,
    };

    Ok(ExportedData {
        format,
        data: formatted,
        generated: Timestamp::now(),
        checksum: calculate_checksum(&formatted),
    })
}

/// Import data from external sources
pub async fn import_data(
    community: &str,
    source_format: ImportFormat,
    data: &[u8],
    mapping: DataMapping,
) -> Result<ImportResult, ImportError> {
    // Parse incoming data
    let parsed = parse_import_data(source_format, data)?;

    // Validate against mapping
    let validated = validate_import(&parsed, &mapping)?;

    // Transform to Mycelix format
    let transformed = transform_import(&validated, &mapping)?;

    // Import with conflict resolution
    let result = execute_import(community, &transformed).await?;

    Ok(result)
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ImportFormat {
    /// JSON (various schemas)
    Json { schema: Option<String> },

    /// CSV with header mapping
    Csv { delimiter: char, has_header: bool },

    /// iCalendar
    ICal,

    /// vCard
    VCard,

    /// Loomio export
    Loomio,

    /// Slack export
    Slack,

    /// Discord export
    Discord,

    /// Facebook group export
    Facebook,
}
```

### Schema Registry

```rust
/// Registry of data schemas for interoperability
#[hdk_entry_helper]
pub struct SchemaRegistration {
    pub schema_id: String,
    pub name: String,
    pub version: String,
    pub schema_url: String,
    pub schema_definition: serde_json::Value,  // JSON Schema
    pub mappings: Vec<SchemaMapping>,
    pub registered_by: AgentPubKey,
    pub registered: Timestamp,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SchemaMapping {
    pub source_schema: String,
    pub target_schema: String,
    pub field_mappings: Vec<FieldMapping>,
    pub transformation_rules: Vec<TransformationRule>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FieldMapping {
    pub source_field: String,
    pub target_field: String,
    pub transform: Option<String>,  // e.g., "uppercase", "date_format"
    pub required: bool,
    pub default: Option<serde_json::Value>,
}
```

---

## Federation Protocol

### Inter-Community Federation

```rust
/// Federation between Mycelix communities
pub struct FederationProtocol {
    pub protocol_version: String,
    pub supported_operations: Vec<FederationOperation>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum FederationOperation {
    /// Discover other communities
    Discovery,

    /// Establish federation agreement
    Handshake,

    /// Share public data
    DataSync,

    /// Cross-community identity verification
    IdentityVerification,

    /// Cross-community trust attestation
    TrustAttestation,

    /// Cross-community governance
    GovernanceCoordination,

    /// Emergency coordination
    EmergencyAlert,
}

/// Federation agreement between communities
#[hdk_entry_helper]
pub struct FederationAgreement {
    pub agreement_id: String,
    pub community_a: CommunityInfo,
    pub community_b: CommunityInfo,

    /// What's shared
    pub shared_data: Vec<SharedDataType>,

    /// Trust recognition
    pub trust_recognition: TrustRecognitionTerms,

    /// Governance coordination
    pub governance_coordination: Option<GovernanceCoordinationTerms>,

    /// Term and renewal
    pub term: Duration,
    pub auto_renew: bool,

    pub signed_by_a: Signature,
    pub signed_by_b: Signature,
    pub effective_date: Timestamp,
}

/// Federated query across communities
pub async fn federated_query(
    query: FederatedQuery,
    authorized_by: &AgentPubKey,
) -> Result<FederatedQueryResult, FederationError> {
    // Get communities with federation agreements
    let federated_communities = get_federated_communities().await?;

    // Send query to each
    let mut results = Vec::new();
    for community in federated_communities {
        if query.targets.is_empty() || query.targets.contains(&community.id) {
            match send_federated_query(&community, &query).await {
                Ok(result) => results.push(result),
                Err(e) => log::warn!("Query to {} failed: {}", community.id, e),
            }
        }
    }

    // Merge results
    let merged = merge_federated_results(results)?;

    Ok(merged)
}
```

### ActivityPub Integration

```rust
/// ActivityPub integration for social federation
pub struct ActivityPubBridge {
    pub actor_url: String,
    pub inbox_url: String,
    pub outbox_url: String,
}

/// Mycelix as ActivityPub Actor
pub fn get_actor(community: &str, agent: &AgentPubKey) -> ActivityPubActor {
    ActivityPubActor {
        context: vec![
            "https://www.w3.org/ns/activitystreams".into(),
            "https://w3id.org/security/v1".into(),
        ],
        id: format!("https://mycelix.community/{}/users/{}", community, agent),
        actor_type: "Person".into(),
        inbox: format!("https://mycelix.community/{}/users/{}/inbox", community, agent),
        outbox: format!("https://mycelix.community/{}/users/{}/outbox", community, agent),
        followers: format!("https://mycelix.community/{}/users/{}/followers", community, agent),
        following: format!("https://mycelix.community/{}/users/{}/following", community, agent),
        public_key: ActivityPubPublicKey {
            id: format!("https://mycelix.community/{}/users/{}#main-key", community, agent),
            owner: format!("https://mycelix.community/{}/users/{}", community, agent),
            public_key_pem: get_agent_public_key_pem(agent),
        },
    }
}

/// Send ActivityPub activity to followers
pub async fn send_activity(
    activity: ActivityPubActivity,
    recipients: Vec<String>,
) -> Result<(), ActivityPubError> {
    for recipient in recipients {
        let inbox = resolve_inbox(&recipient).await?;
        sign_and_send(&activity, &inbox).await?;
    }
    Ok(())
}

/// Receive ActivityPub activity
pub async fn receive_activity(
    activity: ActivityPubActivity,
) -> Result<(), ActivityPubError> {
    // Verify signature
    verify_activity_signature(&activity).await?;

    // Process based on type
    match activity.activity_type.as_str() {
        "Follow" => handle_follow(&activity).await,
        "Undo" => handle_undo(&activity).await,
        "Create" => handle_create(&activity).await,
        "Announce" => handle_announce(&activity).await,
        "Like" => handle_like(&activity).await,
        _ => Err(ActivityPubError::UnsupportedActivity),
    }
}
```

---

## Security Considerations

### API Security

```rust
/// API security configuration
pub struct APISecurityConfig {
    /// Authentication methods
    pub auth_methods: Vec<AuthMethod>,

    /// Rate limiting
    pub rate_limits: RateLimitConfig,

    /// IP restrictions
    pub ip_restrictions: Option<IPRestrictions>,

    /// TLS requirements
    pub tls_required: bool,
    pub min_tls_version: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum AuthMethod {
    /// API key in header
    ApiKey { header_name: String },

    /// OAuth 2.0 bearer token
    OAuth2Bearer,

    /// Signed request (for high-security)
    SignedRequest { algorithm: String },

    /// mTLS
    MutualTLS,
}

/// Verify API request
pub async fn verify_api_request(
    request: &APIRequest,
    config: &APISecurityConfig,
) -> Result<VerifiedRequest, SecurityError> {
    // Check TLS
    if config.tls_required && !request.is_tls {
        return Err(SecurityError::TLSRequired);
    }

    // Check IP restrictions
    if let Some(restrictions) = &config.ip_restrictions {
        if !restrictions.is_allowed(&request.source_ip) {
            return Err(SecurityError::IPNotAllowed);
        }
    }

    // Check rate limits
    check_rate_limit(&request.source_ip, &config.rate_limits)?;

    // Verify authentication
    let auth_result = verify_authentication(request, &config.auth_methods).await?;

    // Log request
    log_api_request(request, &auth_result)?;

    Ok(VerifiedRequest {
        original: request.clone(),
        authenticated_as: auth_result.identity,
        scopes: auth_result.scopes,
    })
}
```

### Bridge Security

```rust
/// Security for cross-chain bridges
pub struct BridgeSecurityConfig {
    /// Multi-sig requirements
    pub multi_sig: MultiSigConfig,

    /// Value limits
    pub value_limits: ValueLimits,

    /// Monitoring
    pub monitoring: BridgeMonitoring,

    /// Emergency procedures
    pub emergency: EmergencyConfig,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MultiSigConfig {
    pub required_signatures: u32,
    pub total_signers: u32,
    pub signers: Vec<SignerInfo>,
    pub time_lock: Option<Duration>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ValueLimits {
    pub per_transaction: Decimal,
    pub per_day: Decimal,
    pub per_week: Decimal,
    pub large_transaction_threshold: Decimal,
    pub large_transaction_delay: Duration,
}
```

---

## Conclusion

Interoperability enables Mycelix to be part of a larger ecosystem while maintaining its core values of sovereignty, privacy, and community governance. These standards ensure that communities can connect with the broader world without compromising their autonomy.

**Key Principles**:
1. **Sovereignty first** - Always maintain local control
2. **Standards-based** - Use open standards where possible
3. **Secure by default** - Never compromise security for convenience
4. **Graceful degradation** - Work even if external systems fail
5. **Privacy-preserving** - Share minimum necessary data

---

*"To be interoperable is not to be dependent. We connect with the world from a place of strength, not necessity."*

---

*Document Version: 1.0*
*Last Updated: 2025*
*Standards Referenced: W3C DIDs, W3C Verifiable Credentials, ActivityPub, OAuth 2.0, OpenAPI 3.0*
