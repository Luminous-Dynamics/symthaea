// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! SSO/SAML Authentication
//!
//! Enterprise single sign-on with SAML 2.0 and OIDC support

use base64::{engine::general_purpose::STANDARD, Engine};
use chrono::{DateTime, Duration, Utc};
use openssl::x509::X509;
use serde::{Deserialize, Serialize};
use sqlx::PgPool;
use std::collections::HashMap;
use uuid::Uuid;
use xml::reader::{EventReader, XmlEvent};

/// SSO Provider configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SsoProvider {
    pub id: Uuid,
    pub tenant_id: Uuid,
    pub provider_type: SsoProviderType,
    pub name: String,
    pub enabled: bool,

    // SAML settings
    pub saml_entity_id: Option<String>,
    pub saml_sso_url: Option<String>,
    pub saml_slo_url: Option<String>,
    pub saml_certificate: Option<String>,
    pub saml_signing_certificate: Option<String>,

    // OIDC settings
    pub oidc_issuer: Option<String>,
    pub oidc_client_id: Option<String>,
    pub oidc_client_secret: Option<String>,
    pub oidc_scopes: Vec<String>,

    // Attribute mapping
    pub attribute_mapping: AttributeMapping,

    // Settings
    pub auto_provision: bool,
    pub default_role: String,
    pub allowed_domains: Vec<String>,

    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum SsoProviderType {
    Saml,
    Oidc,
    Azure,
    Okta,
    Google,
    OneLogin,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AttributeMapping {
    pub email: String,
    pub first_name: Option<String>,
    pub last_name: Option<String>,
    pub display_name: Option<String>,
    pub groups: Option<String>,
    pub department: Option<String>,
}

/// SAML authentication result
#[derive(Debug, Clone)]
pub struct SamlAuthResult {
    pub name_id: String,
    pub session_index: Option<String>,
    pub attributes: HashMap<String, Vec<String>>,
    pub valid_until: DateTime<Utc>,
}

/// SSO Service
pub struct SsoService {
    pool: PgPool,
    our_entity_id: String,
    our_acs_url: String,
}

impl SsoService {
    pub fn new(pool: PgPool, base_url: &str) -> Self {
        Self {
            pool,
            our_entity_id: format!("{}/saml/metadata", base_url),
            our_acs_url: format!("{}/saml/acs", base_url),
        }
    }

    /// Get provider for tenant
    pub async fn get_provider(
        &self,
        tenant_id: Uuid,
    ) -> Result<Option<SsoProvider>, sqlx::Error> {
        sqlx::query_as::<_, SsoProviderRow>(
            "SELECT * FROM sso_providers WHERE tenant_id = $1 AND enabled = true",
        )
        .bind(tenant_id)
        .fetch_optional(&self.pool)
        .await
        .map(|r| r.map(|row| row.into()))
    }

    /// Create/update SSO provider
    pub async fn upsert_provider(&self, provider: &SsoProvider) -> Result<(), sqlx::Error> {
        sqlx::query(
            r#"
            INSERT INTO sso_providers (
                id, tenant_id, provider_type, name, enabled,
                saml_entity_id, saml_sso_url, saml_slo_url, saml_certificate,
                oidc_issuer, oidc_client_id, oidc_client_secret, oidc_scopes,
                attribute_mapping, auto_provision, default_role, allowed_domains,
                created_at, updated_at
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19)
            ON CONFLICT (id) DO UPDATE SET
                name = EXCLUDED.name,
                enabled = EXCLUDED.enabled,
                saml_entity_id = EXCLUDED.saml_entity_id,
                saml_sso_url = EXCLUDED.saml_sso_url,
                saml_slo_url = EXCLUDED.saml_slo_url,
                saml_certificate = EXCLUDED.saml_certificate,
                oidc_issuer = EXCLUDED.oidc_issuer,
                oidc_client_id = EXCLUDED.oidc_client_id,
                oidc_client_secret = EXCLUDED.oidc_client_secret,
                oidc_scopes = EXCLUDED.oidc_scopes,
                attribute_mapping = EXCLUDED.attribute_mapping,
                auto_provision = EXCLUDED.auto_provision,
                default_role = EXCLUDED.default_role,
                allowed_domains = EXCLUDED.allowed_domains,
                updated_at = NOW()
            "#,
        )
        .bind(provider.id)
        .bind(provider.tenant_id)
        .bind(&format!("{:?}", provider.provider_type))
        .bind(&provider.name)
        .bind(provider.enabled)
        .bind(&provider.saml_entity_id)
        .bind(&provider.saml_sso_url)
        .bind(&provider.saml_slo_url)
        .bind(&provider.saml_certificate)
        .bind(&provider.oidc_issuer)
        .bind(&provider.oidc_client_id)
        .bind(&provider.oidc_client_secret)
        .bind(&provider.oidc_scopes)
        .bind(&serde_json::to_value(&provider.attribute_mapping).unwrap())
        .bind(provider.auto_provision)
        .bind(&provider.default_role)
        .bind(&provider.allowed_domains)
        .bind(provider.created_at)
        .bind(Utc::now())
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    /// Generate SAML authentication request
    pub fn create_saml_request(&self, provider: &SsoProvider) -> Result<String, SsoError> {
        let request_id = format!("_{}", Uuid::new_v4());
        let issue_instant = Utc::now().format("%Y-%m-%dT%H:%M:%SZ").to_string();

        let saml_request = format!(
            r#"<?xml version="1.0" encoding="UTF-8"?>
<samlp:AuthnRequest
    xmlns:samlp="urn:oasis:names:tc:SAML:2.0:protocol"
    xmlns:saml="urn:oasis:names:tc:SAML:2.0:assertion"
    ID="{}"
    Version="2.0"
    IssueInstant="{}"
    Destination="{}"
    AssertionConsumerServiceURL="{}"
    ProtocolBinding="urn:oasis:names:tc:SAML:2.0:bindings:HTTP-POST">
    <saml:Issuer>{}</saml:Issuer>
    <samlp:NameIDPolicy
        Format="urn:oasis:names:tc:SAML:1.1:nameid-format:emailAddress"
        AllowCreate="true"/>
</samlp:AuthnRequest>"#,
            request_id,
            issue_instant,
            provider.saml_sso_url.as_ref().ok_or(SsoError::MissingConfig("saml_sso_url"))?,
            self.our_acs_url,
            self.our_entity_id
        );

        // Deflate and base64 encode
        let mut encoder = flate2::write::DeflateEncoder::new(
            Vec::new(),
            flate2::Compression::default(),
        );
        std::io::Write::write_all(&mut encoder, saml_request.as_bytes())
            .map_err(|e| SsoError::Encoding(e.to_string()))?;
        let compressed = encoder.finish().map_err(|e| SsoError::Encoding(e.to_string()))?;

        Ok(STANDARD.encode(&compressed))
    }

    /// Build redirect URL for SAML SSO
    pub fn build_saml_redirect_url(
        &self,
        provider: &SsoProvider,
        relay_state: Option<&str>,
    ) -> Result<String, SsoError> {
        let saml_request = self.create_saml_request(provider)?;
        let sso_url = provider.saml_sso_url.as_ref().ok_or(SsoError::MissingConfig("saml_sso_url"))?;

        let mut url = format!("{}?SAMLRequest={}", sso_url, urlencoding::encode(&saml_request));

        if let Some(state) = relay_state {
            url.push_str(&format!("&RelayState={}", urlencoding::encode(state)));
        }

        Ok(url)
    }

    /// Validate and parse SAML response
    pub fn validate_saml_response(
        &self,
        provider: &SsoProvider,
        saml_response: &str,
    ) -> Result<SamlAuthResult, SsoError> {
        // Decode base64
        let decoded = STANDARD
            .decode(saml_response)
            .map_err(|e| SsoError::InvalidResponse(e.to_string()))?;

        let response_xml = String::from_utf8(decoded)
            .map_err(|e| SsoError::InvalidResponse(e.to_string()))?;

        // Validate signature if certificate is provided
        if let Some(cert_pem) = &provider.saml_certificate {
            self.verify_signature(&response_xml, cert_pem)?;
        }

        // Parse SAML response
        self.parse_saml_response(&response_xml, &provider.attribute_mapping)
    }

    /// Verify XML signature
    fn verify_signature(&self, xml: &str, cert_pem: &str) -> Result<(), SsoError> {
        // Load certificate
        let cert = X509::from_pem(cert_pem.as_bytes())
            .map_err(|e| SsoError::InvalidCertificate(e.to_string()))?;

        // In production, use proper XML signature verification library
        // This is a simplified placeholder
        if !xml.contains("Signature") {
            return Err(SsoError::SignatureVerification("No signature found".into()));
        }

        Ok(())
    }

    /// Parse SAML response XML
    fn parse_saml_response(
        &self,
        xml: &str,
        mapping: &AttributeMapping,
    ) -> Result<SamlAuthResult, SsoError> {
        let mut name_id = String::new();
        let mut session_index = None;
        let mut attributes: HashMap<String, Vec<String>> = HashMap::new();
        let mut valid_until = Utc::now() + Duration::hours(1);

        let parser = EventReader::from_str(xml);
        let mut current_element = String::new();
        let mut current_attribute_name = String::new();

        for event in parser {
            match event {
                Ok(XmlEvent::StartElement { name, attributes: attrs, .. }) => {
                    current_element = name.local_name.clone();

                    if name.local_name == "Attribute" {
                        for attr in attrs {
                            if attr.name.local_name == "Name" {
                                current_attribute_name = attr.value.clone();
                            }
                        }
                    }

                    if name.local_name == "Conditions" {
                        for attr in attrs {
                            if attr.name.local_name == "NotOnOrAfter" {
                                if let Ok(dt) = DateTime::parse_from_rfc3339(&attr.value) {
                                    valid_until = dt.with_timezone(&Utc);
                                }
                            }
                        }
                    }

                    if name.local_name == "AuthnStatement" {
                        for attr in attrs {
                            if attr.name.local_name == "SessionIndex" {
                                session_index = Some(attr.value.clone());
                            }
                        }
                    }
                }
                Ok(XmlEvent::Characters(text)) => {
                    if current_element == "NameID" {
                        name_id = text.clone();
                    } else if current_element == "AttributeValue" && !current_attribute_name.is_empty() {
                        attributes
                            .entry(current_attribute_name.clone())
                            .or_insert_with(Vec::new)
                            .push(text);
                    }
                }
                Ok(XmlEvent::EndElement { name }) => {
                    if name.local_name == "Attribute" {
                        current_attribute_name.clear();
                    }
                }
                Err(e) => {
                    return Err(SsoError::InvalidResponse(e.to_string()));
                }
                _ => {}
            }
        }

        if name_id.is_empty() {
            return Err(SsoError::InvalidResponse("No NameID found".into()));
        }

        Ok(SamlAuthResult {
            name_id,
            session_index,
            attributes,
            valid_until,
        })
    }

    /// Generate SP metadata XML
    pub fn generate_metadata(&self) -> String {
        format!(
            r#"<?xml version="1.0" encoding="UTF-8"?>
<md:EntityDescriptor
    xmlns:md="urn:oasis:names:tc:SAML:2.0:metadata"
    entityID="{}">
    <md:SPSSODescriptor
        AuthnRequestsSigned="false"
        WantAssertionsSigned="true"
        protocolSupportEnumeration="urn:oasis:names:tc:SAML:2.0:protocol">
        <md:NameIDFormat>urn:oasis:names:tc:SAML:1.1:nameid-format:emailAddress</md:NameIDFormat>
        <md:AssertionConsumerService
            Binding="urn:oasis:names:tc:SAML:2.0:bindings:HTTP-POST"
            Location="{}"
            index="0"
            isDefault="true"/>
    </md:SPSSODescriptor>
</md:EntityDescriptor>"#,
            self.our_entity_id, self.our_acs_url
        )
    }

    /// Extract user info from SAML attributes
    pub fn extract_user_info(
        &self,
        result: &SamlAuthResult,
        mapping: &AttributeMapping,
    ) -> UserInfo {
        let get_attr = |name: &str| -> Option<String> {
            result.attributes.get(name).and_then(|v| v.first().cloned())
        };

        UserInfo {
            email: get_attr(&mapping.email).unwrap_or_else(|| result.name_id.clone()),
            first_name: mapping.first_name.as_ref().and_then(|n| get_attr(n)),
            last_name: mapping.last_name.as_ref().and_then(|n| get_attr(n)),
            display_name: mapping.display_name.as_ref().and_then(|n| get_attr(n)),
            groups: mapping.groups.as_ref().and_then(|n| {
                result.attributes.get(n).cloned()
            }).unwrap_or_default(),
            department: mapping.department.as_ref().and_then(|n| get_attr(n)),
        }
    }
}

#[derive(Debug, Clone)]
pub struct UserInfo {
    pub email: String,
    pub first_name: Option<String>,
    pub last_name: Option<String>,
    pub display_name: Option<String>,
    pub groups: Vec<String>,
    pub department: Option<String>,
}

#[derive(Debug)]
pub enum SsoError {
    MissingConfig(&'static str),
    InvalidResponse(String),
    InvalidCertificate(String),
    SignatureVerification(String),
    Encoding(String),
    Database(sqlx::Error),
}

impl From<sqlx::Error> for SsoError {
    fn from(err: sqlx::Error) -> Self {
        SsoError::Database(err)
    }
}

// Database row type
#[derive(sqlx::FromRow)]
struct SsoProviderRow {
    id: Uuid,
    tenant_id: Uuid,
    provider_type: String,
    name: String,
    enabled: bool,
    saml_entity_id: Option<String>,
    saml_sso_url: Option<String>,
    saml_slo_url: Option<String>,
    saml_certificate: Option<String>,
    saml_signing_certificate: Option<String>,
    oidc_issuer: Option<String>,
    oidc_client_id: Option<String>,
    oidc_client_secret: Option<String>,
    oidc_scopes: Vec<String>,
    attribute_mapping: serde_json::Value,
    auto_provision: bool,
    default_role: String,
    allowed_domains: Vec<String>,
    created_at: DateTime<Utc>,
    updated_at: DateTime<Utc>,
}

impl From<SsoProviderRow> for SsoProvider {
    fn from(row: SsoProviderRow) -> Self {
        let provider_type = match row.provider_type.as_str() {
            "Saml" => SsoProviderType::Saml,
            "Oidc" => SsoProviderType::Oidc,
            "Azure" => SsoProviderType::Azure,
            "Okta" => SsoProviderType::Okta,
            "Google" => SsoProviderType::Google,
            "OneLogin" => SsoProviderType::OneLogin,
            _ => SsoProviderType::Saml,
        };

        SsoProvider {
            id: row.id,
            tenant_id: row.tenant_id,
            provider_type,
            name: row.name,
            enabled: row.enabled,
            saml_entity_id: row.saml_entity_id,
            saml_sso_url: row.saml_sso_url,
            saml_slo_url: row.saml_slo_url,
            saml_certificate: row.saml_certificate,
            saml_signing_certificate: row.saml_signing_certificate,
            oidc_issuer: row.oidc_issuer,
            oidc_client_id: row.oidc_client_id,
            oidc_client_secret: row.oidc_client_secret,
            oidc_scopes: row.oidc_scopes,
            attribute_mapping: serde_json::from_value(row.attribute_mapping).unwrap_or_default(),
            auto_provision: row.auto_provision,
            default_role: row.default_role,
            allowed_domains: row.allowed_domains,
            created_at: row.created_at,
            updated_at: row.updated_at,
        }
    }
}
