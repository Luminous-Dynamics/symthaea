// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! `symthaea-iso20022` — SWIFT / ISO 20022 bridge for Mycelix
//!
//! Provides:
//! - Parsing of ISO 20022 `pacs.008` (FI-to-FI customer credit transfer) XML messages
//! - Mapping SWIFT BIC codes to Mycelix DIDs via [`DidRegistry`]
//! - Exchange rate conversion from fiat currencies to community currencies via [`RateSource`]
//! - HTLC (Hash Time-Locked Contract) atomic settlement via [`HtlcManager`]
//!
//! # Pipeline
//! ```text
//! SWIFT pacs.008 XML
//!   → parse_pacs008()        → PaymentInstruction
//!   → map_to_mycelix()       → MappingResult (Vec<MycelixOp> + unmapped)
//!   → create_settlement_htlc() → (htlc_id, hash, secret)
//!   → htlc_mgr.lock/claim/settle → settled on Holochain
//! ```
//!
//! # Example
//! ```rust
//! use symthaea_iso20022::{parse_pacs008, DidRegistry, RateSource, map_to_mycelix,
//!                         HtlcManager, create_settlement_htlc};
//!
//! let xml = r#"<Document xmlns="urn:iso:std:iso:20022:tech:xsd:pacs.008.001.12">
//!   <FIToFICstmrCdtTrf>
//!     <GrpHdr><MsgId>SWIFT-BRICS-001</MsgId><CreDtTm>2026-03-19T10:00:00Z</CreDtTm><NbOfTxs>1</NbOfTxs></GrpHdr>
//!     <CdtTrfTxInf>
//!       <PmtId><EndToEndId>TX-ZAR-001</EndToEndId></PmtId>
//!       <InstdAmt Ccy="ZAR">5000.00</InstdAmt>
//!       <Dbtr><Nm>Mycelix Community Cooperative</Nm></Dbtr>
//!       <DbtrAgt><FinInstnId><BICFI>ABORZA001</BICFI></FinInstnId></DbtrAgt>
//!       <Cdtr><Nm>Food Cooperative Johannesburg</Nm></Cdtr>
//!       <CdtrAgt><FinInstnId><BICFI>NEDBZAJJ</BICFI></FinInstnId></CdtrAgt>
//!     </CdtTrfTxInf>
//!   </FIToFICstmrCdtTrf>
//! </Document>"#;
//!
//! let instruction = parse_pacs008(xml).unwrap();
//! assert_eq!(instruction.message_id, "SWIFT-BRICS-001");
//!
//! let mut registry = DidRegistry::new();
//! registry.register_bic("ABORZA001", "did:mycelix:example-community");
//! registry.register_bic("NEDBZAJJ", "did:mycelix:food-coop-jhb");
//!
//! let rate = RateSource::community_only(0.1);
//! let mapping = map_to_mycelix(&instruction, &registry, &rate);
//! assert!(!mapping.operations.is_empty());
//!
//! let mut htlc_mgr = HtlcManager::new();
//! let (htlc_id, hash, secret) = create_settlement_htlc(&mut htlc_mgr, &mapping.operations[0], None).unwrap();
//!
//! htlc_mgr.lock(&htlc_id).unwrap();
//! htlc_mgr.claim(&htlc_id, &secret).unwrap();
//! htlc_mgr.settle(&htlc_id).unwrap();
//! ```

use std::collections::HashMap;
use std::time::Duration;

use chrono::{DateTime, Utc};
use quick_xml::events::Event;
use quick_xml::reader::Reader;
use rand::RngCore;
use thiserror::Error;
use tracing::{debug, warn};

// =============================================================================
// Errors
// =============================================================================

/// Errors returned by this crate.
#[derive(Debug, Error)]
pub enum Iso20022Error {
    #[error("XML parse error: {0}")]
    XmlParse(String),

    #[error("Missing required field: {0}")]
    MissingField(String),

    #[error("Invalid amount: {0}")]
    InvalidAmount(String),

    #[error("Invalid timestamp: {0}")]
    InvalidTimestamp(String),

    #[error("HTLC error: {0}")]
    Htlc(String),
}

// =============================================================================
// Core types — PaymentInstruction / Transaction / Amount
// =============================================================================

/// A monetary amount with currency code.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Amount {
    /// ISO 4217 currency code (e.g. "ZAR", "USD", "EUR").
    pub currency: String,
    /// Decimal value.
    pub value: f64,
}

/// A single credit transfer transaction within a pacs.008 message.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Transaction {
    /// End-to-end identifier assigned by the originator.
    pub end_to_end_id: String,
    /// Instructed amount and currency.
    pub amount: Amount,
    /// Debtor (sender) name.
    pub debtor_name: Option<String>,
    /// Debtor agent BIC (bank identifier code).
    pub debtor_bic: Option<String>,
    /// Creditor (receiver) name.
    pub creditor_name: Option<String>,
    /// Creditor agent BIC.
    pub creditor_bic: Option<String>,
}

/// Parsed ISO 20022 pacs.008 FI-to-FI customer credit transfer instruction.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PaymentInstruction {
    /// Group header message identifier.
    pub message_id: String,
    /// Creation date/time from the group header.
    pub created_at: Option<DateTime<Utc>>,
    /// Declared number of transactions.
    pub number_of_transactions: u32,
    /// Parsed transactions.
    pub transactions: Vec<Transaction>,
}

// =============================================================================
// Parser — pacs.008.001.12
// =============================================================================

/// Parse an ISO 20022 `pacs.008` XML message.
///
/// Accepts any pacs.008 version (001.08 through 001.12). Extracts the group
/// header and all `CdtTrfTxInf` blocks. Unknown or future extensions are
/// silently ignored.
///
/// # Errors
/// Returns [`Iso20022Error::XmlParse`] if the XML is malformed, or
/// [`Iso20022Error::MissingField`] if required group-header fields are absent.
pub fn parse_pacs008(xml: &str) -> Result<PaymentInstruction, Iso20022Error> {
    let mut reader = Reader::from_str(xml);
    reader.config_mut().trim_text(true);

    // State machine fields
    let mut message_id: Option<String> = None;
    let mut created_at: Option<DateTime<Utc>> = None;
    let mut number_of_transactions: u32 = 0;
    let mut transactions: Vec<Transaction> = Vec::new();

    // Current element path (shallow stack via current tag name)
    let mut current_tag = String::new();
    let mut in_grp_hdr = false;
    let mut in_cdt_trf = false;

    // Per-transaction state
    let mut cur_end_to_end: Option<String> = None;
    let mut cur_amount: Option<Amount> = None;
    let mut cur_debtor_name: Option<String> = None;
    let mut cur_debtor_bic: Option<String> = None;
    let mut cur_creditor_name: Option<String> = None;
    let mut cur_creditor_bic: Option<String> = None;
    let mut in_dbtr = false;
    let mut in_cdtr = false;
    let mut in_dbtr_agt = false;
    let mut in_cdtr_agt = false;

    let mut buf = Vec::new();

    loop {
        match reader.read_event_into(&mut buf) {
            Err(e) => return Err(Iso20022Error::XmlParse(e.to_string())),
            Ok(Event::Eof) => break,

            Ok(Event::Start(ref e)) => {
                let local = local_name(e.name().as_ref());
                current_tag = local.clone();

                match local.as_str() {
                    "GrpHdr" => in_grp_hdr = true,
                    "CdtTrfTxInf" => {
                        in_cdt_trf = true;
                        in_grp_hdr = false;
                        // Reset per-tx state
                        cur_end_to_end = None;
                        cur_amount = None;
                        cur_debtor_name = None;
                        cur_debtor_bic = None;
                        cur_creditor_name = None;
                        cur_creditor_bic = None;
                        in_dbtr = false;
                        in_cdtr = false;
                        in_dbtr_agt = false;
                        in_cdtr_agt = false;
                    }
                    "Dbtr" if in_cdt_trf => {
                        in_dbtr = true;
                        in_cdtr = false;
                    }
                    "Cdtr" if in_cdt_trf => {
                        in_cdtr = true;
                        in_dbtr = false;
                    }
                    "DbtrAgt" if in_cdt_trf => {
                        in_dbtr_agt = true;
                        in_cdtr_agt = false;
                    }
                    "CdtrAgt" if in_cdt_trf => {
                        in_cdtr_agt = true;
                        in_dbtr_agt = false;
                    }
                    "InstdAmt" if in_cdt_trf => {
                        // Currency is an attribute: <InstdAmt Ccy="ZAR">5000.00</InstdAmt>
                        let currency = e
                            .attributes()
                            .flatten()
                            .find(|a| local_name(a.key.as_ref()) == "Ccy")
                            .map(|a| String::from_utf8_lossy(&a.value).into_owned())
                            .unwrap_or_else(|| "XXX".to_string());
                        // Value will be read in Text event
                        cur_amount = Some(Amount {
                            currency,
                            value: 0.0,
                        });
                    }
                    _ => {}
                }
            }

            Ok(Event::End(ref e)) => {
                let local = local_name(e.name().as_ref());
                match local.as_str() {
                    "GrpHdr" => in_grp_hdr = false,
                    "CdtTrfTxInf" => {
                        // Commit the current transaction
                        if let Some(amount) = cur_amount.take() {
                            transactions.push(Transaction {
                                end_to_end_id: cur_end_to_end
                                    .take()
                                    .unwrap_or_else(|| format!("auto-{}", transactions.len())),
                                amount,
                                debtor_name: cur_debtor_name.take(),
                                debtor_bic: cur_debtor_bic.take(),
                                creditor_name: cur_creditor_name.take(),
                                creditor_bic: cur_creditor_bic.take(),
                            });
                        }
                        in_cdt_trf = false;
                        in_dbtr = false;
                        in_cdtr = false;
                        in_dbtr_agt = false;
                        in_cdtr_agt = false;
                    }
                    "Dbtr" => in_dbtr = false,
                    "Cdtr" => in_cdtr = false,
                    "DbtrAgt" => in_dbtr_agt = false,
                    "CdtrAgt" => in_cdtr_agt = false,
                    _ => {}
                }
                current_tag.clear();
            }

            Ok(Event::Text(ref e)) => {
                let text = e.unescape().unwrap_or_default().trim().to_string();
                if text.is_empty() {
                    continue;
                }

                if in_grp_hdr {
                    match current_tag.as_str() {
                        "MsgId" => {
                            debug!("pacs.008 MsgId: {}", text);
                            message_id = Some(text);
                        }
                        "CreDtTm" => {
                            created_at = text.parse::<DateTime<Utc>>().ok();
                        }
                        "NbOfTxs" => {
                            number_of_transactions = text.parse().unwrap_or(0);
                        }
                        _ => {}
                    }
                } else if in_cdt_trf {
                    match current_tag.as_str() {
                        "EndToEndId" => cur_end_to_end = Some(text),
                        "InstdAmt" => {
                            if let Some(ref mut amt) = cur_amount {
                                amt.value = text
                                    .parse()
                                    .map_err(|_| Iso20022Error::InvalidAmount(text.clone()))?;
                            }
                        }
                        "Nm" if in_dbtr => cur_debtor_name = Some(text),
                        "Nm" if in_cdtr => cur_creditor_name = Some(text),
                        "BICFI" if in_dbtr_agt => cur_debtor_bic = Some(text),
                        "BICFI" if in_cdtr_agt => cur_creditor_bic = Some(text),
                        _ => {}
                    }
                }
            }

            _ => {}
        }
        buf.clear();
    }

    let message_id =
        message_id.ok_or_else(|| Iso20022Error::MissingField("GrpHdr/MsgId".into()))?;

    if transactions.is_empty() && number_of_transactions > 0 {
        warn!(
            "pacs.008 declared {} transactions but none were parsed",
            number_of_transactions
        );
    }

    Ok(PaymentInstruction {
        message_id,
        created_at,
        number_of_transactions,
        transactions,
    })
}

/// Extract the local name from a qualified XML name, stripping any namespace prefix.
fn local_name(name: &[u8]) -> String {
    let s = std::str::from_utf8(name).unwrap_or("");
    // Strip namespace prefix (e.g. "ns:MsgId" → "MsgId")
    s.rsplit(':').next().unwrap_or(s).to_string()
}

// =============================================================================
// DID Registry — BIC → Mycelix DID mapping
// =============================================================================

/// Registry mapping SWIFT BIC codes to Mycelix DIDs.
///
/// Used during [`map_to_mycelix`] to resolve sender/receiver identities.
/// BIC codes that are not registered result in the transaction being placed
/// in [`MappingResult::unmapped`].
#[derive(Debug, Default, Clone)]
pub struct DidRegistry {
    bic_to_did: HashMap<String, String>,
}

impl DidRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a BIC code → Mycelix DID mapping.
    ///
    /// BICs are normalised to uppercase. Existing mappings are overwritten.
    pub fn register_bic(&mut self, bic: &str, did: &str) {
        self.bic_to_did.insert(bic.to_uppercase(), did.to_string());
    }

    /// Look up a DID by BIC code.
    pub fn resolve_bic(&self, bic: &str) -> Option<&str> {
        self.bic_to_did.get(&bic.to_uppercase()).map(|s| s.as_str())
    }

    /// Number of registered BIC mappings.
    pub fn len(&self) -> usize {
        self.bic_to_did.len()
    }

    /// True if the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.bic_to_did.is_empty()
    }
}

// =============================================================================
// Rate Source — fiat → community currency exchange rates
// =============================================================================

/// Source of exchange rates for converting fiat currency amounts to community
/// currency denominations used in Mycelix operations.
#[derive(Debug, Clone)]
pub struct RateSource {
    mode: RateMode,
}

#[derive(Debug, Clone)]
enum RateMode {
    /// Fixed rate: 1 unit of fiat = `rate` units of community currency.
    CommunityOnly(f64),
    /// Per-currency rates table.
    Table(HashMap<String, f64>),
}

impl RateSource {
    /// A single fixed exchange rate applied to all currencies.
    ///
    /// Useful for simple community deployments where fiat is converted 1:rate
    /// regardless of which currency is being transferred.
    pub fn community_only(rate: f64) -> Self {
        Self {
            mode: RateMode::CommunityOnly(rate),
        }
    }

    /// Per-currency rate table. Falls back to 1.0 for unknown currencies.
    pub fn from_table(rates: HashMap<String, f64>) -> Self {
        Self {
            mode: RateMode::Table(rates),
        }
    }

    /// Look up the rate for a given ISO 4217 currency code.
    pub fn rate_for(&self, currency: &str) -> f64 {
        match &self.mode {
            RateMode::CommunityOnly(r) => *r,
            RateMode::Table(t) => *t.get(currency).unwrap_or(&1.0),
        }
    }
}

// =============================================================================
// Mycelix operation types
// =============================================================================

/// A payment operation expressed in Mycelix terms (DID-to-DID transfer).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MycelixOp {
    /// Source DID (resolved from debtor BIC).
    pub from_did: String,
    /// Destination DID (resolved from creditor BIC).
    pub to_did: String,
    /// Original fiat amount.
    pub fiat_amount: Amount,
    /// Community currency amount after rate conversion.
    pub community_amount: f64,
    /// Original transaction end-to-end ID for traceability.
    pub end_to_end_id: String,
}

/// Result of mapping a [`PaymentInstruction`] to Mycelix operations.
#[derive(Debug, Clone)]
pub struct MappingResult {
    /// Successfully mapped operations (both BICs resolved to DIDs).
    pub operations: Vec<MycelixOp>,
    /// Transactions that could not be mapped (unknown BICs).
    pub unmapped: Vec<Transaction>,
}

/// Map a parsed pacs.008 payment instruction to Mycelix operations.
///
/// Each transaction is mapped if both the debtor and creditor BIC codes are
/// registered in the [`DidRegistry`]. Unresolvable transactions are collected
/// in [`MappingResult::unmapped`] for manual review.
pub fn map_to_mycelix(
    instruction: &PaymentInstruction,
    registry: &DidRegistry,
    rate: &RateSource,
) -> MappingResult {
    let mut operations = Vec::new();
    let mut unmapped = Vec::new();

    for tx in &instruction.transactions {
        let from_did = tx
            .debtor_bic
            .as_deref()
            .and_then(|bic| registry.resolve_bic(bic))
            .map(|s| s.to_string());

        let to_did = tx
            .creditor_bic
            .as_deref()
            .and_then(|bic| registry.resolve_bic(bic))
            .map(|s| s.to_string());

        match (from_did, to_did) {
            (Some(from), Some(to)) => {
                let r = rate.rate_for(&tx.amount.currency);
                let community_amount = tx.amount.value * r;
                debug!(
                    "Mapped {} {} → {} community units ({} → {})",
                    tx.amount.value, tx.amount.currency, community_amount, from, to
                );
                operations.push(MycelixOp {
                    from_did: from,
                    to_did: to,
                    fiat_amount: tx.amount.clone(),
                    community_amount,
                    end_to_end_id: tx.end_to_end_id.clone(),
                });
            }
            _ => {
                warn!(
                    "Could not resolve BICs for transaction {}: debtor={:?} creditor={:?}",
                    tx.end_to_end_id, tx.debtor_bic, tx.creditor_bic
                );
                unmapped.push(tx.clone());
            }
        }
    }

    MappingResult {
        operations,
        unmapped,
    }
}

// =============================================================================
// HTLC — Hash Time-Locked Contracts for atomic settlement
// =============================================================================

/// State of an HTLC.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HtlcState {
    /// Created, not yet locked.
    Pending,
    /// Funds locked, waiting for preimage reveal.
    Locked,
    /// Preimage revealed, funds claimed.
    Claimed,
    /// Settlement confirmed on both sides.
    Settled,
    /// Timed out — funds returned to sender.
    Expired,
}

/// An individual HTLC record.
#[derive(Debug, Clone)]
pub struct Htlc {
    pub id: String,
    pub op: MycelixOp,
    /// Blake3 hash of the secret preimage.
    pub hash: Vec<u8>,
    /// Timeout duration (None = no expiry in test contexts).
    pub timeout: Option<Duration>,
    /// Creation timestamp.
    pub created_at: DateTime<Utc>,
    pub state: HtlcState,
}

/// Manages the lifecycle of multiple HTLCs.
///
/// In production this would persist to the Holochain DHT or an on-chain contract.
/// Here it provides an in-memory manager that satisfies the test pipeline.
#[derive(Debug, Default)]
pub struct HtlcManager {
    htlcs: HashMap<String, Htlc>,
    counter: u64,
}

impl HtlcManager {
    /// Create a new empty manager.
    pub fn new() -> Self {
        Self::default()
    }

    /// Retrieve an HTLC by ID (immutable).
    pub fn get(&self, htlc_id: &str) -> Option<&Htlc> {
        self.htlcs.get(htlc_id)
    }

    /// Transition a Pending HTLC to Locked.
    ///
    /// # Errors
    /// Returns an error if the HTLC is not found or not in `Pending` state.
    pub fn lock(&mut self, htlc_id: &str) -> Result<(), Iso20022Error> {
        let htlc = self
            .htlcs
            .get_mut(htlc_id)
            .ok_or_else(|| Iso20022Error::Htlc(format!("HTLC not found: {htlc_id}")))?;

        if htlc.state != HtlcState::Pending {
            return Err(Iso20022Error::Htlc(format!(
                "HTLC {htlc_id} is in state {:?}, expected Pending",
                htlc.state
            )));
        }

        htlc.state = HtlcState::Locked;
        debug!("HTLC {} locked", htlc_id);
        Ok(())
    }

    /// Claim a Locked HTLC by revealing the preimage `secret`.
    ///
    /// Verifies `blake3(secret) == htlc.hash` before transitioning to `Claimed`.
    ///
    /// # Errors
    /// Returns an error if not found, not Locked, or secret is invalid.
    pub fn claim(&mut self, htlc_id: &str, secret: &[u8]) -> Result<(), Iso20022Error> {
        let htlc = self
            .htlcs
            .get_mut(htlc_id)
            .ok_or_else(|| Iso20022Error::Htlc(format!("HTLC not found: {htlc_id}")))?;

        if htlc.state != HtlcState::Locked {
            return Err(Iso20022Error::Htlc(format!(
                "HTLC {htlc_id} is in state {:?}, expected Locked",
                htlc.state
            )));
        }

        // Verify preimage
        let computed = blake3::hash(secret);
        if computed.as_bytes().as_slice() != htlc.hash.as_slice() {
            return Err(Iso20022Error::Htlc(format!(
                "Invalid preimage for HTLC {htlc_id}"
            )));
        }

        htlc.state = HtlcState::Claimed;
        debug!("HTLC {} claimed", htlc_id);
        Ok(())
    }

    /// Settle a Claimed HTLC (final state).
    ///
    /// # Errors
    /// Returns an error if not found or not in `Claimed` state.
    pub fn settle(&mut self, htlc_id: &str) -> Result<(), Iso20022Error> {
        let htlc = self
            .htlcs
            .get_mut(htlc_id)
            .ok_or_else(|| Iso20022Error::Htlc(format!("HTLC not found: {htlc_id}")))?;

        if htlc.state != HtlcState::Claimed {
            return Err(Iso20022Error::Htlc(format!(
                "HTLC {htlc_id} is in state {:?}, expected Claimed",
                htlc.state
            )));
        }

        htlc.state = HtlcState::Settled;
        debug!(
            "HTLC {} settled — {} → {}",
            htlc_id, htlc.op.from_did, htlc.op.to_did
        );
        Ok(())
    }
}

/// Create an HTLC for settling a [`MycelixOp`] and register it with the manager.
///
/// Generates a cryptographically random 32-byte secret, computes its Blake3 hash,
/// and returns `(htlc_id, hash, secret)`. The caller stores the secret and uses it
/// in [`HtlcManager::claim`] after the counterparty locks funds.
///
/// # Arguments
/// - `manager` — the HTLC manager to register with
/// - `op` — the Mycelix operation this HTLC settles
/// - `timeout` — optional expiry duration (None = no expiry)
///
/// # Returns
/// `(htlc_id, hash_bytes, secret_bytes)` — hash and secret are each 32 bytes.
pub fn create_settlement_htlc(
    manager: &mut HtlcManager,
    op: &MycelixOp,
    timeout: Option<Duration>,
) -> Result<(String, Vec<u8>, Vec<u8>), Iso20022Error> {
    // Generate random secret
    let mut secret = vec![0u8; 32];
    rand::thread_rng().fill_bytes(&mut secret);

    // Compute Blake3 hash of secret
    let hash = blake3::hash(&secret);
    let hash_bytes = hash.as_bytes().to_vec();

    // Assign ID
    manager.counter += 1;
    let htlc_id = format!(
        "htlc:{}:{}:{}",
        op.end_to_end_id,
        op.from_did.split(':').last().unwrap_or("unknown"),
        manager.counter
    );

    let htlc = Htlc {
        id: htlc_id.clone(),
        op: op.clone(),
        hash: hash_bytes.clone(),
        timeout,
        created_at: Utc::now(),
        state: HtlcState::Pending,
    };

    manager.htlcs.insert(htlc_id.clone(), htlc);
    debug!(
        "Created HTLC {} for {} → {}",
        htlc_id, op.from_did, op.to_did
    );

    Ok((htlc_id, hash_bytes, secret))
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    const PACS008_SAMPLE: &str = r#"<Document xmlns="urn:iso:std:iso:20022:tech:xsd:pacs.008.001.12">
  <FIToFICstmrCdtTrf>
    <GrpHdr>
      <MsgId>SWIFT-BRICS-001</MsgId>
      <CreDtTm>2026-03-19T10:00:00Z</CreDtTm>
      <NbOfTxs>1</NbOfTxs>
    </GrpHdr>
    <CdtTrfTxInf>
      <PmtId><EndToEndId>TX-ZAR-001</EndToEndId></PmtId>
      <InstdAmt Ccy="ZAR">5000.00</InstdAmt>
      <Dbtr><Nm>Mycelix Community Cooperative</Nm></Dbtr>
      <DbtrAgt><FinInstnId><BICFI>ABORZA001</BICFI></FinInstnId></DbtrAgt>
      <Cdtr><Nm>Food Cooperative Johannesburg</Nm></Cdtr>
      <CdtrAgt><FinInstnId><BICFI>NEDBZAJJ</BICFI></FinInstnId></CdtrAgt>
    </CdtTrfTxInf>
  </FIToFICstmrCdtTrf>
</Document>"#;

    #[test]
    fn test_parse_pacs008_basic() {
        let instruction = parse_pacs008(PACS008_SAMPLE).unwrap();
        assert_eq!(instruction.message_id, "SWIFT-BRICS-001");
        assert_eq!(instruction.transactions.len(), 1);
        assert_eq!(instruction.transactions[0].amount.currency, "ZAR");
        assert!((instruction.transactions[0].amount.value - 5000.0).abs() < 0.01);
        assert_eq!(
            instruction.transactions[0].debtor_bic.as_deref(),
            Some("ABORZA001")
        );
        assert_eq!(
            instruction.transactions[0].creditor_bic.as_deref(),
            Some("NEDBZAJJ")
        );
    }

    #[test]
    fn test_parse_pacs008_missing_msgid() {
        let bad_xml = r#"<Document xmlns="urn:iso:std:iso:20022:tech:xsd:pacs.008.001.12">
  <FIToFICstmrCdtTrf><GrpHdr><NbOfTxs>0</NbOfTxs></GrpHdr></FIToFICstmrCdtTrf>
</Document>"#;
        assert!(matches!(
            parse_pacs008(bad_xml),
            Err(Iso20022Error::MissingField(_))
        ));
    }

    #[test]
    fn test_did_registry() {
        let mut registry = DidRegistry::new();
        assert!(registry.is_empty());
        registry.register_bic("ABORZA001", "did:mycelix:example-community");
        registry.register_bic("NEDBZAJJ", "did:mycelix:food-coop-jhb");
        assert_eq!(registry.len(), 2);
        assert_eq!(
            registry.resolve_bic("ABORZA001"),
            Some("did:mycelix:example-community")
        );
        // BIC lookup is case-insensitive
        assert_eq!(
            registry.resolve_bic("aborza001"),
            Some("did:mycelix:example-community")
        );
        assert!(registry.resolve_bic("UNKNOWN").is_none());
    }

    #[test]
    fn test_map_to_mycelix_full() {
        let instruction = parse_pacs008(PACS008_SAMPLE).unwrap();

        let mut registry = DidRegistry::new();
        registry.register_bic("ABORZA001", "did:mycelix:example-community");
        registry.register_bic("NEDBZAJJ", "did:mycelix:food-coop-jhb");

        let rate = RateSource::community_only(0.1);
        let mapping = map_to_mycelix(&instruction, &registry, &rate);

        assert_eq!(mapping.operations.len(), 1);
        assert!(mapping.unmapped.is_empty());

        let op = &mapping.operations[0];
        assert_eq!(op.from_did, "did:mycelix:example-community");
        assert_eq!(op.to_did, "did:mycelix:food-coop-jhb");
        assert_eq!(op.fiat_amount.currency, "ZAR");
        assert!((op.community_amount - 500.0).abs() < 0.01); // 5000 * 0.1
    }

    #[test]
    fn test_map_to_mycelix_unmapped() {
        let instruction = parse_pacs008(PACS008_SAMPLE).unwrap();
        let registry = DidRegistry::new(); // Empty — no BICs registered
        let rate = RateSource::community_only(1.0);
        let mapping = map_to_mycelix(&instruction, &registry, &rate);

        assert!(mapping.operations.is_empty());
        assert_eq!(mapping.unmapped.len(), 1);
    }

    #[test]
    fn test_htlc_full_lifecycle() {
        let op = MycelixOp {
            from_did: "did:mycelix:sender".into(),
            to_did: "did:mycelix:receiver".into(),
            fiat_amount: Amount {
                currency: "ZAR".into(),
                value: 100.0,
            },
            community_amount: 10.0,
            end_to_end_id: "TX-001".into(),
        };

        let mut mgr = HtlcManager::new();
        let (htlc_id, _hash, secret) = create_settlement_htlc(&mut mgr, &op, None).unwrap();

        assert!(mgr.get(&htlc_id).is_some());
        assert_eq!(mgr.get(&htlc_id).unwrap().state, HtlcState::Pending);

        mgr.lock(&htlc_id).unwrap();
        assert_eq!(mgr.get(&htlc_id).unwrap().state, HtlcState::Locked);

        mgr.claim(&htlc_id, &secret).unwrap();
        assert_eq!(mgr.get(&htlc_id).unwrap().state, HtlcState::Claimed);

        mgr.settle(&htlc_id).unwrap();
        assert_eq!(mgr.get(&htlc_id).unwrap().state, HtlcState::Settled);
    }

    #[test]
    fn test_htlc_invalid_secret_rejected() {
        let op = MycelixOp {
            from_did: "did:mycelix:sender".into(),
            to_did: "did:mycelix:receiver".into(),
            fiat_amount: Amount {
                currency: "USD".into(),
                value: 50.0,
            },
            community_amount: 50.0,
            end_to_end_id: "TX-002".into(),
        };

        let mut mgr = HtlcManager::new();
        let (htlc_id, _hash, _secret) = create_settlement_htlc(&mut mgr, &op, None).unwrap();
        mgr.lock(&htlc_id).unwrap();

        let wrong_secret = vec![0u8; 32]; // all zeros
        assert!(matches!(
            mgr.claim(&htlc_id, &wrong_secret),
            Err(Iso20022Error::Htlc(_))
        ));
        // State should still be Locked
        assert_eq!(mgr.get(&htlc_id).unwrap().state, HtlcState::Locked);
    }

    #[test]
    fn test_htlc_invalid_state_transitions() {
        let op = MycelixOp {
            from_did: "did:mycelix:a".into(),
            to_did: "did:mycelix:b".into(),
            fiat_amount: Amount {
                currency: "EUR".into(),
                value: 200.0,
            },
            community_amount: 20.0,
            end_to_end_id: "TX-003".into(),
        };
        let mut mgr = HtlcManager::new();
        let (htlc_id, _, secret) = create_settlement_htlc(&mut mgr, &op, None).unwrap();

        // Can't claim before locking
        assert!(mgr.claim(&htlc_id, &secret).is_err());
        // Can't settle before claiming
        mgr.lock(&htlc_id).unwrap();
        assert!(mgr.settle(&htlc_id).is_err());
    }

    #[test]
    fn test_full_pipeline() {
        // Mirrors the integration test in conductor_integration.rs (offline version)
        let xml = PACS008_SAMPLE;
        let instruction = parse_pacs008(xml).unwrap();
        assert_eq!(instruction.message_id, "SWIFT-BRICS-001");
        assert_eq!(instruction.transactions.len(), 1);
        assert_eq!(instruction.transactions[0].amount.currency, "ZAR");

        let mut registry = DidRegistry::new();
        registry.register_bic("ABORZA001", "did:mycelix:example-community");
        registry.register_bic("NEDBZAJJ", "did:mycelix:food-coop-jhb");
        let rate = RateSource::community_only(0.1);
        let mapping = map_to_mycelix(&instruction, &registry, &rate);
        assert!(!mapping.operations.is_empty());

        let mut htlc_mgr = HtlcManager::new();
        let (htlc_id, hash, secret) =
            create_settlement_htlc(&mut htlc_mgr, &mapping.operations[0], None).unwrap();
        println!("HTLC created: id={htlc_id}, hash={:02x?}", &hash[..4]);

        htlc_mgr.lock(&htlc_id).unwrap();
        htlc_mgr.claim(&htlc_id, &secret).unwrap();
        htlc_mgr.settle(&htlc_id).unwrap();

        assert_eq!(htlc_mgr.get(&htlc_id).unwrap().state, HtlcState::Settled);
        println!("Full pipeline: SWIFT pacs.008 → parse → map → HTLC → settled ✓");
    }
}
