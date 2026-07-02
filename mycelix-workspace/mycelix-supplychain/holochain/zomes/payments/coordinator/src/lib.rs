// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Payments Coordinator Zome - Business logic for payment processing
use hdk::prelude::*;
use payments_integrity::*;

use mycelix_zome_helpers as _;
fn ensure_path(path: Path, link_type: LinkTypes) -> ExternResult<EntryHash> {
    let typed = path.typed(link_type)?;
    typed.ensure()?;
    typed.path_entry_hash()
}

// ============================================================================
// Payment Management
// ============================================================================

#[derive(Serialize, Deserialize, Debug)]
pub struct CreatePaymentInput {
    pub po_hash: ActionHash,
    pub amount: u64,
    pub currency: String,
    pub method: PaymentMethod,
    pub payee: AgentPubKey,
    pub reference: String,
}

#[hdk_extern]
pub fn create_payment(input: CreatePaymentInput) -> ExternResult<ActionHash> {
    // Input validation
    if input.amount == 0 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Payment amount must be greater than 0".to_string()
        )));
    }
    if input.currency.is_empty() || input.currency.len() > 10 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Currency must be 1-10 characters".to_string()
        )));
    }
    if input.reference.len() > 200 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Reference cannot exceed 200 characters".to_string()
        )));
    }

    let payer = agent_info()?.agent_initial_pubkey;

    let payment = Payment {
        po_hash: input.po_hash.clone(),
        amount: input.amount,
        currency: input.currency,
        method: input.method,
        status: PaymentStatus::Pending,
        payer: payer.clone(),
        payee: input.payee.clone(),
        reference: input.reference,
        created_at: sys_time()?,
        completed_at: None,
    };

    let action_hash = create_entry(EntryTypes::Payment(payment.clone()))?;
    create_link(
        input.po_hash,
        action_hash.clone(),
        LinkTypes::PoToPayments,
        (),
    )?;

    let payer_path = Path::from(format!("payer/{}", payer));
    let payer_hash = ensure_path(payer_path, LinkTypes::PayerToPayments)?;
    create_link(
        payer_hash,
        action_hash.clone(),
        LinkTypes::PayerToPayments,
        (),
    )?;

    let payee_path = Path::from(format!("payee/{}", input.payee));
    let payee_hash = ensure_path(payee_path, LinkTypes::PayeeToPayments)?;
    create_link(
        payee_hash,
        action_hash.clone(),
        LinkTypes::PayeeToPayments,
        (),
    )?;

    Ok(action_hash)
}

#[hdk_extern]
pub fn get_payment(hash: ActionHash) -> ExternResult<Option<Payment>> {
    match get(hash, GetOptions::default())? {
        Some(r) => Ok(r.entry().to_app_option().map_err(|e| wasm_error!(e))?),
        None => Ok(None),
    }
}

#[hdk_extern]
pub fn update_payment_status(input: (ActionHash, PaymentStatus)) -> ExternResult<ActionHash> {
    let (hash, new_status) = input;
    if let Some(record) = get(hash.clone(), GetOptions::default())? {
        if let Some(mut payment) = record
            .entry()
            .to_app_option::<Payment>()
            .map_err(|e| wasm_error!(e))?
        {
            payment.status = new_status.clone();
            if new_status == PaymentStatus::Completed {
                payment.completed_at = Some(sys_time()?);
            }
            return update_entry(hash, EntryTypes::Payment(payment));
        }
    }
    Err(wasm_error!(WasmErrorInner::Guest(
        "Payment not found".into()
    )))
}

#[hdk_extern]
pub fn create_invoice(invoice: Invoice) -> ExternResult<ActionHash> {
    let action_hash = create_entry(EntryTypes::Invoice(invoice.clone()))?;
    create_link(
        invoice.po_hash,
        action_hash.clone(),
        LinkTypes::PoToInvoices,
        (),
    )?;
    Ok(action_hash)
}

#[hdk_extern]
pub fn get_invoice(hash: ActionHash) -> ExternResult<Option<Invoice>> {
    match get(hash, GetOptions::default())? {
        Some(r) => Ok(r.entry().to_app_option().map_err(|e| wasm_error!(e))?),
        None => Ok(None),
    }
}

#[hdk_extern]
pub fn create_escrow(escrow: EscrowAccount) -> ExternResult<ActionHash> {
    let action_hash = create_entry(EntryTypes::EscrowAccount(escrow.clone()))?;
    create_link(
        escrow.po_hash,
        action_hash.clone(),
        LinkTypes::PoToEscrow,
        (),
    )?;
    Ok(action_hash)
}

#[hdk_extern]
pub fn fund_escrow(hash: ActionHash) -> ExternResult<ActionHash> {
    if let Some(record) = get(hash.clone(), GetOptions::default())? {
        if let Some(mut escrow) = record
            .entry()
            .to_app_option::<EscrowAccount>()
            .map_err(|e| wasm_error!(e))?
        {
            escrow.funded_at = Some(sys_time()?);
            return update_entry(hash, EntryTypes::EscrowAccount(escrow));
        }
    }
    Err(wasm_error!(WasmErrorInner::Guest(
        "Escrow not found".into()
    )))
}

#[hdk_extern]
pub fn release_escrow(hash: ActionHash) -> ExternResult<ActionHash> {
    if let Some(record) = get(hash.clone(), GetOptions::default())? {
        if let Some(mut escrow) = record
            .entry()
            .to_app_option::<EscrowAccount>()
            .map_err(|e| wasm_error!(e))?
        {
            // Authorization: only the buyer (whose funds are held in escrow) or a
            // designated arbiter may release the escrow. Without this check, any
            // agent — including the seller, who benefits from the release — could
            // call `release_escrow` directly and drain funds without the buyer's
            // consent. The seller is deliberately excluded: they are the
            // beneficiary of the release, not an authorizer of it.
            let caller = agent_info()?.agent_initial_pubkey;
            let is_buyer = caller == escrow.buyer;
            let is_arbiter = escrow
                .arbiter
                .as_ref()
                .map(|a| *a == caller)
                .unwrap_or(false);
            if !is_buyer && !is_arbiter {
                return Err(wasm_error!(WasmErrorInner::Guest(
                    "Only the escrow buyer or designated arbiter may release escrow".to_string()
                )));
            }

            if escrow.funded_at.is_none() {
                return Err(wasm_error!(WasmErrorInner::Guest(
                    "Escrow not funded".into()
                )));
            }
            escrow.released_at = Some(sys_time()?);
            return update_entry(hash, EntryTypes::EscrowAccount(escrow));
        }
    }
    Err(wasm_error!(WasmErrorInner::Guest(
        "Escrow not found".into()
    )))
}

#[hdk_extern]
pub fn get_po_payments(po_hash: ActionHash) -> ExternResult<Vec<Payment>> {
    let filter = LinkTypeFilter::try_from(LinkTypes::PoToPayments)?;
    let links = get_links(LinkQuery::new(po_hash, filter), GetStrategy::default())?;

    let mut payments = Vec::new();
    for link in links {
        if let Some(hash) = link.target.into_action_hash() {
            if let Some(payment) = get_payment(hash)? {
                payments.push(payment);
            }
        }
    }
    Ok(payments)
}

#[hdk_extern]
pub fn confirm_payment(hash: ActionHash) -> ExternResult<ActionHash> {
    update_payment_status((hash, PaymentStatus::Completed))
}

#[hdk_extern]
pub fn refund_payment(hash: ActionHash) -> ExternResult<ActionHash> {
    update_payment_status((hash, PaymentStatus::Refunded))
}

#[hdk_extern]
pub fn get_my_payments(_: ()) -> ExternResult<Vec<Payment>> {
    let my_agent = agent_info()?.agent_initial_pubkey;
    let payer_path = Path::from(format!("payer/{}", my_agent));
    let typed = payer_path.typed(LinkTypes::PayerToPayments)?;
    let filter = LinkTypeFilter::try_from(LinkTypes::PayerToPayments)?;
    let links = get_links(
        LinkQuery::new(typed.path_entry_hash()?, filter),
        GetStrategy::default(),
    )?;

    let mut payments = Vec::new();
    for link in links {
        if let Some(hash) = link.target.into_action_hash() {
            if let Some(payment) = get_payment(hash)? {
                payments.push(payment);
            }
        }
    }
    Ok(payments)
}

#[hdk_extern]
pub fn get_po_escrow(po_hash: ActionHash) -> ExternResult<Option<Record>> {
    let filter = LinkTypeFilter::try_from(LinkTypes::PoToEscrow)?;
    let links = get_links(LinkQuery::new(po_hash, filter), GetStrategy::default())?;
    if let Some(link) = links.first() {
        let hash = link.target.clone().into_action_hash().ok_or_else(|| {
            wasm_error!(WasmErrorInner::Guest(
                "Invalid escrow link target".to_string()
            ))
        })?;
        get(hash, GetOptions::default())
    } else {
        Ok(None)
    }
}

// ============================================================================
// Workflow 3: Payment Automation
// ============================================================================

#[derive(Serialize, Deserialize, Debug)]
pub struct FulfillmentPaymentInput {
    pub po_hash: ActionHash,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct FulfillmentPaymentResult {
    /// "escrow_released" | "payment_created" | "already_paid" | "po_not_received"
    /// | "settlement_failed"
    pub action: String,
    pub payment_hash: Option<ActionHash>,
    pub amount: u64,
    pub currency: String,
    /// Populated only when action == "settlement_failed": why the finance
    /// cluster settlement was not confirmed.
    pub settlement_error: Option<String>,
}

#[hdk_extern]
pub fn process_fulfillment_payment(
    input: FulfillmentPaymentInput,
) -> ExternResult<FulfillmentPaymentResult> {
    // Step 1: Get the purchase order via local call to procurement
    let po_response = call(
        CallTargetCell::Local,
        "procurement_coordinator",
        "get_purchase_order".into(),
        None,
        input.po_hash.clone(),
    )?;

    let po_value: serde_json::Value = match po_response {
        ZomeCallResponse::Ok(result) => serde_json::from_slice(
            result
                .decode::<serde_bytes::ByteBuf>()
                .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
                .as_ref(),
        )
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?,
        ZomeCallResponse::NetworkError(e) => {
            return Err(wasm_error!(WasmErrorInner::Guest(format!(
                "Network error fetching PO: {}",
                e
            ))));
        }
        ZomeCallResponse::Unauthorized(_, _, _, _) => {
            return Err(wasm_error!(WasmErrorInner::Guest(
                "Unauthorized to fetch PO".to_string()
            )));
        }
        ZomeCallResponse::CountersigningSession(_) => {
            return Err(wasm_error!(WasmErrorInner::Guest(
                "Countersigning failed".to_string()
            )));
        }
        _ => {
            return Err(wasm_error!(WasmErrorInner::Guest(
                "Unexpected response fetching PO".to_string()
            )));
        }
    };

    // Unwrap the Option<PurchaseOrder>
    let po = match po_value {
        serde_json::Value::Null => {
            return Ok(FulfillmentPaymentResult {
                action: "po_not_received".to_string(),
                payment_hash: None,
                amount: 0,
                currency: "USD".to_string(),
                settlement_error: None,
            });
        }
        v => v,
    };

    let status = po
        .get("status")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    let total_amount = po.get("total_amount").and_then(|v| v.as_u64()).unwrap_or(0);

    let currency = po
        .get("currency")
        .and_then(|v| v.as_str())
        .unwrap_or("USD")
        .to_string();

    let supplier: AgentPubKey = match po.get("supplier") {
        Some(v) => serde_json::from_value(v.clone())
            .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?,
        None => {
            return Err(wasm_error!(WasmErrorInner::Guest(
                "PO missing supplier field".to_string()
            )));
        }
    };

    let po_number = po
        .get("po_number")
        .and_then(|v| v.as_str())
        .unwrap_or("UNKNOWN")
        .to_string();

    // Step 2: Check PO has been received
    if status != "Received" {
        return Ok(FulfillmentPaymentResult {
            action: "po_not_received".to_string(),
            payment_hash: None,
            amount: total_amount,
            currency,
            settlement_error: None,
        });
    }

    // Step 3: Check if already paid
    let total_paid = get_po_total_paid(input.po_hash.clone())?;
    if total_paid >= total_amount {
        return Ok(FulfillmentPaymentResult {
            action: "already_paid".to_string(),
            payment_hash: None,
            amount: total_amount,
            currency,
            settlement_error: None,
        });
    }

    // Step 4: Check for funded escrow and release it.
    //
    // NOTE: escrow release here is local-only and does not attempt finance
    // settlement (unlike Step 5-6 below). Escrow amounts/currencies are
    // PO-defined (frequently non-SAP, e.g. "USD"), and finance's
    // process_payment currently only accepts SAP — so there's no settlement
    // call to gate on yet. Extending real settlement to escrow release needs
    // either restricting supplychain escrow to SAP-only or adding
    // multi-currency support to finance first; out of scope for this pass.
    if let Some(escrow_record) = get_po_escrow(input.po_hash.clone())? {
        if let Some(escrow) = escrow_record
            .entry()
            .to_app_option::<EscrowAccount>()
            .map_err(|e| wasm_error!(e))?
        {
            if escrow.funded_at.is_some() && escrow.released_at.is_none() {
                let escrow_hash = escrow_record.action_address().clone();
                release_escrow(escrow_hash.clone())?;
                return Ok(FulfillmentPaymentResult {
                    action: "escrow_released".to_string(),
                    payment_hash: Some(escrow_hash),
                    amount: escrow.amount,
                    currency: escrow.currency,
                    settlement_error: None,
                });
            }
        }
    }

    // Step 5: Create a direct payment (starts Pending — NOT confirmed yet).
    let remaining = total_amount.saturating_sub(total_paid);
    let payment_input = CreatePaymentInput {
        po_hash: input.po_hash.clone(),
        amount: remaining,
        currency: currency.clone(),
        method: PaymentMethod::BankTransfer,
        payee: supplier,
        reference: format!("Auto-payment for PO {}", po_number),
    };
    let payment_hash = create_payment(payment_input)?;

    // Step 6: Settlement in the finance cluster is now a precondition for
    // confirming the payment, not a best-effort afterthought. Previously
    // this called confirm_payment() first and then discarded the
    // settle_in_finance() result (`let _ = ...`) — meaning the local record
    // was marked Completed, and process_fulfillment_payment reported
    // "payment_created" success, regardless of whether the finance cluster
    // was reachable or whether the settlement call was rejected. Goods could
    // ship on a payment that never actually moved any SAP. See
    // MYCELIX_REVIEW.md P1 #4.
    let settlement = settle_in_finance(payment_hash.clone())?;
    if !settlement.settled {
        update_payment_status((payment_hash.clone(), PaymentStatus::Failed))?;
        return Ok(FulfillmentPaymentResult {
            action: "settlement_failed".to_string(),
            payment_hash: Some(payment_hash),
            amount: remaining,
            currency,
            settlement_error: settlement.error,
        });
    }
    confirm_payment(payment_hash.clone())?;

    Ok(FulfillmentPaymentResult {
        action: "payment_created".to_string(),
        payment_hash: Some(payment_hash),
        amount: remaining,
        currency,
        settlement_error: None,
    })
}

#[hdk_extern]
pub fn get_po_total_paid(po_hash: ActionHash) -> ExternResult<u64> {
    let payments = get_po_payments(po_hash)?;
    let total: u64 = payments
        .iter()
        .filter(|p| p.status == PaymentStatus::Completed)
        .map(|p| p.amount)
        .sum();
    Ok(total)
}

// ============================================================================
// Gap 1: Finance Settlement Bridge
// ============================================================================

/// Result of attempting to settle a supplychain payment in the finance cluster.
#[derive(Serialize, Deserialize, Debug)]
pub struct FinanceSettlementResult {
    pub settled: bool,
    pub finance_reference: Option<String>,
    pub error: Option<String>,
}

/// Settle a local supplychain payment in the finance cluster.
///
/// Looks up the local payment by `payment_hash`, then calls the finance cluster's
/// `process_payment` function via `CallTargetCell::OtherRole("finance")`.
/// If the finance cluster is not installed (role not found) the call will return
/// an `Err`, which is caught and surfaced as a non-fatal `FinanceSettlementResult`.
pub fn settle_in_finance(payment_hash: ActionHash) -> ExternResult<FinanceSettlementResult> {
    // Retrieve the local payment record
    let payment = match get_payment(payment_hash.clone())? {
        Some(p) => p,
        None => {
            return Ok(FinanceSettlementResult {
                settled: false,
                finance_reference: None,
                error: Some(format!("Payment {} not found locally", payment_hash)),
            });
        }
    };

    // Build a finance-compatible cross-hApp payment payload.
    // Finance bridge `process_payment` expects: source_happ, from_did (payer),
    // to_did (payee), amount, currency, reference.
    //
    // DID format MUST be "did:mycelix:{agent_pubkey}" — finance's
    // verify_caller_is_did compares this string against
    // format!("did:mycelix:{}", agent_info()?.agent_initial_pubkey)
    // (mycelix-finance/zomes/shared/src/lib.rs). This previously used
    // "did:key:{...}", a different prefix that could never match — every
    // settlement attempt was guaranteed to fail auth regardless of whether
    // the underlying agent keys lined up. See MYCELIX_REVIEW.md P1 #4.
    let payload = serde_json::json!({
        "source_happ": "mycelix-supplychain",
        "from_did": format!("did:mycelix:{}", payment.payer),
        "to_did": format!("did:mycelix:{}", payment.payee),
        "amount": payment.amount,
        "currency": payment.currency,
        "reference": payment.reference,
    });

    let encoded =
        ExternIO::encode(payload).map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?;

    match call(
        CallTargetCell::OtherRole("finance".into()),
        ZomeName::from("finance_bridge"),
        FunctionName::from("process_payment"),
        None,
        encoded,
    ) {
        Ok(ZomeCallResponse::Ok(data)) => {
            // Decode the returned Record as a serde_json::Value to extract the ID
            let value: serde_json::Value = data.decode().unwrap_or(serde_json::Value::Null);
            let finance_reference = value
                .get("entry")
                .and_then(|e| e.get("id"))
                .and_then(|id| id.as_str())
                .map(|s| s.to_string());
            Ok(FinanceSettlementResult {
                settled: true,
                finance_reference,
                error: None,
            })
        }
        Ok(other) => {
            // Finance cluster installed but call was rejected (auth, countersigning, etc.)
            Ok(FinanceSettlementResult {
                settled: false,
                finance_reference: None,
                error: Some(format!("Finance cluster rejected settlement: {:?}", other)),
            })
        }
        Err(_) => {
            // Finance cluster not installed or unreachable — degrade gracefully
            Ok(FinanceSettlementResult {
                settled: false,
                finance_reference: None,
                error: Some("Finance cluster not available".to_string()),
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fulfillment_payment_input_serde() {
        let input = FulfillmentPaymentInput {
            po_hash: ActionHash::from_raw_36(vec![5u8; 36]),
        };
        let json = serde_json::to_string(&input).unwrap();
        let back: FulfillmentPaymentInput = serde_json::from_str(&json).unwrap();
        assert_eq!(back.po_hash, ActionHash::from_raw_36(vec![5u8; 36]));
    }

    #[test]
    fn test_fulfillment_payment_result_serde() {
        // escrow_released variant
        let result = FulfillmentPaymentResult {
            action: "escrow_released".to_string(),
            payment_hash: Some(ActionHash::from_raw_36(vec![6u8; 36])),
            amount: 50_000,
            currency: "USD".to_string(),
            settlement_error: None,
        };
        let json = serde_json::to_string(&result).unwrap();
        let back: FulfillmentPaymentResult = serde_json::from_str(&json).unwrap();
        assert_eq!(back.action, "escrow_released");
        assert!(back.payment_hash.is_some());
        assert_eq!(back.amount, 50_000);
        assert_eq!(back.currency, "USD");

        // po_not_received variant
        let result2 = FulfillmentPaymentResult {
            action: "po_not_received".to_string(),
            payment_hash: None,
            amount: 0,
            currency: "EUR".to_string(),
            settlement_error: None,
        };
        let json2 = serde_json::to_string(&result2).unwrap();
        let back2: FulfillmentPaymentResult = serde_json::from_str(&json2).unwrap();
        assert_eq!(back2.action, "po_not_received");
        assert!(back2.payment_hash.is_none());
    }

    #[test]
    fn test_payment_status_serde_roundtrip() {
        let statuses = vec![
            PaymentStatus::Pending,
            PaymentStatus::Authorized,
            PaymentStatus::Captured,
            PaymentStatus::Completed,
            PaymentStatus::Failed,
            PaymentStatus::Refunded,
            PaymentStatus::Disputed,
        ];
        for status in statuses {
            let json = serde_json::to_string(&status).unwrap();
            let back: PaymentStatus = serde_json::from_str(&json).unwrap();
            assert_eq!(back, status);
        }
    }

    #[test]
    fn test_payment_method_serde_roundtrip() {
        let methods = vec![
            PaymentMethod::BankTransfer,
            PaymentMethod::CreditCard,
            PaymentMethod::Crypto,
            PaymentMethod::Escrow,
            PaymentMethod::LetterOfCredit,
        ];
        for method in methods {
            let json = serde_json::to_string(&method).unwrap();
            let back: PaymentMethod = serde_json::from_str(&json).unwrap();
            assert_eq!(back, method);
        }
    }

    #[test]
    fn test_create_payment_input_serde() {
        let input = CreatePaymentInput {
            po_hash: ActionHash::from_raw_36(vec![0u8; 36]),
            amount: 50_000,
            currency: "USD".to_string(),
            method: PaymentMethod::BankTransfer,
            payee: AgentPubKey::from_raw_36(vec![1u8; 36]),
            reference: "INV-2026-001".to_string(),
        };
        let json = serde_json::to_string(&input).unwrap();
        let back: CreatePaymentInput = serde_json::from_str(&json).unwrap();
        assert_eq!(back.amount, 50_000);
        assert_eq!(back.currency, "USD");
        assert_eq!(back.reference, "INV-2026-001");
    }

    #[test]
    fn test_payment_completed_filter() {
        // Simulate the get_po_total_paid filter logic
        struct FakePayment {
            status: PaymentStatus,
            amount: u64,
        }
        let payments = vec![
            FakePayment {
                status: PaymentStatus::Completed,
                amount: 100,
            },
            FakePayment {
                status: PaymentStatus::Pending,
                amount: 200,
            },
            FakePayment {
                status: PaymentStatus::Completed,
                amount: 300,
            },
            FakePayment {
                status: PaymentStatus::Failed,
                amount: 400,
            },
        ];
        let total: u64 = payments
            .iter()
            .filter(|p| p.status == PaymentStatus::Completed)
            .map(|p| p.amount)
            .sum();
        assert_eq!(total, 400); // 100 + 300
    }

    #[test]
    fn test_finance_settlement_result_serde() {
        // settled = true variant
        let result = FinanceSettlementResult {
            settled: true,
            finance_reference: Some("payment:mycelix-supplychain:did:key:abc:1234".to_string()),
            error: None,
        };
        let json = serde_json::to_string(&result).unwrap();
        let back: FinanceSettlementResult = serde_json::from_str(&json).unwrap();
        assert!(back.settled);
        assert!(back.finance_reference.is_some());
        assert!(back.error.is_none());

        // settled = false variant
        let result2 = FinanceSettlementResult {
            settled: false,
            finance_reference: None,
            error: Some("Finance cluster not available".to_string()),
        };
        let json2 = serde_json::to_string(&result2).unwrap();
        let back2: FinanceSettlementResult = serde_json::from_str(&json2).unwrap();
        assert!(!back2.settled);
        assert!(back2.finance_reference.is_none());
        assert!(back2.error.is_some());
    }

    #[test]
    fn test_settlement_result_with_error() {
        let result = FinanceSettlementResult {
            settled: false,
            finance_reference: None,
            error: Some("Finance cluster rejected settlement: Unauthorized".to_string()),
        };
        let json = serde_json::to_string(&result).unwrap();
        let back: FinanceSettlementResult = serde_json::from_str(&json).unwrap();
        assert!(!back.settled);
        assert!(back.error.as_deref().unwrap().contains("rejected"));
    }
}
