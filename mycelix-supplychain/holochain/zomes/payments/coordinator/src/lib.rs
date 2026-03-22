//! Payments Coordinator Zome - Business logic for payment processing
use hdk::prelude::*;
use payments_integrity::*;

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
    create_link(input.po_hash, action_hash.clone(), LinkTypes::PoToPayments, ())?;

    let payer_path = Path::from(format!("payer/{}", payer));
    let payer_hash = ensure_path(payer_path, LinkTypes::PayerToPayments)?;
    create_link(payer_hash, action_hash.clone(), LinkTypes::PayerToPayments, ())?;

    let payee_path = Path::from(format!("payee/{}", input.payee));
    let payee_hash = ensure_path(payee_path, LinkTypes::PayeeToPayments)?;
    create_link(payee_hash, action_hash.clone(), LinkTypes::PayeeToPayments, ())?;

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
        if let Some(mut payment) = record.entry().to_app_option::<Payment>().map_err(|e| wasm_error!(e))? {
            payment.status = new_status.clone();
            if new_status == PaymentStatus::Completed {
                payment.completed_at = Some(sys_time()?);
            }
            return update_entry(hash, EntryTypes::Payment(payment));
        }
    }
    Err(wasm_error!(WasmErrorInner::Guest("Payment not found".into())))
}

#[hdk_extern]
pub fn create_invoice(invoice: Invoice) -> ExternResult<ActionHash> {
    let action_hash = create_entry(EntryTypes::Invoice(invoice.clone()))?;
    create_link(invoice.po_hash, action_hash.clone(), LinkTypes::PoToInvoices, ())?;
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
    create_link(escrow.po_hash, action_hash.clone(), LinkTypes::PoToEscrow, ())?;
    Ok(action_hash)
}

#[hdk_extern]
pub fn fund_escrow(hash: ActionHash) -> ExternResult<ActionHash> {
    if let Some(record) = get(hash.clone(), GetOptions::default())? {
        if let Some(mut escrow) = record.entry().to_app_option::<EscrowAccount>().map_err(|e| wasm_error!(e))? {
            escrow.funded_at = Some(sys_time()?);
            return update_entry(hash, EntryTypes::EscrowAccount(escrow));
        }
    }
    Err(wasm_error!(WasmErrorInner::Guest("Escrow not found".into())))
}

#[hdk_extern]
pub fn release_escrow(hash: ActionHash) -> ExternResult<ActionHash> {
    if let Some(record) = get(hash.clone(), GetOptions::default())? {
        if let Some(mut escrow) = record.entry().to_app_option::<EscrowAccount>().map_err(|e| wasm_error!(e))? {
            if escrow.funded_at.is_none() {
                return Err(wasm_error!(WasmErrorInner::Guest("Escrow not funded".into())));
            }
            escrow.released_at = Some(sys_time()?);
            return update_entry(hash, EntryTypes::EscrowAccount(escrow));
        }
    }
    Err(wasm_error!(WasmErrorInner::Guest("Escrow not found".into())))
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
    let links = get_links(LinkQuery::new(typed.path_entry_hash()?, filter), GetStrategy::default())?;

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
pub fn get_po_total_paid(po_hash: ActionHash) -> ExternResult<u64> {
    let payments = get_po_payments(po_hash)?;
    let total: u64 = payments.iter()
        .filter(|p| p.status == PaymentStatus::Completed)
        .map(|p| p.amount)
        .sum();
    Ok(total)
}

#[cfg(test)]
mod tests {
    use super::*;

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
        struct FakePayment { status: PaymentStatus, amount: u64 }
        let payments = vec![
            FakePayment { status: PaymentStatus::Completed, amount: 100 },
            FakePayment { status: PaymentStatus::Pending, amount: 200 },
            FakePayment { status: PaymentStatus::Completed, amount: 300 },
            FakePayment { status: PaymentStatus::Failed, amount: 400 },
        ];
        let total: u64 = payments.iter()
            .filter(|p| p.status == PaymentStatus::Completed)
            .map(|p| p.amount)
            .sum();
        assert_eq!(total, 400); // 100 + 300
    }
}
