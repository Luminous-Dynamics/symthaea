// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Procurement Coordinator Zome
//!
//! Business logic for supply chain procurement operations.
//! Holochain 0.6 compatible (hdk 0.6)

use hdk::prelude::*;
use mycelix_zome_helpers as _;
use procurement_integrity::*;

/// Helper to ensure a path exists and return its entry hash
fn ensure_path(path: Path, link_type: LinkTypes) -> ExternResult<EntryHash> {
    let typed = path.typed(link_type)?;
    typed.ensure()?;
    typed.path_entry_hash()
}

// ============================================================================
// Purchase Order Functions
// ============================================================================

#[derive(Serialize, Deserialize, Debug)]
pub struct CreatePurchaseOrderInput {
    pub po_number: String,
    pub supplier: AgentPubKey,
    pub items: Vec<PurchaseOrderItem>,
    pub currency: String,
    pub due_date: Option<Timestamp>,
    pub notes: Option<String>,
}

#[hdk_extern]
pub fn create_purchase_order(input: CreatePurchaseOrderInput) -> ExternResult<ActionHash> {
    // Input validation
    if input.po_number.is_empty() || input.po_number.len() > 50 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "PO number must be 1-50 characters".to_string()
        )));
    }
    if input.items.is_empty() {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "At least one item is required".to_string()
        )));
    }
    if input.currency.is_empty() || input.currency.len() > 10 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Currency must be 1-10 characters".to_string()
        )));
    }

    let buyer = agent_info()?.agent_initial_pubkey;
    let now = sys_time()?;

    let total_amount: u64 = input
        .items
        .iter()
        .map(|item| item.quantity * item.unit_price)
        .sum();

    if total_amount == 0 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Total amount must be greater than 0".to_string()
        )));
    }

    let po = PurchaseOrder {
        po_number: input.po_number,
        supplier: input.supplier.clone(),
        buyer: buyer.clone(),
        items: input.items,
        total_amount,
        currency: input.currency,
        status: PurchaseOrderStatus::Draft,
        created_at: now,
        due_date: input.due_date,
        notes: input.notes,
        trust_score_at_creation: None,
    };

    let action_hash = create_entry(EntryTypes::PurchaseOrder(po))?;

    // Link from buyer
    let buyer_path = Path::from(format!("buyer_pos/{}", buyer));
    let buyer_hash = ensure_path(buyer_path, LinkTypes::BuyerToPurchaseOrders)?;
    create_link(
        buyer_hash,
        action_hash.clone(),
        LinkTypes::BuyerToPurchaseOrders,
        (),
    )?;

    // Link from supplier
    let supplier_path = Path::from(format!("supplier_pos/{}", input.supplier));
    let supplier_hash = ensure_path(supplier_path, LinkTypes::SupplierToPurchaseOrders)?;
    create_link(
        supplier_hash,
        action_hash.clone(),
        LinkTypes::SupplierToPurchaseOrders,
        (),
    )?;

    Ok(action_hash)
}

#[hdk_extern]
pub fn get_purchase_order(action_hash: ActionHash) -> ExternResult<Option<PurchaseOrder>> {
    let record = get(action_hash, GetOptions::default())?;
    match record {
        Some(r) => Ok(r.entry().to_app_option().map_err(|e| wasm_error!(e))?),
        None => Ok(None),
    }
}

#[hdk_extern]
pub fn get_my_purchase_orders(_: ()) -> ExternResult<Vec<PurchaseOrder>> {
    let my_agent = agent_info()?.agent_initial_pubkey;
    let buyer_path = Path::from(format!("buyer_pos/{}", my_agent));
    let typed_path = buyer_path.typed(LinkTypes::BuyerToPurchaseOrders)?;
    let filter = LinkTypeFilter::try_from(LinkTypes::BuyerToPurchaseOrders)?;
    let links = get_links(
        LinkQuery::new(typed_path.path_entry_hash()?, filter),
        GetStrategy::default(),
    )?;

    let mut orders = Vec::new();
    for link in links {
        if let Some(action_hash) = link.target.into_action_hash() {
            if let Some(po) = get_purchase_order(action_hash)? {
                orders.push(po);
            }
        }
    }
    Ok(orders)
}

#[hdk_extern]
pub fn update_po_status(input: (ActionHash, PurchaseOrderStatus)) -> ExternResult<ActionHash> {
    let (action_hash, new_status) = input;

    let record = get(action_hash.clone(), GetOptions::default())?
        .ok_or_else(|| wasm_error!(WasmErrorInner::Guest("PO not found".to_string())))?;

    let mut po: PurchaseOrder = record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(e))?
        .ok_or_else(|| wasm_error!(WasmErrorInner::Guest("Invalid PO".to_string())))?;

    po.status = new_status;
    update_entry(action_hash, EntryTypes::PurchaseOrder(po))
}

// ============================================================================
// Supplier Profile Functions
// ============================================================================

#[derive(Serialize, Deserialize, Debug)]
pub struct CreateSupplierProfileInput {
    pub company_name: String,
    pub contact_email: String,
    pub contact_phone: Option<String>,
    pub address: Option<String>,
    pub categories: Vec<String>,
    pub certifications: Vec<String>,
    pub payment_terms: String,
    pub lead_time_days: u32,
    pub minimum_order_value: u64,
}

#[hdk_extern]
pub fn create_supplier_profile(input: CreateSupplierProfileInput) -> ExternResult<ActionHash> {
    let my_agent = agent_info()?.agent_initial_pubkey;
    let now = sys_time()?;

    let profile = SupplierProfile {
        agent: my_agent.clone(),
        company_name: input.company_name,
        contact_email: input.contact_email,
        contact_phone: input.contact_phone,
        address: input.address,
        categories: input.categories.clone(),
        certifications: input.certifications,
        payment_terms: input.payment_terms,
        lead_time_days: input.lead_time_days,
        minimum_order_value: input.minimum_order_value,
        created_at: now,
        updated_at: now,
    };

    let action_hash = create_entry(EntryTypes::SupplierProfile(profile))?;

    // Link from agent
    let agent_path = Path::from(format!("supplier_profile/{}", my_agent));
    let agent_hash = ensure_path(agent_path, LinkTypes::AgentToSupplierProfile)?;
    create_link(
        agent_hash,
        action_hash.clone(),
        LinkTypes::AgentToSupplierProfile,
        (),
    )?;

    // Link to all suppliers
    let all_suppliers_path = Path::from("all_suppliers");
    let all_suppliers_hash = ensure_path(all_suppliers_path, LinkTypes::AllSuppliers)?;
    create_link(
        all_suppliers_hash,
        action_hash.clone(),
        LinkTypes::AllSuppliers,
        (),
    )?;

    // Link to categories
    for category in &input.categories {
        let cat_path = Path::from(format!("category/{}", category.to_lowercase()));
        let cat_hash = ensure_path(cat_path, LinkTypes::CategoryToSuppliers)?;
        create_link(
            cat_hash,
            action_hash.clone(),
            LinkTypes::CategoryToSuppliers,
            (),
        )?;
    }

    Ok(action_hash)
}

#[hdk_extern]
pub fn get_supplier_profile(agent: AgentPubKey) -> ExternResult<Option<SupplierProfile>> {
    let agent_path = Path::from(format!("supplier_profile/{}", agent));
    let typed_path = agent_path.typed(LinkTypes::AgentToSupplierProfile)?;
    let filter = LinkTypeFilter::try_from(LinkTypes::AgentToSupplierProfile)?;
    let links = get_links(
        LinkQuery::new(typed_path.path_entry_hash()?, filter),
        GetStrategy::default(),
    )?;

    if let Some(link) = links.last() {
        if let Some(action_hash) = link.target.clone().into_action_hash() {
            let record = get(action_hash, GetOptions::default())?;
            return match record {
                Some(r) => Ok(r.entry().to_app_option().map_err(|e| wasm_error!(e))?),
                None => Ok(None),
            };
        }
    }
    Ok(None)
}

#[hdk_extern]
pub fn get_all_suppliers(_: ()) -> ExternResult<Vec<SupplierProfile>> {
    let all_suppliers_path = Path::from("all_suppliers");
    let typed_path = all_suppliers_path.typed(LinkTypes::AllSuppliers)?;
    let filter = LinkTypeFilter::try_from(LinkTypes::AllSuppliers)?;
    let links = get_links(
        LinkQuery::new(typed_path.path_entry_hash()?, filter),
        GetStrategy::default(),
    )?;

    let mut suppliers = Vec::new();
    for link in links {
        if let Some(action_hash) = link.target.into_action_hash() {
            if let Some(record) = get(action_hash, GetOptions::default())? {
                if let Some(profile) = record
                    .entry()
                    .to_app_option::<SupplierProfile>()
                    .map_err(|e| wasm_error!(e))?
                {
                    suppliers.push(profile);
                }
            }
        }
    }
    Ok(suppliers)
}

#[hdk_extern]
pub fn get_suppliers_by_category(category: String) -> ExternResult<Vec<SupplierProfile>> {
    let cat_path = Path::from(format!("category/{}", category.to_lowercase()));
    let typed_path = cat_path.typed(LinkTypes::CategoryToSuppliers)?;
    let filter = LinkTypeFilter::try_from(LinkTypes::CategoryToSuppliers)?;
    let links = get_links(
        LinkQuery::new(typed_path.path_entry_hash()?, filter),
        GetStrategy::default(),
    )?;

    let mut suppliers = Vec::new();
    for link in links {
        if let Some(action_hash) = link.target.into_action_hash() {
            if let Some(record) = get(action_hash, GetOptions::default())? {
                if let Some(profile) = record
                    .entry()
                    .to_app_option::<SupplierProfile>()
                    .map_err(|e| wasm_error!(e))?
                {
                    suppliers.push(profile);
                }
            }
        }
    }
    Ok(suppliers)
}

// ============================================================================
// RFQ Functions
// ============================================================================

#[derive(Serialize, Deserialize, Debug)]
pub struct CreateRfqInput {
    pub rfq_number: String,
    pub items: Vec<RfqItem>,
    pub deadline: Timestamp,
    pub notes: Option<String>,
}

#[hdk_extern]
pub fn create_rfq(input: CreateRfqInput) -> ExternResult<ActionHash> {
    let buyer = agent_info()?.agent_initial_pubkey;
    let now = sys_time()?;

    let rfq = RequestForQuotation {
        rfq_number: input.rfq_number,
        buyer: buyer.clone(),
        items: input.items,
        deadline: input.deadline,
        notes: input.notes,
        status: RfqStatus::Open,
        created_at: now,
    };

    let action_hash = create_entry(EntryTypes::RequestForQuotation(rfq))?;

    // Link from buyer
    let buyer_path = Path::from(format!("buyer_rfqs/{}", buyer));
    let buyer_hash = ensure_path(buyer_path, LinkTypes::BuyerToRfqs)?;
    create_link(buyer_hash, action_hash.clone(), LinkTypes::BuyerToRfqs, ())?;

    Ok(action_hash)
}

#[hdk_extern]
pub fn submit_quotation(input: Quotation) -> ExternResult<ActionHash> {
    let supplier = agent_info()?.agent_initial_pubkey;

    let mut quote = input;
    quote.supplier = supplier.clone();
    quote.submitted_at = sys_time()?;

    let action_hash = create_entry(EntryTypes::Quotation(quote.clone()))?;

    // Link from RFQ
    create_link(
        quote.rfq_hash.clone(),
        action_hash.clone(),
        LinkTypes::RfqToQuotations,
        (),
    )?;

    // Link from supplier
    let supplier_path = Path::from(format!("supplier_quotes/{}", supplier));
    let supplier_hash = ensure_path(supplier_path, LinkTypes::SupplierToQuotations)?;
    create_link(
        supplier_hash,
        action_hash.clone(),
        LinkTypes::SupplierToQuotations,
        (),
    )?;

    Ok(action_hash)
}

#[hdk_extern]
pub fn get_quotations_for_rfq(rfq_hash: ActionHash) -> ExternResult<Vec<Quotation>> {
    let filter = LinkTypeFilter::try_from(LinkTypes::RfqToQuotations)?;
    let links = get_links(LinkQuery::new(rfq_hash, filter), GetStrategy::default())?;

    let mut quotes = Vec::new();
    for link in links {
        if let Some(action_hash) = link.target.into_action_hash() {
            if let Some(record) = get(action_hash, GetOptions::default())? {
                if let Some(quote) = record
                    .entry()
                    .to_app_option::<Quotation>()
                    .map_err(|e| wasm_error!(e))?
                {
                    quotes.push(quote);
                }
            }
        }
    }
    Ok(quotes)
}

#[hdk_extern]
pub fn approve_purchase_order(po_hash: ActionHash) -> ExternResult<ActionHash> {
    update_po_status((po_hash, PurchaseOrderStatus::Approved))
}

#[hdk_extern]
pub fn fulfill_purchase_order(po_hash: ActionHash) -> ExternResult<ActionHash> {
    update_po_status((po_hash, PurchaseOrderStatus::Received))
}

#[hdk_extern]
pub fn cancel_purchase_order(po_hash: ActionHash) -> ExternResult<ActionHash> {
    update_po_status((po_hash, PurchaseOrderStatus::Cancelled))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_score_supplier_perfect() {
        // Max reputation (5.0), min lead time (1 out of 30), tiny min order
        let score = score_supplier(5.0, 1, 30, 1, 1000);
        // rep_norm=1.0, lead_score=1-(1/30)≈0.9667, order_fit=(1-1/1000)≈0.999
        // 1.0*0.4 + 0.9667*0.3 + 0.999*0.3 ≈ 0.4 + 0.290 + 0.300 = 0.990
        assert!(
            score > 0.9,
            "Perfect supplier should score > 0.9, got {}",
            score
        );
    }

    #[test]
    fn test_score_supplier_worst() {
        // Zero reputation, max lead time, unaffordable min order
        let score = score_supplier(0.0, 30, 30, 10_000, 1);
        // rep_norm=0.0, lead_score=1-(30/30)=0.0, order_fit clamped to 0.0
        assert_eq!(score, 0.0, "Worst supplier should score 0.0");
    }

    #[test]
    fn test_score_supplier_balanced() {
        // Mid reputation (2.5/5), mid lead time (15/30), order fits half
        let score = score_supplier(2.5, 15, 30, 500, 1000);
        // rep_norm=0.5, lead_score=0.5, order_fit=0.5
        // 0.5*0.4 + 0.5*0.3 + 0.5*0.3 = 0.2 + 0.15 + 0.15 = 0.5
        assert!(
            (score - 0.5).abs() < 0.001,
            "Balanced supplier should score ~0.5, got {}",
            score
        );
    }

    #[test]
    fn test_score_max_lead_zero_guard() {
        // max_lead_time=0 should not panic (guarded by .max(1))
        let score = score_supplier(3.0, 0, 0, 100, 500);
        // max_lt=1, lead_score=1-(0/1)=1.0
        assert!(
            score.is_finite(),
            "Score must be finite even when max_lead=0"
        );
    }

    #[test]
    fn test_supplier_selection_input_serde() {
        let input = SupplierSelectionInput {
            category: "electronics".to_string(),
            required_quantity: 1000,
        };
        let json = serde_json::to_string(&input).unwrap();
        let back: SupplierSelectionInput = serde_json::from_str(&json).unwrap();
        assert_eq!(back.category, "electronics");
        assert_eq!(back.required_quantity, 1000);
    }

    #[test]
    fn test_ranked_supplier_serde() {
        let rs = RankedSupplier {
            agent: AgentPubKey::from_raw_36(vec![2u8; 36]),
            company_name: "BestParts Inc".to_string(),
            score: 0.85,
            reputation_score: 4.2,
            lead_time_days: 7,
            minimum_order_value: 500,
        };
        let json = serde_json::to_string(&rs).unwrap();
        let back: RankedSupplier = serde_json::from_str(&json).unwrap();
        assert_eq!(back.company_name, "BestParts Inc");
        assert!((back.score - 0.85).abs() < 0.001);
        assert_eq!(back.lead_time_days, 7);
        assert_eq!(back.minimum_order_value, 500);
    }

    /// All valid PurchaseOrderStatus transitions used in the coordinator.
    /// Draft → Submitted → Approved → Received (happy path via approve/fulfill).
    /// Draft → Cancelled, any → Cancelled (cancel_purchase_order).
    #[test]
    fn test_po_status_serde_roundtrip() {
        let statuses = vec![
            PurchaseOrderStatus::Draft,
            PurchaseOrderStatus::Submitted,
            PurchaseOrderStatus::Approved,
            PurchaseOrderStatus::Rejected,
            PurchaseOrderStatus::Sent,
            PurchaseOrderStatus::Acknowledged,
            PurchaseOrderStatus::PartiallyReceived,
            PurchaseOrderStatus::Received,
            PurchaseOrderStatus::Cancelled,
            PurchaseOrderStatus::Closed,
        ];
        for status in statuses {
            let json = serde_json::to_string(&status).unwrap();
            let back: PurchaseOrderStatus = serde_json::from_str(&json).unwrap();
            assert_eq!(back, status);
        }
    }

    #[test]
    fn test_po_status_approve_path() {
        // Verify the status values used in approve/fulfill/cancel helpers
        let approved = PurchaseOrderStatus::Approved;
        let received = PurchaseOrderStatus::Received;
        let cancelled = PurchaseOrderStatus::Cancelled;
        assert_ne!(approved, received);
        assert_ne!(approved, cancelled);
        assert_ne!(received, cancelled);
    }

    #[test]
    fn test_purchase_order_item_serde() {
        let item = PurchaseOrderItem {
            item_code: "BOLT-M6-50".to_string(),
            description: "M6 bolt 50mm".to_string(),
            quantity: 1000,
            unit_price: 5,
            unit: "each".to_string(),
        };
        let json = serde_json::to_string(&item).unwrap();
        let back: PurchaseOrderItem = serde_json::from_str(&json).unwrap();
        assert_eq!(back.item_code, "BOLT-M6-50");
        assert_eq!(back.quantity, 1000);
        assert_eq!(back.unit_price, 5);
    }

    #[test]
    fn test_total_amount_calculation() {
        // Reproduce the total_amount logic from create_purchase_order
        let items = vec![
            PurchaseOrderItem {
                item_code: "A".to_string(),
                description: "".to_string(),
                quantity: 10,
                unit_price: 100,
                unit: "each".to_string(),
            },
            PurchaseOrderItem {
                item_code: "B".to_string(),
                description: "".to_string(),
                quantity: 5,
                unit_price: 200,
                unit: "each".to_string(),
            },
        ];
        let total: u64 = items.iter().map(|i| i.quantity * i.unit_price).sum();
        assert_eq!(total, 2000); // 10*100 + 5*200
    }

    #[test]
    fn test_create_purchase_order_input_serde() {
        let input = CreatePurchaseOrderInput {
            po_number: "PO-2026-001".to_string(),
            supplier: AgentPubKey::from_raw_36(vec![0u8; 36]),
            items: vec![PurchaseOrderItem {
                item_code: "SKU-001".to_string(),
                description: "Widget".to_string(),
                quantity: 100,
                unit_price: 50,
                unit: "pcs".to_string(),
            }],
            currency: "USD".to_string(),
            due_date: None,
            notes: Some("Urgent".to_string()),
        };
        let json = serde_json::to_string(&input).unwrap();
        let back: CreatePurchaseOrderInput = serde_json::from_str(&json).unwrap();
        assert_eq!(back.po_number, "PO-2026-001");
        assert_eq!(back.items.len(), 1);
        assert_eq!(back.currency, "USD");
    }
}

// ============================================================================
// Workflow 2: Supplier Selection
// ============================================================================

#[derive(Serialize, Deserialize, Debug)]
pub struct SupplierSelectionInput {
    pub category: String,
    pub required_quantity: u64,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct RankedSupplier {
    pub agent: AgentPubKey,
    pub company_name: String,
    pub score: f64,
    pub reputation_score: f64,
    pub lead_time_days: u32,
    pub minimum_order_value: u64,
}

/// Pure scoring function — testable without HDK context.
fn score_supplier(
    reputation: f64,
    lead_time_days: u32,
    max_lead_time: u32,
    min_order_value: u64,
    required_quantity: u64,
) -> f64 {
    let rep_norm = (reputation / 5.0).clamp(0.0, 1.0);
    let max_lt = max_lead_time.max(1) as f64;
    let lead_score = 1.0 - (lead_time_days as f64 / max_lt);
    let order_fit = if required_quantity == 0 {
        0.0
    } else {
        (1.0 - (min_order_value as f64 / required_quantity as f64)).clamp(0.0, 1.0)
    };
    rep_norm * 0.4 + lead_score * 0.3 + order_fit * 0.3
}

#[hdk_extern]
pub fn select_best_supplier(input: SupplierSelectionInput) -> ExternResult<Vec<RankedSupplier>> {
    let suppliers = get_suppliers_by_category(input.category)?;

    if suppliers.is_empty() {
        return Ok(vec![]);
    }

    // Find max lead time for normalization
    let max_lead_time = suppliers
        .iter()
        .map(|s| s.lead_time_days)
        .max()
        .unwrap_or(1);

    let mut ranked: Vec<RankedSupplier> = Vec::new();

    for supplier in &suppliers {
        // Get reputation rating from trust zome
        let rating_result = call(
            CallTargetCell::Local,
            "trust_coordinator",
            "get_provider_rating".into(),
            None,
            supplier.agent.clone(),
        );

        let reputation_score = match rating_result {
            Ok(ZomeCallResponse::Ok(result)) => result.decode::<f64>().unwrap_or(0.0),
            _ => 0.0,
        };

        let score = score_supplier(
            reputation_score,
            supplier.lead_time_days,
            max_lead_time,
            supplier.minimum_order_value,
            input.required_quantity,
        );

        ranked.push(RankedSupplier {
            agent: supplier.agent.clone(),
            company_name: supplier.company_name.clone(),
            score,
            reputation_score,
            lead_time_days: supplier.lead_time_days,
            minimum_order_value: supplier.minimum_order_value,
        });
    }

    // Sort descending by score
    ranked.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    Ok(ranked)
}

#[hdk_extern]
pub fn get_supplier_orders(supplier: AgentPubKey) -> ExternResult<Vec<PurchaseOrder>> {
    let supplier_path = Path::from(format!("supplier_pos/{}", supplier));
    let typed_path = supplier_path.typed(LinkTypes::SupplierToPurchaseOrders)?;
    let filter = LinkTypeFilter::try_from(LinkTypes::SupplierToPurchaseOrders)?;
    let links = get_links(
        LinkQuery::new(typed_path.path_entry_hash()?, filter),
        GetStrategy::default(),
    )?;

    let mut orders = Vec::new();
    for link in links {
        if let Some(action_hash) = link.target.into_action_hash() {
            if let Some(po) = get_purchase_order(action_hash)? {
                orders.push(po);
            }
        }
    }
    Ok(orders)
}
