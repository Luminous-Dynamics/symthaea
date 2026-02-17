//! Procurement Coordinator Zome
//!
//! Business logic for supply chain procurement operations.
//! Holochain 0.6 compatible (hdk 0.6)

use hdk::prelude::*;
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

    let total_amount: u64 = input.items.iter()
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
    create_link(buyer_hash, action_hash.clone(), LinkTypes::BuyerToPurchaseOrders, ())?;

    // Link from supplier
    let supplier_path = Path::from(format!("supplier_pos/{}", input.supplier));
    let supplier_hash = ensure_path(supplier_path, LinkTypes::SupplierToPurchaseOrders)?;
    create_link(supplier_hash, action_hash.clone(), LinkTypes::SupplierToPurchaseOrders, ())?;

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

    let mut po: PurchaseOrder = record.entry()
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
    create_link(agent_hash, action_hash.clone(), LinkTypes::AgentToSupplierProfile, ())?;

    // Link to all suppliers
    let all_suppliers_path = Path::from("all_suppliers");
    let all_suppliers_hash = ensure_path(all_suppliers_path, LinkTypes::AllSuppliers)?;
    create_link(all_suppliers_hash, action_hash.clone(), LinkTypes::AllSuppliers, ())?;

    // Link to categories
    for category in &input.categories {
        let cat_path = Path::from(format!("category/{}", category.to_lowercase()));
        let cat_hash = ensure_path(cat_path, LinkTypes::CategoryToSuppliers)?;
        create_link(cat_hash, action_hash.clone(), LinkTypes::CategoryToSuppliers, ())?;
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
                if let Some(profile) = record.entry().to_app_option::<SupplierProfile>().map_err(|e| wasm_error!(e))? {
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
                if let Some(profile) = record.entry().to_app_option::<SupplierProfile>().map_err(|e| wasm_error!(e))? {
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
    create_link(quote.rfq_hash.clone(), action_hash.clone(), LinkTypes::RfqToQuotations, ())?;

    // Link from supplier
    let supplier_path = Path::from(format!("supplier_quotes/{}", supplier));
    let supplier_hash = ensure_path(supplier_path, LinkTypes::SupplierToQuotations)?;
    create_link(supplier_hash, action_hash.clone(), LinkTypes::SupplierToQuotations, ())?;

    Ok(action_hash)
}

#[hdk_extern]
pub fn get_quotations_for_rfq(rfq_hash: ActionHash) -> ExternResult<Vec<Quotation>> {
    let filter = LinkTypeFilter::try_from(LinkTypes::RfqToQuotations)?;
    let links = get_links(
        LinkQuery::new(rfq_hash, filter),
        GetStrategy::default(),
    )?;

    let mut quotes = Vec::new();
    for link in links {
        if let Some(action_hash) = link.target.into_action_hash() {
            if let Some(record) = get(action_hash, GetOptions::default())? {
                if let Some(quote) = record.entry().to_app_option::<Quotation>().map_err(|e| wasm_error!(e))? {
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
