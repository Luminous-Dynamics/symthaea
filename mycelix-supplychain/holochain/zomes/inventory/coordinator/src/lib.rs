//! Inventory Coordinator Zome - Business logic for inventory management
use hdk::prelude::*;
use inventory_integrity::*;

fn ensure_path(path: Path, link_type: LinkTypes) -> ExternResult<EntryHash> {
    let typed = path.typed(link_type)?;
    typed.ensure()?;
    typed.path_entry_hash()
}

// ============================================================================
// Inventory Item Management
// ============================================================================

#[derive(Serialize, Deserialize, Debug)]
pub struct CreateItemInput {
    pub sku: String,
    pub name: String,
    pub description: Option<String>,
    pub category: String,
    pub unit: String,
    pub reorder_point: u64,
    pub reorder_quantity: u64,
}

#[hdk_extern]
pub fn create_item(input: CreateItemInput) -> ExternResult<ActionHash> {
    // Input validation
    if input.sku.is_empty() || input.sku.len() > 100 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "SKU must be 1-100 characters".to_string()
        )));
    }
    if input.name.is_empty() || input.name.len() > 200 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Name must be 1-200 characters".to_string()
        )));
    }
    if input.category.is_empty() {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Category is required".to_string()
        )));
    }
    if input.unit.is_empty() {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Unit is required".to_string()
        )));
    }

    let item = InventoryItem {
        sku: input.sku,
        name: input.name,
        description: input.description,
        category: input.category.clone(),
        unit: input.unit,
        reorder_point: input.reorder_point,
        reorder_quantity: input.reorder_quantity,
        created_at: sys_time()?,
    };

    let action_hash = create_entry(EntryTypes::InventoryItem(item.clone()))?;

    // Link to all items
    let all_path = Path::from("all_items");
    let all_hash = ensure_path(all_path, LinkTypes::AllItems)?;
    create_link(all_hash, action_hash.clone(), LinkTypes::AllItems, ())?;

    // Link to category
    let cat_path = Path::from(format!("category/{}", item.category.to_lowercase()));
    let cat_hash = ensure_path(cat_path, LinkTypes::CategoryToItems)?;
    create_link(cat_hash, action_hash.clone(), LinkTypes::CategoryToItems, ())?;

    Ok(action_hash)
}

#[hdk_extern]
pub fn get_item(hash: ActionHash) -> ExternResult<Option<InventoryItem>> {
    match get(hash, GetOptions::default())? {
        Some(r) => Ok(r.entry().to_app_option().map_err(|e| wasm_error!(e))?),
        None => Ok(None),
    }
}

#[hdk_extern]
pub fn get_all_items(_: ()) -> ExternResult<Vec<InventoryItem>> {
    let path = Path::from("all_items");
    let typed = path.typed(LinkTypes::AllItems)?;
    let filter = LinkTypeFilter::try_from(LinkTypes::AllItems)?;
    let links = get_links(LinkQuery::new(typed.path_entry_hash()?, filter), GetStrategy::default())?;

    let mut items = Vec::new();
    for link in links {
        if let Some(hash) = link.target.into_action_hash() {
            if let Some(item) = get_item(hash)? {
                items.push(item);
            }
        }
    }
    Ok(items)
}

#[hdk_extern]
pub fn get_items_by_category(category: String) -> ExternResult<Vec<InventoryItem>> {
    let cat_path = Path::from(format!("category/{}", category.to_lowercase()));
    let typed = cat_path.typed(LinkTypes::CategoryToItems)?;
    let filter = LinkTypeFilter::try_from(LinkTypes::CategoryToItems)?;
    let links = get_links(LinkQuery::new(typed.path_entry_hash()?, filter), GetStrategy::default())?;

    let mut items = Vec::new();
    for link in links {
        if let Some(hash) = link.target.into_action_hash() {
            if let Some(item) = get_item(hash)? {
                items.push(item);
            }
        }
    }
    Ok(items)
}

// ============================================================================
// Stock Level Management
// ============================================================================

#[derive(Serialize, Deserialize, Debug)]
pub struct UpdateStockInput {
    pub item_hash: ActionHash,
    pub location: String,
    pub quantity: u64,
    pub reserved: u64,
}

#[hdk_extern]
pub fn update_stock(input: UpdateStockInput) -> ExternResult<ActionHash> {
    // Validate input
    if input.location.is_empty() || input.location.len() > 100 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Location must be 1-100 characters".to_string()
        )));
    }

    // Verify item exists
    if get_item(input.item_hash.clone())?.is_none() {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Item not found".to_string()
        )));
    }

    let stock_level = StockLevel {
        item_hash: input.item_hash.clone(),
        location: input.location.clone(),
        quantity: input.quantity,
        reserved: input.reserved,
        updated_at: sys_time()?,
    };

    let action_hash = create_entry(EntryTypes::StockLevel(stock_level.clone()))?;

    // Link from item to stock levels
    create_link(input.item_hash, action_hash.clone(), LinkTypes::ItemToStockLevels, ())?;

    // Link from location to stock
    let loc_path = Path::from(format!("location/{}", stock_level.location));
    let loc_hash = ensure_path(loc_path, LinkTypes::LocationToStock)?;
    create_link(loc_hash, action_hash.clone(), LinkTypes::LocationToStock, ())?;

    Ok(action_hash)
}

#[hdk_extern]
pub fn get_stock_levels(item_hash: ActionHash) -> ExternResult<Vec<StockLevel>> {
    let filter = LinkTypeFilter::try_from(LinkTypes::ItemToStockLevels)?;
    let links = get_links(LinkQuery::new(item_hash, filter), GetStrategy::default())?;

    let mut levels = Vec::new();
    for link in links {
        if let Some(hash) = link.target.into_action_hash() {
            if let Some(record) = get(hash, GetOptions::default())? {
                if let Some(level) = record.entry().to_app_option::<StockLevel>().map_err(|e| wasm_error!(e))? {
                    levels.push(level);
                }
            }
        }
    }
    Ok(levels)
}

#[hdk_extern]
pub fn get_stock_by_location(location: String) -> ExternResult<Vec<StockLevel>> {
    let loc_path = Path::from(format!("location/{}", location));
    let typed = loc_path.typed(LinkTypes::LocationToStock)?;
    let filter = LinkTypeFilter::try_from(LinkTypes::LocationToStock)?;
    let links = get_links(LinkQuery::new(typed.path_entry_hash()?, filter), GetStrategy::default())?;

    let mut levels = Vec::new();
    for link in links {
        if let Some(hash) = link.target.into_action_hash() {
            if let Some(record) = get(hash, GetOptions::default())? {
                if let Some(level) = record.entry().to_app_option::<StockLevel>().map_err(|e| wasm_error!(e))? {
                    levels.push(level);
                }
            }
        }
    }
    Ok(levels)
}

#[hdk_extern]
pub fn get_total_stock(item_hash: ActionHash) -> ExternResult<u64> {
    let levels = get_stock_levels(item_hash)?;
    let total: u64 = levels.iter().map(|l| l.quantity).sum();
    Ok(total)
}

#[hdk_extern]
pub fn get_available_stock(item_hash: ActionHash) -> ExternResult<u64> {
    let levels = get_stock_levels(item_hash)?;
    let available: u64 = levels.iter()
        .map(|l| l.quantity.saturating_sub(l.reserved))
        .sum();
    Ok(available)
}

// ============================================================================
// Stock Movement Tracking
// ============================================================================

#[derive(Serialize, Deserialize, Debug)]
pub struct RecordMovementInput {
    pub item_hash: ActionHash,
    pub movement_type: MovementType,
    pub quantity: u64,
    pub from_location: Option<String>,
    pub to_location: Option<String>,
    pub reference: Option<String>,
    pub notes: Option<String>,
}

#[hdk_extern]
pub fn record_movement(input: RecordMovementInput) -> ExternResult<ActionHash> {
    // Validate input
    if input.quantity == 0 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Quantity must be greater than 0".to_string()
        )));
    }

    // Verify item exists
    if get_item(input.item_hash.clone())?.is_none() {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Item not found".to_string()
        )));
    }

    // Validate locations based on movement type
    match input.movement_type {
        MovementType::Transfer => {
            if input.from_location.is_none() || input.to_location.is_none() {
                return Err(wasm_error!(WasmErrorInner::Guest(
                    "Transfer requires both from_location and to_location".to_string()
                )));
            }
        }
        MovementType::Inbound => {
            if input.to_location.is_none() {
                return Err(wasm_error!(WasmErrorInner::Guest(
                    "Inbound requires to_location".to_string()
                )));
            }
        }
        MovementType::Outbound => {
            if input.from_location.is_none() {
                return Err(wasm_error!(WasmErrorInner::Guest(
                    "Outbound requires from_location".to_string()
                )));
            }
        }
        _ => {}
    }

    let movement = StockMovement {
        item_hash: input.item_hash.clone(),
        movement_type: input.movement_type,
        quantity: input.quantity,
        from_location: input.from_location,
        to_location: input.to_location,
        reference: input.reference,
        notes: input.notes,
        created_at: sys_time()?,
        created_by: agent_info()?.agent_initial_pubkey,
    };

    let action_hash = create_entry(EntryTypes::StockMovement(movement))?;
    create_link(input.item_hash, action_hash.clone(), LinkTypes::ItemToMovements, ())?;

    Ok(action_hash)
}

#[hdk_extern]
pub fn get_item_movements(item_hash: ActionHash) -> ExternResult<Vec<StockMovement>> {
    let filter = LinkTypeFilter::try_from(LinkTypes::ItemToMovements)?;
    let links = get_links(LinkQuery::new(item_hash, filter), GetStrategy::default())?;

    let mut movements = Vec::new();
    for link in links {
        if let Some(hash) = link.target.into_action_hash() {
            if let Some(record) = get(hash, GetOptions::default())? {
                if let Some(movement) = record.entry().to_app_option::<StockMovement>().map_err(|e| wasm_error!(e))? {
                    movements.push(movement);
                }
            }
        }
    }
    Ok(movements)
}

// ============================================================================
// Advanced Queries
// ============================================================================

#[hdk_extern]
pub fn get_low_stock_items(_: ()) -> ExternResult<Vec<(InventoryItem, u64)>> {
    let items = get_all_items(())?;
    let mut low_stock = Vec::new();

    for item in items {
        // Get the item's action hash by looking it up
        let all_path = Path::from("all_items");
        let typed = all_path.typed(LinkTypes::AllItems)?;
        let filter = LinkTypeFilter::try_from(LinkTypes::AllItems)?;
        let links = get_links(LinkQuery::new(typed.path_entry_hash()?, filter), GetStrategy::default())?;

        for link in links {
            if let Some(hash) = link.target.clone().into_action_hash() {
                if let Some(found_item) = get_item(hash.clone())? {
                    if found_item.sku == item.sku {
                        let total = get_total_stock(hash)?;
                        if total < item.reorder_point {
                            low_stock.push((item.clone(), total));
                        }
                        break;
                    }
                }
            }
        }
    }

    Ok(low_stock)
}
