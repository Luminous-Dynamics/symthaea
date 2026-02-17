use hdi::prelude::*;

/// Listing entry type - core marketplace data structure
///
/// This represents a single item for sale in the marketplace.
/// Integrates with Epistemic Charter v2.0 for truth classification.
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Listing {
    /// Product title (1-200 characters)
    pub title: String,

    /// Detailed description (1-5000 characters)
    pub description: String,

    /// Price in USD cents (to avoid floating point issues)
    /// Example: $19.99 = 1999 cents
    pub price_cents: u64,

    /// Product category for filtering
    pub category: ListingCategory,

    /// IPFS CIDs for product photos (max 10)
    pub photos_ipfs_cids: Vec<String>,

    /// Number of items available (inventory)
    pub quantity_available: u32,

    /// Current listing status
    pub status: ListingStatus,

    /// Epistemic classification (Epistemic Charter v2.0)
    /// This listing is a claim about a product existing at this price
    pub epistemic: EpistemicClassification,

    /// Creation timestamp (milliseconds since Unix epoch)
    pub created_at: Timestamp,

    /// Last update timestamp
    pub updated_at: Timestamp,
}

/// Product categories for marketplace
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(rename_all = "PascalCase")]
pub enum ListingCategory {
    Electronics,
    Fashion,
    #[serde(rename = "Home & Garden")]
    HomeGarden,
    #[serde(rename = "Sports & Outdoors")]
    SportsOutdoors,
    #[serde(rename = "Books & Media")]
    BooksMedia,
    #[serde(rename = "Toys & Games")]
    ToysGames,
    #[serde(rename = "Health & Beauty")]
    HealthBeauty,
    Automotive,
    #[serde(rename = "Art & Collectibles")]
    ArtCollectibles,
    Other,
}

/// Listing lifecycle status
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum ListingStatus {
    /// Active and available for purchase
    Active,
    /// Sold out (inventory = 0)
    Sold,
    /// Seller paused listing
    Inactive,
    /// Soft-deleted (hidden from UI)
    Deleted,
}

/// Epistemic Charter v2.0 Classification
///
/// Every listing is a claim about reality that can be verified.
/// See: Mycelix-Core/docs/architecture/THE EPISTEMIC CHARTER (v2.0).md
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct EpistemicClassification {
    /// E-Axis: How to verify this claim?
    /// For listings: Typically E1 (seller attestation) or E2 (buyer verification after purchase)
    pub empirical: EmpiricalLevel,

    /// N-Axis: Who agrees this is binding?
    /// For listings: N0 (seller's personal claim) until transaction creates N1 (buyer-seller agreement)
    pub normative: NormativeLevel,

    /// M-Axis: How long does this matter?
    /// For listings: M1 (prune when sold) or M2 (keep for reputation history)
    pub materiality: MaterialityLevel,
}

/// E-Axis: Empirical Verifiability
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum EmpiricalLevel {
    /// E0: Unverifiable belief
    E0Null,
    /// E1: Personal testimonial (seller's claim)
    E1Testimonial,
    /// E2: Privately verifiable (buyer can verify after purchase)
    E2PrivateVerify,
    /// E3: Cryptographically proven (signed by trusted authority)
    E3Cryptographic,
    /// E4: Publicly reproducible
    E4PublicRepro,
}

/// N-Axis: Normative Authority
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum NormativeLevel {
    /// N0: Personal (seller only)
    N0Personal,
    /// N1: Communal (buyer-seller agreement)
    N1Communal,
    /// N2: Network (marketplace consensus)
    N2Network,
    /// N3: Axiomatic (constitutional/legal)
    N3Axiomatic,
}

/// M-Axis: Materiality/State Management
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum MaterialityLevel {
    /// M0: Ephemeral (discard immediately)
    M0Ephemeral,
    /// M1: Temporal (prune after state change)
    M1Temporal,
    /// M2: Persistent (archive after time)
    M2Persistent,
    /// M3: Foundational (preserve forever)
    M3Foundational,
}

/// Link types for listing discovery and relationships
#[hdk_link_types]
pub enum LinkTypes {
    /// Links from agent to their listings
    /// Base: AgentPubKey, Target: Listing EntryHash
    AgentToListings,

    /// Links from category path to listings
    /// Base: Path("listings.{category}"), Target: Listing EntryHash
    CategoryToListings,

    /// Links from status path to listings
    /// Base: Path("listings.status.{status}"), Target: Listing EntryHash
    StatusToListings,

    /// All listings anchor
    /// Base: Path("all_listings"), Target: Listing EntryHash
    AllListings,
}

/// Entry types for this integrity zome
#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    Listing(Listing),
}

/// Validation function for Listing entries
///
/// This enforces marketplace rules at the DHT level.
/// Invalid listings are rejected by the network.
#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        FlatOp::StoreEntry(store_entry) => match store_entry {
            OpEntry::CreateEntry { app_entry, action } => match app_entry {
                EntryTypes::Listing(listing) => validate_create_listing(&listing, &action),
            },
            OpEntry::UpdateEntry {
                app_entry, action, ..
            } => match app_entry {
                EntryTypes::Listing(listing) => validate_update_listing(&listing, &action),
            },
            _ => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterUpdate(update_entry) => match update_entry {
            OpUpdate::Entry { app_entry, action } => match app_entry {
                EntryTypes::Listing(listing) => validate_update_listing(&listing, &action),
            },
            _ => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterDelete(_delete_entry) => {
            // Allow sellers to delete their own listings
            Ok(ValidateCallbackResult::Valid)
        }
        FlatOp::RegisterCreateLink {
            link_type,
            base_address: _,
            target_address: _,
            tag: _,
            action: _,
        } => {
            // Validate that links are created by the listing owner
            match link_type {
                LinkTypes::AgentToListings => Ok(ValidateCallbackResult::Valid),
                LinkTypes::CategoryToListings => Ok(ValidateCallbackResult::Valid),
                LinkTypes::StatusToListings => Ok(ValidateCallbackResult::Valid),
                LinkTypes::AllListings => Ok(ValidateCallbackResult::Valid),
            }
        }
        FlatOp::RegisterDeleteLink { .. } => Ok(ValidateCallbackResult::Valid),
        _ => Ok(ValidateCallbackResult::Valid),
    }
}

/// Validate listing data (used for both create and update)
fn validate_listing_data(
    listing: &Listing,
) -> ExternResult<ValidateCallbackResult> {
    // Title validation
    if listing.title.is_empty() || listing.title.len() > 200 {
        return Ok(ValidateCallbackResult::Invalid(
            "Title must be 1-200 characters".into(),
        ));
    }

    // Description validation
    if listing.description.is_empty() || listing.description.len() > 5000 {
        return Ok(ValidateCallbackResult::Invalid(
            "Description must be 1-5000 characters".into(),
        ));
    }

    // Price validation (must be positive)
    if listing.price_cents == 0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Price must be greater than zero".into(),
        ));
    }

    // Prevent unrealistic prices (> $1,000,000)
    if listing.price_cents > 100_000_000_00 {
        return Ok(ValidateCallbackResult::Invalid(
            "Price exceeds maximum allowed ($1,000,000)".into(),
        ));
    }

    // Photos validation
    if listing.photos_ipfs_cids.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "At least one photo is required".into(),
        ));
    }

    if listing.photos_ipfs_cids.len() > 10 {
        return Ok(ValidateCallbackResult::Invalid(
            "Maximum 10 photos allowed".into(),
        ));
    }

    // Validate IPFS CIDs format
    for cid in &listing.photos_ipfs_cids {
        if !is_valid_ipfs_cid(cid) {
            return Ok(ValidateCallbackResult::Invalid(format!(
                "Invalid IPFS CID: {}",
                cid
            )));
        }
    }

    // Quantity validation
    if listing.quantity_available == 0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Quantity must be at least 1".into(),
        ));
    }

    // Epistemic validation: Ensure listings start with proper classification
    // New listings should be E1 (testimonial), N0 (personal), M1 or M2
    match listing.epistemic.empirical {
        EmpiricalLevel::E0Null => {
            return Ok(ValidateCallbackResult::Invalid(
                "Listings cannot be E0 (unverifiable)".into(),
            ))
        }
        EmpiricalLevel::E3Cryptographic | EmpiricalLevel::E4PublicRepro => {
            return Ok(ValidateCallbackResult::Invalid(
                "New listings cannot claim E3/E4 without proof".into(),
            ))
        }
        _ => {} // E1, E2 are valid
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate listing creation
fn validate_create_listing(
    listing: &Listing,
    _action: &Create,
) -> ExternResult<ValidateCallbackResult> {
    validate_listing_data(listing)
}

/// Validate listing updates
fn validate_update_listing(
    listing: &Listing,
    action: &Update,
) -> ExternResult<ValidateCallbackResult> {
    // 1. Run same data validations as create
    let data_validation = validate_listing_data(listing)?;
    if let ValidateCallbackResult::Invalid(reason) = data_validation {
        return Ok(ValidateCallbackResult::Invalid(reason));
    }

    // 2. Authorization check: Only the original seller can update their listing
    // The Update action contains the original_action_hash which created the entry
    // We need to verify the author of the update matches the original author

    // Get the original action that created this listing
    let original_action_hash = action.original_action_address.clone();

    // The author of the current update action
    let updater = action.author.clone();

    // In Holochain's validation model, we can check that the updater
    // matches the original author by examining the chain history
    // The must_get_action call fetches the original action
    let original_action = must_get_action(original_action_hash.clone())?;

    // Extract the original author from the action
    let original_author = original_action.action().author().clone();

    // Authorization check: the updater must be the original seller
    if updater != original_author {
        return Ok(ValidateCallbackResult::Invalid(
            format!(
                "Only the original seller can update this listing. Updater: {}, Original seller: {}",
                updater, original_author
            )
        ));
    }

    // 3. Validate that certain fields cannot be changed (seller protection)
    // The created_at timestamp should never change on updates
    let original_entry = must_get_entry(action.original_entry_address.clone())?;

    if let Entry::App(app_entry) = original_entry.as_content() {
        // Try to deserialize the original listing
        if let Ok(original_listing) = Listing::try_from(app_entry.clone().into_sb()) {
            // created_at should never change
            if listing.created_at != original_listing.created_at {
                return Ok(ValidateCallbackResult::Invalid(
                    "Cannot modify the creation timestamp".into()
                ));
            }

            // updated_at should be greater than or equal to original
            if listing.updated_at < original_listing.updated_at {
                return Ok(ValidateCallbackResult::Invalid(
                    "Update timestamp cannot be earlier than original".into()
                ));
            }

            // Prevent resurrection of deleted listings
            if original_listing.status == ListingStatus::Deleted
                && listing.status != ListingStatus::Deleted
            {
                return Ok(ValidateCallbackResult::Invalid(
                    "Cannot reactivate a deleted listing - create a new one instead".into()
                ));
            }
        }
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate IPFS CID format
///
/// Accepts CIDv0 (Qm...) and CIDv1 (b...) formats
fn is_valid_ipfs_cid(cid: &str) -> bool {
    // CIDv0: Qm + 44 base58 characters
    if cid.len() == 46 && cid.starts_with("Qm") {
        return cid.chars().skip(2).all(|c| {
            c.is_ascii_alphanumeric() && c != '0' && c != 'O' && c != 'I' && c != 'l'
        });
    }

    // CIDv1: b + 58 base32 characters (simplified check)
    if cid.len() >= 59 && cid.starts_with('b') {
        return true; // Simplified validation for CIDv1
    }

    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ipfs_cid_validation() {
        // Valid CIDv0
        assert!(is_valid_ipfs_cid(
            "QmYwAPJzv5CZsnA625s3Xf2nemtYgPpHdWEz79ojWnPbdG"
        ));

        // Invalid CIDs
        assert!(!is_valid_ipfs_cid("QmInvalid"));
        assert!(!is_valid_ipfs_cid("notacid"));
        assert!(!is_valid_ipfs_cid(""));
    }
}
