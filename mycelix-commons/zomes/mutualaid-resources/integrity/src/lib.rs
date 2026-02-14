//! Resources Integrity Zome
//!
//! This zome defines entry types and validation rules for resource sharing
//! in the Mycelix Mutual Aid hApp. Supports tools, vehicles, spaces, and equipment.

use hdi::prelude::*;
use mutualaid_common::*;

/// Entry types for the resources zome
#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    /// A shared resource
    #[entry_type(visibility = "public")]
    SharedResource(SharedResource),
    /// A booking for a resource
    #[entry_type(visibility = "public")]
    Booking(Booking),
    /// Usage record
    #[entry_type(visibility = "public")]
    Usage(Usage),
    /// Maintenance record
    #[entry_type(visibility = "public")]
    Maintenance(Maintenance),
}

/// Link types for the resources zome
#[hdk_link_types]
pub enum LinkTypes {
    /// Link from owner to their resources
    OwnerToResources,
    /// Link from resource type anchor to resources
    TypeToResources,
    /// Link from resource to its bookings
    ResourceToBookings,
    /// Link from booker to their bookings
    BookerToBookings,
    /// Link from resource to usage records
    ResourceToUsage,
    /// Link from resource to maintenance records
    ResourceToMaintenance,
    /// Link for all resources discovery
    AllResources,
    /// Link for available resources
    AvailableResources,
}

/// Genesis self-check
#[hdk_extern]
pub fn genesis_self_check(_data: GenesisSelfCheckData) -> ExternResult<ValidateCallbackResult> {
    Ok(ValidateCallbackResult::Valid)
}

/// Main validation callback
#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        FlatOp::StoreEntry(store_entry) => match store_entry {
            OpEntry::CreateEntry { app_entry, .. } => validate_create_entry(app_entry),
            OpEntry::UpdateEntry { app_entry, .. } => validate_create_entry(app_entry),
            _ => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterCreateLink {
            link_type,
            base_address,
            target_address,
            tag,
            ..
        } => validate_create_link(link_type, base_address, target_address, tag),
        FlatOp::RegisterDeleteLink { link_type, .. } => {
            let _ = link_type;
            Ok(ValidateCallbackResult::Valid)
        }
        _ => Ok(ValidateCallbackResult::Valid),
    }
}

/// Validate entry creation
fn validate_create_entry(entry: EntryTypes) -> ExternResult<ValidateCallbackResult> {
    match entry {
        EntryTypes::SharedResource(resource) => validate_shared_resource(resource),
        EntryTypes::Booking(booking) => validate_booking(booking),
        EntryTypes::Usage(usage) => validate_usage(usage),
        EntryTypes::Maintenance(maintenance) => validate_maintenance(maintenance),
    }
}

/// Validate a shared resource
fn validate_shared_resource(resource: SharedResource) -> ExternResult<ValidateCallbackResult> {
    // ID must not be empty
    if resource.id.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Resource ID cannot be empty".to_string(),
        ));
    }

    // Name must not be empty
    if resource.name.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Resource name cannot be empty".to_string(),
        ));
    }

    // Name length limit
    if resource.name.len() > 150 {
        return Ok(ValidateCallbackResult::Invalid(
            "Resource name cannot exceed 150 characters".to_string(),
        ));
    }

    // Description length limit
    if resource.description.len() > 3000 {
        return Ok(ValidateCallbackResult::Invalid(
            "Resource description cannot exceed 3000 characters".to_string(),
        ));
    }

    // Photos limit
    if resource.photos.len() > 10 {
        return Ok(ValidateCallbackResult::Invalid(
            "Cannot have more than 10 photos".to_string(),
        ));
    }

    // Sharing model validation
    if resource.sharing_model.hourly_rate < 0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Hourly rate cannot be negative".to_string(),
        ));
    }

    if let Some(daily) = resource.sharing_model.daily_rate {
        if daily < 0 {
            return Ok(ValidateCallbackResult::Invalid(
                "Daily rate cannot be negative".to_string(),
            ));
        }
    }

    if let Some(deposit) = resource.sharing_model.deposit {
        if deposit < 0 {
            return Ok(ValidateCallbackResult::Invalid(
                "Deposit cannot be negative".to_string(),
            ));
        }
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate a booking
fn validate_booking(booking: Booking) -> ExternResult<ValidateCallbackResult> {
    // ID must not be empty
    if booking.id.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Booking ID cannot be empty".to_string(),
        ));
    }

    // End time must be after start time
    if booking.end_time <= booking.start_time {
        return Ok(ValidateCallbackResult::Invalid(
            "End time must be after start time".to_string(),
        ));
    }

    // Purpose length limit
    if booking.purpose.len() > 500 {
        return Ok(ValidateCallbackResult::Invalid(
            "Booking purpose cannot exceed 500 characters".to_string(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate a usage record
fn validate_usage(usage: Usage) -> ExternResult<ValidateCallbackResult> {
    // If actual_end is set, it must be after actual_start
    if let Some(end) = usage.actual_end {
        if end <= usage.actual_start {
            return Ok(ValidateCallbackResult::Invalid(
                "End time must be after start time".to_string(),
            ));
        }
    }

    // Notes length limit
    if usage.notes.len() > 1000 {
        return Ok(ValidateCallbackResult::Invalid(
            "Usage notes cannot exceed 1000 characters".to_string(),
        ));
    }

    // Issues limit
    if usage.issues.len() > 10 {
        return Ok(ValidateCallbackResult::Invalid(
            "Cannot report more than 10 issues".to_string(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate a maintenance record
fn validate_maintenance(maintenance: Maintenance) -> ExternResult<ValidateCallbackResult> {
    // Description must not be empty
    if maintenance.description.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Maintenance description cannot be empty".to_string(),
        ));
    }

    // Description length limit
    if maintenance.description.len() > 2000 {
        return Ok(ValidateCallbackResult::Invalid(
            "Maintenance description cannot exceed 2000 characters".to_string(),
        ));
    }

    // Hours spent must be non-negative
    if maintenance.hours_spent < 0.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Hours spent cannot be negative".to_string(),
        ));
    }

    // Hours spent should be reasonable
    if maintenance.hours_spent > 100.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Hours spent cannot exceed 100".to_string(),
        ));
    }

    // Cost must be non-negative if present
    if let Some(cost) = maintenance.cost {
        if cost < 0 {
            return Ok(ValidateCallbackResult::Invalid(
                "Maintenance cost cannot be negative".to_string(),
            ));
        }
    }

    // Parts limit
    if maintenance.parts_used.len() > 20 {
        return Ok(ValidateCallbackResult::Invalid(
            "Cannot list more than 20 parts".to_string(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate link creation
fn validate_create_link(
    link_type: LinkTypes,
    _base_address: AnyLinkableHash,
    _target_address: AnyLinkableHash,
    _tag: LinkTag,
) -> ExternResult<ValidateCallbackResult> {
    match link_type {
        LinkTypes::OwnerToResources => Ok(ValidateCallbackResult::Valid),
        LinkTypes::TypeToResources => Ok(ValidateCallbackResult::Valid),
        LinkTypes::ResourceToBookings => Ok(ValidateCallbackResult::Valid),
        LinkTypes::BookerToBookings => Ok(ValidateCallbackResult::Valid),
        LinkTypes::ResourceToUsage => Ok(ValidateCallbackResult::Valid),
        LinkTypes::ResourceToMaintenance => Ok(ValidateCallbackResult::Valid),
        LinkTypes::AllResources => Ok(ValidateCallbackResult::Valid),
        LinkTypes::AvailableResources => Ok(ValidateCallbackResult::Valid),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // =============================================================================
    // FACTORY FUNCTIONS
    // =============================================================================

    fn test_agent() -> AgentPubKey {
        AgentPubKey::from_raw_36(vec![0xdb; 36])
    }

    fn test_action_hash() -> ActionHash {
        ActionHash::from_raw_36(vec![0xab; 36])
    }

    fn test_timestamp_early() -> Timestamp {
        Timestamp::from_micros(1000000)
    }

    fn test_timestamp_late() -> Timestamp {
        Timestamp::from_micros(2000000)
    }

    fn valid_shared_resource() -> SharedResource {
        SharedResource {
            id: "resource-1".to_string(),
            owner: test_agent(),
            name: "Test Power Drill".to_string(),
            description: "A reliable power drill for community use".to_string(),
            resource_type: ResourceType::PowerTool,
            condition: ResourceCondition::Good,
            photos: vec![],
            location: LocationConstraint::FixedLocation("123 Main St".to_string()),
            availability: Availability::default(),
            sharing_model: SharingModel {
                free: false,
                deposit: Some(50),
                hourly_rate: 10,
                daily_rate: Some(50),
                accepts_time_credits: true,
                accepts_circle_credits: true,
                circle_hash: None,
            },
            usage_instructions: "Handle with care".to_string(),
            liability_notes: Some("User responsible for damages".to_string()),
            currently_available: true,
            created_at: test_timestamp_early(),
            updated_at: test_timestamp_early(),
        }
    }

    fn valid_booking() -> Booking {
        Booking {
            id: "booking-1".to_string(),
            resource_hash: test_action_hash(),
            booker: test_agent(),
            start_time: test_timestamp_early(),
            end_time: test_timestamp_late(),
            purpose: "Weekend project".to_string(),
            status: BookingStatus::Pending,
            payment_method: Some(PaymentMethod::TimeCredits(2.0)),
            payment_hash: None,
            created_at: test_timestamp_early(),
        }
    }

    fn valid_usage() -> Usage {
        Usage {
            booking_hash: test_action_hash(),
            actual_start: test_timestamp_early(),
            actual_end: Some(test_timestamp_late()),
            condition_before: ResourceCondition::Good,
            condition_after: Some(ResourceCondition::Good),
            notes: "Used as expected".to_string(),
            issues: vec![],
        }
    }

    fn valid_maintenance() -> Maintenance {
        Maintenance {
            resource_hash: test_action_hash(),
            maintainer: test_agent(),
            maintenance_type: MaintenanceType::Routine,
            description: "Oil change and inspection".to_string(),
            cost: Some(25),
            hours_spent: 2.0,
            parts_used: vec!["Oil filter".to_string()],
            performed_at: test_timestamp_early(),
            next_due: Some(test_timestamp_late()),
        }
    }

    // =============================================================================
    // SHARED RESOURCE VALIDATION TESTS
    // =============================================================================

    #[test]
    fn test_valid_shared_resource() {
        let resource = valid_shared_resource();
        let result = validate_shared_resource(resource);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_shared_resource_empty_id() {
        let mut resource = valid_shared_resource();
        resource.id = "".to_string();
        let result = validate_shared_resource(resource);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn test_shared_resource_empty_name() {
        let mut resource = valid_shared_resource();
        resource.name = "".to_string();
        let result = validate_shared_resource(resource);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn test_shared_resource_name_exact_limit() {
        let mut resource = valid_shared_resource();
        resource.name = "a".repeat(150);
        let result = validate_shared_resource(resource);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_shared_resource_name_exceeds_limit() {
        let mut resource = valid_shared_resource();
        resource.name = "a".repeat(151);
        let result = validate_shared_resource(resource);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn test_shared_resource_description_exact_limit() {
        let mut resource = valid_shared_resource();
        resource.description = "a".repeat(3000);
        let result = validate_shared_resource(resource);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_shared_resource_description_exceeds_limit() {
        let mut resource = valid_shared_resource();
        resource.description = "a".repeat(3001);
        let result = validate_shared_resource(resource);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn test_shared_resource_photos_exact_limit() {
        let mut resource = valid_shared_resource();
        resource.photos = (0..10).map(|i| format!("photo{}", i)).collect();
        let result = validate_shared_resource(resource);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_shared_resource_photos_exceeds_limit() {
        let mut resource = valid_shared_resource();
        resource.photos = (0..11).map(|i| format!("photo{}", i)).collect();
        let result = validate_shared_resource(resource);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn test_shared_resource_hourly_rate_zero() {
        let mut resource = valid_shared_resource();
        resource.sharing_model.hourly_rate = 0;
        let result = validate_shared_resource(resource);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_shared_resource_hourly_rate_negative() {
        let mut resource = valid_shared_resource();
        resource.sharing_model.hourly_rate = -1;
        let result = validate_shared_resource(resource);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn test_shared_resource_daily_rate_zero() {
        let mut resource = valid_shared_resource();
        resource.sharing_model.daily_rate = Some(0);
        let result = validate_shared_resource(resource);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_shared_resource_daily_rate_negative() {
        let mut resource = valid_shared_resource();
        resource.sharing_model.daily_rate = Some(-1);
        let result = validate_shared_resource(resource);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn test_shared_resource_daily_rate_none() {
        let mut resource = valid_shared_resource();
        resource.sharing_model.daily_rate = None;
        let result = validate_shared_resource(resource);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_shared_resource_deposit_zero() {
        let mut resource = valid_shared_resource();
        resource.sharing_model.deposit = Some(0);
        let result = validate_shared_resource(resource);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_shared_resource_deposit_negative() {
        let mut resource = valid_shared_resource();
        resource.sharing_model.deposit = Some(-1);
        let result = validate_shared_resource(resource);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn test_shared_resource_deposit_none() {
        let mut resource = valid_shared_resource();
        resource.sharing_model.deposit = None;
        let result = validate_shared_resource(resource);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    // =============================================================================
    // BOOKING VALIDATION TESTS
    // =============================================================================

    #[test]
    fn test_valid_booking() {
        let booking = valid_booking();
        let result = validate_booking(booking);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_booking_empty_id() {
        let mut booking = valid_booking();
        booking.id = "".to_string();
        let result = validate_booking(booking);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn test_booking_end_before_start() {
        let mut booking = valid_booking();
        booking.start_time = test_timestamp_late();
        booking.end_time = test_timestamp_early();
        let result = validate_booking(booking);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn test_booking_end_equals_start() {
        let mut booking = valid_booking();
        booking.start_time = test_timestamp_early();
        booking.end_time = test_timestamp_early();
        let result = validate_booking(booking);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn test_booking_purpose_exact_limit() {
        let mut booking = valid_booking();
        booking.purpose = "a".repeat(500);
        let result = validate_booking(booking);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_booking_purpose_exceeds_limit() {
        let mut booking = valid_booking();
        booking.purpose = "a".repeat(501);
        let result = validate_booking(booking);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn test_booking_purpose_empty() {
        let mut booking = valid_booking();
        booking.purpose = "".to_string();
        let result = validate_booking(booking);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    // =============================================================================
    // USAGE VALIDATION TESTS
    // =============================================================================

    #[test]
    fn test_valid_usage() {
        let usage = valid_usage();
        let result = validate_usage(usage);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_usage_actual_end_none() {
        let mut usage = valid_usage();
        usage.actual_end = None;
        let result = validate_usage(usage);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_usage_actual_end_before_start() {
        let mut usage = valid_usage();
        usage.actual_start = test_timestamp_late();
        usage.actual_end = Some(test_timestamp_early());
        let result = validate_usage(usage);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn test_usage_actual_end_equals_start() {
        let mut usage = valid_usage();
        usage.actual_start = test_timestamp_early();
        usage.actual_end = Some(test_timestamp_early());
        let result = validate_usage(usage);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn test_usage_notes_exact_limit() {
        let mut usage = valid_usage();
        usage.notes = "a".repeat(1000);
        let result = validate_usage(usage);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_usage_notes_exceeds_limit() {
        let mut usage = valid_usage();
        usage.notes = "a".repeat(1001);
        let result = validate_usage(usage);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn test_usage_notes_empty() {
        let mut usage = valid_usage();
        usage.notes = "".to_string();
        let result = validate_usage(usage);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_usage_issues_exact_limit() {
        let mut usage = valid_usage();
        usage.issues = (0..10).map(|i| format!("issue{}", i)).collect();
        let result = validate_usage(usage);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_usage_issues_exceeds_limit() {
        let mut usage = valid_usage();
        usage.issues = (0..11).map(|i| format!("issue{}", i)).collect();
        let result = validate_usage(usage);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn test_usage_no_issues() {
        let mut usage = valid_usage();
        usage.issues = vec![];
        let result = validate_usage(usage);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    // =============================================================================
    // MAINTENANCE VALIDATION TESTS
    // =============================================================================

    #[test]
    fn test_valid_maintenance() {
        let maintenance = valid_maintenance();
        let result = validate_maintenance(maintenance);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_maintenance_empty_description() {
        let mut maintenance = valid_maintenance();
        maintenance.description = "".to_string();
        let result = validate_maintenance(maintenance);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn test_maintenance_description_exact_limit() {
        let mut maintenance = valid_maintenance();
        maintenance.description = "a".repeat(2000);
        let result = validate_maintenance(maintenance);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_maintenance_description_exceeds_limit() {
        let mut maintenance = valid_maintenance();
        maintenance.description = "a".repeat(2001);
        let result = validate_maintenance(maintenance);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn test_maintenance_hours_zero() {
        let mut maintenance = valid_maintenance();
        maintenance.hours_spent = 0.0;
        let result = validate_maintenance(maintenance);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_maintenance_hours_negative() {
        let mut maintenance = valid_maintenance();
        maintenance.hours_spent = -0.1;
        let result = validate_maintenance(maintenance);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn test_maintenance_hours_exact_limit() {
        let mut maintenance = valid_maintenance();
        maintenance.hours_spent = 100.0;
        let result = validate_maintenance(maintenance);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_maintenance_hours_exceeds_limit() {
        let mut maintenance = valid_maintenance();
        maintenance.hours_spent = 100.1;
        let result = validate_maintenance(maintenance);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn test_maintenance_cost_zero() {
        let mut maintenance = valid_maintenance();
        maintenance.cost = Some(0);
        let result = validate_maintenance(maintenance);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_maintenance_cost_negative() {
        let mut maintenance = valid_maintenance();
        maintenance.cost = Some(-1);
        let result = validate_maintenance(maintenance);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn test_maintenance_cost_none() {
        let mut maintenance = valid_maintenance();
        maintenance.cost = None;
        let result = validate_maintenance(maintenance);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_maintenance_parts_exact_limit() {
        let mut maintenance = valid_maintenance();
        maintenance.parts_used = (0..20).map(|i| format!("part{}", i)).collect();
        let result = validate_maintenance(maintenance);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_maintenance_parts_exceeds_limit() {
        let mut maintenance = valid_maintenance();
        maintenance.parts_used = (0..21).map(|i| format!("part{}", i)).collect();
        let result = validate_maintenance(maintenance);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn test_maintenance_no_parts() {
        let mut maintenance = valid_maintenance();
        maintenance.parts_used = vec![];
        let result = validate_maintenance(maintenance);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    // =============================================================================
    // ENUM SERIALIZATION TESTS
    // =============================================================================

    #[test]
    fn test_resource_type_serde_power_tool() {
        let rt = ResourceType::PowerTool;
        let json = serde_json::to_string(&rt).unwrap();
        let parsed: ResourceType = serde_json::from_str(&json).unwrap();
        assert_eq!(rt, parsed);
    }

    #[test]
    fn test_resource_type_serde_car() {
        let rt = ResourceType::Car;
        let json = serde_json::to_string(&rt).unwrap();
        let parsed: ResourceType = serde_json::from_str(&json).unwrap();
        assert_eq!(rt, parsed);
    }

    #[test]
    fn test_resource_type_serde_custom() {
        let rt = ResourceType::Custom("Specialized Tool".to_string());
        let json = serde_json::to_string(&rt).unwrap();
        let parsed: ResourceType = serde_json::from_str(&json).unwrap();
        assert_eq!(rt, parsed);
    }

    #[test]
    fn test_resource_condition_serde_excellent() {
        let rc = ResourceCondition::Excellent;
        let json = serde_json::to_string(&rc).unwrap();
        let parsed: ResourceCondition = serde_json::from_str(&json).unwrap();
        assert_eq!(rc, parsed);
    }

    #[test]
    fn test_resource_condition_serde_needs_repair() {
        let rc = ResourceCondition::NeedsRepair;
        let json = serde_json::to_string(&rc).unwrap();
        let parsed: ResourceCondition = serde_json::from_str(&json).unwrap();
        assert_eq!(rc, parsed);
    }

    #[test]
    fn test_booking_status_serde_pending() {
        let bs = BookingStatus::Pending;
        let json = serde_json::to_string(&bs).unwrap();
        let parsed: BookingStatus = serde_json::from_str(&json).unwrap();
        assert_eq!(bs, parsed);
    }

    #[test]
    fn test_booking_status_serde_confirmed() {
        let bs = BookingStatus::Confirmed;
        let json = serde_json::to_string(&bs).unwrap();
        let parsed: BookingStatus = serde_json::from_str(&json).unwrap();
        assert_eq!(bs, parsed);
    }

    #[test]
    fn test_booking_status_serde_no_show() {
        let bs = BookingStatus::NoShow;
        let json = serde_json::to_string(&bs).unwrap();
        let parsed: BookingStatus = serde_json::from_str(&json).unwrap();
        assert_eq!(bs, parsed);
    }

    #[test]
    fn test_maintenance_type_serde_routine() {
        let mt = MaintenanceType::Routine;
        let json = serde_json::to_string(&mt).unwrap();
        let parsed: MaintenanceType = serde_json::from_str(&json).unwrap();
        assert_eq!(mt, parsed);
    }

    #[test]
    fn test_maintenance_type_serde_repair() {
        let mt = MaintenanceType::Repair;
        let json = serde_json::to_string(&mt).unwrap();
        let parsed: MaintenanceType = serde_json::from_str(&json).unwrap();
        assert_eq!(mt, parsed);
    }

    #[test]
    fn test_maintenance_type_serde_custom() {
        let mt = MaintenanceType::Other("Deep Clean".to_string());
        let json = serde_json::to_string(&mt).unwrap();
        let parsed: MaintenanceType = serde_json::from_str(&json).unwrap();
        assert_eq!(mt, parsed);
    }

    // =============================================================================
    // ADDITIONAL EDGE CASES
    // =============================================================================

    #[test]
    fn test_shared_resource_all_resource_types() {
        let types = vec![
            ResourceType::PowerTool,
            ResourceType::HandTool,
            ResourceType::GardenTool,
            ResourceType::Car,
            ResourceType::Bicycle,
            ResourceType::MeetingRoom,
            ResourceType::Workshop,
            ResourceType::CampingGear,
            ResourceType::Custom("Test".to_string()),
        ];

        for rt in types {
            let mut resource = valid_shared_resource();
            resource.resource_type = rt;
            let result = validate_shared_resource(resource);
            assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
        }
    }

    #[test]
    fn test_shared_resource_all_conditions() {
        let conditions = vec![
            ResourceCondition::Excellent,
            ResourceCondition::Good,
            ResourceCondition::Fair,
            ResourceCondition::NeedsRepair,
            ResourceCondition::BeingRepaired,
        ];

        for cond in conditions {
            let mut resource = valid_shared_resource();
            resource.condition = cond;
            let result = validate_shared_resource(resource);
            assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
        }
    }

    #[test]
    fn test_booking_all_statuses() {
        let statuses = vec![
            BookingStatus::Pending,
            BookingStatus::Confirmed,
            BookingStatus::Active,
            BookingStatus::Completed,
            BookingStatus::Cancelled,
            BookingStatus::NoShow,
        ];

        for status in statuses {
            let mut booking = valid_booking();
            booking.status = status;
            let result = validate_booking(booking);
            assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
        }
    }

    #[test]
    fn test_maintenance_all_types() {
        let types = vec![
            MaintenanceType::Routine,
            MaintenanceType::Repair,
            MaintenanceType::Cleaning,
            MaintenanceType::Upgrade,
            MaintenanceType::SafetyCheck,
            MaintenanceType::Other("Custom".to_string()),
        ];

        for mt in types {
            let mut maintenance = valid_maintenance();
            maintenance.maintenance_type = mt;
            let result = validate_maintenance(maintenance);
            assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
        }
    }

    #[test]
    fn test_shared_resource_free_model() {
        let mut resource = valid_shared_resource();
        resource.sharing_model = SharingModel {
            free: true,
            deposit: None,
            hourly_rate: 0,
            daily_rate: None,
            accepts_time_credits: false,
            accepts_circle_credits: false,
            circle_hash: None,
        };
        let result = validate_shared_resource(resource);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_booking_with_payment_methods() {
        let payment_methods = vec![
            Some(PaymentMethod::Free),
            Some(PaymentMethod::TimeCredits(5.0)),
            Some(PaymentMethod::CircleCredits {
                circle_hash: test_action_hash(),
                amount: 100,
            }),
            Some(PaymentMethod::External("Cash".to_string())),
            None,
        ];

        for pm in payment_methods {
            let mut booking = valid_booking();
            booking.payment_method = pm;
            let result = validate_booking(booking);
            assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
        }
    }

    #[test]
    fn test_usage_condition_changes() {
        let conditions = vec![
            (ResourceCondition::Excellent, Some(ResourceCondition::Good)),
            (ResourceCondition::Good, Some(ResourceCondition::Fair)),
            (ResourceCondition::Fair, Some(ResourceCondition::NeedsRepair)),
            (ResourceCondition::Good, None),
        ];

        for (before, after) in conditions {
            let mut usage = valid_usage();
            usage.condition_before = before;
            usage.condition_after = after;
            let result = validate_usage(usage);
            assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
        }
    }

    #[test]
    fn test_maintenance_fractional_hours() {
        let hours = vec![0.0, 0.5, 1.5, 10.5, 50.25, 99.9, 100.0];

        for h in hours {
            let mut maintenance = valid_maintenance();
            maintenance.hours_spent = h;
            let result = validate_maintenance(maintenance);
            assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
        }
    }
}
