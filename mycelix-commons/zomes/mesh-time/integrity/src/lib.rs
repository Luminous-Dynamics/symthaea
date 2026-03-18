use hdi::prelude::*;

/// A time anchor recorded on the DHT for mesh-time consensus.
#[hdk_entry_helper]
#[derive(Clone)]
pub struct TimeAnchor {
    /// Mesh time in microseconds since Unix epoch.
    pub timestamp_us: u64,
    /// Reporter's stratum level.
    pub stratum: u8,
    /// Reporter's Phi at anchor time.
    pub phi: f32,
    /// Reporter's estimated drift in ppm.
    pub drift_ppm: f32,
    /// Monotonic counter (prevents replay).
    pub counter: u64,
}

/// Challenge to a skewed time anchor.
#[hdk_entry_helper]
#[derive(Clone)]
pub struct TimeDispute {
    /// Action hash of the disputed TimeAnchor.
    pub anchor_hash: ActionHash,
    /// Challenger's own timestamp at dispute time.
    pub challenger_timestamp_us: u64,
    /// Claimed offset discrepancy (µs).
    pub claimed_offset_us: i64,
    /// Reason for dispute.
    pub reason: String,
}

/// Anchor entry for deterministic link bases.
#[hdk_entry_helper]
#[derive(Clone)]
pub struct Anchor(pub String);

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    #[entry_type(name = "TimeAnchor", visibility = "public")]
    TimeAnchor(TimeAnchor),
    #[entry_type(name = "TimeDispute", visibility = "public")]
    TimeDispute(TimeDispute),
    #[entry_type(name = "Anchor", visibility = "public")]
    Anchor(Anchor),
}

#[hdk_link_types]
pub enum LinkTypes {
    AllTimeAnchors,
    AnchorToDisputes,
    AgentToAnchors,
}

#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        FlatOp::StoreEntry(store_entry) => match store_entry {
            OpEntry::CreateEntry { app_entry, .. } | OpEntry::UpdateEntry { app_entry, .. } => {
                match app_entry {
                    EntryTypes::TimeAnchor(anchor) => {
                        if anchor.stratum > 15 {
                            return Ok(ValidateCallbackResult::Invalid(
                                "Stratum must be <= 15".to_string(),
                            ));
                        }
                        if !anchor.phi.is_finite() || anchor.phi < 0.0 || anchor.phi > 2.0 {
                            return Ok(ValidateCallbackResult::Invalid(
                                "Phi must be finite and in [0.0, 2.0]".to_string(),
                            ));
                        }
                        if !anchor.drift_ppm.is_finite() {
                            return Ok(ValidateCallbackResult::Invalid(
                                "Drift must be finite".to_string(),
                            ));
                        }
                        Ok(ValidateCallbackResult::Valid)
                    }
                    EntryTypes::TimeDispute(dispute) => {
                        if dispute.reason.len() > 1024 {
                            return Ok(ValidateCallbackResult::Invalid(
                                "Dispute reason too long (max 1024)".to_string(),
                            ));
                        }
                        Ok(ValidateCallbackResult::Valid)
                    }
                    EntryTypes::Anchor(_) => Ok(ValidateCallbackResult::Valid),
                }
            }
            _ => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterCreateLink { .. }
        | FlatOp::RegisterDeleteLink { .. }
        | FlatOp::StoreRecord(_)
        | FlatOp::RegisterUpdate(_)
        | FlatOp::RegisterDelete(_)
        | FlatOp::RegisterAgentActivity(_) => Ok(ValidateCallbackResult::Valid),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_time_anchor_serde() {
        let anchor = TimeAnchor {
            timestamp_us: 1_710_000_000_000_000,
            stratum: 2,
            phi: 0.75,
            drift_ppm: -0.5,
            counter: 42,
        };
        let bytes = serde_json::to_vec(&anchor).unwrap();
        let decoded: TimeAnchor = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(decoded.timestamp_us, anchor.timestamp_us);
        assert_eq!(decoded.stratum, anchor.stratum);
    }
}
