// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Shared validation helpers for integrity zomes.
//!
//! The 68+ integrity zomes across mycelix-commons, mycelix-civic, and
//! mycelix-hearth all perform nearly identical author-match checks in their
//! `validate()` callbacks.  These helpers factor out the common pattern so
//! each zome can call a single function instead of duplicating the logic.
//!
//! # Usage
//!
//! In an integrity zome's `validate()` function:
//!
//! ```ignore
//! use mycelix_bridge_common::validation::check_author_match;
//!
//! // Inside RegisterDelete arm:
//! let original_action = must_get_action(action.deletes_address.clone())?;
//! let result = check_author_match(
//!     original_action.action().author(),
//!     &action.author,
//!     "delete",
//! );
//! if let ValidateCallbackResult::Invalid(_) = &result {
//!     return Ok(result);
//! }
//! ```

use hdk::prelude::*;

/// Check that two agent public keys match for an authorship validation.
///
/// Returns `ValidateCallbackResult::Valid` if the authors match, or
/// `ValidateCallbackResult::Invalid` with a descriptive message if they
/// do not.
///
/// The `operation` parameter is interpolated into the error message
/// (e.g., `"update"`, `"delete"`, `"delete this link on"`).
///
/// # Examples
///
/// ```ignore
/// let result = check_author_match(
///     original_action.action().author(),
///     &update_action.author,
///     "update",
/// );
/// // → Valid, or Invalid("Only the original entry author can update their entries")
/// ```
pub fn check_author_match(
    original_author: &AgentPubKey,
    action_author: &AgentPubKey,
    operation: &str,
) -> ValidateCallbackResult {
    if original_author != action_author {
        ValidateCallbackResult::Invalid(format!(
            "Only the original entry author can {} their entries",
            operation
        ))
    } else {
        ValidateCallbackResult::Valid
    }
}

/// Check that two agent public keys match for a link deletion validation.
///
/// This is a specialization of [`check_author_match`] with a link-specific
/// error message ("… delete this link") instead of the generic entry message.
///
/// # Examples
///
/// ```ignore
/// let original_action = must_get_action(action.link_add_address.clone())?;
/// let result = check_link_author_match(
///     original_action.action().author(),
///     &action.author,
/// );
/// if let ValidateCallbackResult::Invalid(_) = &result {
///     return Ok(result);
/// }
/// ```
pub fn check_link_author_match(
    original_author: &AgentPubKey,
    action_author: &AgentPubKey,
) -> ValidateCallbackResult {
    if original_author != action_author {
        ValidateCallbackResult::Invalid("Only the original author can delete this link".into())
    } else {
        ValidateCallbackResult::Valid
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_agent(byte: u8) -> AgentPubKey {
        AgentPubKey::from_raw_36(vec![byte; 36])
    }

    // ---- check_author_match ----

    #[test]
    fn author_match_same_agent_is_valid() {
        let agent = make_agent(1);
        let result = check_author_match(&agent, &agent, "update");
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn author_match_different_agent_is_invalid() {
        let original = make_agent(1);
        let impersonator = make_agent(2);
        let result = check_author_match(&original, &impersonator, "update");
        match &result {
            ValidateCallbackResult::Invalid(msg) => {
                assert!(msg.contains("update"));
                assert!(msg.contains("Only the original entry author"));
            }
            _ => panic!("Expected Invalid, got {:?}", result),
        }
    }

    #[test]
    fn author_match_delete_operation_message() {
        let original = make_agent(10);
        let other = make_agent(20);
        let result = check_author_match(&original, &other, "delete");
        match &result {
            ValidateCallbackResult::Invalid(msg) => {
                assert!(msg.contains("delete"));
            }
            _ => panic!("Expected Invalid"),
        }
    }

    #[test]
    fn author_match_cloned_keys_are_equal() {
        let agent = make_agent(42);
        let cloned = agent.clone();
        let result = check_author_match(&agent, &cloned, "update");
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    // ---- check_link_author_match ----

    #[test]
    fn link_author_match_same_agent_is_valid() {
        let agent = make_agent(5);
        let result = check_link_author_match(&agent, &agent);
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn link_author_match_different_agent_is_invalid() {
        let original = make_agent(5);
        let other = make_agent(6);
        let result = check_link_author_match(&original, &other);
        match &result {
            ValidateCallbackResult::Invalid(msg) => {
                assert!(msg.contains("delete this link"));
                assert!(msg.contains("Only the original author"));
            }
            _ => panic!("Expected Invalid, got {:?}", result),
        }
    }

    #[test]
    fn link_author_match_message_differs_from_entry() {
        let original = make_agent(1);
        let other = make_agent(2);
        let entry_result = check_author_match(&original, &other, "delete");
        let link_result = check_link_author_match(&original, &other);
        // Both are Invalid but with different messages
        match (&entry_result, &link_result) {
            (
                ValidateCallbackResult::Invalid(entry_msg),
                ValidateCallbackResult::Invalid(link_msg),
            ) => {
                assert_ne!(entry_msg, link_msg);
                assert!(link_msg.contains("link"));
            }
            _ => panic!("Expected both Invalid"),
        }
    }
}
