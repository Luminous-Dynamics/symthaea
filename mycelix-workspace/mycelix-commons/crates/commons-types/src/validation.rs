// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Shared validation macros for Holochain integrity zomes.
//!
//! These macros reduce boilerplate across the ~50 integrity zomes in
//! mycelix-commons, mycelix-civic, and mycelix-governance by extracting
//! common validation patterns into reusable, composable macros.
//!
//! All macros are designed to be used inside functions that return
//! `ExternResult<ValidateCallbackResult>`. They perform an early return
//! with `ValidateCallbackResult::Invalid(...)` when the check fails.
//!
//! # Usage
//!
//! ```ignore
//! use commons_types::*; // brings macros into scope via #[macro_export]
//!
//! fn validate_my_entry(entry: MyEntry) -> ExternResult<ValidateCallbackResult> {
//!     require_non_empty!(entry.name, "name");
//!     require_max_str_len!(entry.name, 256, "name");
//!     require_finite!(entry.value, "value");
//!     require_in_range!(entry.latitude, -90.0, 90.0, "latitude");
//!     require_max_len!(entry.items, 100, "items");
//!     require_basis_points!(entry.share_bps, "share");
//!     Ok(ValidateCallbackResult::Valid)
//! }
//! ```

/// Validates that an f64 value is finite (not NaN, not Infinity, not -Infinity).
///
/// Must be used inside a function returning `ExternResult<ValidateCallbackResult>`.
/// Performs an early return with `Invalid` if the value is not finite.
///
/// # Example
///
/// ```ignore
/// require_finite!(geo.latitude, "Latitude");
/// ```
#[macro_export]
macro_rules! require_finite {
    ($val:expr, $field:expr) => {
        if !($val).is_finite() {
            return Ok(ValidateCallbackResult::Invalid(format!(
                "{} must be a finite number, got {}",
                $field, $val
            )));
        }
    };
}

/// Validates that a Vec does not exceed a maximum length.
///
/// Must be used inside a function returning `ExternResult<ValidateCallbackResult>`.
/// Performs an early return with `Invalid` if the Vec length exceeds `max`.
///
/// # Example
///
/// ```ignore
/// require_max_len!(property.co_owners, 100, "co-owners");
/// ```
#[macro_export]
macro_rules! require_max_len {
    ($vec:expr, $max:expr, $field:expr) => {
        if ($vec).len() > $max {
            return Ok(ValidateCallbackResult::Invalid(format!(
                "{} exceeds maximum length of {}",
                $field, $max
            )));
        }
    };
}

/// Validates that a string is not empty after trimming whitespace.
///
/// Must be used inside a function returning `ExternResult<ValidateCallbackResult>`.
/// Performs an early return with `Invalid` if the trimmed string is empty.
///
/// # Example
///
/// ```ignore
/// require_non_empty!(entry.name, "Entry name");
/// ```
#[macro_export]
macro_rules! require_non_empty {
    ($s:expr, $field:expr) => {
        if ($s).trim().is_empty() {
            return Ok(ValidateCallbackResult::Invalid(format!(
                "{} cannot be empty",
                $field
            )));
        }
    };
}

/// Validates that a string does not exceed a maximum byte length.
///
/// Uses `.len()` which counts bytes (consistent with existing zome validation).
/// Must be used inside a function returning `ExternResult<ValidateCallbackResult>`.
/// Performs an early return with `Invalid` if the string length exceeds `max_bytes`.
///
/// # Example
///
/// ```ignore
/// require_max_str_len!(entry.title, 256, "Title");
/// ```
#[macro_export]
macro_rules! require_max_str_len {
    ($s:expr, $max_bytes:expr, $field:expr) => {
        if ($s).len() > $max_bytes {
            return Ok(ValidateCallbackResult::Invalid(format!(
                "{} must be {} characters or fewer",
                $field, $max_bytes
            )));
        }
    };
}

/// Validates that a numeric value is within an inclusive range `[min, max]`.
///
/// Works with any type that implements `PartialOrd` and `Display` (f64, f32, i32, u32, etc.).
/// Must be used inside a function returning `ExternResult<ValidateCallbackResult>`.
/// Performs an early return with `Invalid` if the value is outside the range.
///
/// # Example
///
/// ```ignore
/// require_in_range!(geo.latitude, -90.0, 90.0, "Latitude");
/// require_in_range!(reading.ph, 0.0, 14.0, "pH");
/// require_in_range!(charge.period_month, 1, 12, "Month");
/// ```
#[macro_export]
macro_rules! require_in_range {
    ($val:expr, $min:expr, $max:expr, $field:expr) => {
        if ($val) < $min || ($val) > $max {
            return Ok(ValidateCallbackResult::Invalid(format!(
                "{} must be between {} and {}",
                $field, $min, $max
            )));
        }
    };
}

/// Validates that a u32 value is valid basis points (1..=10000).
///
/// Basis points represent fractional percentages where 10000 = 100.00%.
/// A value of 0 or above 10000 is rejected.
/// Must be used inside a function returning `ExternResult<ValidateCallbackResult>`.
///
/// # Example
///
/// ```ignore
/// require_basis_points!(co_owner.share_basis_points, "Share");
/// ```
#[macro_export]
macro_rules! require_basis_points {
    ($val:expr, $field:expr) => {
        if ($val) == 0 || ($val) > 10000 {
            return Ok(ValidateCallbackResult::Invalid(format!(
                "{} must be between 1 and 10000 basis points",
                $field
            )));
        }
    };
}

/// Validates that a link tag does not exceed a maximum byte length.
///
/// Must be used inside a function returning `ExternResult<ValidateCallbackResult>`.
/// Performs an early return with `Invalid` if the tag exceeds `max_bytes`.
///
/// # Example
///
/// ```ignore
/// require_max_tag_len!(tag, 256, "OwnerToProperties");
/// ```
#[macro_export]
macro_rules! require_max_tag_len {
    ($tag:expr, $max_bytes:expr, $link_name:expr) => {
        if ($tag).0.len() > $max_bytes {
            return Ok(ValidateCallbackResult::Invalid(format!(
                "{} link tag too long (max {} bytes)",
                $link_name, $max_bytes
            )));
        }
    };
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    // We need ValidateCallbackResult and ExternResult for the macros.
    // In real zomes these come from `hdi::prelude::*`, but for unit testing
    // the commons-types crate we use them directly from hdi.
    use hdi::prelude::*;

    // Helper to check if a result is Valid
    fn is_valid(result: &ExternResult<ValidateCallbackResult>) -> bool {
        matches!(result, Ok(ValidateCallbackResult::Valid))
    }

    // Helper to check if a result is Invalid
    fn is_invalid(result: &ExternResult<ValidateCallbackResult>) -> bool {
        matches!(result, Ok(ValidateCallbackResult::Invalid(_)))
    }

    // Helper to extract the Invalid message
    fn invalid_msg(result: &ExternResult<ValidateCallbackResult>) -> String {
        match result {
            Ok(ValidateCallbackResult::Invalid(msg)) => msg.clone(),
            _ => panic!("Expected Invalid, got {:?}", result),
        }
    }

    // ========================================================================
    // require_finite! tests
    // ========================================================================

    fn check_finite(val: f64) -> ExternResult<ValidateCallbackResult> {
        require_finite!(val, "test_field");
        Ok(ValidateCallbackResult::Valid)
    }

    #[test]
    fn finite_normal_value_passes() {
        assert!(is_valid(&check_finite(42.0)));
    }

    #[test]
    fn finite_zero_passes() {
        assert!(is_valid(&check_finite(0.0)));
    }

    #[test]
    fn finite_negative_zero_passes() {
        assert!(is_valid(&check_finite(-0.0)));
    }

    #[test]
    fn finite_large_value_passes() {
        assert!(is_valid(&check_finite(1e300)));
    }

    #[test]
    fn finite_small_value_passes() {
        assert!(is_valid(&check_finite(1e-300)));
    }

    #[test]
    fn finite_nan_rejected() {
        let result = check_finite(f64::NAN);
        assert!(is_invalid(&result));
        assert!(invalid_msg(&result).contains("test_field"));
        assert!(invalid_msg(&result).contains("finite"));
    }

    #[test]
    fn finite_infinity_rejected() {
        let result = check_finite(f64::INFINITY);
        assert!(is_invalid(&result));
        assert!(invalid_msg(&result).contains("test_field"));
    }

    #[test]
    fn finite_neg_infinity_rejected() {
        let result = check_finite(f64::NEG_INFINITY);
        assert!(is_invalid(&result));
        assert!(invalid_msg(&result).contains("test_field"));
    }

    // ========================================================================
    // require_max_len! tests
    // ========================================================================

    fn check_max_len(items: &[u8], max: usize) -> ExternResult<ValidateCallbackResult> {
        require_max_len!(items, max, "items");
        Ok(ValidateCallbackResult::Valid)
    }

    #[test]
    fn max_len_empty_vec_passes() {
        assert!(is_valid(&check_max_len(&[], 10)));
    }

    #[test]
    fn max_len_under_limit_passes() {
        assert!(is_valid(&check_max_len(&[1, 2, 3], 10)));
    }

    #[test]
    fn max_len_at_limit_passes() {
        assert!(is_valid(&check_max_len(&[1, 2, 3], 3)));
    }

    #[test]
    fn max_len_over_limit_rejected() {
        let result = check_max_len(&[1, 2, 3, 4], 3);
        assert!(is_invalid(&result));
        assert!(invalid_msg(&result).contains("items"));
        assert!(invalid_msg(&result).contains("3"));
    }

    #[test]
    fn max_len_zero_limit_empty_vec_passes() {
        assert!(is_valid(&check_max_len(&[], 0)));
    }

    #[test]
    fn max_len_zero_limit_nonempty_vec_rejected() {
        assert!(is_invalid(&check_max_len(&[1], 0)));
    }

    // ========================================================================
    // require_non_empty! tests
    // ========================================================================

    fn check_non_empty(s: &str) -> ExternResult<ValidateCallbackResult> {
        require_non_empty!(s, "field_name");
        Ok(ValidateCallbackResult::Valid)
    }

    #[test]
    fn non_empty_normal_string_passes() {
        assert!(is_valid(&check_non_empty("hello")));
    }

    #[test]
    fn non_empty_single_char_passes() {
        assert!(is_valid(&check_non_empty("x")));
    }

    #[test]
    fn non_empty_empty_string_rejected() {
        let result = check_non_empty("");
        assert!(is_invalid(&result));
        assert!(invalid_msg(&result).contains("field_name"));
        assert!(invalid_msg(&result).contains("cannot be empty"));
    }

    #[test]
    fn non_empty_whitespace_only_rejected() {
        assert!(is_invalid(&check_non_empty("   ")));
    }

    #[test]
    fn non_empty_tab_only_rejected() {
        assert!(is_invalid(&check_non_empty("\t")));
    }

    #[test]
    fn non_empty_newline_only_rejected() {
        assert!(is_invalid(&check_non_empty("\n")));
    }

    #[test]
    fn non_empty_mixed_whitespace_rejected() {
        assert!(is_invalid(&check_non_empty(" \t\n ")));
    }

    #[test]
    fn non_empty_leading_trailing_whitespace_passes() {
        assert!(is_valid(&check_non_empty("  hello  ")));
    }

    // ========================================================================
    // require_max_str_len! tests
    // ========================================================================

    fn check_max_str_len(s: &str, max: usize) -> ExternResult<ValidateCallbackResult> {
        require_max_str_len!(s, max, "test_str");
        Ok(ValidateCallbackResult::Valid)
    }

    #[test]
    fn max_str_len_empty_passes() {
        assert!(is_valid(&check_max_str_len("", 10)));
    }

    #[test]
    fn max_str_len_under_limit_passes() {
        assert!(is_valid(&check_max_str_len("hello", 10)));
    }

    #[test]
    fn max_str_len_at_limit_passes() {
        assert!(is_valid(&check_max_str_len("hello", 5)));
    }

    #[test]
    fn max_str_len_over_limit_rejected() {
        let result = check_max_str_len("hello!", 5);
        assert!(is_invalid(&result));
        assert!(invalid_msg(&result).contains("test_str"));
        assert!(invalid_msg(&result).contains("5"));
    }

    #[test]
    fn max_str_len_multibyte_uses_byte_count() {
        // Unicode emoji is 4 bytes
        let emoji = "\u{1F600}"; // 4 bytes
        assert!(is_valid(&check_max_str_len(emoji, 4)));
        assert!(is_invalid(&check_max_str_len(emoji, 3)));
    }

    // ========================================================================
    // require_in_range! tests
    // ========================================================================

    fn check_in_range_f64(val: f64, min: f64, max: f64) -> ExternResult<ValidateCallbackResult> {
        require_in_range!(val, min, max, "test_val");
        Ok(ValidateCallbackResult::Valid)
    }

    fn check_in_range_u8(val: u8, min: u8, max: u8) -> ExternResult<ValidateCallbackResult> {
        require_in_range!(val, min, max, "test_int");
        Ok(ValidateCallbackResult::Valid)
    }

    #[test]
    fn in_range_mid_value_passes() {
        assert!(is_valid(&check_in_range_f64(0.0, -90.0, 90.0)));
    }

    #[test]
    fn in_range_at_min_passes() {
        assert!(is_valid(&check_in_range_f64(-90.0, -90.0, 90.0)));
    }

    #[test]
    fn in_range_at_max_passes() {
        assert!(is_valid(&check_in_range_f64(90.0, -90.0, 90.0)));
    }

    #[test]
    fn in_range_below_min_rejected() {
        let result = check_in_range_f64(-90.1, -90.0, 90.0);
        assert!(is_invalid(&result));
        assert!(invalid_msg(&result).contains("test_val"));
        assert!(invalid_msg(&result).contains("-90"));
        assert!(invalid_msg(&result).contains("90"));
    }

    #[test]
    fn in_range_above_max_rejected() {
        let result = check_in_range_f64(90.1, -90.0, 90.0);
        assert!(is_invalid(&result));
    }

    #[test]
    fn in_range_integer_mid_passes() {
        assert!(is_valid(&check_in_range_u8(6, 1, 12)));
    }

    #[test]
    fn in_range_integer_at_min_passes() {
        assert!(is_valid(&check_in_range_u8(1, 1, 12)));
    }

    #[test]
    fn in_range_integer_at_max_passes() {
        assert!(is_valid(&check_in_range_u8(12, 1, 12)));
    }

    #[test]
    fn in_range_integer_below_min_rejected() {
        assert!(is_invalid(&check_in_range_u8(0, 1, 12)));
    }

    #[test]
    fn in_range_integer_above_max_rejected() {
        assert!(is_invalid(&check_in_range_u8(13, 1, 12)));
    }

    // ========================================================================
    // require_basis_points! tests
    // ========================================================================

    fn check_basis_points(val: u32) -> ExternResult<ValidateCallbackResult> {
        require_basis_points!(val, "share");
        Ok(ValidateCallbackResult::Valid)
    }

    #[test]
    fn basis_points_one_passes() {
        assert!(is_valid(&check_basis_points(1)));
    }

    #[test]
    fn basis_points_10000_passes() {
        assert!(is_valid(&check_basis_points(10000)));
    }

    #[test]
    fn basis_points_5000_passes() {
        assert!(is_valid(&check_basis_points(5000)));
    }

    #[test]
    fn basis_points_zero_rejected() {
        let result = check_basis_points(0);
        assert!(is_invalid(&result));
        assert!(invalid_msg(&result).contains("share"));
        assert!(invalid_msg(&result).contains("basis points"));
    }

    #[test]
    fn basis_points_10001_rejected() {
        let result = check_basis_points(10001);
        assert!(is_invalid(&result));
        assert!(invalid_msg(&result).contains("share"));
    }

    #[test]
    fn basis_points_u32_max_rejected() {
        assert!(is_invalid(&check_basis_points(u32::MAX)));
    }

    // ========================================================================
    // require_max_tag_len! tests
    // ========================================================================

    fn check_max_tag_len(tag_bytes: Vec<u8>, max: usize) -> ExternResult<ValidateCallbackResult> {
        let tag = LinkTag(tag_bytes);
        require_max_tag_len!(tag, max, "TestLink");
        Ok(ValidateCallbackResult::Valid)
    }

    #[test]
    fn max_tag_len_empty_passes() {
        assert!(is_valid(&check_max_tag_len(vec![], 256)));
    }

    #[test]
    fn max_tag_len_under_limit_passes() {
        assert!(is_valid(&check_max_tag_len(vec![0u8; 100], 256)));
    }

    #[test]
    fn max_tag_len_at_limit_passes() {
        assert!(is_valid(&check_max_tag_len(vec![0u8; 256], 256)));
    }

    #[test]
    fn max_tag_len_over_limit_rejected() {
        let result = check_max_tag_len(vec![0u8; 257], 256);
        assert!(is_invalid(&result));
        assert!(invalid_msg(&result).contains("TestLink"));
        assert!(invalid_msg(&result).contains("256"));
    }

    #[test]
    fn max_tag_len_over_limit_512_rejected() {
        let result = check_max_tag_len(vec![0u8; 513], 512);
        assert!(is_invalid(&result));
        assert!(invalid_msg(&result).contains("512"));
    }

    #[test]
    fn max_tag_len_at_512_passes() {
        assert!(is_valid(&check_max_tag_len(vec![0u8; 512], 512)));
    }

    // ========================================================================
    // Macro composition test (multiple macros in one function)
    // ========================================================================

    fn validate_composed(
        name: &str,
        value: f64,
        items: &[u8],
        bps: u32,
    ) -> ExternResult<ValidateCallbackResult> {
        require_non_empty!(name, "Name");
        require_max_str_len!(name, 64, "Name");
        require_finite!(value, "Value");
        require_in_range!(value, 0.0, 100.0, "Value");
        require_max_len!(items, 10, "Items");
        require_basis_points!(bps, "Share");
        Ok(ValidateCallbackResult::Valid)
    }

    #[test]
    fn composed_all_valid_passes() {
        assert!(is_valid(&validate_composed("hello", 50.0, &[1, 2, 3], 5000)));
    }

    #[test]
    fn composed_empty_name_rejected_first() {
        let result = validate_composed("", 50.0, &[1, 2, 3], 5000);
        assert!(is_invalid(&result));
        assert!(invalid_msg(&result).contains("cannot be empty"));
    }

    #[test]
    fn composed_name_too_long_rejected() {
        let result = validate_composed(&"x".repeat(65), 50.0, &[1, 2, 3], 5000);
        assert!(is_invalid(&result));
        assert!(invalid_msg(&result).contains("64"));
    }

    #[test]
    fn composed_nan_value_rejected() {
        let result = validate_composed("hello", f64::NAN, &[1, 2, 3], 5000);
        assert!(is_invalid(&result));
        assert!(invalid_msg(&result).contains("finite"));
    }

    #[test]
    fn composed_out_of_range_rejected() {
        let result = validate_composed("hello", 101.0, &[1, 2, 3], 5000);
        assert!(is_invalid(&result));
        assert!(invalid_msg(&result).contains("between"));
    }

    #[test]
    fn composed_vec_too_long_rejected() {
        let result = validate_composed("hello", 50.0, &[0u8; 11], 5000);
        assert!(is_invalid(&result));
        assert!(invalid_msg(&result).contains("Items"));
    }

    #[test]
    fn composed_zero_basis_points_rejected() {
        let result = validate_composed("hello", 50.0, &[1, 2, 3], 0);
        assert!(is_invalid(&result));
        assert!(invalid_msg(&result).contains("basis points"));
    }
}
