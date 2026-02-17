//! Integration tests for Mycelix-DeSci API
//!
//! These tests verify the complete functionality of the REST API,
//! including claim lifecycle, query operations, trust management,
//! and error handling.

mod helpers;

mod test_api_claims;
mod test_api_query;
mod test_api_trust;
mod test_api_system;
mod test_claim_lifecycle;
mod test_concurrent_operations;
mod test_error_handling;
