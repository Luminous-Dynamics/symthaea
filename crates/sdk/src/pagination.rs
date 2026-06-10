// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Pagination Module
//!
//! Provides standardized pagination utilities for Mycelix zomes.
//! Supports cursor-based and offset-based pagination patterns.
//!
//! ## Example
//!
//! ```rust
//! use mycelix_sdk::pagination::{PaginationRequest, PaginatedResponse, paginate_vec};
//!
//! let items = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
//! let request = PaginationRequest::new(3, None); // Page size 3
//!
//! let response = paginate_vec(items, &request);
//! assert_eq!(response.items.len(), 3);
//! assert!(response.has_more);
//! ```

use serde::{Deserialize, Serialize};

/// Default page size when not specified
pub const DEFAULT_PAGE_SIZE: u32 = 50;

/// Maximum allowed page size to prevent resource exhaustion
pub const MAX_PAGE_SIZE: u32 = 500;

/// Pagination request parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PaginationRequest {
    /// Number of items per page (capped at MAX_PAGE_SIZE)
    pub page_size: u32,
    /// Cursor for cursor-based pagination (opaque string)
    pub cursor: Option<String>,
    /// Offset for offset-based pagination
    pub offset: Option<u32>,
    /// Sort direction
    pub sort_desc: bool,
}

impl Default for PaginationRequest {
    fn default() -> Self {
        Self {
            page_size: DEFAULT_PAGE_SIZE,
            cursor: None,
            offset: None,
            sort_desc: true, // Most recent first by default
        }
    }
}

impl PaginationRequest {
    /// Create a new pagination request with page size
    pub fn new(page_size: u32, cursor: Option<String>) -> Self {
        Self {
            page_size: page_size.min(MAX_PAGE_SIZE),
            cursor,
            offset: None,
            sort_desc: true,
        }
    }

    /// Create with offset-based pagination
    pub fn with_offset(page_size: u32, offset: u32) -> Self {
        Self {
            page_size: page_size.min(MAX_PAGE_SIZE),
            cursor: None,
            offset: Some(offset),
            sort_desc: true,
        }
    }

    /// Get effective page size (capped)
    pub fn effective_page_size(&self) -> usize {
        self.page_size.min(MAX_PAGE_SIZE) as usize
    }

    /// Get effective offset
    pub fn effective_offset(&self) -> usize {
        self.offset.unwrap_or(0) as usize
    }
}

/// Paginated response wrapper
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PaginatedResponse<T> {
    /// Items in this page
    pub items: Vec<T>,
    /// Total count of items (if available)
    pub total_count: Option<u64>,
    /// Cursor for next page (if cursor-based)
    pub next_cursor: Option<String>,
    /// Whether there are more items
    pub has_more: bool,
    /// Current page number (if offset-based)
    pub page: Option<u32>,
    /// Total pages (if offset-based and total known)
    pub total_pages: Option<u32>,
}

impl<T> PaginatedResponse<T> {
    /// Create an empty response
    pub fn empty() -> Self {
        Self {
            items: vec![],
            total_count: Some(0),
            next_cursor: None,
            has_more: false,
            page: Some(0),
            total_pages: Some(0),
        }
    }

    /// Create from items with pagination info
    pub fn new(items: Vec<T>, has_more: bool, next_cursor: Option<String>) -> Self {
        Self {
            items,
            total_count: None,
            next_cursor,
            has_more,
            page: None,
            total_pages: None,
        }
    }

    /// Set total count and calculate total pages
    pub fn with_total(mut self, total: u64, page_size: u32) -> Self {
        self.total_count = Some(total);
        self.total_pages = Some((total as f64 / page_size as f64).ceil() as u32);
        self
    }

    /// Set current page
    pub fn with_page(mut self, page: u32) -> Self {
        self.page = Some(page);
        self
    }
}

/// Paginate a vector of items
pub fn paginate_vec<T: Clone>(items: Vec<T>, request: &PaginationRequest) -> PaginatedResponse<T> {
    let total = items.len();
    let offset = request.effective_offset();
    let page_size = request.effective_page_size();

    if offset >= total {
        return PaginatedResponse::empty().with_total(total as u64, request.page_size);
    }

    let end = (offset + page_size).min(total);
    let page_items: Vec<T> = items[offset..end].to_vec();
    let has_more = end < total;

    let next_cursor = if has_more {
        Some(end.to_string())
    } else {
        None
    };

    PaginatedResponse::new(page_items, has_more, next_cursor)
        .with_total(total as u64, request.page_size)
        .with_page((offset / page_size) as u32)
}

/// Cursor encoding/decoding utilities
pub mod cursor {

    /// Encode a cursor from timestamp and optional ID
    pub fn encode(timestamp_micros: i64, id: Option<&str>) -> String {
        match id {
            Some(id) => format!("{}:{}", timestamp_micros, id),
            None => timestamp_micros.to_string(),
        }
    }

    /// Decode a cursor to timestamp and optional ID
    pub fn decode(cursor: &str) -> Option<(i64, Option<String>)> {
        if cursor.contains(':') {
            let parts: Vec<&str> = cursor.splitn(2, ':').collect();
            if parts.len() == 2 {
                let ts = parts[0].parse().ok()?;
                Some((ts, Some(parts[1].to_string())))
            } else {
                None
            }
        } else {
            let ts = cursor.parse().ok()?;
            Some((ts, None))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_paginate_vec() {
        let items: Vec<i32> = (1..=100).collect();

        // First page
        let req = PaginationRequest::with_offset(10, 0);
        let resp = paginate_vec(items.clone(), &req);
        assert_eq!(resp.items.len(), 10);
        assert_eq!(resp.items[0], 1);
        assert!(resp.has_more);
        assert_eq!(resp.total_count, Some(100));
        assert_eq!(resp.page, Some(0));

        // Middle page
        let req = PaginationRequest::with_offset(10, 50);
        let resp = paginate_vec(items.clone(), &req);
        assert_eq!(resp.items.len(), 10);
        assert_eq!(resp.items[0], 51);
        assert!(resp.has_more);
        assert_eq!(resp.page, Some(5));

        // Last page
        let req = PaginationRequest::with_offset(10, 90);
        let resp = paginate_vec(items.clone(), &req);
        assert_eq!(resp.items.len(), 10);
        assert_eq!(resp.items[0], 91);
        assert!(!resp.has_more);
    }

    #[test]
    fn test_cursor_encode_decode() {
        let ts = 1704825600_000_000i64;

        // Without ID
        let cursor = cursor::encode(ts, None);
        let (decoded_ts, decoded_id) = cursor::decode(&cursor).unwrap();
        assert_eq!(decoded_ts, ts);
        assert!(decoded_id.is_none());

        // With ID
        let cursor = cursor::encode(ts, Some("abc123"));
        let (decoded_ts, decoded_id) = cursor::decode(&cursor).unwrap();
        assert_eq!(decoded_ts, ts);
        assert_eq!(decoded_id, Some("abc123".to_string()));
    }

    #[test]
    fn test_page_size_cap() {
        let req = PaginationRequest::new(1000, None);
        assert_eq!(req.effective_page_size(), MAX_PAGE_SIZE as usize);
    }
}
