// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Query filters and sorting

use crate::claims::EpistemicTier;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Query filter for searching claims
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct QueryFilter {
    /// Filter by category (partial match, case-insensitive)
    pub category: Option<String>,

    /// Filter by minimum epistemic tier
    pub min_tier: Option<EpistemicTier>,

    /// Filter by maximum epistemic tier
    pub max_tier: Option<EpistemicTier>,

    /// Filter by keywords (all must match)
    pub keywords: Vec<String>,

    /// Filter by creator
    pub creator: Option<String>,

    /// Filter by date range (created_at)
    pub date_from: Option<DateTime<Utc>>,
    pub date_to: Option<DateTime<Utc>>,

    /// Filter by minimum number of verifications
    pub min_verifications: Option<usize>,

    /// Filter by license
    pub license: Option<String>,

    /// Limit number of results
    pub limit: Option<usize>,

    /// Offset for pagination
    pub offset: Option<usize>,

    /// Sorting
    pub sort_by: Option<SortBy>,
    pub sort_order: Option<SortOrder>,
}

/// Sorting field
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum SortBy {
    /// Sort by creation date
    CreatedAt,

    /// Sort by last update date
    UpdatedAt,

    /// Sort by epistemic tier
    EpistemicTier,

    /// Sort by number of verifications
    VerificationCount,

    /// Sort by category (alphabetically)
    Category,
}

/// Sorting order
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum SortOrder {
    /// Ascending order
    Ascending,

    /// Descending order
    Descending,
}

impl QueryFilter {
    /// Create a new empty filter
    pub fn new() -> Self {
        Self::default()
    }

    /// Set category filter
    pub fn with_category(mut self, category: String) -> Self {
        self.category = Some(category);
        self
    }

    /// Set minimum tier filter
    pub fn with_min_tier(mut self, tier: EpistemicTier) -> Self {
        self.min_tier = Some(tier);
        self
    }

    /// Add keyword to filter
    pub fn with_keyword(mut self, keyword: String) -> Self {
        self.keywords.push(keyword);
        self
    }

    /// Set keywords filter
    pub fn with_keywords(mut self, keywords: Vec<String>) -> Self {
        self.keywords = keywords;
        self
    }

    /// Set creator filter
    pub fn with_creator(mut self, creator: String) -> Self {
        self.creator = Some(creator);
        self
    }

    /// Set date range
    pub fn with_date_range(mut self, from: DateTime<Utc>, to: DateTime<Utc>) -> Self {
        self.date_from = Some(from);
        self.date_to = Some(to);
        self
    }

    /// Set limit
    pub fn with_limit(mut self, limit: usize) -> Self {
        self.limit = Some(limit);
        self
    }

    /// Set offset
    pub fn with_offset(mut self, offset: usize) -> Self {
        self.offset = Some(offset);
        self
    }

    /// Set sorting
    pub fn with_sort(mut self, by: SortBy, order: SortOrder) -> Self {
        self.sort_by = Some(by);
        self.sort_order = Some(order);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_filter() {
        let filter = QueryFilter::new()
            .with_category("genomics".to_string())
            .with_min_tier(EpistemicTier::E3)
            .with_keyword("CRISPR".to_string())
            .with_limit(10);

        assert_eq!(filter.category, Some("genomics".to_string()));
        assert_eq!(filter.min_tier, Some(EpistemicTier::E3));
        assert_eq!(filter.keywords.len(), 1);
        assert_eq!(filter.limit, Some(10));
    }

    #[test]
    fn test_multiple_keywords() {
        let filter = QueryFilter::new()
            .with_keywords(vec!["NAD+".to_string(), "aging".to_string()]);

        assert_eq!(filter.keywords.len(), 2);
    }
}
