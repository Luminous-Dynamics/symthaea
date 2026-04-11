// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Search Components Index
 *
 * Advanced search and filtering components
 */

export { default as EpistemicSearch } from './EpistemicSearch';
export type { EpistemicFilters } from './EpistemicSearch';
export { FilterChip, TrustScoreSlider, PathLengthSelector, ActiveFiltersBar } from './EpistemicSearch';

// Advanced search with filters
export {
  AdvancedSearch,
  AdvancedSearchModal,
  SearchHistoryDropdown,
  SavedSearches,
  useAdvancedSearch,
  useSearchHistory,
} from './AdvancedSearch';
export type { SearchFilters, SavedSearch } from './AdvancedSearch';
