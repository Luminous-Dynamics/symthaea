// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Advanced Search Component
 *
 * Powerful search with epistemic filters:
 * - Full-text search
 * - Trust score filtering
 * - Epistemic tier filtering
 * - Date range
 * - Attachment filters
 * - Saved searches
 */

import { useState, useCallback, useMemo, useEffect } from 'react';
import { useDebounce } from '@/hooks/useDebounce';

// ============================================
// Types
// ============================================

export interface SearchFilters {
  query: string;
  trustScoreMin?: number;
  trustScoreMax?: number;
  epistemicTiers?: number[];
  assuranceLevels?: string[];
  dateFrom?: string;
  dateTo?: string;
  hasAttachments?: boolean;
  isUnread?: boolean;
  isStarred?: boolean;
  folders?: string[];
  senders?: string[];
  hasClaims?: boolean;
  relationshipTypes?: string[];
}

export interface SavedSearch {
  id: string;
  name: string;
  filters: SearchFilters;
  createdAt: string;
  lastUsed?: string;
}

interface SearchSuggestion {
  type: 'contact' | 'subject' | 'term' | 'filter';
  value: string;
  label: string;
  icon?: string;
}

// ============================================
// Search Input with Suggestions
// ============================================

interface SearchInputProps {
  value: string;
  onChange: (value: string) => void;
  onSubmit: () => void;
  suggestions?: SearchSuggestion[];
  placeholder?: string;
  autoFocus?: boolean;
}

function SearchInput({
  value,
  onChange,
  onSubmit,
  suggestions = [],
  placeholder = 'Search emails...',
  autoFocus = false,
}: SearchInputProps) {
  const [showSuggestions, setShowSuggestions] = useState(false);
  const [selectedIndex, setSelectedIndex] = useState(-1);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (!showSuggestions || suggestions.length === 0) {
      if (e.key === 'Enter') {
        onSubmit();
      }
      return;
    }

    switch (e.key) {
      case 'ArrowDown':
        e.preventDefault();
        setSelectedIndex((i) => Math.min(i + 1, suggestions.length - 1));
        break;
      case 'ArrowUp':
        e.preventDefault();
        setSelectedIndex((i) => Math.max(i - 1, -1));
        break;
      case 'Enter':
        e.preventDefault();
        if (selectedIndex >= 0) {
          onChange(suggestions[selectedIndex].value);
        }
        onSubmit();
        setShowSuggestions(false);
        break;
      case 'Escape':
        setShowSuggestions(false);
        break;
    }
  };

  return (
    <div className="relative">
      <div className="relative">
        <svg
          className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-400"
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"
          />
        </svg>
        <input
          type="text"
          value={value}
          onChange={(e) => {
            onChange(e.target.value);
            setShowSuggestions(true);
            setSelectedIndex(-1);
          }}
          onKeyDown={handleKeyDown}
          onFocus={() => setShowSuggestions(true)}
          onBlur={() => setTimeout(() => setShowSuggestions(false), 200)}
          placeholder={placeholder}
          autoFocus={autoFocus}
          className="w-full pl-10 pr-4 py-3 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          data-testid="search-input"
        />
      </div>

      {/* Suggestions dropdown */}
      {showSuggestions && suggestions.length > 0 && (
        <div className="absolute top-full left-0 right-0 mt-1 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg shadow-lg z-50 max-h-64 overflow-auto">
          {suggestions.map((suggestion, index) => (
            <button
              key={`${suggestion.type}-${suggestion.value}`}
              className={`w-full px-4 py-2 text-left flex items-center gap-3 hover:bg-gray-50 dark:hover:bg-gray-700 ${
                index === selectedIndex ? 'bg-gray-50 dark:bg-gray-700' : ''
              }`}
              onClick={() => {
                onChange(suggestion.value);
                onSubmit();
                setShowSuggestions(false);
              }}
            >
              <span className="text-gray-400">{suggestion.icon || '🔍'}</span>
              <div>
                <div className="text-sm text-gray-900 dark:text-gray-100">{suggestion.label}</div>
                <div className="text-xs text-gray-500">{suggestion.type}</div>
              </div>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

// ============================================
// Filter Components
// ============================================

interface TrustScoreFilterProps {
  min?: number;
  max?: number;
  onChange: (min: number | undefined, max: number | undefined) => void;
}

function TrustScoreFilter({ min, max, onChange }: TrustScoreFilterProps) {
  return (
    <div className="space-y-2">
      <label className="text-sm font-medium text-gray-700 dark:text-gray-300">
        Trust Score
      </label>
      <div className="flex items-center gap-2">
        <input
          type="number"
          min="0"
          max="100"
          value={min ?? ''}
          onChange={(e) => onChange(e.target.value ? Number(e.target.value) : undefined, max)}
          placeholder="Min"
          className="w-20 px-2 py-1 text-sm border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-800"
          data-testid="filter-trust-min"
        />
        <span className="text-gray-400">-</span>
        <input
          type="number"
          min="0"
          max="100"
          value={max ?? ''}
          onChange={(e) => onChange(min, e.target.value ? Number(e.target.value) : undefined)}
          placeholder="Max"
          className="w-20 px-2 py-1 text-sm border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-800"
        />
        <span className="text-sm text-gray-500">%</span>
      </div>
    </div>
  );
}

interface TierFilterProps {
  selected: number[];
  onChange: (tiers: number[]) => void;
}

function TierFilter({ selected, onChange }: TierFilterProps) {
  const tiers = [
    { value: 0, label: 'T0', description: 'Unverifiable' },
    { value: 1, label: 'T1', description: 'Email verified' },
    { value: 2, label: 'T2', description: 'Identity verified' },
    { value: 3, label: 'T3', description: 'In network' },
    { value: 4, label: 'T4', description: 'Fully attested' },
  ];

  const toggle = (tier: number) => {
    if (selected.includes(tier)) {
      onChange(selected.filter((t) => t !== tier));
    } else {
      onChange([...selected, tier]);
    }
  };

  return (
    <div className="space-y-2">
      <label className="text-sm font-medium text-gray-700 dark:text-gray-300">
        Epistemic Tier
      </label>
      <div className="flex flex-wrap gap-2">
        {tiers.map((tier) => (
          <button
            key={tier.value}
            onClick={() => toggle(tier.value)}
            className={`px-3 py-1 text-sm rounded-full transition-colors ${
              selected.includes(tier.value)
                ? 'bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300 border-2 border-blue-500'
                : 'bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-400 border-2 border-transparent'
            }`}
            title={tier.description}
          >
            {tier.label}
          </button>
        ))}
      </div>
    </div>
  );
}

interface DateRangeFilterProps {
  from?: string;
  to?: string;
  onChange: (from: string | undefined, to: string | undefined) => void;
}

function DateRangeFilter({ from, to, onChange }: DateRangeFilterProps) {
  const presets = [
    { label: 'Today', days: 0 },
    { label: 'Week', days: 7 },
    { label: 'Month', days: 30 },
    { label: '3 Months', days: 90 },
    { label: 'Year', days: 365 },
  ];

  const applyPreset = (days: number) => {
    const toDate = new Date();
    const fromDate = new Date();
    fromDate.setDate(fromDate.getDate() - days);

    onChange(fromDate.toISOString().split('T')[0], toDate.toISOString().split('T')[0]);
  };

  return (
    <div className="space-y-2">
      <label className="text-sm font-medium text-gray-700 dark:text-gray-300">
        Date Range
      </label>
      <div className="flex flex-wrap gap-1 mb-2">
        {presets.map((preset) => (
          <button
            key={preset.label}
            onClick={() => applyPreset(preset.days)}
            className="px-2 py-1 text-xs bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-400 rounded hover:bg-gray-200 dark:hover:bg-gray-700"
          >
            {preset.label}
          </button>
        ))}
      </div>
      <div className="flex items-center gap-2">
        <input
          type="date"
          value={from ?? ''}
          onChange={(e) => onChange(e.target.value || undefined, to)}
          className="flex-1 px-2 py-1 text-sm border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-800"
        />
        <span className="text-gray-400">-</span>
        <input
          type="date"
          value={to ?? ''}
          onChange={(e) => onChange(from, e.target.value || undefined)}
          className="flex-1 px-2 py-1 text-sm border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-800"
        />
      </div>
    </div>
  );
}

// ============================================
// Saved Searches
// ============================================

interface SavedSearchesProps {
  searches: SavedSearch[];
  onSelect: (search: SavedSearch) => void;
  onDelete: (id: string) => void;
}

function SavedSearches({ searches, onSelect, onDelete }: SavedSearchesProps) {
  if (searches.length === 0) {
    return (
      <div className="text-sm text-gray-500 text-center py-4">
        No saved searches yet
      </div>
    );
  }

  return (
    <div className="space-y-2">
      {searches.map((search) => (
        <div
          key={search.id}
          className="flex items-center justify-between p-2 bg-gray-50 dark:bg-gray-800 rounded-lg"
        >
          <button
            onClick={() => onSelect(search)}
            className="flex-1 text-left"
          >
            <div className="font-medium text-sm text-gray-900 dark:text-gray-100">
              {search.name}
            </div>
            <div className="text-xs text-gray-500">
              {search.filters.query || 'No query'}
            </div>
          </button>
          <button
            onClick={() => onDelete(search.id)}
            className="p-1 text-gray-400 hover:text-red-500"
          >
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>
      ))}
    </div>
  );
}

// ============================================
// Main Advanced Search Component
// ============================================

interface AdvancedSearchProps {
  isOpen: boolean;
  onClose: () => void;
  onSearch: (filters: SearchFilters) => void;
  initialFilters?: SearchFilters;
}

export function AdvancedSearch({
  isOpen,
  onClose,
  onSearch,
  initialFilters,
}: AdvancedSearchProps) {
  const [filters, setFilters] = useState<SearchFilters>(
    initialFilters || { query: '' }
  );
  const [savedSearches, setSavedSearches] = useState<SavedSearch[]>([]);
  const [saveDialogOpen, setSaveDialogOpen] = useState(false);
  const [saveName, setSaveName] = useState('');

  // Load saved searches from localStorage
  useEffect(() => {
    const saved = localStorage.getItem('mycelix_saved_searches');
    if (saved) {
      setSavedSearches(JSON.parse(saved));
    }
  }, []);

  const updateFilter = useCallback(<K extends keyof SearchFilters>(
    key: K,
    value: SearchFilters[K]
  ) => {
    setFilters((prev) => ({ ...prev, [key]: value }));
  }, []);

  const clearFilters = useCallback(() => {
    setFilters({ query: '' });
  }, []);

  const handleSearch = useCallback(() => {
    onSearch(filters);
    onClose();
  }, [filters, onSearch, onClose]);

  const handleSaveSearch = useCallback(() => {
    if (!saveName.trim()) return;

    const newSearch: SavedSearch = {
      id: Date.now().toString(),
      name: saveName,
      filters,
      createdAt: new Date().toISOString(),
    };

    const updated = [...savedSearches, newSearch];
    setSavedSearches(updated);
    localStorage.setItem('mycelix_saved_searches', JSON.stringify(updated));
    setSaveDialogOpen(false);
    setSaveName('');
  }, [saveName, filters, savedSearches]);

  const handleDeleteSavedSearch = useCallback((id: string) => {
    const updated = savedSearches.filter((s) => s.id !== id);
    setSavedSearches(updated);
    localStorage.setItem('mycelix_saved_searches', JSON.stringify(updated));
  }, [savedSearches]);

  const handleSelectSavedSearch = useCallback((search: SavedSearch) => {
    setFilters(search.filters);
  }, []);

  const activeFilterCount = useMemo(() => {
    let count = 0;
    if (filters.trustScoreMin !== undefined) count++;
    if (filters.trustScoreMax !== undefined) count++;
    if (filters.epistemicTiers?.length) count++;
    if (filters.dateFrom) count++;
    if (filters.dateTo) count++;
    if (filters.hasAttachments) count++;
    if (filters.isUnread) count++;
    if (filters.isStarred) count++;
    if (filters.hasClaims) count++;
    return count;
  }, [filters]);

  if (!isOpen) return null;

  return (
    <>
      {/* Backdrop */}
      <div className="fixed inset-0 bg-black/50 z-50" onClick={onClose} />

      {/* Modal */}
      <div
        className="fixed inset-4 md:inset-auto md:top-1/2 md:left-1/2 md:-translate-x-1/2 md:-translate-y-1/2 md:w-[700px] md:max-h-[85vh] bg-white dark:bg-gray-900 rounded-xl shadow-2xl z-50 flex flex-col overflow-hidden"
        data-testid="advanced-search"
      >
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-gray-200 dark:border-gray-700">
          <div className="flex items-center gap-3">
            <span className="text-2xl">🔍</span>
            <h2 className="text-xl font-bold text-gray-900 dark:text-gray-100">
              Advanced Search
            </h2>
            {activeFilterCount > 0 && (
              <span className="px-2 py-0.5 text-xs bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300 rounded-full">
                {activeFilterCount} filter{activeFilterCount !== 1 ? 's' : ''}
              </span>
            )}
          </div>
          <button
            onClick={onClose}
            className="p-2 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors"
          >
            <svg className="w-5 h-5 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-6">
          <div className="space-y-6">
            {/* Search input */}
            <SearchInput
              value={filters.query}
              onChange={(value) => updateFilter('query', value)}
              onSubmit={handleSearch}
              autoFocus
            />

            {/* Filters grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* Trust Score */}
              <TrustScoreFilter
                min={filters.trustScoreMin}
                max={filters.trustScoreMax}
                onChange={(min, max) => {
                  updateFilter('trustScoreMin', min);
                  updateFilter('trustScoreMax', max);
                }}
              />

              {/* Epistemic Tier */}
              <TierFilter
                selected={filters.epistemicTiers || []}
                onChange={(tiers) => updateFilter('epistemicTiers', tiers)}
              />

              {/* Date Range */}
              <div className="md:col-span-2">
                <DateRangeFilter
                  from={filters.dateFrom}
                  to={filters.dateTo}
                  onChange={(from, to) => {
                    updateFilter('dateFrom', from);
                    updateFilter('dateTo', to);
                  }}
                />
              </div>
            </div>

            {/* Toggle filters */}
            <div className="flex flex-wrap gap-3">
              {[
                { key: 'hasAttachments', label: 'Has Attachments', icon: '📎' },
                { key: 'isUnread', label: 'Unread', icon: '📬' },
                { key: 'isStarred', label: 'Starred', icon: '⭐' },
                { key: 'hasClaims', label: 'Has Claims', icon: '🔐' },
              ].map(({ key, label, icon }) => (
                <button
                  key={key}
                  onClick={() => updateFilter(key as keyof SearchFilters, !filters[key as keyof SearchFilters])}
                  className={`flex items-center gap-2 px-3 py-2 rounded-lg transition-colors ${
                    filters[key as keyof SearchFilters]
                      ? 'bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300'
                      : 'bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-400'
                  }`}
                >
                  <span>{icon}</span>
                  <span className="text-sm">{label}</span>
                </button>
              ))}
            </div>

            {/* Saved searches */}
            <div className="border-t border-gray-200 dark:border-gray-700 pt-4">
              <div className="flex items-center justify-between mb-3">
                <h3 className="font-medium text-gray-900 dark:text-gray-100">
                  Saved Searches
                </h3>
                <button
                  onClick={() => setSaveDialogOpen(true)}
                  className="text-sm text-blue-600 dark:text-blue-400 hover:underline"
                >
                  Save current
                </button>
              </div>
              <SavedSearches
                searches={savedSearches}
                onSelect={handleSelectSavedSearch}
                onDelete={handleDeleteSavedSearch}
              />
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="flex items-center justify-between px-6 py-4 border-t border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-800">
          <button
            onClick={clearFilters}
            className="text-sm text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-100"
          >
            Clear all
          </button>
          <div className="flex gap-3">
            <button
              onClick={onClose}
              className="px-4 py-2 text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-100"
            >
              Cancel
            </button>
            <button
              onClick={handleSearch}
              className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors"
              data-testid="apply-filters"
            >
              Search
            </button>
          </div>
        </div>

        {/* Save dialog */}
        {saveDialogOpen && (
          <div className="absolute inset-0 bg-black/50 flex items-center justify-center">
            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 w-80 shadow-xl">
              <h4 className="font-semibold text-gray-900 dark:text-gray-100 mb-4">
                Save Search
              </h4>
              <input
                type="text"
                value={saveName}
                onChange={(e) => setSaveName(e.target.value)}
                placeholder="Search name..."
                className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-900 mb-4"
                autoFocus
              />
              <div className="flex justify-end gap-2">
                <button
                  onClick={() => setSaveDialogOpen(false)}
                  className="px-3 py-1.5 text-gray-600 dark:text-gray-400"
                >
                  Cancel
                </button>
                <button
                  onClick={handleSaveSearch}
                  className="px-3 py-1.5 bg-blue-600 text-white rounded-lg"
                >
                  Save
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    </>
  );
}

// ============================================
// Search Hook
// ============================================

export function useAdvancedSearch() {
  const [isOpen, setIsOpen] = useState(false);
  const [filters, setFilters] = useState<SearchFilters>({ query: '' });

  // Keyboard shortcut to open
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.ctrlKey && e.shiftKey && e.key === 'f') {
        e.preventDefault();
        setIsOpen(true);
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, []);

  return {
    isOpen,
    open: () => setIsOpen(true),
    close: () => setIsOpen(false),
    filters,
    setFilters,
  };
}

export default AdvancedSearch;
