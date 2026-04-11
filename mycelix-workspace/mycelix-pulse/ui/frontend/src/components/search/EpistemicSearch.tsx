// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Epistemic Search Component
 *
 * Advanced search interface for filtering emails by:
 * - Epistemic tiers (T0-T4 verifiability levels)
 * - Trust scores and path lengths
 * - Assurance levels (E0-E4 identity proofs)
 * - Claim types and verification status
 * - AI-detected intent categories
 *
 * Integrates with the email list to provide powerful filtering.
 */

import { useState, useMemo, useCallback } from 'react';
import { useQuery } from '@tanstack/react-query';
import { api } from '@/services/api';
import type { AssuranceLevel } from '@/services/api';

export interface EpistemicFilters {
  // Trust filters
  minTrustScore?: number;
  maxPathLength?: number;
  requireTrustPath?: boolean;

  // Assurance level filters
  minAssuranceLevel?: AssuranceLevel;
  assuranceLevels?: AssuranceLevel[];

  // Epistemic tier filters (T0-T4)
  minTier?: number;
  maxTier?: number;
  tiers?: number[];

  // Claim filters
  hasVerifiedClaims?: boolean;
  claimTypes?: string[];

  // AI intent filters
  intents?: string[];
  minPriority?: number;

  // Standard filters
  searchQuery?: string;
  dateFrom?: string;
  dateTo?: string;
  hasAttachments?: boolean;
  isUnread?: boolean;
}

interface EpistemicSearchProps {
  onFiltersChange: (filters: EpistemicFilters) => void;
  currentFilters?: EpistemicFilters;
  compact?: boolean;
  className?: string;
}

// Epistemic tier configuration
const tierConfig: Record<number, {
  label: string;
  description: string;
  color: string;
  bgColor: string;
  examples: string[];
}> = {
  0: {
    label: 'T0: Unverifiable',
    description: 'No way to verify authenticity',
    color: 'text-gray-600',
    bgColor: 'bg-gray-100 dark:bg-gray-800',
    examples: ['Anonymous sender', 'No cryptographic proof'],
  },
  1: {
    label: 'T1: Sender Known',
    description: 'Email ownership verified',
    color: 'text-blue-600',
    bgColor: 'bg-blue-50 dark:bg-blue-900/30',
    examples: ['DKIM verified', 'SPF pass'],
  },
  2: {
    label: 'T2: Identity Verified',
    description: 'Real identity attestation',
    color: 'text-indigo-600',
    bgColor: 'bg-indigo-50 dark:bg-indigo-900/30',
    examples: ['Gitcoin Passport', 'GitHub verified'],
  },
  3: {
    label: 'T3: Trust Connected',
    description: 'In your trust graph',
    color: 'text-purple-600',
    bgColor: 'bg-purple-50 dark:bg-purple-900/30',
    examples: ['Direct contact', 'Introduced by trusted party'],
  },
  4: {
    label: 'T4: Fully Attested',
    description: 'Multiple verifiable claims',
    color: 'text-emerald-600',
    bgColor: 'bg-emerald-50 dark:bg-emerald-900/30',
    examples: ['Professional credentials', 'Multi-factor verified'],
  },
};

// Assurance level configuration
const assuranceConfig: Record<AssuranceLevel, {
  label: string;
  shortLabel: string;
  description: string;
  color: string;
  bgColor: string;
  attackCost: string;
}> = {
  e0_anonymous: {
    label: 'Anonymous',
    shortLabel: 'E0',
    description: 'No identity verification',
    color: 'text-gray-600',
    bgColor: 'bg-gray-100 dark:bg-gray-800',
    attackCost: '$0',
  },
  e1_verified_email: {
    label: 'Email Verified',
    shortLabel: 'E1',
    description: 'Email address ownership proven',
    color: 'text-blue-600',
    bgColor: 'bg-blue-50 dark:bg-blue-900/30',
    attackCost: '$100',
  },
  e2_gitcoin_passport: {
    label: 'Gitcoin Passport',
    shortLabel: 'E2',
    description: 'Sybil-resistant identity',
    color: 'text-indigo-600',
    bgColor: 'bg-indigo-50 dark:bg-indigo-900/30',
    attackCost: '$1,000',
  },
  e3_multi_factor: {
    label: 'Multi-Factor',
    shortLabel: 'E3',
    description: 'Multiple identity proofs',
    color: 'text-purple-600',
    bgColor: 'bg-purple-50 dark:bg-purple-900/30',
    attackCost: '$100K',
  },
  e4_constitutional: {
    label: 'Constitutional',
    shortLabel: 'E4',
    description: 'Highest assurance level',
    color: 'text-emerald-600',
    bgColor: 'bg-emerald-50 dark:bg-emerald-900/30',
    attackCost: '$10M',
  },
};

// Intent categories
const intentCategories = [
  { id: 'urgent_request', label: 'Urgent Request', icon: '🔴', color: 'red' },
  { id: 'question', label: 'Question', icon: '❓', color: 'blue' },
  { id: 'information', label: 'Information', icon: 'ℹ️', color: 'gray' },
  { id: 'action_required', label: 'Action Required', icon: '⚡', color: 'amber' },
  { id: 'follow_up', label: 'Follow Up', icon: '🔄', color: 'purple' },
  { id: 'introduction', label: 'Introduction', icon: '🤝', color: 'green' },
  { id: 'scheduling', label: 'Scheduling', icon: '📅', color: 'teal' },
  { id: 'promotional', label: 'Promotional', icon: '📢', color: 'orange' },
];

// Filter chip component
function FilterChip({
  label,
  isActive,
  onClick,
  color = 'gray',
}: {
  label: string;
  isActive: boolean;
  onClick: () => void;
  color?: string;
}) {
  const colorClasses: Record<string, string> = {
    gray: isActive ? 'bg-gray-200 dark:bg-gray-700 text-gray-900 dark:text-gray-100' : 'bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-400',
    blue: isActive ? 'bg-blue-100 dark:bg-blue-900/40 text-blue-700 dark:text-blue-300' : 'bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-400',
    indigo: isActive ? 'bg-indigo-100 dark:bg-indigo-900/40 text-indigo-700 dark:text-indigo-300' : 'bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-400',
    purple: isActive ? 'bg-purple-100 dark:bg-purple-900/40 text-purple-700 dark:text-purple-300' : 'bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-400',
    emerald: isActive ? 'bg-emerald-100 dark:bg-emerald-900/40 text-emerald-700 dark:text-emerald-300' : 'bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-400',
    red: isActive ? 'bg-red-100 dark:bg-red-900/40 text-red-700 dark:text-red-300' : 'bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-400',
    amber: isActive ? 'bg-amber-100 dark:bg-amber-900/40 text-amber-700 dark:text-amber-300' : 'bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-400',
    green: isActive ? 'bg-green-100 dark:bg-green-900/40 text-green-700 dark:text-green-300' : 'bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-400',
    teal: isActive ? 'bg-teal-100 dark:bg-teal-900/40 text-teal-700 dark:text-teal-300' : 'bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-400',
    orange: isActive ? 'bg-orange-100 dark:bg-orange-900/40 text-orange-700 dark:text-orange-300' : 'bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-400',
  };

  return (
    <button
      type="button"
      onClick={onClick}
      className={`px-3 py-1.5 rounded-full text-xs font-medium transition-colors ${colorClasses[color]} hover:opacity-80`}
    >
      {label}
    </button>
  );
}

// Trust score slider
function TrustScoreSlider({
  value,
  onChange,
}: {
  value: number;
  onChange: (value: number) => void;
}) {
  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <span className="text-xs text-gray-500 dark:text-gray-400">Min Trust Score</span>
        <span className={`text-sm font-medium ${
          value >= 0.7 ? 'text-emerald-600' :
          value >= 0.4 ? 'text-amber-600' :
          'text-red-600'
        }`}>
          {Math.round(value * 100)}%
        </span>
      </div>
      <input
        type="range"
        min="0"
        max="1"
        step="0.1"
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="w-full h-2 bg-gray-200 dark:bg-gray-700 rounded-lg appearance-none cursor-pointer"
      />
      <div className="flex justify-between text-xs text-gray-400">
        <span>Any</span>
        <span>Medium</span>
        <span>High</span>
      </div>
    </div>
  );
}

// Path length selector
function PathLengthSelector({
  value,
  onChange,
}: {
  value: number | undefined;
  onChange: (value: number | undefined) => void;
}) {
  const options = [
    { value: undefined, label: 'Any' },
    { value: 1, label: '1 hop (Direct)' },
    { value: 2, label: '2 hops' },
    { value: 3, label: '3 hops' },
  ];

  return (
    <div className="space-y-2">
      <span className="text-xs text-gray-500 dark:text-gray-400">Max Path Length</span>
      <div className="flex gap-2">
        {options.map((option) => (
          <button
            key={option.label}
            type="button"
            onClick={() => onChange(option.value)}
            className={`flex-1 px-2 py-1.5 rounded text-xs font-medium transition-colors ${
              value === option.value
                ? 'bg-blue-100 dark:bg-blue-900/40 text-blue-700 dark:text-blue-300'
                : 'bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-400 hover:bg-gray-200 dark:hover:bg-gray-700'
            }`}
          >
            {option.label}
          </button>
        ))}
      </div>
    </div>
  );
}

// Active filters display
function ActiveFiltersBar({
  filters,
  onRemove,
  onClear,
}: {
  filters: EpistemicFilters;
  onRemove: (key: keyof EpistemicFilters) => void;
  onClear: () => void;
}) {
  const activeFilters: Array<{ key: keyof EpistemicFilters; label: string }> = [];

  if (filters.minTrustScore && filters.minTrustScore > 0) {
    activeFilters.push({ key: 'minTrustScore', label: `Trust ≥${Math.round(filters.minTrustScore * 100)}%` });
  }
  if (filters.maxPathLength) {
    activeFilters.push({ key: 'maxPathLength', label: `≤${filters.maxPathLength} hops` });
  }
  if (filters.requireTrustPath) {
    activeFilters.push({ key: 'requireTrustPath', label: 'In trust network' });
  }
  if (filters.tiers && filters.tiers.length > 0) {
    activeFilters.push({ key: 'tiers', label: `Tiers: ${filters.tiers.map(t => `T${t}`).join(', ')}` });
  }
  if (filters.assuranceLevels && filters.assuranceLevels.length > 0) {
    activeFilters.push({
      key: 'assuranceLevels',
      label: `Assurance: ${filters.assuranceLevels.map(a => assuranceConfig[a].shortLabel).join(', ')}`,
    });
  }
  if (filters.intents && filters.intents.length > 0) {
    activeFilters.push({ key: 'intents', label: `Intents: ${filters.intents.length}` });
  }
  if (filters.hasVerifiedClaims) {
    activeFilters.push({ key: 'hasVerifiedClaims', label: 'Has verified claims' });
  }

  if (activeFilters.length === 0) return null;

  return (
    <div className="flex items-center gap-2 flex-wrap py-2">
      <span className="text-xs text-gray-500 dark:text-gray-400">Active:</span>
      {activeFilters.map(({ key, label }) => (
        <span
          key={key}
          className="inline-flex items-center gap-1 px-2 py-1 bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300 rounded text-xs"
        >
          {label}
          <button
            type="button"
            onClick={() => onRemove(key)}
            className="hover:text-blue-900 dark:hover:text-blue-100"
          >
            ×
          </button>
        </span>
      ))}
      <button
        type="button"
        onClick={onClear}
        className="text-xs text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-300"
      >
        Clear all
      </button>
    </div>
  );
}

// Main component
export default function EpistemicSearch({
  onFiltersChange,
  currentFilters = {},
  compact = false,
  className = '',
}: EpistemicSearchProps) {
  const [isExpanded, setIsExpanded] = useState(!compact);
  const [filters, setFilters] = useState<EpistemicFilters>(currentFilters);

  // Update filters and notify parent
  const updateFilters = useCallback((updates: Partial<EpistemicFilters>) => {
    const newFilters = { ...filters, ...updates };
    setFilters(newFilters);
    onFiltersChange(newFilters);
  }, [filters, onFiltersChange]);

  // Toggle tier selection
  const toggleTier = useCallback((tier: number) => {
    const currentTiers = filters.tiers || [];
    const newTiers = currentTiers.includes(tier)
      ? currentTiers.filter(t => t !== tier)
      : [...currentTiers, tier];
    updateFilters({ tiers: newTiers.length > 0 ? newTiers : undefined });
  }, [filters.tiers, updateFilters]);

  // Toggle assurance level selection
  const toggleAssuranceLevel = useCallback((level: AssuranceLevel) => {
    const currentLevels = filters.assuranceLevels || [];
    const newLevels = currentLevels.includes(level)
      ? currentLevels.filter(l => l !== level)
      : [...currentLevels, level];
    updateFilters({ assuranceLevels: newLevels.length > 0 ? newLevels : undefined });
  }, [filters.assuranceLevels, updateFilters]);

  // Toggle intent selection
  const toggleIntent = useCallback((intent: string) => {
    const currentIntents = filters.intents || [];
    const newIntents = currentIntents.includes(intent)
      ? currentIntents.filter(i => i !== intent)
      : [...currentIntents, intent];
    updateFilters({ intents: newIntents.length > 0 ? newIntents : undefined });
  }, [filters.intents, updateFilters]);

  // Remove single filter
  const removeFilter = useCallback((key: keyof EpistemicFilters) => {
    const newFilters = { ...filters };
    delete newFilters[key];
    setFilters(newFilters);
    onFiltersChange(newFilters);
  }, [filters, onFiltersChange]);

  // Clear all filters
  const clearFilters = useCallback(() => {
    setFilters({});
    onFiltersChange({});
  }, [onFiltersChange]);

  // Count active filters
  const activeFilterCount = useMemo(() => {
    let count = 0;
    if (filters.minTrustScore && filters.minTrustScore > 0) count++;
    if (filters.maxPathLength) count++;
    if (filters.requireTrustPath) count++;
    if (filters.tiers && filters.tiers.length > 0) count++;
    if (filters.assuranceLevels && filters.assuranceLevels.length > 0) count++;
    if (filters.intents && filters.intents.length > 0) count++;
    if (filters.hasVerifiedClaims) count++;
    return count;
  }, [filters]);

  // Compact toggle header
  if (compact && !isExpanded) {
    return (
      <div className={`${className}`}>
        <button
          type="button"
          onClick={() => setIsExpanded(true)}
          className="flex items-center gap-2 px-3 py-2 text-sm text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-800 rounded-lg transition-colors"
        >
          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 4a1 1 0 011-1h16a1 1 0 011 1v2.586a1 1 0 01-.293.707l-6.414 6.414a1 1 0 00-.293.707V17l-4 4v-6.586a1 1 0 00-.293-.707L3.293 7.293A1 1 0 013 6.586V4z" />
          </svg>
          <span>Epistemic Filters</span>
          {activeFilterCount > 0 && (
            <span className="px-1.5 py-0.5 bg-blue-100 dark:bg-blue-900/40 text-blue-600 dark:text-blue-400 rounded text-xs">
              {activeFilterCount}
            </span>
          )}
        </button>
        <ActiveFiltersBar filters={filters} onRemove={removeFilter} onClear={clearFilters} />
      </div>
    );
  }

  return (
    <div className={`bg-white dark:bg-gray-900 rounded-lg border border-gray-200 dark:border-gray-700 ${className}`}>
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-gray-200 dark:border-gray-700">
        <div className="flex items-center gap-2">
          <svg className="w-5 h-5 text-blue-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 4a1 1 0 011-1h16a1 1 0 011 1v2.586a1 1 0 01-.293.707l-6.414 6.414a1 1 0 00-.293.707V17l-4 4v-6.586a1 1 0 00-.293-.707L3.293 7.293A1 1 0 013 6.586V4z" />
          </svg>
          <h3 className="text-sm font-semibold text-gray-900 dark:text-gray-100">
            Epistemic Filters
          </h3>
          {activeFilterCount > 0 && (
            <span className="px-1.5 py-0.5 bg-blue-100 dark:bg-blue-900/40 text-blue-600 dark:text-blue-400 rounded text-xs">
              {activeFilterCount} active
            </span>
          )}
        </div>
        {compact && (
          <button
            type="button"
            onClick={() => setIsExpanded(false)}
            className="text-gray-400 hover:text-gray-600 dark:hover:text-gray-300"
          >
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 15l7-7 7 7" />
            </svg>
          </button>
        )}
      </div>

      <div className="p-4 space-y-6">
        {/* Active filters bar */}
        <ActiveFiltersBar filters={filters} onRemove={removeFilter} onClear={clearFilters} />

        {/* Epistemic Tiers */}
        <div>
          <h4 className="text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider mb-3">
            Epistemic Tier
          </h4>
          <div className="flex flex-wrap gap-2">
            {Object.entries(tierConfig).map(([tier, config]) => (
              <FilterChip
                key={tier}
                label={`T${tier}`}
                isActive={filters.tiers?.includes(parseInt(tier)) || false}
                onClick={() => toggleTier(parseInt(tier))}
                color={config.color.replace('text-', '').split('-')[0]}
              />
            ))}
          </div>
        </div>

        {/* Trust Score */}
        <div>
          <h4 className="text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider mb-3">
            Trust Requirements
          </h4>
          <div className="space-y-4">
            <TrustScoreSlider
              value={filters.minTrustScore || 0}
              onChange={(value) => updateFilters({ minTrustScore: value > 0 ? value : undefined })}
            />
            <PathLengthSelector
              value={filters.maxPathLength}
              onChange={(value) => updateFilters({ maxPathLength: value })}
            />
            <label className="flex items-center gap-2 cursor-pointer">
              <input
                type="checkbox"
                checked={filters.requireTrustPath || false}
                onChange={(e) => updateFilters({ requireTrustPath: e.target.checked || undefined })}
                className="w-4 h-4 rounded border-gray-300 text-blue-600 focus:ring-blue-500"
              />
              <span className="text-sm text-gray-700 dark:text-gray-300">
                Only show contacts in my trust network
              </span>
            </label>
          </div>
        </div>

        {/* Assurance Levels */}
        <div>
          <h4 className="text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider mb-3">
            Assurance Level
          </h4>
          <div className="flex flex-wrap gap-2">
            {(Object.keys(assuranceConfig) as AssuranceLevel[]).map((level) => {
              const config = assuranceConfig[level];
              return (
                <FilterChip
                  key={level}
                  label={config.shortLabel}
                  isActive={filters.assuranceLevels?.includes(level) || false}
                  onClick={() => toggleAssuranceLevel(level)}
                  color={config.color.replace('text-', '').split('-')[0]}
                />
              );
            })}
          </div>
        </div>

        {/* Intent Categories */}
        <div>
          <h4 className="text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider mb-3">
            AI-Detected Intent
          </h4>
          <div className="flex flex-wrap gap-2">
            {intentCategories.map((intent) => (
              <FilterChip
                key={intent.id}
                label={`${intent.icon} ${intent.label}`}
                isActive={filters.intents?.includes(intent.id) || false}
                onClick={() => toggleIntent(intent.id)}
                color={intent.color}
              />
            ))}
          </div>
        </div>

        {/* Verification Status */}
        <div>
          <h4 className="text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider mb-3">
            Verification
          </h4>
          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={filters.hasVerifiedClaims || false}
              onChange={(e) => updateFilters({ hasVerifiedClaims: e.target.checked || undefined })}
              className="w-4 h-4 rounded border-gray-300 text-blue-600 focus:ring-blue-500"
            />
            <span className="text-sm text-gray-700 dark:text-gray-300">
              Only show emails with verified claims
            </span>
          </label>
        </div>

        {/* Preset Filters */}
        <div className="pt-4 border-t border-gray-200 dark:border-gray-700">
          <h4 className="text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider mb-3">
            Quick Presets
          </h4>
          <div className="grid grid-cols-2 gap-2">
            <button
              type="button"
              onClick={() => {
                setFilters({
                  requireTrustPath: true,
                  minTrustScore: 0.5,
                });
                onFiltersChange({
                  requireTrustPath: true,
                  minTrustScore: 0.5,
                });
              }}
              className="px-3 py-2 text-xs font-medium text-left bg-emerald-50 dark:bg-emerald-900/20 text-emerald-700 dark:text-emerald-300 rounded-lg hover:bg-emerald-100 dark:hover:bg-emerald-900/30 transition-colors"
            >
              <span className="block font-semibold">Trusted Only</span>
              <span className="text-emerald-600 dark:text-emerald-400">From my trust network</span>
            </button>
            <button
              type="button"
              onClick={() => {
                setFilters({
                  tiers: [3, 4],
                  hasVerifiedClaims: true,
                });
                onFiltersChange({
                  tiers: [3, 4],
                  hasVerifiedClaims: true,
                });
              }}
              className="px-3 py-2 text-xs font-medium text-left bg-purple-50 dark:bg-purple-900/20 text-purple-700 dark:text-purple-300 rounded-lg hover:bg-purple-100 dark:hover:bg-purple-900/30 transition-colors"
            >
              <span className="block font-semibold">High Assurance</span>
              <span className="text-purple-600 dark:text-purple-400">Verified identity only</span>
            </button>
            <button
              type="button"
              onClick={() => {
                setFilters({
                  intents: ['urgent_request', 'action_required'],
                });
                onFiltersChange({
                  intents: ['urgent_request', 'action_required'],
                });
              }}
              className="px-3 py-2 text-xs font-medium text-left bg-red-50 dark:bg-red-900/20 text-red-700 dark:text-red-300 rounded-lg hover:bg-red-100 dark:hover:bg-red-900/30 transition-colors"
            >
              <span className="block font-semibold">Needs Action</span>
              <span className="text-red-600 dark:text-red-400">Urgent & action items</span>
            </button>
            <button
              type="button"
              onClick={() => {
                setFilters({
                  tiers: [0, 1],
                  requireTrustPath: false,
                });
                onFiltersChange({
                  tiers: [0, 1],
                  requireTrustPath: false,
                });
              }}
              className="px-3 py-2 text-xs font-medium text-left bg-amber-50 dark:bg-amber-900/20 text-amber-700 dark:text-amber-300 rounded-lg hover:bg-amber-100 dark:hover:bg-amber-900/30 transition-colors"
            >
              <span className="block font-semibold">Review Needed</span>
              <span className="text-amber-600 dark:text-amber-400">Low trust / unverified</span>
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

// Export types and sub-components
export { FilterChip, TrustScoreSlider, PathLengthSelector, ActiveFiltersBar };
export type { EpistemicFilters };
