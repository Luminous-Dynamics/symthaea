// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * CourseFilters Component
 *
 * Provides search and filter controls for course discovery
 */

import React, { useState } from 'react';

export interface FilterState {
  searchQuery: string;
  difficulty: string;
  tags: string[];
  sortBy: string;
}

interface CourseFiltersProps {
  onFilterChange: (filters: FilterState) => void;
  availableTags?: string[];
}

export const CourseFilters: React.FC<CourseFiltersProps> = ({
  onFilterChange,
  availableTags = [],
}) => {
  const [searchQuery, setSearchQuery] = useState('');
  const [difficulty, setDifficulty] = useState('all');
  const [selectedTags, setSelectedTags] = useState<string[]>([]);
  const [sortBy, setSortBy] = useState('popular');

  const handleFilterChange = (
    newQuery?: string,
    newDifficulty?: string,
    newTags?: string[],
    newSortBy?: string
  ) => {
    const query = newQuery !== undefined ? newQuery : searchQuery;
    const diff = newDifficulty !== undefined ? newDifficulty : difficulty;
    const tags = newTags !== undefined ? newTags : selectedTags;
    const sort = newSortBy !== undefined ? newSortBy : sortBy;

    onFilterChange({
      searchQuery: query,
      difficulty: diff,
      tags,
      sortBy: sort,
    });
  };

  const handleSearchChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const query = e.target.value;
    setSearchQuery(query);
    handleFilterChange(query);
  };

  const handleDifficultyChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const diff = e.target.value;
    setDifficulty(diff);
    handleFilterChange(undefined, diff);
  };

  const handleTagToggle = (tag: string) => {
    const newTags = selectedTags.includes(tag)
      ? selectedTags.filter((t) => t !== tag)
      : [...selectedTags, tag];
    setSelectedTags(newTags);
    handleFilterChange(undefined, undefined, newTags);
  };

  const handleSortChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const sort = e.target.value;
    setSortBy(sort);
    handleFilterChange(undefined, undefined, undefined, sort);
  };

  const handleClearFilters = () => {
    setSearchQuery('');
    setDifficulty('all');
    setSelectedTags([]);
    setSortBy('popular');
    onFilterChange({
      searchQuery: '',
      difficulty: 'all',
      tags: [],
      sortBy: 'popular',
    });
  };

  const hasActiveFilters =
    searchQuery !== '' ||
    difficulty !== 'all' ||
    selectedTags.length > 0 ||
    sortBy !== 'popular';

  return (
    <div style={{ marginBottom: '24px' }}>
      {/* Search and Sort Row */}
      <div style={{ display: 'flex', gap: '12px', marginBottom: '16px' }}>
        {/* Search Input */}
        <div style={{ flex: 1 }}>
          <input
            type="text"
            placeholder="Search courses..."
            value={searchQuery}
            onChange={handleSearchChange}
            style={{
              width: '100%',
              padding: '10px 14px',
              fontSize: '14px',
              border: '1px solid #d1d5db',
              borderRadius: '6px',
              outline: 'none',
            }}
            onFocus={(e) => {
              e.target.style.borderColor = '#3b82f6';
              e.target.style.boxShadow = '0 0 0 3px rgba(59, 130, 246, 0.1)';
            }}
            onBlur={(e) => {
              e.target.style.borderColor = '#d1d5db';
              e.target.style.boxShadow = 'none';
            }}
          />
        </div>

        {/* Difficulty Filter */}
        <select
          value={difficulty}
          onChange={handleDifficultyChange}
          style={{
            padding: '10px 14px',
            fontSize: '14px',
            border: '1px solid #d1d5db',
            borderRadius: '6px',
            backgroundColor: '#ffffff',
            cursor: 'pointer',
            outline: 'none',
          }}
        >
          <option value="all">All Levels</option>
          <option value="beginner">Beginner</option>
          <option value="intermediate">Intermediate</option>
          <option value="advanced">Advanced</option>
        </select>

        {/* Sort By */}
        <select
          value={sortBy}
          onChange={handleSortChange}
          style={{
            padding: '10px 14px',
            fontSize: '14px',
            border: '1px solid #d1d5db',
            borderRadius: '6px',
            backgroundColor: '#ffffff',
            cursor: 'pointer',
            outline: 'none',
          }}
        >
          <option value="popular">Most Popular</option>
          <option value="recent">Most Recent</option>
          <option value="title">Title (A-Z)</option>
        </select>
      </div>

      {/* Tags Row */}
      {availableTags.length > 0 && (
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px', marginBottom: '12px' }}>
          {availableTags.slice(0, 10).map((tag) => {
            const isSelected = selectedTags.includes(tag);
            return (
              <button
                key={tag}
                onClick={() => handleTagToggle(tag)}
                style={{
                  padding: '6px 12px',
                  fontSize: '13px',
                  border: '1px solid',
                  borderColor: isSelected ? '#3b82f6' : '#d1d5db',
                  borderRadius: '6px',
                  backgroundColor: isSelected ? '#eff6ff' : '#ffffff',
                  color: isSelected ? '#3b82f6' : '#4b5563',
                  cursor: 'pointer',
                  transition: 'all 0.2s ease',
                }}
                onMouseEnter={(e) => {
                  if (!isSelected) {
                    e.currentTarget.style.borderColor = '#9ca3af';
                  }
                }}
                onMouseLeave={(e) => {
                  if (!isSelected) {
                    e.currentTarget.style.borderColor = '#d1d5db';
                  }
                }}
              >
                {tag}
              </button>
            );
          })}
        </div>
      )}

      {/* Clear Filters Button */}
      {hasActiveFilters && (
        <button
          onClick={handleClearFilters}
          style={{
            padding: '6px 12px',
            fontSize: '13px',
            border: '1px solid #d1d5db',
            borderRadius: '6px',
            backgroundColor: '#ffffff',
            color: '#6b7280',
            cursor: 'pointer',
          }}
          onMouseEnter={(e) => {
            e.currentTarget.style.backgroundColor = '#f9fafb';
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.backgroundColor = '#ffffff';
          }}
        >
          Clear Filters
        </button>
      )}
    </div>
  );
};
