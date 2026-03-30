// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Advanced Search Component
 *
 * Full-text search with operators and saved searches
 */

import React, { useState, useEffect, useRef } from 'react';

interface SearchResult {
  id: string;
  subject: string;
  fromAddress: string;
  fromName?: string;
  bodyText: string;
  receivedAt: string;
  folder: string;
  isRead: boolean;
  hasAttachments: boolean;
}

interface SavedSearch {
  id: string;
  name: string;
  query: string;
  useCount: number;
}

interface AdvancedSearchProps {
  onSelect: (emailId: string) => void;
  onClose?: () => void;
}

export default function AdvancedSearch({ onSelect, onClose }: AdvancedSearchProps) {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<SearchResult[]>([]);
  const [loading, setLoading] = useState(false);
  const [savedSearches, setSavedSearches] = useState<SavedSearch[]>([]);
  const [searchHistory, setSearchHistory] = useState<string[]>([]);
  const [suggestions, setSuggestions] = useState<string[]>([]);
  const [showSuggestions, setShowSuggestions] = useState(false);
  const [showSaveModal, setShowSaveModal] = useState(false);
  const [sortBy, setSortBy] = useState<'relevance' | 'date' | 'sender'>('relevance');
  const inputRef = useRef<HTMLInputElement>(null);
  const debounceRef = useRef<NodeJS.Timeout>();

  useEffect(() => {
    fetchSavedSearches();
    fetchSearchHistory();
  }, []);

  useEffect(() => {
    if (query.length >= 2) {
      fetchSuggestions(query);
    } else {
      setSuggestions([]);
    }
  }, [query]);

  async function fetchSavedSearches() {
    try {
      const response = await fetch('/api/search/saved');
      if (response.ok) {
        setSavedSearches(await response.json());
      }
    } catch (error) {
      console.error('Failed to fetch saved searches:', error);
    }
  }

  async function fetchSearchHistory() {
    try {
      const response = await fetch('/api/search/history?limit=10');
      if (response.ok) {
        const history = await response.json();
        setSearchHistory(history.map((h: any) => h.query));
      }
    } catch (error) {
      console.error('Failed to fetch search history:', error);
    }
  }

  async function fetchSuggestions(prefix: string) {
    try {
      const response = await fetch(`/api/search/suggestions?prefix=${encodeURIComponent(prefix)}`);
      if (response.ok) {
        setSuggestions(await response.json());
      }
    } catch (error) {
      console.error('Failed to fetch suggestions:', error);
    }
  }

  async function search(searchQuery: string) {
    if (!searchQuery.trim()) {
      setResults([]);
      return;
    }

    setLoading(true);
    try {
      const response = await fetch('/api/search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: searchQuery,
          sortBy,
          limit: 50,
          offset: 0,
        }),
      });

      if (response.ok) {
        const data = await response.json();
        setResults(data.results);
      }
    } catch (error) {
      console.error('Search failed:', error);
    } finally {
      setLoading(false);
    }
  }

  function handleInputChange(e: React.ChangeEvent<HTMLInputElement>) {
    const value = e.target.value;
    setQuery(value);
    setShowSuggestions(true);

    // Debounce search
    if (debounceRef.current) {
      clearTimeout(debounceRef.current);
    }
    debounceRef.current = setTimeout(() => {
      search(value);
    }, 300);
  }

  function handleKeyDown(e: React.KeyboardEvent) {
    if (e.key === 'Enter') {
      setShowSuggestions(false);
      search(query);
    } else if (e.key === 'Escape') {
      setShowSuggestions(false);
      onClose?.();
    }
  }

  function applySuggestion(suggestion: string) {
    setQuery(suggestion);
    setShowSuggestions(false);
    search(suggestion);
    inputRef.current?.focus();
  }

  async function saveSearch(name: string) {
    try {
      await fetch('/api/search/saved', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name, query }),
      });
      fetchSavedSearches();
      setShowSaveModal(false);
    } catch (error) {
      console.error('Failed to save search:', error);
    }
  }

  async function deleteSavedSearch(id: string) {
    try {
      await fetch(`/api/search/saved/${id}`, { method: 'DELETE' });
      fetchSavedSearches();
    } catch (error) {
      console.error('Failed to delete saved search:', error);
    }
  }

  function insertOperator(operator: string) {
    const cursorPos = inputRef.current?.selectionStart || query.length;
    const newQuery = query.slice(0, cursorPos) + operator + query.slice(cursorPos);
    setQuery(newQuery);
    inputRef.current?.focus();
  }

  const operators = [
    { label: 'from:', example: 'from:alice@example.com' },
    { label: 'to:', example: 'to:bob@example.com' },
    { label: 'subject:', example: 'subject:"meeting notes"' },
    { label: 'has:attachment', example: 'has:attachment' },
    { label: 'is:unread', example: 'is:unread' },
    { label: 'is:starred', example: 'is:starred' },
    { label: 'in:', example: 'in:inbox' },
    { label: 'label:', example: 'label:work' },
    { label: 'after:', example: 'after:2024-01-01' },
    { label: 'before:', example: 'before:2024-12-31' },
  ];

  return (
    <div className="h-full flex flex-col">
      {/* Search Bar */}
      <div className="border-b border-border p-4">
        <div className="relative">
          <input
            ref={inputRef}
            type="text"
            value={query}
            onChange={handleInputChange}
            onKeyDown={handleKeyDown}
            onFocus={() => setShowSuggestions(true)}
            placeholder="Search emails... (try: from:alice has:attachment)"
            className="w-full px-4 py-3 pr-24 border border-border rounded-lg text-lg"
            autoFocus
          />
          <div className="absolute right-2 top-1/2 -translate-y-1/2 flex items-center gap-2">
            {query && (
              <button
                onClick={() => setShowSaveModal(true)}
                className="px-2 py-1 text-sm text-primary hover:underline"
              >
                Save
              </button>
            )}
            <button
              onClick={() => search(query)}
              className="px-3 py-1.5 bg-primary text-white rounded text-sm"
            >
              Search
            </button>
          </div>

          {/* Suggestions Dropdown */}
          {showSuggestions && (suggestions.length > 0 || searchHistory.length > 0) && (
            <div className="absolute top-full left-0 right-0 mt-1 bg-background border border-border rounded-lg shadow-lg z-10 max-h-64 overflow-y-auto">
              {suggestions.length > 0 && (
                <div>
                  <div className="px-3 py-2 text-xs font-semibold text-muted bg-muted/10">
                    Suggestions
                  </div>
                  {suggestions.map((s, i) => (
                    <div
                      key={i}
                      className="px-3 py-2 hover:bg-muted/30 cursor-pointer"
                      onClick={() => applySuggestion(s)}
                    >
                      {s}
                    </div>
                  ))}
                </div>
              )}
              {searchHistory.length > 0 && (
                <div>
                  <div className="px-3 py-2 text-xs font-semibold text-muted bg-muted/10">
                    Recent Searches
                  </div>
                  {searchHistory.map((h, i) => (
                    <div
                      key={i}
                      className="px-3 py-2 hover:bg-muted/30 cursor-pointer flex items-center gap-2"
                      onClick={() => applySuggestion(h)}
                    >
                      <span className="text-muted">History</span>
                      {h}
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>

        {/* Operator Chips */}
        <div className="flex flex-wrap gap-2 mt-3">
          {operators.map((op) => (
            <button
              key={op.label}
              onClick={() => insertOperator(op.label)}
              className="px-2 py-1 text-xs bg-muted/30 hover:bg-muted/50 rounded"
              title={op.example}
            >
              {op.label}
            </button>
          ))}
        </div>

        {/* Sort Options */}
        <div className="flex items-center gap-4 mt-3">
          <span className="text-sm text-muted">Sort by:</span>
          {(['relevance', 'date', 'sender'] as const).map((sort) => (
            <button
              key={sort}
              onClick={() => {
                setSortBy(sort);
                if (query) search(query);
              }}
              className={`text-sm px-2 py-1 rounded ${
                sortBy === sort ? 'bg-primary text-white' : 'hover:bg-muted/30'
              }`}
            >
              {sort.charAt(0).toUpperCase() + sort.slice(1)}
            </button>
          ))}
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 flex overflow-hidden">
        {/* Saved Searches Sidebar */}
        <div className="w-64 border-r border-border p-4 overflow-y-auto">
          <h3 className="font-semibold mb-3">Saved Searches</h3>
          {savedSearches.length === 0 ? (
            <p className="text-sm text-muted">No saved searches yet</p>
          ) : (
            <div className="space-y-2">
              {savedSearches.map((saved) => (
                <div
                  key={saved.id}
                  className="flex items-center justify-between p-2 hover:bg-muted/30 rounded cursor-pointer group"
                  onClick={() => {
                    setQuery(saved.query);
                    search(saved.query);
                  }}
                >
                  <div className="min-w-0">
                    <p className="font-medium truncate">{saved.name}</p>
                    <p className="text-xs text-muted truncate">{saved.query}</p>
                  </div>
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      deleteSavedSearch(saved.id);
                    }}
                    className="opacity-0 group-hover:opacity-100 text-red-500 text-sm"
                  >
                    X
                  </button>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Search Results */}
        <div className="flex-1 overflow-y-auto">
          {loading ? (
            <div className="flex items-center justify-center h-32">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary"></div>
            </div>
          ) : results.length === 0 ? (
            <div className="text-center py-12 text-muted">
              {query ? 'No results found' : 'Enter a search query to find emails'}
            </div>
          ) : (
            <div className="divide-y divide-border">
              <div className="px-4 py-2 bg-muted/10 text-sm text-muted">
                {results.length} results
              </div>
              {results.map((result) => (
                <div
                  key={result.id}
                  className={`p-4 hover:bg-muted/30 cursor-pointer ${
                    !result.isRead ? 'bg-blue-50' : ''
                  }`}
                  onClick={() => onSelect(result.id)}
                >
                  <div className="flex items-center justify-between mb-1">
                    <span className={`font-medium ${!result.isRead ? 'text-foreground' : 'text-muted'}`}>
                      {result.fromName || result.fromAddress}
                    </span>
                    <div className="flex items-center gap-2">
                      {result.hasAttachments && <span>Attach</span>}
                      <span className="text-xs text-muted">
                        {new Date(result.receivedAt).toLocaleDateString()}
                      </span>
                    </div>
                  </div>
                  <p className={`${!result.isRead ? 'font-medium' : ''}`}>{result.subject}</p>
                  <p className="text-sm text-muted line-clamp-2 mt-1">
                    {result.bodyText.slice(0, 200)}...
                  </p>
                  <div className="mt-2">
                    <span className="px-2 py-0.5 bg-muted/30 rounded text-xs">{result.folder}</span>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Save Search Modal */}
      {showSaveModal && (
        <SaveSearchModal
          query={query}
          onSave={saveSearch}
          onClose={() => setShowSaveModal(false)}
        />
      )}
    </div>
  );
}

function SaveSearchModal({
  query,
  onSave,
  onClose,
}: {
  query: string;
  onSave: (name: string) => void;
  onClose: () => void;
}) {
  const [name, setName] = useState('');

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
      <div className="bg-background rounded-lg shadow-xl w-full max-w-md">
        <div className="p-4 border-b border-border">
          <h2 className="text-lg font-semibold">Save Search</h2>
        </div>
        <div className="p-4 space-y-4">
          <div>
            <label className="block text-sm font-medium mb-1">Name</label>
            <input
              type="text"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="e.g., Emails from Alice with attachments"
              className="w-full px-3 py-2 border border-border rounded"
              autoFocus
            />
          </div>
          <div>
            <label className="block text-sm font-medium mb-1">Query</label>
            <div className="px-3 py-2 bg-muted/20 rounded text-sm font-mono">{query}</div>
          </div>
        </div>
        <div className="p-4 border-t border-border flex justify-end gap-2">
          <button onClick={onClose} className="px-4 py-2 border border-border rounded">
            Cancel
          </button>
          <button
            onClick={() => onSave(name)}
            disabled={!name.trim()}
            className="px-4 py-2 bg-primary text-white rounded disabled:opacity-50"
          >
            Save
          </button>
        </div>
      </div>
    </div>
  );
}

/**
 * Search Bar Component (for header)
 */
export function SearchBar({ onFocus }: { onFocus: () => void }) {
  const [query, setQuery] = useState('');

  return (
    <div className="relative">
      <input
        type="text"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        onFocus={onFocus}
        placeholder="Search emails..."
        className="w-64 px-4 py-2 pl-10 border border-border rounded-lg"
      />
      <span className="absolute left-3 top-1/2 -translate-y-1/2 text-muted">Search</span>
      <kbd className="absolute right-2 top-1/2 -translate-y-1/2 px-1.5 py-0.5 bg-muted/30 rounded text-xs text-muted">
        /
      </kbd>
    </div>
  );
}
