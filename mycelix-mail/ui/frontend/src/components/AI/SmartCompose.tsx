// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Smart Compose & AI Features
 *
 * AI-powered email composition with suggestions, summaries, and smart replies
 */

import React, { useState, useEffect, useRef, useCallback } from 'react';

interface CompletionSuggestion {
  text: string;
  confidence: number;
}

interface SmartReply {
  shortLabel: string;
  fullText: string;
  tone: string;
}

interface EmailSummary {
  tldr: string;
  keyPoints: string[];
  actionItems: ActionItem[];
  sentiment: Sentiment;
}

interface ActionItem {
  description: string;
  dueDate?: string;
  priority: string;
}

interface Sentiment {
  overall: string;
  urgency: number;
}

interface PriorityScore {
  emailId: string;
  score: number;
  category: string;
  factors: { name: string; value: number }[];
}

// Smart Compose Hook
export function useSmartCompose(onSuggestion: (text: string) => void) {
  const [suggestions, setSuggestions] = useState<CompletionSuggestion[]>([]);
  const [loading, setLoading] = useState(false);
  const [activeSuggestion, setActiveSuggestion] = useState<number>(-1);
  const debounceRef = useRef<NodeJS.Timeout>();

  const fetchSuggestions = useCallback(async (text: string, context: string) => {
    if (text.length < 10) {
      setSuggestions([]);
      return;
    }

    setLoading(true);
    try {
      const response = await fetch('/api/ai/smart-compose', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ context, partial_text: text, max_suggestions: 3 }),
      });

      if (response.ok) {
        const data = await response.json();
        setSuggestions(data);
        setActiveSuggestion(data.length > 0 ? 0 : -1);
      }
    } catch (error) {
      console.error('Smart compose error:', error);
    } finally {
      setLoading(false);
    }
  }, []);

  const debouncedFetch = useCallback((text: string, context: string) => {
    if (debounceRef.current) clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(() => fetchSuggestions(text, context), 500);
  }, [fetchSuggestions]);

  const acceptSuggestion = useCallback((index?: number) => {
    const idx = index ?? activeSuggestion;
    if (idx >= 0 && suggestions[idx]) {
      onSuggestion(suggestions[idx].text);
      setSuggestions([]);
      setActiveSuggestion(-1);
    }
  }, [activeSuggestion, suggestions, onSuggestion]);

  const handleKeyDown = useCallback((e: React.KeyboardEvent) => {
    if (suggestions.length === 0) return false;

    if (e.key === 'Tab' && activeSuggestion >= 0) {
      e.preventDefault();
      acceptSuggestion();
      return true;
    }
    if (e.key === 'ArrowDown') {
      e.preventDefault();
      setActiveSuggestion((i) => Math.min(i + 1, suggestions.length - 1));
      return true;
    }
    if (e.key === 'ArrowUp') {
      e.preventDefault();
      setActiveSuggestion((i) => Math.max(i - 1, 0));
      return true;
    }
    if (e.key === 'Escape') {
      setSuggestions([]);
      setActiveSuggestion(-1);
      return true;
    }
    return false;
  }, [suggestions, activeSuggestion, acceptSuggestion]);

  return {
    suggestions,
    loading,
    activeSuggestion,
    fetchSuggestions: debouncedFetch,
    acceptSuggestion,
    handleKeyDown,
    clearSuggestions: () => { setSuggestions([]); setActiveSuggestion(-1); },
  };
}

// Smart Compose Suggestions Popup
export function SmartComposeSuggestions({
  suggestions,
  activeSuggestion,
  onSelect,
  position,
}: {
  suggestions: CompletionSuggestion[];
  activeSuggestion: number;
  onSelect: (index: number) => void;
  position: { top: number; left: number };
}) {
  if (suggestions.length === 0) return null;

  return (
    <div
      className="absolute z-50 bg-background border border-border rounded-lg shadow-lg py-1 min-w-64"
      style={{ top: position.top, left: position.left }}
    >
      <div className="px-3 py-1 text-xs text-muted border-b border-border">
        Tab to accept, arrows to navigate
      </div>
      {suggestions.map((suggestion, index) => (
        <div
          key={index}
          className={`px-3 py-2 cursor-pointer ${
            index === activeSuggestion ? 'bg-primary/10' : 'hover:bg-muted/30'
          }`}
          onClick={() => onSelect(index)}
        >
          <span className="text-muted">... </span>
          {suggestion.text}
          <span className="ml-2 text-xs text-muted">
            {Math.round(suggestion.confidence * 100)}%
          </span>
        </div>
      ))}
    </div>
  );
}

// Email Summary Component
export function EmailSummaryPanel({ emailId }: { emailId: string }) {
  const [summary, setSummary] = useState<EmailSummary | null>(null);
  const [loading, setLoading] = useState(false);
  const [expanded, setExpanded] = useState(false);

  const fetchSummary = async () => {
    setLoading(true);
    try {
      const response = await fetch(`/api/ai/summarize/${emailId}`);
      if (response.ok) setSummary(await response.json());
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (expanded && !summary) fetchSummary();
  }, [expanded, emailId]);

  const getSentimentColor = (sentiment: string) => {
    switch (sentiment) {
      case 'Positive': return 'text-green-600';
      case 'Negative': return 'text-red-600';
      case 'Mixed': return 'text-yellow-600';
      default: return 'text-gray-600';
    }
  };

  const getPriorityColor = (priority: string) => {
    switch (priority) {
      case 'Urgent': return 'bg-red-100 text-red-800';
      case 'High': return 'bg-orange-100 text-orange-800';
      case 'Medium': return 'bg-yellow-100 text-yellow-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  return (
    <div className="border border-border rounded-lg overflow-hidden">
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full px-4 py-2 flex items-center justify-between bg-muted/20 hover:bg-muted/30"
      >
        <span className="font-medium flex items-center gap-2">
          <span>AI Summary</span>
          {summary?.sentiment && (
            <span className={`text-sm ${getSentimentColor(summary.sentiment.overall)}`}>
              ({summary.sentiment.overall})
            </span>
          )}
        </span>
        <span>{expanded ? '[-]' : '[+]'}</span>
      </button>

      {expanded && (
        <div className="p-4">
          {loading ? (
            <div className="flex items-center justify-center py-4">
              <div className="animate-spin h-5 w-5 border-2 border-primary border-t-transparent rounded-full" />
            </div>
          ) : summary ? (
            <div className="space-y-4">
              <div>
                <h4 className="text-sm font-semibold text-muted mb-1">TL;DR</h4>
                <p>{summary.tldr}</p>
              </div>

              {summary.keyPoints.length > 0 && (
                <div>
                  <h4 className="text-sm font-semibold text-muted mb-1">Key Points</h4>
                  <ul className="list-disc list-inside space-y-1">
                    {summary.keyPoints.map((point, i) => (
                      <li key={i}>{point}</li>
                    ))}
                  </ul>
                </div>
              )}

              {summary.actionItems.length > 0 && (
                <div>
                  <h4 className="text-sm font-semibold text-muted mb-1">Action Items</h4>
                  <div className="space-y-2">
                    {summary.actionItems.map((item, i) => (
                      <div key={i} className="flex items-start gap-2">
                        <input type="checkbox" className="mt-1" />
                        <div>
                          <p>{item.description}</p>
                          <div className="flex gap-2 mt-1">
                            <span className={`px-2 py-0.5 rounded text-xs ${getPriorityColor(item.priority)}`}>
                              {item.priority}
                            </span>
                            {item.dueDate && (
                              <span className="text-xs text-muted">Due: {item.dueDate}</span>
                            )}
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {summary.sentiment.urgency > 0.5 && (
                <div className="bg-red-50 border border-red-200 rounded px-3 py-2 text-sm text-red-800">
                  High urgency detected ({Math.round(summary.sentiment.urgency * 100)}%)
                </div>
              )}
            </div>
          ) : (
            <button onClick={fetchSummary} className="text-primary hover:underline">
              Generate Summary
            </button>
          )}
        </div>
      )}
    </div>
  );
}

// Smart Reply Buttons
export function SmartReplyButtons({
  emailId,
  onSelectReply,
}: {
  emailId: string;
  onSelectReply: (text: string) => void;
}) {
  const [replies, setReplies] = useState<SmartReply[]>([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    fetchReplies();
  }, [emailId]);

  const fetchReplies = async () => {
    setLoading(true);
    try {
      const response = await fetch(`/api/ai/smart-reply/${emailId}`);
      if (response.ok) {
        const data = await response.json();
        setReplies(data.suggestions || []);
      }
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="flex gap-2">
        {[1, 2, 3].map((i) => (
          <div key={i} className="h-8 w-24 bg-muted/30 rounded animate-pulse" />
        ))}
      </div>
    );
  }

  if (replies.length === 0) return null;

  return (
    <div className="flex flex-wrap gap-2">
      {replies.map((reply, i) => (
        <button
          key={i}
          onClick={() => onSelectReply(reply.fullText)}
          className="px-3 py-1.5 bg-primary/10 hover:bg-primary/20 rounded-full text-sm"
          title={reply.fullText}
        >
          {reply.shortLabel}
        </button>
      ))}
    </div>
  );
}

// Priority Inbox View
export function PriorityInbox({ onSelectEmail }: { onSelectEmail: (id: string) => void }) {
  const [priorities, setPriorities] = useState<PriorityScore[]>([]);
  const [loading, setLoading] = useState(true);
  const [emails, setEmails] = useState<Record<string, any>>({});

  useEffect(() => {
    fetchPriorities();
  }, []);

  const fetchPriorities = async () => {
    setLoading(true);
    try {
      const response = await fetch('/api/ai/prioritize-inbox');
      if (response.ok) {
        const data = await response.json();
        setPriorities(data);
        fetchEmailDetails(data.map((p: PriorityScore) => p.emailId));
      }
    } finally {
      setLoading(false);
    }
  };

  const fetchEmailDetails = async (ids: string[]) => {
    const details: Record<string, any> = {};
    for (const id of ids.slice(0, 20)) {
      const response = await fetch(`/api/emails/${id}`);
      if (response.ok) details[id] = await response.json();
    }
    setEmails(details);
  };

  const getCategoryColor = (category: string) => {
    switch (category) {
      case 'Primary': return 'bg-blue-100 text-blue-800';
      case 'Social': return 'bg-purple-100 text-purple-800';
      case 'Promotions': return 'bg-green-100 text-green-800';
      case 'Updates': return 'bg-yellow-100 text-yellow-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  const getScoreColor = (score: number) => {
    if (score >= 0.7) return 'text-red-600';
    if (score >= 0.4) return 'text-yellow-600';
    return 'text-green-600';
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin h-8 w-8 border-2 border-primary border-t-transparent rounded-full" />
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-semibold">Priority Inbox</h2>
        <button onClick={fetchPriorities} className="text-sm text-primary hover:underline">
          Refresh
        </button>
      </div>

      <div className="divide-y divide-border rounded-lg border border-border">
        {priorities.map((priority) => {
          const email = emails[priority.emailId];
          if (!email) return null;

          return (
            <div
              key={priority.emailId}
              className="p-4 hover:bg-muted/30 cursor-pointer"
              onClick={() => onSelectEmail(priority.emailId)}
            >
              <div className="flex items-start justify-between">
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 mb-1">
                    <span className="font-medium truncate">{email.fromName || email.fromAddress}</span>
                    <span className={`px-2 py-0.5 rounded text-xs ${getCategoryColor(priority.category)}`}>
                      {priority.category}
                    </span>
                  </div>
                  <p className="font-medium">{email.subject}</p>
                  <p className="text-sm text-muted truncate">{email.snippet}</p>
                </div>
                <div className="text-right ml-4">
                  <div className={`text-lg font-bold ${getScoreColor(priority.score)}`}>
                    {Math.round(priority.score * 100)}
                  </div>
                  <div className="text-xs text-muted">priority</div>
                </div>
              </div>

              <div className="flex gap-2 mt-2">
                {priority.factors.slice(0, 3).map((factor) => (
                  <span key={factor.name} className="text-xs text-muted">
                    {factor.name}: {Math.round(factor.value * 100)}%
                  </span>
                ))}
              </div>
            </div>
          );
        })}
      </div>

      {priorities.length === 0 && (
        <div className="text-center py-12 text-muted">
          No emails to prioritize
        </div>
      )}
    </div>
  );
}

// AI Settings Panel
export function AiSettingsPanel() {
  const [settings, setSettings] = useState({
    smartComposeEnabled: true,
    smartReplyEnabled: true,
    autoSummarize: false,
    priorityInboxEnabled: true,
    tone: 'professional',
  });

  const handleChange = (key: string, value: any) => {
    setSettings((prev) => ({ ...prev, [key]: value }));
    // Save to backend
    fetch('/api/settings/ai', {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ [key]: value }),
    });
  };

  return (
    <div className="space-y-6">
      <h2 className="text-lg font-semibold">AI Features</h2>

      <div className="space-y-4">
        <label className="flex items-center justify-between">
          <div>
            <p className="font-medium">Smart Compose</p>
            <p className="text-sm text-muted">Get AI-powered writing suggestions as you type</p>
          </div>
          <input
            type="checkbox"
            checked={settings.smartComposeEnabled}
            onChange={(e) => handleChange('smartComposeEnabled', e.target.checked)}
            className="h-5 w-5"
          />
        </label>

        <label className="flex items-center justify-between">
          <div>
            <p className="font-medium">Smart Reply</p>
            <p className="text-sm text-muted">Show suggested quick replies</p>
          </div>
          <input
            type="checkbox"
            checked={settings.smartReplyEnabled}
            onChange={(e) => handleChange('smartReplyEnabled', e.target.checked)}
            className="h-5 w-5"
          />
        </label>

        <label className="flex items-center justify-between">
          <div>
            <p className="font-medium">Auto-Summarize</p>
            <p className="text-sm text-muted">Automatically summarize long emails</p>
          </div>
          <input
            type="checkbox"
            checked={settings.autoSummarize}
            onChange={(e) => handleChange('autoSummarize', e.target.checked)}
            className="h-5 w-5"
          />
        </label>

        <label className="flex items-center justify-between">
          <div>
            <p className="font-medium">Priority Inbox</p>
            <p className="text-sm text-muted">Sort emails by AI-calculated importance</p>
          </div>
          <input
            type="checkbox"
            checked={settings.priorityInboxEnabled}
            onChange={(e) => handleChange('priorityInboxEnabled', e.target.checked)}
            className="h-5 w-5"
          />
        </label>

        <div>
          <p className="font-medium mb-2">Default Writing Tone</p>
          <select
            value={settings.tone}
            onChange={(e) => handleChange('tone', e.target.value)}
            className="w-full px-3 py-2 border border-border rounded"
          >
            <option value="professional">Professional</option>
            <option value="casual">Casual</option>
            <option value="formal">Formal</option>
            <option value="friendly">Friendly</option>
            <option value="concise">Concise</option>
          </select>
        </div>
      </div>
    </div>
  );
}
