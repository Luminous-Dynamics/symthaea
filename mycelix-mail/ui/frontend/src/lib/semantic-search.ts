// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Semantic Search Service
 *
 * Provides:
 * - Natural language email search
 * - Vector embeddings for semantic similarity
 * - Auto-categorization with ML
 * - Meeting/event extraction
 * - Relationship insights and analytics
 */

import { create } from 'zustand';

// ============================================================================
// Types
// ============================================================================

export interface SearchQuery {
  text: string;
  filters?: SearchFilters;
  options?: SearchOptions;
}

export interface SearchFilters {
  from?: string[];
  to?: string[];
  dateRange?: { start: Date; end: Date };
  folders?: string[];
  hasAttachments?: boolean;
  isUnread?: boolean;
  trustScoreMin?: number;
  labels?: string[];
  excludeLabels?: string[];
}

export interface SearchOptions {
  limit?: number;
  offset?: number;
  sortBy?: 'relevance' | 'date' | 'trust';
  includeThreads?: boolean;
  semanticWeight?: number; // 0-1, how much to weight semantic vs keyword
}

export interface SearchResult {
  id: string;
  emailId: string;
  score: number;
  matchType: 'semantic' | 'keyword' | 'hybrid';
  highlights: SearchHighlight[];
  email: SearchResultEmail;
}

export interface SearchHighlight {
  field: 'subject' | 'body' | 'from' | 'to';
  text: string;
  matches: { start: number; end: number }[];
}

export interface SearchResultEmail {
  id: string;
  threadId: string;
  subject: string;
  snippet: string;
  from: { name: string; email: string };
  date: Date;
  isRead: boolean;
  trustScore?: number;
  labels: string[];
}

export interface EmbeddingVector {
  emailId: string;
  vector: number[];
  createdAt: Date;
  model: string;
}

export interface EmailCategory {
  id: string;
  name: string;
  description: string;
  color: string;
  icon: string;
  rules?: CategoryRule[];
  isSystem: boolean;
  emailCount: number;
}

export interface CategoryRule {
  field: 'from' | 'to' | 'subject' | 'body' | 'domain';
  operator: 'contains' | 'equals' | 'regex' | 'semantic_similarity';
  value: string;
  threshold?: number;
}

export interface ExtractedMeeting {
  id: string;
  emailId: string;
  title: string;
  dateTime?: Date;
  duration?: number;
  location?: string;
  attendees: string[];
  description?: string;
  confidence: number;
  status: 'detected' | 'confirmed' | 'declined' | 'added_to_calendar';
}

export interface RelationshipInsight {
  contactEmail: string;
  contactName?: string;
  metrics: {
    totalEmails: number;
    sentByMe: number;
    sentToMe: number;
    avgResponseTime: number; // hours
    lastContact: Date;
    firstContact: Date;
    communicationFrequency: 'daily' | 'weekly' | 'monthly' | 'rare';
  };
  patterns: {
    preferredTimes: { hour: number; count: number }[];
    topTopics: { topic: string; count: number }[];
    sentimentTrend: { month: string; sentiment: number }[];
  };
  trustEvolution: { date: Date; score: number }[];
}

export interface NaturalLanguageQuery {
  original: string;
  parsed: {
    intent: 'search' | 'filter' | 'action' | 'question';
    entities: {
      type: 'person' | 'date' | 'topic' | 'folder' | 'attachment';
      value: string;
      confidence: number;
    }[];
    timeRange?: { start: Date; end: Date };
    sortIntent?: 'newest' | 'oldest' | 'relevant';
  };
  structuredQuery: SearchQuery;
}

// ============================================================================
// Embedding Service
// ============================================================================

class EmbeddingService {
  private modelEndpoint: string;
  private cache: Map<string, number[]> = new Map();

  constructor(endpoint: string = '/api/embeddings') {
    this.modelEndpoint = endpoint;
  }

  async getEmbedding(text: string): Promise<number[]> {
    const cacheKey = this.hashText(text);
    if (this.cache.has(cacheKey)) {
      return this.cache.get(cacheKey)!;
    }

    // In production, this would call the embedding API
    // For now, generate mock embedding
    const embedding = this.mockEmbedding(text);
    this.cache.set(cacheKey, embedding);
    return embedding;
  }

  async getBatchEmbeddings(texts: string[]): Promise<number[][]> {
    return Promise.all(texts.map((t) => this.getEmbedding(t)));
  }

  cosineSimilarity(a: number[], b: number[]): number {
    if (a.length !== b.length) return 0;

    let dotProduct = 0;
    let normA = 0;
    let normB = 0;

    for (let i = 0; i < a.length; i++) {
      dotProduct += a[i] * b[i];
      normA += a[i] * a[i];
      normB += b[i] * b[i];
    }

    return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
  }

  private hashText(text: string): string {
    let hash = 0;
    for (let i = 0; i < text.length; i++) {
      const char = text.charCodeAt(i);
      hash = (hash << 5) - hash + char;
      hash = hash & hash;
    }
    return hash.toString(36);
  }

  private mockEmbedding(text: string): number[] {
    // Generate deterministic mock embedding based on text
    const dimension = 384;
    const embedding: number[] = [];
    const seed = this.hashText(text);

    for (let i = 0; i < dimension; i++) {
      const value = Math.sin(parseInt(seed, 36) * (i + 1)) * 0.5;
      embedding.push(value);
    }

    // Normalize
    const norm = Math.sqrt(embedding.reduce((sum, v) => sum + v * v, 0));
    return embedding.map((v) => v / norm);
  }
}

// ============================================================================
// Natural Language Parser
// ============================================================================

class NLQueryParser {
  private datePatterns = [
    { regex: /last\s+week/i, getValue: () => this.getRelativeDate(-7) },
    { regex: /last\s+month/i, getValue: () => this.getRelativeDate(-30) },
    { regex: /yesterday/i, getValue: () => this.getRelativeDate(-1) },
    { regex: /today/i, getValue: () => this.getRelativeDate(0) },
    { regex: /this\s+week/i, getValue: () => this.getThisWeek() },
    { regex: /this\s+month/i, getValue: () => this.getThisMonth() },
    {
      regex: /(\d{1,2})\/(\d{1,2})\/(\d{4})/,
      getValue: (m: RegExpMatchArray) => ({
        start: new Date(parseInt(m[3]), parseInt(m[1]) - 1, parseInt(m[2])),
        end: new Date(parseInt(m[3]), parseInt(m[1]) - 1, parseInt(m[2]), 23, 59, 59),
      }),
    },
  ];

  private personPatterns = [
    { regex: /from\s+([^\s,]+@[^\s,]+)/i, field: 'from' as const },
    { regex: /from\s+["']([^"']+)["']/i, field: 'from' as const },
    { regex: /to\s+([^\s,]+@[^\s,]+)/i, field: 'to' as const },
    { regex: /to\s+["']([^"']+)["']/i, field: 'to' as const },
  ];

  parse(query: string): NaturalLanguageQuery {
    const entities: NaturalLanguageQuery['parsed']['entities'] = [];
    let timeRange: { start: Date; end: Date } | undefined;
    let remainingQuery = query;

    // Extract dates
    for (const pattern of this.datePatterns) {
      const match = query.match(pattern.regex);
      if (match) {
        timeRange = pattern.getValue(match);
        remainingQuery = remainingQuery.replace(pattern.regex, '').trim();
        entities.push({
          type: 'date',
          value: match[0],
          confidence: 0.9,
        });
      }
    }

    // Extract people
    const fromFilters: string[] = [];
    const toFilters: string[] = [];

    for (const pattern of this.personPatterns) {
      const match = query.match(pattern.regex);
      if (match) {
        const value = match[1];
        if (pattern.field === 'from') fromFilters.push(value);
        else toFilters.push(value);
        remainingQuery = remainingQuery.replace(pattern.regex, '').trim();
        entities.push({
          type: 'person',
          value,
          confidence: 0.95,
        });
      }
    }

    // Extract attachment mentions
    if (/with\s+attachment|has\s+attachment|attached/i.test(query)) {
      entities.push({
        type: 'attachment',
        value: 'true',
        confidence: 0.9,
      });
      remainingQuery = remainingQuery
        .replace(/with\s+attachment|has\s+attachment|attached/gi, '')
        .trim();
    }

    // Detect intent
    let intent: NaturalLanguageQuery['parsed']['intent'] = 'search';
    if (/^(find|search|show|get)\b/i.test(query)) intent = 'search';
    else if (/^(filter|only)\b/i.test(query)) intent = 'filter';
    else if (/\?$/.test(query)) intent = 'question';

    // Detect sort intent
    let sortIntent: 'newest' | 'oldest' | 'relevant' | undefined;
    if (/newest|recent|latest/i.test(query)) sortIntent = 'newest';
    else if (/oldest|first/i.test(query)) sortIntent = 'oldest';

    // Build structured query
    const structuredQuery: SearchQuery = {
      text: remainingQuery.replace(/^(find|search|show|get|filter|only)\s+/i, '').trim(),
      filters: {
        ...(fromFilters.length > 0 && { from: fromFilters }),
        ...(toFilters.length > 0 && { to: toFilters }),
        ...(timeRange && { dateRange: timeRange }),
        ...(entities.some((e) => e.type === 'attachment') && { hasAttachments: true }),
      },
      options: {
        sortBy: sortIntent === 'newest' || sortIntent === 'oldest' ? 'date' : 'relevance',
      },
    };

    return {
      original: query,
      parsed: {
        intent,
        entities,
        timeRange,
        sortIntent,
      },
      structuredQuery,
    };
  }

  private getRelativeDate(daysAgo: number): { start: Date; end: Date } {
    const date = new Date();
    date.setDate(date.getDate() + daysAgo);
    date.setHours(0, 0, 0, 0);
    const end = new Date(date);
    end.setHours(23, 59, 59, 999);
    return { start: date, end };
  }

  private getThisWeek(): { start: Date; end: Date } {
    const now = new Date();
    const start = new Date(now);
    start.setDate(now.getDate() - now.getDay());
    start.setHours(0, 0, 0, 0);
    return { start, end: now };
  }

  private getThisMonth(): { start: Date; end: Date } {
    const now = new Date();
    const start = new Date(now.getFullYear(), now.getMonth(), 1);
    return { start, end: now };
  }
}

// ============================================================================
// Meeting Extractor
// ============================================================================

class MeetingExtractor {
  private timePatterns = [
    /(\d{1,2}):(\d{2})\s*(am|pm)?/gi,
    /(\d{1,2})\s*(am|pm)/gi,
    /at\s+(\d{1,2}):?(\d{2})?\s*(am|pm)?/gi,
  ];

  private datePatterns = [
    /on\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)/gi,
    /on\s+(\d{1,2}\/\d{1,2})/gi,
    /(tomorrow|next\s+week|this\s+friday)/gi,
  ];

  private durationPatterns = [
    /for\s+(\d+)\s*(hour|hr|minute|min)/gi,
    /(\d+)\s*(hour|hr|minute|min)\s*meeting/gi,
  ];

  extract(emailBody: string, emailSubject: string): Partial<ExtractedMeeting> | null {
    const text = `${emailSubject} ${emailBody}`.toLowerCase();

    // Check if this looks like a meeting email
    const meetingKeywords = [
      'meeting',
      'call',
      'sync',
      'discussion',
      'appointment',
      'schedule',
      'calendar',
      'invite',
      'join',
      'zoom',
      'teams',
      'meet',
    ];

    const hasMeetingKeyword = meetingKeywords.some((kw) => text.includes(kw));
    if (!hasMeetingKeyword) return null;

    const meeting: Partial<ExtractedMeeting> = {
      confidence: 0.5,
    };

    // Extract title - often in subject
    if (emailSubject) {
      meeting.title = emailSubject
        .replace(/^(re:|fwd:|fw:)\s*/gi, '')
        .replace(/invitation:/gi, '')
        .trim();
      meeting.confidence += 0.1;
    }

    // Extract time
    for (const pattern of this.timePatterns) {
      const match = text.match(pattern);
      if (match) {
        meeting.confidence += 0.15;
        break;
      }
    }

    // Extract date
    for (const pattern of this.datePatterns) {
      const match = text.match(pattern);
      if (match) {
        meeting.confidence += 0.15;
        break;
      }
    }

    // Extract duration
    for (const pattern of this.durationPatterns) {
      const match = text.match(pattern);
      if (match) {
        const value = parseInt(match[1]);
        const unit = match[2];
        meeting.duration = unit.startsWith('h') ? value * 60 : value;
        meeting.confidence += 0.1;
        break;
      }
    }

    // Extract location/link
    const zoomMatch = text.match(/https:\/\/[^\s]*zoom[^\s]*/i);
    const teamsMatch = text.match(/https:\/\/teams\.microsoft\.com[^\s]*/i);
    const meetMatch = text.match(/https:\/\/meet\.google\.com[^\s]*/i);

    if (zoomMatch || teamsMatch || meetMatch) {
      meeting.location = (zoomMatch || teamsMatch || meetMatch)![0];
      meeting.confidence += 0.2;
    }

    return meeting.confidence > 0.5 ? meeting : null;
  }
}

// ============================================================================
// Categorization Service
// ============================================================================

class CategorizationService {
  private embeddingService: EmbeddingService;
  private categories: EmailCategory[] = [
    {
      id: 'work',
      name: 'Work',
      description: 'Work-related emails',
      color: '#3b82f6',
      icon: 'briefcase',
      isSystem: true,
      emailCount: 0,
    },
    {
      id: 'personal',
      name: 'Personal',
      description: 'Personal correspondence',
      color: '#10b981',
      icon: 'user',
      isSystem: true,
      emailCount: 0,
    },
    {
      id: 'finance',
      name: 'Finance',
      description: 'Financial and billing',
      color: '#f59e0b',
      icon: 'dollar-sign',
      isSystem: true,
      emailCount: 0,
    },
    {
      id: 'shopping',
      name: 'Shopping',
      description: 'Orders and receipts',
      color: '#8b5cf6',
      icon: 'shopping-bag',
      isSystem: true,
      emailCount: 0,
    },
    {
      id: 'travel',
      name: 'Travel',
      description: 'Travel and bookings',
      color: '#ec4899',
      icon: 'plane',
      isSystem: true,
      emailCount: 0,
    },
    {
      id: 'newsletters',
      name: 'Newsletters',
      description: 'Subscriptions and updates',
      color: '#6366f1',
      icon: 'mail',
      isSystem: true,
      emailCount: 0,
    },
  ];

  private categoryEmbeddings: Map<string, number[]> = new Map();

  constructor(embeddingService: EmbeddingService) {
    this.embeddingService = embeddingService;
    this.initializeCategoryEmbeddings();
  }

  private async initializeCategoryEmbeddings(): Promise<void> {
    for (const category of this.categories) {
      const text = `${category.name} ${category.description}`;
      const embedding = await this.embeddingService.getEmbedding(text);
      this.categoryEmbeddings.set(category.id, embedding);
    }
  }

  async categorize(
    email: { subject: string; body: string; from: string }
  ): Promise<{ categoryId: string; confidence: number }[]> {
    const emailText = `${email.subject} ${email.body}`;
    const emailEmbedding = await this.embeddingService.getEmbedding(emailText);

    const scores: { categoryId: string; confidence: number }[] = [];

    for (const [categoryId, categoryEmbedding] of this.categoryEmbeddings) {
      const similarity = this.embeddingService.cosineSimilarity(emailEmbedding, categoryEmbedding);
      // Normalize to 0-1 confidence
      const confidence = (similarity + 1) / 2;
      scores.push({ categoryId, confidence });
    }

    return scores.sort((a, b) => b.confidence - a.confidence);
  }

  getCategories(): EmailCategory[] {
    return this.categories;
  }
}

// ============================================================================
// Relationship Analyzer
// ============================================================================

class RelationshipAnalyzer {
  analyzeRelationship(
    emails: {
      id: string;
      from: string;
      to: string[];
      date: Date;
      subject: string;
      sentiment?: number;
    }[],
    contactEmail: string,
    myEmail: string
  ): RelationshipInsight {
    const relevantEmails = emails.filter(
      (e) => e.from === contactEmail || e.to.includes(contactEmail)
    );

    const sentByMe = relevantEmails.filter((e) => e.from === myEmail).length;
    const sentToMe = relevantEmails.length - sentByMe;

    // Calculate response times
    const responseTimes: number[] = [];
    const sortedEmails = relevantEmails.sort((a, b) => a.date.getTime() - b.date.getTime());

    for (let i = 1; i < sortedEmails.length; i++) {
      const prev = sortedEmails[i - 1];
      const curr = sortedEmails[i];
      if (prev.from !== curr.from) {
        const responseTime = (curr.date.getTime() - prev.date.getTime()) / (1000 * 60 * 60);
        if (responseTime < 168) {
          // Within a week
          responseTimes.push(responseTime);
        }
      }
    }

    const avgResponseTime =
      responseTimes.length > 0
        ? responseTimes.reduce((a, b) => a + b, 0) / responseTimes.length
        : 0;

    // Calculate communication frequency
    const daySpan =
      relevantEmails.length > 1
        ? (sortedEmails[sortedEmails.length - 1].date.getTime() - sortedEmails[0].date.getTime()) /
          (1000 * 60 * 60 * 24)
        : 0;
    const emailsPerDay = daySpan > 0 ? relevantEmails.length / daySpan : 0;

    let frequency: RelationshipInsight['metrics']['communicationFrequency'];
    if (emailsPerDay >= 1) frequency = 'daily';
    else if (emailsPerDay >= 0.14) frequency = 'weekly';
    else if (emailsPerDay >= 0.03) frequency = 'monthly';
    else frequency = 'rare';

    // Analyze preferred times
    const hourCounts: Map<number, number> = new Map();
    for (const email of relevantEmails) {
      const hour = email.date.getHours();
      hourCounts.set(hour, (hourCounts.get(hour) || 0) + 1);
    }

    const preferredTimes = Array.from(hourCounts.entries())
      .map(([hour, count]) => ({ hour, count }))
      .sort((a, b) => b.count - a.count)
      .slice(0, 5);

    // Extract topics from subjects
    const topicCounts: Map<string, number> = new Map();
    for (const email of relevantEmails) {
      const words = email.subject.toLowerCase().split(/\s+/);
      for (const word of words) {
        if (word.length > 4 && !['about', 'these', 'those', 'their'].includes(word)) {
          topicCounts.set(word, (topicCounts.get(word) || 0) + 1);
        }
      }
    }

    const topTopics = Array.from(topicCounts.entries())
      .map(([topic, count]) => ({ topic, count }))
      .sort((a, b) => b.count - a.count)
      .slice(0, 5);

    // Sentiment trend by month
    const sentimentByMonth: Map<string, { sum: number; count: number }> = new Map();
    for (const email of relevantEmails) {
      const month = `${email.date.getFullYear()}-${String(email.date.getMonth() + 1).padStart(2, '0')}`;
      const current = sentimentByMonth.get(month) || { sum: 0, count: 0 };
      sentimentByMonth.set(month, {
        sum: current.sum + (email.sentiment || 0.5),
        count: current.count + 1,
      });
    }

    const sentimentTrend = Array.from(sentimentByMonth.entries())
      .map(([month, data]) => ({
        month,
        sentiment: data.sum / data.count,
      }))
      .sort((a, b) => a.month.localeCompare(b.month));

    return {
      contactEmail,
      metrics: {
        totalEmails: relevantEmails.length,
        sentByMe,
        sentToMe,
        avgResponseTime,
        lastContact: sortedEmails[sortedEmails.length - 1]?.date || new Date(),
        firstContact: sortedEmails[0]?.date || new Date(),
        communicationFrequency: frequency,
      },
      patterns: {
        preferredTimes,
        topTopics,
        sentimentTrend,
      },
      trustEvolution: [], // Would be populated from trust service
    };
  }
}

// ============================================================================
// Store
// ============================================================================

interface SemanticSearchState {
  searchHistory: SearchQuery[];
  recentResults: SearchResult[];
  suggestedCategories: Map<string, { categoryId: string; confidence: number }[]>;
  detectedMeetings: ExtractedMeeting[];
  relationshipInsights: Map<string, RelationshipInsight>;

  addToHistory: (query: SearchQuery) => void;
  setResults: (results: SearchResult[]) => void;
  setSuggestedCategory: (
    emailId: string,
    suggestions: { categoryId: string; confidence: number }[]
  ) => void;
  addDetectedMeeting: (meeting: ExtractedMeeting) => void;
  setRelationshipInsight: (email: string, insight: RelationshipInsight) => void;
}

export const useSemanticSearchStore = create<SemanticSearchState>((set) => ({
  searchHistory: [],
  recentResults: [],
  suggestedCategories: new Map(),
  detectedMeetings: [],
  relationshipInsights: new Map(),

  addToHistory: (query) =>
    set((state) => ({
      searchHistory: [query, ...state.searchHistory.slice(0, 49)],
    })),

  setResults: (results) => set({ recentResults: results }),

  setSuggestedCategory: (emailId, suggestions) =>
    set((state) => {
      const map = new Map(state.suggestedCategories);
      map.set(emailId, suggestions);
      return { suggestedCategories: map };
    }),

  addDetectedMeeting: (meeting) =>
    set((state) => ({
      detectedMeetings: [...state.detectedMeetings, meeting],
    })),

  setRelationshipInsight: (email, insight) =>
    set((state) => {
      const map = new Map(state.relationshipInsights);
      map.set(email, insight);
      return { relationshipInsights: map };
    }),
}));

// ============================================================================
// Singleton Services
// ============================================================================

const embeddingService = new EmbeddingService();
const nlParser = new NLQueryParser();
const meetingExtractor = new MeetingExtractor();
const categorizationService = new CategorizationService(embeddingService);
const relationshipAnalyzer = new RelationshipAnalyzer();

// ============================================================================
// React Hooks
// ============================================================================

import { useState, useCallback, useMemo } from 'react';

export function useSemanticSearch() {
  const [isSearching, setIsSearching] = useState(false);
  const [results, setResults] = useState<SearchResult[]>([]);
  const { addToHistory, setResults: storeResults } = useSemanticSearchStore();

  const search = useCallback(
    async (query: string | SearchQuery): Promise<SearchResult[]> => {
      setIsSearching(true);

      try {
        // Parse natural language if string
        const searchQuery =
          typeof query === 'string' ? nlParser.parse(query).structuredQuery : query;

        addToHistory(searchQuery);

        // In production, this would call the search API
        // For now, return mock results
        const mockResults: SearchResult[] = [
          {
            id: '1',
            emailId: 'email_1',
            score: 0.95,
            matchType: 'semantic',
            highlights: [
              {
                field: 'subject',
                text: 'Meeting tomorrow at 3pm',
                matches: [{ start: 0, end: 7 }],
              },
            ],
            email: {
              id: 'email_1',
              threadId: 'thread_1',
              subject: 'Meeting tomorrow at 3pm',
              snippet: 'Let me know if the time works for you...',
              from: { name: 'John Doe', email: 'john@example.com' },
              date: new Date(),
              isRead: true,
              trustScore: 85,
              labels: ['work'],
            },
          },
        ];

        setResults(mockResults);
        storeResults(mockResults);
        return mockResults;
      } finally {
        setIsSearching(false);
      }
    },
    [addToHistory, storeResults]
  );

  const parseQuery = useCallback((query: string) => {
    return nlParser.parse(query);
  }, []);

  return {
    search,
    parseQuery,
    results,
    isSearching,
  };
}

export function useNaturalLanguageSearch() {
  const [parsedQuery, setParsedQuery] = useState<NaturalLanguageQuery | null>(null);

  const parse = useCallback((query: string) => {
    const parsed = nlParser.parse(query);
    setParsedQuery(parsed);
    return parsed;
  }, []);

  return {
    parse,
    parsedQuery,
    entities: parsedQuery?.parsed.entities || [],
    timeRange: parsedQuery?.parsed.timeRange,
    structuredQuery: parsedQuery?.structuredQuery,
  };
}

export function useAutoCategorization() {
  const { setSuggestedCategory, suggestedCategories } = useSemanticSearchStore();

  const categorize = useCallback(
    async (email: { id: string; subject: string; body: string; from: string }) => {
      const suggestions = await categorizationService.categorize(email);
      setSuggestedCategory(email.id, suggestions);
      return suggestions;
    },
    [setSuggestedCategory]
  );

  const getSuggestions = useCallback(
    (emailId: string) => {
      return suggestedCategories.get(emailId) || [];
    },
    [suggestedCategories]
  );

  const categories = useMemo(() => categorizationService.getCategories(), []);

  return {
    categorize,
    getSuggestions,
    categories,
  };
}

export function useMeetingExtraction() {
  const { addDetectedMeeting, detectedMeetings } = useSemanticSearchStore();

  const extractMeeting = useCallback(
    (emailId: string, subject: string, body: string) => {
      const extracted = meetingExtractor.extract(body, subject);
      if (extracted) {
        const meeting: ExtractedMeeting = {
          id: `meeting_${Date.now()}`,
          emailId,
          title: extracted.title || subject,
          dateTime: extracted.dateTime,
          duration: extracted.duration,
          location: extracted.location,
          attendees: [],
          confidence: extracted.confidence || 0,
          status: 'detected',
        };
        addDetectedMeeting(meeting);
        return meeting;
      }
      return null;
    },
    [addDetectedMeeting]
  );

  return {
    extractMeeting,
    detectedMeetings,
  };
}

export function useRelationshipInsights() {
  const { setRelationshipInsight, relationshipInsights } = useSemanticSearchStore();

  const analyzeRelationship = useCallback(
    (
      emails: {
        id: string;
        from: string;
        to: string[];
        date: Date;
        subject: string;
        sentiment?: number;
      }[],
      contactEmail: string,
      myEmail: string
    ) => {
      const insight = relationshipAnalyzer.analyzeRelationship(emails, contactEmail, myEmail);
      setRelationshipInsight(contactEmail, insight);
      return insight;
    },
    [setRelationshipInsight]
  );

  const getInsight = useCallback(
    (email: string) => {
      return relationshipInsights.get(email);
    },
    [relationshipInsights]
  );

  return {
    analyzeRelationship,
    getInsight,
    allInsights: Array.from(relationshipInsights.values()),
  };
}
