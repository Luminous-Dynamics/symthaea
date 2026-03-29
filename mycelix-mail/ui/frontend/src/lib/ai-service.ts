// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * AI Service Integration Layer
 *
 * Provides intelligent email analysis, summarization, and insights
 * using LLM APIs (OpenAI, Anthropic, local models).
 */

import { create } from 'zustand';
import { persist } from 'zustand/middleware';

// ============================================
// Types
// ============================================

export type AIProvider = 'openai' | 'anthropic' | 'local' | 'mock';

export interface AIConfig {
  provider: AIProvider;
  apiKey?: string;
  baseUrl?: string;
  model?: string;
  maxTokens?: number;
  temperature?: number;
}

export interface EmailAnalysis {
  id: string;
  emailId: string;
  summary: string;
  keyPoints: string[];
  sentiment: 'positive' | 'neutral' | 'negative' | 'mixed';
  sentimentScore: number; // -1 to 1
  intent: EmailIntent;
  urgency: 'low' | 'medium' | 'high' | 'critical';
  urgencyScore: number; // 0 to 1
  topics: string[];
  entities: ExtractedEntity[];
  claims: ExtractedClaim[];
  suggestedActions: SuggestedAction[];
  riskIndicators: RiskIndicator[];
  analyzedAt: string;
}

export interface EmailIntent {
  primary: 'informational' | 'request' | 'action_required' | 'fyi' | 'social' | 'marketing' | 'spam' | 'phishing';
  confidence: number;
  secondary?: string[];
}

export interface ExtractedEntity {
  type: 'person' | 'organization' | 'location' | 'date' | 'money' | 'url' | 'email' | 'phone';
  value: string;
  context: string;
  confidence: number;
}

export interface ExtractedClaim {
  statement: string;
  claimant: string;
  category: 'factual' | 'opinion' | 'promise' | 'request' | 'threat';
  verifiable: boolean;
  confidence: number;
  suggestedVerification?: string;
}

export interface SuggestedAction {
  type: 'reply' | 'forward' | 'archive' | 'delete' | 'snooze' | 'create_task' | 'add_calendar' | 'verify_claim';
  description: string;
  priority: number;
  draftContent?: string;
}

export interface RiskIndicator {
  type: 'phishing' | 'scam' | 'impersonation' | 'malware' | 'social_engineering' | 'urgency_manipulation';
  severity: 'low' | 'medium' | 'high' | 'critical';
  description: string;
  evidence: string[];
  confidence: number;
}

export interface ThreadSummary {
  id: string;
  threadId: string;
  participantCount: number;
  messageCount: number;
  summary: string;
  timeline: ThreadEvent[];
  decisions: string[];
  actionItems: ActionItem[];
  openQuestions: string[];
  sentiment: 'positive' | 'neutral' | 'negative' | 'mixed';
  analyzedAt: string;
}

export interface ThreadEvent {
  timestamp: string;
  participant: string;
  type: 'message' | 'decision' | 'question' | 'action_item' | 'sentiment_shift';
  description: string;
}

export interface ActionItem {
  description: string;
  assignee?: string;
  dueDate?: string;
  status: 'pending' | 'in_progress' | 'completed';
  source: string;
}

export interface SmartReply {
  id: string;
  content: string;
  tone: 'professional' | 'friendly' | 'brief' | 'detailed';
  intent: 'accept' | 'decline' | 'ask_questions' | 'acknowledge' | 'custom';
  confidence: number;
}

export interface ComposeSuggestion {
  type: 'subject' | 'opening' | 'body' | 'closing' | 'tone_adjustment';
  original?: string;
  suggestion: string;
  reason: string;
}

// ============================================
// AI Service Store
// ============================================

interface AIServiceState {
  config: AIConfig;
  isConfigured: boolean;
  analysisCache: Map<string, EmailAnalysis>;
  threadCache: Map<string, ThreadSummary>;
  isProcessing: boolean;
  error: string | null;
}

interface AIServiceActions {
  configure: (config: Partial<AIConfig>) => void;
  clearCache: () => void;
  setProcessing: (processing: boolean) => void;
  setError: (error: string | null) => void;
  cacheAnalysis: (analysis: EmailAnalysis) => void;
  cacheThreadSummary: (summary: ThreadSummary) => void;
}

export const useAIServiceStore = create<AIServiceState & AIServiceActions>()(
  persist(
    (set, get) => ({
      config: {
        provider: 'mock',
        temperature: 0.7,
        maxTokens: 2000,
      },
      isConfigured: false,
      analysisCache: new Map(),
      threadCache: new Map(),
      isProcessing: false,
      error: null,

      configure: (config) => {
        set((state) => ({
          config: { ...state.config, ...config },
          isConfigured: !!(config.apiKey || config.provider === 'mock' || config.provider === 'local'),
        }));
      },

      clearCache: () => {
        set({
          analysisCache: new Map(),
          threadCache: new Map(),
        });
      },

      setProcessing: (processing) => set({ isProcessing: processing }),
      setError: (error) => set({ error }),

      cacheAnalysis: (analysis) => {
        set((state) => {
          const cache = new Map(state.analysisCache);
          cache.set(analysis.emailId, analysis);
          return { analysisCache: cache };
        });
      },

      cacheThreadSummary: (summary) => {
        set((state) => {
          const cache = new Map(state.threadCache);
          cache.set(summary.threadId, summary);
          return { threadCache: cache };
        });
      },
    }),
    {
      name: 'ai-service-config',
      partialize: (state) => ({
        config: {
          ...state.config,
          apiKey: undefined, // Don't persist API key
        },
      }),
    }
  )
);

// ============================================
// AI Service Class
// ============================================

class AIService {
  private config: AIConfig;

  constructor(config: AIConfig) {
    this.config = config;
  }

  private async callAPI(prompt: string, systemPrompt?: string): Promise<string> {
    const { provider, apiKey, baseUrl, model, maxTokens, temperature } = this.config;

    if (provider === 'mock') {
      return this.mockResponse(prompt);
    }

    if (provider === 'openai') {
      const response = await fetch(baseUrl || 'https://api.openai.com/v1/chat/completions', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${apiKey}`,
        },
        body: JSON.stringify({
          model: model || 'gpt-4-turbo-preview',
          messages: [
            ...(systemPrompt ? [{ role: 'system', content: systemPrompt }] : []),
            { role: 'user', content: prompt },
          ],
          max_tokens: maxTokens,
          temperature,
        }),
      });

      if (!response.ok) {
        throw new Error(`OpenAI API error: ${response.statusText}`);
      }

      const data = await response.json();
      return data.choices[0].message.content;
    }

    if (provider === 'anthropic') {
      const response = await fetch(baseUrl || 'https://api.anthropic.com/v1/messages', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'x-api-key': apiKey!,
          'anthropic-version': '2023-06-01',
        },
        body: JSON.stringify({
          model: model || 'claude-3-opus-20240229',
          max_tokens: maxTokens,
          system: systemPrompt,
          messages: [{ role: 'user', content: prompt }],
        }),
      });

      if (!response.ok) {
        throw new Error(`Anthropic API error: ${response.statusText}`);
      }

      const data = await response.json();
      return data.content[0].text;
    }

    if (provider === 'local') {
      const response = await fetch(baseUrl || 'http://localhost:11434/api/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: model || 'llama2',
          prompt: systemPrompt ? `${systemPrompt}\n\n${prompt}` : prompt,
          stream: false,
        }),
      });

      if (!response.ok) {
        throw new Error(`Local API error: ${response.statusText}`);
      }

      const data = await response.json();
      return data.response;
    }

    throw new Error(`Unknown provider: ${provider}`);
  }

  private mockResponse(prompt: string): string {
    // Generate plausible mock responses for testing
    if (prompt.includes('analyze')) {
      return JSON.stringify({
        summary: 'This email discusses project updates and timeline changes.',
        keyPoints: ['Project deadline moved to next month', 'Budget approved', 'Team expansion planned'],
        sentiment: 'positive',
        sentimentScore: 0.6,
        intent: { primary: 'informational', confidence: 0.85 },
        urgency: 'medium',
        urgencyScore: 0.5,
        topics: ['project management', 'timeline', 'budget'],
        entities: [
          { type: 'date', value: 'next month', context: 'deadline', confidence: 0.9 },
          { type: 'money', value: '$50,000', context: 'budget', confidence: 0.95 },
        ],
        claims: [
          { statement: 'Budget has been approved', claimant: 'sender', category: 'factual', verifiable: true, confidence: 0.8 },
        ],
        suggestedActions: [
          { type: 'reply', description: 'Acknowledge receipt and confirm timeline', priority: 1 },
          { type: 'add_calendar', description: 'Add new deadline to calendar', priority: 2 },
        ],
        riskIndicators: [],
      });
    }

    if (prompt.includes('summarize thread')) {
      return JSON.stringify({
        summary: 'Discussion about Q4 planning with multiple stakeholders.',
        timeline: [
          { timestamp: '2024-01-15T10:00:00Z', participant: 'Alice', type: 'message', description: 'Initiated planning discussion' },
          { timestamp: '2024-01-15T14:00:00Z', participant: 'Bob', type: 'question', description: 'Asked about budget constraints' },
        ],
        decisions: ['Move forward with Option A', 'Delay Phase 2 by two weeks'],
        actionItems: [
          { description: 'Prepare budget proposal', assignee: 'Alice', dueDate: '2024-01-20', status: 'pending', source: 'email' },
        ],
        openQuestions: ['What is the final headcount?'],
        sentiment: 'neutral',
      });
    }

    if (prompt.includes('smart replies')) {
      return JSON.stringify([
        { content: 'Thank you for the update. I\'ll review and get back to you by EOD.', tone: 'professional', intent: 'acknowledge', confidence: 0.9 },
        { content: 'Got it, thanks! Let me know if you need anything else.', tone: 'friendly', intent: 'acknowledge', confidence: 0.85 },
        { content: 'Thanks. A few questions before I proceed...', tone: 'professional', intent: 'ask_questions', confidence: 0.8 },
      ]);
    }

    if (prompt.includes('phishing') || prompt.includes('risk')) {
      return JSON.stringify({
        riskIndicators: [
          { type: 'urgency_manipulation', severity: 'medium', description: 'Uses urgent language to pressure action', evidence: ['Act now', 'Limited time'], confidence: 0.7 },
        ],
        overallRisk: 'low',
        recommendation: 'Exercise caution but likely safe',
      });
    }

    return JSON.stringify({ result: 'Mock response', success: true });
  }

  async analyzeEmail(email: {
    id: string;
    subject: string;
    body: string;
    from: string;
    to: string[];
    date: string;
  }): Promise<EmailAnalysis> {
    const systemPrompt = `You are an AI assistant specialized in email analysis. Analyze emails for:
- Summary and key points
- Sentiment and urgency
- Intent classification
- Entity extraction (people, organizations, dates, money)
- Claim identification
- Risk indicators (phishing, scam, impersonation)
- Suggested actions

Respond in JSON format only.`;

    const prompt = `Analyze this email:
Subject: ${email.subject}
From: ${email.from}
To: ${email.to.join(', ')}
Date: ${email.date}

Body:
${email.body}

Provide a comprehensive analysis in JSON format.`;

    const response = await this.callAPI(prompt, systemPrompt);
    const analysis = JSON.parse(response);

    return {
      id: crypto.randomUUID(),
      emailId: email.id,
      ...analysis,
      analyzedAt: new Date().toISOString(),
    };
  }

  async summarizeThread(thread: {
    id: string;
    subject: string;
    messages: Array<{
      from: string;
      body: string;
      date: string;
    }>;
  }): Promise<ThreadSummary> {
    const systemPrompt = `You are an AI assistant specialized in email thread analysis. Summarize threads to identify:
- Overall summary
- Key timeline events
- Decisions made
- Action items with assignees
- Open questions
- Overall sentiment

Respond in JSON format only.`;

    const messagesText = thread.messages
      .map((m, i) => `[${i + 1}] From: ${m.from}\nDate: ${m.date}\n${m.body}`)
      .join('\n\n---\n\n');

    const prompt = `Summarize this email thread:
Subject: ${thread.subject}

Messages:
${messagesText}

Provide a comprehensive thread summary in JSON format.`;

    const response = await this.callAPI(prompt, systemPrompt);
    const summary = JSON.parse(response);

    return {
      id: crypto.randomUUID(),
      threadId: thread.id,
      participantCount: new Set(thread.messages.map((m) => m.from)).size,
      messageCount: thread.messages.length,
      ...summary,
      analyzedAt: new Date().toISOString(),
    };
  }

  async generateSmartReplies(email: {
    subject: string;
    body: string;
    from: string;
  }): Promise<SmartReply[]> {
    const systemPrompt = `You are an AI assistant that generates contextually appropriate email replies. Generate 3-5 reply options with different tones and intents. Respond in JSON format as an array of replies.`;

    const prompt = `Generate smart replies for this email:
Subject: ${email.subject}
From: ${email.from}

Body:
${email.body}

Generate reply options in JSON format.`;

    const response = await this.callAPI(prompt, systemPrompt);
    const replies = JSON.parse(response);

    return replies.map((r: Omit<SmartReply, 'id'>) => ({
      id: crypto.randomUUID(),
      ...r,
    }));
  }

  async detectRisks(email: {
    subject: string;
    body: string;
    from: string;
    headers?: Record<string, string>;
  }): Promise<RiskIndicator[]> {
    const systemPrompt = `You are a security AI specialized in detecting email threats. Analyze for:
- Phishing attempts
- Scam indicators
- Impersonation
- Social engineering
- Urgency manipulation
- Suspicious links/attachments

Be thorough but avoid false positives. Respond in JSON format.`;

    const prompt = `Analyze this email for security risks:
Subject: ${email.subject}
From: ${email.from}
${email.headers ? `Headers: ${JSON.stringify(email.headers)}` : ''}

Body:
${email.body}

Provide risk analysis in JSON format.`;

    const response = await this.callAPI(prompt, systemPrompt);
    const analysis = JSON.parse(response);

    return analysis.riskIndicators || [];
  }

  async suggestCompose(draft: {
    to: string[];
    subject?: string;
    body?: string;
    context?: string;
  }): Promise<ComposeSuggestion[]> {
    const systemPrompt = `You are an AI writing assistant for professional emails. Suggest improvements for:
- Subject lines
- Opening lines
- Body content
- Closing lines
- Tone adjustments

Be helpful but preserve the author's voice. Respond in JSON format.`;

    const prompt = `Help improve this email draft:
To: ${draft.to.join(', ')}
${draft.subject ? `Subject: ${draft.subject}` : ''}
${draft.context ? `Context: ${draft.context}` : ''}

Draft:
${draft.body || '(empty)'}

Provide suggestions in JSON format.`;

    const response = await this.callAPI(prompt, systemPrompt);
    return JSON.parse(response);
  }

  async extractClaims(text: string): Promise<ExtractedClaim[]> {
    const systemPrompt = `You are an AI specialized in identifying claims and assertions in text. Extract:
- Factual claims that can be verified
- Opinions and subjective statements
- Promises and commitments
- Requests and demands
- Potential threats

For each claim, assess if it's verifiable and how. Respond in JSON format.`;

    const prompt = `Extract claims from this text:
${text}

Provide extracted claims in JSON format.`;

    const response = await this.callAPI(prompt, systemPrompt);
    return JSON.parse(response);
  }
}

// ============================================
// React Hooks
// ============================================

export function useAIService() {
  const config = useAIServiceStore((s) => s.config);
  const isConfigured = useAIServiceStore((s) => s.isConfigured);
  const configure = useAIServiceStore((s) => s.configure);

  const service = isConfigured ? new AIService(config) : null;

  return {
    service,
    isConfigured,
    configure,
    provider: config.provider,
  };
}

export function useEmailAnalysis(emailId: string | null, email?: {
  subject: string;
  body: string;
  from: string;
  to: string[];
  date: string;
}) {
  const { service } = useAIService();
  const cache = useAIServiceStore((s) => s.analysisCache);
  const cacheAnalysis = useAIServiceStore((s) => s.cacheAnalysis);
  const setProcessing = useAIServiceStore((s) => s.setProcessing);
  const setError = useAIServiceStore((s) => s.setError);

  const [analysis, setAnalysis] = useState<EmailAnalysis | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    if (!emailId) {
      setAnalysis(null);
      return;
    }

    // Check cache first
    const cached = cache.get(emailId);
    if (cached) {
      setAnalysis(cached);
      return;
    }

    // Analyze if we have the email data
    if (email && service) {
      setIsLoading(true);
      setProcessing(true);

      service
        .analyzeEmail({ id: emailId, ...email })
        .then((result) => {
          setAnalysis(result);
          cacheAnalysis(result);
        })
        .catch((err) => {
          setError(err.message);
        })
        .finally(() => {
          setIsLoading(false);
          setProcessing(false);
        });
    }
  }, [emailId, email?.body, service]);

  return { analysis, isLoading };
}

export function useSmartReplies(email: {
  subject: string;
  body: string;
  from: string;
} | null) {
  const { service } = useAIService();
  const [replies, setReplies] = useState<SmartReply[]>([]);
  const [isLoading, setIsLoading] = useState(false);

  const generate = async () => {
    if (!email || !service) return;

    setIsLoading(true);
    try {
      const result = await service.generateSmartReplies(email);
      setReplies(result);
    } catch (err) {
      console.error('Failed to generate smart replies:', err);
    } finally {
      setIsLoading(false);
    }
  };

  return { replies, isLoading, generate };
}

export function useRiskDetection(email: {
  subject: string;
  body: string;
  from: string;
  headers?: Record<string, string>;
} | null) {
  const { service } = useAIService();
  const [risks, setRisks] = useState<RiskIndicator[]>([]);
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    if (!email || !service) {
      setRisks([]);
      return;
    }

    setIsLoading(true);
    service
      .detectRisks(email)
      .then(setRisks)
      .catch((err) => {
        console.error('Failed to detect risks:', err);
        setRisks([]);
      })
      .finally(() => setIsLoading(false));
  }, [email?.body, service]);

  return { risks, isLoading, hasHighRisk: risks.some((r) => r.severity === 'high' || r.severity === 'critical') };
}

// Import useState/useEffect
import { useState, useEffect } from 'react';

export default AIService;
