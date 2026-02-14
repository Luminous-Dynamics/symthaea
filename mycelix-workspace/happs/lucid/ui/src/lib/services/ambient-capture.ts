/**
 * Ambient Capture System
 *
 * Effortless knowledge capture through:
 * - Voice streaming with insight detection
 * - Screen/browser context awareness
 * - Activity pattern recognition
 * - Automatic thought candidate generation
 */

import type { CreateThoughtInput } from '@mycelix/lucid-client';
import { ThoughtType, EmpiricalLevel, NormativeLevel, MaterialityLevel, HarmonicLevel } from '@mycelix/lucid-client';

// Web Speech API types (not in lib.dom by default)
interface SpeechRecognitionEvent extends Event {
  results: SpeechRecognitionResultList;
  resultIndex: number;
}

interface SpeechRecognitionErrorEvent extends Event {
  error: string;
  message: string;
}

interface SpeechRecognition extends EventTarget {
  continuous: boolean;
  interimResults: boolean;
  lang: string;
  maxAlternatives: number;
  onresult: ((event: SpeechRecognitionEvent) => void) | null;
  onerror: ((event: SpeechRecognitionErrorEvent) => void) | null;
  onend: (() => void) | null;
  start(): void;
  stop(): void;
  abort(): void;
}

// ============================================================================
// TYPES
// ============================================================================

export interface CaptureContext {
  timestamp: number;
  source: 'voice' | 'screen' | 'manual' | 'integration';
  confidence: number;
  metadata: {
    duration?: number;
    url?: string;
    title?: string;
    app?: string;
    emotion?: string;
    energy?: 'low' | 'medium' | 'high';
  };
}

export interface ThoughtCandidate {
  content: string;
  suggestedType: ThoughtType;
  context: CaptureContext;
  insightScore: number;
  tags: string[];
  approved: boolean;
}

export interface VoiceSegment {
  text: string;
  timestamp: number;
  confidence: number;
  isFinal: boolean;
}

export interface AmbientState {
  isListening: boolean;
  isProcessing: boolean;
  currentTranscript: string;
  candidates: ThoughtCandidate[];
  captureMode: 'continuous' | 'triggered' | 'paused';
  insightThreshold: number;
}

// ============================================================================
// VOICE CAPTURE SERVICE
// ============================================================================

class VoiceCaptureService {
  private recognition: SpeechRecognition | null = null;
  private isSupported: boolean;
  private segments: VoiceSegment[] = [];
  private onSegmentCallbacks: ((segment: VoiceSegment) => void)[] = [];
  private onInsightCallbacks: ((candidate: ThoughtCandidate) => void)[] = [];
  private silenceTimeout: number | null = null;
  private processingBuffer: string[] = [];

  constructor() {
    this.isSupported = 'SpeechRecognition' in window || 'webkitSpeechRecognition' in window;
  }

  get supported(): boolean {
    return this.isSupported;
  }

  initialize(): boolean {
    if (!this.isSupported) return false;

    const SpeechRecognitionAPI = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
    const recognition = new SpeechRecognitionAPI() as SpeechRecognition;
    this.recognition = recognition;

    recognition.continuous = true;
    recognition.interimResults = true;
    recognition.lang = 'en-US';
    recognition.maxAlternatives = 1;

    recognition.onresult = (event) => {
      const result = event.results[event.results.length - 1];
      const transcript = result[0].transcript;
      const confidence = result[0].confidence;

      const segment: VoiceSegment = {
        text: transcript,
        timestamp: Date.now(),
        confidence,
        isFinal: result.isFinal,
      };

      this.segments.push(segment);
      this.onSegmentCallbacks.forEach((cb) => cb(segment));

      if (result.isFinal) {
        this.processingBuffer.push(transcript);
        this.scheduleInsightDetection();
      }
    };

    recognition.onerror = (event) => {
      console.error('Speech recognition error:', event.error);
      if (event.error === 'no-speech') {
        // Restart on silence
        this.restart();
      }
    };

    recognition.onend = () => {
      // Auto-restart if still supposed to be listening
      if (this.recognition) {
        try {
          this.recognition.start();
        } catch (e) {
          // Already started, ignore
        }
      }
    };

    return true;
  }

  start(): void {
    if (!this.recognition) {
      this.initialize();
    }
    try {
      this.recognition?.start();
    } catch (e) {
      // Already running
    }
  }

  stop(): void {
    this.recognition?.stop();
  }

  restart(): void {
    this.stop();
    setTimeout(() => this.start(), 100);
  }

  onSegment(callback: (segment: VoiceSegment) => void): void {
    this.onSegmentCallbacks.push(callback);
  }

  onInsight(callback: (candidate: ThoughtCandidate) => void): void {
    this.onInsightCallbacks.push(callback);
  }

  private scheduleInsightDetection(): void {
    if (this.silenceTimeout) {
      clearTimeout(this.silenceTimeout);
    }

    // Process after 2 seconds of silence
    this.silenceTimeout = window.setTimeout(() => {
      if (this.processingBuffer.length > 0) {
        const fullText = this.processingBuffer.join(' ');
        this.processingBuffer = [];
        this.detectInsights(fullText);
      }
    }, 2000);
  }

  private async detectInsights(text: string): Promise<void> {
    const insights = await detectInsightMoments(text);

    for (const insight of insights) {
      const candidate: ThoughtCandidate = {
        content: insight.text,
        suggestedType: insight.type,
        context: {
          timestamp: Date.now(),
          source: 'voice',
          confidence: insight.score,
          metadata: {},
        },
        insightScore: insight.score,
        tags: insight.tags,
        approved: false,
      };

      this.onInsightCallbacks.forEach((cb) => cb(candidate));
    }
  }
}

// ============================================================================
// INSIGHT DETECTION
// ============================================================================

interface InsightMoment {
  text: string;
  type: ThoughtType;
  score: number;
  tags: string[];
}

// Patterns that indicate insight moments
const INSIGHT_PATTERNS = [
  // Realizations
  { pattern: /\b(i just realized|it hit me|i finally understand|now i see)\b/i, type: ThoughtType.Insight, boost: 0.3 },
  { pattern: /\b(the key is|the secret is|what matters is)\b/i, type: ThoughtType.Claim, boost: 0.25 },

  // Questions worth capturing
  { pattern: /\b(what if|i wonder|how come|why do)\b.*\?/i, type: ThoughtType.Question, boost: 0.3 },
  { pattern: /\b(the real question is|we should ask)\b/i, type: ThoughtType.Question, boost: 0.25 },

  // Beliefs and values
  { pattern: /\b(i believe|i think|in my view|my sense is)\b/i, type: ThoughtType.Claim, boost: 0.2 },
  { pattern: /\b(what's important is|what matters most)\b/i, type: ThoughtType.Claim, boost: 0.25 },

  // Hypotheses
  { pattern: /\b(maybe|perhaps|could be that|might be)\b/i, type: ThoughtType.Hypothesis, boost: 0.15 },
  { pattern: /\b(my hypothesis is|i suspect|my theory)\b/i, type: ThoughtType.Hypothesis, boost: 0.3 },

  // Goals and intentions
  { pattern: /\b(i want to|i need to|my goal is|i'm going to)\b/i, type: ThoughtType.Task, boost: 0.2 },
  { pattern: /\b(the plan is|next step is|we should)\b/i, type: ThoughtType.Task, boost: 0.2 },

  // Definitions and clarifications
  { pattern: /\b(what i mean is|in other words|to clarify)\b/i, type: ThoughtType.Definition, boost: 0.15 },

  // Evidence and arguments
  { pattern: /\b(because|therefore|this proves|evidence shows)\b/i, type: ThoughtType.Claim, boost: 0.2 },

  // Memories
  { pattern: /\b(i remember|back when|that time when)\b/i, type: ThoughtType.Reflection, boost: 0.15 },

  // Intuitions
  { pattern: /\b(i feel like|something tells me|gut feeling)\b/i, type: ThoughtType.Insight, boost: 0.2 },

  // Reflections
  { pattern: /\b(looking back|on reflection|thinking about it)\b/i, type: ThoughtType.Reflection, boost: 0.2 },
];

// Keywords that boost insight score
const INSIGHT_KEYWORDS = [
  'insight', 'realize', 'understand', 'discover', 'learn', 'notice',
  'important', 'key', 'crucial', 'essential', 'fundamental',
  'interesting', 'surprising', 'unexpected', 'remarkable',
  'pattern', 'connection', 'relationship', 'link',
  'always', 'never', 'every', 'all', 'none', // Universal claims
];

// Domain keywords for tagging
const DOMAIN_KEYWORDS: Record<string, string[]> = {
  philosophy: ['meaning', 'truth', 'reality', 'existence', 'consciousness', 'ethics', 'morality'],
  psychology: ['mind', 'behavior', 'emotion', 'feeling', 'thought', 'mental', 'cognitive'],
  relationships: ['relationship', 'friend', 'family', 'love', 'trust', 'communication'],
  work: ['project', 'meeting', 'deadline', 'client', 'team', 'task', 'goal'],
  health: ['health', 'exercise', 'sleep', 'diet', 'energy', 'stress', 'wellness'],
  creativity: ['idea', 'creative', 'design', 'art', 'write', 'build', 'make'],
  learning: ['learn', 'study', 'understand', 'practice', 'skill', 'knowledge'],
  personal: ['life', 'self', 'growth', 'change', 'journey', 'path'],
};

/**
 * Detect insight-worthy moments in text
 */
async function detectInsightMoments(text: string): Promise<InsightMoment[]> {
  const insights: InsightMoment[] = [];
  const sentences = splitIntoSentences(text);

  for (const sentence of sentences) {
    const score = calculateInsightScore(sentence);

    if (score >= 0.3) {
      insights.push({
        text: sentence.trim(),
        type: detectThoughtType(sentence),
        score,
        tags: extractTags(sentence),
      });
    }
  }

  return insights;
}

/**
 * Split text into sentences
 */
function splitIntoSentences(text: string): string[] {
  return text
    .split(/[.!?]+/)
    .map((s) => s.trim())
    .filter((s) => s.length > 10);
}

/**
 * Calculate insight score for a piece of text
 */
function calculateInsightScore(text: string): number {
  let score = 0.1; // Base score
  const lower = text.toLowerCase();

  // Check insight patterns
  for (const { pattern, boost } of INSIGHT_PATTERNS) {
    if (pattern.test(text)) {
      score += boost;
    }
  }

  // Check insight keywords
  for (const keyword of INSIGHT_KEYWORDS) {
    if (lower.includes(keyword)) {
      score += 0.05;
    }
  }

  // Length bonus (longer = more substantial)
  if (text.length > 50) score += 0.05;
  if (text.length > 100) score += 0.05;
  if (text.length > 200) score += 0.05;

  // Specificity bonus (numbers, proper nouns)
  if (/\d+/.test(text)) score += 0.05;
  if (/[A-Z][a-z]+\s[A-Z][a-z]+/.test(text)) score += 0.05; // Names

  // Cap at 1.0
  return Math.min(1.0, score);
}

/**
 * Detect thought type from content
 */
function detectThoughtType(text: string): ThoughtType {
  for (const { pattern, type } of INSIGHT_PATTERNS) {
    if (pattern.test(text)) {
      return type;
    }
  }

  if (text.includes('?')) return ThoughtType.Question;
  return ThoughtType.Note;
}

/**
 * Extract relevant tags from text
 */
function extractTags(text: string): string[] {
  const tags: string[] = [];
  const lower = text.toLowerCase();

  for (const [domain, keywords] of Object.entries(DOMAIN_KEYWORDS)) {
    if (keywords.some((kw) => lower.includes(kw))) {
      tags.push(domain);
    }
  }

  return tags.slice(0, 5);
}

// ============================================================================
// SCREEN CONTEXT CAPTURE
// ============================================================================

export interface ScreenContext {
  url?: string;
  title?: string;
  selectedText?: string;
  timestamp: number;
}

/**
 * Capture current browser context
 */
export function captureScreenContext(): ScreenContext {
  return {
    url: window.location.href,
    title: document.title,
    selectedText: window.getSelection()?.toString(),
    timestamp: Date.now(),
  };
}

/**
 * Create thought from screen context
 */
export function createThoughtFromContext(context: ScreenContext): ThoughtCandidate | null {
  if (!context.selectedText || context.selectedText.length < 20) {
    return null;
  }

  return {
    content: context.selectedText + (context.url ? `\n\nSource: ${context.url}` : ''),
    suggestedType: ThoughtType.Quote,
    context: {
      timestamp: context.timestamp,
      source: 'screen',
      confidence: 0.6,
      metadata: {
        url: context.url,
        title: context.title,
      },
    },
    insightScore: 0.5,
    tags: ['web-capture'],
    approved: false,
  };
}

// ============================================================================
// CONTEXT AGGREGATOR
// ============================================================================

export interface AggregatedContext {
  time: {
    hour: number;
    dayOfWeek: number;
    isWeekend: boolean;
    timeOfDay: 'morning' | 'afternoon' | 'evening' | 'night';
  };
  activity: {
    recentTopics: string[];
    currentFocus?: string;
    sessionDuration: number;
  };
  environment: {
    url?: string;
    app?: string;
  };
}

/**
 * Aggregate current context
 */
export function aggregateContext(recentCandidates: ThoughtCandidate[]): AggregatedContext {
  const now = new Date();
  const hour = now.getHours();

  let timeOfDay: 'morning' | 'afternoon' | 'evening' | 'night';
  if (hour < 6) timeOfDay = 'night';
  else if (hour < 12) timeOfDay = 'morning';
  else if (hour < 18) timeOfDay = 'afternoon';
  else if (hour < 22) timeOfDay = 'evening';
  else timeOfDay = 'night';

  // Extract recent topics from candidates
  const recentTopics = [...new Set(recentCandidates.flatMap((c) => c.tags))].slice(0, 5);

  // Find most common recent topic
  const topicCounts = new Map<string, number>();
  recentCandidates.forEach((c) => {
    c.tags.forEach((t) => topicCounts.set(t, (topicCounts.get(t) || 0) + 1));
  });
  const currentFocus = [...topicCounts.entries()].sort((a, b) => b[1] - a[1])[0]?.[0];

  return {
    time: {
      hour,
      dayOfWeek: now.getDay(),
      isWeekend: now.getDay() === 0 || now.getDay() === 6,
      timeOfDay,
    },
    activity: {
      recentTopics,
      currentFocus,
      sessionDuration: recentCandidates.length > 0
        ? Date.now() - recentCandidates[0].context.timestamp
        : 0,
    },
    environment: {
      url: window.location.href,
    },
  };
}

// ============================================================================
// AMBIENT CAPTURE MANAGER
// ============================================================================

export class AmbientCaptureManager {
  private voiceService: VoiceCaptureService;
  private candidates: ThoughtCandidate[] = [];
  private state: AmbientState;
  private stateCallbacks: ((state: AmbientState) => void)[] = [];
  private candidateCallbacks: ((candidate: ThoughtCandidate) => void)[] = [];

  constructor() {
    this.voiceService = new VoiceCaptureService();
    this.state = {
      isListening: false,
      isProcessing: false,
      currentTranscript: '',
      candidates: [],
      captureMode: 'paused',
      insightThreshold: 0.3,
    };
  }

  get voiceSupported(): boolean {
    return this.voiceService.supported;
  }

  initialize(): void {
    this.voiceService.initialize();

    this.voiceService.onSegment((segment) => {
      this.updateState({ currentTranscript: segment.text });
    });

    this.voiceService.onInsight((candidate) => {
      if (candidate.insightScore >= this.state.insightThreshold) {
        this.addCandidate(candidate);
      }
    });
  }

  startListening(): void {
    this.voiceService.start();
    this.updateState({ isListening: true, captureMode: 'continuous' });
  }

  stopListening(): void {
    this.voiceService.stop();
    this.updateState({ isListening: false, captureMode: 'paused' });
  }

  setThreshold(threshold: number): void {
    this.updateState({ insightThreshold: Math.max(0.1, Math.min(1.0, threshold)) });
  }

  addCandidate(candidate: ThoughtCandidate): void {
    this.candidates.unshift(candidate);
    this.updateState({ candidates: [...this.candidates] });
    this.candidateCallbacks.forEach((cb) => cb(candidate));
  }

  approveCandidate(index: number): ThoughtCandidate | null {
    if (index < 0 || index >= this.candidates.length) return null;
    this.candidates[index].approved = true;
    this.updateState({ candidates: [...this.candidates] });
    return this.candidates[index];
  }

  dismissCandidate(index: number): void {
    this.candidates.splice(index, 1);
    this.updateState({ candidates: [...this.candidates] });
  }

  clearCandidates(): void {
    this.candidates = [];
    this.updateState({ candidates: [] });
  }

  onStateChange(callback: (state: AmbientState) => void): void {
    this.stateCallbacks.push(callback);
  }

  onCandidate(callback: (candidate: ThoughtCandidate) => void): void {
    this.candidateCallbacks.push(callback);
  }

  getState(): AmbientState {
    return { ...this.state };
  }

  private updateState(partial: Partial<AmbientState>): void {
    this.state = { ...this.state, ...partial };
    this.stateCallbacks.forEach((cb) => cb(this.state));
  }

  /**
   * Convert approved candidate to thought input
   */
  candidateToThoughtInput(candidate: ThoughtCandidate): CreateThoughtInput {
    return {
      content: candidate.content,
      thought_type: candidate.suggestedType,
      confidence: candidate.context.confidence,
      tags: candidate.tags,
      epistemic: {
        empirical: EmpiricalLevel.E2,
        normative: NormativeLevel.N0,
        materiality: MaterialityLevel.M2,
        harmonic: HarmonicLevel.H2,
      },
    };
  }
}

// Singleton instance
export const ambientCapture = new AmbientCaptureManager();
