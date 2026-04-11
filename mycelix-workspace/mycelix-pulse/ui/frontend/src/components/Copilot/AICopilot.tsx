// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Track S: AI Copilot & Summarization
 *
 * Thread summarization, action extraction, smart replies,
 * meeting extraction, tone analysis, and translation.
 */

import React, { useState, useCallback, useEffect } from 'react';

// ============================================================================
// Types
// ============================================================================

interface ThreadSummary {
  threadId: string;
  summary: string;
  keyPoints: string[];
  participants: ParticipantSummary[];
  timeline: TimelineEvent[];
  wordCountOriginal: number;
  wordCountSummary: number;
  compressionRatio: number;
  generatedAt: string;
}

interface ParticipantSummary {
  email: string;
  name?: string;
  messageCount: number;
  sentiment: SentimentScore;
  keyContributions: string[];
}

interface TimelineEvent {
  timestamp: string;
  eventType: string;
  description: string;
  messageId?: string;
}

interface SentimentScore {
  overall: number;
  urgency: number;
  formality: number;
  positivity: number;
}

interface ActionItem {
  id: string;
  emailId: string;
  description: string;
  assignee?: string;
  deadline?: string;
  priority: 'Critical' | 'High' | 'Medium' | 'Low';
  status: 'Pending' | 'InProgress' | 'Completed' | 'Dismissed';
  confidence: number;
  sourceText: string;
}

interface SmartReply {
  id: string;
  replyType: string;
  content: string;
  tone: string;
  confidence: number;
  personalizationScore: number;
}

interface MeetingExtraction {
  id: string;
  title?: string;
  proposedTimes: ProposedTime[];
  location?: MeetingLocation;
  attendees: string[];
  agenda?: string;
  durationMinutes?: number;
  confidence: number;
}

interface ProposedTime {
  start: string;
  end?: string;
  timezone?: string;
  isFlexible: boolean;
  sourceText: string;
}

interface MeetingLocation {
  type: 'Physical' | 'Virtual' | 'Hybrid' | 'ToBeDetermined';
  platform?: string;
  address?: string;
  link?: string;
}

interface ToneAnalysis {
  detectedTone: string;
  suggestedTone?: string;
  issues: ToneIssue[];
  improvements: ToneImprovement[];
  readabilityScore: number;
  clarityScore: number;
}

interface ToneIssue {
  issueType: string;
  description: string;
  severity: 'Critical' | 'Warning' | 'Suggestion';
}

interface ToneImprovement {
  original: string;
  suggestion: string;
  reason: string;
}

interface Translation {
  sourceLanguage: string;
  targetLanguage: string;
  translatedText: string;
  culturalNotes: CulturalNote[];
  confidence: number;
}

interface CulturalNote {
  noteType: string;
  description: string;
  suggestion?: string;
}

// ============================================================================
// Hooks
// ============================================================================

function useSummarization() {
  const [loading, setLoading] = useState(false);
  const [summary, setSummary] = useState<ThreadSummary | null>(null);

  const summarizeThread = useCallback(async (threadId: string) => {
    setLoading(true);
    try {
      const response = await fetch(`/api/copilot/summarize/thread/${threadId}`, {
        method: 'POST',
      });
      const data = await response.json();
      setSummary(data);
      return data;
    } finally {
      setLoading(false);
    }
  }, []);

  const summarizeEmail = useCallback(async (emailId: string) => {
    setLoading(true);
    try {
      const response = await fetch(`/api/copilot/summarize/email/${emailId}`, {
        method: 'POST',
      });
      return await response.text();
    } finally {
      setLoading(false);
    }
  }, []);

  return { summarizeThread, summarizeEmail, summary, loading };
}

function useActionExtraction() {
  const [actions, setActions] = useState<ActionItem[]>([]);
  const [loading, setLoading] = useState(false);

  const extractActions = useCallback(async (emailId: string) => {
    setLoading(true);
    try {
      const response = await fetch(`/api/copilot/actions/extract/${emailId}`, {
        method: 'POST',
      });
      const data = await response.json();
      setActions(data);
      return data;
    } finally {
      setLoading(false);
    }
  }, []);

  const updateActionStatus = useCallback(async (actionId: string, status: ActionItem['status']) => {
    await fetch(`/api/copilot/actions/${actionId}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ status }),
    });
    setActions(prev => prev.map(a => a.id === actionId ? { ...a, status } : a));
  }, []);

  return { actions, extractActions, updateActionStatus, loading };
}

function useSmartReplies() {
  const [replies, setReplies] = useState<SmartReply[]>([]);
  const [loading, setLoading] = useState(false);

  const generateReplies = useCallback(async (emailId: string, count: number = 3) => {
    setLoading(true);
    try {
      const response = await fetch(`/api/copilot/replies/${emailId}?count=${count}`, {
        method: 'POST',
      });
      const data = await response.json();
      setReplies(data);
      return data;
    } finally {
      setLoading(false);
    }
  }, []);

  return { replies, generateReplies, loading };
}

function useMeetingExtraction() {
  const [meeting, setMeeting] = useState<MeetingExtraction | null>(null);
  const [loading, setLoading] = useState(false);

  const extractMeeting = useCallback(async (emailId: string) => {
    setLoading(true);
    try {
      const response = await fetch(`/api/copilot/meeting/extract/${emailId}`, {
        method: 'POST',
      });
      const data = await response.json();
      setMeeting(data);
      return data;
    } finally {
      setLoading(false);
    }
  }, []);

  const createCalendarEvent = useCallback(async (meetingId: string) => {
    const response = await fetch(`/api/copilot/meeting/${meetingId}/calendar`, {
      method: 'POST',
    });
    return await response.json();
  }, []);

  return { meeting, extractMeeting, createCalendarEvent, loading };
}

function useToneAnalysis() {
  const [analysis, setAnalysis] = useState<ToneAnalysis | null>(null);
  const [loading, setLoading] = useState(false);

  const analyzeTone = useCallback(async (content: string) => {
    setLoading(true);
    try {
      const response = await fetch('/api/copilot/tone/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ content }),
      });
      const data = await response.json();
      setAnalysis(data);
      return data;
    } finally {
      setLoading(false);
    }
  }, []);

  return { analysis, analyzeTone, loading };
}

function useTranslation() {
  const [translation, setTranslation] = useState<Translation | null>(null);
  const [loading, setLoading] = useState(false);
  const [languages, setLanguages] = useState<{ code: string; name: string }[]>([]);

  useEffect(() => {
    fetch('/api/copilot/translate/languages')
      .then(res => res.json())
      .then(setLanguages);
  }, []);

  const translate = useCallback(async (emailId: string, targetLanguage: string) => {
    setLoading(true);
    try {
      const response = await fetch(`/api/copilot/translate/${emailId}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ targetLanguage }),
      });
      const data = await response.json();
      setTranslation(data);
      return data;
    } finally {
      setLoading(false);
    }
  }, []);

  return { translation, translate, languages, loading };
}

// ============================================================================
// Components
// ============================================================================

interface ThreadSummaryPanelProps {
  threadId: string;
  onClose: () => void;
}

export function ThreadSummaryPanel({ threadId, onClose }: ThreadSummaryPanelProps) {
  const { summarizeThread, summary, loading } = useSummarization();

  useEffect(() => {
    summarizeThread(threadId);
  }, [threadId, summarizeThread]);

  if (loading) {
    return (
      <div className="copilot-panel loading">
        <div className="spinner" />
        <p>Generating summary...</p>
      </div>
    );
  }

  if (!summary) return null;

  return (
    <div className="copilot-panel thread-summary">
      <header>
        <h3>Thread Summary</h3>
        <button onClick={onClose} className="close-btn">×</button>
      </header>

      <div className="summary-stats">
        <span>{summary.wordCountOriginal} words → {summary.wordCountSummary} words</span>
        <span className="compression">
          {Math.round((1 - summary.compressionRatio) * 100)}% shorter
        </span>
      </div>

      <div className="summary-content">
        <p>{summary.summary}</p>
      </div>

      {summary.keyPoints.length > 0 && (
        <div className="key-points">
          <h4>Key Points</h4>
          <ul>
            {summary.keyPoints.map((point, i) => (
              <li key={i}>{point}</li>
            ))}
          </ul>
        </div>
      )}

      {summary.participants.length > 0 && (
        <div className="participants">
          <h4>Participants</h4>
          {summary.participants.map(p => (
            <div key={p.email} className="participant">
              <span className="name">{p.name || p.email}</span>
              <span className="count">{p.messageCount} messages</span>
              <SentimentBadge score={p.sentiment} />
            </div>
          ))}
        </div>
      )}

      {summary.timeline.length > 0 && (
        <div className="timeline">
          <h4>Timeline</h4>
          {summary.timeline.map((event, i) => (
            <div key={i} className={`timeline-event ${event.eventType.toLowerCase()}`}>
              <span className="time">
                {new Date(event.timestamp).toLocaleDateString()}
              </span>
              <span className="description">{event.description}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function SentimentBadge({ score }: { score: SentimentScore }) {
  const sentiment = score.overall > 0.3 ? 'positive' :
                    score.overall < -0.3 ? 'negative' : 'neutral';

  return (
    <span className={`sentiment-badge ${sentiment}`}>
      {sentiment}
      {score.urgency > 0.7 && ' · urgent'}
    </span>
  );
}

interface ActionItemsPanelProps {
  emailId: string;
}

export function ActionItemsPanel({ emailId }: ActionItemsPanelProps) {
  const { actions, extractActions, updateActionStatus, loading } = useActionExtraction();

  useEffect(() => {
    extractActions(emailId);
  }, [emailId, extractActions]);

  const priorityColors: Record<ActionItem['priority'], string> = {
    Critical: '#dc2626',
    High: '#ea580c',
    Medium: '#ca8a04',
    Low: '#65a30d',
  };

  return (
    <div className="copilot-panel action-items">
      <header>
        <h3>Action Items</h3>
        <span className="count">{actions.length} detected</span>
      </header>

      {loading ? (
        <div className="loading">Extracting actions...</div>
      ) : (
        <div className="actions-list">
          {actions.map(action => (
            <div
              key={action.id}
              className={`action-item ${action.status.toLowerCase()}`}
            >
              <div className="action-header">
                <span
                  className="priority"
                  style={{ backgroundColor: priorityColors[action.priority] }}
                >
                  {action.priority}
                </span>
                <span className="confidence">
                  {Math.round(action.confidence * 100)}% confidence
                </span>
              </div>

              <p className="description">{action.description}</p>

              {action.assignee && (
                <span className="assignee">Assigned to: {action.assignee}</span>
              )}

              {action.deadline && (
                <span className="deadline">
                  Due: {new Date(action.deadline).toLocaleDateString()}
                </span>
              )}

              <blockquote className="source">{action.sourceText}</blockquote>

              <div className="action-buttons">
                {action.status === 'Pending' && (
                  <>
                    <button onClick={() => updateActionStatus(action.id, 'InProgress')}>
                      Start
                    </button>
                    <button onClick={() => updateActionStatus(action.id, 'Dismissed')}>
                      Dismiss
                    </button>
                  </>
                )}
                {action.status === 'InProgress' && (
                  <button onClick={() => updateActionStatus(action.id, 'Completed')}>
                    Complete
                  </button>
                )}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

interface SmartReplyPanelProps {
  emailId: string;
  onSelectReply: (content: string) => void;
}

export function SmartReplyPanel({ emailId, onSelectReply }: SmartReplyPanelProps) {
  const { replies, generateReplies, loading } = useSmartReplies();
  const [selectedTone, setSelectedTone] = useState<string>('Professional');

  useEffect(() => {
    generateReplies(emailId);
  }, [emailId, generateReplies]);

  const toneIcons: Record<string, string> = {
    Formal: '🎩',
    Professional: '💼',
    Friendly: '😊',
    Casual: '👋',
    Urgent: '⚡',
    Apologetic: '🙏',
    Grateful: '🙏',
  };

  return (
    <div className="copilot-panel smart-replies">
      <header>
        <h3>Smart Replies</h3>
        <select value={selectedTone} onChange={e => setSelectedTone(e.target.value)}>
          {Object.keys(toneIcons).map(tone => (
            <option key={tone} value={tone}>{toneIcons[tone]} {tone}</option>
          ))}
        </select>
      </header>

      {loading ? (
        <div className="loading">Generating replies...</div>
      ) : (
        <div className="replies-list">
          {replies.map(reply => (
            <div
              key={reply.id}
              className="reply-option"
              onClick={() => onSelectReply(reply.content)}
            >
              <div className="reply-header">
                <span className="type">{reply.replyType}</span>
                <span className="tone">{toneIcons[reply.tone]} {reply.tone}</span>
              </div>
              <p className="content">{reply.content}</p>
              <div className="reply-footer">
                <span className="confidence">
                  {Math.round(reply.confidence * 100)}% match
                </span>
                <span className="personalization">
                  {Math.round(reply.personalizationScore * 100)}% personalized
                </span>
              </div>
            </div>
          ))}
        </div>
      )}

      <button
        className="regenerate-btn"
        onClick={() => generateReplies(emailId, 5)}
        disabled={loading}
      >
        Generate More
      </button>
    </div>
  );
}

interface MeetingExtractionPanelProps {
  emailId: string;
}

export function MeetingExtractionPanel({ emailId }: MeetingExtractionPanelProps) {
  const { meeting, extractMeeting, createCalendarEvent, loading } = useMeetingExtraction();
  const [creating, setCreating] = useState(false);

  useEffect(() => {
    extractMeeting(emailId);
  }, [emailId, extractMeeting]);

  const handleCreateEvent = async () => {
    if (!meeting) return;
    setCreating(true);
    try {
      await createCalendarEvent(meeting.id);
      alert('Calendar event created!');
    } finally {
      setCreating(false);
    }
  };

  if (loading) {
    return <div className="copilot-panel loading">Analyzing email for meeting details...</div>;
  }

  if (!meeting) {
    return (
      <div className="copilot-panel no-meeting">
        <p>No meeting details detected in this email.</p>
      </div>
    );
  }

  return (
    <div className="copilot-panel meeting-extraction">
      <header>
        <h3>Meeting Detected</h3>
        <span className="confidence">
          {Math.round(meeting.confidence * 100)}% confidence
        </span>
      </header>

      {meeting.title && (
        <div className="field">
          <label>Title</label>
          <span>{meeting.title}</span>
        </div>
      )}

      {meeting.proposedTimes.length > 0 && (
        <div className="proposed-times">
          <label>Proposed Times</label>
          {meeting.proposedTimes.map((time, i) => (
            <div key={i} className="time-option">
              <span className="datetime">
                {new Date(time.start).toLocaleString()}
              </span>
              {time.isFlexible && <span className="flexible">Flexible</span>}
              <span className="source">{time.sourceText}</span>
            </div>
          ))}
        </div>
      )}

      {meeting.location && (
        <div className="field">
          <label>Location</label>
          <span>
            {meeting.location.type === 'Virtual' ? (
              <>
                {meeting.location.platform}
                {meeting.location.link && (
                  <a href={meeting.location.link} target="_blank" rel="noopener noreferrer">
                    Join
                  </a>
                )}
              </>
            ) : meeting.location.type === 'Physical' ? (
              meeting.location.address
            ) : (
              'To be determined'
            )}
          </span>
        </div>
      )}

      {meeting.attendees.length > 0 && (
        <div className="field">
          <label>Attendees</label>
          <div className="attendees">
            {meeting.attendees.map(email => (
              <span key={email} className="attendee">{email}</span>
            ))}
          </div>
        </div>
      )}

      {meeting.agenda && (
        <div className="field">
          <label>Agenda</label>
          <p>{meeting.agenda}</p>
        </div>
      )}

      {meeting.durationMinutes && (
        <div className="field">
          <label>Duration</label>
          <span>{meeting.durationMinutes} minutes</span>
        </div>
      )}

      <button
        className="create-event-btn"
        onClick={handleCreateEvent}
        disabled={creating}
      >
        {creating ? 'Creating...' : 'Add to Calendar'}
      </button>
    </div>
  );
}

interface ToneAnalysisPanelProps {
  content: string;
  onChange: (content: string) => void;
}

export function ToneAnalysisPanel({ content, onChange }: ToneAnalysisPanelProps) {
  const { analysis, analyzeTone, loading } = useToneAnalysis();

  useEffect(() => {
    if (content.length > 50) {
      const timer = setTimeout(() => analyzeTone(content), 500);
      return () => clearTimeout(timer);
    }
  }, [content, analyzeTone]);

  const applyImprovement = (improvement: ToneImprovement) => {
    onChange(content.replace(improvement.original, improvement.suggestion));
  };

  if (!analysis) return null;

  return (
    <div className="copilot-panel tone-analysis">
      <header>
        <h3>Tone Analysis</h3>
        {loading && <span className="analyzing">Analyzing...</span>}
      </header>

      <div className="scores">
        <div className="score">
          <label>Detected Tone</label>
          <span className="value">{analysis.detectedTone}</span>
        </div>
        <div className="score">
          <label>Readability</label>
          <div className="bar">
            <div
              className="fill"
              style={{ width: `${analysis.readabilityScore * 100}%` }}
            />
          </div>
        </div>
        <div className="score">
          <label>Clarity</label>
          <div className="bar">
            <div
              className="fill"
              style={{ width: `${analysis.clarityScore * 100}%` }}
            />
          </div>
        </div>
      </div>

      {analysis.issues.length > 0 && (
        <div className="issues">
          <h4>Issues Found</h4>
          {analysis.issues.map((issue, i) => (
            <div key={i} className={`issue ${issue.severity.toLowerCase()}`}>
              <span className="type">{issue.issueType}</span>
              <p>{issue.description}</p>
            </div>
          ))}
        </div>
      )}

      {analysis.improvements.length > 0 && (
        <div className="improvements">
          <h4>Suggested Improvements</h4>
          {analysis.improvements.map((imp, i) => (
            <div key={i} className="improvement">
              <div className="comparison">
                <span className="original">{imp.original}</span>
                <span className="arrow">→</span>
                <span className="suggestion">{imp.suggestion}</span>
              </div>
              <p className="reason">{imp.reason}</p>
              <button onClick={() => applyImprovement(imp)}>Apply</button>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

interface TranslationPanelProps {
  emailId: string;
}

export function TranslationPanel({ emailId }: TranslationPanelProps) {
  const { translation, translate, languages, loading } = useTranslation();
  const [targetLang, setTargetLang] = useState('en');

  const handleTranslate = () => {
    translate(emailId, targetLang);
  };

  return (
    <div className="copilot-panel translation">
      <header>
        <h3>Translation</h3>
      </header>

      <div className="language-selector">
        <label>Translate to:</label>
        <select value={targetLang} onChange={e => setTargetLang(e.target.value)}>
          {languages.map(lang => (
            <option key={lang.code} value={lang.code}>{lang.name}</option>
          ))}
        </select>
        <button onClick={handleTranslate} disabled={loading}>
          {loading ? 'Translating...' : 'Translate'}
        </button>
      </div>

      {translation && (
        <>
          <div className="translation-info">
            <span>{translation.sourceLanguage} → {translation.targetLanguage}</span>
            <span className="confidence">
              {Math.round(translation.confidence * 100)}% confidence
            </span>
          </div>

          <div className="translated-text">
            {translation.translatedText}
          </div>

          {translation.culturalNotes.length > 0 && (
            <div className="cultural-notes">
              <h4>Cultural Notes</h4>
              {translation.culturalNotes.map((note, i) => (
                <div key={i} className="note">
                  <span className="type">{note.noteType}</span>
                  <p>{note.description}</p>
                  {note.suggestion && <p className="suggestion">{note.suggestion}</p>}
                </div>
              ))}
            </div>
          )}
        </>
      )}
    </div>
  );
}

// ============================================================================
// Main Copilot Component
// ============================================================================

interface CopilotProps {
  emailId?: string;
  threadId?: string;
  draftContent?: string;
  onUpdateDraft?: (content: string) => void;
}

export function AICopilot({ emailId, threadId, draftContent, onUpdateDraft }: CopilotProps) {
  const [activeTab, setActiveTab] = useState<string>('summary');

  const tabs = [
    { id: 'summary', label: 'Summary', icon: '📋' },
    { id: 'actions', label: 'Actions', icon: '✅' },
    { id: 'replies', label: 'Smart Reply', icon: '💬' },
    { id: 'meeting', label: 'Meeting', icon: '📅' },
    { id: 'tone', label: 'Tone', icon: '🎭' },
    { id: 'translate', label: 'Translate', icon: '🌐' },
  ];

  return (
    <div className="ai-copilot">
      <nav className="copilot-tabs">
        {tabs.map(tab => (
          <button
            key={tab.id}
            className={activeTab === tab.id ? 'active' : ''}
            onClick={() => setActiveTab(tab.id)}
          >
            <span className="icon">{tab.icon}</span>
            <span className="label">{tab.label}</span>
          </button>
        ))}
      </nav>

      <div className="copilot-content">
        {activeTab === 'summary' && threadId && (
          <ThreadSummaryPanel threadId={threadId} onClose={() => {}} />
        )}
        {activeTab === 'actions' && emailId && (
          <ActionItemsPanel emailId={emailId} />
        )}
        {activeTab === 'replies' && emailId && (
          <SmartReplyPanel
            emailId={emailId}
            onSelectReply={(content) => onUpdateDraft?.(content)}
          />
        )}
        {activeTab === 'meeting' && emailId && (
          <MeetingExtractionPanel emailId={emailId} />
        )}
        {activeTab === 'tone' && draftContent && (
          <ToneAnalysisPanel
            content={draftContent}
            onChange={(content) => onUpdateDraft?.(content)}
          />
        )}
        {activeTab === 'translate' && emailId && (
          <TranslationPanel emailId={emailId} />
        )}
      </div>
    </div>
  );
}

export default AICopilot;
