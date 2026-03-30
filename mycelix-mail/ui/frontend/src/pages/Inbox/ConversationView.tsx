// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Conversation View
 *
 * Threaded email conversation display with AI summaries
 */

import React, { useEffect, useState } from 'react';

interface ThreadMessage {
  id: string;
  fromAddress: string;
  fromName?: string;
  toAddresses: string[];
  ccAddresses: string[];
  subject: string;
  bodyText: string;
  bodyHtml?: string;
  receivedAt: string;
  isRead: boolean;
  hasAttachments: boolean;
  threadPosition: number;
}

interface Conversation {
  threadId: string;
  subject: string;
  participants: string[];
  messageCount: number;
  messages: ThreadMessage[];
  firstMessageAt: string;
  lastMessageAt: string;
  hasUnread: boolean;
}

interface ThreadSummary {
  summary: string;
  keyPoints: string[];
  actionItems: ActionItem[];
  sentiment: 'positive' | 'neutral' | 'negative';
}

interface ActionItem {
  text: string;
  assignedTo?: string;
  dueDate?: string;
  completed: boolean;
}

interface ConversationViewProps {
  threadId: string;
  onClose: () => void;
  onReply: (emailId: string) => void;
}

export default function ConversationView({ threadId, onClose, onReply }: ConversationViewProps) {
  const [conversation, setConversation] = useState<Conversation | null>(null);
  const [summary, setSummary] = useState<ThreadSummary | null>(null);
  const [expandedMessages, setExpandedMessages] = useState<Set<string>>(new Set());
  const [loading, setLoading] = useState(true);
  const [showSummary, setShowSummary] = useState(false);

  useEffect(() => {
    fetchConversation();
  }, [threadId]);

  async function fetchConversation() {
    try {
      const response = await fetch(`/api/threads/${threadId}`);
      if (response.ok) {
        const data = await response.json();
        setConversation(data);
        // Expand last message by default
        if (data.messages.length > 0) {
          setExpandedMessages(new Set([data.messages[data.messages.length - 1].id]));
        }
      }
    } catch (error) {
      console.error('Failed to fetch conversation:', error);
    } finally {
      setLoading(false);
    }
  }

  async function fetchSummary() {
    try {
      const response = await fetch(`/api/threads/${threadId}/summary`);
      if (response.ok) {
        setSummary(await response.json());
        setShowSummary(true);
      }
    } catch (error) {
      console.error('Failed to fetch summary:', error);
    }
  }

  function toggleMessage(messageId: string) {
    setExpandedMessages((prev) => {
      const next = new Set(prev);
      if (next.has(messageId)) {
        next.delete(messageId);
      } else {
        next.add(messageId);
      }
      return next;
    });
  }

  function expandAll() {
    if (conversation) {
      setExpandedMessages(new Set(conversation.messages.map((m) => m.id)));
    }
  }

  function collapseAll() {
    if (conversation?.messages.length) {
      setExpandedMessages(new Set([conversation.messages[conversation.messages.length - 1].id]));
    }
  }

  const sentimentColors = {
    positive: 'text-green-600',
    neutral: 'text-gray-600',
    negative: 'text-red-600',
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary"></div>
      </div>
    );
  }

  if (!conversation) {
    return (
      <div className="text-center py-12">
        <p className="text-muted">Conversation not found</p>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="border-b border-border p-4">
        <div className="flex items-start justify-between">
          <div>
            <h1 className="text-xl font-bold">{conversation.subject}</h1>
            <div className="flex items-center gap-2 mt-1 text-sm text-muted">
              <span>{conversation.messageCount} messages</span>
              <span>-</span>
              <span>{conversation.participants.length} participants</span>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={fetchSummary}
              className="px-3 py-1.5 text-sm border border-border rounded-lg hover:bg-muted/30"
            >
              AI Summary
            </button>
            <button
              onClick={expandAll}
              className="px-3 py-1.5 text-sm border border-border rounded-lg hover:bg-muted/30"
            >
              Expand All
            </button>
            <button
              onClick={collapseAll}
              className="px-3 py-1.5 text-sm border border-border rounded-lg hover:bg-muted/30"
            >
              Collapse
            </button>
            <button
              onClick={onClose}
              className="p-2 hover:bg-muted/30 rounded-lg"
            >
              X
            </button>
          </div>
        </div>

        {/* Participants */}
        <div className="flex flex-wrap gap-2 mt-3">
          {conversation.participants.map((participant) => (
            <span
              key={participant}
              className="px-2 py-1 bg-muted/30 rounded-full text-sm"
            >
              {participant}
            </span>
          ))}
        </div>
      </div>

      {/* AI Summary Panel */}
      {showSummary && summary && (
        <div className="border-b border-border p-4 bg-blue-50">
          <div className="flex items-center justify-between mb-3">
            <h3 className="font-semibold">AI Summary</h3>
            <button
              onClick={() => setShowSummary(false)}
              className="text-sm text-muted hover:text-foreground"
            >
              Hide
            </button>
          </div>
          <p className="text-sm mb-3">{summary.summary}</p>

          {summary.keyPoints.length > 0 && (
            <div className="mb-3">
              <h4 className="text-sm font-medium mb-1">Key Points</h4>
              <ul className="list-disc list-inside text-sm space-y-1">
                {summary.keyPoints.map((point, idx) => (
                  <li key={idx}>{point}</li>
                ))}
              </ul>
            </div>
          )}

          {summary.actionItems.length > 0 && (
            <div className="mb-3">
              <h4 className="text-sm font-medium mb-1">Action Items</h4>
              <ul className="space-y-1">
                {summary.actionItems.map((item, idx) => (
                  <li key={idx} className="flex items-center gap-2 text-sm">
                    <input type="checkbox" checked={item.completed} readOnly />
                    <span>{item.text}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}

          <div className={`text-sm ${sentimentColors[summary.sentiment]}`}>
            Sentiment: {summary.sentiment}
          </div>
        </div>
      )}

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {conversation.messages.map((message, index) => {
          const isExpanded = expandedMessages.has(message.id);
          const isLast = index === conversation.messages.length - 1;

          return (
            <div
              key={message.id}
              className={`border rounded-lg ${!message.isRead ? 'border-primary bg-primary/5' : 'border-border'}`}
            >
              {/* Message Header */}
              <div
                className="p-3 cursor-pointer hover:bg-muted/30"
                onClick={() => toggleMessage(message.id)}
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <div className="w-8 h-8 rounded-full bg-primary/20 flex items-center justify-center">
                      {(message.fromName || message.fromAddress)[0].toUpperCase()}
                    </div>
                    <div>
                      <p className="font-medium">
                        {message.fromName || message.fromAddress}
                      </p>
                      <p className="text-xs text-muted">
                        to {message.toAddresses.join(', ')}
                        {message.ccAddresses.length > 0 && (
                          <span> (cc: {message.ccAddresses.join(', ')})</span>
                        )}
                      </p>
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    {message.hasAttachments && <span>📎</span>}
                    <span className="text-sm text-muted">
                      {new Date(message.receivedAt).toLocaleString()}
                    </span>
                    <span className="text-muted">{isExpanded ? '▼' : '▶'}</span>
                  </div>
                </div>

                {/* Preview when collapsed */}
                {!isExpanded && (
                  <p className="mt-2 text-sm text-muted line-clamp-2">
                    {message.bodyText.slice(0, 200)}...
                  </p>
                )}
              </div>

              {/* Expanded Content */}
              {isExpanded && (
                <div className="border-t border-border">
                  <div className="p-4">
                    {message.bodyHtml ? (
                      <div
                        className="prose prose-sm max-w-none"
                        dangerouslySetInnerHTML={{ __html: message.bodyHtml }}
                      />
                    ) : (
                      <pre className="whitespace-pre-wrap font-sans text-sm">
                        {message.bodyText}
                      </pre>
                    )}
                  </div>

                  {/* Message Actions */}
                  <div className="border-t border-border p-3 flex gap-2">
                    <button
                      onClick={() => onReply(message.id)}
                      className="px-3 py-1.5 text-sm bg-primary text-white rounded hover:bg-primary/90"
                    >
                      Reply
                    </button>
                    <button className="px-3 py-1.5 text-sm border border-border rounded hover:bg-muted/30">
                      Reply All
                    </button>
                    <button className="px-3 py-1.5 text-sm border border-border rounded hover:bg-muted/30">
                      Forward
                    </button>
                  </div>
                </div>
              )}
            </div>
          );
        })}
      </div>

      {/* Quick Reply */}
      <div className="border-t border-border p-4">
        <div className="flex items-center gap-2">
          <input
            type="text"
            placeholder="Write a quick reply..."
            className="flex-1 px-4 py-2 border border-border rounded-lg"
            onKeyDown={(e) => {
              if (e.key === 'Enter' && conversation.messages.length > 0) {
                onReply(conversation.messages[conversation.messages.length - 1].id);
              }
            }}
          />
          <button
            onClick={() => {
              if (conversation.messages.length > 0) {
                onReply(conversation.messages[conversation.messages.length - 1].id);
              }
            }}
            className="px-4 py-2 bg-primary text-white rounded-lg hover:bg-primary/90"
          >
            Reply
          </button>
        </div>
      </div>
    </div>
  );
}

/**
 * Thread List Item component for inbox view
 */
export function ThreadListItem({
  thread,
  onClick,
  selected,
}: {
  thread: {
    threadId: string;
    subject: string;
    messageCount: number;
    participantCount: number;
    lastMessageAt: string;
    hasUnread: boolean;
    lastSender: string;
    lastSenderName?: string;
    preview: string;
  };
  onClick: () => void;
  selected: boolean;
}) {
  return (
    <div
      className={`p-4 border-b border-border cursor-pointer hover:bg-muted/30 ${
        selected ? 'bg-primary/10' : ''
      } ${thread.hasUnread ? 'bg-blue-50' : ''}`}
      onClick={onClick}
    >
      <div className="flex items-center justify-between mb-1">
        <div className="flex items-center gap-2">
          <span className={`font-medium ${thread.hasUnread ? 'text-foreground' : 'text-muted'}`}>
            {thread.lastSenderName || thread.lastSender}
          </span>
          {thread.messageCount > 1 && (
            <span className="px-1.5 py-0.5 bg-muted/50 rounded text-xs">
              {thread.messageCount}
            </span>
          )}
        </div>
        <span className="text-xs text-muted">
          {new Date(thread.lastMessageAt).toLocaleDateString()}
        </span>
      </div>
      <p className={`text-sm ${thread.hasUnread ? 'font-medium' : ''}`}>
        {thread.subject}
      </p>
      <p className="text-sm text-muted line-clamp-1 mt-1">{thread.preview}</p>
    </div>
  );
}
