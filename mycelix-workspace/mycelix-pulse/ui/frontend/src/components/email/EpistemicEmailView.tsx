// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Epistemic Email View
 *
 * Enhanced email view with integrated epistemic features:
 * - Sidebar with AI insights, claims, and trust information
 * - Quick actions for trust management
 * - Keyboard shortcuts for power users
 * - Contact profile integration
 *
 * Wraps the existing EmailView with epistemic enhancements.
 */

import { useState, useEffect, useCallback } from 'react';
import { useQuery } from '@tanstack/react-query';
import { api } from '@/services/api';
import { useAuthStore } from '@/store/authStore';
import EmailView from '../EmailView';
import { EpistemicInsightsPanel } from '../epistemic';
import ContactProfile from '../contacts/ContactProfile';
import ThreadSummaryPanel from '../thread/ThreadSummaryPanel';
import type { Email } from '@/types';

interface EpistemicEmailViewProps {
  emailId: string;
  onBack?: () => void;
  onReply?: (email: Email) => void;
  onReplyAll?: (email: Email) => void;
  onForward?: (email: Email) => void;
  onDeleteAndNext?: () => void;
  onNextEmail?: () => void;
  onPreviousEmail?: () => void;
}

type SidebarView = 'insights' | 'contact' | 'thread' | null;

export default function EpistemicEmailView({
  emailId,
  onBack,
  onReply,
  onReplyAll,
  onForward,
  onDeleteAndNext,
  onNextEmail,
  onPreviousEmail,
}: EpistemicEmailViewProps) {
  const [sidebarView, setSidebarView] = useState<SidebarView>('insights');
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const user = useAuthStore((state) => state.user);
  const userDid = user?.did || 'did:mycelix:self';

  // Fetch email data
  const { data: email } = useQuery({
    queryKey: ['email', emailId],
    queryFn: () => api.getEmail(emailId),
  });

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Ignore if typing in input
      if (
        e.target instanceof HTMLInputElement ||
        e.target instanceof HTMLTextAreaElement
      ) {
        return;
      }

      // Alt+I: Toggle insights sidebar
      if (e.altKey && e.key === 'i') {
        e.preventDefault();
        setSidebarView(sidebarView === 'insights' ? null : 'insights');
        setSidebarCollapsed(false);
      }

      // Alt+C: Toggle contact sidebar
      if (e.altKey && e.key === 'c') {
        e.preventDefault();
        setSidebarView(sidebarView === 'contact' ? null : 'contact');
        setSidebarCollapsed(false);
      }

      // Alt+T: Toggle thread summary sidebar
      if (e.altKey && e.key === 't') {
        e.preventDefault();
        setSidebarView(sidebarView === 'thread' ? null : 'thread');
        setSidebarCollapsed(false);
      }

      // Escape: Close sidebar
      if (e.key === 'Escape' && sidebarView) {
        e.preventDefault();
        setSidebarView(null);
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [sidebarView]);

  const handleViewContact = useCallback((did: string) => {
    setSidebarView('contact');
    setSidebarCollapsed(false);
  }, []);

  const handleViewInGraph = useCallback(() => {
    // This would open the trust graph visualization
    console.log('View in graph:', email?.from?.address);
  }, [email?.from?.address]);

  // Extract sender info
  const senderEmail = email?.from?.address || '';
  const senderName = email?.from?.name;
  const senderDid = (email as any)?.senderDid || `did:mycelix:${senderEmail.split('@')[0]}`;

  return (
    <div className="h-full flex">
      {/* Main Email View */}
      <div className={`flex-1 min-w-0 ${sidebarView && !sidebarCollapsed ? 'mr-80' : ''}`}>
        <EmailView
          emailId={emailId}
          onBack={onBack}
          onReply={onReply}
          onReplyAll={onReplyAll}
          onForward={onForward}
          onDeleteAndNext={onDeleteAndNext}
          onNextEmail={onNextEmail}
          onPreviousEmail={onPreviousEmail}
        />
      </div>

      {/* Sidebar Toggle Buttons (when collapsed) */}
      {(sidebarCollapsed || !sidebarView) && (
        <div className="fixed right-4 top-1/2 -translate-y-1/2 flex flex-col gap-2 z-20">
          <button
            onClick={() => {
              setSidebarView('insights');
              setSidebarCollapsed(false);
            }}
            className={`p-2 rounded-lg shadow-lg transition-colors ${
              sidebarView === 'insights'
                ? 'bg-blue-600 text-white'
                : 'bg-white dark:bg-gray-800 text-gray-600 dark:text-gray-400 hover:bg-gray-50 dark:hover:bg-gray-700'
            }`}
            title="AI Insights (Alt+I)"
          >
            <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
            </svg>
          </button>
          <button
            onClick={() => {
              setSidebarView('contact');
              setSidebarCollapsed(false);
            }}
            className={`p-2 rounded-lg shadow-lg transition-colors ${
              sidebarView === 'contact'
                ? 'bg-blue-600 text-white'
                : 'bg-white dark:bg-gray-800 text-gray-600 dark:text-gray-400 hover:bg-gray-50 dark:hover:bg-gray-700'
            }`}
            title="Contact Profile (Alt+C)"
          >
            <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
            </svg>
          </button>
          <button
            onClick={() => {
              setSidebarView('thread');
              setSidebarCollapsed(false);
            }}
            className={`p-2 rounded-lg shadow-lg transition-colors ${
              sidebarView === 'thread'
                ? 'bg-blue-600 text-white'
                : 'bg-white dark:bg-gray-800 text-gray-600 dark:text-gray-400 hover:bg-gray-50 dark:hover:bg-gray-700'
            }`}
            title="Thread Summary (Alt+T)"
          >
            <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z" />
            </svg>
          </button>
        </div>
      )}

      {/* Sidebar */}
      {sidebarView && !sidebarCollapsed && (
        <div className="fixed right-0 top-0 bottom-0 w-80 bg-white dark:bg-gray-900 border-l border-gray-200 dark:border-gray-700 shadow-xl z-30 flex flex-col">
          {/* Sidebar Header */}
          <div className="flex items-center justify-between px-4 py-3 border-b border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-800">
            <div className="flex items-center gap-1">
              <button
                onClick={() => setSidebarView('insights')}
                className={`px-2 py-1 text-xs font-medium rounded transition-colors ${
                  sidebarView === 'insights'
                    ? 'bg-blue-100 dark:bg-blue-900/40 text-blue-700 dark:text-blue-300'
                    : 'text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300'
                }`}
              >
                Insights
              </button>
              <button
                onClick={() => setSidebarView('contact')}
                className={`px-2 py-1 text-xs font-medium rounded transition-colors ${
                  sidebarView === 'contact'
                    ? 'bg-blue-100 dark:bg-blue-900/40 text-blue-700 dark:text-blue-300'
                    : 'text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300'
                }`}
              >
                Contact
              </button>
              <button
                onClick={() => setSidebarView('thread')}
                className={`px-2 py-1 text-xs font-medium rounded transition-colors ${
                  sidebarView === 'thread'
                    ? 'bg-blue-100 dark:bg-blue-900/40 text-blue-700 dark:text-blue-300'
                    : 'text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300'
                }`}
              >
                Thread
              </button>
            </div>
            <button
              onClick={() => setSidebarCollapsed(true)}
              className="p-1 rounded hover:bg-gray-200 dark:hover:bg-gray-700 transition-colors"
              title="Collapse sidebar"
            >
              <svg className="w-4 h-4 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
              </svg>
            </button>
          </div>

          {/* Sidebar Content */}
          <div className="flex-1 overflow-hidden">
            {sidebarView === 'insights' && email && (
              <EpistemicInsightsPanel
                email={email}
                userDid={userDid}
                onViewContact={handleViewContact}
              />
            )}

            {sidebarView === 'contact' && (
              <ContactProfile
                contactDid={senderDid}
                contactEmail={senderEmail}
                contactName={senderName}
                userDid={userDid}
                onViewInGraph={handleViewInGraph}
              />
            )}

            {sidebarView === 'thread' && email && (
              <ThreadSummaryPanel
                threadId={(email as any).threadId || emailId}
                emails={[email]}
                userDid={userDid}
                onSelectEmail={(id) => console.log('Select email:', id)}
                onViewContact={handleViewContact}
              />
            )}
          </div>

          {/* Keyboard Shortcuts Help */}
          <div className="px-4 py-2 border-t border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-800">
            <p className="text-xs text-gray-500 dark:text-gray-400 text-center">
              Alt+I/C/T to switch • Esc to close
            </p>
          </div>
        </div>
      )}
    </div>
  );
}
