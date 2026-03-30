// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
import { useState, useRef, useEffect, useCallback } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { useNavigate } from 'react-router-dom';
import { api } from '@/services/api';
import { useAuthStore } from '@/store/authStore';
import { useSnoozeStore, getSnoozeDate } from '@/store/snoozeStore';
import { useLabelStore } from '@/store/labelStore';
import { toast } from '@/store/toastStore';
import { useActionHistoryStore } from '@/store/actionHistoryStore';
import { useLayout } from '@/hooks/useLayout';
import { useSmartActions } from '@/hooks/useSmartActions';
import Layout from '@/components/Layout';
import EmailList, { type EmailListRef } from '@/components/EmailList';
import EmailView from '@/components/EmailView';
import FolderList from '@/components/FolderList';
import ComposeEmail from '@/components/ComposeEmail';
import KeyboardShortcutsHelp from '@/components/KeyboardShortcutsHelp';
import SnoozedFolderView from '@/components/SnoozedFolderView';
import SmartFolderView from '@/components/SmartFolderView';
import LabelPicker from '@/components/LabelPicker';
import { useKeyboardShortcuts, EMAIL_SHORTCUTS } from '@/hooks/useKeyboardShortcuts';
import type { Email, Folder } from '@/types';
import TrustProviderAlert from '@/components/TrustProviderAlert';

type ComposeMode = 'new' | 'reply' | 'replyAll' | 'forward';

// Virtual folder IDs
const SNOOZED_FOLDER_ID = '__snoozed__';
const SMART_FOLDERS = {
  ALL_MAIL: '__all_mail__',
  STARRED: '__starred__',
  IMPORTANT: '__important__',
  UNREAD: '__unread__',
  ATTACHMENTS: '__attachments__',
} as const;

export default function DashboardPage() {
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const user = useAuthStore((state) => state.user);
  const { snoozedEmails, snoozeEmail } = useSnoozeStore();
  const { labels } = useLabelStore();
  const { undo, redo, canUndo, canRedo } = useActionHistoryStore();
  const { config: layoutConfig } = useLayout();
  const emailListRef = useRef<EmailListRef>(null);

  const [selectedEmailId, setSelectedEmailId] = useState<string | null>(null);
  const [selectedFolderId, setSelectedFolderId] = useState<string | null>(null);
  const [isComposeOpen, setIsComposeOpen] = useState(false);
  const [composeMode, setComposeMode] = useState<ComposeMode>('new');
  const [composeOriginalEmail, setComposeOriginalEmail] = useState<Email | undefined>();
  const [showShortcutsHelp, setShowShortcutsHelp] = useState(false);
  const [showLabelPicker, setShowLabelPicker] = useState(false);
  const [labelPickerMode, setLabelPickerMode] = useState<'single' | 'bulk'>('single');

  // Clear selection when folder changes
  useEffect(() => {
    emailListRef.current?.clearSelection();
  }, [selectedFolderId]);

  const { data: accounts } = useQuery({
    queryKey: ['accounts'],
    queryFn: () => api.getAccounts(),
  });

  const { data: apiFolders } = useQuery({
    queryKey: ['folders'],
    queryFn: () => api.getFolders(),
  });

  // Get Important label for smart folder
  const importantLabel = labels.find(l => l.name === 'Important');

  // Build folders array with smart folders and virtual folders
  const folders: Folder[] = [
    ...(apiFolders || []),
    // Smart Folders
    {
      id: SMART_FOLDERS.ALL_MAIL,
      name: 'All Mail',
      type: 'CUSTOM',
      unreadCount: 0,
    } as Folder,
    {
      id: SMART_FOLDERS.STARRED,
      name: 'Starred',
      type: 'CUSTOM',
      unreadCount: 0,
    } as Folder,
    {
      id: SMART_FOLDERS.IMPORTANT,
      name: 'Important',
      type: 'CUSTOM',
      unreadCount: importantLabel?.emailCount || 0,
    } as Folder,
    {
      id: SMART_FOLDERS.UNREAD,
      name: 'Unread',
      type: 'CUSTOM',
      unreadCount: 0,
    } as Folder,
    {
      id: SMART_FOLDERS.ATTACHMENTS,
      name: 'Attachments',
      type: 'CUSTOM',
      unreadCount: 0,
    } as Folder,
    // Snoozed Folder
    {
      id: SNOOZED_FOLDER_ID,
      name: 'Snoozed',
      type: 'CUSTOM',
      unreadCount: snoozedEmails.length,
    } as Folder,
  ];

  // Check if current folder is a smart folder
  const isSmartFolder = selectedFolderId && Object.values(SMART_FOLDERS).includes(selectedFolderId as any);
  const isSnoozedFolder = selectedFolderId === SNOOZED_FOLDER_ID;
  const isVirtualFolder = isSmartFolder || isSnoozedFolder;

  // Fetch emails based on folder type
  const { data: emails } = useQuery({
    queryKey: ['emails', isSmartFolder ? null : selectedFolderId],
    queryFn: () => {
      // For smart folders, fetch all emails
      if (isSmartFolder) {
        return api.getEmails({});
      }
      // For regular folders, fetch by folder ID
      return api.getEmails({ folderId: selectedFolderId || undefined });
    },
    enabled: !!selectedFolderId && !isSnoozedFolder,
  });

  const { data: currentEmail } = useQuery({
    queryKey: ['email', selectedEmailId],
    queryFn: () => api.getEmail(selectedEmailId!),
    enabled: !!selectedEmailId,
  });

  const markStarredMutation = useMutation({
    mutationFn: ({ emailId, isStarred }: { emailId: string; isStarred: boolean }) =>
      api.markEmailStarred(emailId, isStarred),
    onSuccess: (_, variables) => {
      queryClient.invalidateQueries({ queryKey: ['emails'] });
      queryClient.invalidateQueries({ queryKey: ['email', selectedEmailId] });
      toast.success(variables.isStarred ? 'Email starred' : 'Email unstarred');
    },
    onError: () => {
      toast.error('Failed to update email');
    },
  });

  const markReadMutation = useMutation({
    mutationFn: ({ emailId, isRead }: { emailId: string; isRead: boolean }) =>
      api.markEmailRead(emailId, isRead),
    onSuccess: (_, variables) => {
      queryClient.invalidateQueries({ queryKey: ['emails'] });
      queryClient.invalidateQueries({ queryKey: ['email', selectedEmailId] });
      toast.success(variables.isRead ? 'Marked as read' : 'Marked as unread');
    },
    onError: () => {
      toast.error('Failed to update email');
    },
  });

  const deleteEmailMutation = useMutation({
    mutationFn: (emailId: string) => api.deleteEmail(emailId),
    onSuccess: () => {
      setSelectedEmailId(null);
      queryClient.invalidateQueries({ queryKey: ['emails'] });
      toast.success('Email deleted');
    },
    onError: () => {
      toast.error('Failed to delete email');
    },
  });

  // Bulk operations mutations
  const bulkMarkReadMutation = useMutation({
    mutationFn: async ({ emailIds, isRead }: { emailIds: string[]; isRead: boolean }) => {
      const results = await Promise.allSettled(
        emailIds.map((id) => api.markEmailRead(id, isRead))
      );

      const successful = results.filter((r) => r.status === 'fulfilled').length;
      const failed = results.filter((r) => r.status === 'rejected').length;

      return { successful, failed, total: emailIds.length, isRead };
    },
    onSuccess: (result) => {
      queryClient.invalidateQueries({ queryKey: ['emails'] });
      const action = result.isRead ? 'read' : 'unread';

      if (result.failed > 0) {
        toast.warning(
          `${result.successful}/${result.total} email${result.total > 1 ? 's' : ''} marked as ${action}. ${result.failed} failed.`
        );
      } else {
        toast.success(`Marked ${result.successful} email${result.successful > 1 ? 's' : ''} as ${action}`);
      }
    },
    onError: () => {
      toast.error('Failed to update emails');
    },
  });

  const bulkStarMutation = useMutation({
    mutationFn: async ({ emailIds, isStarred }: { emailIds: string[]; isStarred: boolean }) => {
      const results = await Promise.allSettled(
        emailIds.map((id) => api.markEmailStarred(id, isStarred))
      );

      const successful = results.filter((r) => r.status === 'fulfilled').length;
      const failed = results.filter((r) => r.status === 'rejected').length;

      return { successful, failed, total: emailIds.length, isStarred };
    },
    onSuccess: (result) => {
      queryClient.invalidateQueries({ queryKey: ['emails'] });
      const action = result.isStarred ? 'starred' : 'unstarred';

      if (result.failed > 0) {
        toast.warning(
          `${result.successful}/${result.total} email${result.total > 1 ? 's' : ''} ${action}. ${result.failed} failed.`
        );
      } else {
        toast.success(`${result.successful} email${result.successful > 1 ? 's' : ''} ${action}`);
      }
    },
    onError: () => {
      toast.error('Failed to update emails');
    },
  });

  const bulkDeleteMutation = useMutation({
    mutationFn: async (emailIds: string[]) => {
      const results = await Promise.allSettled(
        emailIds.map((id) => api.deleteEmail(id))
      );

      const successful = results.filter((r) => r.status === 'fulfilled').length;
      const failed = results.filter((r) => r.status === 'rejected').length;

      return { successful, failed, total: emailIds.length };
    },
    onSuccess: (result) => {
      setSelectedEmailId(null);
      queryClient.invalidateQueries({ queryKey: ['emails'] });

      if (result.failed > 0) {
        toast.warning(
          `Deleted ${result.successful}/${result.total} email${result.total > 1 ? 's' : ''}. ${result.failed} failed.`
        );
      } else {
        toast.success(`Deleted ${result.successful} email${result.successful > 1 ? 's' : ''}`);
      }
    },
    onError: () => {
      toast.error('Failed to delete emails');
    },
  });

  const defaultAccount = accounts?.find((acc) => acc.isDefault);
  const emailList = emails?.emails || [];
  const selectedEmailIndex = selectedEmailId
    ? emailList.findIndex((e) => e.id === selectedEmailId)
    : -1;

  const openCompose = useCallback((mode: ComposeMode, email?: Email) => {
    setComposeMode(mode);
    setComposeOriginalEmail(email);
    setIsComposeOpen(true);
  }, []);

  const handleReply = useCallback((email: Email) => {
    openCompose('reply', email);
  }, [openCompose]);

  const handleReplyAll = useCallback((email: Email) => {
    openCompose('replyAll', email);
  }, [openCompose]);

  const handleForward = useCallback((email: Email) => {
    openCompose('forward', email);
  }, [openCompose]);

  // Smart actions hook for "and next" patterns
  const { selectNextEmail, selectPreviousEmail } = useSmartActions({
    emails: emailList,
    currentEmailId: selectedEmailId,
    onSelectEmail: setSelectedEmailId,
  });

  // Smart action: Delete and Next
  const handleDeleteAndNext = useCallback(() => {
    if (!selectedEmailId) return;

    deleteEmailMutation.mutate(selectedEmailId, {
      onSuccess: () => {
        // Select next email after deletion
        selectNextEmail({ unreadOnly: false });
      },
    });
  }, [selectedEmailId, deleteEmailMutation, selectNextEmail]);

  // Smart action: Next Email
  const handleNextEmail = useCallback(() => {
    selectNextEmail({ unreadOnly: false });
  }, [selectNextEmail]);

  // Smart action: Previous Email
  const handlePreviousEmail = useCallback(() => {
    selectPreviousEmail({ unreadOnly: false });
  }, [selectPreviousEmail]);

  // Keyboard shortcut actions
  const shortcuts = [
    {
      key: EMAIL_SHORTCUTS.COMPOSE.key,
      description: EMAIL_SHORTCUTS.COMPOSE.description,
      action: () => openCompose('new'),
    },
    {
      key: EMAIL_SHORTCUTS.REPLY.key,
      description: EMAIL_SHORTCUTS.REPLY.description,
      action: () => {
        if (currentEmail) handleReply(currentEmail);
      },
    },
    {
      key: EMAIL_SHORTCUTS.REPLY_ALL.key,
      description: EMAIL_SHORTCUTS.REPLY_ALL.description,
      action: () => {
        if (currentEmail) handleReplyAll(currentEmail);
      },
    },
    {
      key: EMAIL_SHORTCUTS.FORWARD.key,
      description: EMAIL_SHORTCUTS.FORWARD.description,
      action: () => {
        if (currentEmail) handleForward(currentEmail);
      },
    },
    {
      key: EMAIL_SHORTCUTS.NEXT_EMAIL.key,
      description: EMAIL_SHORTCUTS.NEXT_EMAIL.description,
      action: () => {
        if (selectedEmailIndex >= 0 && selectedEmailIndex < emailList.length - 1) {
          setSelectedEmailId(emailList[selectedEmailIndex + 1].id);
        }
      },
    },
    {
      key: EMAIL_SHORTCUTS.PREV_EMAIL.key,
      description: EMAIL_SHORTCUTS.PREV_EMAIL.description,
      action: () => {
        if (selectedEmailIndex > 0) {
          setSelectedEmailId(emailList[selectedEmailIndex - 1].id);
        }
      },
    },
    {
      key: EMAIL_SHORTCUTS.SEARCH.key,
      description: EMAIL_SHORTCUTS.SEARCH.description,
      action: () => {
        emailListRef.current?.focusSearch();
      },
    },
    {
      key: EMAIL_SHORTCUTS.ESCAPE.key,
      description: EMAIL_SHORTCUTS.ESCAPE.description,
      action: () => {
        if (isComposeOpen) {
          setIsComposeOpen(false);
        } else if (showShortcutsHelp) {
          setShowShortcutsHelp(false);
        } else {
          setSelectedEmailId(null);
        }
      },
    },
    {
      key: EMAIL_SHORTCUTS.STAR.key,
      description: EMAIL_SHORTCUTS.STAR.description,
      action: () => {
        if (currentEmail) {
          markStarredMutation.mutate({
            emailId: currentEmail.id,
            isStarred: !currentEmail.isStarred,
          });
        }
      },
    },
    {
      key: EMAIL_SHORTCUTS.MARK_READ.key,
      description: EMAIL_SHORTCUTS.MARK_READ.description,
      action: () => {
        if (currentEmail) {
          markReadMutation.mutate({
            emailId: currentEmail.id,
            isRead: !currentEmail.isRead,
          });
        }
      },
    },
    {
      key: EMAIL_SHORTCUTS.DELETE.key,
      description: EMAIL_SHORTCUTS.DELETE.description,
      action: () => {
        if (selectedEmailId) {
          deleteEmailMutation.mutate(selectedEmailId);
        }
      },
    },
    {
      key: EMAIL_SHORTCUTS.HELP.key,
      description: EMAIL_SHORTCUTS.HELP.description,
      action: () => setShowShortcutsHelp(true),
    },
  ];

  useKeyboardShortcuts(shortcuts, !isComposeOpen);

  // Bulk operation keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Don't trigger if typing in an input or if compose is open
      if (
        e.target instanceof HTMLInputElement ||
        e.target instanceof HTMLTextAreaElement ||
        isComposeOpen ||
        showShortcutsHelp
      ) {
        return;
      }

      const selectedIds = emailListRef.current?.getSelectedIds();

      // Undo (Ctrl/Cmd + Z)
      if ((e.ctrlKey || e.metaKey) && e.key === 'z' && !e.shiftKey) {
        if (canUndo()) {
          e.preventDefault();
          undo().then(() => {
            toast.success('Action undone');
          }).catch((err) => {
            console.error('Failed to undo:', err);
            toast.error('Failed to undo action');
          });
        }
        return;
      }

      // Redo (Ctrl/Cmd + Shift + Z or Ctrl/Cmd + Y)
      if ((e.ctrlKey || e.metaKey) && ((e.key === 'z' && e.shiftKey) || e.key === 'y')) {
        if (canRedo()) {
          e.preventDefault();
          redo().then(() => {
            toast.success('Action redone');
          }).catch((err) => {
            console.error('Failed to redo:', err);
            toast.error('Failed to redo action');
          });
        }
        return;
      }

      // Settings (Ctrl/Cmd + ,)
      if ((e.ctrlKey || e.metaKey) && e.key === ',') {
        e.preventDefault();
        navigate('/settings');
        return;
      }

      // Select all (Ctrl/Cmd + A)
      if ((e.ctrlKey || e.metaKey) && e.key === 'a' && selectedFolderId) {
        e.preventDefault();
        emailListRef.current?.selectAll();
      }

      // Snooze current email (Z key)
      if (e.key.toLowerCase() === 'z' && selectedEmailId && !isSnoozedFolder) {
        e.preventDefault();
        // Snooze until tomorrow 9 AM
        const snoozeDate = getSnoozeDate('tomorrow');
        if (snoozeDate && selectedFolderId) {
          snoozeEmail(selectedEmailId, snoozeDate, selectedFolderId);
          setSelectedEmailId(null); // Deselect after snoozing
        }
        return;
      }

      // Label picker shortcuts
      if (e.key.toLowerCase() === 'l') {
        e.preventDefault();

        if (e.shiftKey && selectedIds && selectedIds.length > 0) {
          // Shift+L: Bulk label
          setLabelPickerMode('bulk');
          setShowLabelPicker(true);
        } else if (selectedEmailId) {
          // L: Label single email
          setLabelPickerMode('single');
          setShowLabelPicker(true);
        }
        return;
      }

      // Bulk operations (only if emails are selected)
      if (selectedIds && selectedIds.length > 0) {
        if (e.shiftKey && e.key === 'Delete') {
          e.preventDefault();
          if (confirm(`Delete ${selectedIds.length} email${selectedIds.length > 1 ? 's' : ''}?`)) {
            bulkDeleteMutation.mutate(selectedIds);
          }
        } else if (e.shiftKey && e.key.toLowerCase() === 'u') {
          e.preventDefault();
          bulkMarkReadMutation.mutate({ emailIds: selectedIds, isRead: false });
        } else if (e.shiftKey && e.key.toLowerCase() === 'r') {
          e.preventDefault();
          bulkMarkReadMutation.mutate({ emailIds: selectedIds, isRead: true });
        } else if (e.shiftKey && e.key.toLowerCase() === 's') {
          e.preventDefault();
          bulkStarMutation.mutate({ emailIds: selectedIds, isStarred: true });
        } else if (e.shiftKey && e.key.toLowerCase() === 'd') {
          e.preventDefault();
          emailListRef.current?.clearSelection();
        }
      }

      // Smart navigation shortcuts (only when viewing an email)
      if (selectedEmailId) {
        // # - Delete and next
        if (e.key === '#' || (e.shiftKey && e.key === '3')) {
          e.preventDefault();
          handleDeleteAndNext();
          return;
        }

        // ] - Next email
        if (e.key === ']') {
          e.preventDefault();
          handleNextEmail();
          return;
        }

        // [ - Previous email
        if (e.key === '[') {
          e.preventDefault();
          handlePreviousEmail();
          return;
        }
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [
    selectedFolderId,
    selectedEmailId,
    isSnoozedFolder,
    isComposeOpen,
    showShortcutsHelp,
    navigate,
    snoozeEmail,
    bulkDeleteMutation,
    bulkMarkReadMutation,
    bulkStarMutation,
    handleDeleteAndNext,
    handleNextEmail,
    handlePreviousEmail,
  ]);

  return (
    <Layout>
      <div className="flex h-[calc(100vh-64px)]">
        {/* Sidebar */}
        <div className="w-64 bg-white dark:bg-gray-800 border-r border-gray-200 dark:border-gray-700 overflow-y-auto">
          <div className="p-4">
            <button
              onClick={() => openCompose('new')}
              className="btn btn-primary w-full mb-6 flex items-center justify-center space-x-2"
            >
              <span>✏️</span>
              <span>Compose</span>
            </button>

            <h2 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-4">Folders</h2>
            <FolderList
              folders={folders || []}
              selectedFolderId={selectedFolderId}
              onSelectFolder={setSelectedFolderId}
            />
          </div>
        </div>

        {/* Main content area with dynamic layout */}
        <div className={layoutConfig.containerClass}>
          {/* Email List */}
          <div className={`bg-white dark:bg-gray-800 ${layoutConfig.listClass}`}>
            <div className="px-4 pt-3">
              <TrustProviderAlert />
            </div>
            {isSnoozedFolder ? (
              <SnoozedFolderView
                onSelectEmail={setSelectedEmailId}
                selectedEmailId={selectedEmailId}
              />
            ) : isSmartFolder ? (
              <SmartFolderView
                folderId={selectedFolderId!}
                emails={emails?.emails || []}
                onSelectEmail={setSelectedEmailId}
                selectedEmailId={selectedEmailId}
              />
            ) : (
              <EmailList
                ref={emailListRef}
                folderId={selectedFolderId}
                selectedEmailId={selectedEmailId}
                onSelectEmail={setSelectedEmailId}
                onBulkMarkRead={(emailIds, isRead) =>
                  bulkMarkReadMutation.mutate({ emailIds, isRead })
                }
                onBulkStar={(emailIds, isStarred) =>
                  bulkStarMutation.mutate({ emailIds, isStarred })
                }
                onBulkDelete={(emailIds) => bulkDeleteMutation.mutate(emailIds)}
                bulkOperationsPending={
                  bulkMarkReadMutation.isPending || bulkStarMutation.isPending || bulkDeleteMutation.isPending
                }
              />
            )}
          </div>

          {/* Email View */}
          {layoutConfig.showPreview && (
            <div className={`bg-gray-50 dark:bg-gray-900 ${layoutConfig.previewClass}`}>
              {selectedEmailId ? (
                <EmailView
                  emailId={selectedEmailId}
                  onBack={() => setSelectedEmailId(null)}
                  onReply={handleReply}
                  onReplyAll={handleReplyAll}
                  onForward={handleForward}
                  onDeleteAndNext={handleDeleteAndNext}
                  onNextEmail={handleNextEmail}
                  onPreviousEmail={handlePreviousEmail}
                />
              ) : (
                <div className="h-full flex items-center justify-center text-gray-500 dark:text-gray-400">
                  <div className="text-center">
                    <svg className="mx-auto h-12 w-12 text-gray-400 dark:text-gray-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 8l7.89 5.26a2 2 0 002.22 0L21 8M5 19h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
                    </svg>
                    <p className="mt-4 text-lg font-medium dark:text-gray-300">No email selected</p>
                    <p className="text-sm text-gray-500 dark:text-gray-400">Select an email to view its content</p>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      {/* Compose Modal */}
      {isComposeOpen && (
        <ComposeEmail
          onClose={() => {
            setIsComposeOpen(false);
            setComposeMode('new');
            setComposeOriginalEmail(undefined);
          }}
          mode={composeMode}
          originalEmail={composeOriginalEmail}
        />
      )}

      {/* Keyboard Shortcuts Help */}
      {showShortcutsHelp && <KeyboardShortcutsHelp onClose={() => setShowShortcutsHelp(false)} />}

      {/* Label Picker */}
      {showLabelPicker && selectedEmailId && (
        <LabelPicker
          emailId={selectedEmailId}
          onClose={() => setShowLabelPicker(false)}
          mode={labelPickerMode}
          emailIds={labelPickerMode === 'bulk' ? emailListRef.current?.getSelectedIds() || [] : undefined}
        />
      )}

      {/* Help Button */}
      <button
        onClick={() => setShowShortcutsHelp(true)}
        className="fixed bottom-6 right-6 w-12 h-12 bg-primary-600 text-white rounded-full shadow-lg hover:bg-primary-700 flex items-center justify-center transition-colors z-40"
        aria-label="Keyboard shortcuts help"
        title="Keyboard shortcuts (?)"
      >
        <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M8.228 9c.549-1.165 2.03-2 3.772-2 2.21 0 4 1.343 4 3 0 1.4-1.278 2.575-3.006 2.907-.542.104-.994.54-.994 1.093m0 3h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
          />
        </svg>
      </button>
    </Layout>
  );
}
