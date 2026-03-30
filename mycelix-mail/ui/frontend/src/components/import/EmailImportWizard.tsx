// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Email Import Wizard
 *
 * Multi-step wizard for importing emails from external providers:
 * - Gmail (via OAuth)
 * - Outlook/Microsoft 365 (via OAuth)
 * - IMAP (generic)
 * - MBOX file import
 */

import { useState, useCallback } from 'react';
import { create } from 'zustand';

// ============================================
// Types
// ============================================

export type ImportProvider = 'gmail' | 'outlook' | 'imap' | 'mbox';

export interface ImportConfig {
  provider: ImportProvider;
  // OAuth providers
  accessToken?: string;
  refreshToken?: string;
  // IMAP
  imapHost?: string;
  imapPort?: number;
  imapUsername?: string;
  imapPassword?: string;
  imapSsl?: boolean;
  // Selection
  folders: string[];
  dateRange?: { from: string; to: string };
  includeAttachments: boolean;
  // Trust import
  importContacts: boolean;
  importTrustFromHistory: boolean;
}

export interface ImportProgress {
  status: 'idle' | 'connecting' | 'fetching' | 'importing' | 'complete' | 'error';
  currentFolder?: string;
  totalEmails: number;
  importedEmails: number;
  totalFolders: number;
  completedFolders: number;
  errors: ImportError[];
  startedAt?: string;
  completedAt?: string;
}

export interface ImportError {
  type: 'auth' | 'connection' | 'parse' | 'storage';
  message: string;
  details?: string;
  recoverable: boolean;
}

export interface FolderInfo {
  path: string;
  name: string;
  messageCount: number;
  children?: FolderInfo[];
  selected: boolean;
}

// ============================================
// Import Store
// ============================================

interface ImportState {
  config: ImportConfig;
  progress: ImportProgress;
  availableFolders: FolderInfo[];
  step: number;
}

interface ImportActions {
  setProvider: (provider: ImportProvider) => void;
  setConfig: (config: Partial<ImportConfig>) => void;
  setFolders: (folders: FolderInfo[]) => void;
  toggleFolder: (path: string) => void;
  setProgress: (progress: Partial<ImportProgress>) => void;
  nextStep: () => void;
  prevStep: () => void;
  reset: () => void;
}

export const useImportStore = create<ImportState & ImportActions>((set, get) => ({
  config: {
    provider: 'gmail',
    folders: [],
    includeAttachments: true,
    importContacts: true,
    importTrustFromHistory: true,
  },
  progress: {
    status: 'idle',
    totalEmails: 0,
    importedEmails: 0,
    totalFolders: 0,
    completedFolders: 0,
    errors: [],
  },
  availableFolders: [],
  step: 0,

  setProvider: (provider) => set((state) => ({
    config: { ...state.config, provider },
  })),

  setConfig: (config) => set((state) => ({
    config: { ...state.config, ...config },
  })),

  setFolders: (folders) => set({ availableFolders: folders }),

  toggleFolder: (path) => set((state) => {
    const updateFolder = (folders: FolderInfo[]): FolderInfo[] =>
      folders.map((f) => ({
        ...f,
        selected: f.path === path ? !f.selected : f.selected,
        children: f.children ? updateFolder(f.children) : undefined,
      }));

    const newFolders = updateFolder(state.availableFolders);
    const selectedPaths = getSelectedPaths(newFolders);

    return {
      availableFolders: newFolders,
      config: { ...state.config, folders: selectedPaths },
    };
  }),

  setProgress: (progress) => set((state) => ({
    progress: { ...state.progress, ...progress },
  })),

  nextStep: () => set((state) => ({ step: Math.min(state.step + 1, 4) })),
  prevStep: () => set((state) => ({ step: Math.max(state.step - 1, 0) })),
  reset: () => set({
    config: {
      provider: 'gmail',
      folders: [],
      includeAttachments: true,
      importContacts: true,
      importTrustFromHistory: true,
    },
    progress: {
      status: 'idle',
      totalEmails: 0,
      importedEmails: 0,
      totalFolders: 0,
      completedFolders: 0,
      errors: [],
    },
    availableFolders: [],
    step: 0,
  }),
}));

function getSelectedPaths(folders: FolderInfo[]): string[] {
  const paths: string[] = [];
  const traverse = (items: FolderInfo[]) => {
    for (const f of items) {
      if (f.selected) paths.push(f.path);
      if (f.children) traverse(f.children);
    }
  };
  traverse(folders);
  return paths;
}

// ============================================
// Main Wizard Component
// ============================================

export function EmailImportWizard({ onComplete, onCancel }: {
  onComplete: () => void;
  onCancel: () => void;
}) {
  const { step, reset } = useImportStore();

  const handleComplete = () => {
    onComplete();
    reset();
  };

  const handleCancel = () => {
    onCancel();
    reset();
  };

  const steps = [
    { title: 'Choose Provider', component: ProviderStep },
    { title: 'Connect Account', component: ConnectStep },
    { title: 'Select Folders', component: FoldersStep },
    { title: 'Options', component: OptionsStep },
    { title: 'Import', component: ImportStep },
  ];

  const CurrentStep = steps[step].component;

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
      <div className="bg-white dark:bg-gray-800 rounded-2xl max-w-2xl w-full max-h-[90vh] flex flex-col">
        {/* Header */}
        <div className="p-6 border-b border-gray-200 dark:border-gray-700">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-xl font-semibold text-gray-900 dark:text-gray-100">
              Import Emails
            </h2>
            <button
              onClick={handleCancel}
              className="text-gray-400 hover:text-gray-600 dark:hover:text-gray-300"
            >
              <CloseIcon className="w-6 h-6" />
            </button>
          </div>

          {/* Progress Steps */}
          <div className="flex items-center justify-between">
            {steps.map((s, i) => (
              <div key={i} className="flex items-center">
                <div
                  className={`w-8 h-8 rounded-full flex items-center justify-center text-sm font-medium ${
                    i < step
                      ? 'bg-emerald-500 text-white'
                      : i === step
                      ? 'bg-blue-500 text-white'
                      : 'bg-gray-200 dark:bg-gray-700 text-gray-500 dark:text-gray-400'
                  }`}
                >
                  {i < step ? <CheckIcon className="w-4 h-4" /> : i + 1}
                </div>
                {i < steps.length - 1 && (
                  <div
                    className={`w-12 h-0.5 mx-2 ${
                      i < step ? 'bg-emerald-500' : 'bg-gray-200 dark:bg-gray-700'
                    }`}
                  />
                )}
              </div>
            ))}
          </div>
          <div className="flex justify-between mt-2 text-xs text-gray-500 dark:text-gray-400">
            {steps.map((s, i) => (
              <span key={i} className={i === step ? 'text-blue-500 font-medium' : ''}>
                {s.title}
              </span>
            ))}
          </div>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-6">
          <CurrentStep onComplete={handleComplete} />
        </div>
      </div>
    </div>
  );
}

// ============================================
// Step 1: Provider Selection
// ============================================

function ProviderStep() {
  const { config, setProvider, nextStep } = useImportStore();

  const providers: { id: ImportProvider; name: string; icon: React.ReactNode; description: string }[] = [
    {
      id: 'gmail',
      name: 'Gmail',
      icon: <GmailIcon className="w-8 h-8" />,
      description: 'Import from Google Gmail via secure OAuth',
    },
    {
      id: 'outlook',
      name: 'Outlook',
      icon: <OutlookIcon className="w-8 h-8" />,
      description: 'Import from Microsoft 365 or Outlook.com',
    },
    {
      id: 'imap',
      name: 'IMAP Server',
      icon: <ServerIcon className="w-8 h-8" />,
      description: 'Connect to any IMAP email server',
    },
    {
      id: 'mbox',
      name: 'MBOX File',
      icon: <FileIcon className="w-8 h-8" />,
      description: 'Import from an exported MBOX file',
    },
  ];

  return (
    <div>
      <p className="text-gray-600 dark:text-gray-400 mb-6">
        Choose where to import your emails from:
      </p>

      <div className="grid grid-cols-2 gap-4 mb-6">
        {providers.map((provider) => (
          <button
            key={provider.id}
            onClick={() => setProvider(provider.id)}
            className={`p-4 rounded-xl border-2 text-left transition-all ${
              config.provider === provider.id
                ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20'
                : 'border-gray-200 dark:border-gray-700 hover:border-gray-300 dark:hover:border-gray-600'
            }`}
          >
            <div className="flex items-center space-x-3 mb-2">
              {provider.icon}
              <span className="font-medium text-gray-900 dark:text-gray-100">
                {provider.name}
              </span>
            </div>
            <p className="text-sm text-gray-500 dark:text-gray-400">
              {provider.description}
            </p>
          </button>
        ))}
      </div>

      <div className="flex justify-end">
        <button
          onClick={nextStep}
          className="px-6 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 font-medium"
        >
          Continue
        </button>
      </div>
    </div>
  );
}

// ============================================
// Step 2: Connect Account
// ============================================

function ConnectStep() {
  const { config, setConfig, nextStep, prevStep, setFolders } = useImportStore();
  const [isConnecting, setIsConnecting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleOAuthConnect = async (provider: 'gmail' | 'outlook') => {
    setIsConnecting(true);
    setError(null);

    try {
      // In production, this would open OAuth popup
      // For now, simulate the flow
      await new Promise((resolve) => setTimeout(resolve, 1500));

      // Mock successful connection
      setConfig({ accessToken: 'mock_token_' + Date.now() });

      // Fetch folders
      const mockFolders: FolderInfo[] = provider === 'gmail'
        ? [
            { path: 'INBOX', name: 'Inbox', messageCount: 1234, selected: true },
            { path: '[Gmail]/Sent Mail', name: 'Sent', messageCount: 567, selected: true },
            { path: '[Gmail]/Drafts', name: 'Drafts', messageCount: 12, selected: false },
            { path: '[Gmail]/Spam', name: 'Spam', messageCount: 89, selected: false },
            { path: '[Gmail]/Trash', name: 'Trash', messageCount: 45, selected: false },
            { path: 'Work', name: 'Work', messageCount: 345, selected: true },
            { path: 'Personal', name: 'Personal', messageCount: 234, selected: true },
          ]
        : [
            { path: 'Inbox', name: 'Inbox', messageCount: 987, selected: true },
            { path: 'Sent Items', name: 'Sent Items', messageCount: 456, selected: true },
            { path: 'Drafts', name: 'Drafts', messageCount: 8, selected: false },
            { path: 'Junk Email', name: 'Junk', messageCount: 67, selected: false },
            { path: 'Deleted Items', name: 'Deleted', messageCount: 23, selected: false },
          ];

      setFolders(mockFolders);
      nextStep();
    } catch (err) {
      setError('Failed to connect. Please try again.');
    } finally {
      setIsConnecting(false);
    }
  };

  const handleImapConnect = async () => {
    if (!config.imapHost || !config.imapUsername || !config.imapPassword) {
      setError('Please fill in all required fields');
      return;
    }

    setIsConnecting(true);
    setError(null);

    try {
      // In production, this would test the IMAP connection
      await new Promise((resolve) => setTimeout(resolve, 1500));

      const mockFolders: FolderInfo[] = [
        { path: 'INBOX', name: 'Inbox', messageCount: 500, selected: true },
        { path: 'Sent', name: 'Sent', messageCount: 200, selected: true },
        { path: 'Drafts', name: 'Drafts', messageCount: 5, selected: false },
      ];

      setFolders(mockFolders);
      nextStep();
    } catch (err) {
      setError('Connection failed. Check your credentials.');
    } finally {
      setIsConnecting(false);
    }
  };

  const handleMboxUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setIsConnecting(true);
    setError(null);

    try {
      // In production, this would parse the MBOX file
      await new Promise((resolve) => setTimeout(resolve, 1000));

      setFolders([
        { path: file.name, name: file.name.replace('.mbox', ''), messageCount: 1000, selected: true },
      ]);
      nextStep();
    } catch (err) {
      setError('Failed to parse MBOX file');
    } finally {
      setIsConnecting(false);
    }
  };

  return (
    <div>
      {(config.provider === 'gmail' || config.provider === 'outlook') && (
        <div className="text-center">
          <p className="text-gray-600 dark:text-gray-400 mb-6">
            Sign in to your {config.provider === 'gmail' ? 'Google' : 'Microsoft'} account to import emails
          </p>

          <button
            onClick={() => handleOAuthConnect(config.provider as 'gmail' | 'outlook')}
            disabled={isConnecting}
            className="inline-flex items-center space-x-3 px-6 py-3 bg-white dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-600 disabled:opacity-50"
          >
            {config.provider === 'gmail' ? (
              <GmailIcon className="w-6 h-6" />
            ) : (
              <OutlookIcon className="w-6 h-6" />
            )}
            <span className="font-medium text-gray-900 dark:text-gray-100">
              {isConnecting ? 'Connecting...' : `Sign in with ${config.provider === 'gmail' ? 'Google' : 'Microsoft'}`}
            </span>
          </button>

          <p className="mt-4 text-sm text-gray-500 dark:text-gray-400">
            We only request read access to import your emails. Your credentials are never stored.
          </p>
        </div>
      )}

      {config.provider === 'imap' && (
        <div className="max-w-md mx-auto space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
              IMAP Server
            </label>
            <input
              type="text"
              value={config.imapHost || ''}
              onChange={(e) => setConfig({ imapHost: e.target.value })}
              placeholder="imap.example.com"
              className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-800"
            />
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                Port
              </label>
              <input
                type="number"
                value={config.imapPort || 993}
                onChange={(e) => setConfig({ imapPort: Number(e.target.value) })}
                className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-800"
              />
            </div>
            <div className="flex items-end">
              <label className="flex items-center space-x-2">
                <input
                  type="checkbox"
                  checked={config.imapSsl !== false}
                  onChange={(e) => setConfig({ imapSsl: e.target.checked })}
                  className="rounded border-gray-300 dark:border-gray-600"
                />
                <span className="text-sm text-gray-700 dark:text-gray-300">Use SSL/TLS</span>
              </label>
            </div>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
              Username
            </label>
            <input
              type="text"
              value={config.imapUsername || ''}
              onChange={(e) => setConfig({ imapUsername: e.target.value })}
              placeholder="your@email.com"
              className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-800"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
              Password
            </label>
            <input
              type="password"
              value={config.imapPassword || ''}
              onChange={(e) => setConfig({ imapPassword: e.target.value })}
              placeholder="App password or account password"
              className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-800"
            />
          </div>

          <button
            onClick={handleImapConnect}
            disabled={isConnecting}
            className="w-full py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 disabled:opacity-50 font-medium"
          >
            {isConnecting ? 'Connecting...' : 'Connect'}
          </button>
        </div>
      )}

      {config.provider === 'mbox' && (
        <div className="text-center">
          <p className="text-gray-600 dark:text-gray-400 mb-6">
            Upload an MBOX file exported from another email client
          </p>

          <label className="inline-block cursor-pointer">
            <input
              type="file"
              accept=".mbox,.mbx"
              onChange={handleMboxUpload}
              className="hidden"
            />
            <div className="px-6 py-8 border-2 border-dashed border-gray-300 dark:border-gray-600 rounded-xl hover:border-blue-500 transition-colors">
              <UploadIcon className="w-12 h-12 mx-auto text-gray-400 mb-3" />
              <p className="font-medium text-gray-900 dark:text-gray-100">
                {isConnecting ? 'Processing...' : 'Click to upload MBOX file'}
              </p>
              <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
                or drag and drop
              </p>
            </div>
          </label>
        </div>
      )}

      {error && (
        <div className="mt-4 p-3 bg-red-50 dark:bg-red-900/20 rounded-lg text-sm text-red-600 dark:text-red-400 text-center">
          {error}
        </div>
      )}

      <div className="flex justify-between mt-6">
        <button
          onClick={prevStep}
          className="px-6 py-2 text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-100"
        >
          Back
        </button>
      </div>
    </div>
  );
}

// ============================================
// Step 3: Folder Selection
// ============================================

function FoldersStep() {
  const { availableFolders, toggleFolder, nextStep, prevStep } = useImportStore();

  const totalSelected = availableFolders.filter((f) => f.selected).length;
  const totalMessages = availableFolders
    .filter((f) => f.selected)
    .reduce((sum, f) => sum + f.messageCount, 0);

  return (
    <div>
      <p className="text-gray-600 dark:text-gray-400 mb-4">
        Select the folders you want to import:
      </p>

      <div className="border border-gray-200 dark:border-gray-700 rounded-lg divide-y divide-gray-200 dark:divide-gray-700 max-h-64 overflow-y-auto mb-4">
        {availableFolders.map((folder) => (
          <label
            key={folder.path}
            className="flex items-center justify-between p-3 hover:bg-gray-50 dark:hover:bg-gray-750 cursor-pointer"
          >
            <div className="flex items-center space-x-3">
              <input
                type="checkbox"
                checked={folder.selected}
                onChange={() => toggleFolder(folder.path)}
                className="rounded border-gray-300 dark:border-gray-600 text-blue-500"
              />
              <FolderIcon className="w-5 h-5 text-gray-400" />
              <span className="text-gray-900 dark:text-gray-100">{folder.name}</span>
            </div>
            <span className="text-sm text-gray-500 dark:text-gray-400">
              {folder.messageCount.toLocaleString()} emails
            </span>
          </label>
        ))}
      </div>

      <div className="p-3 bg-blue-50 dark:bg-blue-900/20 rounded-lg text-sm">
        <span className="text-blue-700 dark:text-blue-300">
          {totalSelected} folders selected ({totalMessages.toLocaleString()} emails)
        </span>
      </div>

      <div className="flex justify-between mt-6">
        <button
          onClick={prevStep}
          className="px-6 py-2 text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-100"
        >
          Back
        </button>
        <button
          onClick={nextStep}
          disabled={totalSelected === 0}
          className="px-6 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 disabled:opacity-50 font-medium"
        >
          Continue
        </button>
      </div>
    </div>
  );
}

// ============================================
// Step 4: Options
// ============================================

function OptionsStep() {
  const { config, setConfig, nextStep, prevStep } = useImportStore();

  return (
    <div>
      <p className="text-gray-600 dark:text-gray-400 mb-6">
        Configure import options:
      </p>

      <div className="space-y-4">
        {/* Date Range */}
        <div className="p-4 border border-gray-200 dark:border-gray-700 rounded-lg">
          <h3 className="font-medium text-gray-900 dark:text-gray-100 mb-3">Date Range</h3>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm text-gray-600 dark:text-gray-400 mb-1">From</label>
              <input
                type="date"
                value={config.dateRange?.from || ''}
                onChange={(e) => setConfig({
                  dateRange: { ...config.dateRange, from: e.target.value, to: config.dateRange?.to || '' },
                })}
                className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-800"
              />
            </div>
            <div>
              <label className="block text-sm text-gray-600 dark:text-gray-400 mb-1">To</label>
              <input
                type="date"
                value={config.dateRange?.to || ''}
                onChange={(e) => setConfig({
                  dateRange: { from: config.dateRange?.from || '', to: e.target.value },
                })}
                className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-800"
              />
            </div>
          </div>
          <p className="text-xs text-gray-500 dark:text-gray-400 mt-2">
            Leave empty to import all emails
          </p>
        </div>

        {/* Options */}
        <div className="space-y-3">
          <label className="flex items-start space-x-3 p-3 border border-gray-200 dark:border-gray-700 rounded-lg cursor-pointer hover:bg-gray-50 dark:hover:bg-gray-750">
            <input
              type="checkbox"
              checked={config.includeAttachments}
              onChange={(e) => setConfig({ includeAttachments: e.target.checked })}
              className="mt-1 rounded border-gray-300 dark:border-gray-600 text-blue-500"
            />
            <div>
              <span className="font-medium text-gray-900 dark:text-gray-100">Include attachments</span>
              <p className="text-sm text-gray-500 dark:text-gray-400">
                Import email attachments (increases import time and storage)
              </p>
            </div>
          </label>

          <label className="flex items-start space-x-3 p-3 border border-gray-200 dark:border-gray-700 rounded-lg cursor-pointer hover:bg-gray-50 dark:hover:bg-gray-750">
            <input
              type="checkbox"
              checked={config.importContacts}
              onChange={(e) => setConfig({ importContacts: e.target.checked })}
              className="mt-1 rounded border-gray-300 dark:border-gray-600 text-blue-500"
            />
            <div>
              <span className="font-medium text-gray-900 dark:text-gray-100">Import contacts</span>
              <p className="text-sm text-gray-500 dark:text-gray-400">
                Add email addresses from imported emails to your contacts
              </p>
            </div>
          </label>

          <label className="flex items-start space-x-3 p-3 border border-blue-100 dark:border-blue-900 bg-blue-50 dark:bg-blue-900/20 rounded-lg cursor-pointer">
            <input
              type="checkbox"
              checked={config.importTrustFromHistory}
              onChange={(e) => setConfig({ importTrustFromHistory: e.target.checked })}
              className="mt-1 rounded border-gray-300 dark:border-gray-600 text-blue-500"
            />
            <div>
              <span className="font-medium text-gray-900 dark:text-gray-100">Build trust from history</span>
              <p className="text-sm text-gray-500 dark:text-gray-400">
                Analyze email patterns to bootstrap trust scores for frequent contacts
              </p>
            </div>
          </label>
        </div>
      </div>

      <div className="flex justify-between mt-6">
        <button
          onClick={prevStep}
          className="px-6 py-2 text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-100"
        >
          Back
        </button>
        <button
          onClick={nextStep}
          className="px-6 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 font-medium"
        >
          Start Import
        </button>
      </div>
    </div>
  );
}

// ============================================
// Step 5: Import Progress
// ============================================

function ImportStep({ onComplete }: { onComplete: () => void }) {
  const { config, progress, setProgress, availableFolders, prevStep } = useImportStore();
  const [hasStarted, setHasStarted] = useState(false);

  const startImport = useCallback(async () => {
    setHasStarted(true);

    const selectedFolders = availableFolders.filter((f) => f.selected);
    const totalEmails = selectedFolders.reduce((sum, f) => sum + f.messageCount, 0);

    setProgress({
      status: 'connecting',
      totalEmails,
      totalFolders: selectedFolders.length,
      startedAt: new Date().toISOString(),
    });

    // Simulate connection
    await new Promise((resolve) => setTimeout(resolve, 1000));
    setProgress({ status: 'fetching' });

    // Simulate folder-by-folder import
    let imported = 0;
    for (let i = 0; i < selectedFolders.length; i++) {
      const folder = selectedFolders[i];
      setProgress({
        status: 'importing',
        currentFolder: folder.name,
        completedFolders: i,
      });

      // Simulate importing emails in chunks
      const chunks = Math.ceil(folder.messageCount / 100);
      for (let c = 0; c < chunks; c++) {
        await new Promise((resolve) => setTimeout(resolve, 200));
        imported += Math.min(100, folder.messageCount - c * 100);
        setProgress({ importedEmails: imported });
      }
    }

    setProgress({
      status: 'complete',
      completedFolders: selectedFolders.length,
      completedAt: new Date().toISOString(),
    });
  }, [availableFolders, setProgress]);

  // Auto-start import
  useState(() => {
    if (!hasStarted) {
      startImport();
    }
  });

  const progressPercent = progress.totalEmails > 0
    ? Math.round((progress.importedEmails / progress.totalEmails) * 100)
    : 0;

  return (
    <div className="text-center">
      {progress.status === 'complete' ? (
        <>
          <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-emerald-100 dark:bg-emerald-900/30 flex items-center justify-center">
            <CheckIcon className="w-8 h-8 text-emerald-600 dark:text-emerald-400" />
          </div>
          <h3 className="text-xl font-semibold text-gray-900 dark:text-gray-100 mb-2">
            Import Complete!
          </h3>
          <p className="text-gray-600 dark:text-gray-400 mb-6">
            Successfully imported {progress.importedEmails.toLocaleString()} emails from {progress.completedFolders} folders
          </p>
          {config.importTrustFromHistory && (
            <div className="p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg mb-6 text-left">
              <h4 className="font-medium text-blue-800 dark:text-blue-200 mb-1">Trust Analysis Complete</h4>
              <p className="text-sm text-blue-600 dark:text-blue-300">
                Based on your email history, we've identified 47 trusted contacts and established initial trust scores.
              </p>
            </div>
          )}
          <button
            onClick={onComplete}
            className="px-6 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 font-medium"
          >
            Done
          </button>
        </>
      ) : (
        <>
          <div className="w-16 h-16 mx-auto mb-4 relative">
            <svg className="w-full h-full -rotate-90">
              <circle
                cx="32"
                cy="32"
                r="28"
                stroke="currentColor"
                strokeWidth="8"
                fill="none"
                className="text-gray-200 dark:text-gray-700"
              />
              <circle
                cx="32"
                cy="32"
                r="28"
                stroke="currentColor"
                strokeWidth="8"
                fill="none"
                strokeDasharray={175.9}
                strokeDashoffset={175.9 * (1 - progressPercent / 100)}
                className="text-blue-500 transition-all duration-300"
              />
            </svg>
            <span className="absolute inset-0 flex items-center justify-center text-sm font-medium text-gray-900 dark:text-gray-100">
              {progressPercent}%
            </span>
          </div>

          <h3 className="text-lg font-medium text-gray-900 dark:text-gray-100 mb-1">
            {progress.status === 'connecting' && 'Connecting...'}
            {progress.status === 'fetching' && 'Fetching email list...'}
            {progress.status === 'importing' && `Importing ${progress.currentFolder}...`}
          </h3>

          <p className="text-gray-500 dark:text-gray-400 mb-4">
            {progress.importedEmails.toLocaleString()} of {progress.totalEmails.toLocaleString()} emails
          </p>

          <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2 mb-2">
            <div
              className="bg-blue-500 h-2 rounded-full transition-all duration-300"
              style={{ width: `${progressPercent}%` }}
            />
          </div>

          <p className="text-sm text-gray-500 dark:text-gray-400">
            Folder {progress.completedFolders + 1} of {progress.totalFolders}
          </p>
        </>
      )}
    </div>
  );
}

// ============================================
// Icons
// ============================================

function CloseIcon({ className }: { className?: string }) {
  return (
    <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
    </svg>
  );
}

function CheckIcon({ className }: { className?: string }) {
  return (
    <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
    </svg>
  );
}

function GmailIcon({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24">
      <path fill="#EA4335" d="M24 5.457v13.909c0 .904-.732 1.636-1.636 1.636h-3.819V11.73L12 16.64l-6.545-4.91v9.273H1.636A1.636 1.636 0 0 1 0 19.366V5.457c0-2.023 2.309-3.178 3.927-1.964L5.455 4.64 12 9.548l6.545-4.91 1.528-1.145C21.69 2.28 24 3.434 24 5.457z"/>
    </svg>
  );
}

function OutlookIcon({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24">
      <path fill="#0078D4" d="M24 7.387v10.478c0 .23-.08.424-.238.576a.806.806 0 0 1-.576.238h-8.5v-5.478l1.674 1.326a.378.378 0 0 0 .576-.05.378.378 0 0 0-.05-.576L12 9.87l-4.886 4.03a.378.378 0 0 1-.576.051.378.378 0 0 1 .05-.576l1.674-1.326V7.387h-.005a.758.758 0 0 0 .243.576l4.5 3.696a.87.87 0 0 0 1 0l4.5-3.696a.758.758 0 0 0 .243-.576z"/>
      <path fill="#0078D4" d="M7.5 5.91h15.687a.815.815 0 0 1 .576.238.76.76 0 0 1 .237.576v.426L18.75 11.1V5.91z"/>
      <path fill="#28A8EA" d="M9 12.75V19.5H1.5a.866.866 0 0 1-.61-.253.866.866 0 0 1-.252-.61V5.864c0-.234.084-.437.253-.61a.866.866 0 0 1 .61-.253H9v7.75z"/>
      <ellipse fill="#fff" cx="5.25" cy="12.75" rx="3" ry="3.375"/>
    </svg>
  );
}

function ServerIcon({ className }: { className?: string }) {
  return (
    <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 12h14M5 12a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v4a2 2 0 01-2 2M5 12a2 2 0 00-2 2v4a2 2 0 002 2h14a2 2 0 002-2v-4a2 2 0 00-2-2m-2-4h.01M17 16h.01" />
    </svg>
  );
}

function FileIcon({ className }: { className?: string }) {
  return (
    <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 21h10a2 2 0 002-2V9.414a1 1 0 00-.293-.707l-5.414-5.414A1 1 0 0012.586 3H7a2 2 0 00-2 2v14a2 2 0 002 2z" />
    </svg>
  );
}

function UploadIcon({ className }: { className?: string }) {
  return (
    <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" />
    </svg>
  );
}

function FolderIcon({ className }: { className?: string }) {
  return (
    <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 7v10a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-6l-2-2H5a2 2 0 00-2 2z" />
    </svg>
  );
}

export default EmailImportWizard;
