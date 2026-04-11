// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Import Page
 *
 * Email migration from Gmail, Outlook, and file imports
 */

import React, { useEffect, useState } from 'react';

interface MigrationJob {
  id: string;
  sourceType: string;
  status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled';
  totalItems: number;
  processedItems: number;
  failedItems: number;
  currentPhase: string;
  errorLog: string[];
  createdAt: string;
  startedAt?: string;
  completedAt?: string;
}

export default function ImportPage() {
  const [jobs, setJobs] = useState<MigrationJob[]>([]);
  const [showGmailModal, setShowGmailModal] = useState(false);
  const [showOutlookModal, setShowOutlookModal] = useState(false);
  const [showFileModal, setShowFileModal] = useState(false);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchJobs();
    // Poll for updates while jobs are running
    const interval = setInterval(() => {
      if (jobs.some((j) => j.status === 'running' || j.status === 'pending')) {
        fetchJobs();
      }
    }, 5000);
    return () => clearInterval(interval);
  }, []);

  async function fetchJobs() {
    try {
      const response = await fetch('/api/migration/jobs');
      if (response.ok) {
        setJobs(await response.json());
      }
    } catch (error) {
      console.error('Failed to fetch jobs:', error);
    } finally {
      setLoading(false);
    }
  }

  async function cancelJob(jobId: string) {
    try {
      await fetch(`/api/migration/jobs/${jobId}/cancel`, { method: 'POST' });
      fetchJobs();
    } catch (error) {
      console.error('Failed to cancel job:', error);
    }
  }

  function getStatusColor(status: MigrationJob['status']): string {
    switch (status) {
      case 'completed':
        return 'text-green-600 bg-green-100';
      case 'running':
        return 'text-blue-600 bg-blue-100';
      case 'pending':
        return 'text-yellow-600 bg-yellow-100';
      case 'failed':
        return 'text-red-600 bg-red-100';
      case 'cancelled':
        return 'text-gray-600 bg-gray-100';
    }
  }

  function getSourceIcon(source: string): string {
    switch (source) {
      case 'gmail':
        return 'Gmail';
      case 'outlook':
        return 'Outlook';
      case 'mbox':
        return 'MBOX';
      case 'eml':
        return 'EML';
      default:
        return 'Import';
    }
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary"></div>
      </div>
    );
  }

  return (
    <div className="p-6 max-w-4xl mx-auto">
      <div className="mb-8">
        <h1 className="text-2xl font-bold">Import Emails</h1>
        <p className="text-muted">Migrate your emails from other providers</p>
      </div>

      {/* Import Options */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
        <div
          className="p-6 border border-border rounded-lg hover:border-primary cursor-pointer text-center"
          onClick={() => setShowGmailModal(true)}
        >
          <div className="text-4xl mb-3">Gmail</div>
          <h3 className="font-semibold">Import from Gmail</h3>
          <p className="text-sm text-muted mt-1">Connect with Google OAuth</p>
        </div>

        <div
          className="p-6 border border-border rounded-lg hover:border-primary cursor-pointer text-center"
          onClick={() => setShowOutlookModal(true)}
        >
          <div className="text-4xl mb-3">Outlook</div>
          <h3 className="font-semibold">Import from Outlook</h3>
          <p className="text-sm text-muted mt-1">Connect with Microsoft OAuth</p>
        </div>

        <div
          className="p-6 border border-border rounded-lg hover:border-primary cursor-pointer text-center"
          onClick={() => setShowFileModal(true)}
        >
          <div className="text-4xl mb-3">File</div>
          <h3 className="font-semibold">Import from File</h3>
          <p className="text-sm text-muted mt-1">MBOX, EML, or PST files</p>
        </div>
      </div>

      {/* Active/Recent Jobs */}
      <div>
        <h2 className="text-lg font-semibold mb-4">Import History</h2>

        {jobs.length === 0 ? (
          <div className="text-center py-12 bg-surface rounded-lg border border-border">
            <p className="text-muted">No imports yet</p>
          </div>
        ) : (
          <div className="space-y-4">
            {jobs.map((job) => (
              <div key={job.id} className="border border-border rounded-lg p-4">
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center gap-3">
                    <span className="font-medium">{getSourceIcon(job.sourceType)}</span>
                    <span className={`px-2 py-0.5 rounded text-sm ${getStatusColor(job.status)}`}>
                      {job.status}
                    </span>
                  </div>
                  <div className="text-sm text-muted">
                    {new Date(job.createdAt).toLocaleString()}
                  </div>
                </div>

                {(job.status === 'running' || job.status === 'pending') && (
                  <>
                    <div className="mb-2">
                      <div className="flex justify-between text-sm mb-1">
                        <span>{job.currentPhase}</span>
                        <span>
                          {job.processedItems} / {job.totalItems || '?'}
                        </span>
                      </div>
                      <div className="h-2 bg-gray-200 rounded-full overflow-hidden">
                        <div
                          className="h-full bg-primary transition-all duration-300"
                          style={{
                            width: job.totalItems
                              ? `${(job.processedItems / job.totalItems) * 100}%`
                              : '0%',
                          }}
                        />
                      </div>
                    </div>
                    <button
                      onClick={() => cancelJob(job.id)}
                      className="text-sm text-red-600 hover:underline"
                    >
                      Cancel Import
                    </button>
                  </>
                )}

                {job.status === 'completed' && (
                  <div className="text-sm text-green-600">
                    Successfully imported {job.processedItems} emails
                    {job.failedItems > 0 && (
                      <span className="text-red-600"> ({job.failedItems} failed)</span>
                    )}
                  </div>
                )}

                {job.status === 'failed' && job.errorLog.length > 0 && (
                  <div className="mt-2 p-2 bg-red-50 rounded text-sm text-red-600">
                    {job.errorLog[0]}
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Gmail Modal */}
      {showGmailModal && (
        <GmailImportModal
          onClose={() => setShowGmailModal(false)}
          onStarted={() => {
            setShowGmailModal(false);
            fetchJobs();
          }}
        />
      )}

      {/* Outlook Modal */}
      {showOutlookModal && (
        <OutlookImportModal
          onClose={() => setShowOutlookModal(false)}
          onStarted={() => {
            setShowOutlookModal(false);
            fetchJobs();
          }}
        />
      )}

      {/* File Import Modal */}
      {showFileModal && (
        <FileImportModal
          onClose={() => setShowFileModal(false)}
          onStarted={() => {
            setShowFileModal(false);
            fetchJobs();
          }}
        />
      )}
    </div>
  );
}

function GmailImportModal({
  onClose,
  onStarted,
}: {
  onClose: () => void;
  onStarted: () => void;
}) {
  const [options, setOptions] = useState({
    includeSpam: false,
    includeTrash: false,
    dateFrom: '',
    dateTo: '',
  });

  async function startImport() {
    try {
      // Would redirect to Google OAuth
      // For now, simulate starting import
      const response = await fetch('/api/migration/gmail/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(options),
      });
      if (response.ok) {
        onStarted();
      }
    } catch (error) {
      console.error('Failed to start import:', error);
    }
  }

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
      <div className="bg-background rounded-lg shadow-xl w-full max-w-md">
        <div className="p-4 border-b border-border">
          <h2 className="text-lg font-semibold">Import from Gmail</h2>
        </div>

        <div className="p-4 space-y-4">
          <p className="text-sm text-muted">
            Connect your Google account to import emails. You'll be redirected to
            Google to authorize access.
          </p>

          <div className="space-y-3">
            <label className="flex items-center gap-2">
              <input
                type="checkbox"
                checked={options.includeSpam}
                onChange={(e) => setOptions({ ...options, includeSpam: e.target.checked })}
              />
              Include spam folder
            </label>
            <label className="flex items-center gap-2">
              <input
                type="checkbox"
                checked={options.includeTrash}
                onChange={(e) => setOptions({ ...options, includeTrash: e.target.checked })}
              />
              Include trash folder
            </label>
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium mb-1">From Date</label>
              <input
                type="date"
                value={options.dateFrom}
                onChange={(e) => setOptions({ ...options, dateFrom: e.target.value })}
                className="w-full px-3 py-2 border border-border rounded"
              />
            </div>
            <div>
              <label className="block text-sm font-medium mb-1">To Date</label>
              <input
                type="date"
                value={options.dateTo}
                onChange={(e) => setOptions({ ...options, dateTo: e.target.value })}
                className="w-full px-3 py-2 border border-border rounded"
              />
            </div>
          </div>
        </div>

        <div className="p-4 border-t border-border flex justify-end gap-2">
          <button
            onClick={onClose}
            className="px-4 py-2 border border-border rounded-lg"
          >
            Cancel
          </button>
          <button
            onClick={startImport}
            className="px-4 py-2 bg-primary text-white rounded-lg"
          >
            Connect Google Account
          </button>
        </div>
      </div>
    </div>
  );
}

function OutlookImportModal({
  onClose,
  onStarted,
}: {
  onClose: () => void;
  onStarted: () => void;
}) {
  const [options, setOptions] = useState({
    includeCalendar: true,
    includeContacts: true,
  });

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
      <div className="bg-background rounded-lg shadow-xl w-full max-w-md">
        <div className="p-4 border-b border-border">
          <h2 className="text-lg font-semibold">Import from Outlook</h2>
        </div>

        <div className="p-4 space-y-4">
          <p className="text-sm text-muted">
            Connect your Microsoft account to import emails, calendar, and contacts.
          </p>

          <div className="space-y-3">
            <label className="flex items-center gap-2">
              <input
                type="checkbox"
                checked={options.includeCalendar}
                onChange={(e) => setOptions({ ...options, includeCalendar: e.target.checked })}
              />
              Include calendar events
            </label>
            <label className="flex items-center gap-2">
              <input
                type="checkbox"
                checked={options.includeContacts}
                onChange={(e) => setOptions({ ...options, includeContacts: e.target.checked })}
              />
              Include contacts
            </label>
          </div>
        </div>

        <div className="p-4 border-t border-border flex justify-end gap-2">
          <button onClick={onClose} className="px-4 py-2 border border-border rounded-lg">
            Cancel
          </button>
          <button className="px-4 py-2 bg-primary text-white rounded-lg">
            Connect Microsoft Account
          </button>
        </div>
      </div>
    </div>
  );
}

function FileImportModal({
  onClose,
  onStarted,
}: {
  onClose: () => void;
  onStarted: () => void;
}) {
  const [file, setFile] = useState<File | null>(null);
  const [targetFolder, setTargetFolder] = useState('Inbox');
  const [uploading, setUploading] = useState(false);

  async function handleUpload() {
    if (!file) return;

    setUploading(true);
    try {
      const formData = new FormData();
      formData.append('file', file);
      formData.append('folder', targetFolder);

      const response = await fetch('/api/migration/file/upload', {
        method: 'POST',
        body: formData,
      });

      if (response.ok) {
        onStarted();
      }
    } catch (error) {
      console.error('Failed to upload file:', error);
    } finally {
      setUploading(false);
    }
  }

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
      <div className="bg-background rounded-lg shadow-xl w-full max-w-md">
        <div className="p-4 border-b border-border">
          <h2 className="text-lg font-semibold">Import from File</h2>
        </div>

        <div className="p-4 space-y-4">
          <div>
            <label className="block text-sm font-medium mb-2">Select File</label>
            <input
              type="file"
              accept=".mbox,.eml,.pst"
              onChange={(e) => setFile(e.target.files?.[0] || null)}
              className="w-full"
            />
            <p className="text-xs text-muted mt-1">
              Supported formats: MBOX, EML, PST
            </p>
          </div>

          <div>
            <label className="block text-sm font-medium mb-1">Import to Folder</label>
            <select
              value={targetFolder}
              onChange={(e) => setTargetFolder(e.target.value)}
              className="w-full px-3 py-2 border border-border rounded"
            >
              <option value="Inbox">Inbox</option>
              <option value="Archive">Archive</option>
              <option value="Imported">Imported</option>
            </select>
          </div>
        </div>

        <div className="p-4 border-t border-border flex justify-end gap-2">
          <button onClick={onClose} className="px-4 py-2 border border-border rounded-lg">
            Cancel
          </button>
          <button
            onClick={handleUpload}
            disabled={!file || uploading}
            className="px-4 py-2 bg-primary text-white rounded-lg disabled:opacity-50"
          >
            {uploading ? 'Uploading...' : 'Start Import'}
          </button>
        </div>
      </div>
    </div>
  );
}
