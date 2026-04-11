// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Batch Operations for Trust Management
 *
 * Bulk actions for attestations and trust management:
 * - Batch attestation creation
 * - Bulk revocation
 * - Import/export trust network
 * - Batch verification
 */

import { useState, useCallback, useMemo } from 'react';
import { useMutation, useQueryClient } from '@tanstack/react-query';

// ============================================
// Types
// ============================================

interface BatchContact {
  did: string;
  name: string;
  email?: string;
  selected: boolean;
}

interface BatchAttestationRequest {
  subjectDid: string;
  relationship: string;
  trustScore: number;
  message?: string;
}

interface ExportedTrustNetwork {
  version: string;
  exportedAt: string;
  userDid: string;
  attestations: Array<{
    subjectDid: string;
    relationship: string;
    trustScore: number;
    message?: string;
    createdAt: string;
  }>;
  settings: {
    trustDecayDays: number;
    maxPathLength: number;
  };
}

// ============================================
// Batch Attestation Creator
// ============================================

interface BatchAttestationProps {
  contacts: BatchContact[];
  onComplete?: (results: { success: number; failed: number }) => void;
}

export function BatchAttestationCreator({ contacts, onComplete }: BatchAttestationProps) {
  const [selectedContacts, setSelectedContacts] = useState<Set<string>>(new Set());
  const [relationship, setRelationship] = useState('direct_trust');
  const [trustScore, setTrustScore] = useState(0.7);
  const [message, setMessage] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);
  const [progress, setProgress] = useState({ current: 0, total: 0 });

  const queryClient = useQueryClient();

  const toggleContact = useCallback((did: string) => {
    setSelectedContacts((prev) => {
      const next = new Set(prev);
      if (next.has(did)) {
        next.delete(did);
      } else {
        next.add(did);
      }
      return next;
    });
  }, []);

  const selectAll = useCallback(() => {
    setSelectedContacts(new Set(contacts.map((c) => c.did)));
  }, [contacts]);

  const selectNone = useCallback(() => {
    setSelectedContacts(new Set());
  }, []);

  const processBatch = useCallback(async () => {
    if (selectedContacts.size === 0) return;

    setIsProcessing(true);
    setProgress({ current: 0, total: selectedContacts.size });

    let success = 0;
    let failed = 0;

    for (const did of selectedContacts) {
      try {
        // Simulate API call - replace with actual API
        await new Promise((resolve) => setTimeout(resolve, 200));
        success++;
      } catch {
        failed++;
      }
      setProgress((prev) => ({ ...prev, current: prev.current + 1 }));
    }

    setIsProcessing(false);
    queryClient.invalidateQueries({ queryKey: ['trust'] });
    queryClient.invalidateQueries({ queryKey: ['attestations'] });
    onComplete?.({ success, failed });
  }, [selectedContacts, relationship, trustScore, message, queryClient, onComplete]);

  const relationshipOptions = [
    { value: 'direct_trust', label: 'Direct Trust' },
    { value: 'introduction', label: 'Introduction' },
    { value: 'organization_member', label: 'Organization Member' },
    { value: 'vouch', label: 'Vouch' },
  ];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100">
          Batch Create Attestations
        </h3>
        <div className="text-sm text-gray-500">
          {selectedContacts.size} of {contacts.length} selected
        </div>
      </div>

      {/* Selection controls */}
      <div className="flex gap-2">
        <button
          onClick={selectAll}
          className="px-3 py-1.5 text-sm bg-gray-100 dark:bg-gray-800 hover:bg-gray-200 dark:hover:bg-gray-700 rounded-lg transition-colors"
        >
          Select All
        </button>
        <button
          onClick={selectNone}
          className="px-3 py-1.5 text-sm bg-gray-100 dark:bg-gray-800 hover:bg-gray-200 dark:hover:bg-gray-700 rounded-lg transition-colors"
        >
          Clear Selection
        </button>
      </div>

      {/* Contact list */}
      <div className="max-h-64 overflow-y-auto border border-gray-200 dark:border-gray-700 rounded-lg divide-y divide-gray-100 dark:divide-gray-800">
        {contacts.map((contact) => (
          <label
            key={contact.did}
            className="flex items-center gap-3 px-4 py-3 hover:bg-gray-50 dark:hover:bg-gray-800/50 cursor-pointer"
          >
            <input
              type="checkbox"
              checked={selectedContacts.has(contact.did)}
              onChange={() => toggleContact(contact.did)}
              className="w-4 h-4 rounded border-gray-300 text-blue-600 focus:ring-blue-500"
            />
            <div className="flex-1 min-w-0">
              <div className="font-medium text-gray-900 dark:text-gray-100 truncate">
                {contact.name}
              </div>
              {contact.email && (
                <div className="text-sm text-gray-500 truncate">{contact.email}</div>
              )}
            </div>
            <div className="text-xs text-gray-400 font-mono">
              {contact.did.slice(-8)}
            </div>
          </label>
        ))}
      </div>

      {/* Attestation settings */}
      <div className="grid grid-cols-2 gap-4">
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
            Relationship Type
          </label>
          <select
            value={relationship}
            onChange={(e) => setRelationship(e.target.value)}
            className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-800 focus:ring-2 focus:ring-blue-500"
          >
            {relationshipOptions.map((opt) => (
              <option key={opt.value} value={opt.value}>
                {opt.label}
              </option>
            ))}
          </select>
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
            Trust Score: {Math.round(trustScore * 100)}%
          </label>
          <input
            type="range"
            min="0"
            max="100"
            value={trustScore * 100}
            onChange={(e) => setTrustScore(Number(e.target.value) / 100)}
            className="w-full"
          />
        </div>
      </div>

      <div>
        <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
          Message (optional)
        </label>
        <input
          type="text"
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          placeholder="Reason for attestation..."
          className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-800 focus:ring-2 focus:ring-blue-500"
        />
      </div>

      {/* Progress */}
      {isProcessing && (
        <div className="space-y-2">
          <div className="flex items-center justify-between text-sm">
            <span>Processing attestations...</span>
            <span>{progress.current} / {progress.total}</span>
          </div>
          <div className="h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
            <div
              className="h-full bg-blue-500 transition-all duration-300"
              style={{ width: `${(progress.current / progress.total) * 100}%` }}
            />
          </div>
        </div>
      )}

      {/* Actions */}
      <div className="flex justify-end gap-3">
        <button
          onClick={processBatch}
          disabled={selectedContacts.size === 0 || isProcessing}
          className="px-4 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-400 text-white rounded-lg transition-colors"
        >
          {isProcessing
            ? 'Processing...'
            : `Create ${selectedContacts.size} Attestation${selectedContacts.size !== 1 ? 's' : ''}`}
        </button>
      </div>
    </div>
  );
}

// ============================================
// Trust Network Export/Import
// ============================================

interface ExportImportProps {
  userDid: string;
  onImportComplete?: (count: number) => void;
}

export function TrustNetworkExportImport({ userDid, onImportComplete }: ExportImportProps) {
  const [isExporting, setIsExporting] = useState(false);
  const [isImporting, setIsImporting] = useState(false);
  const [importError, setImportError] = useState<string | null>(null);
  const [importPreview, setImportPreview] = useState<ExportedTrustNetwork | null>(null);

  const queryClient = useQueryClient();

  // Export trust network
  const handleExport = useCallback(async () => {
    setIsExporting(true);

    try {
      // Fetch current attestations - replace with actual API
      const attestations = await Promise.resolve([
        {
          subjectDid: 'did:mycelix:example1',
          relationship: 'direct_trust',
          trustScore: 0.8,
          message: 'Trusted colleague',
          createdAt: new Date().toISOString(),
        },
      ]);

      const exportData: ExportedTrustNetwork = {
        version: '1.0',
        exportedAt: new Date().toISOString(),
        userDid,
        attestations,
        settings: {
          trustDecayDays: 90,
          maxPathLength: 3,
        },
      };

      // Create and download file
      const blob = new Blob([JSON.stringify(exportData, null, 2)], {
        type: 'application/json',
      });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `trust-network-${new Date().toISOString().split('T')[0]}.json`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    } catch (error) {
      console.error('Export failed:', error);
    } finally {
      setIsExporting(false);
    }
  }, [userDid]);

  // Handle file selection for import
  const handleFileSelect = useCallback((event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    setImportError(null);
    setImportPreview(null);

    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const data = JSON.parse(e.target?.result as string) as ExportedTrustNetwork;

        // Validate structure
        if (!data.version || !data.attestations || !Array.isArray(data.attestations)) {
          throw new Error('Invalid trust network file format');
        }

        setImportPreview(data);
      } catch (error) {
        setImportError(error instanceof Error ? error.message : 'Failed to parse file');
      }
    };
    reader.readAsText(file);
  }, []);

  // Perform import
  const handleImport = useCallback(async () => {
    if (!importPreview) return;

    setIsImporting(true);
    setImportError(null);

    try {
      // Process each attestation - replace with actual API
      for (const attestation of importPreview.attestations) {
        await new Promise((resolve) => setTimeout(resolve, 100));
      }

      queryClient.invalidateQueries({ queryKey: ['trust'] });
      queryClient.invalidateQueries({ queryKey: ['attestations'] });
      onImportComplete?.(importPreview.attestations.length);
      setImportPreview(null);
    } catch (error) {
      setImportError(error instanceof Error ? error.message : 'Import failed');
    } finally {
      setIsImporting(false);
    }
  }, [importPreview, queryClient, onImportComplete]);

  return (
    <div className="space-y-6">
      <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100">
        Export / Import Trust Network
      </h3>

      {/* Export section */}
      <div className="p-4 bg-gray-50 dark:bg-gray-800 rounded-lg">
        <div className="flex items-start gap-4">
          <div className="p-2 bg-blue-100 dark:bg-blue-900/30 rounded-lg">
            <svg className="w-6 h-6 text-blue-600 dark:text-blue-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" />
            </svg>
          </div>
          <div className="flex-1">
            <h4 className="font-medium text-gray-900 dark:text-gray-100">Export</h4>
            <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
              Download your trust network as a JSON file. Includes all attestations and settings.
            </p>
            <button
              onClick={handleExport}
              disabled={isExporting}
              className="mt-3 px-4 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-400 text-white text-sm rounded-lg transition-colors"
            >
              {isExporting ? 'Exporting...' : 'Export Trust Network'}
            </button>
          </div>
        </div>
      </div>

      {/* Import section */}
      <div className="p-4 bg-gray-50 dark:bg-gray-800 rounded-lg">
        <div className="flex items-start gap-4">
          <div className="p-2 bg-emerald-100 dark:bg-emerald-900/30 rounded-lg">
            <svg className="w-6 h-6 text-emerald-600 dark:text-emerald-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
            </svg>
          </div>
          <div className="flex-1">
            <h4 className="font-medium text-gray-900 dark:text-gray-100">Import</h4>
            <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
              Restore your trust network from a previously exported file.
            </p>

            <label className="mt-3 inline-flex items-center gap-2 px-4 py-2 bg-emerald-600 hover:bg-emerald-700 text-white text-sm rounded-lg cursor-pointer transition-colors">
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
              </svg>
              Select File
              <input
                type="file"
                accept=".json"
                onChange={handleFileSelect}
                className="hidden"
              />
            </label>
          </div>
        </div>

        {/* Import error */}
        {importError && (
          <div className="mt-4 p-3 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg text-sm text-red-700 dark:text-red-300">
            {importError}
          </div>
        )}

        {/* Import preview */}
        {importPreview && (
          <div className="mt-4 p-4 bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-700 rounded-lg">
            <h5 className="font-medium text-gray-900 dark:text-gray-100 mb-3">
              Import Preview
            </h5>
            <dl className="grid grid-cols-2 gap-4 text-sm">
              <div>
                <dt className="text-gray-500">Exported</dt>
                <dd className="font-medium">
                  {new Date(importPreview.exportedAt).toLocaleDateString()}
                </dd>
              </div>
              <div>
                <dt className="text-gray-500">Attestations</dt>
                <dd className="font-medium">{importPreview.attestations.length}</dd>
              </div>
              <div>
                <dt className="text-gray-500">Version</dt>
                <dd className="font-medium">{importPreview.version}</dd>
              </div>
              <div>
                <dt className="text-gray-500">Source DID</dt>
                <dd className="font-medium font-mono text-xs truncate">
                  {importPreview.userDid.slice(-12)}
                </dd>
              </div>
            </dl>

            <div className="mt-4 flex gap-3">
              <button
                onClick={handleImport}
                disabled={isImporting}
                className="px-4 py-2 bg-emerald-600 hover:bg-emerald-700 disabled:bg-gray-400 text-white text-sm rounded-lg transition-colors"
              >
                {isImporting
                  ? 'Importing...'
                  : `Import ${importPreview.attestations.length} Attestations`}
              </button>
              <button
                onClick={() => setImportPreview(null)}
                className="px-4 py-2 text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-100 text-sm transition-colors"
              >
                Cancel
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

// ============================================
// Batch Revocation
// ============================================

interface BatchRevocationProps {
  attestations: Array<{
    id: string;
    subjectDid: string;
    subjectName: string;
    createdAt: string;
  }>;
  onComplete?: (count: number) => void;
}

export function BatchRevocation({ attestations, onComplete }: BatchRevocationProps) {
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
  const [isProcessing, setIsProcessing] = useState(false);
  const [confirmOpen, setConfirmOpen] = useState(false);

  const queryClient = useQueryClient();

  const toggleAttestation = useCallback((id: string) => {
    setSelectedIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) {
        next.delete(id);
      } else {
        next.add(id);
      }
      return next;
    });
  }, []);

  const processRevocation = useCallback(async () => {
    if (selectedIds.size === 0) return;

    setIsProcessing(true);

    try {
      for (const id of selectedIds) {
        // Simulate API call
        await new Promise((resolve) => setTimeout(resolve, 150));
      }

      queryClient.invalidateQueries({ queryKey: ['trust'] });
      queryClient.invalidateQueries({ queryKey: ['attestations'] });
      onComplete?.(selectedIds.size);
      setSelectedIds(new Set());
    } finally {
      setIsProcessing(false);
      setConfirmOpen(false);
    }
  }, [selectedIds, queryClient, onComplete]);

  return (
    <div className="space-y-4">
      <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100">
        Revoke Attestations
      </h3>

      <div className="max-h-64 overflow-y-auto border border-gray-200 dark:border-gray-700 rounded-lg divide-y divide-gray-100 dark:divide-gray-800">
        {attestations.map((att) => (
          <label
            key={att.id}
            className="flex items-center gap-3 px-4 py-3 hover:bg-gray-50 dark:hover:bg-gray-800/50 cursor-pointer"
          >
            <input
              type="checkbox"
              checked={selectedIds.has(att.id)}
              onChange={() => toggleAttestation(att.id)}
              className="w-4 h-4 rounded border-gray-300 text-red-600 focus:ring-red-500"
            />
            <div className="flex-1 min-w-0">
              <div className="font-medium text-gray-900 dark:text-gray-100">
                {att.subjectName}
              </div>
              <div className="text-xs text-gray-500">
                Created {new Date(att.createdAt).toLocaleDateString()}
              </div>
            </div>
          </label>
        ))}
      </div>

      <button
        onClick={() => setConfirmOpen(true)}
        disabled={selectedIds.size === 0}
        className="px-4 py-2 bg-red-600 hover:bg-red-700 disabled:bg-gray-400 text-white rounded-lg transition-colors"
      >
        Revoke {selectedIds.size} Attestation{selectedIds.size !== 1 ? 's' : ''}
      </button>

      {/* Confirmation modal */}
      {confirmOpen && (
        <>
          <div className="fixed inset-0 bg-black/50 z-50" onClick={() => setConfirmOpen(false)} />
          <div className="fixed top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-full max-w-md p-6 bg-white dark:bg-gray-900 rounded-xl shadow-xl z-50">
            <h4 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-2">
              Confirm Revocation
            </h4>
            <p className="text-gray-600 dark:text-gray-400 mb-4">
              Are you sure you want to revoke {selectedIds.size} attestation
              {selectedIds.size !== 1 ? 's' : ''}? This action cannot be undone.
            </p>
            <div className="flex justify-end gap-3">
              <button
                onClick={() => setConfirmOpen(false)}
                className="px-4 py-2 text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-100 transition-colors"
              >
                Cancel
              </button>
              <button
                onClick={processRevocation}
                disabled={isProcessing}
                className="px-4 py-2 bg-red-600 hover:bg-red-700 disabled:bg-gray-400 text-white rounded-lg transition-colors"
              >
                {isProcessing ? 'Revoking...' : 'Revoke'}
              </button>
            </div>
          </div>
        </>
      )}
    </div>
  );
}

export default {
  BatchAttestationCreator,
  TrustNetworkExportImport,
  BatchRevocation,
};
