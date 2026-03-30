// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Key Management Page
 *
 * PGP key generation, import, and trust management
 */

import React, { useEffect, useState } from 'react';

interface UserKey {
  id: string;
  fingerprint: string;
  algorithm: string;
  keySize: number;
  createdAt: string;
  expiresAt?: string;
  revoked: boolean;
  publishedTo: string[];
}

interface ContactKey {
  id: string;
  email: string;
  fingerprint: string;
  verified: boolean;
  trustLevel: 'untrusted' | 'unknown' | 'marginal' | 'full' | 'ultimate';
  importedAt: string;
}

export default function KeyManagement() {
  const [userKeys, setUserKeys] = useState<UserKey[]>([]);
  const [contactKeys, setContactKeys] = useState<ContactKey[]>([]);
  const [loading, setLoading] = useState(true);
  const [showGenerateModal, setShowGenerateModal] = useState(false);
  const [showImportModal, setShowImportModal] = useState(false);

  useEffect(() => {
    fetchKeys();
  }, []);

  async function fetchKeys() {
    try {
      const [userRes, contactsRes] = await Promise.all([
        fetch('/api/encryption/keys'),
        fetch('/api/encryption/contacts'),
      ]);
      if (userRes.ok) setUserKeys(await userRes.json());
      if (contactsRes.ok) setContactKeys(await contactsRes.json());
    } catch (error) {
      console.error('Failed to fetch keys:', error);
    } finally {
      setLoading(false);
    }
  }

  async function generateKey(params: {
    name: string;
    email: string;
    keySize: number;
    algorithm: string;
    passphrase: string;
  }) {
    try {
      const response = await fetch('/api/encryption/keys/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(params),
      });
      if (response.ok) {
        fetchKeys();
        setShowGenerateModal(false);
      }
    } catch (error) {
      console.error('Failed to generate key:', error);
    }
  }

  async function importKey(armoredKey: string) {
    try {
      const response = await fetch('/api/encryption/keys/import', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ armored_key: armoredKey }),
      });
      if (response.ok) {
        fetchKeys();
        setShowImportModal(false);
      }
    } catch (error) {
      console.error('Failed to import key:', error);
    }
  }

  async function revokeKey(keyId: string) {
    if (!confirm('Revoke this key? This action cannot be undone.')) return;
    try {
      await fetch(`/api/encryption/keys/${keyId}/revoke`, { method: 'POST' });
      fetchKeys();
    } catch (error) {
      console.error('Failed to revoke key:', error);
    }
  }

  async function updateTrust(contactId: string, trustLevel: ContactKey['trustLevel']) {
    try {
      await fetch(`/api/encryption/contacts/${contactId}/trust`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ trust_level: trustLevel }),
      });
      fetchKeys();
    } catch (error) {
      console.error('Failed to update trust:', error);
    }
  }

  async function publishKey(keyId: string) {
    try {
      await fetch(`/api/encryption/keys/${keyId}/publish`, { method: 'POST' });
      fetchKeys();
    } catch (error) {
      console.error('Failed to publish key:', error);
    }
  }

  function formatFingerprint(fp: string): string {
    return fp.match(/.{1,4}/g)?.join(' ') || fp;
  }

  const trustColors = {
    untrusted: 'bg-red-100 text-red-800',
    unknown: 'bg-gray-100 text-gray-800',
    marginal: 'bg-yellow-100 text-yellow-800',
    full: 'bg-green-100 text-green-800',
    ultimate: 'bg-blue-100 text-blue-800',
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary"></div>
      </div>
    );
  }

  return (
    <div className="p-6 max-w-4xl mx-auto">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-2xl font-bold">Encryption Keys</h1>
          <p className="text-muted">Manage your PGP keys for end-to-end encryption</p>
        </div>
        <div className="flex gap-2">
          <button
            onClick={() => setShowImportModal(true)}
            className="px-4 py-2 border border-border rounded-lg hover:bg-muted/30"
          >
            Import Key
          </button>
          <button
            onClick={() => setShowGenerateModal(true)}
            className="px-4 py-2 bg-primary text-white rounded-lg hover:bg-primary/90"
          >
            Generate Key
          </button>
        </div>
      </div>

      {/* Your Keys */}
      <div className="mb-8">
        <h2 className="text-lg font-semibold mb-4">Your Keys</h2>
        {userKeys.length === 0 ? (
          <div className="text-center py-8 bg-surface rounded-lg border border-border">
            <div className="text-4xl mb-4">🔐</div>
            <h3 className="font-medium mb-2">No encryption keys</h3>
            <p className="text-muted mb-4">Generate a key pair to start using encryption</p>
            <button
              onClick={() => setShowGenerateModal(true)}
              className="px-4 py-2 bg-primary text-white rounded-lg"
            >
              Generate Key Pair
            </button>
          </div>
        ) : (
          <div className="space-y-4">
            {userKeys.map((key) => (
              <div
                key={key.id}
                className={`bg-surface border rounded-lg p-4 ${
                  key.revoked ? 'border-red-200 bg-red-50' : 'border-border'
                }`}
              >
                <div className="flex items-start justify-between">
                  <div>
                    <div className="flex items-center gap-2 mb-2">
                      <span className="font-mono text-sm">
                        {formatFingerprint(key.fingerprint)}
                      </span>
                      {key.revoked && (
                        <span className="px-2 py-0.5 bg-red-100 text-red-800 text-xs rounded">
                          Revoked
                        </span>
                      )}
                    </div>
                    <div className="text-sm text-muted space-y-1">
                      <p>
                        {key.algorithm} {key.keySize}-bit
                      </p>
                      <p>Created: {new Date(key.createdAt).toLocaleDateString()}</p>
                      {key.expiresAt && (
                        <p>Expires: {new Date(key.expiresAt).toLocaleDateString()}</p>
                      )}
                      {key.publishedTo.length > 0 && (
                        <p>Published to: {key.publishedTo.join(', ')}</p>
                      )}
                    </div>
                  </div>
                  {!key.revoked && (
                    <div className="flex gap-2">
                      <button
                        onClick={() => publishKey(key.id)}
                        className="px-3 py-1.5 text-sm border border-border rounded hover:bg-muted/30"
                      >
                        Publish
                      </button>
                      <button
                        onClick={() => {
                          navigator.clipboard.writeText(key.fingerprint);
                        }}
                        className="px-3 py-1.5 text-sm border border-border rounded hover:bg-muted/30"
                      >
                        Copy
                      </button>
                      <button
                        onClick={() => revokeKey(key.id)}
                        className="px-3 py-1.5 text-sm text-red-600 border border-red-200 rounded hover:bg-red-50"
                      >
                        Revoke
                      </button>
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Contact Keys */}
      <div>
        <h2 className="text-lg font-semibold mb-4">Contact Keys</h2>
        {contactKeys.length === 0 ? (
          <div className="text-center py-8 bg-surface rounded-lg border border-border">
            <p className="text-muted">No contact keys imported yet</p>
          </div>
        ) : (
          <div className="bg-surface rounded-lg border border-border overflow-hidden">
            <table className="w-full">
              <thead className="bg-muted/30">
                <tr>
                  <th className="px-4 py-3 text-left text-sm font-medium">Email</th>
                  <th className="px-4 py-3 text-left text-sm font-medium">Fingerprint</th>
                  <th className="px-4 py-3 text-left text-sm font-medium">Trust</th>
                  <th className="px-4 py-3 text-left text-sm font-medium">Actions</th>
                </tr>
              </thead>
              <tbody>
                {contactKeys.map((contact) => (
                  <tr key={contact.id} className="border-t border-border">
                    <td className="px-4 py-3">
                      <div className="flex items-center gap-2">
                        {contact.email}
                        {contact.verified && (
                          <span title="Verified" className="text-green-500">
                            ✓
                          </span>
                        )}
                      </div>
                    </td>
                    <td className="px-4 py-3 font-mono text-sm">
                      {contact.fingerprint.slice(-16)}
                    </td>
                    <td className="px-4 py-3">
                      <select
                        value={contact.trustLevel}
                        onChange={(e) =>
                          updateTrust(contact.id, e.target.value as ContactKey['trustLevel'])
                        }
                        className={`px-2 py-1 rounded text-sm ${trustColors[contact.trustLevel]}`}
                      >
                        <option value="untrusted">Untrusted</option>
                        <option value="unknown">Unknown</option>
                        <option value="marginal">Marginal</option>
                        <option value="full">Full</option>
                        <option value="ultimate">Ultimate</option>
                      </select>
                    </td>
                    <td className="px-4 py-3">
                      <button
                        onClick={() => {
                          // Show full key details
                        }}
                        className="text-sm text-primary hover:underline"
                      >
                        View Details
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Generate Key Modal */}
      {showGenerateModal && (
        <GenerateKeyModal
          onGenerate={generateKey}
          onClose={() => setShowGenerateModal(false)}
        />
      )}

      {/* Import Key Modal */}
      {showImportModal && (
        <ImportKeyModal
          onImport={importKey}
          onClose={() => setShowImportModal(false)}
        />
      )}
    </div>
  );
}

function GenerateKeyModal({
  onGenerate,
  onClose,
}: {
  onGenerate: (params: {
    name: string;
    email: string;
    keySize: number;
    algorithm: string;
    passphrase: string;
  }) => void;
  onClose: () => void;
}) {
  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const [keySize, setKeySize] = useState(4096);
  const [algorithm, setAlgorithm] = useState('RSA');
  const [passphrase, setPassphrase] = useState('');
  const [confirmPassphrase, setConfirmPassphrase] = useState('');
  const [generating, setGenerating] = useState(false);

  async function handleGenerate() {
    if (passphrase !== confirmPassphrase) {
      alert('Passphrases do not match');
      return;
    }
    setGenerating(true);
    await onGenerate({ name, email, keySize, algorithm, passphrase });
    setGenerating(false);
  }

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
      <div className="bg-background rounded-lg shadow-xl w-full max-w-md">
        <div className="p-4 border-b border-border">
          <h2 className="text-lg font-semibold">Generate Key Pair</h2>
        </div>

        <div className="p-4 space-y-4">
          <div>
            <label className="block text-sm font-medium mb-1">Name</label>
            <input
              type="text"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="Your Name"
              className="w-full px-3 py-2 border border-border rounded-lg"
            />
          </div>

          <div>
            <label className="block text-sm font-medium mb-1">Email</label>
            <input
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              placeholder="you@example.com"
              className="w-full px-3 py-2 border border-border rounded-lg"
            />
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium mb-1">Algorithm</label>
              <select
                value={algorithm}
                onChange={(e) => setAlgorithm(e.target.value)}
                className="w-full px-3 py-2 border border-border rounded-lg"
              >
                <option value="RSA">RSA</option>
                <option value="Ed25519">Ed25519</option>
                <option value="Curve25519">Curve25519</option>
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium mb-1">Key Size</label>
              <select
                value={keySize}
                onChange={(e) => setKeySize(parseInt(e.target.value))}
                className="w-full px-3 py-2 border border-border rounded-lg"
                disabled={algorithm !== 'RSA'}
              >
                <option value={2048}>2048 bits</option>
                <option value={3072}>3072 bits</option>
                <option value={4096}>4096 bits</option>
              </select>
            </div>
          </div>

          <div>
            <label className="block text-sm font-medium mb-1">Passphrase</label>
            <input
              type="password"
              value={passphrase}
              onChange={(e) => setPassphrase(e.target.value)}
              placeholder="Strong passphrase"
              className="w-full px-3 py-2 border border-border rounded-lg"
            />
          </div>

          <div>
            <label className="block text-sm font-medium mb-1">Confirm Passphrase</label>
            <input
              type="password"
              value={confirmPassphrase}
              onChange={(e) => setConfirmPassphrase(e.target.value)}
              placeholder="Confirm passphrase"
              className="w-full px-3 py-2 border border-border rounded-lg"
            />
          </div>

          <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-3 text-sm">
            <strong>Important:</strong> Your passphrase protects your private key. Store it
            securely - it cannot be recovered if lost.
          </div>
        </div>

        <div className="p-4 border-t border-border flex justify-end gap-2">
          <button
            onClick={onClose}
            className="px-4 py-2 border border-border rounded-lg hover:bg-muted/30"
          >
            Cancel
          </button>
          <button
            onClick={handleGenerate}
            disabled={generating || !name || !email || !passphrase}
            className="px-4 py-2 bg-primary text-white rounded-lg hover:bg-primary/90 disabled:opacity-50"
          >
            {generating ? 'Generating...' : 'Generate Key'}
          </button>
        </div>
      </div>
    </div>
  );
}

function ImportKeyModal({
  onImport,
  onClose,
}: {
  onImport: (armoredKey: string) => void;
  onClose: () => void;
}) {
  const [armoredKey, setArmoredKey] = useState('');

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
      <div className="bg-background rounded-lg shadow-xl w-full max-w-lg">
        <div className="p-4 border-b border-border">
          <h2 className="text-lg font-semibold">Import Public Key</h2>
        </div>

        <div className="p-4">
          <label className="block text-sm font-medium mb-1">
            Paste the armored public key (PGP PUBLIC KEY BLOCK)
          </label>
          <textarea
            value={armoredKey}
            onChange={(e) => setArmoredKey(e.target.value)}
            placeholder="-----BEGIN PGP PUBLIC KEY BLOCK-----
...
-----END PGP PUBLIC KEY BLOCK-----"
            rows={10}
            className="w-full px-3 py-2 border border-border rounded-lg font-mono text-sm"
          />
        </div>

        <div className="p-4 border-t border-border flex justify-end gap-2">
          <button
            onClick={onClose}
            className="px-4 py-2 border border-border rounded-lg hover:bg-muted/30"
          >
            Cancel
          </button>
          <button
            onClick={() => onImport(armoredKey)}
            disabled={!armoredKey.includes('BEGIN PGP PUBLIC KEY BLOCK')}
            className="px-4 py-2 bg-primary text-white rounded-lg hover:bg-primary/90 disabled:opacity-50"
          >
            Import Key
          </button>
        </div>
      </div>
    </div>
  );
}
