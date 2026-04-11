// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Key Manager Component
 *
 * UI for managing PGP/GPG encryption keys:
 * - Generate new key pairs
 * - Import/export keys
 * - Key trust management
 * - Unlock/lock keys
 */

import { useState, useCallback } from 'react';
import {
  useKeyStore,
  useEncryption,
  useKeyManagement,
  useKeyUnlock,
  type PublicKey,
  type PrivateKey,
  type KeyGenOptions,
} from '@/lib/encryption';

// ============================================
// Key Manager Main Component
// ============================================

export function KeyManager() {
  const [activeTab, setActiveTab] = useState<'my-keys' | 'contacts' | 'generate' | 'import'>('my-keys');
  const { privateKeys, publicKeys, defaultKeyId } = useEncryption();

  const myKeys = privateKeys;
  const contactKeys = publicKeys.filter(
    (pk) => !privateKeys.some((sk) => sk.fingerprint === pk.fingerprint)
  );

  return (
    <div className="max-w-4xl mx-auto p-6">
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-gray-900 dark:text-gray-100">
          Encryption Keys
        </h1>
        <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
          Manage your PGP keys for end-to-end encrypted email
        </p>
      </div>

      {/* Tabs */}
      <div className="border-b border-gray-200 dark:border-gray-700 mb-6">
        <nav className="flex space-x-8">
          {[
            { id: 'my-keys', label: 'My Keys', count: myKeys.length },
            { id: 'contacts', label: 'Contact Keys', count: contactKeys.length },
            { id: 'generate', label: 'Generate' },
            { id: 'import', label: 'Import' },
          ].map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id as typeof activeTab)}
              className={`py-3 px-1 border-b-2 font-medium text-sm transition-colors ${
                activeTab === tab.id
                  ? 'border-blue-500 text-blue-600 dark:text-blue-400'
                  : 'border-transparent text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300'
              }`}
            >
              {tab.label}
              {tab.count !== undefined && (
                <span className="ml-2 px-2 py-0.5 text-xs rounded-full bg-gray-100 dark:bg-gray-800">
                  {tab.count}
                </span>
              )}
            </button>
          ))}
        </nav>
      </div>

      {/* Tab Content */}
      {activeTab === 'my-keys' && (
        <MyKeysTab keys={myKeys} defaultKeyId={defaultKeyId} />
      )}
      {activeTab === 'contacts' && (
        <ContactKeysTab keys={contactKeys} />
      )}
      {activeTab === 'generate' && (
        <GenerateKeyTab />
      )}
      {activeTab === 'import' && (
        <ImportKeyTab />
      )}
    </div>
  );
}

// ============================================
// My Keys Tab
// ============================================

function MyKeysTab({ keys, defaultKeyId }: { keys: PrivateKey[]; defaultKeyId: string | null }) {
  const { setDefaultKey, deleteKey } = useKeyManagement();
  const [expandedKey, setExpandedKey] = useState<string | null>(null);

  if (keys.length === 0) {
    return (
      <div className="text-center py-12">
        <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-gray-100 dark:bg-gray-800 flex items-center justify-center">
          <KeyIcon className="w-8 h-8 text-gray-400" />
        </div>
        <h3 className="text-lg font-medium text-gray-900 dark:text-gray-100 mb-2">
          No keys yet
        </h3>
        <p className="text-sm text-gray-600 dark:text-gray-400 mb-4">
          Generate or import a key to start using encrypted email
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {keys.map((key) => (
        <PrivateKeyCard
          key={key.fingerprint}
          keyData={key}
          isDefault={key.fingerprint === defaultKeyId}
          isExpanded={expandedKey === key.fingerprint}
          onToggleExpand={() => setExpandedKey(
            expandedKey === key.fingerprint ? null : key.fingerprint
          )}
          onSetDefault={() => setDefaultKey(key.fingerprint)}
          onDelete={() => deleteKey(key.fingerprint)}
        />
      ))}
    </div>
  );
}

// ============================================
// Private Key Card
// ============================================

function PrivateKeyCard({
  keyData,
  isDefault,
  isExpanded,
  onToggleExpand,
  onSetDefault,
  onDelete,
}: {
  keyData: PrivateKey;
  isDefault: boolean;
  isExpanded: boolean;
  onToggleExpand: () => void;
  onSetDefault: () => void;
  onDelete: () => void;
}) {
  const { isUnlocked, isUnlocking, error, unlock } = useKeyUnlock(keyData.fingerprint);
  const [passphrase, setPassphrase] = useState('');
  const [showPassphrase, setShowPassphrase] = useState(false);
  const lockKey = useKeyStore((s) => s.lockKey);

  const handleUnlock = async () => {
    const success = await unlock(passphrase);
    if (success) {
      setPassphrase('');
    }
  };

  return (
    <div className="border border-gray-200 dark:border-gray-700 rounded-lg overflow-hidden">
      {/* Header */}
      <div
        className="p-4 bg-white dark:bg-gray-800 cursor-pointer hover:bg-gray-50 dark:hover:bg-gray-750"
        onClick={onToggleExpand}
      >
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className={`w-10 h-10 rounded-full flex items-center justify-center ${
              isUnlocked
                ? 'bg-emerald-100 dark:bg-emerald-900/30'
                : 'bg-gray-100 dark:bg-gray-700'
            }`}>
              {isUnlocked ? (
                <UnlockedIcon className="w-5 h-5 text-emerald-600 dark:text-emerald-400" />
              ) : (
                <LockedIcon className="w-5 h-5 text-gray-500 dark:text-gray-400" />
              )}
            </div>
            <div>
              <div className="flex items-center space-x-2">
                <span className="font-medium text-gray-900 dark:text-gray-100">
                  {keyData.userId}
                </span>
                {isDefault && (
                  <span className="px-2 py-0.5 text-xs font-medium rounded-full bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300">
                    Default
                  </span>
                )}
              </div>
              <div className="text-sm text-gray-500 dark:text-gray-400">
                {keyData.email}
              </div>
            </div>
          </div>
          <div className="flex items-center space-x-2">
            <span className="text-xs text-gray-500 dark:text-gray-400">
              {keyData.algorithm} {keyData.keySize}
            </span>
            <ChevronIcon
              className={`w-5 h-5 text-gray-400 transition-transform ${
                isExpanded ? 'rotate-180' : ''
              }`}
            />
          </div>
        </div>
      </div>

      {/* Expanded Content */}
      {isExpanded && (
        <div className="border-t border-gray-200 dark:border-gray-700 p-4 bg-gray-50 dark:bg-gray-900">
          {/* Fingerprint */}
          <div className="mb-4">
            <label className="text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
              Fingerprint
            </label>
            <div className="mt-1 font-mono text-sm text-gray-900 dark:text-gray-100 break-all">
              {formatFingerprint(keyData.fingerprint)}
            </div>
          </div>

          {/* Key Details */}
          <div className="grid grid-cols-2 gap-4 mb-4">
            <div>
              <label className="text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                Created
              </label>
              <div className="mt-1 text-sm text-gray-900 dark:text-gray-100">
                {new Date(keyData.createdAt).toLocaleDateString()}
              </div>
            </div>
            <div>
              <label className="text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                Expires
              </label>
              <div className="mt-1 text-sm text-gray-900 dark:text-gray-100">
                {keyData.expiresAt
                  ? new Date(keyData.expiresAt).toLocaleDateString()
                  : 'Never'}
              </div>
            </div>
          </div>

          {/* Unlock/Lock */}
          {!isUnlocked ? (
            <div className="mb-4">
              <label className="text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                Unlock Key
              </label>
              <div className="mt-1 flex space-x-2">
                <div className="relative flex-1">
                  <input
                    type={showPassphrase ? 'text' : 'password'}
                    value={passphrase}
                    onChange={(e) => setPassphrase(e.target.value)}
                    placeholder="Enter passphrase"
                    className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100 pr-10"
                    onKeyDown={(e) => e.key === 'Enter' && handleUnlock()}
                  />
                  <button
                    type="button"
                    onClick={() => setShowPassphrase(!showPassphrase)}
                    className="absolute right-2 top-1/2 -translate-y-1/2 text-gray-400 hover:text-gray-600"
                  >
                    {showPassphrase ? <EyeOffIcon className="w-5 h-5" /> : <EyeIcon className="w-5 h-5" />}
                  </button>
                </div>
                <button
                  onClick={handleUnlock}
                  disabled={isUnlocking || !passphrase}
                  className="px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {isUnlocking ? 'Unlocking...' : 'Unlock'}
                </button>
              </div>
              {error && (
                <p className="mt-1 text-sm text-red-600 dark:text-red-400">{error}</p>
              )}
            </div>
          ) : (
            <div className="mb-4 p-3 bg-emerald-50 dark:bg-emerald-900/20 rounded-lg flex items-center justify-between">
              <div className="flex items-center space-x-2">
                <UnlockedIcon className="w-5 h-5 text-emerald-600 dark:text-emerald-400" />
                <span className="text-sm text-emerald-700 dark:text-emerald-300">
                  Key is unlocked and ready for use
                </span>
              </div>
              <button
                onClick={() => lockKey(keyData.fingerprint)}
                className="text-sm text-emerald-600 dark:text-emerald-400 hover:underline"
              >
                Lock now
              </button>
            </div>
          )}

          {/* Actions */}
          <div className="flex items-center justify-between pt-4 border-t border-gray-200 dark:border-gray-700">
            <div className="flex space-x-2">
              {!isDefault && (
                <button
                  onClick={onSetDefault}
                  className="px-3 py-1.5 text-sm font-medium text-blue-600 dark:text-blue-400 hover:bg-blue-50 dark:hover:bg-blue-900/20 rounded-lg"
                >
                  Set as Default
                </button>
              )}
              <button className="px-3 py-1.5 text-sm font-medium text-gray-600 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-800 rounded-lg">
                Export Public Key
              </button>
              <button className="px-3 py-1.5 text-sm font-medium text-gray-600 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-800 rounded-lg">
                Publish to Keyserver
              </button>
            </div>
            <button
              onClick={onDelete}
              className="px-3 py-1.5 text-sm font-medium text-red-600 dark:text-red-400 hover:bg-red-50 dark:hover:bg-red-900/20 rounded-lg"
            >
              Delete Key
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

// ============================================
// Contact Keys Tab
// ============================================

function ContactKeysTab({ keys }: { keys: PublicKey[] }) {
  const { setTrustLevel, deleteKey } = useKeyManagement();
  const [searchQuery, setSearchQuery] = useState('');

  const filteredKeys = keys.filter(
    (key) =>
      key.userId.toLowerCase().includes(searchQuery.toLowerCase()) ||
      key.email.toLowerCase().includes(searchQuery.toLowerCase()) ||
      key.fingerprint.toLowerCase().includes(searchQuery.toLowerCase())
  );

  return (
    <div>
      {/* Search */}
      <div className="mb-4">
        <input
          type="text"
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          placeholder="Search by name, email, or fingerprint..."
          className="w-full px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100"
        />
      </div>

      {filteredKeys.length === 0 ? (
        <div className="text-center py-12">
          <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-gray-100 dark:bg-gray-800 flex items-center justify-center">
            <UsersIcon className="w-8 h-8 text-gray-400" />
          </div>
          <h3 className="text-lg font-medium text-gray-900 dark:text-gray-100 mb-2">
            {searchQuery ? 'No matching keys' : 'No contact keys'}
          </h3>
          <p className="text-sm text-gray-600 dark:text-gray-400">
            {searchQuery
              ? 'Try a different search term'
              : 'Import public keys from your contacts to send them encrypted email'}
          </p>
        </div>
      ) : (
        <div className="space-y-3">
          {filteredKeys.map((key) => (
            <PublicKeyCard
              key={key.fingerprint}
              keyData={key}
              onSetTrust={(level) => setTrustLevel(key.fingerprint, level)}
              onDelete={() => deleteKey(key.fingerprint)}
            />
          ))}
        </div>
      )}
    </div>
  );
}

// ============================================
// Public Key Card
// ============================================

function PublicKeyCard({
  keyData,
  onSetTrust,
  onDelete,
}: {
  keyData: PublicKey;
  onSetTrust: (level: PublicKey['trustLevel']) => void;
  onDelete: () => void;
}) {
  const [showDetails, setShowDetails] = useState(false);

  const trustColors: Record<PublicKey['trustLevel'], string> = {
    unknown: 'bg-gray-100 text-gray-700 dark:bg-gray-700 dark:text-gray-300',
    never: 'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-300',
    marginal: 'bg-yellow-100 text-yellow-700 dark:bg-yellow-900/30 dark:text-yellow-300',
    full: 'bg-emerald-100 text-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-300',
    ultimate: 'bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-300',
  };

  return (
    <div className="border border-gray-200 dark:border-gray-700 rounded-lg p-4 bg-white dark:bg-gray-800">
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <div className="w-10 h-10 rounded-full bg-gray-100 dark:bg-gray-700 flex items-center justify-center">
            <span className="text-lg font-medium text-gray-600 dark:text-gray-300">
              {keyData.userId.charAt(0).toUpperCase()}
            </span>
          </div>
          <div>
            <div className="font-medium text-gray-900 dark:text-gray-100">
              {keyData.userId}
            </div>
            <div className="text-sm text-gray-500 dark:text-gray-400">
              {keyData.email}
            </div>
          </div>
        </div>
        <div className="flex items-center space-x-2">
          <select
            value={keyData.trustLevel}
            onChange={(e) => onSetTrust(e.target.value as PublicKey['trustLevel'])}
            className={`px-2 py-1 text-xs font-medium rounded-full border-0 cursor-pointer ${trustColors[keyData.trustLevel]}`}
          >
            <option value="unknown">Unknown</option>
            <option value="never">Never Trust</option>
            <option value="marginal">Marginal</option>
            <option value="full">Full Trust</option>
            <option value="ultimate">Ultimate</option>
          </select>
          <button
            onClick={() => setShowDetails(!showDetails)}
            className="p-1 text-gray-400 hover:text-gray-600 dark:hover:text-gray-300"
          >
            <ChevronIcon className={`w-5 h-5 transition-transform ${showDetails ? 'rotate-180' : ''}`} />
          </button>
        </div>
      </div>

      {showDetails && (
        <div className="mt-4 pt-4 border-t border-gray-200 dark:border-gray-700">
          <div className="text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider mb-1">
            Fingerprint
          </div>
          <div className="font-mono text-sm text-gray-900 dark:text-gray-100 mb-3">
            {formatFingerprint(keyData.fingerprint)}
          </div>
          <div className="flex items-center justify-between text-sm">
            <span className="text-gray-500 dark:text-gray-400">
              {keyData.algorithm} {keyData.keySize} • Created {new Date(keyData.createdAt).toLocaleDateString()}
            </span>
            <button
              onClick={onDelete}
              className="text-red-600 dark:text-red-400 hover:underline"
            >
              Remove
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

// ============================================
// Generate Key Tab
// ============================================

function GenerateKeyTab() {
  const { generateKeyPair } = useKeyManagement();
  const [isGenerating, setIsGenerating] = useState(false);
  const [success, setSuccess] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [formData, setFormData] = useState<KeyGenOptions>({
    type: 'ECC',
    curve: 'curve25519',
    userId: '',
    email: '',
    passphrase: '',
    expirationDays: 365 * 2,
  });
  const [confirmPassphrase, setConfirmPassphrase] = useState('');

  const handleGenerate = async () => {
    if (formData.passphrase !== confirmPassphrase) {
      setError('Passphrases do not match');
      return;
    }

    if (formData.passphrase.length < 8) {
      setError('Passphrase must be at least 8 characters');
      return;
    }

    setIsGenerating(true);
    setError(null);

    try {
      await generateKeyPair(formData);
      setSuccess(true);
      setFormData({
        type: 'ECC',
        curve: 'curve25519',
        userId: '',
        email: '',
        passphrase: '',
        expirationDays: 365 * 2,
      });
      setConfirmPassphrase('');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to generate key');
    } finally {
      setIsGenerating(false);
    }
  };

  if (success) {
    return (
      <div className="text-center py-12">
        <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-emerald-100 dark:bg-emerald-900/30 flex items-center justify-center">
          <CheckIcon className="w-8 h-8 text-emerald-600 dark:text-emerald-400" />
        </div>
        <h3 className="text-lg font-medium text-gray-900 dark:text-gray-100 mb-2">
          Key Generated Successfully
        </h3>
        <p className="text-sm text-gray-600 dark:text-gray-400 mb-4">
          Your new encryption key is ready to use
        </p>
        <button
          onClick={() => setSuccess(false)}
          className="px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600"
        >
          Generate Another
        </button>
      </div>
    );
  }

  return (
    <div className="max-w-lg">
      <div className="space-y-4">
        {/* Name */}
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
            Your Name
          </label>
          <input
            type="text"
            value={formData.userId}
            onChange={(e) => setFormData({ ...formData, userId: e.target.value })}
            placeholder="John Doe"
            className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100"
          />
        </div>

        {/* Email */}
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
            Email Address
          </label>
          <input
            type="email"
            value={formData.email}
            onChange={(e) => setFormData({ ...formData, email: e.target.value })}
            placeholder="john@example.com"
            className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100"
          />
        </div>

        {/* Key Type */}
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
            Key Type
          </label>
          <div className="grid grid-cols-2 gap-2">
            <button
              type="button"
              onClick={() => setFormData({ ...formData, type: 'ECC', curve: 'curve25519' })}
              className={`p-3 rounded-lg border text-left ${
                formData.type === 'ECC'
                  ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20'
                  : 'border-gray-300 dark:border-gray-600'
              }`}
            >
              <div className="font-medium text-gray-900 dark:text-gray-100">ECC (Recommended)</div>
              <div className="text-xs text-gray-500 dark:text-gray-400">Modern, fast, secure</div>
            </button>
            <button
              type="button"
              onClick={() => setFormData({ ...formData, type: 'RSA', rsaBits: 4096 })}
              className={`p-3 rounded-lg border text-left ${
                formData.type === 'RSA'
                  ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20'
                  : 'border-gray-300 dark:border-gray-600'
              }`}
            >
              <div className="font-medium text-gray-900 dark:text-gray-100">RSA</div>
              <div className="text-xs text-gray-500 dark:text-gray-400">Classic, widely supported</div>
            </button>
          </div>
        </div>

        {/* Expiration */}
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
            Key Expiration
          </label>
          <select
            value={formData.expirationDays}
            onChange={(e) => setFormData({ ...formData, expirationDays: Number(e.target.value) || undefined })}
            className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100"
          >
            <option value={365}>1 year</option>
            <option value={365 * 2}>2 years</option>
            <option value={365 * 5}>5 years</option>
            <option value="">Never</option>
          </select>
        </div>

        {/* Passphrase */}
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
            Passphrase
          </label>
          <input
            type="password"
            value={formData.passphrase}
            onChange={(e) => setFormData({ ...formData, passphrase: e.target.value })}
            placeholder="Strong passphrase to protect your key"
            className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100"
          />
          <p className="mt-1 text-xs text-gray-500 dark:text-gray-400">
            Choose a strong passphrase you'll remember. You'll need it to unlock your key.
          </p>
        </div>

        {/* Confirm Passphrase */}
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
            Confirm Passphrase
          </label>
          <input
            type="password"
            value={confirmPassphrase}
            onChange={(e) => setConfirmPassphrase(e.target.value)}
            placeholder="Confirm your passphrase"
            className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100"
          />
        </div>

        {error && (
          <div className="p-3 bg-red-50 dark:bg-red-900/20 rounded-lg text-sm text-red-600 dark:text-red-400">
            {error}
          </div>
        )}

        <button
          onClick={handleGenerate}
          disabled={isGenerating || !formData.userId || !formData.email || !formData.passphrase}
          className="w-full py-3 bg-blue-500 text-white rounded-lg hover:bg-blue-600 disabled:opacity-50 disabled:cursor-not-allowed font-medium"
        >
          {isGenerating ? 'Generating Key...' : 'Generate Key Pair'}
        </button>
      </div>
    </div>
  );
}

// ============================================
// Import Key Tab
// ============================================

function ImportKeyTab() {
  const { importPublicKey, importPrivateKey } = useKeyManagement();
  const [keyType, setKeyType] = useState<'public' | 'private'>('public');
  const [keyText, setKeyText] = useState('');
  const [passphrase, setPassphrase] = useState('');
  const [isImporting, setIsImporting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState(false);

  const handleImport = async () => {
    setIsImporting(true);
    setError(null);

    try {
      if (keyType === 'public') {
        await importPublicKey(keyText);
      } else {
        await importPrivateKey(keyText, passphrase);
      }
      setSuccess(true);
      setKeyText('');
      setPassphrase('');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to import key');
    } finally {
      setIsImporting(false);
    }
  };

  const handleFileImport = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const text = await file.text();
    setKeyText(text);
  };

  if (success) {
    return (
      <div className="text-center py-12">
        <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-emerald-100 dark:bg-emerald-900/30 flex items-center justify-center">
          <CheckIcon className="w-8 h-8 text-emerald-600 dark:text-emerald-400" />
        </div>
        <h3 className="text-lg font-medium text-gray-900 dark:text-gray-100 mb-2">
          Key Imported Successfully
        </h3>
        <button
          onClick={() => setSuccess(false)}
          className="px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600"
        >
          Import Another
        </button>
      </div>
    );
  }

  return (
    <div className="max-w-lg">
      <div className="space-y-4">
        {/* Key Type Selection */}
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
            Key Type
          </label>
          <div className="grid grid-cols-2 gap-2">
            <button
              type="button"
              onClick={() => setKeyType('public')}
              className={`p-3 rounded-lg border text-center ${
                keyType === 'public'
                  ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20'
                  : 'border-gray-300 dark:border-gray-600'
              }`}
            >
              <div className="font-medium text-gray-900 dark:text-gray-100">Public Key</div>
              <div className="text-xs text-gray-500 dark:text-gray-400">For encrypting to others</div>
            </button>
            <button
              type="button"
              onClick={() => setKeyType('private')}
              className={`p-3 rounded-lg border text-center ${
                keyType === 'private'
                  ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20'
                  : 'border-gray-300 dark:border-gray-600'
              }`}
            >
              <div className="font-medium text-gray-900 dark:text-gray-100">Private Key</div>
              <div className="text-xs text-gray-500 dark:text-gray-400">Your personal key</div>
            </button>
          </div>
        </div>

        {/* File Upload */}
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
            Import from File
          </label>
          <input
            type="file"
            accept=".asc,.gpg,.pgp,.key"
            onChange={handleFileImport}
            className="w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-lg file:border-0 file:text-sm file:font-medium file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100 dark:file:bg-blue-900/30 dark:file:text-blue-300"
          />
        </div>

        {/* Key Text */}
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
            Or Paste Key
          </label>
          <textarea
            value={keyText}
            onChange={(e) => setKeyText(e.target.value)}
            placeholder={`-----BEGIN PGP ${keyType.toUpperCase()} KEY BLOCK-----\n...\n-----END PGP ${keyType.toUpperCase()} KEY BLOCK-----`}
            rows={8}
            className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100 font-mono text-sm"
          />
        </div>

        {/* Passphrase for private key */}
        {keyType === 'private' && (
          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
              Key Passphrase
            </label>
            <input
              type="password"
              value={passphrase}
              onChange={(e) => setPassphrase(e.target.value)}
              placeholder="Enter the key's passphrase"
              className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100"
            />
          </div>
        )}

        {error && (
          <div className="p-3 bg-red-50 dark:bg-red-900/20 rounded-lg text-sm text-red-600 dark:text-red-400">
            {error}
          </div>
        )}

        <button
          onClick={handleImport}
          disabled={isImporting || !keyText || (keyType === 'private' && !passphrase)}
          className="w-full py-3 bg-blue-500 text-white rounded-lg hover:bg-blue-600 disabled:opacity-50 disabled:cursor-not-allowed font-medium"
        >
          {isImporting ? 'Importing...' : 'Import Key'}
        </button>
      </div>
    </div>
  );
}

// ============================================
// Utility Functions
// ============================================

function formatFingerprint(fingerprint: string): string {
  return fingerprint.match(/.{1,4}/g)?.join(' ') || fingerprint;
}

// ============================================
// Icons
// ============================================

function KeyIcon({ className }: { className?: string }) {
  return (
    <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 7a2 2 0 012 2m4 0a6 6 0 01-7.743 5.743L11 17H9v2H7v2H4a1 1 0 01-1-1v-2.586a1 1 0 01.293-.707l5.964-5.964A6 6 0 1121 9z" />
    </svg>
  );
}

function LockedIcon({ className }: { className?: string }) {
  return (
    <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" />
    </svg>
  );
}

function UnlockedIcon({ className }: { className?: string }) {
  return (
    <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 11V7a4 4 0 118 0m-4 8v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2z" />
    </svg>
  );
}

function ChevronIcon({ className }: { className?: string }) {
  return (
    <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
    </svg>
  );
}

function EyeIcon({ className }: { className?: string }) {
  return (
    <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
    </svg>
  );
}

function EyeOffIcon({ className }: { className?: string }) {
  return (
    <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13.875 18.825A10.05 10.05 0 0112 19c-4.478 0-8.268-2.943-9.543-7a9.97 9.97 0 011.563-3.029m5.858.908a3 3 0 114.243 4.243M9.878 9.878l4.242 4.242M9.88 9.88l-3.29-3.29m7.532 7.532l3.29 3.29M3 3l3.59 3.59m0 0A9.953 9.953 0 0112 5c4.478 0 8.268 2.943 9.543 7a10.025 10.025 0 01-4.132 5.411m0 0L21 21" />
    </svg>
  );
}

function UsersIcon({ className }: { className?: string }) {
  return (
    <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4.354a4 4 0 110 5.292M15 21H3v-1a6 6 0 0112 0v1zm0 0h6v-1a6 6 0 00-9-5.197M13 7a4 4 0 11-8 0 4 4 0 018 0z" />
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

export default KeyManager;
