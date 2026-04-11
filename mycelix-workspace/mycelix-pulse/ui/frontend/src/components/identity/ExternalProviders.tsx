// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * External Identity Provider Integration
 *
 * Connect to external identity systems for enhanced trust:
 * - ENS (Ethereum Name Service)
 * - Lens Protocol
 * - Farcaster
 * - GitHub verification
 * - Domain ownership
 */

import { useState, useCallback } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';

// ============================================
// Types
// ============================================

type ProviderStatus = 'disconnected' | 'connecting' | 'connected' | 'error';

interface IdentityProvider {
  id: string;
  name: string;
  icon: string;
  description: string;
  status: ProviderStatus;
  identifier?: string;
  verifiedAt?: string;
  assuranceBoost: number; // How much this improves assurance level
}

interface VerificationResult {
  success: boolean;
  identifier: string;
  proof: string;
  timestamp: string;
}

// ============================================
// Provider Card Component
// ============================================

interface ProviderCardProps {
  provider: IdentityProvider;
  onConnect: () => void;
  onDisconnect: () => void;
  isConnecting: boolean;
}

function ProviderCard({ provider, onConnect, onDisconnect, isConnecting }: ProviderCardProps) {
  const statusColors: Record<ProviderStatus, string> = {
    disconnected: 'bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-400',
    connecting: 'bg-amber-100 dark:bg-amber-900/30 text-amber-700 dark:text-amber-300',
    connected: 'bg-emerald-100 dark:bg-emerald-900/30 text-emerald-700 dark:text-emerald-300',
    error: 'bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-300',
  };

  const statusLabels: Record<ProviderStatus, string> = {
    disconnected: 'Not connected',
    connecting: 'Connecting...',
    connected: 'Connected',
    error: 'Error',
  };

  return (
    <div className="p-4 bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 shadow-sm">
      <div className="flex items-start gap-4">
        {/* Icon */}
        <div className="text-3xl">{provider.icon}</div>

        {/* Info */}
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2">
            <h4 className="font-semibold text-gray-900 dark:text-gray-100">
              {provider.name}
            </h4>
            <span className={`px-2 py-0.5 text-xs rounded-full ${statusColors[provider.status]}`}>
              {statusLabels[provider.status]}
            </span>
          </div>

          <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
            {provider.description}
          </p>

          {provider.identifier && (
            <div className="mt-2 flex items-center gap-2">
              <span className="text-sm font-mono text-gray-700 dark:text-gray-300">
                {provider.identifier}
              </span>
              {provider.verifiedAt && (
                <span className="text-xs text-gray-500">
                  verified {new Date(provider.verifiedAt).toLocaleDateString()}
                </span>
              )}
            </div>
          )}

          <div className="mt-2 flex items-center gap-2">
            <span className="text-xs text-emerald-600 dark:text-emerald-400">
              +{provider.assuranceBoost}% assurance boost
            </span>
          </div>
        </div>

        {/* Action */}
        <div>
          {provider.status === 'connected' ? (
            <button
              onClick={onDisconnect}
              className="px-3 py-1.5 text-sm text-red-600 dark:text-red-400 hover:bg-red-50 dark:hover:bg-red-900/20 rounded-lg transition-colors"
            >
              Disconnect
            </button>
          ) : (
            <button
              onClick={onConnect}
              disabled={isConnecting}
              className="px-3 py-1.5 text-sm bg-blue-600 hover:bg-blue-700 disabled:bg-gray-400 text-white rounded-lg transition-colors"
            >
              {isConnecting ? 'Connecting...' : 'Connect'}
            </button>
          )}
        </div>
      </div>
    </div>
  );
}

// ============================================
// ENS Verification
// ============================================

interface ENSVerificationProps {
  onVerified: (result: VerificationResult) => void;
}

export function ENSVerification({ onVerified }: ENSVerificationProps) {
  const [ensName, setEnsName] = useState('');
  const [isVerifying, setIsVerifying] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleVerify = useCallback(async () => {
    if (!ensName.endsWith('.eth')) {
      setError('Please enter a valid .eth name');
      return;
    }

    setIsVerifying(true);
    setError(null);

    try {
      // Simulate ENS resolution - replace with actual Web3 call
      await new Promise((resolve) => setTimeout(resolve, 1500));

      // Mock successful verification
      onVerified({
        success: true,
        identifier: ensName,
        proof: `ens:${ensName}:verified`,
        timestamp: new Date().toISOString(),
      });
    } catch (err) {
      setError('Failed to verify ENS name. Please check the name and try again.');
    } finally {
      setIsVerifying(false);
    }
  }, [ensName, onVerified]);

  return (
    <div className="space-y-4">
      <div>
        <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
          ENS Name
        </label>
        <div className="flex gap-2">
          <input
            type="text"
            value={ensName}
            onChange={(e) => setEnsName(e.target.value)}
            placeholder="vitalik.eth"
            className="flex-1 px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-800 focus:ring-2 focus:ring-blue-500"
          />
          <button
            onClick={handleVerify}
            disabled={!ensName || isVerifying}
            className="px-4 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-400 text-white rounded-lg transition-colors"
          >
            {isVerifying ? 'Verifying...' : 'Verify'}
          </button>
        </div>
      </div>

      {error && (
        <p className="text-sm text-red-600 dark:text-red-400">{error}</p>
      )}

      <p className="text-xs text-gray-500">
        Connect your Ethereum wallet to verify ownership of your ENS name.
        This provides cryptographic proof of your identity.
      </p>
    </div>
  );
}

// ============================================
// GitHub Verification
// ============================================

interface GitHubVerificationProps {
  onVerified: (result: VerificationResult) => void;
}

export function GitHubVerification({ onVerified }: GitHubVerificationProps) {
  const [isVerifying, setIsVerifying] = useState(false);
  const [gistCode, setGistCode] = useState<string | null>(null);

  const startVerification = useCallback(() => {
    // Generate verification code
    const code = `mycelix-verify-${Math.random().toString(36).substring(2, 10)}`;
    setGistCode(code);
  }, []);

  const completeVerification = useCallback(async () => {
    if (!gistCode) return;

    setIsVerifying(true);

    try {
      // Simulate GitHub gist check - replace with actual API call
      await new Promise((resolve) => setTimeout(resolve, 2000));

      onVerified({
        success: true,
        identifier: 'github:username',
        proof: gistCode,
        timestamp: new Date().toISOString(),
      });

      setGistCode(null);
    } catch {
      // Handle error
    } finally {
      setIsVerifying(false);
    }
  }, [gistCode, onVerified]);

  return (
    <div className="space-y-4">
      {!gistCode ? (
        <button
          onClick={startVerification}
          className="w-full px-4 py-3 flex items-center justify-center gap-2 bg-gray-900 hover:bg-gray-800 text-white rounded-lg transition-colors"
        >
          <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
            <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/>
          </svg>
          Start GitHub Verification
        </button>
      ) : (
        <div className="space-y-4">
          <div className="p-4 bg-gray-50 dark:bg-gray-800 rounded-lg">
            <h5 className="font-medium text-gray-900 dark:text-gray-100 mb-2">
              Step 1: Create a public gist
            </h5>
            <p className="text-sm text-gray-600 dark:text-gray-400 mb-3">
              Create a new public gist at{' '}
              <a
                href="https://gist.github.com"
                target="_blank"
                rel="noopener noreferrer"
                className="text-blue-600 hover:underline"
              >
                gist.github.com
              </a>{' '}
              with this exact content:
            </p>
            <div className="p-3 bg-gray-900 rounded-lg font-mono text-sm text-emerald-400">
              {gistCode}
            </div>
          </div>

          <div className="p-4 bg-gray-50 dark:bg-gray-800 rounded-lg">
            <h5 className="font-medium text-gray-900 dark:text-gray-100 mb-2">
              Step 2: Verify
            </h5>
            <p className="text-sm text-gray-600 dark:text-gray-400 mb-3">
              Once created, click below to verify your GitHub account.
            </p>
            <button
              onClick={completeVerification}
              disabled={isVerifying}
              className="px-4 py-2 bg-emerald-600 hover:bg-emerald-700 disabled:bg-gray-400 text-white rounded-lg transition-colors"
            >
              {isVerifying ? 'Verifying...' : 'Complete Verification'}
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

// ============================================
// Domain Verification
// ============================================

interface DomainVerificationProps {
  onVerified: (result: VerificationResult) => void;
}

export function DomainVerification({ onVerified }: DomainVerificationProps) {
  const [domain, setDomain] = useState('');
  const [verificationRecord, setVerificationRecord] = useState<string | null>(null);
  const [isVerifying, setIsVerifying] = useState(false);

  const startVerification = useCallback(() => {
    if (!domain) return;

    // Generate DNS TXT record value
    const record = `mycelix-verify=${Math.random().toString(36).substring(2, 15)}`;
    setVerificationRecord(record);
  }, [domain]);

  const completeVerification = useCallback(async () => {
    if (!domain || !verificationRecord) return;

    setIsVerifying(true);

    try {
      // Simulate DNS lookup - replace with actual verification
      await new Promise((resolve) => setTimeout(resolve, 2000));

      onVerified({
        success: true,
        identifier: `domain:${domain}`,
        proof: verificationRecord,
        timestamp: new Date().toISOString(),
      });

      setVerificationRecord(null);
      setDomain('');
    } finally {
      setIsVerifying(false);
    }
  }, [domain, verificationRecord, onVerified]);

  return (
    <div className="space-y-4">
      <div>
        <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
          Domain
        </label>
        <div className="flex gap-2">
          <input
            type="text"
            value={domain}
            onChange={(e) => setDomain(e.target.value)}
            placeholder="example.com"
            disabled={!!verificationRecord}
            className="flex-1 px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-800 focus:ring-2 focus:ring-blue-500 disabled:bg-gray-100 dark:disabled:bg-gray-700"
          />
          {!verificationRecord && (
            <button
              onClick={startVerification}
              disabled={!domain}
              className="px-4 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-400 text-white rounded-lg transition-colors"
            >
              Start
            </button>
          )}
        </div>
      </div>

      {verificationRecord && (
        <div className="p-4 bg-gray-50 dark:bg-gray-800 rounded-lg space-y-4">
          <div>
            <h5 className="font-medium text-gray-900 dark:text-gray-100 mb-2">
              Add this DNS TXT record:
            </h5>
            <div className="p-3 bg-gray-900 rounded-lg">
              <div className="flex items-center justify-between text-sm">
                <span className="text-gray-400">Type:</span>
                <span className="font-mono text-emerald-400">TXT</span>
              </div>
              <div className="flex items-center justify-between text-sm mt-1">
                <span className="text-gray-400">Name:</span>
                <span className="font-mono text-emerald-400">_mycelix</span>
              </div>
              <div className="flex items-center justify-between text-sm mt-1">
                <span className="text-gray-400">Value:</span>
                <span className="font-mono text-emerald-400 break-all">
                  {verificationRecord}
                </span>
              </div>
            </div>
          </div>

          <button
            onClick={completeVerification}
            disabled={isVerifying}
            className="w-full px-4 py-2 bg-emerald-600 hover:bg-emerald-700 disabled:bg-gray-400 text-white rounded-lg transition-colors"
          >
            {isVerifying ? 'Checking DNS...' : 'Verify Domain'}
          </button>
        </div>
      )}
    </div>
  );
}

// ============================================
// Main External Providers Component
// ============================================

export function ExternalIdentityProviders() {
  const [expandedProvider, setExpandedProvider] = useState<string | null>(null);
  const queryClient = useQueryClient();

  // Mock provider data - replace with actual state
  const [providers, setProviders] = useState<IdentityProvider[]>([
    {
      id: 'ens',
      name: 'ENS',
      icon: '🔷',
      description: 'Ethereum Name Service - link your .eth name',
      status: 'disconnected',
      assuranceBoost: 15,
    },
    {
      id: 'lens',
      name: 'Lens Protocol',
      icon: '🌿',
      description: 'Decentralized social identity on Polygon',
      status: 'disconnected',
      assuranceBoost: 10,
    },
    {
      id: 'farcaster',
      name: 'Farcaster',
      icon: '🟣',
      description: 'Decentralized social network identity',
      status: 'disconnected',
      assuranceBoost: 10,
    },
    {
      id: 'github',
      name: 'GitHub',
      icon: '🐙',
      description: 'Developer identity verification',
      status: 'disconnected',
      assuranceBoost: 8,
    },
    {
      id: 'domain',
      name: 'Domain',
      icon: '🌐',
      description: 'Prove ownership of a web domain',
      status: 'disconnected',
      assuranceBoost: 12,
    },
  ]);

  const handleConnect = useCallback((providerId: string) => {
    setExpandedProvider(providerId);
  }, []);

  const handleDisconnect = useCallback((providerId: string) => {
    setProviders((prev) =>
      prev.map((p) =>
        p.id === providerId
          ? { ...p, status: 'disconnected', identifier: undefined, verifiedAt: undefined }
          : p
      )
    );
  }, []);

  const handleVerified = useCallback((providerId: string, result: VerificationResult) => {
    setProviders((prev) =>
      prev.map((p) =>
        p.id === providerId
          ? { ...p, status: 'connected', identifier: result.identifier, verifiedAt: result.timestamp }
          : p
      )
    );
    setExpandedProvider(null);
    queryClient.invalidateQueries({ queryKey: ['credentials'] });
  }, [queryClient]);

  // Calculate total assurance boost
  const totalBoost = providers
    .filter((p) => p.status === 'connected')
    .reduce((sum, p) => sum + p.assuranceBoost, 0);

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100">
            External Identity Providers
          </h3>
          <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
            Connect external accounts to boost your assurance level
          </p>
        </div>
        {totalBoost > 0 && (
          <div className="px-3 py-1.5 bg-emerald-100 dark:bg-emerald-900/30 text-emerald-700 dark:text-emerald-300 rounded-full text-sm font-medium">
            +{totalBoost}% assurance
          </div>
        )}
      </div>

      <div className="space-y-3">
        {providers.map((provider) => (
          <div key={provider.id}>
            <ProviderCard
              provider={provider}
              onConnect={() => handleConnect(provider.id)}
              onDisconnect={() => handleDisconnect(provider.id)}
              isConnecting={expandedProvider === provider.id}
            />

            {/* Expanded verification UI */}
            {expandedProvider === provider.id && (
              <div className="mt-2 ml-16 p-4 bg-gray-50 dark:bg-gray-800/50 rounded-lg border border-gray-200 dark:border-gray-700">
                {provider.id === 'ens' && (
                  <ENSVerification
                    onVerified={(result) => handleVerified('ens', result)}
                  />
                )}
                {provider.id === 'github' && (
                  <GitHubVerification
                    onVerified={(result) => handleVerified('github', result)}
                  />
                )}
                {provider.id === 'domain' && (
                  <DomainVerification
                    onVerified={(result) => handleVerified('domain', result)}
                  />
                )}
                {(provider.id === 'lens' || provider.id === 'farcaster') && (
                  <div className="text-center py-4 text-gray-500">
                    Coming soon - connect your wallet to verify
                  </div>
                )}
              </div>
            )}
          </div>
        ))}
      </div>

      <div className="p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
        <p className="text-sm text-blue-700 dark:text-blue-300">
          <strong>Privacy note:</strong> Your external identities are stored locally and only shared
          when you explicitly attach them to outgoing emails. Verifications use cryptographic proofs
          that don't expose your private keys.
        </p>
      </div>
    </div>
  );
}

export default ExternalIdentityProviders;
