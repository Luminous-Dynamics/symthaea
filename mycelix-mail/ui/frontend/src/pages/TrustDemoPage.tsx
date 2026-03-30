// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Trust Demo Page
 *
 * Demonstrates the core Mycelix value proposition:
 * - Trust scoring with MATL
 * - Trust attestations
 * - Cross-hApp reputation
 *
 * This is the "tiny thing end-to-end" that proves the concept.
 */

import React, { useState, useEffect } from 'react';
import {
  useHolochainConnection,
  useContacts,
  useTrustScore,
  useCrossHappReputation,
  useRecordPositiveInteraction,
  useReportSpam,
  useInbox,
} from '../lib/hooks/useHolochainMail';

// Trust Level Badge
function TrustBadge({ score, size = 'medium' }: { score: number; size?: 'small' | 'medium' | 'large' }) {
  const getColor = () => {
    if (score >= 0.8) return '#22c55e'; // green
    if (score >= 0.5) return '#eab308'; // yellow
    if (score >= 0.2) return '#f97316'; // orange
    return '#ef4444'; // red
  };

  const getLabel = () => {
    if (score >= 0.8) return 'Trusted';
    if (score >= 0.5) return 'Known';
    if (score >= 0.2) return 'Caution';
    return 'Unknown';
  };

  const sizes = { small: '1.25rem', medium: '2rem', large: '3rem' };

  return (
    <div
      style={{
        display: 'inline-flex',
        alignItems: 'center',
        gap: '0.5rem',
        padding: '0.25rem 0.75rem',
        borderRadius: '9999px',
        backgroundColor: getColor(),
        color: 'white',
        fontSize: sizes[size],
        fontWeight: 'bold',
      }}
    >
      {score >= 0.8 ? '✓' : score >= 0.5 ? '~' : score >= 0.2 ? '!' : '?'}
      <span style={{ fontSize: '0.875rem' }}>
        {getLabel()} ({Math.round(score * 100)}%)
      </span>
    </div>
  );
}

// Trust Breakdown Card
function TrustBreakdown({ did }: { did: string }) {
  const { data: trust, isLoading } = useTrustScore(did);
  const { data: crossHapp } = useCrossHappReputation(did);

  if (isLoading) {
    return <div style={{ padding: '1rem', color: '#666' }}>Loading trust data...</div>;
  }

  if (!trust) {
    return <div style={{ padding: '1rem', color: '#666' }}>No trust data available</div>;
  }

  return (
    <div
      style={{
        padding: '1.5rem',
        backgroundColor: '#f8fafc',
        borderRadius: '0.75rem',
        border: '1px solid #e2e8f0',
      }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
        <h3 style={{ margin: 0, fontSize: '1.125rem' }}>Trust Analysis</h3>
        <TrustBadge score={trust.score} />
      </div>

      {trust.is_byzantine && (
        <div
          style={{
            padding: '0.75rem',
            backgroundColor: '#fef2f2',
            borderRadius: '0.5rem',
            color: '#dc2626',
            marginBottom: '1rem',
            fontSize: '0.875rem',
          }}
        >
          ⚠️ Byzantine behavior detected - proceed with caution
        </div>
      )}

      <div style={{ marginBottom: '1rem' }}>
        <div style={{ fontSize: '0.875rem', color: '#64748b', marginBottom: '0.5rem' }}>Score Breakdown</div>
        <div style={{ display: 'grid', gap: '0.5rem' }}>
          <ScoreBar label="Direct Trust" value={trust.breakdown.direct} color="#3b82f6" />
          <ScoreBar label="Network Trust" value={trust.breakdown.network} color="#8b5cf6" />
          <ScoreBar label="Behavior" value={trust.breakdown.behavior} color="#06b6d4" />
        </div>
      </div>

      {crossHapp && (
        <div style={{ borderTop: '1px solid #e2e8f0', paddingTop: '1rem' }}>
          <div style={{ fontSize: '0.875rem', color: '#64748b', marginBottom: '0.5rem' }}>Cross-hApp Reputation</div>
          <div style={{ display: 'flex', gap: '1rem', fontSize: '0.875rem' }}>
            <div>
              <span style={{ color: '#64748b' }}>Score:</span>{' '}
              <strong>{Math.round(crossHapp.cross_happ_score * 100)}%</strong>
            </div>
            <div>
              <span style={{ color: '#64748b' }}>Known in:</span>{' '}
              <strong>{crossHapp.happ_count} hApps</strong>
            </div>
            <div>
              <span style={{ color: '#64748b' }}>Confidence:</span>{' '}
              <strong>{Math.round(crossHapp.confidence * 100)}%</strong>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

function ScoreBar({ label, value, color }: { label: string; value: number; color: string }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
      <span style={{ fontSize: '0.75rem', color: '#64748b', width: '6rem' }}>{label}</span>
      <div style={{ flex: 1, height: '0.5rem', backgroundColor: '#e2e8f0', borderRadius: '9999px' }}>
        <div
          style={{
            width: `${Math.min(100, value * 100)}%`,
            height: '100%',
            backgroundColor: color,
            borderRadius: '9999px',
            transition: 'width 0.3s ease',
          }}
        />
      </div>
      <span style={{ fontSize: '0.75rem', fontWeight: 'bold', width: '2.5rem' }}>
        {Math.round(value * 100)}%
      </span>
    </div>
  );
}

// Contact Card with Trust Actions
function ContactCard({
  contact,
  isSelected,
  onSelect,
}: {
  contact: { did: string; name?: string; email: string; trust_score: number };
  isSelected: boolean;
  onSelect: () => void;
}) {
  const recordPositive = useRecordPositiveInteraction();

  return (
    <div
      onClick={onSelect}
      style={{
        padding: '1rem',
        backgroundColor: isSelected ? '#eff6ff' : 'white',
        borderRadius: '0.75rem',
        border: isSelected ? '2px solid #3b82f6' : '1px solid #e2e8f0',
        cursor: 'pointer',
        transition: 'all 0.2s',
      }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
        <div>
          <div style={{ fontWeight: 600 }}>{contact.name || contact.email}</div>
          <div style={{ fontSize: '0.875rem', color: '#64748b' }}>{contact.email}</div>
          <div style={{ fontSize: '0.75rem', color: '#94a3b8', marginTop: '0.25rem' }}>{contact.did}</div>
        </div>
        <TrustBadge score={contact.trust_score} size="small" />
      </div>

      {isSelected && (
        <div style={{ marginTop: '1rem', display: 'flex', gap: '0.5rem' }}>
          <button
            onClick={(e) => {
              e.stopPropagation();
              recordPositive.mutate(contact.did);
            }}
            disabled={recordPositive.isPending}
            style={{
              padding: '0.5rem 1rem',
              backgroundColor: '#22c55e',
              color: 'white',
              border: 'none',
              borderRadius: '0.5rem',
              cursor: 'pointer',
              fontSize: '0.875rem',
            }}
          >
            {recordPositive.isPending ? 'Recording...' : '👍 Vouch for this person'}
          </button>
        </div>
      )}
    </div>
  );
}

// Main Demo Page
export default function TrustDemoPage() {
  const { isConnected, isConnecting, error } = useHolochainConnection({ mockMode: true });
  const { data: contacts = [], isLoading: contactsLoading } = useContacts();
  const { data: inbox = [] } = useInbox();
  const [selectedDid, setSelectedDid] = useState<string | null>(null);

  return (
    <div style={{ minHeight: '100vh', backgroundColor: '#f1f5f9' }}>
      {/* Header */}
      <header
        style={{
          padding: '1.5rem 2rem',
          backgroundColor: 'white',
          borderBottom: '1px solid #e2e8f0',
        }}
      >
        <div style={{ maxWidth: '1200px', margin: '0 auto' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <div>
              <h1 style={{ margin: 0, fontSize: '1.5rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                🍄 Mycelix Trust Demo
              </h1>
              <p style={{ margin: '0.25rem 0 0', color: '#64748b', fontSize: '0.875rem' }}>
                Trust-first communication powered by Holochain
              </p>
            </div>

            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '0.5rem',
                padding: '0.5rem 1rem',
                backgroundColor: isConnected ? '#dcfce7' : isConnecting ? '#fef9c3' : '#fee2e2',
                borderRadius: '9999px',
                fontSize: '0.875rem',
              }}
            >
              <div
                style={{
                  width: '0.5rem',
                  height: '0.5rem',
                  borderRadius: '50%',
                  backgroundColor: isConnected ? '#22c55e' : isConnecting ? '#eab308' : '#ef4444',
                }}
              />
              {isConnecting ? 'Connecting...' : isConnected ? 'Connected (Mock Mode)' : 'Disconnected'}
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main style={{ maxWidth: '1200px', margin: '0 auto', padding: '2rem' }}>
        {error && (
          <div
            style={{
              padding: '1rem',
              backgroundColor: '#fef2f2',
              borderRadius: '0.75rem',
              color: '#dc2626',
              marginBottom: '1.5rem',
            }}
          >
            Connection error: {error.message}
          </div>
        )}

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '2rem' }}>
          {/* Left Column: Contacts */}
          <div>
            <h2 style={{ fontSize: '1.25rem', marginBottom: '1rem' }}>Your Trust Network</h2>
            <p style={{ color: '#64748b', fontSize: '0.875rem', marginBottom: '1rem' }}>
              Select a contact to see their trust breakdown. Vouch for people you trust to improve their score.
            </p>

            {contactsLoading ? (
              <div style={{ color: '#64748b' }}>Loading contacts...</div>
            ) : (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
                {contacts.map((contact) => (
                  <ContactCard
                    key={contact.did}
                    contact={contact}
                    isSelected={selectedDid === contact.did}
                    onSelect={() => setSelectedDid(contact.did)}
                  />
                ))}
              </div>
            )}
          </div>

          {/* Right Column: Trust Details */}
          <div>
            <h2 style={{ fontSize: '1.25rem', marginBottom: '1rem' }}>Trust Analysis</h2>

            {selectedDid ? (
              <TrustBreakdown did={selectedDid} />
            ) : (
              <div
                style={{
                  padding: '3rem',
                  backgroundColor: '#f8fafc',
                  borderRadius: '0.75rem',
                  border: '1px dashed #cbd5e1',
                  textAlign: 'center',
                  color: '#64748b',
                }}
              >
                <div style={{ fontSize: '2rem', marginBottom: '0.5rem' }}>👆</div>
                <div>Select a contact to see their trust analysis</div>
              </div>
            )}

            {/* How It Works */}
            <div
              style={{
                marginTop: '2rem',
                padding: '1.5rem',
                backgroundColor: 'white',
                borderRadius: '0.75rem',
                border: '1px solid #e2e8f0',
              }}
            >
              <h3 style={{ margin: '0 0 1rem', fontSize: '1rem' }}>How MATL Trust Works</h3>
              <ul style={{ margin: 0, paddingLeft: '1.25rem', color: '#64748b', fontSize: '0.875rem' }}>
                <li style={{ marginBottom: '0.5rem' }}>
                  <strong>Direct Trust:</strong> Your personal interactions with this person
                </li>
                <li style={{ marginBottom: '0.5rem' }}>
                  <strong>Network Trust:</strong> What others in your network say about them
                </li>
                <li style={{ marginBottom: '0.5rem' }}>
                  <strong>Behavior:</strong> Patterns like response time, message quality
                </li>
                <li>
                  <strong>Cross-hApp:</strong> Their reputation across the entire Mycelix network
                </li>
              </ul>
            </div>
          </div>
        </div>

        {/* Email Preview with Trust */}
        <div style={{ marginTop: '3rem' }}>
          <h2 style={{ fontSize: '1.25rem', marginBottom: '1rem' }}>Recent Messages (Trust-Filtered)</h2>
          <p style={{ color: '#64748b', fontSize: '0.875rem', marginBottom: '1rem' }}>
            Messages from untrusted senders are filtered automatically. You only see what's worth seeing.
          </p>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
            {inbox.slice(0, 5).map((msg) => (
              <div
                key={msg.id}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '1rem',
                  padding: '1rem',
                  backgroundColor: msg.is_read ? 'white' : '#eff6ff',
                  borderRadius: '0.5rem',
                  border: '1px solid #e2e8f0',
                }}
              >
                <TrustBadge score={msg.trust_score} size="small" />
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div style={{ fontWeight: msg.is_read ? 400 : 600, truncate: true }}>{msg.subject}</div>
                  <div style={{ fontSize: '0.875rem', color: '#64748b' }}>
                    From: {msg.from_did.split(':').pop()}
                  </div>
                </div>
                <div style={{ fontSize: '0.75rem', color: '#94a3b8' }}>
                  {new Date(msg.timestamp).toLocaleDateString()}
                </div>
              </div>
            ))}
          </div>
        </div>
      </main>
    </div>
  );
}
