// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * CredentialCard Component
 *
 * Displays a W3C Verifiable Credential in a card format
 * Memoized to prevent unnecessary re-renders when props haven't changed
 */

import React from 'react';
import type { VerifiableCredential } from '../data/mockCredentials';
import { getScoreBandColor, getCredentialTypeIcon } from '../data/mockCredentials';

interface CredentialCardProps {
  credential: VerifiableCredential;
  onClick?: (credential: VerifiableCredential) => void;
}

export const CredentialCard = React.memo<CredentialCardProps>(({ credential, onClick }) => {
  const handleClick = () => {
    if (onClick) {
      onClick(credential);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (onClick && (e.key === 'Enter' || e.key === ' ')) {
      e.preventDefault();
      onClick(credential);
    }
  };

  const isAchievement = credential.type.includes('EduAchievementCredential');
  const icon = getCredentialTypeIcon(credential.type);
  const scoreBandColor = getScoreBandColor(credential.credentialSubject.scoreBand);

  const credentialType = credential.type[credential.type.length - 1]
    .replace('Edu', '')
    .replace('Credential', '');

  return (
    <div
      className="credential-card"
      onClick={handleClick}
      onKeyDown={handleKeyDown}
      role={onClick ? 'button' : undefined}
      tabIndex={onClick ? 0 : undefined}
      aria-label={`${credentialType} credential for ${credential.credentialSubject.courseId}. ${isAchievement ? `Score: ${credential.credentialSubject.score.toFixed(1)}. ` : ''}${credential.proof ? 'Verified. ' : ''}Click for details.`}
      style={{
        border: '2px solid #e5e7eb',
        borderRadius: '12px',
        padding: '24px',
        cursor: onClick ? 'pointer' : 'default',
        transition: 'all 0.2s ease',
        backgroundColor: '#ffffff',
        boxShadow: '0 2px 6px rgba(0, 0, 0, 0.1)',
        position: 'relative',
        overflow: 'hidden',
      }}
      onMouseEnter={(e) => {
        if (onClick) {
          e.currentTarget.style.boxShadow = '0 6px 16px rgba(0, 0, 0, 0.15)';
          e.currentTarget.style.transform = 'translateY(-4px)';
          e.currentTarget.style.borderColor = '#3b82f6';
        }
      }}
      onMouseLeave={(e) => {
        e.currentTarget.style.boxShadow = '0 2px 6px rgba(0, 0, 0, 0.1)';
        e.currentTarget.style.transform = 'translateY(0)';
        e.currentTarget.style.borderColor = '#e5e7eb';
      }}
    >
      {/* Top Accent Bar */}
      <div
        style={{
          position: 'absolute',
          top: 0,
          left: 0,
          right: 0,
          height: '4px',
          background: `linear-gradient(90deg, ${scoreBandColor}, transparent)`,
        }}
      />

      {/* Header */}
      <div style={{ marginBottom: '16px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '8px' }}>
          <span style={{ fontSize: '24px' }}>{icon}</span>
          <h3
            style={{
              margin: 0,
              fontSize: '18px',
              fontWeight: '600',
              color: '#111827',
            }}
          >
            {credential.type[credential.type.length - 1]
              .replace('Edu', '')
              .replace('Credential', '')}
          </h3>
        </div>
        <p style={{ margin: 0, fontSize: '13px', color: '#6b7280' }}>
          Course: {credential.credentialSubject.courseId}
        </p>
      </div>

      {/* Score (if achievement) */}
      {isAchievement && credential.credentialSubject.score > 0 && (
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '12px',
            marginBottom: '16px',
          }}
        >
          <div
            style={{
              fontSize: '32px',
              fontWeight: '700',
              color: scoreBandColor,
            }}
          >
            {credential.credentialSubject.score.toFixed(1)}
          </div>
          <div>
            <div
              style={{
                fontSize: '18px',
                fontWeight: '600',
                padding: '4px 12px',
                borderRadius: '6px',
                backgroundColor: scoreBandColor + '20',
                color: scoreBandColor,
                display: 'inline-block',
              }}
            >
              {credential.credentialSubject.scoreBand}
            </div>
          </div>
        </div>
      )}

      {/* Skills */}
      <div style={{ marginBottom: '16px' }}>
        <div style={{ fontSize: '12px', fontWeight: '600', color: '#6b7280', marginBottom: '8px' }}>
          Skills Acquired
        </div>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px' }}>
          {credential.credentialSubject.skills.map((skill, index) => (
            <span
              key={index}
              style={{
                fontSize: '12px',
                padding: '4px 10px',
                borderRadius: '6px',
                backgroundColor: '#f3f4f6',
                color: '#4b5563',
              }}
            >
              {skill}
            </span>
          ))}
        </div>
      </div>

      {/* FL Contributions */}
      {credential.credentialSubject.flContributions && (
        <div
          style={{
            padding: '8px 12px',
            backgroundColor: '#ecfdf5',
            borderRadius: '6px',
            marginBottom: '16px',
          }}
        >
          <span style={{ fontSize: '12px', color: '#059669' }}>
            🤝 Contributed to {credential.credentialSubject.flContributions} FL round
            {credential.credentialSubject.flContributions !== 1 ? 's' : ''}
          </span>
        </div>
      )}

      {/* Footer */}
      <div
        style={{
          paddingTop: '16px',
          borderTop: '1px solid #e5e7eb',
          fontSize: '12px',
          color: '#9ca3af',
        }}
      >
        <div style={{ marginBottom: '4px' }}>
          Completed: {credential.credentialSubject.completionDate}
        </div>
        <div>
          Issued by: {credential.issuer.slice(0, 20)}...
        </div>
      </div>

      {/* Verified Badge */}
      {credential.proof && (
        <div
          style={{
            position: 'absolute',
            top: '16px',
            right: '16px',
            padding: '4px 10px',
            borderRadius: '12px',
            backgroundColor: '#10b981',
            color: '#ffffff',
            fontSize: '11px',
            fontWeight: '600',
            display: 'flex',
            alignItems: 'center',
            gap: '4px',
          }}
        >
          <span>✓</span>
          <span>Verified</span>
        </div>
      )}
    </div>
  );
});
