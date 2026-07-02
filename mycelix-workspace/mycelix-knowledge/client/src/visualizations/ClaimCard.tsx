// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * ClaimCard - Comprehensive claim display component
 *
 * @example
 * ```tsx
 * <ClaimCard
 *   claim={claim}
 *   credibility={credibilityScore}
 *   onViewDetails={() => navigate(`/claim/${claim.id}`)}
 *   showRelationships
 * />
 * ```
 */

import React from 'react';
import { CredibilityBadge } from './CredibilityBadge';
import { EpistemicPositionBar } from './EpistemicPositionBar';
import { VerdictIndicator, Verdict } from './VerdictIndicator';

export interface ClaimData {
  id: string;
  content: string;
  classification: {
    empirical: number;
    normative: number;
    mythic: number;
  };
  author?: string;
  createdAt?: number;
  tags?: string[];
  sourceCount?: number;
}

export interface CredibilityData {
  score: number;
  factors?: {
    sourceDiversity: number;
    authorReputation: number;
    temporalConsistency: number;
    crossValidation: number;
  };
  verdict?: Verdict;
  confidence?: number;
}

export interface RelationshipData {
  supports: number;
  contradicts: number;
  refines: number;
  dependsOn: number;
}

export interface ClaimCardProps {
  /** The claim data */
  claim: ClaimData;
  /** Credibility score and factors */
  credibility?: CredibilityData;
  /** Relationship counts */
  relationships?: RelationshipData;
  /** Card variant */
  variant?: 'default' | 'compact' | 'detailed';
  /** Show the E-N-M position bar */
  showEpistemicBar?: boolean;
  /** Show credibility factors breakdown */
  showCredibilityFactors?: boolean;
  /** Show relationship summary */
  showRelationships?: boolean;
  /** Show tags */
  showTags?: boolean;
  /** Callback when clicking "View Details" */
  onViewDetails?: () => void;
  /** Callback when clicking "Verify" */
  onVerify?: () => void;
  /** Whether the card is selected */
  selected?: boolean;
  /** Custom className */
  className?: string;
  /** Inline styles */
  style?: React.CSSProperties;
}

function formatDate(timestamp?: number): string {
  if (!timestamp) return 'Unknown';
  return new Date(timestamp).toLocaleDateString('en-US', {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
  });
}

function truncateText(text: string, maxLength: number): string {
  if (text.length <= maxLength) return text;
  return text.slice(0, maxLength - 3) + '...';
}

export function ClaimCard({
  claim,
  credibility,
  relationships,
  variant = 'default',
  showEpistemicBar = true,
  showCredibilityFactors = false,
  showRelationships = false,
  showTags = true,
  onViewDetails,
  onVerify,
  selected = false,
  className,
  style,
}: ClaimCardProps): JSX.Element {
  const isCompact = variant === 'compact';
  const isDetailed = variant === 'detailed';

  return (
    <div
      className={className}
      style={{
        backgroundColor: '#ffffff',
        borderRadius: '12px',
        border: `2px solid ${selected ? '#3b82f6' : '#e5e7eb'}`,
        padding: isCompact ? '12px' : '16px',
        boxShadow: selected
          ? '0 4px 12px rgba(59, 130, 246, 0.2)'
          : '0 1px 3px rgba(0, 0, 0, 0.1)',
        transition: 'all 0.2s ease',
        cursor: onViewDetails ? 'pointer' : 'default',
        ...style,
      }}
      onClick={onViewDetails}
    >
      {/* Header */}
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'flex-start',
          marginBottom: isCompact ? '8px' : '12px',
        }}
      >
        <div style={{ flex: 1, marginRight: '12px' }}>
          <p
            style={{
              margin: 0,
              fontSize: isCompact ? '14px' : '16px',
              fontWeight: 500,
              color: '#1f2937',
              lineHeight: 1.5,
            }}
          >
            {isCompact ? truncateText(claim.content, 100) : claim.content}
          </p>

          {/* Meta info */}
          {!isCompact && (
            <div
              style={{
                display: 'flex',
                gap: '12px',
                marginTop: '8px',
                fontSize: '12px',
                color: '#6b7280',
              }}
            >
              {claim.author && (
                <span>By: {claim.author.slice(0, 8)}...</span>
              )}
              {claim.createdAt && (
                <span>{formatDate(claim.createdAt)}</span>
              )}
              {claim.sourceCount !== undefined && (
                <span>{claim.sourceCount} sources</span>
              )}
            </div>
          )}
        </div>

        {/* Credibility badge */}
        {credibility && (
          <CredibilityBadge
            score={credibility.score}
            size={isCompact ? 'sm' : 'md'}
            showDescription={isDetailed}
          />
        )}
      </div>

      {/* Verdict */}
      {credibility?.verdict && (
        <div style={{ marginBottom: '12px' }}>
          <VerdictIndicator
            verdict={credibility.verdict}
            confidence={credibility.confidence}
            size="sm"
            showConfidence={isDetailed}
          />
        </div>
      )}

      {/* Epistemic position bar */}
      {showEpistemicBar && (
        <div style={{ marginBottom: isCompact ? '8px' : '12px' }}>
          <EpistemicPositionBar
            empirical={claim.classification.empirical}
            normative={claim.classification.normative}
            mythic={claim.classification.mythic}
            height={isCompact ? 16 : 20}
            showLabels={!isCompact}
            showValues={isDetailed}
          />
        </div>
      )}

      {/* Credibility factors (detailed view) */}
      {showCredibilityFactors && isDetailed && credibility?.factors && (
        <div
          style={{
            marginBottom: '12px',
            padding: '12px',
            backgroundColor: '#f9fafb',
            borderRadius: '8px',
          }}
        >
          <div style={{ fontSize: '12px', fontWeight: 600, marginBottom: '8px', color: '#374151' }}>
            Credibility Factors
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px' }}>
            {Object.entries(credibility.factors).map(([key, value]) => (
              <div key={key} style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ fontSize: '11px', color: '#6b7280', textTransform: 'capitalize' }}>
                  {key.replace(/([A-Z])/g, ' $1').trim()}
                </span>
                <span style={{ fontSize: '11px', fontWeight: 500, color: '#374151' }}>
                  {Math.round(value * 100)}%
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Relationships summary */}
      {showRelationships && relationships && (
        <div
          style={{
            display: 'flex',
            gap: '12px',
            marginBottom: isCompact ? '8px' : '12px',
            fontSize: '11px',
          }}
        >
          {relationships.supports > 0 && (
            <span style={{ color: '#22c55e' }}>
              +{relationships.supports} supporting
            </span>
          )}
          {relationships.contradicts > 0 && (
            <span style={{ color: '#ef4444' }}>
              -{relationships.contradicts} contradicting
            </span>
          )}
          {relationships.refines > 0 && (
            <span style={{ color: '#3b82f6' }}>
              ~{relationships.refines} refining
            </span>
          )}
          {relationships.dependsOn > 0 && (
            <span style={{ color: '#f59e0b' }}>
              &rarr;{relationships.dependsOn} dependencies
            </span>
          )}
        </div>
      )}

      {/* Tags */}
      {showTags && claim.tags && claim.tags.length > 0 && (
        <div
          style={{
            display: 'flex',
            flexWrap: 'wrap',
            gap: '6px',
            marginBottom: '12px',
          }}
        >
          {claim.tags.slice(0, isCompact ? 3 : 6).map((tag, i) => (
            <span
              key={i}
              style={{
                padding: '2px 8px',
                backgroundColor: '#f3f4f6',
                borderRadius: '12px',
                fontSize: '11px',
                color: '#6b7280',
              }}
            >
              {tag}
            </span>
          ))}
          {claim.tags.length > (isCompact ? 3 : 6) && (
            <span style={{ fontSize: '11px', color: '#9ca3af' }}>
              +{claim.tags.length - (isCompact ? 3 : 6)} more
            </span>
          )}
        </div>
      )}

      {/* Actions */}
      {(onViewDetails || onVerify) && !isCompact && (
        <div
          style={{
            display: 'flex',
            gap: '8px',
            paddingTop: '12px',
            borderTop: '1px solid #e5e7eb',
          }}
        >
          {onViewDetails && (
            <button
              onClick={(e) => {
                e.stopPropagation();
                onViewDetails();
              }}
              style={{
                padding: '6px 12px',
                fontSize: '12px',
                fontWeight: 500,
                color: '#3b82f6',
                backgroundColor: '#eff6ff',
                border: 'none',
                borderRadius: '6px',
                cursor: 'pointer',
              }}
            >
              View Details
            </button>
          )}
          {onVerify && (
            <button
              onClick={(e) => {
                e.stopPropagation();
                onVerify();
              }}
              style={{
                padding: '6px 12px',
                fontSize: '12px',
                fontWeight: 500,
                color: '#22c55e',
                backgroundColor: '#dcfce7',
                border: 'none',
                borderRadius: '6px',
                cursor: 'pointer',
              }}
            >
              Verify
            </button>
          )}
        </div>
      )}
    </div>
  );
}

export default ClaimCard;
