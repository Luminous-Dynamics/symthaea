// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * RoundCard Component
 *
 * Displays an FL round in a card format with state, participants, and metadata
 * Memoized to prevent unnecessary re-renders when props haven't changed
 */

import React from 'react';
import type { FlRound } from '../data/mockRounds';
import { getRoundProgress, getRoundStateDescription } from '../data/mockRounds';

interface RoundCardProps {
  round: FlRound;
  onClick?: (round: FlRound) => void;
}

export const RoundCard = React.memo<RoundCardProps>(({ round, onClick }) => {
  const handleClick = () => {
    if (onClick) {
      onClick(round);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (onClick && (e.key === 'Enter' || e.key === ' ')) {
      e.preventDefault();
      onClick(round);
    }
  };

  const getStateColor = (state: string): string => {
    switch (state) {
      case 'DISCOVER':
        return '#8b5cf6'; // purple
      case 'JOIN':
        return '#3b82f6'; // blue
      case 'ASSIGN':
      case 'UPDATE':
        return '#f59e0b'; // amber
      case 'AGGREGATE':
        return '#f97316'; // orange
      case 'RELEASE':
        return '#10b981'; // green
      case 'COMPLETED':
        return '#10b981'; // green
      case 'FAILED':
        return '#ef4444'; // red
      default:
        return '#6b7280'; // gray
    }
  };

  const progress = getRoundProgress(round);
  const stateDescription = getRoundStateDescription(round.state);
  const isActive = ['JOIN', 'ASSIGN', 'UPDATE', 'AGGREGATE'].includes(round.state);

  return (
    <div
      className="round-card"
      onClick={handleClick}
      onKeyDown={handleKeyDown}
      role={onClick ? 'button' : undefined}
      tabIndex={onClick ? 0 : undefined}
      aria-label={`FL Round ${round.round_id}. State: ${round.state}. ${round.current_participants} of ${round.max_participants} participants. Aggregation: ${round.aggregation_method}. Click for details.`}
      style={{
        border: '1px solid #e5e7eb',
        borderRadius: '8px',
        padding: '20px',
        cursor: onClick ? 'pointer' : 'default',
        transition: 'all 0.2s ease',
        backgroundColor: '#ffffff',
        boxShadow: '0 1px 3px rgba(0, 0, 0, 0.1)',
        position: 'relative',
        overflow: 'hidden',
      }}
      onMouseEnter={(e) => {
        if (onClick) {
          e.currentTarget.style.boxShadow = '0 4px 12px rgba(0, 0, 0, 0.15)';
          e.currentTarget.style.transform = 'translateY(-2px)';
        }
      }}
      onMouseLeave={(e) => {
        e.currentTarget.style.boxShadow = '0 1px 3px rgba(0, 0, 0, 0.1)';
        e.currentTarget.style.transform = 'translateY(0)';
      }}
    >
      {/* Active Indicator */}
      {isActive && (
        <div
          style={{
            position: 'absolute',
            top: 0,
            left: 0,
            right: 0,
            height: '3px',
            background: `linear-gradient(90deg, ${getStateColor(round.state)}, transparent)`,
          }}
        />
      )}

      {/* Header */}
      <div style={{ marginBottom: '16px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
          <div>
            <h3
              style={{
                margin: 0,
                fontSize: '16px',
                fontWeight: '600',
                color: '#111827',
                marginBottom: '4px',
              }}
            >
              {round.round_id}
            </h3>
            <p style={{ margin: 0, fontSize: '13px', color: '#6b7280' }}>
              Model: {round.model_id}
            </p>
          </div>
          <span
            style={{
              fontSize: '11px',
              fontWeight: '600',
              padding: '4px 10px',
              borderRadius: '12px',
              backgroundColor: getStateColor(round.state) + '20',
              color: getStateColor(round.state),
              textTransform: 'uppercase',
              letterSpacing: '0.5px',
            }}
          >
            {round.state}
          </span>
        </div>
      </div>

      {/* State Description */}
      <p
        style={{
          margin: '0 0 16px 0',
          fontSize: '14px',
          color: '#4b5563',
        }}
      >
        {stateDescription}
      </p>

      {/* Progress Bar */}
      <div style={{ marginBottom: '16px' }}>
        <div
          style={{
            height: '6px',
            backgroundColor: '#e5e7eb',
            borderRadius: '3px',
            overflow: 'hidden',
          }}
        >
          <div
            style={{
              height: '100%',
              width: `${progress}%`,
              backgroundColor: getStateColor(round.state),
              transition: 'width 0.3s ease',
            }}
          />
        </div>
      </div>

      {/* Participants */}
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          marginBottom: '16px',
        }}
      >
        <div style={{ fontSize: '14px', color: '#4b5563' }}>
          <span style={{ fontWeight: '600', fontSize: '18px', color: '#111827' }}>
            {round.current_participants}
          </span>
          <span style={{ color: '#9ca3af' }}> / {round.max_participants}</span>
          <span style={{ marginLeft: '4px' }}>participants</span>
        </div>
        {round.min_participants && round.current_participants < round.min_participants && (
          <span
            style={{
              fontSize: '12px',
              color: '#f59e0b',
              backgroundColor: '#fef3c7',
              padding: '2px 8px',
              borderRadius: '4px',
            }}
          >
            Need {round.min_participants - round.current_participants} more
          </span>
        )}
      </div>

      {/* Metadata */}
      <div
        style={{
          display: 'flex',
          flexWrap: 'wrap',
          gap: '12px',
          paddingTop: '16px',
          borderTop: '1px solid #e5e7eb',
          fontSize: '13px',
          color: '#6b7280',
        }}
      >
        <span>📊 {round.aggregation_method}</span>
        <span>🔒 Clip: {round.clip_norm}</span>
        {round.privacy_params.epsilon && (
          <span>🛡️ ε={round.privacy_params.epsilon}</span>
        )}
      </div>

      {/* Timestamps */}
      <div
        style={{
          marginTop: '12px',
          fontSize: '12px',
          color: '#9ca3af',
        }}
      >
        Created: {new Date(round.created_at).toLocaleDateString()}
        {round.completed_at && (
          <span style={{ marginLeft: '12px' }}>
            Completed: {new Date(round.completed_at).toLocaleDateString()}
          </span>
        )}
      </div>
    </div>
  );
});
