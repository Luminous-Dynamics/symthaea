// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * RoundTimeline Component
 *
 * Visualizes the FL round lifecycle with phase indicators
 */

import React from 'react';
import type { RoundState } from '../data/mockRounds';

interface RoundTimelineProps {
  currentState: RoundState;
}

const PHASES: RoundState[] = [
  'DISCOVER',
  'JOIN',
  'ASSIGN',
  'UPDATE',
  'AGGREGATE',
  'RELEASE',
];

export const RoundTimeline: React.FC<RoundTimelineProps> = ({ currentState }) => {
  const getCurrentPhaseIndex = (): number => {
    if (currentState === 'COMPLETED') return PHASES.length;
    if (currentState === 'FAILED') return -1;
    return PHASES.indexOf(currentState);
  };

  const currentPhaseIndex = getCurrentPhaseIndex();

  const getPhaseStatus = (index: number): 'completed' | 'current' | 'pending' | 'failed' => {
    if (currentState === 'FAILED') return 'failed';
    if (index < currentPhaseIndex) return 'completed';
    if (index === currentPhaseIndex) return 'current';
    return 'pending';
  };

  const getStatusColor = (status: string): string => {
    switch (status) {
      case 'completed':
        return '#10b981'; // green
      case 'current':
        return '#3b82f6'; // blue
      case 'pending':
        return '#d1d5db'; // gray
      case 'failed':
        return '#ef4444'; // red
      default:
        return '#d1d5db';
    }
  };

  const getStatusIcon = (status: string): string => {
    switch (status) {
      case 'completed':
        return '✓';
      case 'current':
        return '●';
      case 'pending':
        return '○';
      case 'failed':
        return '✕';
      default:
        return '○';
    }
  };

  return (
    <div style={{ padding: '20px 0' }}>
      {/* Timeline */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        {PHASES.map((phase, index) => {
          const status = getPhaseStatus(index);
          const color = getStatusColor(status);
          const isLast = index === PHASES.length - 1;

          return (
            <React.Fragment key={phase}>
              {/* Phase Node */}
              <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
                {/* Circle */}
                <div
                  style={{
                    width: '40px',
                    height: '40px',
                    borderRadius: '50%',
                    backgroundColor: status === 'pending' ? '#ffffff' : color,
                    border: `2px solid ${color}`,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    fontSize: '16px',
                    fontWeight: '600',
                    color: status === 'pending' ? color : '#ffffff',
                    transition: 'all 0.3s ease',
                    ...(status === 'current' && {
                      animation: 'pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite',
                    }),
                  }}
                >
                  {getStatusIcon(status)}
                </div>

                {/* Label */}
                <div
                  style={{
                    marginTop: '8px',
                    fontSize: '11px',
                    fontWeight: status === 'current' ? '600' : '500',
                    color: status === 'current' ? color : '#6b7280',
                    textAlign: 'center',
                    textTransform: 'uppercase',
                    letterSpacing: '0.5px',
                  }}
                >
                  {phase}
                </div>
              </div>

              {/* Connector Line */}
              {!isLast && (
                <div
                  style={{
                    flex: 1,
                    height: '2px',
                    backgroundColor: index < currentPhaseIndex ? color : '#d1d5db',
                    margin: '0 8px',
                    marginBottom: '32px', // Offset for labels
                    transition: 'background-color 0.3s ease',
                  }}
                />
              )}
            </React.Fragment>
          );
        })}
      </div>

      {/* Completed State */}
      {currentState === 'COMPLETED' && (
        <div
          style={{
            marginTop: '24px',
            padding: '16px',
            backgroundColor: '#ecfdf5',
            border: '1px solid #10b981',
            borderRadius: '8px',
            textAlign: 'center',
          }}
        >
          <div style={{ fontSize: '16px', fontWeight: '600', color: '#059669' }}>
            ✓ Round Completed Successfully
          </div>
          <div style={{ fontSize: '13px', color: '#047857', marginTop: '4px' }}>
            Aggregated model is now available
          </div>
        </div>
      )}

      {/* Failed State */}
      {currentState === 'FAILED' && (
        <div
          style={{
            marginTop: '24px',
            padding: '16px',
            backgroundColor: '#fef2f2',
            border: '1px solid #ef4444',
            borderRadius: '8px',
            textAlign: 'center',
          }}
        >
          <div style={{ fontSize: '16px', fontWeight: '600', color: '#dc2626' }}>
            ✕ Round Failed
          </div>
          <div style={{ fontSize: '13px', color: '#b91c1c', marginTop: '4px' }}>
            Please check the logs for details
          </div>
        </div>
      )}

      {/* Pulse Animation */}
      <style>
        {`
          @keyframes pulse {
            0%, 100% {
              opacity: 1;
            }
            50% {
              opacity: 0.7;
            }
          }
        `}
      </style>
    </div>
  );
};
