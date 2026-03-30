// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * FLRoundsPage Component
 *
 * Main page for viewing and participating in Federated Learning rounds
 */

import React, { useState, useMemo, useEffect, useCallback } from 'react';
import { RoundCard } from '../components/RoundCard';
import { RoundTimeline } from '../components/RoundTimeline';
import { LoadingCard } from '../components/LoadingSkeleton';
import { ErrorState } from '../components/ErrorState';
import { EmptyState } from '../components/EmptyState';
import { useToast } from '../contexts/ToastContext';
import { mockFlRounds, FlRound } from '../data/mockRounds';
import {
  ANIMATION_TIMING,
  ERROR_SIMULATION,
  ERROR_MESSAGES,
  SUCCESS_MESSAGES,
  INFO_MESSAGES,
  FL_ROUND_STATES,
} from '../config/constants';

type FilterTab = 'all' | 'active' | 'completed';

export const FLRoundsPage: React.FC = () => {
  const { toast } = useToast();
  const [selectedTab, setSelectedTab] = useState<FilterTab>('all');
  const [selectedRound, setSelectedRound] = useState<FlRound | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [rounds, setRounds] = useState<FlRound[]>([]);

  // Simulate data fetching
  useEffect(() => {
    const fetchRounds = async () => {
      setLoading(true);
      setError(null);

      try {
        // Simulate API call delay
        await new Promise(resolve => setTimeout(resolve, ANIMATION_TIMING.LOADING_DELAY.FL_ROUNDS));

        // Simulate occasional error
        if (Math.random() < ERROR_SIMULATION.FL_ROUNDS_ERROR_RATE) {
          throw new Error(ERROR_MESSAGES.FL_ROUNDS_LOAD_FAILED);
        }

        setRounds(mockFlRounds);
        toast.success(SUCCESS_MESSAGES.FL_ROUNDS_LOADED(mockFlRounds.length));
      } catch (err) {
        const errorMessage = err instanceof Error ? err.message : ERROR_MESSAGES.GENERIC;
        setError(errorMessage);
        toast.error(ERROR_MESSAGES.FL_ROUNDS_LOAD_FAILED);
      } finally {
        setLoading(false);
      }
    };

    fetchRounds();
  }, [toast]);

  const filteredRounds = useMemo(() => {
    switch (selectedTab) {
      case 'active':
        return rounds.filter((r) => FL_ROUND_STATES.ACTIVE.includes(r.state as any));
      case 'completed':
        return rounds.filter((r) => r.state === FL_ROUND_STATES.COMPLETED);
      default:
        return rounds;
    }
  }, [selectedTab, rounds]);

  const activeCount = useMemo(
    () => rounds.filter((r) => FL_ROUND_STATES.ACTIVE.includes(r.state as any)).length,
    [rounds]
  );

  const completedCount = useMemo(
    () => rounds.filter((r) => r.state === FL_ROUND_STATES.COMPLETED).length,
    [rounds]
  );

  const handleRoundClick = useCallback((round: FlRound) => {
    setSelectedRound(round);
    toast.info(INFO_MESSAGES.VIEWING_ROUND(round.round_id));
  }, [toast]);

  const handleCloseModal = useCallback(() => {
    setSelectedRound(null);
  }, []);

  const handleJoinRound = useCallback((round: FlRound) => {
    toast.success(SUCCESS_MESSAGES.ROUND_JOINED(round.round_id));
    toast.info(INFO_MESSAGES.HOLOCHAIN_COMING_SOON);
  }, [toast]);

  const TabButton: React.FC<{ tab: FilterTab; label: string; count?: number }> = ({
    tab,
    label,
    count,
  }) => {
    const isActive = selectedTab === tab;
    return (
      <button
        onClick={() => setSelectedTab(tab)}
        style={{
          padding: '10px 20px',
          fontSize: '14px',
          fontWeight: '500',
          color: isActive ? '#3b82f6' : '#6b7280',
          backgroundColor: isActive ? '#eff6ff' : 'transparent',
          border: 'none',
          borderBottom: isActive ? '2px solid #3b82f6' : '2px solid transparent',
          cursor: 'pointer',
          transition: 'all 0.2s ease',
        }}
        onMouseEnter={(e) => {
          if (!isActive) {
            e.currentTarget.style.color = '#3b82f6';
          }
        }}
        onMouseLeave={(e) => {
          if (!isActive) {
            e.currentTarget.style.color = '#6b7280';
          }
        }}
      >
        {label}
        {count !== undefined && (
          <span
            style={{
              marginLeft: '6px',
              padding: '2px 8px',
              fontSize: '12px',
              borderRadius: '10px',
              backgroundColor: isActive ? '#3b82f6' : '#e5e7eb',
              color: isActive ? '#ffffff' : '#6b7280',
            }}
          >
            {count}
          </span>
        )}
      </button>
    );
  };

  // Show loading state
  if (loading) {
    return (
      <div style={{ maxWidth: '1200px', margin: '0 auto', padding: '24px' }}>
        {/* Header */}
        <div style={{ marginBottom: '32px' }}>
          <h1 style={{ margin: 0, fontSize: '32px', fontWeight: '700', color: '#111827' }}>
            Federated Learning Rounds
          </h1>
          <p style={{ margin: '8px 0 0 0', fontSize: '16px', color: '#6b7280' }}>
            Loading FL rounds...
          </p>
        </div>

        {/* Loading skeleton grid */}
        <div
          style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fill, minmax(350px, 1fr))',
            gap: '20px',
          }}
        >
          {[1, 2, 3, 4].map((i) => (
            <LoadingCard key={i} height="200px" />
          ))}
        </div>
      </div>
    );
  }

  // Show error state
  if (error) {
    return (
      <div style={{ maxWidth: '1200px', margin: '0 auto', padding: '24px' }}>
        {/* Header */}
        <div style={{ marginBottom: '32px' }}>
          <h1 style={{ margin: 0, fontSize: '32px', fontWeight: '700', color: '#111827' }}>
            Federated Learning Rounds
          </h1>
        </div>

        <ErrorState
          title="Failed to Load FL Rounds"
          message={error}
          onRetry={() => window.location.reload()}
        />
      </div>
    );
  }

  return (
    <div style={{ maxWidth: '1200px', margin: '0 auto', padding: '24px' }}>
      {/* Header */}
      <div style={{ marginBottom: '32px' }}>
        <h1 style={{ margin: 0, fontSize: '32px', fontWeight: '700', color: '#111827' }}>
          Federated Learning Rounds
        </h1>
        <p style={{ margin: '8px 0 0 0', fontSize: '16px', color: '#6b7280' }}>
          Participate in privacy-preserving collaborative model training
        </p>
      </div>

      {/* Tabs */}
      <div
        style={{
          display: 'flex',
          gap: '8px',
          borderBottom: '1px solid #e5e7eb',
          marginBottom: '24px',
        }}
      >
        <TabButton tab="all" label="All Rounds" count={rounds.length} />
        <TabButton tab="active" label="Active" count={activeCount} />
        <TabButton tab="completed" label="Completed" count={completedCount} />
      </div>

      {/* Rounds Grid */}
      {filteredRounds.length > 0 ? (
        <div
          style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fill, minmax(350px, 1fr))',
            gap: '20px',
          }}
        >
          {filteredRounds.map((round) => (
            <RoundCard key={round.round_id} round={round} onClick={handleRoundClick} />
          ))}
        </div>
      ) : (
        <EmptyState
          icon={selectedTab === 'active' ? '⏸️' : selectedTab === 'completed' ? '✅' : '📭'}
          title={
            selectedTab === 'active'
              ? 'No Active Rounds'
              : selectedTab === 'completed'
              ? 'No Completed Rounds'
              : 'No Rounds Available'
          }
          description={
            selectedTab === 'active'
              ? 'There are currently no active FL rounds accepting participants. Check back soon or view all rounds.'
              : selectedTab === 'completed'
              ? 'No rounds have been completed yet. Join an active round to start contributing!'
              : 'No federated learning rounds are available at this time. New rounds will appear here when created.'
          }
          action={{
            label: selectedTab === 'all' ? 'Refresh' : 'View All Rounds',
            onClick: () => {
              if (selectedTab === 'all') {
                window.location.reload();
              } else {
                setSelectedTab('all');
                toast.info(INFO_MESSAGES.SHOWING_ALL_ROUNDS);
              }
            },
          }}
          secondaryAction={
            selectedTab !== 'active'
              ? {
                  label: 'View Active Rounds',
                  onClick: () => {
                    setSelectedTab('active');
                    toast.info(INFO_MESSAGES.SHOWING_ACTIVE_ROUNDS);
                  },
                }
              : undefined
          }
        />
      )}

      {/* Round Detail Modal */}
      {selectedRound && (
        <div
          style={{
            position: 'fixed',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            backgroundColor: 'rgba(0, 0, 0, 0.5)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 1000,
            padding: '24px',
          }}
          onClick={handleCloseModal}
        >
          <div
            style={{
              backgroundColor: '#ffffff',
              borderRadius: '12px',
              maxWidth: '800px',
              width: '100%',
              maxHeight: '90vh',
              overflow: 'auto',
              padding: '32px',
              position: 'relative',
            }}
            onClick={(e) => e.stopPropagation()}
          >
            {/* Close Button */}
            <button
              onClick={handleCloseModal}
              style={{
                position: 'absolute',
                top: '16px',
                right: '16px',
                width: '32px',
                height: '32px',
                borderRadius: '50%',
                border: '1px solid #d1d5db',
                backgroundColor: '#ffffff',
                cursor: 'pointer',
                fontSize: '18px',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
              }}
            >
              ×
            </button>

            {/* Round Details */}
            <h2 style={{ margin: '0 0 8px 0', fontSize: '24px', fontWeight: '700' }}>
              {selectedRound.round_id}
            </h2>
            <p style={{ margin: '0 0 24px 0', fontSize: '14px', color: '#6b7280' }}>
              Model: {selectedRound.model_id}
            </p>

            {/* Timeline */}
            <RoundTimeline currentState={selectedRound.state} />

            {/* Stats Grid */}
            <div
              style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
                gap: '16px',
                marginTop: '32px',
                marginBottom: '24px',
              }}
            >
              <div
                style={{
                  padding: '16px',
                  backgroundColor: '#f9fafb',
                  borderRadius: '8px',
                }}
              >
                <div style={{ fontSize: '12px', color: '#6b7280', marginBottom: '4px' }}>
                  Participants
                </div>
                <div style={{ fontSize: '24px', fontWeight: '700', color: '#111827' }}>
                  {selectedRound.current_participants}
                  <span style={{ fontSize: '14px', color: '#9ca3af', fontWeight: '400' }}>
                    {' '}
                    / {selectedRound.max_participants}
                  </span>
                </div>
              </div>

              <div
                style={{
                  padding: '16px',
                  backgroundColor: '#f9fafb',
                  borderRadius: '8px',
                }}
              >
                <div style={{ fontSize: '12px', color: '#6b7280', marginBottom: '4px' }}>
                  Aggregation Method
                </div>
                <div style={{ fontSize: '18px', fontWeight: '600', color: '#111827' }}>
                  {selectedRound.aggregation_method}
                </div>
              </div>

              <div
                style={{
                  padding: '16px',
                  backgroundColor: '#f9fafb',
                  borderRadius: '8px',
                }}
              >
                <div style={{ fontSize: '12px', color: '#6b7280', marginBottom: '4px' }}>
                  Clip Norm
                </div>
                <div style={{ fontSize: '18px', fontWeight: '600', color: '#111827' }}>
                  {selectedRound.clip_norm}
                </div>
              </div>
            </div>

            {/* Privacy Params */}
            {selectedRound.privacy_params.epsilon && (
              <div
                style={{
                  padding: '16px',
                  backgroundColor: '#ecfdf5',
                  border: '1px solid #10b981',
                  borderRadius: '8px',
                  marginBottom: '24px',
                }}
              >
                <div style={{ fontSize: '14px', fontWeight: '600', color: '#059669', marginBottom: '8px' }}>
                  🛡️ Differential Privacy Enabled
                </div>
                <div style={{ fontSize: '13px', color: '#047857' }}>
                  ε = {selectedRound.privacy_params.epsilon}, δ ={' '}
                  {selectedRound.privacy_params.delta}
                </div>
              </div>
            )}

            {/* Provenance */}
            {selectedRound.provenance && (
              <div
                style={{
                  padding: '16px',
                  backgroundColor: '#f9fafb',
                  borderRadius: '8px',
                  marginBottom: '24px',
                }}
              >
                <div style={{ fontSize: '14px', fontWeight: '600', marginBottom: '12px' }}>
                  Round Provenance
                </div>
                <div style={{ fontSize: '13px', color: '#4b5563', lineHeight: '1.6' }}>
                  <div>Contributors: {selectedRound.provenance.contributor_count}</div>
                  <div>Median Val Loss: {selectedRound.provenance.update_quality_metrics.median_val_loss}</div>
                  <div>Mean Clipped Norm: {selectedRound.provenance.update_quality_metrics.mean_clipped_norm}</div>
                  <div>Outliers Trimmed: {selectedRound.provenance.update_quality_metrics.outliers_trimmed}</div>
                </div>
              </div>
            )}

            {/* Join Button */}
            {selectedRound.state === 'JOIN' && (
              <button
                style={{
                  width: '100%',
                  padding: '12px',
                  fontSize: '16px',
                  fontWeight: '600',
                  color: '#ffffff',
                  backgroundColor: '#3b82f6',
                  border: 'none',
                  borderRadius: '8px',
                  cursor: 'pointer',
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.backgroundColor = '#2563eb';
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.backgroundColor = '#3b82f6';
                }}
                onClick={() => handleJoinRound(selectedRound)}
              >
                Join this Round
              </button>
            )}
          </div>
        </div>
      )}
    </div>
  );
};
