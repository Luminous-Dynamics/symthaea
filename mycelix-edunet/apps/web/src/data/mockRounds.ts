// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Mock FL round data
 */

export type RoundState =
  | 'DISCOVER'
  | 'JOIN'
  | 'ASSIGN'
  | 'UPDATE'
  | 'AGGREGATE'
  | 'RELEASE'
  | 'COMPLETED'
  | 'FAILED';

export interface FlRound {
  round_id: string;
  model_id: string;
  course_id: string;
  state: RoundState;
  current_participants: number;
  min_participants: number;
  max_participants: number;
  aggregation_method: string;
  clip_norm: number;
  privacy_params: {
    epsilon: number | null;
    delta: number | null;
    clip_norm: number;
  };
  current_model_hash?: string;
  aggregated_model_hash?: string;
  provenance?: {
    contributor_count: number;
    aggregation_method: string;
    update_quality_metrics: {
      median_val_loss: number;
      mean_clipped_norm: number;
      outliers_trimmed: number;
    };
  };
  created_at: string;
  updated_at?: string;
  completed_at?: string;
}

export const mockFlRounds: FlRound[] = [
  {
    round_id: 'fl-round-001-2025-11-15',
    model_id: 'rust-fundamentals-model-v1',
    course_id: 'rust-fundamentals-2025',
    state: 'COMPLETED',
    current_participants: 37,
    min_participants: 10,
    max_participants: 100,
    aggregation_method: 'trimmed_mean',
    clip_norm: 1.0,
    privacy_params: {
      epsilon: null,
      delta: null,
      clip_norm: 1.0,
    },
    current_model_hash: 'blake3:7b2a9f8e4d1c3a5b6e2f8d9c4a1b3e5f7a9c2d4e6b8f1a3c5d7e9b2f4a6c8e1d',
    aggregated_model_hash:
      'blake3:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855',
    provenance: {
      contributor_count: 37,
      aggregation_method: 'trimmed_mean',
      update_quality_metrics: {
        median_val_loss: 0.42,
        mean_clipped_norm: 0.87,
        outliers_trimmed: 3,
      },
    },
    created_at: '2025-11-14T10:00:00Z',
    updated_at: '2025-11-15T14:30:00Z',
    completed_at: '2025-11-15T14:30:00Z',
  },
  {
    round_id: 'fl-round-002-2025-11-16',
    model_id: 'spanish-beginner-model-v2',
    course_id: 'spanish-beginner-2025',
    state: 'UPDATE',
    current_participants: 28,
    min_participants: 10,
    max_participants: 100,
    aggregation_method: 'trimmed_mean',
    clip_norm: 1.0,
    privacy_params: {
      epsilon: null,
      delta: null,
      clip_norm: 1.0,
    },
    current_model_hash: 'blake3:a1b2c3d4e5f67890abcdef1234567890abcdef1234567890abcdef1234567890',
    created_at: '2025-11-15T08:00:00Z',
    updated_at: '2025-11-16T12:15:00Z',
  },
  {
    round_id: 'fl-round-003-2025-11-17',
    model_id: 'machine-learning-model-v1',
    course_id: 'machine-learning-intro',
    state: 'JOIN',
    current_participants: 5,
    min_participants: 10,
    max_participants: 100,
    aggregation_method: 'median',
    clip_norm: 2.0,
    privacy_params: {
      epsilon: 1.0,
      delta: 0.00001,
      clip_norm: 2.0,
    },
    created_at: '2025-11-16T14:00:00Z',
    updated_at: '2025-11-17T09:30:00Z',
  },
  {
    round_id: 'fl-round-004-2025-11-18',
    model_id: 'web3-dev-model-v1',
    course_id: 'web3-dev',
    state: 'DISCOVER',
    current_participants: 0,
    min_participants: 10,
    max_participants: 50,
    aggregation_method: 'trimmed_mean',
    clip_norm: 1.0,
    privacy_params: {
      epsilon: null,
      delta: null,
      clip_norm: 1.0,
    },
    created_at: '2025-11-17T16:00:00Z',
  },
  {
    round_id: 'fl-round-005-2025-11-19',
    model_id: 'data-structures-model-v3',
    course_id: 'data-structures',
    state: 'AGGREGATE',
    current_participants: 42,
    min_participants: 10,
    max_participants: 100,
    aggregation_method: 'trimmed_mean',
    clip_norm: 1.0,
    privacy_params: {
      epsilon: null,
      delta: null,
      clip_norm: 1.0,
    },
    current_model_hash: 'blake3:1a2b3c4d5e6f7890abcdef1234567890abcdef1234567890abcdef1234567890',
    created_at: '2025-11-18T10:00:00Z',
    updated_at: '2025-11-19T11:45:00Z',
  },
  {
    round_id: 'fl-round-006-2025-11-20',
    model_id: 'deep-learning-model-v1',
    course_id: 'deep-learning',
    state: 'JOIN',
    current_participants: 8,
    min_participants: 15,
    max_participants: 50,
    aggregation_method: 'trimmed_mean',
    clip_norm: 2.0,
    privacy_params: {
      epsilon: 1.0,
      delta: 0.00001,
      clip_norm: 2.0,
    },
    created_at: '2025-11-20T08:00:00Z',
  },
  {
    round_id: 'fl-round-007-2025-11-21',
    model_id: 'federated-learning-model-v2',
    course_id: 'federated-learning',
    state: 'UPDATE',
    current_participants: 23,
    min_participants: 10,
    max_participants: 40,
    aggregation_method: 'median',
    clip_norm: 1.5,
    privacy_params: {
      epsilon: 0.5,
      delta: 0.00001,
      clip_norm: 1.5,
    },
    current_model_hash: 'blake3:9f8e7d6c5b4a39281764e5d3c2b1a098f7e6d5c4b3a29180f7e6d5c4b3a2918',
    created_at: '2025-11-20T12:00:00Z',
    updated_at: '2025-11-21T10:30:00Z',
  },
  {
    round_id: 'fl-round-008-2025-11-22',
    model_id: 'react-modern-web-model-v3',
    course_id: 'react-modern-web',
    state: 'DISCOVER',
    current_participants: 0,
    min_participants: 20,
    max_participants: 80,
    aggregation_method: 'trimmed_mean',
    clip_norm: 1.0,
    privacy_params: {
      epsilon: null,
      delta: null,
      clip_norm: 1.0,
    },
    created_at: '2025-11-22T06:00:00Z',
  },
  {
    round_id: 'fl-round-009-2025-11-19',
    model_id: 'python-data-science-model-v4',
    course_id: 'python-data-science',
    state: 'COMPLETED',
    current_participants: 54,
    min_participants: 10,
    max_participants: 80,
    aggregation_method: 'trimmed_mean',
    clip_norm: 1.0,
    privacy_params: {
      epsilon: null,
      delta: null,
      clip_norm: 1.0,
    },
    current_model_hash: 'blake3:a1b2c3d4e5f6789012345678901234567890123456789012345678901234567',
    aggregated_model_hash: 'blake3:f0e1d2c3b4a5968778695a4b3c2d1e0f9a8b7c6d5e4f3a2b1c0d9e8f7a6b5c4',
    provenance: {
      contributor_count: 54,
      aggregation_method: 'trimmed_mean',
      update_quality_metrics: {
        median_val_loss: 0.38,
        mean_clipped_norm: 0.92,
        outliers_trimmed: 5,
      },
    },
    created_at: '2025-11-18T14:00:00Z',
    updated_at: '2025-11-19T16:20:00Z',
    completed_at: '2025-11-19T16:20:00Z',
  },
  {
    round_id: 'fl-round-010-2025-11-21',
    model_id: 'golang-concurrency-model-v1',
    course_id: 'golang-concurrency',
    state: 'JOIN',
    current_participants: 12,
    min_participants: 8,
    max_participants: 40,
    aggregation_method: 'trimmed_mean',
    clip_norm: 1.0,
    privacy_params: {
      epsilon: null,
      delta: null,
      clip_norm: 1.0,
    },
    created_at: '2025-11-21T09:00:00Z',
  },
  {
    round_id: 'fl-round-011-2025-11-22',
    model_id: 'holochain-dev-model-v2',
    course_id: 'holochain-dev',
    state: 'DISCOVER',
    current_participants: 0,
    min_participants: 5,
    max_participants: 25,
    aggregation_method: 'median',
    clip_norm: 1.5,
    privacy_params: {
      epsilon: 0.8,
      delta: 0.00001,
      clip_norm: 1.5,
    },
    created_at: '2025-11-22T11:00:00Z',
  },
  {
    round_id: 'fl-round-012-2025-11-20',
    model_id: 'ethical-hacking-model-v1',
    course_id: 'ethical-hacking',
    state: 'AGGREGATE',
    current_participants: 31,
    min_participants: 10,
    max_participants: 50,
    aggregation_method: 'trimmed_mean',
    clip_norm: 1.0,
    privacy_params: {
      epsilon: null,
      delta: null,
      clip_norm: 1.0,
    },
    current_model_hash: 'blake3:5f4e3d2c1b0a9f8e7d6c5b4a3f2e1d0c9b8a7f6e5d4c3b2a1f0e9d8c7b6a5f4',
    created_at: '2025-11-19T15:00:00Z',
    updated_at: '2025-11-20T17:25:00Z',
  },
  {
    round_id: 'fl-round-013-2025-11-18',
    model_id: 'node-backend-model-v2',
    course_id: 'node-backend',
    state: 'COMPLETED',
    current_participants: 48,
    min_participants: 15,
    max_participants: 70,
    aggregation_method: 'trimmed_mean',
    clip_norm: 1.0,
    privacy_params: {
      epsilon: null,
      delta: null,
      clip_norm: 1.0,
    },
    current_model_hash: 'blake3:3e2d1c0b9a8f7e6d5c4b3a2f1e0d9c8b7a6f5e4d3c2b1a0f9e8d7c6b5a4f3e2',
    aggregated_model_hash: 'blake3:d1e2f3a4b5c6d7e8f9a0b1c2d3e4f5a6b7c8d9e0f1a2b3c4d5e6f7a8b9c0d1e',
    provenance: {
      contributor_count: 48,
      aggregation_method: 'trimmed_mean',
      update_quality_metrics: {
        median_val_loss: 0.35,
        mean_clipped_norm: 0.89,
        outliers_trimmed: 4,
      },
    },
    created_at: '2025-11-17T08:00:00Z',
    updated_at: '2025-11-18T12:40:00Z',
    completed_at: '2025-11-18T12:40:00Z',
  },
  {
    round_id: 'fl-round-014-2025-11-21',
    model_id: 'linear-algebra-model-v3',
    course_id: 'linear-algebra',
    state: 'RELEASE',
    current_participants: 36,
    min_participants: 10,
    max_participants: 60,
    aggregation_method: 'trimmed_mean',
    clip_norm: 1.0,
    privacy_params: {
      epsilon: null,
      delta: null,
      clip_norm: 1.0,
    },
    current_model_hash: 'blake3:7a6b5c4d3e2f1a0b9c8d7e6f5a4b3c2d1e0f9a8b7c6d5e4f3a2b1c0d9e8f7a6',
    aggregated_model_hash: 'blake3:c2d3e4f5a6b7c8d9e0f1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b1c2d',
    provenance: {
      contributor_count: 36,
      aggregation_method: 'trimmed_mean',
      update_quality_metrics: {
        median_val_loss: 0.41,
        mean_clipped_norm: 0.85,
        outliers_trimmed: 3,
      },
    },
    created_at: '2025-11-20T07:00:00Z',
    updated_at: '2025-11-21T11:15:00Z',
    completed_at: '2025-11-21T11:15:00Z',
  },
];

export function getRoundById(id: string): FlRound | undefined {
  return mockFlRounds.find(r => r.round_id === id);
}

export function getRoundsByCourse(courseId: string): FlRound[] {
  return mockFlRounds.filter(r => r.course_id === courseId);
}

export function getRoundsByState(state: RoundState): FlRound[] {
  return mockFlRounds.filter(r => r.state === state);
}

export function getActiveRounds(): FlRound[] {
  const activeStates: RoundState[] = ['JOIN', 'ASSIGN', 'UPDATE', 'AGGREGATE'];
  return mockFlRounds.filter(r => activeStates.includes(r.state));
}

export function getCompletedRounds(): FlRound[] {
  return mockFlRounds.filter(r => r.state === 'COMPLETED');
}

/**
 * Get progress percentage for a round based on its state
 */
export function getRoundProgress(round: FlRound): number {
  const stateProgress: Record<RoundState, number> = {
    DISCOVER: 10,
    JOIN: 25,
    ASSIGN: 40,
    UPDATE: 60,
    AGGREGATE: 80,
    RELEASE: 95,
    COMPLETED: 100,
    FAILED: 0,
  };
  return stateProgress[round.state];
}

/**
 * Get human-readable state description
 */
export function getRoundStateDescription(state: RoundState): string {
  const descriptions: Record<RoundState, string> = {
    DISCOVER: 'Discovering participants',
    JOIN: 'Accepting participants',
    ASSIGN: 'Assigning model updates',
    UPDATE: 'Participants training locally',
    AGGREGATE: 'Aggregating model updates',
    RELEASE: 'Releasing aggregated model',
    COMPLETED: 'Round completed successfully',
    FAILED: 'Round failed',
  };
  return descriptions[state];
}
