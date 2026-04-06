// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * ErrorState Component
 *
 * Displays error states for failed data fetching or other non-critical errors
 */

import React from 'react';

interface ErrorStateProps {
  title?: string;
  message?: string;
  onRetry?: () => void;
  icon?: string;
}

export const ErrorState: React.FC<ErrorStateProps> = ({
  title = 'Error Loading Data',
  message = 'Something went wrong while loading this content. Please try again.',
  onRetry,
  icon = '⚠️',
}) => {
  return (
    <div
      style={{
        padding: '64px 24px',
        textAlign: 'center',
        backgroundColor: '#fef2f2',
        borderRadius: '8px',
        border: '1px solid #fecaca',
      }}
    >
      <div style={{ fontSize: '48px', marginBottom: '16px' }}>{icon}</div>
      <h3
        style={{
          margin: '0 0 8px 0',
          fontSize: '20px',
          fontWeight: '600',
          color: '#dc2626',
        }}
      >
        {title}
      </h3>
      <p style={{ margin: '0 0 24px 0', fontSize: '14px', color: '#991b1b' }}>
        {message}
      </p>

      {onRetry && (
        <button
          onClick={onRetry}
          style={{
            padding: '10px 20px',
            fontSize: '14px',
            fontWeight: '600',
            color: '#ffffff',
            backgroundColor: '#dc2626',
            border: 'none',
            borderRadius: '6px',
            cursor: 'pointer',
          }}
          onMouseEnter={(e) => {
            e.currentTarget.style.backgroundColor = '#b91c1c';
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.backgroundColor = '#dc2626';
          }}
        >
          Try Again
        </button>
      )}
    </div>
  );
};

/**
 * EmptyState - Displays when no data is available
 */
export const EmptyState: React.FC<{
  title?: string;
  message?: string;
  action?: {
    label: string;
    onClick: () => void;
  };
  icon?: string;
}> = ({
  title = 'No Data Available',
  message = 'There is no data to display at this time.',
  action,
  icon = '📭',
}) => {
  return (
    <div
      style={{
        padding: '64px 24px',
        textAlign: 'center',
        backgroundColor: '#f9fafb',
        borderRadius: '8px',
      }}
    >
      <div style={{ fontSize: '48px', marginBottom: '16px' }}>{icon}</div>
      <h3
        style={{
          margin: '0 0 8px 0',
          fontSize: '20px',
          fontWeight: '600',
          color: '#111827',
        }}
      >
        {title}
      </h3>
      <p style={{ margin: '0 0 24px 0', fontSize: '14px', color: '#6b7280' }}>
        {message}
      </p>

      {action && (
        <button
          onClick={action.onClick}
          style={{
            padding: '10px 20px',
            fontSize: '14px',
            fontWeight: '600',
            color: '#ffffff',
            backgroundColor: '#3b82f6',
            border: 'none',
            borderRadius: '6px',
            cursor: 'pointer',
          }}
          onMouseEnter={(e) => {
            e.currentTarget.style.backgroundColor = '#2563eb';
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.backgroundColor = '#3b82f6';
          }}
        >
          {action.label}
        </button>
      )}
    </div>
  );
};
