// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * LoadingSkeleton Component
 *
 * Animated skeleton loader for content that is being fetched
 */

import React from 'react';

interface LoadingSkeletonProps {
  width?: string;
  height?: string;
  borderRadius?: string;
  className?: string;
}

export const LoadingSkeleton: React.FC<LoadingSkeletonProps> = ({
  width = '100%',
  height = '20px',
  borderRadius = '4px',
  className,
}) => {
  return (
    <div
      className={className}
      style={{
        width,
        height,
        borderRadius,
        background: 'linear-gradient(90deg, #f0f0f0 25%, #e0e0e0 50%, #f0f0f0 75%)',
        backgroundSize: '200% 100%',
        animation: 'shimmer 1.5s infinite',
      }}
    >
      <style>
        {`
          @keyframes shimmer {
            0% {
              background-position: 200% 0;
            }
            100% {
              background-position: -200% 0;
            }
          }
        `}
      </style>
    </div>
  );
};

/**
 * LoadingCard - Skeleton for card layouts
 */
interface LoadingCardProps {
  height?: string;
}

export const LoadingCard: React.FC<LoadingCardProps> = ({ height }) => {
  return (
    <div
      style={{
        border: '1px solid #e5e7eb',
        borderRadius: '8px',
        padding: '20px',
        backgroundColor: '#ffffff',
        minHeight: height,
      }}
    >
      {/* Header */}
      <div style={{ marginBottom: '16px' }}>
        <LoadingSkeleton height="24px" width="70%" />
        <div style={{ marginTop: '8px' }}>
          <LoadingSkeleton height="16px" width="40%" />
        </div>
      </div>

      {/* Body */}
      <div style={{ marginBottom: '16px' }}>
        <LoadingSkeleton height="14px" width="100%" />
        <div style={{ marginTop: '8px' }}>
          <LoadingSkeleton height="14px" width="90%" />
        </div>
      </div>

      {/* Tags */}
      <div style={{ display: 'flex', gap: '8px', marginBottom: '16px' }}>
        <LoadingSkeleton height="24px" width="60px" borderRadius="12px" />
        <LoadingSkeleton height="24px" width="80px" borderRadius="12px" />
        <LoadingSkeleton height="24px" width="70px" borderRadius="12px" />
      </div>

      {/* Footer */}
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          paddingTop: '16px',
          borderTop: '1px solid #e5e7eb',
        }}
      >
        <LoadingSkeleton height="14px" width="100px" />
        <LoadingSkeleton height="14px" width="80px" />
      </div>
    </div>
  );
};

/**
 * LoadingSpinner - Circular spinner for inline loading
 */
export const LoadingSpinner: React.FC<{ size?: number; color?: string }> = ({
  size = 24,
  color = '#3b82f6',
}) => {
  return (
    <div
      style={{
        width: size,
        height: size,
        border: `3px solid #e5e7eb`,
        borderTop: `3px solid ${color}`,
        borderRadius: '50%',
        animation: 'spin 0.8s linear infinite',
      }}
    >
      <style>
        {`
          @keyframes spin {
            0% {
              transform: rotate(0deg);
            }
            100% {
              transform: rotate(360deg);
            }
          }
        `}
      </style>
    </div>
  );
};

/**
 * LoadingPage - Full page loading state
 */
export const LoadingPage: React.FC<{ message?: string }> = ({
  message = 'Loading...',
}) => {
  return (
    <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        minHeight: '400px',
        gap: '16px',
      }}
    >
      <LoadingSpinner size={48} />
      <p style={{ color: '#6b7280', fontSize: '16px' }}>{message}</p>
    </div>
  );
};
