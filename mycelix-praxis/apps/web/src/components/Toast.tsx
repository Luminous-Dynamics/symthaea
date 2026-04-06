// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Toast Component
 *
 * Displays temporary notification messages with different variants
 * Accessible with ARIA live regions and keyboard dismissal
 */

import React, { useEffect } from 'react';

export type ToastVariant = 'success' | 'error' | 'info' | 'warning';

export interface ToastProps {
  id: string;
  message: string;
  variant: ToastVariant;
  duration?: number;
  onDismiss: (id: string) => void;
}

export const Toast: React.FC<ToastProps> = ({
  id,
  message,
  variant,
  duration = 5000,
  onDismiss,
}) => {
  useEffect(() => {
    if (duration > 0) {
      const timer = setTimeout(() => {
        onDismiss(id);
      }, duration);

      return () => clearTimeout(timer);
    }
  }, [id, duration, onDismiss]);

  const getVariantStyles = () => {
    switch (variant) {
      case 'success':
        return {
          backgroundColor: '#ecfdf5',
          borderColor: '#10b981',
          color: '#047857',
          icon: '✓',
        };
      case 'error':
        return {
          backgroundColor: '#fef2f2',
          borderColor: '#ef4444',
          color: '#991b1b',
          icon: '✕',
        };
      case 'warning':
        return {
          backgroundColor: '#fef3c7',
          borderColor: '#f59e0b',
          color: '#92400e',
          icon: '⚠',
        };
      case 'info':
      default:
        return {
          backgroundColor: '#eff6ff',
          borderColor: '#3b82f6',
          color: '#1e40af',
          icon: 'ℹ',
        };
    }
  };

  const styles = getVariantStyles();

  return (
    <div
      role="status"
      aria-live="polite"
      aria-atomic="true"
      className="toast-item"
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: '12px',
        padding: '12px 16px',
        borderRadius: '8px',
        backgroundColor: styles.backgroundColor,
        border: `1px solid ${styles.borderColor}`,
        color: styles.color,
        boxShadow: '0 4px 12px rgba(0, 0, 0, 0.15)',
        minWidth: '300px',
        maxWidth: '500px',
        animation: 'slideInRight 0.3s ease-out',
        marginBottom: '8px',
      }}
    >
      <span
        aria-hidden="true"
        style={{
          fontSize: '18px',
          fontWeight: '600',
          flexShrink: 0,
        }}
      >
        {styles.icon}
      </span>
      <span
        style={{
          flex: 1,
          fontSize: '14px',
          lineHeight: '1.5',
        }}
      >
        {message}
      </span>
      <button
        onClick={() => onDismiss(id)}
        aria-label="Dismiss notification"
        style={{
          background: 'none',
          border: 'none',
          color: styles.color,
          cursor: 'pointer',
          fontSize: '20px',
          padding: '0',
          width: '20px',
          height: '20px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          opacity: 0.7,
          transition: 'opacity 0.2s',
          flexShrink: 0,
        }}
        onMouseEnter={(e) => {
          e.currentTarget.style.opacity = '1';
        }}
        onMouseLeave={(e) => {
          e.currentTarget.style.opacity = '0.7';
        }}
      >
        ×
      </button>
    </div>
  );
};

export const ToastContainer: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  return (
    <div
      aria-live="polite"
      aria-atomic="false"
      style={{
        position: 'fixed',
        top: '24px',
        right: '24px',
        zIndex: 9999,
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'flex-end',
        pointerEvents: 'none',
      }}
    >
      <div style={{ pointerEvents: 'auto' }}>{children}</div>
    </div>
  );
};
