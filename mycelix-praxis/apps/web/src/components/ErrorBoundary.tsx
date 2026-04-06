// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * ErrorBoundary Component
 *
 * Catches React errors and displays a fallback UI
 */

import { Component, ErrorInfo, ReactNode } from 'react';

interface Props {
  children: ReactNode;
  fallback?: ReactNode;
}

interface State {
  hasError: boolean;
  error: Error | null;
}

export class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = {
      hasError: false,
      error: null,
    };
  }

  static getDerivedStateFromError(error: Error): State {
    return {
      hasError: true,
      error,
    };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error('ErrorBoundary caught an error:', error, errorInfo);
    // In production, you could send this to an error tracking service
    // like Sentry, LogRocket, etc.
  }

  handleReset = () => {
    this.setState({
      hasError: false,
      error: null,
    });
  };

  render() {
    if (this.state.hasError) {
      if (this.props.fallback) {
        return this.props.fallback;
      }

      const isDevelopment = import.meta.env.DEV;

      return (
        <div
          role="alert"
          style={{
            minHeight: '100vh',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            padding: '40px 24px',
            backgroundColor: '#f9fafb',
          }}
        >
          <div
            style={{
              textAlign: 'center',
              maxWidth: '600px',
              backgroundColor: '#ffffff',
              padding: '48px',
              borderRadius: '16px',
              boxShadow: '0 4px 20px rgba(0, 0, 0, 0.1)',
            }}
          >
            <div
              aria-hidden="true"
              style={{
                fontSize: '64px',
                marginBottom: '24px',
              }}
            >
              ⚠️
            </div>
            <h1
              style={{
                fontSize: '28px',
                fontWeight: '700',
                color: '#111827',
                marginBottom: '12px',
                margin: 0,
              }}
            >
              Oops! Something went wrong
            </h1>
            <p
              style={{
                fontSize: '16px',
                color: '#6b7280',
                marginBottom: '32px',
                lineHeight: '1.6',
              }}
            >
              We encountered an unexpected error. Don't worry, your data is safe.
              Try reloading the page or returning to the home page.
            </p>

            {this.state.error && isDevelopment && (
              <details
                style={{
                  marginBottom: '32px',
                  padding: '16px',
                  backgroundColor: '#fef2f2',
                  border: '2px solid #fecaca',
                  borderRadius: '8px',
                  textAlign: 'left',
                }}
              >
                <summary
                  style={{
                    cursor: 'pointer',
                    fontWeight: '600',
                    color: '#dc2626',
                    marginBottom: '12px',
                    fontSize: '14px',
                  }}
                >
                  🐛 Error Details (Development Only)
                </summary>
                <pre
                  style={{
                    fontSize: '12px',
                    color: '#991b1b',
                    overflow: 'auto',
                    margin: 0,
                    lineHeight: '1.5',
                    fontFamily: 'monospace',
                  }}
                >
                  {this.state.error.toString()}
                  {this.state.error.stack && `\n\n${this.state.error.stack}`}
                </pre>
              </details>
            )}

            <div style={{ display: 'flex', gap: '12px', justifyContent: 'center', flexWrap: 'wrap' }}>
              <button
                onClick={() => window.location.reload()}
                style={{
                  padding: '12px 24px',
                  fontSize: '16px',
                  fontWeight: '600',
                  color: '#ffffff',
                  backgroundColor: '#3b82f6',
                  border: 'none',
                  borderRadius: '8px',
                  cursor: 'pointer',
                  transition: 'all 0.2s ease',
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.backgroundColor = '#2563eb';
                  e.currentTarget.style.transform = 'translateY(-1px)';
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.backgroundColor = '#3b82f6';
                  e.currentTarget.style.transform = 'translateY(0)';
                }}
              >
                🔄 Reload Page
              </button>
              <button
                onClick={() => window.location.href = '/'}
                style={{
                  padding: '12px 24px',
                  fontSize: '16px',
                  fontWeight: '600',
                  color: '#4b5563',
                  backgroundColor: '#f3f4f6',
                  border: '1px solid #d1d5db',
                  borderRadius: '8px',
                  cursor: 'pointer',
                  transition: 'all 0.2s ease',
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.backgroundColor = '#e5e7eb';
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.backgroundColor = '#f3f4f6';
                }}
              >
                🏠 Go Home
              </button>
            </div>

            <div style={{ marginTop: '32px', paddingTop: '24px', borderTop: '1px solid #e5e7eb' }}>
              <p style={{ fontSize: '14px', color: '#9ca3af', marginBottom: '12px' }}>
                If this problem persists, please report it on GitHub
              </p>
              <a
                href="https://github.com/Luminous-Dynamics/mycelix-edunet/issues"
                target="_blank"
                rel="noopener noreferrer"
                style={{
                  color: '#3b82f6',
                  textDecoration: 'none',
                  fontSize: '14px',
                  fontWeight: '500',
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.textDecoration = 'underline';
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.textDecoration = 'none';
                }}
              >
                Report Issue →
              </a>
            </div>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}
