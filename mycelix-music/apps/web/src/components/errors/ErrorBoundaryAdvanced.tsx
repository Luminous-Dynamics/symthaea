// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Advanced Error Boundary
 * Feature-rich error boundary with reporting, retry, and recovery
 */

'use client';

import React, { Component, ErrorInfo, ReactNode, createContext, useContext } from 'react';

// Error severity levels
export type ErrorSeverity = 'low' | 'medium' | 'high' | 'critical';

// Error metadata
export interface ErrorMetadata {
  componentStack?: string;
  errorId?: string;
  timestamp: number;
  userAgent?: string;
  url?: string;
  userId?: string;
  sessionId?: string;
  severity: ErrorSeverity;
  tags?: Record<string, string>;
}

// Error report
export interface ErrorReport {
  error: Error;
  metadata: ErrorMetadata;
  context?: Record<string, unknown>;
}

// Error reporter interface
export interface ErrorReporter {
  report(report: ErrorReport): Promise<void>;
}

// Console reporter (default)
const consoleReporter: ErrorReporter = {
  async report(report: ErrorReport): Promise<void> {
    console.error('[ErrorBoundary]', {
      message: report.error.message,
      stack: report.error.stack,
      metadata: report.metadata,
      context: report.context,
    });
  },
};

// Error context for nested boundaries
interface ErrorContextValue {
  onError?: (error: Error, metadata: ErrorMetadata) => void;
  resetError?: () => void;
}

const ErrorContext = createContext<ErrorContextValue>({});

export const useErrorBoundary = () => useContext(ErrorContext);

// Props for ErrorBoundary
export interface ErrorBoundaryProps {
  children: ReactNode;
  fallback?: ReactNode | ((props: FallbackProps) => ReactNode);
  onError?: (error: Error, errorInfo: ErrorInfo) => void;
  onReset?: () => void;
  reporter?: ErrorReporter;
  resetKeys?: unknown[];
  level?: string;
  severity?: ErrorSeverity;
  context?: Record<string, unknown>;
  showReportButton?: boolean;
  maxRetries?: number;
}

// Fallback component props
export interface FallbackProps {
  error: Error;
  errorId: string;
  retry: () => void;
  reset: () => void;
  canRetry: boolean;
  retryCount: number;
  report: () => void;
}

// State
interface ErrorBoundaryState {
  hasError: boolean;
  error: Error | null;
  errorId: string | null;
  retryCount: number;
  isReporting: boolean;
}

export class ErrorBoundaryAdvanced extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  private prevResetKeys: unknown[] = [];
  private reporter: ErrorReporter;

  constructor(props: ErrorBoundaryProps) {
    super(props);
    this.state = {
      hasError: false,
      error: null,
      errorId: null,
      retryCount: 0,
      isReporting: false,
    };
    this.reporter = props.reporter || consoleReporter;
  }

  static getDerivedStateFromError(error: Error): Partial<ErrorBoundaryState> {
    return {
      hasError: true,
      error,
      errorId: generateErrorId(),
    };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo): void {
    const { onError, severity = 'high', context, level } = this.props;
    const { errorId } = this.state;

    const metadata: ErrorMetadata = {
      componentStack: errorInfo.componentStack || undefined,
      errorId: errorId || undefined,
      timestamp: Date.now(),
      userAgent: typeof window !== 'undefined' ? navigator.userAgent : undefined,
      url: typeof window !== 'undefined' ? window.location.href : undefined,
      severity,
      tags: { level: level || 'unknown' },
    };

    // Report error
    this.reporter.report({
      error,
      metadata,
      context,
    });

    // Call onError callback
    onError?.(error, errorInfo);
  }

  componentDidUpdate(prevProps: ErrorBoundaryProps): void {
    const { resetKeys } = this.props;
    const { hasError } = this.state;

    if (hasError && resetKeys) {
      if (resetKeys.length !== this.prevResetKeys.length ||
          resetKeys.some((key, i) => key !== this.prevResetKeys[i])) {
        this.reset();
      }
    }
    this.prevResetKeys = resetKeys || [];
  }

  private reset = (): void => {
    const { onReset } = this.props;
    this.setState({
      hasError: false,
      error: null,
      errorId: null,
      retryCount: 0,
    });
    onReset?.();
  };

  private retry = (): void => {
    const { maxRetries = 3 } = this.props;
    const { retryCount } = this.state;

    if (retryCount < maxRetries) {
      this.setState((prev) => ({
        hasError: false,
        error: null,
        retryCount: prev.retryCount + 1,
      }));
    }
  };

  private report = async (): Promise<void> => {
    const { error, errorId } = this.state;
    const { context, severity = 'high' } = this.props;

    if (!error) return;

    this.setState({ isReporting: true });

    try {
      await this.reporter.report({
        error,
        metadata: {
          errorId: errorId || undefined,
          timestamp: Date.now(),
          severity,
          tags: { source: 'user_report' },
        },
        context,
      });
    } finally {
      this.setState({ isReporting: false });
    }
  };

  render(): ReactNode {
    const { children, fallback, maxRetries = 3, showReportButton = true } = this.props;
    const { hasError, error, errorId, retryCount } = this.state;

    if (hasError && error && errorId) {
      const canRetry = retryCount < maxRetries;

      const fallbackProps: FallbackProps = {
        error,
        errorId,
        retry: this.retry,
        reset: this.reset,
        canRetry,
        retryCount,
        report: this.report,
      };

      if (typeof fallback === 'function') {
        return fallback(fallbackProps);
      }

      if (fallback) {
        return fallback;
      }

      return (
        <DefaultErrorFallback
          {...fallbackProps}
          showReportButton={showReportButton}
        />
      );
    }

    return (
      <ErrorContext.Provider value={{ onError: undefined, resetError: this.reset }}>
        {children}
      </ErrorContext.Provider>
    );
  }
}

// Default fallback component
interface DefaultFallbackProps extends FallbackProps {
  showReportButton?: boolean;
}

function DefaultErrorFallback({
  error,
  errorId,
  retry,
  reset,
  canRetry,
  retryCount,
  report,
  showReportButton = true,
}: DefaultFallbackProps): JSX.Element {
  return (
    <div className="min-h-[200px] flex items-center justify-center p-6">
      <div className="bg-red-500/10 border border-red-500/30 rounded-xl p-6 max-w-md w-full">
        <div className="flex items-center gap-3 mb-4">
          <div className="w-10 h-10 rounded-full bg-red-500/20 flex items-center justify-center">
            <svg className="w-6 h-6 text-red-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
            </svg>
          </div>
          <div>
            <h3 className="text-lg font-semibold text-white">Something went wrong</h3>
            <p className="text-sm text-gray-400">Error ID: {errorId}</p>
          </div>
        </div>

        <p className="text-gray-300 text-sm mb-4">
          {error.message || 'An unexpected error occurred.'}
        </p>

        {process.env.NODE_ENV === 'development' && error.stack && (
          <pre className="bg-black/30 rounded p-3 text-xs text-gray-400 overflow-auto max-h-32 mb-4">
            {error.stack}
          </pre>
        )}

        <div className="flex flex-wrap gap-2">
          {canRetry && (
            <button
              onClick={retry}
              className="px-4 py-2 bg-primary-600 hover:bg-primary-700 text-white rounded-lg text-sm font-medium transition-colors"
            >
              Retry ({retryCount}/{3})
            </button>
          )}
          <button
            onClick={reset}
            className="px-4 py-2 bg-white/10 hover:bg-white/20 text-white rounded-lg text-sm font-medium transition-colors"
          >
            Reset
          </button>
          {showReportButton && (
            <button
              onClick={report}
              className="px-4 py-2 bg-transparent border border-white/20 hover:border-white/40 text-white rounded-lg text-sm font-medium transition-colors"
            >
              Report Issue
            </button>
          )}
        </div>
      </div>
    </div>
  );
}

// Specialized error boundaries

export function PlayerErrorBoundary({ children }: { children: ReactNode }): JSX.Element {
  return (
    <ErrorBoundaryAdvanced
      level="player"
      severity="high"
      fallback={({ retry, canRetry }) => (
        <div className="bg-black/50 backdrop-blur rounded-lg p-4 flex items-center gap-4">
          <div className="w-12 h-12 bg-red-500/20 rounded-lg flex items-center justify-center">
            <svg className="w-6 h-6 text-red-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                d="M9.172 16.172a4 4 0 015.656 0M9 10h.01M15 10h.01M12 12h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
          </div>
          <div className="flex-1">
            <p className="text-white font-medium">Playback Error</p>
            <p className="text-gray-400 text-sm">Unable to play this track</p>
          </div>
          {canRetry && (
            <button onClick={retry} className="px-4 py-2 bg-primary-600 text-white rounded-lg text-sm">
              Retry
            </button>
          )}
        </div>
      )}
    >
      {children}
    </ErrorBoundaryAdvanced>
  );
}

export function StudioErrorBoundary({ children }: { children: ReactNode }): JSX.Element {
  return (
    <ErrorBoundaryAdvanced
      level="studio"
      severity="critical"
      maxRetries={1}
      fallback={({ reset, errorId }) => (
        <div className="min-h-screen bg-gray-900 flex items-center justify-center">
          <div className="bg-gray-800 border border-gray-700 rounded-xl p-8 max-w-lg text-center">
            <div className="w-16 h-16 bg-red-500/20 rounded-full flex items-center justify-center mx-auto mb-6">
              <svg className="w-8 h-8 text-red-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                  d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
              </svg>
            </div>
            <h2 className="text-2xl font-bold text-white mb-2">Studio Crashed</h2>
            <p className="text-gray-400 mb-4">
              The studio encountered an error. Your work has been auto-saved.
            </p>
            <p className="text-gray-500 text-sm mb-6">Error ID: {errorId}</p>
            <button
              onClick={reset}
              className="px-6 py-3 bg-primary-600 hover:bg-primary-700 text-white rounded-lg font-medium transition-colors"
            >
              Reload Studio
            </button>
          </div>
        </div>
      )}
    >
      {children}
    </ErrorBoundaryAdvanced>
  );
}

export function FormErrorBoundary({ children }: { children: ReactNode }): JSX.Element {
  return (
    <ErrorBoundaryAdvanced
      level="form"
      severity="medium"
      fallback={({ retry }) => (
        <div className="bg-yellow-500/10 border border-yellow-500/30 rounded-lg p-4">
          <p className="text-yellow-500 font-medium">Form Error</p>
          <p className="text-gray-400 text-sm mt-1">Unable to load form. Please try again.</p>
          <button onClick={retry} className="mt-3 text-primary-500 text-sm hover:underline">
            Retry
          </button>
        </div>
      )}
    >
      {children}
    </ErrorBoundaryAdvanced>
  );
}

// Utility: Generate unique error ID
function generateErrorId(): string {
  return `err_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 8)}`;
}

// Error reporting service (can be swapped out)
export function createSentryReporter(dsn: string): ErrorReporter {
  return {
    async report(report: ErrorReport): Promise<void> {
      // Sentry integration would go here
      if (typeof window !== 'undefined' && (window as any).Sentry) {
        (window as any).Sentry.captureException(report.error, {
          extra: report.context,
          tags: report.metadata.tags,
        });
      }
    },
  };
}

export function createApiReporter(endpoint: string): ErrorReporter {
  return {
    async report(report: ErrorReport): Promise<void> {
      try {
        await fetch(endpoint, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            error: {
              name: report.error.name,
              message: report.error.message,
              stack: report.error.stack,
            },
            metadata: report.metadata,
            context: report.context,
          }),
        });
      } catch {
        console.error('[ErrorReporter] Failed to report error');
      }
    },
  };
}

export default ErrorBoundaryAdvanced;
