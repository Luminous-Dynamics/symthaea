// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Error Boundary Components
 *
 * Graceful error handling for epistemic components:
 * - Component-level error boundaries
 * - Retry functionality
 * - Error reporting
 * - Fallback UIs
 */

import { Component, ReactNode, useState, useCallback } from 'react';

// ============================================
// Class-based Error Boundary
// ============================================

interface ErrorBoundaryProps {
  children: ReactNode;
  fallback?: ReactNode | ((error: Error, reset: () => void) => ReactNode);
  onError?: (error: Error, errorInfo: React.ErrorInfo) => void;
  resetKeys?: unknown[];
}

interface ErrorBoundaryState {
  hasError: boolean;
  error: Error | null;
}

export class ErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  constructor(props: ErrorBoundaryProps) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
    console.error('[ErrorBoundary] Caught error:', error, errorInfo);
    this.props.onError?.(error, errorInfo);
  }

  componentDidUpdate(prevProps: ErrorBoundaryProps) {
    // Reset on key change
    if (
      this.state.hasError &&
      this.props.resetKeys &&
      prevProps.resetKeys &&
      this.props.resetKeys.some((key, i) => key !== prevProps.resetKeys![i])
    ) {
      this.reset();
    }
  }

  reset = () => {
    this.setState({ hasError: false, error: null });
  };

  render() {
    if (this.state.hasError && this.state.error) {
      if (typeof this.props.fallback === 'function') {
        return this.props.fallback(this.state.error, this.reset);
      }
      if (this.props.fallback) {
        return this.props.fallback;
      }
      return <DefaultErrorFallback error={this.state.error} onReset={this.reset} />;
    }

    return this.props.children;
  }
}

// ============================================
// Default Error Fallback
// ============================================

interface ErrorFallbackProps {
  error: Error;
  onReset?: () => void;
  compact?: boolean;
}

export function DefaultErrorFallback({ error, onReset, compact = false }: ErrorFallbackProps) {
  const [showDetails, setShowDetails] = useState(false);

  if (compact) {
    return (
      <div className="flex items-center gap-2 px-3 py-2 text-sm text-red-600 dark:text-red-400 bg-red-50 dark:bg-red-900/20 rounded-lg">
        <svg className="w-4 h-4 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
        </svg>
        <span className="truncate">{error.message || 'Something went wrong'}</span>
        {onReset && (
          <button
            onClick={onReset}
            className="ml-auto text-red-700 dark:text-red-300 hover:underline"
          >
            Retry
          </button>
        )}
      </div>
    );
  }

  return (
    <div className="p-6 text-center" role="alert">
      <div className="inline-flex items-center justify-center w-16 h-16 rounded-full bg-red-100 dark:bg-red-900/30 mb-4">
        <svg className="w-8 h-8 text-red-600 dark:text-red-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
        </svg>
      </div>

      <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-2">
        Something went wrong
      </h3>

      <p className="text-gray-600 dark:text-gray-400 mb-4 max-w-md mx-auto">
        {error.message || 'An unexpected error occurred. Please try again.'}
      </p>

      <div className="flex items-center justify-center gap-3">
        {onReset && (
          <button
            onClick={onReset}
            className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors"
          >
            Try Again
          </button>
        )}
        <button
          onClick={() => setShowDetails(!showDetails)}
          className="px-4 py-2 text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-100 transition-colors"
        >
          {showDetails ? 'Hide Details' : 'Show Details'}
        </button>
      </div>

      {showDetails && (
        <pre className="mt-4 p-4 bg-gray-100 dark:bg-gray-800 rounded-lg text-left text-xs text-gray-700 dark:text-gray-300 overflow-auto max-h-48">
          {error.stack || error.toString()}
        </pre>
      )}
    </div>
  );
}

// ============================================
// Specialized Error Boundaries
// ============================================

// Trust component error boundary
export function TrustErrorBoundary({ children }: { children: ReactNode }) {
  return (
    <ErrorBoundary
      fallback={(error, reset) => (
        <div className="p-4 bg-amber-50 dark:bg-amber-900/20 rounded-lg border border-amber-200 dark:border-amber-800">
          <div className="flex items-start gap-3">
            <svg className="w-5 h-5 text-amber-600 dark:text-amber-400 mt-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
            </svg>
            <div className="flex-1">
              <h4 className="font-medium text-amber-800 dark:text-amber-200">
                Trust verification unavailable
              </h4>
              <p className="text-sm text-amber-700 dark:text-amber-300 mt-1">
                Unable to load trust information. The sender's trust status could not be verified.
              </p>
              <button
                onClick={reset}
                className="mt-2 text-sm text-amber-800 dark:text-amber-200 hover:underline"
              >
                Retry verification
              </button>
            </div>
          </div>
        </div>
      )}
    >
      {children}
    </ErrorBoundary>
  );
}

// AI insights error boundary
export function AIErrorBoundary({ children }: { children: ReactNode }) {
  return (
    <ErrorBoundary
      fallback={(error, reset) => (
        <div className="p-4 bg-purple-50 dark:bg-purple-900/20 rounded-lg border border-purple-200 dark:border-purple-800">
          <div className="flex items-start gap-3">
            <svg className="w-5 h-5 text-purple-600 dark:text-purple-400 mt-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
            </svg>
            <div className="flex-1">
              <h4 className="font-medium text-purple-800 dark:text-purple-200">
                AI insights unavailable
              </h4>
              <p className="text-sm text-purple-700 dark:text-purple-300 mt-1">
                Unable to generate AI analysis. The local model may be unavailable.
              </p>
              <button
                onClick={reset}
                className="mt-2 text-sm text-purple-800 dark:text-purple-200 hover:underline"
              >
                Retry analysis
              </button>
            </div>
          </div>
        </div>
      )}
    >
      {children}
    </ErrorBoundary>
  );
}

// Network error boundary (for WebSocket/API)
export function NetworkErrorBoundary({ children }: { children: ReactNode }) {
  return (
    <ErrorBoundary
      fallback={(error, reset) => (
        <div className="p-4 bg-gray-50 dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700">
          <div className="flex items-start gap-3">
            <svg className="w-5 h-5 text-gray-600 dark:text-gray-400 mt-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8.111 16.404a5.5 5.5 0 017.778 0M12 20h.01m-7.08-7.071c3.904-3.905 10.236-3.905 14.141 0M1.394 9.393c5.857-5.857 15.355-5.857 21.213 0" />
            </svg>
            <div className="flex-1">
              <h4 className="font-medium text-gray-800 dark:text-gray-200">
                Connection error
              </h4>
              <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                Unable to connect to the server. Please check your connection.
              </p>
              <button
                onClick={reset}
                className="mt-2 text-sm text-blue-600 dark:text-blue-400 hover:underline"
              >
                Retry connection
              </button>
            </div>
          </div>
        </div>
      )}
    >
      {children}
    </ErrorBoundary>
  );
}

// ============================================
// Query Error Handler
// ============================================

interface QueryErrorProps {
  error: Error;
  onRetry?: () => void;
  message?: string;
}

export function QueryError({ error, onRetry, message }: QueryErrorProps) {
  return (
    <div className="flex flex-col items-center justify-center p-8 text-center">
      <div className="w-12 h-12 rounded-full bg-red-100 dark:bg-red-900/30 flex items-center justify-center mb-4">
        <svg className="w-6 h-6 text-red-600 dark:text-red-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
        </svg>
      </div>
      <p className="text-gray-600 dark:text-gray-400 mb-4">
        {message || error.message || 'Failed to load data'}
      </p>
      {onRetry && (
        <button
          onClick={onRetry}
          className="px-4 py-2 text-sm bg-gray-100 dark:bg-gray-800 hover:bg-gray-200 dark:hover:bg-gray-700 rounded-lg transition-colors"
        >
          Try Again
        </button>
      )}
    </div>
  );
}

// ============================================
// Async Boundary (Suspense + Error)
// ============================================

interface AsyncBoundaryProps {
  children: ReactNode;
  loading?: ReactNode;
  error?: ReactNode | ((error: Error, reset: () => void) => ReactNode);
}

export function AsyncBoundary({ children, loading, error }: AsyncBoundaryProps) {
  return (
    <ErrorBoundary fallback={error}>
      {/* Note: Suspense would go here if using React.lazy or use() */}
      {children}
    </ErrorBoundary>
  );
}

// ============================================
// Hook for error handling in components
// ============================================

export function useErrorHandler() {
  const [error, setError] = useState<Error | null>(null);

  const handleError = useCallback((err: unknown) => {
    const error = err instanceof Error ? err : new Error(String(err));
    setError(error);
    console.error('[useErrorHandler]', error);
  }, []);

  const clearError = useCallback(() => {
    setError(null);
  }, []);

  const withErrorHandling = useCallback(
    <T extends (...args: unknown[]) => Promise<unknown>>(fn: T) => {
      return async (...args: Parameters<T>) => {
        try {
          clearError();
          return await fn(...args);
        } catch (err) {
          handleError(err);
          throw err;
        }
      };
    },
    [handleError, clearError]
  );

  return {
    error,
    handleError,
    clearError,
    withErrorHandling,
  };
}

export default ErrorBoundary;
