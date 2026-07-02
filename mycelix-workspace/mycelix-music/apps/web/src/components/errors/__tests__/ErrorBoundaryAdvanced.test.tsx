// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * ErrorBoundaryAdvanced Tests
 */

import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { vi, describe, it, expect, beforeEach, afterEach } from 'vitest';
import {
  ErrorBoundaryAdvanced,
  PlayerErrorBoundary,
  StudioErrorBoundary,
  FormErrorBoundary,
  createApiReporter,
} from '../ErrorBoundaryAdvanced';

// Mock console.error to avoid noise in tests
const originalError = console.error;
beforeEach(() => {
  console.error = vi.fn();
});
afterEach(() => {
  console.error = originalError;
});

// Component that throws an error
const ThrowError = ({ shouldThrow = true }: { shouldThrow?: boolean }) => {
  if (shouldThrow) {
    throw new Error('Test error message');
  }
  return <div>No error</div>;
};

describe('ErrorBoundaryAdvanced', () => {
  it('renders children when there is no error', () => {
    render(
      <ErrorBoundaryAdvanced>
        <div>Child content</div>
      </ErrorBoundaryAdvanced>
    );

    expect(screen.getByText('Child content')).toBeInTheDocument();
  });

  it('renders fallback UI when an error occurs', () => {
    render(
      <ErrorBoundaryAdvanced>
        <ThrowError />
      </ErrorBoundaryAdvanced>
    );

    expect(screen.getByText('Something went wrong')).toBeInTheDocument();
    expect(screen.getByText('Test error message')).toBeInTheDocument();
  });

  it('displays error ID in fallback', () => {
    render(
      <ErrorBoundaryAdvanced>
        <ThrowError />
      </ErrorBoundaryAdvanced>
    );

    expect(screen.getByText(/Error ID:/)).toBeInTheDocument();
  });

  it('calls onError callback when error occurs', () => {
    const onError = vi.fn();

    render(
      <ErrorBoundaryAdvanced onError={onError}>
        <ThrowError />
      </ErrorBoundaryAdvanced>
    );

    expect(onError).toHaveBeenCalledTimes(1);
    expect(onError).toHaveBeenCalledWith(
      expect.any(Error),
      expect.objectContaining({ componentStack: expect.any(String) })
    );
  });

  it('allows retry when maxRetries > 0', () => {
    const { rerender } = render(
      <ErrorBoundaryAdvanced maxRetries={3}>
        <ThrowError />
      </ErrorBoundaryAdvanced>
    );

    const retryButton = screen.getByText(/Retry/);
    expect(retryButton).toBeInTheDocument();

    fireEvent.click(retryButton);

    // Should still show error after retry (component still throws)
    expect(screen.getByText('Something went wrong')).toBeInTheDocument();
  });

  it('calls onReset when reset button is clicked', () => {
    const onReset = vi.fn();

    render(
      <ErrorBoundaryAdvanced onReset={onReset}>
        <ThrowError />
      </ErrorBoundaryAdvanced>
    );

    const resetButton = screen.getByText('Reset');
    fireEvent.click(resetButton);

    expect(onReset).toHaveBeenCalledTimes(1);
  });

  it('renders custom fallback component', () => {
    render(
      <ErrorBoundaryAdvanced
        fallback={<div>Custom fallback</div>}
      >
        <ThrowError />
      </ErrorBoundaryAdvanced>
    );

    expect(screen.getByText('Custom fallback')).toBeInTheDocument();
  });

  it('renders custom fallback function with props', () => {
    render(
      <ErrorBoundaryAdvanced
        fallback={({ error, errorId, retry }) => (
          <div>
            <span>Custom: {error.message}</span>
            <span>ID: {errorId}</span>
            <button onClick={retry}>Custom Retry</button>
          </div>
        )}
      >
        <ThrowError />
      </ErrorBoundaryAdvanced>
    );

    expect(screen.getByText('Custom: Test error message')).toBeInTheDocument();
    expect(screen.getByText(/ID: err_/)).toBeInTheDocument();
    expect(screen.getByText('Custom Retry')).toBeInTheDocument();
  });

  it('resets when resetKeys change', async () => {
    const { rerender } = render(
      <ErrorBoundaryAdvanced resetKeys={['key1']}>
        <ThrowError />
      </ErrorBoundaryAdvanced>
    );

    expect(screen.getByText('Something went wrong')).toBeInTheDocument();

    // Change resetKeys - this should trigger a reset
    rerender(
      <ErrorBoundaryAdvanced resetKeys={['key2']}>
        <ThrowError shouldThrow={false} />
      </ErrorBoundaryAdvanced>
    );

    // After reset with non-throwing component, should render children
    await waitFor(() => {
      expect(screen.getByText('No error')).toBeInTheDocument();
    });
  });

  it('shows report button when showReportButton is true', () => {
    render(
      <ErrorBoundaryAdvanced showReportButton={true}>
        <ThrowError />
      </ErrorBoundaryAdvanced>
    );

    expect(screen.getByText('Report Issue')).toBeInTheDocument();
  });

  it('hides report button when showReportButton is false', () => {
    render(
      <ErrorBoundaryAdvanced showReportButton={false}>
        <ThrowError />
      </ErrorBoundaryAdvanced>
    );

    expect(screen.queryByText('Report Issue')).not.toBeInTheDocument();
  });
});

describe('PlayerErrorBoundary', () => {
  it('renders player-specific error UI', () => {
    render(
      <PlayerErrorBoundary>
        <ThrowError />
      </PlayerErrorBoundary>
    );

    expect(screen.getByText('Playback Error')).toBeInTheDocument();
    expect(screen.getByText('Unable to play this track')).toBeInTheDocument();
  });

  it('shows retry button', () => {
    render(
      <PlayerErrorBoundary>
        <ThrowError />
      </PlayerErrorBoundary>
    );

    expect(screen.getByText('Retry')).toBeInTheDocument();
  });
});

describe('StudioErrorBoundary', () => {
  it('renders studio crash UI', () => {
    render(
      <StudioErrorBoundary>
        <ThrowError />
      </StudioErrorBoundary>
    );

    expect(screen.getByText('Studio Crashed')).toBeInTheDocument();
    expect(screen.getByText(/Your work has been auto-saved/)).toBeInTheDocument();
  });

  it('shows reload button', () => {
    render(
      <StudioErrorBoundary>
        <ThrowError />
      </StudioErrorBoundary>
    );

    expect(screen.getByText('Reload Studio')).toBeInTheDocument();
  });
});

describe('FormErrorBoundary', () => {
  it('renders form error UI', () => {
    render(
      <FormErrorBoundary>
        <ThrowError />
      </FormErrorBoundary>
    );

    expect(screen.getByText('Form Error')).toBeInTheDocument();
    expect(screen.getByText('Unable to load form. Please try again.')).toBeInTheDocument();
  });
});

describe('createApiReporter', () => {
  it('sends error report to API endpoint', async () => {
    const mockFetch = vi.fn().mockResolvedValue({ ok: true });
    global.fetch = mockFetch;

    const reporter = createApiReporter('https://api.example.com/errors');

    await reporter.report({
      error: new Error('Test error'),
      metadata: {
        timestamp: Date.now(),
        severity: 'high',
      },
    });

    expect(mockFetch).toHaveBeenCalledWith(
      'https://api.example.com/errors',
      expect.objectContaining({
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: expect.any(String),
      })
    );

    const body = JSON.parse(mockFetch.mock.calls[0][1].body);
    expect(body.error.message).toBe('Test error');
  });
});
