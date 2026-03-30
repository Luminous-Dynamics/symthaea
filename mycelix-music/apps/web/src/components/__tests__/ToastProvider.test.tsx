// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
import React from 'react';
import { render, screen, act, waitFor } from '@testing-library/react';
import { ToastProvider, useToast } from '../ToastProvider';

// Test component that uses the toast hook
function ToastTester() {
  const { show } = useToast();

  return (
    <div>
      <button onClick={() => show('Info message', 'info')}>Show Info</button>
      <button onClick={() => show('Success message', 'success')}>Show Success</button>
      <button onClick={() => show('Error message', 'error')}>Show Error</button>
      <button onClick={() => show('Default type')}>Show Default</button>
    </div>
  );
}

describe('ToastProvider', () => {
  beforeEach(() => {
    jest.useFakeTimers();
  });

  afterEach(() => {
    jest.runOnlyPendingTimers();
    jest.useRealTimers();
  });

  it('renders children correctly', () => {
    render(
      <ToastProvider>
        <div>Child content</div>
      </ToastProvider>
    );

    expect(screen.getByText('Child content')).toBeInTheDocument();
  });

  it('shows info toast when triggered', () => {
    render(
      <ToastProvider>
        <ToastTester />
      </ToastProvider>
    );

    act(() => {
      screen.getByText('Show Info').click();
    });

    expect(screen.getByText('Info message')).toBeInTheDocument();
  });

  it('shows success toast when triggered', () => {
    render(
      <ToastProvider>
        <ToastTester />
      </ToastProvider>
    );

    act(() => {
      screen.getByText('Show Success').click();
    });

    expect(screen.getByText('Success message')).toBeInTheDocument();
  });

  it('shows error toast when triggered', () => {
    render(
      <ToastProvider>
        <ToastTester />
      </ToastProvider>
    );

    act(() => {
      screen.getByText('Show Error').click();
    });

    expect(screen.getByText('Error message')).toBeInTheDocument();
  });

  it('uses info type as default when no type specified', () => {
    render(
      <ToastProvider>
        <ToastTester />
      </ToastProvider>
    );

    act(() => {
      screen.getByText('Show Default').click();
    });

    expect(screen.getByText('Default type')).toBeInTheDocument();
  });

  it('auto-dismisses toast after timeout', async () => {
    render(
      <ToastProvider>
        <ToastTester />
      </ToastProvider>
    );

    act(() => {
      screen.getByText('Show Info').click();
    });

    expect(screen.getByText('Info message')).toBeInTheDocument();

    // Fast-forward past the auto-dismiss timeout (2500ms)
    act(() => {
      jest.advanceTimersByTime(3000);
    });

    await waitFor(() => {
      expect(screen.queryByText('Info message')).not.toBeInTheDocument();
    });
  });

  it('can show multiple toasts at once', () => {
    render(
      <ToastProvider>
        <ToastTester />
      </ToastProvider>
    );

    act(() => {
      screen.getByText('Show Info').click();
      screen.getByText('Show Success').click();
      screen.getByText('Show Error').click();
    });

    expect(screen.getByText('Info message')).toBeInTheDocument();
    expect(screen.getByText('Success message')).toBeInTheDocument();
    expect(screen.getByText('Error message')).toBeInTheDocument();
  });
});

describe('useToast', () => {
  it('throws error when used outside ToastProvider', () => {
    const TestComponent = () => {
      useToast();
      return null;
    };

    // Suppress console.error for this test
    const originalError = console.error;
    console.error = jest.fn();

    expect(() => render(<TestComponent />)).toThrow(
      'useToast must be used within ToastProvider'
    );

    console.error = originalError;
  });
});
