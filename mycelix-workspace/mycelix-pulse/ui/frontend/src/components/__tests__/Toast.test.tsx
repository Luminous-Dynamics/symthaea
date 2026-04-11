// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import Toast from '../Toast';
import type { Toast as ToastType } from '@/store/toastStore';

describe('Toast', () => {
  const mockOnClose = vi.fn();

  beforeEach(() => {
    mockOnClose.mockClear();
  });

  it('should render success toast with message', () => {
    const toast: ToastType = {
      id: '1',
      type: 'success',
      message: 'Email sent successfully',
    };

    render(<Toast toast={toast} onClose={mockOnClose} />);

    expect(screen.getByText('Email sent successfully')).toBeInTheDocument();
    expect(screen.getByRole('alert')).toHaveClass('bg-green-50');
  });

  it('should render error toast with message', () => {
    const toast: ToastType = {
      id: '2',
      type: 'error',
      message: 'Failed to send email',
    };

    render(<Toast toast={toast} onClose={mockOnClose} />);

    expect(screen.getByText('Failed to send email')).toBeInTheDocument();
    expect(screen.getByRole('alert')).toHaveClass('bg-red-50');
  });

  it('should render warning toast with message', () => {
    const toast: ToastType = {
      id: '3',
      type: 'warning',
      message: 'Low trust sender detected',
    };

    render(<Toast toast={toast} onClose={mockOnClose} />);

    expect(screen.getByText('Low trust sender detected')).toBeInTheDocument();
    expect(screen.getByRole('alert')).toHaveClass('bg-yellow-50');
  });

  it('should render info toast with message', () => {
    const toast: ToastType = {
      id: '4',
      type: 'info',
      message: 'New email received',
    };

    render(<Toast toast={toast} onClose={mockOnClose} />);

    expect(screen.getByText('New email received')).toBeInTheDocument();
    expect(screen.getByRole('alert')).toHaveClass('bg-blue-50');
  });

  it('should call onClose when close button is clicked', async () => {
    const toast: ToastType = {
      id: '5',
      type: 'success',
      message: 'Test message',
    };

    render(<Toast toast={toast} onClose={mockOnClose} />);

    const closeButton = screen.getByLabelText('Close');
    fireEvent.click(closeButton);

    await waitFor(
      () => {
        expect(mockOnClose).toHaveBeenCalledTimes(1);
      },
      { timeout: 500 }
    );
  });

  it('should render action button when action is provided', () => {
    const actionHandler = vi.fn();
    const toast: ToastType = {
      id: '6',
      type: 'info',
      message: 'Email moved to trash',
      action: {
        label: 'Undo',
        onClick: actionHandler,
      },
    };

    render(<Toast toast={toast} onClose={mockOnClose} />);

    expect(screen.getByText('Undo')).toBeInTheDocument();
  });

  it('should call action onClick and close when action button is clicked', async () => {
    const actionHandler = vi.fn();
    const toast: ToastType = {
      id: '7',
      type: 'info',
      message: 'Email deleted',
      action: {
        label: 'Undo',
        onClick: actionHandler,
      },
    };

    render(<Toast toast={toast} onClose={mockOnClose} />);

    const actionButton = screen.getByText('Undo');
    fireEvent.click(actionButton);

    expect(actionHandler).toHaveBeenCalledTimes(1);

    await waitFor(
      () => {
        expect(mockOnClose).toHaveBeenCalledTimes(1);
      },
      { timeout: 500 }
    );
  });

  it('should have correct accessibility role', () => {
    const toast: ToastType = {
      id: '8',
      type: 'error',
      message: 'Something went wrong',
    };

    render(<Toast toast={toast} onClose={mockOnClose} />);

    expect(screen.getByRole('alert')).toBeInTheDocument();
  });
});
