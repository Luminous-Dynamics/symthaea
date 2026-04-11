// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
import { render, screen, fireEvent } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import Avatar, { AvatarGroup } from '../Avatar';

describe('Avatar', () => {
  it('should render with email', () => {
    render(<Avatar email="test@example.com" />);

    // Avatar should be present (either as image or initials)
    const avatar = screen.getByRole('img', { hidden: true });
    expect(avatar).toBeInTheDocument();
  });

  it('should display initials from name when Gravatar fails', async () => {
    render(<Avatar email="test@example.com" name="John Doe" />);

    // Find the image and trigger error
    const img = screen.getByRole('img', { hidden: true });
    fireEvent.error(img);

    // Should show initials
    expect(screen.getByText('JD')).toBeInTheDocument();
  });

  it('should display initials from email when no name and Gravatar fails', () => {
    render(<Avatar email="alice@example.com" />);

    // Trigger Gravatar failure
    const img = screen.getByRole('img', { hidden: true });
    fireEvent.error(img);

    // Should show first two letters of email username
    expect(screen.getByText('AL')).toBeInTheDocument();
  });

  it('should handle single word names', () => {
    render(<Avatar email="test@example.com" name="Alice" />);

    // Trigger Gravatar failure
    const img = screen.getByRole('img', { hidden: true });
    fireEvent.error(img);

    // Should show first two letters of name
    expect(screen.getByText('AL')).toBeInTheDocument();
  });

  it('should call onClick when clicked', () => {
    const handleClick = vi.fn();
    render(<Avatar email="test@example.com" onClick={handleClick} />);

    // Trigger Gravatar failure to ensure we can click
    const img = screen.getByRole('img', { hidden: true });
    fireEvent.error(img);

    // Find and click the avatar container
    const container = screen.getByText('TE').closest('div');
    fireEvent.click(container!);

    expect(handleClick).toHaveBeenCalledTimes(1);
  });

  it('should apply correct size classes', () => {
    const { rerender } = render(<Avatar email="test@example.com" size="sm" />);

    let container = screen.getByRole('img', { hidden: true }).closest('div');
    expect(container).toHaveClass('w-8', 'h-8');

    rerender(<Avatar email="test@example.com" size="lg" />);
    container = screen.getByRole('img', { hidden: true }).closest('div');
    expect(container).toHaveClass('w-12', 'h-12');

    rerender(<Avatar email="test@example.com" size="xl" />);
    container = screen.getByRole('img', { hidden: true }).closest('div');
    expect(container).toHaveClass('w-16', 'h-16');
  });

  it('should apply custom className', () => {
    render(<Avatar email="test@example.com" className="custom-avatar" />);

    const container = screen.getByRole('img', { hidden: true }).closest('div');
    expect(container).toHaveClass('custom-avatar');
  });
});

describe('AvatarGroup', () => {
  const mockEmails = [
    { email: 'alice@example.com', name: 'Alice Smith' },
    { email: 'bob@example.com', name: 'Bob Jones' },
    { email: 'charlie@example.com', name: 'Charlie Brown' },
    { email: 'diana@example.com', name: 'Diana Prince' },
    { email: 'eve@example.com', name: 'Eve Wilson' },
  ];

  it('should display up to max avatars', () => {
    render(<AvatarGroup emails={mockEmails} max={3} />);

    // Should show 3 avatars + overflow indicator
    const avatars = screen.getAllByRole('img', { hidden: true });
    expect(avatars.length).toBe(3);

    // Should show +2 indicator
    expect(screen.getByText('+2')).toBeInTheDocument();
  });

  it('should not show overflow when emails fit within max', () => {
    render(<AvatarGroup emails={mockEmails.slice(0, 2)} max={3} />);

    // Should show 2 avatars
    const avatars = screen.getAllByRole('img', { hidden: true });
    expect(avatars.length).toBe(2);

    // Should not show overflow indicator
    expect(screen.queryByText(/\+/)).not.toBeInTheDocument();
  });

  it('should apply custom className', () => {
    const { container } = render(
      <AvatarGroup emails={mockEmails} className="custom-group" />
    );

    expect(container.firstChild).toHaveClass('custom-group');
  });
});
