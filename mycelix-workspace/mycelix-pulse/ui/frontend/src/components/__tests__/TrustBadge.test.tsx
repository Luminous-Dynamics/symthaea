// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
import { render, screen } from '@testing-library/react';
import { describe, it, expect } from 'vitest';
import { TrustBadge } from '../TrustBadge';
import type { TrustSummary } from '@/store/trustStore';

describe('TrustBadge', () => {
  it('should render high trust with correct styling', () => {
    const summary: TrustSummary = {
      tier: 'high',
      score: 85,
      reasons: ['Verified sender'],
    };

    render(<TrustBadge summary={summary} />);

    expect(screen.getByText('High trust')).toBeInTheDocument();
    expect(screen.getByText('Score 85')).toBeInTheDocument();
  });

  it('should render medium trust correctly', () => {
    const summary: TrustSummary = {
      tier: 'medium',
      score: 55,
      reasons: ['New sender'],
    };

    render(<TrustBadge summary={summary} />);

    expect(screen.getByText('Neutral trust')).toBeInTheDocument();
    expect(screen.getByText('Score 55')).toBeInTheDocument();
  });

  it('should render low trust correctly', () => {
    const summary: TrustSummary = {
      tier: 'low',
      score: 15,
      reasons: ['Suspicious content', 'Unknown sender'],
    };

    render(<TrustBadge summary={summary} />);

    expect(screen.getByText('Low trust')).toBeInTheDocument();
    expect(screen.getByText('Score 15')).toBeInTheDocument();
  });

  it('should render unknown trust correctly', () => {
    const summary: TrustSummary = {
      tier: 'unknown',
      reasons: [],
    };

    render(<TrustBadge summary={summary} />);

    expect(screen.getByText('Unknown trust')).toBeInTheDocument();
    expect(screen.getByText(/Score/)).toBeInTheDocument();
  });

  it('should hide label in compact mode', () => {
    const summary: TrustSummary = {
      tier: 'high',
      score: 90,
      reasons: [],
    };

    render(<TrustBadge summary={summary} compact />);

    expect(screen.queryByText('High trust')).not.toBeInTheDocument();
    expect(screen.getByText('Score 90')).toBeInTheDocument();
  });

  it('should display dash when score is undefined', () => {
    const summary: TrustSummary = {
      tier: 'unknown',
      reasons: [],
      score: undefined,
    };

    render(<TrustBadge summary={summary} />);

    expect(screen.getByText(/–/)).toBeInTheDocument();
  });

  it('should include reasons in title attribute', () => {
    const summary: TrustSummary = {
      tier: 'low',
      score: 20,
      reasons: ['No display name', 'Tagged sender address'],
    };

    render(<TrustBadge summary={summary} />);

    // The outer span has the title attribute
    const badge = screen.getByTitle(/Low trust/);
    expect(badge).toHaveAttribute(
      'title',
      'Low trust • No display name, Tagged sender address'
    );
  });

  it('should apply custom className', () => {
    const summary: TrustSummary = {
      tier: 'high',
      score: 85,
      reasons: [],
    };

    const { container } = render(<TrustBadge summary={summary} className="custom-class" />);

    // The root span element should have the custom class
    const badge = container.firstChild;
    expect(badge).toHaveClass('custom-class');
  });
});
