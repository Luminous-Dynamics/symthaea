// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Accessibility Utilities Tests
 */

import React, { useRef } from 'react';
import { render, screen, fireEvent, waitFor, act } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { vi, describe, it, expect, beforeEach, afterEach } from 'vitest';
import {
  A11yProvider,
  useA11y,
  SkipLink,
  VisuallyHidden,
  useFocusTrap,
  useRovingTabindex,
  useFocusRestore,
  IconButton,
  LiveRegion,
  ProgressIndicator,
  LoadingState,
  KeyboardShortcut,
  Tabs,
  getContrastRatio,
  meetsContrastRequirements,
} from '../accessibility';

// Test component that uses useA11y
const TestA11yConsumer = () => {
  const { announce, reducedMotion, focusVisible } = useA11y();

  return (
    <div>
      <button onClick={() => announce('Test announcement')}>Announce</button>
      <span data-testid="reduced-motion">{reducedMotion.toString()}</span>
      <span data-testid="focus-visible">{focusVisible.toString()}</span>
    </div>
  );
};

describe('A11yProvider', () => {
  it('provides context to children', () => {
    render(
      <A11yProvider>
        <TestA11yConsumer />
      </A11yProvider>
    );

    expect(screen.getByTestId('reduced-motion')).toBeInTheDocument();
    expect(screen.getByTestId('focus-visible')).toBeInTheDocument();
  });

  it('announces messages', async () => {
    render(
      <A11yProvider>
        <TestA11yConsumer />
      </A11yProvider>
    );

    const button = screen.getByText('Announce');
    fireEvent.click(button);

    // Check live region contains announcement
    await waitFor(() => {
      const liveRegion = document.querySelector('[role="status"]');
      expect(liveRegion).toHaveTextContent('Test announcement');
    });
  });

  it('throws error when useA11y is used outside provider', () => {
    const consoleError = vi.spyOn(console, 'error').mockImplementation(() => {});

    expect(() => {
      render(<TestA11yConsumer />);
    }).toThrow('useA11y must be used within A11yProvider');

    consoleError.mockRestore();
  });
});

describe('SkipLink', () => {
  it('renders with default text', () => {
    render(<SkipLink />);

    const link = screen.getByText('Skip to main content');
    expect(link).toBeInTheDocument();
    expect(link).toHaveAttribute('href', '#main-content');
  });

  it('renders with custom href and text', () => {
    render(<SkipLink href="#custom" children="Skip to custom" />);

    const link = screen.getByText('Skip to custom');
    expect(link).toHaveAttribute('href', '#custom');
  });

  it('is visually hidden by default', () => {
    render(<SkipLink />);

    const link = screen.getByText('Skip to main content');
    expect(link).toHaveClass('sr-only');
  });
});

describe('VisuallyHidden', () => {
  it('renders with sr-only class', () => {
    render(<VisuallyHidden>Hidden text</VisuallyHidden>);

    const element = screen.getByText('Hidden text');
    expect(element).toHaveClass('sr-only');
  });

  it('renders as specified element', () => {
    render(<VisuallyHidden as="div">Hidden text</VisuallyHidden>);

    const element = screen.getByText('Hidden text');
    expect(element.tagName).toBe('DIV');
  });
});

describe('useFocusTrap', () => {
  const FocusTrapTest = ({ isActive }: { isActive: boolean }) => {
    const containerRef = useFocusTrap(isActive);

    return (
      <div ref={containerRef} data-testid="trap-container">
        <button data-testid="first">First</button>
        <button data-testid="second">Second</button>
        <button data-testid="last">Last</button>
      </div>
    );
  };

  it('focuses first element when active', async () => {
    render(<FocusTrapTest isActive={true} />);

    await waitFor(() => {
      expect(screen.getByTestId('first')).toHaveFocus();
    });
  });

  it('does not trap focus when inactive', () => {
    render(<FocusTrapTest isActive={false} />);

    expect(screen.getByTestId('first')).not.toHaveFocus();
  });
});

describe('IconButton', () => {
  it('renders with aria-label', () => {
    render(
      <IconButton
        icon={<span>Icon</span>}
        label="Test button"
        onClick={() => {}}
      />
    );

    const button = screen.getByRole('button', { name: 'Test button' });
    expect(button).toBeInTheDocument();
    expect(button).toHaveAttribute('aria-label', 'Test button');
  });

  it('calls onClick when clicked', async () => {
    const onClick = vi.fn();
    const user = userEvent.setup();

    render(
      <IconButton
        icon={<span>Icon</span>}
        label="Test button"
        onClick={onClick}
      />
    );

    await user.click(screen.getByRole('button'));
    expect(onClick).toHaveBeenCalledTimes(1);
  });

  it('is disabled when disabled prop is true', () => {
    render(
      <IconButton
        icon={<span>Icon</span>}
        label="Test button"
        disabled={true}
      />
    );

    expect(screen.getByRole('button')).toBeDisabled();
  });
});

describe('LiveRegion', () => {
  it('renders with correct ARIA attributes', () => {
    render(<LiveRegion message="Test message" />);

    const region = screen.getByRole('status');
    expect(region).toHaveAttribute('aria-live', 'polite');
    expect(region).toHaveAttribute('aria-atomic', 'true');
    expect(region).toHaveTextContent('Test message');
  });

  it('uses assertive priority when specified', () => {
    render(<LiveRegion message="Urgent message" priority="assertive" />);

    const region = screen.getByRole('status');
    expect(region).toHaveAttribute('aria-live', 'assertive');
  });
});

describe('ProgressIndicator', () => {
  it('renders with correct ARIA attributes', () => {
    render(<ProgressIndicator value={50} max={100} label="Progress" />);

    const progressbar = screen.getByRole('progressbar');
    expect(progressbar).toHaveAttribute('aria-valuenow', '50');
    expect(progressbar).toHaveAttribute('aria-valuemin', '0');
    expect(progressbar).toHaveAttribute('aria-valuemax', '100');
    expect(progressbar).toHaveAttribute('aria-label', 'Progress');
  });

  it('displays percentage when showValue is true', () => {
    render(<ProgressIndicator value={75} max={100} label="Progress" showValue={true} />);

    expect(screen.getByText('75%')).toBeInTheDocument();
  });

  it('hides percentage when showValue is false', () => {
    render(<ProgressIndicator value={75} max={100} label="Progress" showValue={false} />);

    expect(screen.queryByText('75%')).not.toBeInTheDocument();
  });
});

describe('LoadingState', () => {
  it('renders with status role', () => {
    render(<LoadingState />);

    const status = screen.getByRole('status');
    expect(status).toBeInTheDocument();
    expect(status).toHaveAttribute('aria-label', 'Loading...');
  });

  it('uses custom label', () => {
    render(<LoadingState label="Fetching data..." />);

    expect(screen.getByRole('status')).toHaveAttribute('aria-label', 'Fetching data...');
  });
});

describe('KeyboardShortcut', () => {
  it('renders keyboard keys', () => {
    render(<KeyboardShortcut keys={['Ctrl', 'S']} />);

    expect(screen.getByText('Ctrl')).toBeInTheDocument();
    expect(screen.getByText('S')).toBeInTheDocument();
  });

  it('renders plus signs between keys', () => {
    render(<KeyboardShortcut keys={['Ctrl', 'Shift', 'P']} />);

    const plusSigns = screen.getAllByText('+');
    expect(plusSigns).toHaveLength(2);
  });
});

describe('Tabs', () => {
  const tabs = [
    { id: 'tab1', label: 'Tab 1', content: <div>Content 1</div> },
    { id: 'tab2', label: 'Tab 2', content: <div>Content 2</div> },
    { id: 'tab3', label: 'Tab 3', content: <div>Content 3</div> },
  ];

  it('renders tabs with correct ARIA attributes', () => {
    render(
      <Tabs
        tabs={tabs}
        activeTab="tab1"
        onTabChange={() => {}}
        label="Test tabs"
      />
    );

    const tablist = screen.getByRole('tablist');
    expect(tablist).toHaveAttribute('aria-label', 'Test tabs');

    const tabButtons = screen.getAllByRole('tab');
    expect(tabButtons).toHaveLength(3);
  });

  it('shows active tab content', () => {
    render(
      <Tabs
        tabs={tabs}
        activeTab="tab1"
        onTabChange={() => {}}
        label="Test tabs"
      />
    );

    expect(screen.getByText('Content 1')).toBeVisible();
  });

  it('calls onTabChange when tab is clicked', async () => {
    const onTabChange = vi.fn();
    const user = userEvent.setup();

    render(
      <Tabs
        tabs={tabs}
        activeTab="tab1"
        onTabChange={onTabChange}
        label="Test tabs"
      />
    );

    await user.click(screen.getByText('Tab 2'));
    expect(onTabChange).toHaveBeenCalledWith('tab2');
  });

  it('supports keyboard navigation', async () => {
    const onTabChange = vi.fn();
    const user = userEvent.setup();

    render(
      <Tabs
        tabs={tabs}
        activeTab="tab1"
        onTabChange={onTabChange}
        label="Test tabs"
      />
    );

    // Focus first tab
    await user.click(screen.getByText('Tab 1'));

    // Arrow right to next tab
    await user.keyboard('{ArrowRight}');
    expect(onTabChange).toHaveBeenCalledWith('tab2');
  });
});

describe('Contrast Utilities', () => {
  describe('getContrastRatio', () => {
    it('returns 21 for black and white', () => {
      const ratio = getContrastRatio('#000000', '#ffffff');
      expect(ratio).toBeCloseTo(21, 0);
    });

    it('returns 1 for same colors', () => {
      const ratio = getContrastRatio('#ff0000', '#ff0000');
      expect(ratio).toBe(1);
    });
  });

  describe('meetsContrastRequirements', () => {
    it('returns true for high contrast colors at AA level', () => {
      expect(meetsContrastRequirements('#000000', '#ffffff', 'AA')).toBe(true);
    });

    it('returns true for high contrast colors at AAA level', () => {
      expect(meetsContrastRequirements('#000000', '#ffffff', 'AAA')).toBe(true);
    });

    it('returns false for low contrast colors', () => {
      expect(meetsContrastRequirements('#777777', '#888888', 'AA')).toBe(false);
    });

    it('has lower threshold for large text', () => {
      // Medium gray on white might pass for large text but not small
      const result = meetsContrastRequirements('#767676', '#ffffff', 'AA', true);
      expect(result).toBe(true);
    });
  });
});
