// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
import type { Meta, StoryObj } from '@storybook/react';
import React, { useState } from 'react';

/**
 * Accessibility Components Stories
 *
 * Components and utilities for building accessible interfaces
 * that comply with WCAG 2.1 AA standards.
 */

// Mock implementations - replace with actual imports
interface SkipLinkProps {
  href?: string;
  children?: React.ReactNode;
}

const SkipLink = ({ href = '#main-content', children = 'Skip to main content' }: SkipLinkProps) => (
  <a
    href={href}
    className="sr-only focus:not-sr-only focus:absolute focus:top-4 focus:left-4 focus:z-50 focus:px-4 focus:py-2 focus:bg-purple-600 focus:text-white focus:rounded-md"
  >
    {children}
  </a>
);

const VisuallyHidden = ({ children }: { children: React.ReactNode }) => (
  <span className="sr-only">{children}</span>
);

interface IconButtonProps {
  icon: React.ReactNode;
  label: string;
  onClick?: () => void;
  disabled?: boolean;
  variant?: 'default' | 'ghost';
}

const IconButton = ({
  icon,
  label,
  onClick,
  disabled = false,
  variant = 'default',
}: IconButtonProps) => (
  <button
    onClick={onClick}
    disabled={disabled}
    aria-label={label}
    className={`
      p-2 rounded-lg transition-colors focus:outline-none focus:ring-2 focus:ring-purple-500
      ${variant === 'default' ? 'bg-gray-700 hover:bg-gray-600' : 'hover:bg-gray-800'}
      ${disabled ? 'opacity-50 cursor-not-allowed' : ''}
    `}
  >
    {icon}
  </button>
);

interface LiveRegionProps {
  message: string;
  priority?: 'polite' | 'assertive';
}

const LiveRegion = ({ message, priority = 'polite' }: LiveRegionProps) => (
  <div
    role="status"
    aria-live={priority}
    aria-atomic="true"
    className="text-sm text-gray-400"
  >
    {message}
  </div>
);

interface ProgressIndicatorProps {
  value: number;
  max?: number;
  label: string;
  showValue?: boolean;
}

const ProgressIndicator = ({
  value,
  max = 100,
  label,
  showValue = true,
}: ProgressIndicatorProps) => {
  const percentage = Math.round((value / max) * 100);

  return (
    <div className="w-full">
      <div className="flex justify-between mb-1">
        <span className="text-sm text-gray-400">{label}</span>
        {showValue && <span className="text-sm text-gray-400">{percentage}%</span>}
      </div>
      <div
        role="progressbar"
        aria-valuenow={value}
        aria-valuemin={0}
        aria-valuemax={max}
        aria-label={label}
        className="h-2 bg-gray-700 rounded-full overflow-hidden"
      >
        <div
          className="h-full bg-purple-500 transition-all duration-300"
          style={{ width: `${percentage}%` }}
        />
      </div>
    </div>
  );
};

interface LoadingStateProps {
  label?: string;
  size?: 'sm' | 'md' | 'lg';
}

const LoadingState = ({ label = 'Loading...', size = 'md' }: LoadingStateProps) => {
  const sizes = {
    sm: 'w-4 h-4',
    md: 'w-8 h-8',
    lg: 'w-12 h-12',
  };

  return (
    <div
      role="status"
      aria-label={label}
      className="flex items-center justify-center gap-2"
    >
      <svg
        className={`${sizes[size]} animate-spin text-purple-500`}
        viewBox="0 0 24 24"
        fill="none"
      >
        <circle
          className="opacity-25"
          cx="12"
          cy="12"
          r="10"
          stroke="currentColor"
          strokeWidth="4"
        />
        <path
          className="opacity-75"
          fill="currentColor"
          d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"
        />
      </svg>
      <VisuallyHidden>{label}</VisuallyHidden>
    </div>
  );
};

interface KeyboardShortcutProps {
  keys: string[];
}

const KeyboardShortcut = ({ keys }: KeyboardShortcutProps) => (
  <span className="inline-flex items-center gap-1">
    {keys.map((key, index) => (
      <React.Fragment key={key}>
        <kbd className="px-2 py-1 text-xs font-mono bg-gray-800 border border-gray-700 rounded">
          {key}
        </kbd>
        {index < keys.length - 1 && <span className="text-gray-500">+</span>}
      </React.Fragment>
    ))}
  </span>
);

interface Tab {
  id: string;
  label: string;
  content: React.ReactNode;
}

interface TabsProps {
  tabs: Tab[];
  activeTab: string;
  onTabChange: (id: string) => void;
  label: string;
}

const Tabs = ({ tabs, activeTab, onTabChange, label }: TabsProps) => (
  <div className="w-full">
    <div role="tablist" aria-label={label} className="flex border-b border-gray-700">
      {tabs.map((tab) => (
        <button
          key={tab.id}
          role="tab"
          id={`tab-${tab.id}`}
          aria-selected={activeTab === tab.id}
          aria-controls={`panel-${tab.id}`}
          onClick={() => onTabChange(tab.id)}
          className={`
            px-4 py-2 text-sm font-medium transition-colors
            ${activeTab === tab.id
              ? 'text-purple-400 border-b-2 border-purple-400'
              : 'text-gray-400 hover:text-white'
            }
          `}
        >
          {tab.label}
        </button>
      ))}
    </div>
    {tabs.map((tab) => (
      <div
        key={tab.id}
        role="tabpanel"
        id={`panel-${tab.id}`}
        aria-labelledby={`tab-${tab.id}`}
        hidden={activeTab !== tab.id}
        className="p-4"
      >
        {tab.content}
      </div>
    ))}
  </div>
);

// Stories

const meta: Meta = {
  title: 'Accessibility/Components',
  parameters: {
    layout: 'centered',
    docs: {
      description: {
        component: 'A collection of accessible components following WCAG 2.1 AA guidelines.',
      },
    },
  },
  tags: ['autodocs'],
};

export default meta;

export const SkipLinkDemo: StoryObj = {
  name: 'Skip Link',
  render: () => (
    <div className="relative">
      <SkipLink />
      <p className="text-gray-400 text-sm">
        Press Tab to reveal the skip link (useful for keyboard users)
      </p>
    </div>
  ),
  parameters: {
    docs: {
      description: {
        story: 'Skip links allow keyboard users to bypass navigation and jump directly to main content.',
      },
    },
  },
};

export const VisuallyHiddenDemo: StoryObj = {
  name: 'Visually Hidden',
  render: () => (
    <div>
      <button className="p-2 bg-purple-600 rounded-lg">
        <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
          <path d="M6.3 2.841A1.5 1.5 0 004 4.11v11.78a1.5 1.5 0 002.3 1.269l9.344-5.89a1.5 1.5 0 000-2.538L6.3 2.84z" />
        </svg>
        <VisuallyHidden>Play track</VisuallyHidden>
      </button>
      <p className="mt-4 text-gray-400 text-sm">
        The text "Play track" is hidden visually but readable by screen readers
      </p>
    </div>
  ),
};

export const IconButtonDemo: StoryObj = {
  name: 'Icon Button',
  render: () => (
    <div className="flex gap-4">
      <IconButton
        icon={
          <svg className="w-5 h-5 text-white" fill="currentColor" viewBox="0 0 20 20">
            <path d="M6.3 2.841A1.5 1.5 0 004 4.11v11.78a1.5 1.5 0 002.3 1.269l9.344-5.89a1.5 1.5 0 000-2.538L6.3 2.84z" />
          </svg>
        }
        label="Play"
      />
      <IconButton
        icon={
          <svg className="w-5 h-5 text-white" fill="currentColor" viewBox="0 0 20 20">
            <path d="M5 4h3v12H5V4zm7 0h3v12h-3V4z" />
          </svg>
        }
        label="Pause"
      />
      <IconButton
        icon={
          <svg className="w-5 h-5 text-white" fill="currentColor" viewBox="0 0 20 20">
            <path d="M3.172 5.172a4 4 0 015.656 0L10 6.343l1.172-1.171a4 4 0 115.656 5.656L10 17.657l-6.828-6.829a4 4 0 010-5.656z" />
          </svg>
        }
        label="Like"
        variant="ghost"
      />
      <IconButton
        icon={
          <svg className="w-5 h-5 text-white" fill="currentColor" viewBox="0 0 20 20">
            <path d="M10 6a2 2 0 110-4 2 2 0 010 4zM10 12a2 2 0 110-4 2 2 0 010 4zM10 18a2 2 0 110-4 2 2 0 010 4z" />
          </svg>
        }
        label="More options"
        disabled
      />
    </div>
  ),
};

export const LiveRegionDemo: StoryObj = {
  name: 'Live Region',
  render: function LiveRegionStory() {
    const [message, setMessage] = useState('');

    return (
      <div className="space-y-4">
        <div className="flex gap-2">
          <button
            onClick={() => setMessage('Track added to playlist')}
            className="px-4 py-2 bg-purple-600 rounded-lg text-white"
          >
            Add to Playlist
          </button>
          <button
            onClick={() => setMessage('Download started')}
            className="px-4 py-2 bg-gray-700 rounded-lg text-white"
          >
            Download
          </button>
        </div>
        <LiveRegion message={message} priority="polite" />
      </div>
    );
  },
};

export const ProgressIndicatorDemo: StoryObj = {
  name: 'Progress Indicator',
  render: function ProgressDemo() {
    const [progress, setProgress] = useState(45);

    return (
      <div className="w-64 space-y-4">
        <ProgressIndicator value={progress} label="Upload Progress" />
        <ProgressIndicator value={75} label="Download Progress" />
        <ProgressIndicator value={100} label="Complete" showValue={false} />
        <input
          type="range"
          min="0"
          max="100"
          value={progress}
          onChange={(e) => setProgress(Number(e.target.value))}
          className="w-full"
        />
      </div>
    );
  },
};

export const LoadingStateDemo: StoryObj = {
  name: 'Loading State',
  render: () => (
    <div className="flex gap-8 items-center">
      <div className="text-center">
        <LoadingState size="sm" />
        <span className="text-xs text-gray-400 mt-2 block">Small</span>
      </div>
      <div className="text-center">
        <LoadingState size="md" />
        <span className="text-xs text-gray-400 mt-2 block">Medium</span>
      </div>
      <div className="text-center">
        <LoadingState size="lg" label="Loading tracks..." />
        <span className="text-xs text-gray-400 mt-2 block">Large</span>
      </div>
    </div>
  ),
};

export const KeyboardShortcutDemo: StoryObj = {
  name: 'Keyboard Shortcuts',
  render: () => (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <span className="text-gray-400">Play/Pause</span>
        <KeyboardShortcut keys={['Space']} />
      </div>
      <div className="flex items-center justify-between">
        <span className="text-gray-400">Previous Track</span>
        <KeyboardShortcut keys={['Ctrl', 'Left']} />
      </div>
      <div className="flex items-center justify-between">
        <span className="text-gray-400">Next Track</span>
        <KeyboardShortcut keys={['Ctrl', 'Right']} />
      </div>
      <div className="flex items-center justify-between">
        <span className="text-gray-400">Toggle Shuffle</span>
        <KeyboardShortcut keys={['S']} />
      </div>
      <div className="flex items-center justify-between">
        <span className="text-gray-400">Toggle Repeat</span>
        <KeyboardShortcut keys={['R']} />
      </div>
    </div>
  ),
};

export const TabsDemo: StoryObj = {
  name: 'Accessible Tabs',
  render: function TabsStory() {
    const [activeTab, setActiveTab] = useState('tracks');

    const tabs = [
      { id: 'tracks', label: 'Tracks', content: <p className="text-gray-300">Track listing...</p> },
      { id: 'albums', label: 'Albums', content: <p className="text-gray-300">Album collection...</p> },
      { id: 'playlists', label: 'Playlists', content: <p className="text-gray-300">User playlists...</p> },
    ];

    return (
      <div className="w-96">
        <Tabs
          tabs={tabs}
          activeTab={activeTab}
          onTabChange={setActiveTab}
          label="Library Navigation"
        />
      </div>
    );
  },
};

export const ContrastChecker: StoryObj = {
  name: 'Color Contrast Examples',
  render: () => (
    <div className="space-y-4">
      <div className="p-4 bg-gray-900 rounded-lg">
        <p className="text-white">White on Dark (21:1 ratio) - AAA</p>
      </div>
      <div className="p-4 bg-gray-800 rounded-lg">
        <p className="text-purple-400">Purple on Gray (5.3:1 ratio) - AA</p>
      </div>
      <div className="p-4 bg-purple-600 rounded-lg">
        <p className="text-white">White on Purple (4.6:1 ratio) - AA</p>
      </div>
      <div className="p-4 bg-gray-700 rounded-lg">
        <p className="text-gray-300">Light Gray on Gray (4.5:1 ratio) - AA minimum</p>
      </div>
    </div>
  ),
};

export const FocusManagement: StoryObj = {
  name: 'Focus Indicators',
  render: () => (
    <div className="space-y-4">
      <p className="text-gray-400 text-sm mb-4">
        Tab through these elements to see focus indicators
      </p>
      <div className="flex gap-4">
        <button className="px-4 py-2 bg-purple-600 rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-400 focus:ring-offset-2 focus:ring-offset-gray-900">
          Button
        </button>
        <a href="#" className="px-4 py-2 text-purple-400 underline focus:outline-none focus:ring-2 focus:ring-purple-400 rounded">
          Link
        </a>
        <input
          type="text"
          placeholder="Input"
          className="px-4 py-2 bg-gray-800 border border-gray-700 rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-400 focus:border-transparent"
        />
      </div>
    </div>
  ),
};
