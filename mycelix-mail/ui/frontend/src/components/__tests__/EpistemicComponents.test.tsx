// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Epistemic Components Tests
 *
 * Comprehensive tests for all epistemic UI components
 */

import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';

// Test utilities
const createTestQueryClient = () =>
  new QueryClient({
    defaultOptions: {
      queries: { retry: false },
      mutations: { retry: false },
    },
  });

const TestWrapper = ({ children }: { children: React.ReactNode }) => (
  <QueryClientProvider client={createTestQueryClient()}>
    {children}
  </QueryClientProvider>
);

// ============================================
// Accessibility Utils Tests
// ============================================

import {
  announce,
  formatTrustScoreForSR,
  formatTierForSR,
  formatAssuranceLevelForSR,
  getFocusableElements,
  focusFirstElement,
} from '@/utils/accessibility';

describe('Accessibility Utils', () => {
  describe('formatTrustScoreForSR', () => {
    it('should format high trust scores', () => {
      expect(formatTrustScoreForSR(0.85)).toBe('High trust, 85 percent');
      expect(formatTrustScoreForSR(0.70)).toBe('High trust, 70 percent');
      expect(formatTrustScoreForSR(1.0)).toBe('High trust, 100 percent');
    });

    it('should format medium trust scores', () => {
      expect(formatTrustScoreForSR(0.55)).toBe('Medium trust, 55 percent');
      expect(formatTrustScoreForSR(0.40)).toBe('Medium trust, 40 percent');
    });

    it('should format low trust scores', () => {
      expect(formatTrustScoreForSR(0.20)).toBe('Low trust, 20 percent');
      expect(formatTrustScoreForSR(0.0)).toBe('Low trust, 0 percent');
    });
  });

  describe('formatTierForSR', () => {
    it('should format all tiers correctly', () => {
      expect(formatTierForSR(0)).toBe('Tier 0, unverifiable sender');
      expect(formatTierForSR(1)).toBe('Tier 1, sender email verified');
      expect(formatTierForSR(2)).toBe('Tier 2, sender identity verified');
      expect(formatTierForSR(3)).toBe('Tier 3, sender in trust network');
      expect(formatTierForSR(4)).toBe('Tier 4, fully attested sender');
    });

    it('should handle unknown tiers', () => {
      expect(formatTierForSR(5)).toBe('Tier 5');
      expect(formatTierForSR(99)).toBe('Tier 99');
    });
  });

  describe('formatAssuranceLevelForSR', () => {
    it('should format all assurance levels', () => {
      expect(formatAssuranceLevelForSR('e0_anonymous')).toContain('E0');
      expect(formatAssuranceLevelForSR('e1_verified_email')).toContain('E1');
      expect(formatAssuranceLevelForSR('e2_gitcoin_passport')).toContain('E2');
      expect(formatAssuranceLevelForSR('e3_multi_factor')).toContain('E3');
      expect(formatAssuranceLevelForSR('e4_constitutional')).toContain('E4');
    });

    it('should return original for unknown levels', () => {
      expect(formatAssuranceLevelForSR('unknown_level')).toBe('unknown_level');
    });
  });

  describe('getFocusableElements', () => {
    it('should find all focusable elements', () => {
      const container = document.createElement('div');
      container.innerHTML = `
        <button>Button</button>
        <a href="#">Link</a>
        <input type="text" />
        <select><option>Option</option></select>
        <textarea></textarea>
        <div tabindex="0">Focusable div</div>
        <div tabindex="-1">Not focusable</div>
        <button disabled>Disabled button</button>
      `;
      document.body.appendChild(container);

      const focusable = getFocusableElements(container);

      expect(focusable.length).toBe(6); // Excludes disabled and tabindex=-1
      expect(focusable[0].tagName).toBe('BUTTON');
      expect(focusable[1].tagName).toBe('A');

      document.body.removeChild(container);
    });
  });

  describe('focusFirstElement', () => {
    it('should focus the first focusable element', () => {
      const container = document.createElement('div');
      container.innerHTML = `
        <button id="first">First</button>
        <button id="second">Second</button>
      `;
      document.body.appendChild(container);

      const result = focusFirstElement(container);

      expect(result).toBe(true);
      expect(document.activeElement?.id).toBe('first');

      document.body.removeChild(container);
    });

    it('should return false when no focusable elements', () => {
      const container = document.createElement('div');
      container.innerHTML = '<span>No focusable</span>';
      document.body.appendChild(container);

      const result = focusFirstElement(container);

      expect(result).toBe(false);

      document.body.removeChild(container);
    });
  });

  describe('announce', () => {
    it('should create live region and announce message', () => {
      announce('Test announcement', 'polite');

      const liveRegion = document.querySelector('[aria-live]');
      expect(liveRegion).toBeInTheDocument();
      expect(liveRegion).toHaveAttribute('aria-live', 'polite');
    });
  });
});

// ============================================
// Demo Data Tests
// ============================================

import {
  generateContact,
  generateContacts,
  generateEmail,
  generateEmails,
  generateClaim,
  generateTrustEdge,
  generateTrustNetwork,
  generateAttestation,
} from '@/utils/demoData';

describe('Demo Data Generators', () => {
  describe('generateContact', () => {
    it('should generate valid contact', () => {
      const contact = generateContact();

      expect(contact.did).toMatch(/^did:mycelix:/);
      expect(contact.email).toContain('@');
      expect(contact.name).toBeTruthy();
      expect(contact.trustScore).toBeGreaterThanOrEqual(0);
      expect(contact.trustScore).toBeLessThanOrEqual(1);
      expect(contact.pathLength).toBeGreaterThanOrEqual(1);
      expect(contact.pathLength).toBeLessThanOrEqual(4);
    });

    it('should respect override options', () => {
      const contact = generateContact({
        name: 'Custom Name',
        trustScore: 0.99,
      });

      expect(contact.name).toBe('Custom Name');
      expect(contact.trustScore).toBe(0.99);
    });
  });

  describe('generateContacts', () => {
    it('should generate specified number of contacts', () => {
      const contacts = generateContacts(5);

      expect(contacts.length).toBe(5);
      contacts.forEach((c) => {
        expect(c.did).toBeDefined();
        expect(c.email).toBeDefined();
      });
    });
  });

  describe('generateEmail', () => {
    it('should generate valid email with epistemic metadata', () => {
      const contacts = generateContacts(3);
      const email = generateEmail(contacts);

      expect(email.id).toBeTruthy();
      expect(email.subject).toBeTruthy();
      expect(email.from.address).toBeTruthy();
      expect(email.to.length).toBeGreaterThan(0);
      expect(email.senderDid).toMatch(/^did:mycelix:/);
      expect(email.epistemicTier).toBeGreaterThanOrEqual(0);
      expect(email.epistemicTier).toBeLessThanOrEqual(4);
      expect(email.aiInsights).toBeDefined();
      expect(email.aiInsights.intent).toBeTruthy();
    });
  });

  describe('generateClaim', () => {
    it('should generate valid verifiable claim', () => {
      const claim = generateClaim();

      expect(claim.id).toBeTruthy();
      expect(claim.type).toBeTruthy();
      expect(claim.issuer).toMatch(/^did:mycelix:/);
      expect(claim.subject).toMatch(/^did:mycelix:/);
      expect(claim.proof).toBeDefined();
      expect(claim.assurance_level).toBeTruthy();
    });
  });

  describe('generateTrustNetwork', () => {
    it('should generate valid trust network', () => {
      const network = generateTrustNetwork('did:mycelix:user', 10);

      expect(network.userDid).toBe('did:mycelix:user');
      expect(network.contacts.length).toBe(10);
      expect(network.edges.length).toBeGreaterThan(0);
      expect(network.stats.totalConnections).toBe(10);
      expect(network.stats.avgTrustScore).toBeGreaterThan(0);
    });
  });

  describe('generateAttestation', () => {
    it('should generate valid attestation', () => {
      const contacts = generateContacts(3);
      const attestation = generateAttestation(contacts);

      expect(attestation.id).toBeTruthy();
      expect(attestation.attestorDid).toBeTruthy();
      expect(attestation.subjectDid).toBeTruthy();
      expect(attestation.trustScore).toBeGreaterThan(0);
      expect(attestation.createdAt).toBeTruthy();
    });
  });
});

// ============================================
// Keyboard Shortcuts Tests
// ============================================

import KeyboardShortcuts, { useKeyboardShortcuts } from '@/components/help/KeyboardShortcuts';
import { renderHook, act } from '@testing-library/react';

describe('KeyboardShortcuts', () => {
  it('should render when isOpen is true', () => {
    const onClose = vi.fn();

    render(<KeyboardShortcuts isOpen={true} onClose={onClose} />);

    expect(screen.getByText('Keyboard Shortcuts')).toBeInTheDocument();
    expect(screen.getByText('Navigation')).toBeInTheDocument();
    expect(screen.getByText('Email Actions')).toBeInTheDocument();
  });

  it('should not render when isOpen is false', () => {
    const onClose = vi.fn();

    render(<KeyboardShortcuts isOpen={false} onClose={onClose} />);

    expect(screen.queryByText('Keyboard Shortcuts')).not.toBeInTheDocument();
  });

  it('should call onClose when clicking backdrop', async () => {
    const user = userEvent.setup();
    const onClose = vi.fn();

    render(<KeyboardShortcuts isOpen={true} onClose={onClose} />);

    const backdrop = document.querySelector('.fixed.inset-0.bg-black\\/50');
    if (backdrop) {
      await user.click(backdrop);
    }

    expect(onClose).toHaveBeenCalled();
  });

  it('should filter shortcuts by search', async () => {
    const user = userEvent.setup();
    const onClose = vi.fn();

    render(<KeyboardShortcuts isOpen={true} onClose={onClose} />);

    const searchInput = screen.getByPlaceholderText('Search shortcuts...');
    await user.type(searchInput, 'compose');

    expect(screen.getByText('Compose new email')).toBeInTheDocument();
    expect(screen.queryByText('Delete email')).not.toBeInTheDocument();
  });

  it('should show no results message for invalid search', async () => {
    const user = userEvent.setup();
    const onClose = vi.fn();

    render(<KeyboardShortcuts isOpen={true} onClose={onClose} />);

    const searchInput = screen.getByPlaceholderText('Search shortcuts...');
    await user.type(searchInput, 'xyznonexistent');

    expect(screen.getByText(/No shortcuts found/)).toBeInTheDocument();
  });
});

describe('useKeyboardShortcuts hook', () => {
  it('should start closed', () => {
    const { result } = renderHook(() => useKeyboardShortcuts());

    expect(result.current.isOpen).toBe(false);
  });

  it('should open with open()', () => {
    const { result } = renderHook(() => useKeyboardShortcuts());

    act(() => {
      result.current.open();
    });

    expect(result.current.isOpen).toBe(true);
  });

  it('should close with close()', () => {
    const { result } = renderHook(() => useKeyboardShortcuts());

    act(() => {
      result.current.open();
    });
    act(() => {
      result.current.close();
    });

    expect(result.current.isOpen).toBe(false);
  });

  it('should toggle with toggle()', () => {
    const { result } = renderHook(() => useKeyboardShortcuts());

    act(() => {
      result.current.toggle();
    });
    expect(result.current.isOpen).toBe(true);

    act(() => {
      result.current.toggle();
    });
    expect(result.current.isOpen).toBe(false);
  });
});

// ============================================
// Settings Store Tests
// ============================================

import { useEpistemicSettings } from '@/components/settings/EpistemicSettings';

describe('useEpistemicSettings store', () => {
  beforeEach(() => {
    // Reset store state
    useEpistemicSettings.setState({
      trust: {
        enabled: true,
        showTierBadges: true,
        showTrustScores: true,
        autoVerifyClaims: true,
        trustDecayEnabled: true,
        trustDecayDays: 90,
        minTrustThreshold: 0.3,
        maxPathLength: 3,
      },
      quarantine: {
        enabled: true,
        autoQuarantineUnknown: false,
        quarantineThreshold: 0.2,
        notifyOnQuarantine: true,
      },
      ai: {
        enabled: true,
        intentDetection: true,
        priorityScoring: true,
        replySuggestions: true,
        threadSummarization: true,
        modelPreference: 'local',
      },
      notifications: {
        attestationReceived: true,
        trustScoreChange: true,
        introductionRequest: true,
        credentialExpiring: true,
        quarantineAlert: true,
      },
      privacy: {
        shareAttestations: true,
        allowIntroductions: true,
        showOnlineStatus: false,
        analyticsEnabled: false,
      },
      display: {
        compactMode: false,
        showAvatars: true,
        animationsEnabled: true,
        highContrast: false,
      },
    });
  });

  it('should have default trust settings', () => {
    const state = useEpistemicSettings.getState();

    expect(state.trust.enabled).toBe(true);
    expect(state.trust.maxPathLength).toBe(3);
    expect(state.trust.minTrustThreshold).toBe(0.3);
  });

  it('should update trust settings', () => {
    const { updateTrust } = useEpistemicSettings.getState();

    updateTrust({ maxPathLength: 5, minTrustThreshold: 0.5 });

    const state = useEpistemicSettings.getState();
    expect(state.trust.maxPathLength).toBe(5);
    expect(state.trust.minTrustThreshold).toBe(0.5);
  });

  it('should update AI settings', () => {
    const { updateAI } = useEpistemicSettings.getState();

    updateAI({ modelPreference: 'cloud', intentDetection: false });

    const state = useEpistemicSettings.getState();
    expect(state.ai.modelPreference).toBe('cloud');
    expect(state.ai.intentDetection).toBe(false);
  });

  it('should update privacy settings', () => {
    const { updatePrivacy } = useEpistemicSettings.getState();

    updatePrivacy({ analyticsEnabled: true });

    const state = useEpistemicSettings.getState();
    expect(state.privacy.analyticsEnabled).toBe(true);
  });

  it('should reset to defaults', () => {
    const { updateTrust, resetToDefaults } = useEpistemicSettings.getState();

    updateTrust({ maxPathLength: 10 });
    expect(useEpistemicSettings.getState().trust.maxPathLength).toBe(10);

    resetToDefaults();
    expect(useEpistemicSettings.getState().trust.maxPathLength).toBe(3);
  });
});

// ============================================
// Integration Tests
// ============================================

describe('Component Integration', () => {
  it('should render epistemic components without crashing', () => {
    // Smoke test - just ensure components can be imported and referenced
    expect(typeof formatTrustScoreForSR).toBe('function');
    expect(typeof generateContact).toBe('function');
    expect(typeof useEpistemicSettings).toBe('function');
  });
});
