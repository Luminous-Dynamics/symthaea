import { useState, useEffect, useCallback, useRef } from 'react';
import type { Email } from '@/types';

export interface UseEmailNavigationProps {
  emails: Email[];
  onSelectEmail: (emailId: string) => void;
  selectedEmailId: string | null;
  enabled?: boolean;
}

export interface UseEmailNavigationReturn {
  focusedIndex: number;
  setFocusedIndex: (index: number) => void;
  handleNext: () => void;
  handlePrevious: () => void;
  handleOpen: () => void;
  handleToggleSelect: () => void;
  scrollToIndex: (index: number) => void;
}

/**
 * Hook for Gmail-style keyboard navigation in email lists
 * Provides j/k navigation, enter to open, x to select/deselect
 */
export const useEmailNavigation = ({
  emails,
  onSelectEmail,
  selectedEmailId,
  enabled = true,
}: UseEmailNavigationProps): UseEmailNavigationReturn => {
  // Focused index is the email that has the visual highlight (separate from checkbox selection)
  const [focusedIndex, setFocusedIndex] = useState(0);
  const emailListRef = useRef<HTMLDivElement | null>(null);

  // Keep focused index in sync with selected email
  useEffect(() => {
    if (selectedEmailId) {
      const index = emails.findIndex((e) => e.id === selectedEmailId);
      if (index !== -1) {
        setFocusedIndex(index);
      }
    }
  }, [selectedEmailId, emails]);

  // Navigate to next email (j key)
  const handleNext = useCallback(() => {
    if (!enabled || emails.length === 0) return;

    setFocusedIndex((prev) => {
      const next = Math.min(prev + 1, emails.length - 1);
      scrollToIndex(next);
      return next;
    });
  }, [enabled, emails.length]);

  // Navigate to previous email (k key)
  const handlePrevious = useCallback(() => {
    if (!enabled || emails.length === 0) return;

    setFocusedIndex((prev) => {
      const next = Math.max(prev - 1, 0);
      scrollToIndex(next);
      return next;
    });
  }, [enabled, emails.length]);

  // Open currently focused email (o or Enter key)
  const handleOpen = useCallback(() => {
    if (!enabled || emails.length === 0 || focusedIndex < 0 || focusedIndex >= emails.length) return;

    const email = emails[focusedIndex];
    if (email) {
      onSelectEmail(email.id);
    }
  }, [enabled, emails, focusedIndex, onSelectEmail]);

  // Toggle checkbox selection for focused email (x key)
  const handleToggleSelect = useCallback(() => {
    if (!enabled || emails.length === 0 || focusedIndex < 0 || focusedIndex >= emails.length) return;

    const email = emails[focusedIndex];
    if (email) {
      // This will be handled by the parent component
      // Just return the email ID for now
      return email.id;
    }
  }, [enabled, emails, focusedIndex]);

  // Scroll email at index into view
  const scrollToIndex = useCallback((index: number) => {
    // Use setTimeout to ensure the element is rendered before scrolling
    setTimeout(() => {
      const element = document.querySelector(`[data-email-index="${index}"]`);
      if (element) {
        element.scrollIntoView({
          behavior: 'smooth',
          block: 'nearest',
        });
      }
    }, 0);
  }, []);

  // Reset focused index when emails change significantly
  useEffect(() => {
    if (focusedIndex >= emails.length) {
      setFocusedIndex(Math.max(0, emails.length - 1));
    }
  }, [emails.length, focusedIndex]);

  return {
    focusedIndex,
    setFocusedIndex,
    handleNext,
    handlePrevious,
    handleOpen,
    handleToggleSelect,
    scrollToIndex,
  };
};
