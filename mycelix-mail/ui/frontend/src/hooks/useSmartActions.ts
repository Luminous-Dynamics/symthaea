import { useCallback } from 'react';
import type { Email } from '@/types';

export interface UseSmartActionsProps {
  emails: Email[];
  currentEmailId: string | null;
  onSelectEmail: (emailId: string | null) => void;
}

export interface UseSmartActionsReturn {
  getNextEmail: (options?: { unreadOnly?: boolean }) => Email | null;
  getPreviousEmail: (options?: { unreadOnly?: boolean }) => Email | null;
  selectNextEmail: (options?: { unreadOnly?: boolean }) => void;
  selectPreviousEmail: (options?: { unreadOnly?: boolean }) => void;
}

/**
 * Hook for smart email navigation patterns
 * Provides "and next" functionality for Archive/Delete/etc actions
 */
export const useSmartActions = ({
  emails,
  currentEmailId,
  onSelectEmail,
}: UseSmartActionsProps): UseSmartActionsReturn => {

  /**
   * Get the next email in the list
   */
  const getNextEmail = useCallback(
    (options: { unreadOnly?: boolean } = {}) => {
      if (!currentEmailId || emails.length === 0) return null;

      const currentIndex = emails.findIndex((e) => e.id === currentEmailId);
      if (currentIndex === -1) return null;

      // Search forward from current position
      for (let i = currentIndex + 1; i < emails.length; i++) {
        const email = emails[i];
        if (!options.unreadOnly || !email.isRead) {
          return email;
        }
      }

      // If unreadOnly and no unread found forward, try from beginning
      if (options.unreadOnly) {
        for (let i = 0; i < currentIndex; i++) {
          const email = emails[i];
          if (!email.isRead) {
            return email;
          }
        }
      }

      return null;
    },
    [emails, currentEmailId]
  );

  /**
   * Get the previous email in the list
   */
  const getPreviousEmail = useCallback(
    (options: { unreadOnly?: boolean } = {}) => {
      if (!currentEmailId || emails.length === 0) return null;

      const currentIndex = emails.findIndex((e) => e.id === currentEmailId);
      if (currentIndex === -1) return null;

      // Search backward from current position
      for (let i = currentIndex - 1; i >= 0; i--) {
        const email = emails[i];
        if (!options.unreadOnly || !email.isRead) {
          return email;
        }
      }

      // If unreadOnly and no unread found backward, try from end
      if (options.unreadOnly) {
        for (let i = emails.length - 1; i > currentIndex; i--) {
          const email = emails[i];
          if (!email.isRead) {
            return email;
          }
        }
      }

      return null;
    },
    [emails, currentEmailId]
  );

  /**
   * Select the next email
   */
  const selectNextEmail = useCallback(
    (options: { unreadOnly?: boolean } = {}) => {
      const nextEmail = getNextEmail(options);
      if (nextEmail) {
        onSelectEmail(nextEmail.id);
      } else {
        // No next email, close the email view
        onSelectEmail(null);
      }
    },
    [getNextEmail, onSelectEmail]
  );

  /**
   * Select the previous email
   */
  const selectPreviousEmail = useCallback(
    (options: { unreadOnly?: boolean } = {}) => {
      const prevEmail = getPreviousEmail(options);
      if (prevEmail) {
        onSelectEmail(prevEmail.id);
      } else {
        // No previous email, stay on current or close
        onSelectEmail(null);
      }
    },
    [getPreviousEmail, onSelectEmail]
  );

  return {
    getNextEmail,
    getPreviousEmail,
    selectNextEmail,
    selectPreviousEmail,
  };
};
