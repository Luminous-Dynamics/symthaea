// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Secure State Hooks
 * React hooks for secure state management
 */

'use client';

import {
  useState,
  useCallback,
  useEffect,
  useRef,
  createContext,
  useContext,
  ReactNode,
} from 'react';

// Encryption utilities (using Web Crypto API)
async function encrypt(data: string, key: CryptoKey): Promise<string> {
  const encoder = new TextEncoder();
  const iv = crypto.getRandomValues(new Uint8Array(12));
  const encrypted = await crypto.subtle.encrypt(
    { name: 'AES-GCM', iv },
    key,
    encoder.encode(data)
  );

  const combined = new Uint8Array(iv.length + encrypted.byteLength);
  combined.set(iv);
  combined.set(new Uint8Array(encrypted), iv.length);

  return btoa(String.fromCharCode(...combined));
}

async function decrypt(data: string, key: CryptoKey): Promise<string> {
  const combined = Uint8Array.from(atob(data), c => c.charCodeAt(0));
  const iv = combined.slice(0, 12);
  const encrypted = combined.slice(12);

  const decrypted = await crypto.subtle.decrypt(
    { name: 'AES-GCM', iv },
    key,
    encrypted
  );

  return new TextDecoder().decode(decrypted);
}

async function generateKey(): Promise<CryptoKey> {
  return crypto.subtle.generateKey(
    { name: 'AES-GCM', length: 256 },
    false,
    ['encrypt', 'decrypt']
  );
}

// Security context
interface SecurityContextValue {
  isSecure: boolean;
  csrfToken: string | null;
  refreshCsrfToken: () => Promise<void>;
  validateInput: (input: string, type: InputType) => ValidationResult;
}

type InputType = 'text' | 'email' | 'url' | 'ethereum_address' | 'username';

interface ValidationResult {
  valid: boolean;
  sanitized: string;
  errors: string[];
}

const SecurityContext = createContext<SecurityContextValue | null>(null);

export function SecurityProvider({ children }: { children: ReactNode }) {
  const [csrfToken, setCsrfToken] = useState<string | null>(null);
  const [isSecure] = useState(() => {
    if (typeof window === 'undefined') return true;
    return window.location.protocol === 'https:' || window.location.hostname === 'localhost';
  });

  const refreshCsrfToken = useCallback(async () => {
    try {
      const response = await fetch('/api/csrf-token', {
        credentials: 'include',
      });
      const data = await response.json();
      setCsrfToken(data.csrfToken);
    } catch (error) {
      console.error('Failed to refresh CSRF token:', error);
    }
  }, []);

  const validateInput = useCallback((input: string, type: InputType): ValidationResult => {
    const errors: string[] = [];
    let sanitized = input.trim();

    // Remove null bytes and control characters
    sanitized = sanitized.replace(/[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]/g, '');

    switch (type) {
      case 'email': {
        const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
        if (!emailRegex.test(sanitized)) {
          errors.push('Invalid email format');
        }
        sanitized = sanitized.toLowerCase();
        break;
      }
      case 'url': {
        try {
          const url = new URL(sanitized);
          if (!['http:', 'https:'].includes(url.protocol)) {
            errors.push('Only HTTP(S) URLs allowed');
          }
          sanitized = url.href;
        } catch {
          errors.push('Invalid URL format');
        }
        break;
      }
      case 'ethereum_address': {
        if (!/^0x[a-fA-F0-9]{40}$/.test(sanitized)) {
          errors.push('Invalid Ethereum address');
        }
        sanitized = sanitized.toLowerCase();
        break;
      }
      case 'username': {
        if (!/^[a-zA-Z0-9_-]{3,30}$/.test(sanitized)) {
          errors.push('Username must be 3-30 characters, alphanumeric with _ or -');
        }
        sanitized = sanitized.toLowerCase();
        break;
      }
      case 'text':
      default: {
        // Strip HTML tags
        sanitized = sanitized.replace(/<[^>]*>/g, '');
        // Escape special characters
        sanitized = sanitized
          .replace(/&/g, '&amp;')
          .replace(/</g, '&lt;')
          .replace(/>/g, '&gt;')
          .replace(/"/g, '&quot;')
          .replace(/'/g, '&#x27;');
        break;
      }
    }

    return {
      valid: errors.length === 0,
      sanitized,
      errors,
    };
  }, []);

  useEffect(() => {
    refreshCsrfToken();
  }, [refreshCsrfToken]);

  return (
    <SecurityContext.Provider value={{ isSecure, csrfToken, refreshCsrfToken, validateInput }}>
      {children}
    </SecurityContext.Provider>
  );
}

export function useSecurity() {
  const context = useContext(SecurityContext);
  if (!context) {
    throw new Error('useSecurity must be used within SecurityProvider');
  }
  return context;
}

/**
 * Secure state hook - encrypts sensitive data in memory
 */
export function useSecureState<T>(initialValue: T): [T, (value: T) => void, () => void] {
  const [state, setState] = useState<T>(initialValue);
  const keyRef = useRef<CryptoKey | null>(null);
  const encryptedRef = useRef<string | null>(null);

  // Initialize encryption key
  useEffect(() => {
    generateKey().then((key) => {
      keyRef.current = key;
    });

    // Clear on unmount
    return () => {
      keyRef.current = null;
      encryptedRef.current = null;
    };
  }, []);

  const setSecureState = useCallback((value: T) => {
    setState(value);

    // Encrypt in background
    if (keyRef.current && value !== null && value !== undefined) {
      encrypt(JSON.stringify(value), keyRef.current)
        .then((encrypted) => {
          encryptedRef.current = encrypted;
        })
        .catch(console.error);
    }
  }, []);

  const clearState = useCallback(() => {
    setState(initialValue);
    encryptedRef.current = null;
  }, [initialValue]);

  return [state, setSecureState, clearState];
}

/**
 * Secure session storage hook
 */
export function useSecureSessionStorage<T>(
  key: string,
  initialValue: T
): [T, (value: T) => void, () => void] {
  const [storedValue, setStoredValue] = useState<T>(() => {
    if (typeof window === 'undefined') return initialValue;

    try {
      const item = sessionStorage.getItem(key);
      return item ? JSON.parse(item) : initialValue;
    } catch {
      return initialValue;
    }
  });

  const setValue = useCallback(
    (value: T) => {
      setStoredValue(value);
      if (typeof window !== 'undefined') {
        try {
          sessionStorage.setItem(key, JSON.stringify(value));
        } catch (error) {
          console.error('Failed to store in sessionStorage:', error);
        }
      }
    },
    [key]
  );

  const removeValue = useCallback(() => {
    setStoredValue(initialValue);
    if (typeof window !== 'undefined') {
      sessionStorage.removeItem(key);
    }
  }, [key, initialValue]);

  // Clear on window close
  useEffect(() => {
    const handleBeforeUnload = () => {
      sessionStorage.removeItem(key);
    };

    window.addEventListener('beforeunload', handleBeforeUnload);
    return () => window.removeEventListener('beforeunload', handleBeforeUnload);
  }, [key]);

  return [storedValue, setValue, removeValue];
}

/**
 * Protected form state with CSRF
 */
export function useProtectedForm<T extends Record<string, unknown>>(initialState: T) {
  const [formData, setFormData] = useState<T>(initialState);
  const [errors, setErrors] = useState<Partial<Record<keyof T, string>>>({});
  const [isSubmitting, setIsSubmitting] = useState(false);
  const { csrfToken, validateInput } = useSecurity();

  const updateField = useCallback(
    <K extends keyof T>(field: K, value: T[K], inputType: InputType = 'text') => {
      const validation = validateInput(String(value), inputType);

      if (!validation.valid) {
        setErrors((prev) => ({ ...prev, [field]: validation.errors[0] }));
      } else {
        setErrors((prev) => {
          const { [field]: _, ...rest } = prev;
          return rest as Partial<Record<keyof T, string>>;
        });
      }

      setFormData((prev) => ({ ...prev, [field]: validation.sanitized as T[K] }));
    },
    [validateInput]
  );

  const submit = useCallback(
    async (
      url: string,
      options: RequestInit = {}
    ): Promise<{ success: boolean; data?: unknown; error?: string }> => {
      if (!csrfToken) {
        return { success: false, error: 'CSRF token not available' };
      }

      if (Object.keys(errors).length > 0) {
        return { success: false, error: 'Form has validation errors' };
      }

      setIsSubmitting(true);

      try {
        const response = await fetch(url, {
          method: 'POST',
          ...options,
          headers: {
            'Content-Type': 'application/json',
            'X-CSRF-Token': csrfToken,
            ...options.headers,
          },
          credentials: 'include',
          body: JSON.stringify(formData),
        });

        const data = await response.json();

        if (!response.ok) {
          return { success: false, error: data.message || 'Request failed' };
        }

        return { success: true, data };
      } catch (error) {
        return { success: false, error: (error as Error).message };
      } finally {
        setIsSubmitting(false);
      }
    },
    [csrfToken, errors, formData]
  );

  const reset = useCallback(() => {
    setFormData(initialState);
    setErrors({});
  }, [initialState]);

  return {
    formData,
    errors,
    isSubmitting,
    updateField,
    submit,
    reset,
    isValid: Object.keys(errors).length === 0,
  };
}

/**
 * Rate limited callback hook
 */
export function useRateLimitedCallback<T extends (...args: any[]) => any>(
  callback: T,
  maxCalls: number,
  windowMs: number
): [T, { remaining: number; resetIn: number }] {
  const callsRef = useRef<number[]>([]);
  const [remaining, setRemaining] = useState(maxCalls);
  const [resetIn, setResetIn] = useState(0);

  const rateLimitedCallback = useCallback(
    ((...args: Parameters<T>) => {
      const now = Date.now();
      const windowStart = now - windowMs;

      // Clean old calls
      callsRef.current = callsRef.current.filter((time) => time > windowStart);

      if (callsRef.current.length >= maxCalls) {
        const oldestCall = callsRef.current[0];
        setResetIn(oldestCall + windowMs - now);
        return undefined;
      }

      callsRef.current.push(now);
      setRemaining(maxCalls - callsRef.current.length);

      return callback(...args);
    }) as T,
    [callback, maxCalls, windowMs]
  );

  return [rateLimitedCallback, { remaining, resetIn }];
}

/**
 * Inactivity timeout hook
 */
export function useInactivityTimeout(
  timeout: number,
  onTimeout: () => void
): { resetTimer: () => void; remainingTime: number } {
  const [remainingTime, setRemainingTime] = useState(timeout);
  const timeoutRef = useRef<NodeJS.Timeout>();
  const intervalRef = useRef<NodeJS.Timeout>();

  const resetTimer = useCallback(() => {
    if (timeoutRef.current) clearTimeout(timeoutRef.current);
    if (intervalRef.current) clearInterval(intervalRef.current);

    setRemainingTime(timeout);

    intervalRef.current = setInterval(() => {
      setRemainingTime((prev) => Math.max(0, prev - 1000));
    }, 1000);

    timeoutRef.current = setTimeout(() => {
      if (intervalRef.current) clearInterval(intervalRef.current);
      onTimeout();
    }, timeout);
  }, [timeout, onTimeout]);

  useEffect(() => {
    const events = ['mousedown', 'mousemove', 'keypress', 'scroll', 'touchstart'];

    events.forEach((event) => {
      document.addEventListener(event, resetTimer, { passive: true });
    });

    resetTimer();

    return () => {
      events.forEach((event) => {
        document.removeEventListener(event, resetTimer);
      });
      if (timeoutRef.current) clearTimeout(timeoutRef.current);
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, [resetTimer]);

  return { resetTimer, remainingTime };
}

export default {
  SecurityProvider,
  useSecurity,
  useSecureState,
  useSecureSessionStorage,
  useProtectedForm,
  useRateLimitedCallback,
  useInactivityTimeout,
};
