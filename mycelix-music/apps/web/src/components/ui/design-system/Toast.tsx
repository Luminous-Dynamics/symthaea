// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
'use client';

import React, { createContext, useContext, useCallback, useState, useEffect } from 'react';
import { createPortal } from 'react-dom';
import { cn } from '@/lib/utils';
import { X, CheckCircle, XCircle, AlertCircle, Info, Loader2 } from 'lucide-react';

export type ToastType = 'success' | 'error' | 'warning' | 'info' | 'loading';
export type ToastPosition =
  | 'top-left'
  | 'top-center'
  | 'top-right'
  | 'bottom-left'
  | 'bottom-center'
  | 'bottom-right';

export interface Toast {
  id: string;
  type: ToastType;
  title: string;
  description?: string;
  duration?: number;
  dismissible?: boolean;
  action?: {
    label: string;
    onClick: () => void;
  };
}

interface ToastContextValue {
  toasts: Toast[];
  addToast: (toast: Omit<Toast, 'id'>) => string;
  removeToast: (id: string) => void;
  updateToast: (id: string, toast: Partial<Toast>) => void;
}

const ToastContext = createContext<ToastContextValue | null>(null);

export function useToast() {
  const context = useContext(ToastContext);
  if (!context) {
    throw new Error('useToast must be used within a ToastProvider');
  }
  return context;
}

// Toast helper functions
export function toast(options: Omit<Toast, 'id'>) {
  // This will be set by the provider
  return (window as any).__toast_add?.(options) || '';
}

toast.success = (title: string, description?: string) =>
  toast({ type: 'success', title, description });

toast.error = (title: string, description?: string) =>
  toast({ type: 'error', title, description });

toast.warning = (title: string, description?: string) =>
  toast({ type: 'warning', title, description });

toast.info = (title: string, description?: string) =>
  toast({ type: 'info', title, description });

toast.loading = (title: string, description?: string) =>
  toast({ type: 'loading', title, description, duration: 0 });

toast.promise = async <T,>(
  promise: Promise<T>,
  options: {
    loading: string;
    success: string | ((data: T) => string);
    error: string | ((err: any) => string);
  }
): Promise<T> => {
  const id = toast.loading(options.loading);
  try {
    const result = await promise;
    const successMessage = typeof options.success === 'function'
      ? options.success(result)
      : options.success;
    (window as any).__toast_update?.(id, {
      type: 'success',
      title: successMessage,
      duration: 4000
    });
    return result;
  } catch (error) {
    const errorMessage = typeof options.error === 'function'
      ? options.error(error)
      : options.error;
    (window as any).__toast_update?.(id, {
      type: 'error',
      title: errorMessage,
      duration: 4000
    });
    throw error;
  }
};

export interface ToastProviderProps {
  children: React.ReactNode;
  position?: ToastPosition;
  maxToasts?: number;
  defaultDuration?: number;
}

export function ToastProvider({
  children,
  position = 'bottom-right',
  maxToasts = 5,
  defaultDuration = 4000,
}: ToastProviderProps) {
  const [toasts, setToasts] = useState<Toast[]>([]);

  const addToast = useCallback(
    (toast: Omit<Toast, 'id'>) => {
      const id = Math.random().toString(36).slice(2);
      const newToast: Toast = {
        ...toast,
        id,
        duration: toast.duration ?? defaultDuration,
        dismissible: toast.dismissible ?? true,
      };

      setToasts((prev) => {
        const updated = [newToast, ...prev];
        return updated.slice(0, maxToasts);
      });

      return id;
    },
    [defaultDuration, maxToasts]
  );

  const removeToast = useCallback((id: string) => {
    setToasts((prev) => prev.filter((t) => t.id !== id));
  }, []);

  const updateToast = useCallback((id: string, updates: Partial<Toast>) => {
    setToasts((prev) =>
      prev.map((t) => (t.id === id ? { ...t, ...updates } : t))
    );
  }, []);

  // Expose toast functions globally
  useEffect(() => {
    (window as any).__toast_add = addToast;
    (window as any).__toast_update = updateToast;
    return () => {
      delete (window as any).__toast_add;
      delete (window as any).__toast_update;
    };
  }, [addToast, updateToast]);

  const positionClasses: Record<ToastPosition, string> = {
    'top-left': 'top-4 left-4',
    'top-center': 'top-4 left-1/2 -translate-x-1/2',
    'top-right': 'top-4 right-4',
    'bottom-left': 'bottom-4 left-4',
    'bottom-center': 'bottom-4 left-1/2 -translate-x-1/2',
    'bottom-right': 'bottom-4 right-4',
  };

  return (
    <ToastContext.Provider value={{ toasts, addToast, removeToast, updateToast }}>
      {children}
      {typeof document !== 'undefined' &&
        createPortal(
          <div
            className={cn(
              'fixed z-[100] flex flex-col gap-2 pointer-events-none',
              positionClasses[position]
            )}
          >
            {toasts.map((toast) => (
              <ToastItem
                key={toast.id}
                toast={toast}
                onDismiss={() => removeToast(toast.id)}
              />
            ))}
          </div>,
          document.body
        )}
    </ToastContext.Provider>
  );
}

interface ToastItemProps {
  toast: Toast;
  onDismiss: () => void;
}

function ToastItem({ toast, onDismiss }: ToastItemProps) {
  const [isExiting, setIsExiting] = useState(false);

  useEffect(() => {
    if (toast.duration && toast.duration > 0) {
      const timer = setTimeout(() => {
        setIsExiting(true);
        setTimeout(onDismiss, 200);
      }, toast.duration);
      return () => clearTimeout(timer);
    }
  }, [toast.duration, onDismiss]);

  const handleDismiss = () => {
    setIsExiting(true);
    setTimeout(onDismiss, 200);
  };

  const icons: Record<ToastType, React.ReactNode> = {
    success: <CheckCircle className="h-5 w-5 text-green-500" />,
    error: <XCircle className="h-5 w-5 text-red-500" />,
    warning: <AlertCircle className="h-5 w-5 text-yellow-500" />,
    info: <Info className="h-5 w-5 text-blue-500" />,
    loading: <Loader2 className="h-5 w-5 text-primary-500 animate-spin" />,
  };

  const bgClasses: Record<ToastType, string> = {
    success: 'bg-green-50 dark:bg-green-900/20 border-green-200 dark:border-green-800',
    error: 'bg-red-50 dark:bg-red-900/20 border-red-200 dark:border-red-800',
    warning: 'bg-yellow-50 dark:bg-yellow-900/20 border-yellow-200 dark:border-yellow-800',
    info: 'bg-blue-50 dark:bg-blue-900/20 border-blue-200 dark:border-blue-800',
    loading: 'bg-gray-50 dark:bg-gray-800 border-gray-200 dark:border-gray-700',
  };

  return (
    <div
      className={cn(
        'pointer-events-auto w-80 rounded-lg border shadow-lg transition-all duration-200',
        bgClasses[toast.type],
        isExiting
          ? 'opacity-0 translate-x-4'
          : 'opacity-100 translate-x-0 animate-in slide-in-from-right-4'
      )}
      role="alert"
    >
      <div className="flex items-start gap-3 p-4">
        <div className="flex-shrink-0">{icons[toast.type]}</div>
        <div className="flex-1 min-w-0">
          <p className="text-sm font-medium text-gray-900 dark:text-white">
            {toast.title}
          </p>
          {toast.description && (
            <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">
              {toast.description}
            </p>
          )}
          {toast.action && (
            <button
              onClick={toast.action.onClick}
              className="mt-2 text-sm font-medium text-primary-600 hover:text-primary-500 dark:text-primary-400"
            >
              {toast.action.label}
            </button>
          )}
        </div>
        {toast.dismissible && toast.type !== 'loading' && (
          <button
            onClick={handleDismiss}
            className="flex-shrink-0 text-gray-400 hover:text-gray-600 dark:hover:text-gray-300"
          >
            <X className="h-4 w-4" />
          </button>
        )}
      </div>
    </div>
  );
}

// Sonner-style toast animations
export function Toaster({ position = 'bottom-right' }: { position?: ToastPosition }) {
  return <ToastProvider position={position}>{null}</ToastProvider>;
}
