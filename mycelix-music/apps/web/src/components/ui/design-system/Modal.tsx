// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
'use client';

import React, { useEffect, useCallback, useRef } from 'react';
import { createPortal } from 'react-dom';
import { cva, type VariantProps } from 'class-variance-authority';
import { cn } from '@/lib/utils';
import { X } from 'lucide-react';
import { Button } from './Button';

const overlayVariants = cva(
  'fixed inset-0 z-50 transition-opacity duration-300',
  {
    variants: {
      blur: {
        none: 'bg-black/50',
        sm: 'bg-black/40 backdrop-blur-sm',
        md: 'bg-black/30 backdrop-blur-md',
        lg: 'bg-black/20 backdrop-blur-lg',
      },
    },
    defaultVariants: {
      blur: 'sm',
    },
  }
);

const modalVariants = cva(
  'relative bg-white dark:bg-gray-900 shadow-2xl transition-all duration-300 max-h-[90vh] overflow-hidden flex flex-col',
  {
    variants: {
      size: {
        xs: 'w-full max-w-xs',
        sm: 'w-full max-w-sm',
        md: 'w-full max-w-md',
        lg: 'w-full max-w-lg',
        xl: 'w-full max-w-xl',
        '2xl': 'w-full max-w-2xl',
        '3xl': 'w-full max-w-3xl',
        '4xl': 'w-full max-w-4xl',
        full: 'w-full max-w-[95vw] h-[90vh]',
      },
      rounded: {
        none: 'rounded-none',
        sm: 'rounded-lg',
        md: 'rounded-xl',
        lg: 'rounded-2xl',
        full: 'rounded-3xl',
      },
      position: {
        center: '',
        top: '',
        bottom: '',
      },
    },
    defaultVariants: {
      size: 'md',
      rounded: 'md',
      position: 'center',
    },
  }
);

export interface ModalProps extends VariantProps<typeof modalVariants> {
  isOpen: boolean;
  onClose: () => void;
  children: React.ReactNode;
  title?: string;
  description?: string;
  showCloseButton?: boolean;
  closeOnOverlayClick?: boolean;
  closeOnEscape?: boolean;
  blur?: 'none' | 'sm' | 'md' | 'lg';
  footer?: React.ReactNode;
  className?: string;
  overlayClassName?: string;
}

export function Modal({
  isOpen,
  onClose,
  children,
  title,
  description,
  showCloseButton = true,
  closeOnOverlayClick = true,
  closeOnEscape = true,
  size,
  rounded,
  position = 'center',
  blur,
  footer,
  className,
  overlayClassName,
}: ModalProps) {
  const modalRef = useRef<HTMLDivElement>(null);
  const previousActiveElement = useRef<HTMLElement | null>(null);

  const handleEscape = useCallback(
    (e: KeyboardEvent) => {
      if (e.key === 'Escape' && closeOnEscape) {
        onClose();
      }
    },
    [closeOnEscape, onClose]
  );

  const handleOverlayClick = (e: React.MouseEvent) => {
    if (e.target === e.currentTarget && closeOnOverlayClick) {
      onClose();
    }
  };

  useEffect(() => {
    if (isOpen) {
      previousActiveElement.current = document.activeElement as HTMLElement;
      document.addEventListener('keydown', handleEscape);
      document.body.style.overflow = 'hidden';
      modalRef.current?.focus();
    }

    return () => {
      document.removeEventListener('keydown', handleEscape);
      document.body.style.overflow = '';
      previousActiveElement.current?.focus();
    };
  }, [isOpen, handleEscape]);

  if (!isOpen) return null;

  const positionClasses = {
    center: 'items-center justify-center',
    top: 'items-start justify-center pt-20',
    bottom: 'items-end justify-center pb-20',
  };

  const modal = (
    <div
      className={cn(
        overlayVariants({ blur }),
        'flex',
        positionClasses[position || 'center'],
        overlayClassName
      )}
      onClick={handleOverlayClick}
      role="dialog"
      aria-modal="true"
      aria-labelledby={title ? 'modal-title' : undefined}
      aria-describedby={description ? 'modal-description' : undefined}
    >
      <div
        ref={modalRef}
        className={cn(
          modalVariants({ size, rounded, position }),
          'animate-in fade-in-0 zoom-in-95',
          className
        )}
        tabIndex={-1}
      >
        {(title || showCloseButton) && (
          <div className="flex items-start justify-between p-4 border-b border-gray-200 dark:border-gray-800">
            <div className="flex-1 pr-4">
              {title && (
                <h2
                  id="modal-title"
                  className="text-lg font-semibold text-gray-900 dark:text-white"
                >
                  {title}
                </h2>
              )}
              {description && (
                <p
                  id="modal-description"
                  className="mt-1 text-sm text-gray-500 dark:text-gray-400"
                >
                  {description}
                </p>
              )}
            </div>
            {showCloseButton && (
              <Button
                variant="ghost"
                size="icon-sm"
                onClick={onClose}
                aria-label="Close modal"
              >
                <X className="h-4 w-4" />
              </Button>
            )}
          </div>
        )}

        <div className="flex-1 overflow-y-auto p-4">{children}</div>

        {footer && (
          <div className="flex items-center justify-end gap-2 p-4 border-t border-gray-200 dark:border-gray-800">
            {footer}
          </div>
        )}
      </div>
    </div>
  );

  return typeof document !== 'undefined'
    ? createPortal(modal, document.body)
    : null;
}

// Confirmation Modal
export interface ConfirmModalProps {
  isOpen: boolean;
  onClose: () => void;
  onConfirm: () => void;
  title: string;
  message: string;
  confirmText?: string;
  cancelText?: string;
  variant?: 'danger' | 'warning' | 'info';
  loading?: boolean;
}

export function ConfirmModal({
  isOpen,
  onClose,
  onConfirm,
  title,
  message,
  confirmText = 'Confirm',
  cancelText = 'Cancel',
  variant = 'info',
  loading = false,
}: ConfirmModalProps) {
  const variantStyles = {
    danger: 'danger',
    warning: 'primary',
    info: 'primary',
  } as const;

  return (
    <Modal
      isOpen={isOpen}
      onClose={onClose}
      title={title}
      size="sm"
      footer={
        <>
          <Button variant="ghost" onClick={onClose} disabled={loading}>
            {cancelText}
          </Button>
          <Button
            variant={variantStyles[variant]}
            onClick={onConfirm}
            loading={loading}
          >
            {confirmText}
          </Button>
        </>
      }
    >
      <p className="text-gray-600 dark:text-gray-300">{message}</p>
    </Modal>
  );
}

// Sheet (Slide-in panel)
export interface SheetProps {
  isOpen: boolean;
  onClose: () => void;
  children: React.ReactNode;
  title?: string;
  side?: 'left' | 'right' | 'top' | 'bottom';
  size?: 'sm' | 'md' | 'lg' | 'xl' | 'full';
  showCloseButton?: boolean;
}

export function Sheet({
  isOpen,
  onClose,
  children,
  title,
  side = 'right',
  size = 'md',
  showCloseButton = true,
}: SheetProps) {
  const sizeClasses = {
    left: {
      sm: 'w-64',
      md: 'w-80',
      lg: 'w-96',
      xl: 'w-[480px]',
      full: 'w-full',
    },
    right: {
      sm: 'w-64',
      md: 'w-80',
      lg: 'w-96',
      xl: 'w-[480px]',
      full: 'w-full',
    },
    top: {
      sm: 'h-32',
      md: 'h-48',
      lg: 'h-64',
      xl: 'h-96',
      full: 'h-full',
    },
    bottom: {
      sm: 'h-32',
      md: 'h-48',
      lg: 'h-64',
      xl: 'h-96',
      full: 'h-full',
    },
  };

  const positionClasses = {
    left: 'inset-y-0 left-0',
    right: 'inset-y-0 right-0',
    top: 'inset-x-0 top-0',
    bottom: 'inset-x-0 bottom-0',
  };

  const translateClasses = {
    left: isOpen ? 'translate-x-0' : '-translate-x-full',
    right: isOpen ? 'translate-x-0' : 'translate-x-full',
    top: isOpen ? 'translate-y-0' : '-translate-y-full',
    bottom: isOpen ? 'translate-y-0' : 'translate-y-full',
  };

  useEffect(() => {
    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };

    if (isOpen) {
      document.addEventListener('keydown', handleEscape);
      document.body.style.overflow = 'hidden';
    }

    return () => {
      document.removeEventListener('keydown', handleEscape);
      document.body.style.overflow = '';
    };
  }, [isOpen, onClose]);

  if (!isOpen && !document.querySelector('[data-sheet-open]')) return null;

  const sheet = (
    <>
      <div
        className={cn(
          'fixed inset-0 z-50 bg-black/50 backdrop-blur-sm transition-opacity duration-300',
          isOpen ? 'opacity-100' : 'opacity-0 pointer-events-none'
        )}
        onClick={onClose}
      />
      <div
        data-sheet-open={isOpen}
        className={cn(
          'fixed z-50 bg-white dark:bg-gray-900 shadow-2xl transition-transform duration-300 ease-out flex flex-col',
          positionClasses[side],
          sizeClasses[side][size],
          translateClasses[side]
        )}
      >
        {(title || showCloseButton) && (
          <div className="flex items-center justify-between p-4 border-b border-gray-200 dark:border-gray-800">
            {title && (
              <h2 className="text-lg font-semibold text-gray-900 dark:text-white">
                {title}
              </h2>
            )}
            {showCloseButton && (
              <Button variant="ghost" size="icon-sm" onClick={onClose}>
                <X className="h-4 w-4" />
              </Button>
            )}
          </div>
        )}
        <div className="flex-1 overflow-y-auto p-4">{children}</div>
      </div>
    </>
  );

  return typeof document !== 'undefined'
    ? createPortal(sheet, document.body)
    : null;
}

export { overlayVariants, modalVariants };
