import React, { useState } from 'react';

type Variant = 'info' | 'warning' | 'success' | 'error';

export function StatusBanner({
  children,
  variant = 'info',
  dismissible = false,
  onDismiss,
  className = '',
}: {
  children: React.ReactNode;
  variant?: Variant;
  dismissible?: boolean;
  onDismiss?: () => void;
  className?: string;
}) {
  const [closed, setClosed] = useState(false);
  if (closed) return null;

  const colors = {
    info: 'bg-purple-500/10 border-purple-500/30 text-purple-200',
    warning: 'bg-yellow-500/10 border-yellow-500/30 text-yellow-200',
    success: 'bg-emerald-500/10 border-emerald-500/30 text-emerald-200',
    error: 'bg-red-500/10 border-red-500/30 text-red-200',
  }[variant];

  return (
    <div className={`rounded-xl px-4 py-3 border ${colors} ${className}`}>
      <div className="flex items-start justify-between gap-3">
        <div className="text-sm">{children}</div>
        {dismissible && (
          <button
            onClick={() => { setClosed(true); onDismiss?.(); }}
            className="text-current/80 hover:text-current"
            aria-label="Dismiss"
          >
            ×
          </button>
        )}
      </div>
    </div>
  );
}

