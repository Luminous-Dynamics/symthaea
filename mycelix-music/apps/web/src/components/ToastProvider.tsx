import React, { createContext, useContext, useMemo, useState, useCallback } from 'react';

type ToastType = 'success' | 'error' | 'info';

interface Toast {
  id: string;
  message: string;
  type: ToastType;
}

interface ToastContextValue {
  show: (message: string, type?: ToastType) => void;
}

const ToastContext = createContext<ToastContextValue | null>(null);

export function ToastProvider({ children }: { children: React.ReactNode }) {
  const [toasts, setToasts] = useState<Toast[]>([]);

  const show: ToastContextValue['show'] = useCallback((message, type = 'info') => {
    const id = Math.random().toString(36).slice(2);
    setToasts((prev) => [...prev, { id, message, type }]);
    setTimeout(() => {
      setToasts((prev) => prev.filter((t) => t.id !== id));
    }, 2500);
  }, []);

  const value = useMemo(() => ({ show }), [show]);

  return (
    <ToastContext.Provider value={value}>
      {children}
      {/* Toast Container */}
      <div className="fixed bottom-4 left-1/2 -translate-x-1/2 z-50 space-y-2 w-[90%] max-w-md">
        {toasts.map((t) => (
          <div
            key={t.id}
            className={[
              'px-4 py-2 rounded-lg shadow backdrop-blur-sm border',
              'text-white',
              t.type === 'success' && 'bg-green-600/80 border-green-400/40',
              t.type === 'error' && 'bg-red-600/80 border-red-400/40',
              t.type === 'info' && 'bg-purple-600/80 border-purple-400/40',
            ].filter(Boolean).join(' ')}
          >
            {t.message}
          </div>
        ))}
      </div>
    </ToastContext.Provider>
  );
}

export function useToast() {
  const ctx = useContext(ToastContext);
  if (!ctx) throw new Error('useToast must be used within ToastProvider');
  return ctx;
}

