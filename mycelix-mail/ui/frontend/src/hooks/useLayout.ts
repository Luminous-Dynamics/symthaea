import { useState, useEffect } from 'react';

export type LayoutMode = 'vertical' | 'horizontal' | 'no-preview';

export interface LayoutConfig {
  label: string;
  description: string;
  icon: string;
  listClass: string;
  previewClass: string;
  containerClass: string;
  showPreview: boolean;
}

export const LAYOUT_CONFIGS: Record<LayoutMode, LayoutConfig> = {
  vertical: {
    label: 'Vertical Split',
    description: 'Email list on the left, preview on the right',
    icon: '⬌',
    containerClass: 'flex flex-row h-full',
    listClass: 'w-96 flex-shrink-0 border-r border-gray-200 dark:border-gray-700 overflow-auto',
    previewClass: 'flex-1 overflow-auto',
    showPreview: true,
  },
  horizontal: {
    label: 'Horizontal Split',
    description: 'Email list on top, preview on bottom',
    icon: '⬍',
    containerClass: 'flex flex-col h-full',
    listClass: 'h-80 flex-shrink-0 border-b border-gray-200 dark:border-gray-700 overflow-auto',
    previewClass: 'flex-1 overflow-auto',
    showPreview: true,
  },
  'no-preview': {
    label: 'No Preview',
    description: 'Full-screen email list, click to open',
    icon: '☰',
    containerClass: 'h-full',
    listClass: 'h-full overflow-auto',
    previewClass: 'hidden',
    showPreview: false,
  },
};

/**
 * Hook for managing email layout mode (vertical split, horizontal split, or no preview)
 * Persists user preference to localStorage
 */
export const useLayout = () => {
  const [layout, setLayoutState] = useState<LayoutMode>(() => {
    const saved = localStorage.getItem('email-layout');
    return (saved as LayoutMode) || 'vertical';
  });

  useEffect(() => {
    localStorage.setItem('email-layout', layout);
  }, [layout]);

  const setLayout = (newLayout: LayoutMode) => {
    setLayoutState(newLayout);
  };

  return {
    layout,
    setLayout,
    config: LAYOUT_CONFIGS[layout],
  };
};
