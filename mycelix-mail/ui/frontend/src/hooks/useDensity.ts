import { useState, useEffect } from 'react';

export type DensityLevel = 'comfortable' | 'cozy' | 'compact';

interface DensityConfig {
  level: DensityLevel;
  padding: string;
  textSize: string;
  label: string;
  description: string;
  emailsVisible: number;
}

export const DENSITY_CONFIGS: Record<DensityLevel, DensityConfig> = {
  comfortable: {
    level: 'comfortable',
    padding: 'py-3 px-4',
    textSize: 'text-sm',
    label: 'Comfortable',
    description: 'Spacious layout with maximum readability (~8 emails visible)',
    emailsVisible: 8,
  },
  cozy: {
    level: 'cozy',
    padding: 'py-2.5 px-3.5',
    textSize: 'text-sm',
    label: 'Cozy',
    description: 'Balanced spacing for good readability (~10 emails visible)',
    emailsVisible: 10,
  },
  compact: {
    level: 'compact',
    padding: 'py-2 px-3',
    textSize: 'text-xs',
    label: 'Compact',
    description: 'Dense layout to see more emails at once (~12 emails visible)',
    emailsVisible: 12,
  },
};

const STORAGE_KEY = 'email-density';

export const useDensity = () => {
  const [density, setDensityState] = useState<DensityLevel>(() => {
    const saved = localStorage.getItem(STORAGE_KEY);
    return (saved as DensityLevel) || 'comfortable';
  });

  useEffect(() => {
    localStorage.setItem(STORAGE_KEY, density);
  }, [density]);

  const setDensity = (level: DensityLevel) => {
    setDensityState(level);
  };

  const cycleDensity = () => {
    const levels: DensityLevel[] = ['comfortable', 'cozy', 'compact'];
    const currentIndex = levels.indexOf(density);
    const nextIndex = (currentIndex + 1) % levels.length;
    setDensity(levels[nextIndex]);
  };

  const config = DENSITY_CONFIGS[density];

  return {
    density,
    setDensity,
    cycleDensity,
    config,
    allConfigs: DENSITY_CONFIGS,
  };
};
