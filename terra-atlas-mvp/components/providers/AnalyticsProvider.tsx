'use client';

import { useEffect } from 'react';
import { usePathname } from 'next/navigation';
import { initializeAnalytics, getAnalytics } from '@/lib/analytics';

interface AnalyticsProviderProps {
  children: React.ReactNode;
}

export default function AnalyticsProvider({ children }: AnalyticsProviderProps) {
  const pathname = usePathname();

  useEffect(() => {
    // Initialize analytics with configuration
    // In production, these would come from environment variables
    initializeAnalytics({
      gtmId: process.env.NEXT_PUBLIC_GTM_ID,
      gaId: process.env.NEXT_PUBLIC_GA_ID,
      plausibleDomain: process.env.NEXT_PUBLIC_PLAUSIBLE_DOMAIN,
      debug: process.env.NODE_ENV === 'development'
    });

    // Track initial page view
    const analytics = getAnalytics();
    if (analytics) {
      analytics.trackPageView(pathname);
    }

    // Set up session end tracking
    const handleBeforeUnload = () => {
      const analytics = getAnalytics();
      if (analytics) {
        analytics.trackSessionEnd();
      }
    };

    window.addEventListener('beforeunload', handleBeforeUnload);

    return () => {
      window.removeEventListener('beforeunload', handleBeforeUnload);
    };
  }, []);

  useEffect(() => {
    // Track page views on route change
    const analytics = getAnalytics();
    if (analytics) {
      analytics.trackPageView(pathname);
    }
  }, [pathname]);

  return <>{children}</>;
}