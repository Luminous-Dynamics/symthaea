import React from 'react';
import ReactDOM from 'react-dom/client';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import App from './App';
import './index.css';
import './styles/animations.css';

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 30000, // 30 seconds - data is fresh for 30s
      cacheTime: 5 * 60 * 1000, // 5 minutes - keep in cache for 5 min
      retry: 1, // Retry failed requests once
      refetchOnWindowFocus: true, // Refresh on window focus
      refetchOnReconnect: true, // Refresh on reconnect
    },
    mutations: {
      retry: 0, // Don't retry mutations
    },
  },
});

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <QueryClientProvider client={queryClient}>
      <App />
    </QueryClientProvider>
  </React.StrictMode>
);
