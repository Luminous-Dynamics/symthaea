/**
 * Mycelix Mail - Example Application Entry Point
 */

import React from 'react';
import { createRoot } from 'react-dom/client';
import App from './App';

// Import Tailwind CSS (would be configured in build)
// import './styles.css';

const container = document.getElementById('root');
if (container) {
  const root = createRoot(container);
  root.render(
    <React.StrictMode>
      <App />
    </React.StrictMode>
  );
}

export { App };
