import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import '@fortawesome/fontawesome-svg-core/styles.css';
import { config } from '@fortawesome/fontawesome-svg-core';
import App from './App.jsx';
import './styles.css';

config.autoAddCss = false;

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
