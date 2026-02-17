/**
 * AI Components
 *
 * React components for AI-powered features:
 * - AudioAnalyzer - Analyze audio files with AI
 * - ModelTrainer - Train custom models
 */

export { AudioAnalyzer } from './AudioAnalyzer';
export { ModelTrainer } from './ModelTrainer';

export default {
  AudioAnalyzer: () => import('./AudioAnalyzer'),
  ModelTrainer: () => import('./ModelTrainer'),
};
