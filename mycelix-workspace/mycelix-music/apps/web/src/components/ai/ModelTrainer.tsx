// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Model Trainer Component
 *
 * UI for custom model training:
 * - Dataset creation and management
 * - Training configuration
 * - Progress visualization
 * - Model export/import
 */

import React, { useState, useEffect, useCallback, useRef } from 'react';
import {
  getModelTrainer,
  type Dataset,
  type TrainingExample,
  type TrainingConfig,
  type TrainingProgress,
  type TrainingResult,
} from '../../lib/ai';

// ==================== Types ====================

interface ModelTrainerProps {
  className?: string;
  onModelTrained?: (result: TrainingResult) => void;
}

// ==================== Sub-components ====================

const DatasetCard: React.FC<{
  dataset: Dataset;
  selected: boolean;
  onSelect: () => void;
  onDelete: () => void;
}> = ({ dataset, selected, onSelect, onDelete }) => (
  <div
    className={`
      p-4 rounded-lg cursor-pointer transition-all
      ${selected ? 'bg-purple-600 ring-2 ring-purple-400' : 'bg-gray-800 hover:bg-gray-700'}
    `}
    onClick={onSelect}
  >
    <div className="flex justify-between items-start mb-2">
      <h4 className="font-semibold">{dataset.name}</h4>
      <button
        className="text-gray-400 hover:text-red-400 transition-colors"
        onClick={(e) => {
          e.stopPropagation();
          onDelete();
        }}
      >
        &times;
      </button>
    </div>
    <div className="text-sm text-gray-400 space-y-1">
      <p>{dataset.examples.length} examples</p>
      <p>{dataset.labels.length} labels: {dataset.labels.join(', ')}</p>
      <p>Updated: {new Date(dataset.updatedAt).toLocaleDateString()}</p>
    </div>
  </div>
);

const ExampleRow: React.FC<{
  example: TrainingExample;
  onDelete: () => void;
  onPlay: () => void;
}> = ({ example, onDelete, onPlay }) => (
  <div className="flex items-center bg-gray-800 rounded p-2 mb-2">
    <button
      className="w-8 h-8 flex items-center justify-center bg-gray-700 rounded-full mr-3 hover:bg-gray-600"
      onClick={onPlay}
    >
      ▶
    </button>
    <div className="flex-1">
      <div className="text-sm font-medium">{example.id.slice(-8)}</div>
      <div className="text-xs text-gray-400">
        Label: <span className="text-purple-400">{example.label}</span>
        {' | '}
        Duration: {(example.audio.length / 22050).toFixed(1)}s
      </div>
    </div>
    <button
      className="text-gray-400 hover:text-red-400 px-2"
      onClick={onDelete}
    >
      &times;
    </button>
  </div>
);

const TrainingChart: React.FC<{
  history: TrainingResult['epochHistory'] | null;
}> = ({ history }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    if (!history || !canvasRef.current) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const width = canvas.width;
    const height = canvas.height;
    const padding = 40;

    // Clear
    ctx.fillStyle = '#1F2937';
    ctx.fillRect(0, 0, width, height);

    if (history.loss.length === 0) return;

    // Calculate scales
    const maxLoss = Math.max(...history.loss, ...history.valLoss);
    const epochs = history.loss.length;

    // Draw grid
    ctx.strokeStyle = '#374151';
    ctx.lineWidth = 1;
    for (let i = 0; i <= 4; i++) {
      const y = padding + (height - 2 * padding) * (i / 4);
      ctx.beginPath();
      ctx.moveTo(padding, y);
      ctx.lineTo(width - padding, y);
      ctx.stroke();
    }

    // Draw loss line
    ctx.strokeStyle = '#EF4444';
    ctx.lineWidth = 2;
    ctx.beginPath();
    history.loss.forEach((loss, i) => {
      const x = padding + (width - 2 * padding) * (i / (epochs - 1 || 1));
      const y = padding + (height - 2 * padding) * (loss / maxLoss);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();

    // Draw validation loss line
    ctx.strokeStyle = '#F97316';
    ctx.lineWidth = 2;
    ctx.setLineDash([5, 5]);
    ctx.beginPath();
    history.valLoss.forEach((loss, i) => {
      const x = padding + (width - 2 * padding) * (i / (epochs - 1 || 1));
      const y = padding + (height - 2 * padding) * (loss / maxLoss);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
    ctx.setLineDash([]);

    // Draw accuracy line
    ctx.strokeStyle = '#10B981';
    ctx.lineWidth = 2;
    ctx.beginPath();
    history.accuracy.forEach((acc, i) => {
      const x = padding + (width - 2 * padding) * (i / (epochs - 1 || 1));
      const y = padding + (height - 2 * padding) * (1 - acc);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();

    // Legend
    ctx.font = '12px sans-serif';
    ctx.fillStyle = '#EF4444';
    ctx.fillText('Training Loss', padding, 20);
    ctx.fillStyle = '#F97316';
    ctx.fillText('Val Loss', padding + 100, 20);
    ctx.fillStyle = '#10B981';
    ctx.fillText('Accuracy', padding + 180, 20);
  }, [history]);

  return (
    <canvas
      ref={canvasRef}
      width={400}
      height={200}
      className="w-full rounded-lg"
    />
  );
};

// ==================== Main Component ====================

export const ModelTrainer: React.FC<ModelTrainerProps> = ({
  className = '',
  onModelTrained,
}) => {
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [selectedDataset, setSelectedDataset] = useState<Dataset | null>(null);
  const [isTraining, setIsTraining] = useState(false);
  const [progress, setProgress] = useState<TrainingProgress | null>(null);
  const [result, setResult] = useState<TrainingResult | null>(null);
  const [newDatasetName, setNewDatasetName] = useState('');
  const [newDatasetLabels, setNewDatasetLabels] = useState('');
  const [showNewDataset, setShowNewDataset] = useState(false);

  const [config, setConfig] = useState<TrainingConfig>({
    epochs: 50,
    batchSize: 32,
    learningRate: 0.001,
    validationSplit: 0.2,
    earlyStoppingPatience: 5,
    augmentation: {
      timeStretch: true,
      pitchShift: true,
      noiseInjection: true,
      gainVariation: true,
    },
  });

  const fileInputRef = useRef<HTMLInputElement>(null);
  const audioContextRef = useRef<AudioContext | null>(null);

  // Load datasets on mount
  useEffect(() => {
    const trainer = getModelTrainer();
    trainer.initialize().then(() => {
      setDatasets(trainer.getDatasetManager().getAllDatasets());
    });
  }, []);

  // Create new dataset
  const handleCreateDataset = useCallback(async () => {
    if (!newDatasetName || !newDatasetLabels) return;

    const trainer = getModelTrainer();
    const labels = newDatasetLabels.split(',').map(l => l.trim());
    const dataset = await trainer.getDatasetManager().createDataset(newDatasetName, labels);

    setDatasets([...datasets, dataset]);
    setSelectedDataset(dataset);
    setShowNewDataset(false);
    setNewDatasetName('');
    setNewDatasetLabels('');
  }, [newDatasetName, newDatasetLabels, datasets]);

  // Delete dataset
  const handleDeleteDataset = useCallback(async (datasetId: string) => {
    const trainer = getModelTrainer();
    await trainer.getDatasetManager().deleteDataset(datasetId);
    setDatasets(datasets.filter(d => d.id !== datasetId));
    if (selectedDataset?.id === datasetId) {
      setSelectedDataset(null);
    }
  }, [datasets, selectedDataset]);

  // Add example to dataset
  const handleAddExample = useCallback(async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (!selectedDataset || !e.target.files?.length) return;

    const file = e.target.files[0];
    const label = prompt('Enter label for this example:', selectedDataset.labels[0]);
    if (!label || !selectedDataset.labels.includes(label)) {
      alert(`Label must be one of: ${selectedDataset.labels.join(', ')}`);
      return;
    }

    // Decode audio
    const arrayBuffer = await file.arrayBuffer();
    if (!audioContextRef.current) {
      audioContextRef.current = new AudioContext({ sampleRate: 22050 });
    }
    const audioBuffer = await audioContextRef.current.decodeAudioData(arrayBuffer);
    const audio = audioBuffer.getChannelData(0);

    // Add to dataset
    const trainer = getModelTrainer();
    await trainer.getDatasetManager().addExample(selectedDataset.id, { audio, label });

    // Refresh datasets
    setDatasets(trainer.getDatasetManager().getAllDatasets());
    setSelectedDataset(trainer.getDatasetManager().getDataset(selectedDataset.id) || null);
  }, [selectedDataset]);

  // Delete example
  const handleDeleteExample = useCallback(async (exampleId: string) => {
    if (!selectedDataset) return;

    const trainer = getModelTrainer();
    await trainer.getDatasetManager().removeExample(selectedDataset.id, exampleId);

    // Refresh
    setDatasets(trainer.getDatasetManager().getAllDatasets());
    setSelectedDataset(trainer.getDatasetManager().getDataset(selectedDataset.id) || null);
  }, [selectedDataset]);

  // Play example audio
  const handlePlayExample = useCallback(async (audio: Float32Array) => {
    if (!audioContextRef.current) {
      audioContextRef.current = new AudioContext({ sampleRate: 22050 });
    }

    const buffer = audioContextRef.current.createBuffer(1, audio.length, 22050);
    buffer.copyToChannel(audio, 0);

    const source = audioContextRef.current.createBufferSource();
    source.buffer = buffer;
    source.connect(audioContextRef.current.destination);
    source.start();
  }, []);

  // Start training
  const handleTrain = useCallback(async () => {
    if (!selectedDataset) return;

    setIsTraining(true);
    setProgress(null);
    setResult(null);

    try {
      const trainer = getModelTrainer();
      const trainingResult = await trainer.train(
        selectedDataset.id,
        config,
        (p) => setProgress(p)
      );

      setResult(trainingResult);
      onModelTrained?.(trainingResult);
    } catch (error) {
      console.error('Training failed:', error);
      alert(`Training failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
    } finally {
      setIsTraining(false);
    }
  }, [selectedDataset, config, onModelTrained]);

  // Export model
  const handleExport = useCallback(async () => {
    if (!result || !selectedDataset) return;

    const trainer = getModelTrainer();
    const exported = await trainer.exportModel(result.modelId, selectedDataset.labels);

    const blob = new Blob([JSON.stringify(exported)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${result.modelId}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }, [result, selectedDataset]);

  return (
    <div className={`${className}`}>
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Datasets Panel */}
        <div className="bg-gray-900 rounded-lg p-4">
          <div className="flex justify-between items-center mb-4">
            <h3 className="text-lg font-semibold">Datasets</h3>
            <button
              className="px-3 py-1 bg-purple-600 rounded text-sm hover:bg-purple-500"
              onClick={() => setShowNewDataset(true)}
            >
              + New
            </button>
          </div>

          {showNewDataset && (
            <div className="bg-gray-800 rounded-lg p-4 mb-4">
              <input
                type="text"
                placeholder="Dataset name"
                className="w-full bg-gray-700 rounded px-3 py-2 mb-2"
                value={newDatasetName}
                onChange={(e) => setNewDatasetName(e.target.value)}
              />
              <input
                type="text"
                placeholder="Labels (comma-separated)"
                className="w-full bg-gray-700 rounded px-3 py-2 mb-2"
                value={newDatasetLabels}
                onChange={(e) => setNewDatasetLabels(e.target.value)}
              />
              <div className="flex gap-2">
                <button
                  className="flex-1 px-3 py-2 bg-purple-600 rounded hover:bg-purple-500"
                  onClick={handleCreateDataset}
                >
                  Create
                </button>
                <button
                  className="px-3 py-2 bg-gray-700 rounded hover:bg-gray-600"
                  onClick={() => setShowNewDataset(false)}
                >
                  Cancel
                </button>
              </div>
            </div>
          )}

          <div className="space-y-2">
            {datasets.map((dataset) => (
              <DatasetCard
                key={dataset.id}
                dataset={dataset}
                selected={selectedDataset?.id === dataset.id}
                onSelect={() => setSelectedDataset(dataset)}
                onDelete={() => handleDeleteDataset(dataset.id)}
              />
            ))}

            {datasets.length === 0 && (
              <p className="text-gray-500 text-center py-8">
                No datasets yet. Create one to get started.
              </p>
            )}
          </div>
        </div>

        {/* Examples Panel */}
        <div className="bg-gray-900 rounded-lg p-4">
          <div className="flex justify-between items-center mb-4">
            <h3 className="text-lg font-semibold">Examples</h3>
            {selectedDataset && (
              <button
                className="px-3 py-1 bg-purple-600 rounded text-sm hover:bg-purple-500"
                onClick={() => fileInputRef.current?.click()}
              >
                + Add Audio
              </button>
            )}
          </div>

          <input
            ref={fileInputRef}
            type="file"
            accept="audio/*"
            className="hidden"
            onChange={handleAddExample}
          />

          {selectedDataset ? (
            <div className="max-h-96 overflow-y-auto">
              {selectedDataset.examples.map((example) => (
                <ExampleRow
                  key={example.id}
                  example={example}
                  onDelete={() => handleDeleteExample(example.id)}
                  onPlay={() => handlePlayExample(example.audio)}
                />
              ))}

              {selectedDataset.examples.length === 0 && (
                <p className="text-gray-500 text-center py-8">
                  No examples yet. Add audio files to train your model.
                </p>
              )}
            </div>
          ) : (
            <p className="text-gray-500 text-center py-8">
              Select a dataset to view examples
            </p>
          )}
        </div>

        {/* Training Panel */}
        <div className="bg-gray-900 rounded-lg p-4">
          <h3 className="text-lg font-semibold mb-4">Training</h3>

          <div className="space-y-4 mb-6">
            <div>
              <label className="block text-sm text-gray-400 mb-1">Epochs</label>
              <input
                type="number"
                className="w-full bg-gray-800 rounded px-3 py-2"
                value={config.epochs}
                onChange={(e) => setConfig({ ...config, epochs: parseInt(e.target.value) || 50 })}
                disabled={isTraining}
              />
            </div>

            <div>
              <label className="block text-sm text-gray-400 mb-1">Batch Size</label>
              <input
                type="number"
                className="w-full bg-gray-800 rounded px-3 py-2"
                value={config.batchSize}
                onChange={(e) => setConfig({ ...config, batchSize: parseInt(e.target.value) || 32 })}
                disabled={isTraining}
              />
            </div>

            <div>
              <label className="block text-sm text-gray-400 mb-1">Learning Rate</label>
              <input
                type="number"
                step="0.0001"
                className="w-full bg-gray-800 rounded px-3 py-2"
                value={config.learningRate}
                onChange={(e) => setConfig({ ...config, learningRate: parseFloat(e.target.value) || 0.001 })}
                disabled={isTraining}
              />
            </div>

            <div>
              <label className="block text-sm text-gray-400 mb-2">Data Augmentation</label>
              <div className="space-y-2">
                {Object.entries(config.augmentation || {}).map(([key, value]) => (
                  <label key={key} className="flex items-center">
                    <input
                      type="checkbox"
                      checked={value}
                      onChange={(e) => setConfig({
                        ...config,
                        augmentation: { ...config.augmentation, [key]: e.target.checked },
                      })}
                      disabled={isTraining}
                      className="mr-2"
                    />
                    <span className="text-sm capitalize">{key.replace(/([A-Z])/g, ' $1')}</span>
                  </label>
                ))}
              </div>
            </div>
          </div>

          {/* Progress */}
          {isTraining && progress && (
            <div className="mb-4">
              <div className="flex justify-between text-sm mb-1">
                <span>Epoch {progress.epoch}/{progress.totalEpochs}</span>
                <span>Accuracy: {(progress.accuracy * 100).toFixed(1)}%</span>
              </div>
              <div className="h-2 bg-gray-700 rounded-full overflow-hidden">
                <div
                  className="h-full bg-purple-500 transition-all duration-300"
                  style={{ width: `${(progress.epoch / progress.totalEpochs) * 100}%` }}
                />
              </div>
              <div className="text-xs text-gray-500 mt-1">
                Loss: {progress.loss.toFixed(4)}
                {progress.valLoss !== undefined && ` | Val Loss: ${progress.valLoss.toFixed(4)}`}
              </div>
            </div>
          )}

          {/* Training Chart */}
          {result && (
            <div className="mb-4">
              <TrainingChart history={result.epochHistory} />
              <div className="grid grid-cols-2 gap-2 mt-2 text-sm">
                <div className="bg-gray-800 rounded p-2 text-center">
                  <div className="text-gray-400">Final Accuracy</div>
                  <div className="text-xl font-bold text-green-400">
                    {(result.finalAccuracy * 100).toFixed(1)}%
                  </div>
                </div>
                <div className="bg-gray-800 rounded p-2 text-center">
                  <div className="text-gray-400">Training Time</div>
                  <div className="text-xl font-bold">
                    {(result.trainingTime / 1000).toFixed(1)}s
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Actions */}
          <div className="space-y-2">
            <button
              className={`
                w-full py-3 rounded-lg font-semibold transition-colors
                ${isTraining
                  ? 'bg-gray-700 cursor-not-allowed'
                  : selectedDataset && selectedDataset.examples.length >= 10
                    ? 'bg-purple-600 hover:bg-purple-500'
                    : 'bg-gray-700 cursor-not-allowed'
                }
              `}
              onClick={handleTrain}
              disabled={isTraining || !selectedDataset || selectedDataset.examples.length < 10}
            >
              {isTraining ? 'Training...' : 'Start Training'}
            </button>

            {result && (
              <button
                className="w-full py-2 bg-gray-800 rounded-lg hover:bg-gray-700"
                onClick={handleExport}
              >
                Export Model
              </button>
            )}
          </div>

          {selectedDataset && selectedDataset.examples.length < 10 && (
            <p className="text-sm text-yellow-500 mt-2 text-center">
              Need at least 10 examples to train (currently: {selectedDataset.examples.length})
            </p>
          )}
        </div>
      </div>
    </div>
  );
};

export default ModelTrainer;
