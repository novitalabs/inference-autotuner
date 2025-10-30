import { useState, useEffect } from 'react';
import { useQuery } from '@tanstack/react-query';
import { presetService } from '../services/presetService';
import type { Preset, MergeStrategy, MergeResult } from '../types/preset';

interface PresetSelectorProps {
  onParametersChange: (parameters: Record<string, any[]>) => void;
  className?: string;
}

export default function PresetSelector({ onParametersChange, className = '' }: PresetSelectorProps) {
  const [selectedPresetIds, setSelectedPresetIds] = useState<number[]>([]);
  const [mergeStrategy, setMergeStrategy] = useState<MergeStrategy>('union');
  const [showPreview, setShowPreview] = useState(false);

  // Fetch all presets
  const { data: presets = [], isLoading: presetsLoading } = useQuery({
    queryKey: ['presets'],
    queryFn: () => presetService.getAll(),
  });

  // Fetch merged parameters when selection changes
  const { data: mergeResult, isLoading: merging } = useQuery<MergeResult>({
    queryKey: ['merge-presets', selectedPresetIds, mergeStrategy],
    queryFn: () => presetService.merge(selectedPresetIds, mergeStrategy),
    enabled: selectedPresetIds.length > 0,
  });

  // Update parent when merged parameters change
  useEffect(() => {
    if (mergeResult?.parameters) {
      onParametersChange(mergeResult.parameters);
    } else if (selectedPresetIds.length === 0) {
      onParametersChange({});
    }
  }, [mergeResult, selectedPresetIds, onParametersChange]);

  const handlePresetToggle = (presetId: number) => {
    setSelectedPresetIds(prev =>
      prev.includes(presetId)
        ? prev.filter(id => id !== presetId)
        : [...prev, presetId]
    );
  };

  const handleClearAll = () => {
    setSelectedPresetIds([]);
  };

  const getPresetById = (id: number) => presets.find(p => p.id === id);

  if (presetsLoading) {
    return <div className="text-gray-500">Loading presets...</div>;
  }

  return (
    <div className={`preset-selector border rounded-lg p-4 bg-gray-50 ${className}`}>
      <div className="flex items-center justify-between mb-3">
        <h3 className="font-semibold text-lg">Apply Parameter Presets</h3>
        <button
          onClick={() => setShowPreview(!showPreview)}
          className="text-sm text-blue-600 hover:text-blue-800"
        >
          {showPreview ? 'Hide Preview' : 'Show Preview'}
        </button>
      </div>

      {/* Preset Selection */}
      <div className="mb-4">
        <label className="block text-sm font-medium text-gray-700 mb-2">
          Select Presets
        </label>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
          {presets.map(preset => (
            <label
              key={preset.id}
              className={`flex items-center p-3 border rounded cursor-pointer transition-colors ${
                selectedPresetIds.includes(preset.id)
                  ? 'bg-blue-50 border-blue-500'
                  : 'bg-white border-gray-300 hover:border-gray-400'
              }`}
            >
              <input
                type="checkbox"
                checked={selectedPresetIds.includes(preset.id)}
                onChange={() => handlePresetToggle(preset.id)}
                className="mr-3"
              />
              <div className="flex-1">
                <div className="flex items-center gap-2">
                  <span className="font-medium">{preset.name}</span>
                  {preset.is_system && (
                    <span className="text-xs px-2 py-0.5 bg-gray-200 rounded">
                      System
                    </span>
                  )}
                </div>
                {preset.description && (
                  <p className="text-xs text-gray-600 mt-1">{preset.description}</p>
                )}
                {preset.category && (
                  <span className="text-xs text-gray-500 mt-1 inline-block">
                    Category: {preset.category}
                  </span>
                )}
              </div>
            </label>
          ))}
        </div>
      </div>

      {/* Applied Presets Chips */}
      {selectedPresetIds.length > 0 && (
        <div className="mb-4">
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Applied Presets ({selectedPresetIds.length})
          </label>
          <div className="flex flex-wrap gap-2">
            {selectedPresetIds.map(id => {
              const preset = getPresetById(id);
              return preset ? (
                <span
                  key={id}
                  className="inline-flex items-center gap-2 bg-blue-100 text-blue-800 px-3 py-1 rounded-full text-sm"
                >
                  {preset.name}
                  <button
                    onClick={() => handlePresetToggle(id)}
                    className="text-blue-600 hover:text-blue-800 font-bold"
                  >
                    ×
                  </button>
                </span>
              ) : null;
            })}
            <button
              onClick={handleClearAll}
              className="text-xs text-gray-600 hover:text-gray-800 underline"
            >
              Clear all
            </button>
          </div>
        </div>
      )}

      {/* Merge Strategy */}
      {selectedPresetIds.length > 1 && (
        <div className="mb-4">
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Merge Strategy
          </label>
          <select
            value={mergeStrategy}
            onChange={(e) => setMergeStrategy(e.target.value as MergeStrategy)}
            className="border border-gray-300 rounded px-3 py-2 w-full md:w-auto"
          >
            <option value="union">Union (Combine All)</option>
            <option value="intersection">Intersection (Common Only)</option>
            <option value="last_wins">Last Wins (Override)</option>
          </select>
          <p className="text-xs text-gray-600 mt-1">
            {mergeStrategy === 'union' && 'Combines all values from all presets'}
            {mergeStrategy === 'intersection' && 'Only keeps values present in all presets'}
            {mergeStrategy === 'last_wins' && 'Later presets override earlier ones'}
          </p>
        </div>
      )}

      {/* Conflicts Warning */}
      {mergeResult?.conflicts && mergeResult.conflicts.length > 0 && (
        <div className="mb-4 bg-yellow-50 border border-yellow-200 rounded p-3">
          <div className="flex items-start gap-2">
            <span className="text-yellow-600 font-bold">⚠️</span>
            <div className="flex-1">
              <p className="font-semibold text-yellow-800 text-sm">Merge Conflicts Detected</p>
              <ul className="text-xs text-yellow-700 mt-1 space-y-1">
                {mergeResult.conflicts.map((conflict, idx) => (
                  <li key={idx}>
                    • <strong>{conflict.parameter}</strong>: {conflict.reason}
                    {conflict.overridden_by && ` (overridden by ${conflict.overridden_by})`}
                  </li>
                ))}
              </ul>
            </div>
          </div>
        </div>
      )}

      {/* Loading indicator */}
      {merging && (
        <div className="text-sm text-gray-500">
          <span className="inline-block animate-spin mr-2">⟳</span>
          Merging parameters...
        </div>
      )}

      {/* Parameter Preview */}
      {showPreview && mergeResult && (
        <div className="mt-4">
          <div className="bg-white border border-gray-300 rounded p-3">
            <div className="flex items-center justify-between mb-2">
              <p className="text-sm font-medium text-gray-700">Merged Parameters Preview</p>
              <span className="text-xs text-gray-500">
                {Object.keys(mergeResult.parameters).length} parameters
              </span>
            </div>
            <pre className="text-xs bg-gray-50 rounded p-2 overflow-auto max-h-64">
              {JSON.stringify(mergeResult.parameters, null, 2)}
            </pre>
            {mergeResult.applied_presets && (
              <p className="text-xs text-gray-600 mt-2">
                Applied: {mergeResult.applied_presets.join(', ')}
              </p>
            )}
          </div>
        </div>
      )}

      {/* Help text */}
      {selectedPresetIds.length === 0 && (
        <div className="text-sm text-gray-500 italic">
          Select one or more presets to automatically populate parameters. You can still edit them manually after applying.
        </div>
      )}
    </div>
  );
}
