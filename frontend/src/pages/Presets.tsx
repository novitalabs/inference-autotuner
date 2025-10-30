import { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { presetService } from '../services/presetService';
import toast from 'react-hot-toast';
import type { Preset } from '../types/preset';
import PresetEditModal from '../components/PresetEditModal';

export default function Presets() {
  const queryClient = useQueryClient();
  const [selectedCategory, setSelectedCategory] = useState<string>('');
  const [deleteConfirm, setDeleteConfirm] = useState<number | null>(null);
  const [editingPreset, setEditingPreset] = useState<Preset | null>(null);

  // Fetch presets
  const { data: presets = [], isLoading } = useQuery({
    queryKey: ['presets', selectedCategory],
    queryFn: () => presetService.getAll(selectedCategory || undefined),
  });

  // Delete mutation
  const deleteMutation = useMutation({
    mutationFn: (id: number) => presetService.delete(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['presets'] });
      toast.success('Preset deleted successfully');
      setDeleteConfirm(null);
    },
    onError: (error: any) => {
      toast.error(error.response?.data?.detail || 'Failed to delete preset');
      setDeleteConfirm(null);
    },
  });

  // Export preset
  const handleExport = async (preset: Preset) => {
    try {
      const blob = await presetService.exportPreset(preset.id);
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `preset-${preset.name.toLowerCase().replace(/\s+/g, '-')}.json`;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
      toast.success('Preset exported successfully');
    } catch (error) {
      toast.error('Failed to export preset');
    }
  };

  // Import preset
  const handleImport = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    try {
      await presetService.importPreset(file);
      queryClient.invalidateQueries({ queryKey: ['presets'] });
      toast.success('Preset imported successfully');
      event.target.value = ''; // Reset input
    } catch (error: any) {
      toast.error(error.response?.data?.detail || 'Failed to import preset');
      event.target.value = '';
    }
  };

  const categories = Array.from(new Set(presets.map(p => p.category).filter(Boolean)));

  if (isLoading) {
    return (
      <div className="p-8 text-center">
        <div className="inline-block animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
        <p className="mt-2 text-gray-600">Loading presets...</p>
      </div>
    );
  }

  return (
    <div className="p-6">
      <div className="mb-6">
        <div className="flex items-center justify-between mb-4">
          <h1 className="text-2xl font-bold">Parameter Presets</h1>
          <div className="flex gap-2">
            <label className="px-4 py-2 bg-green-600 text-white rounded hover:bg-green-700 cursor-pointer">
              Import
              <input
                type="file"
                accept=".json"
                onChange={handleImport}
                className="hidden"
              />
            </label>
          </div>
        </div>

        {/* Category Filter */}
        <div className="flex items-center gap-2">
          <label className="text-sm font-medium text-gray-700">Filter by category:</label>
          <select
            value={selectedCategory}
            onChange={(e) => setSelectedCategory(e.target.value)}
            className="border border-gray-300 rounded px-3 py-1 text-sm"
          >
            <option value="">All</option>
            {categories.map(cat => (
              <option key={cat} value={cat}>{cat}</option>
            ))}
          </select>
          {selectedCategory && (
            <button
              onClick={() => setSelectedCategory('')}
              className="text-sm text-gray-600 hover:text-gray-800 underline"
            >
              Clear filter
            </button>
          )}
        </div>
      </div>

      {/* Presets Table */}
      {presets.length === 0 ? (
        <div className="text-center py-12 bg-gray-50 rounded-lg">
          <p className="text-gray-600">No presets found</p>
          {selectedCategory && (
            <button
              onClick={() => setSelectedCategory('')}
              className="mt-2 text-blue-600 hover:text-blue-800 text-sm"
            >
              Show all presets
            </button>
          )}
        </div>
      ) : (
        <div className="bg-white border border-gray-200 rounded-lg overflow-hidden">
          <table className="min-w-full divide-y divide-gray-200">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Name
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Description
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Category
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Parameters
                </th>
                <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Actions
                </th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              {presets.map((preset) => (
                <tr key={preset.id} className="hover:bg-gray-50">
                  <td className="px-6 py-4 whitespace-nowrap">
                    <div className="flex items-center gap-2">
                      <span className="font-medium text-gray-900">{preset.name}</span>
                      {preset.is_system && (
                        <span className="px-2 py-0.5 text-xs bg-gray-200 text-gray-700 rounded">
                          System
                        </span>
                      )}
                    </div>
                  </td>
                  <td className="px-6 py-4">
                    <div className="text-sm text-gray-600 max-w-md truncate">
                      {preset.description || '-'}
                    </div>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <span className="px-2 py-1 text-xs bg-blue-100 text-blue-800 rounded">
                      {preset.category || 'uncategorized'}
                    </span>
                  </td>
                  <td className="px-6 py-4">
                    <span className="text-sm text-gray-600">
                      {Object.keys(preset.parameters).length} params
                    </span>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-right text-sm">
                    <div className="flex justify-end gap-2">
                      <button
                        onClick={() => handleExport(preset)}
                        className="text-blue-600 hover:text-blue-800"
                        title="Export"
                      >
                        Export
                      </button>
                      <button
                        onClick={() => setEditingPreset(preset)}
                        className="text-green-600 hover:text-green-800"
                        title="Edit"
                      >
                        Edit
                      </button>
                      {deleteConfirm === preset.id ? (
                        <div className="flex gap-2">
                          <button
                            onClick={() => deleteMutation.mutate(preset.id)}
                            className="text-red-600 hover:text-red-800 font-medium"
                          >
                            Confirm
                          </button>
                          <button
                            onClick={() => setDeleteConfirm(null)}
                            className="text-gray-600 hover:text-gray-800"
                          >
                            Cancel
                          </button>
                        </div>
                      ) : (
                        <button
                          onClick={() => setDeleteConfirm(preset.id)}
                          className="text-red-600 hover:text-red-800"
                          title="Delete"
                        >
                          Delete
                        </button>
                      )}
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Summary */}
      <div className="mt-4 text-sm text-gray-600">
        Showing {presets.length} preset{presets.length !== 1 ? 's' : ''}
        {selectedCategory && ` in category "${selectedCategory}"`}
      </div>

      {/* Edit Modal */}
      {editingPreset && (
        <PresetEditModal
          preset={editingPreset}
          onClose={() => setEditingPreset(null)}
        />
      )}
    </div>
  );
}
