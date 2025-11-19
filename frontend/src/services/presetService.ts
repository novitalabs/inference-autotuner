// Preset service for API calls
import axios from 'axios';
import type { Preset, MergeResult, MergeStrategy } from '../types/preset';

// Use environment variable or default to relative path (proxy)
const API_BASE = import.meta.env.VITE_API_URL || '/api';

export const presetService = {
  async getAll(category?: string): Promise<Preset[]> {
    const params = category ? { category } : {};
    const response = await axios.get(`${API_BASE}/presets/`, { params });
    return response.data;
  },

  async getById(id: number): Promise<Preset> {
    const response = await axios.get(`${API_BASE}/presets/${id}`);
    return response.data;
  },

  async create(preset: Omit<Preset, 'id' | 'is_system' | 'created_at' | 'updated_at'>): Promise<Preset> {
    const response = await axios.post(`${API_BASE}/presets/`, preset);
    return response.data;
  },

  async update(id: number, preset: Partial<Preset>): Promise<Preset> {
    const response = await axios.put(`${API_BASE}/presets/${id}`, preset);
    return response.data;
  },

  async delete(id: number): Promise<void> {
    await axios.delete(`${API_BASE}/presets/${id}`);
  },

  async merge(presetIds: number[], strategy: MergeStrategy = 'union'): Promise<MergeResult> {
    const response = await axios.post(`${API_BASE}/presets/merge`, {
      preset_ids: presetIds,
      merge_strategy: strategy
    });
    return response.data;
  },

  async importPreset(file: File): Promise<Preset> {
    const formData = new FormData();
    formData.append('file', file);
    const response = await axios.post(`${API_BASE}/presets/import`, formData, {
      headers: { 'Content-Type': 'multipart/form-data' }
    });
    return response.data;
  },

  async exportPreset(id: number): Promise<Blob> {
    const response = await axios.get(`${API_BASE}/presets/${id}/export`, {
      responseType: 'blob'
    });
    return response.data;
  }
};
