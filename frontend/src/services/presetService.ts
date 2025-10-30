// Preset service for API calls
import axios from 'axios';

const API_BASE = 'http://localhost:8000';

export interface Preset {
  id: number;
  name: string;
  description?: string;
  category?: string;
  is_system: boolean;
  parameters: Record<string, any[]>;
  metadata?: Record<string, any>;
  created_at: string;
  updated_at?: string;
}

export interface MergeResult {
  parameters: Record<string, any[]>;
  applied_presets: string[];
  conflicts?: Array<{
    parameter: string;
    reason: string;
    [key: string]: any;
  }>;
}

export type MergeStrategy = 'union' | 'intersection' | 'last_wins';

export const presetService = {
  async getAll(category?: string): Promise<Preset[]> {
    const params = category ? { category } : {};
    const response = await axios.get(`${API_BASE}/api/presets/`, { params });
    return response.data;
  },

  async getById(id: number): Promise<Preset> {
    const response = await axios.get(`${API_BASE}/api/presets/${id}`);
    return response.data;
  },

  async create(preset: Omit<Preset, 'id' | 'is_system' | 'created_at' | 'updated_at'>): Promise<Preset> {
    const response = await axios.post(`${API_BASE}/api/presets/`, preset);
    return response.data;
  },

  async update(id: number, preset: Partial<Preset>): Promise<Preset> {
    const response = await axios.put(`${API_BASE}/api/presets/${id}`, preset);
    return response.data;
  },

  async delete(id: number): Promise<void> {
    await axios.delete(`${API_BASE}/api/presets/${id}`);
  },

  async merge(presetIds: number[], strategy: MergeStrategy = 'union'): Promise<MergeResult> {
    const response = await axios.post(`${API_BASE}/api/presets/merge`, {
      preset_ids: presetIds,
      merge_strategy: strategy
    });
    return response.data;
  },

  async importPreset(file: File): Promise<Preset> {
    const formData = new FormData();
    formData.append('file', file);
    const response = await axios.post(`${API_BASE}/api/presets/import`, formData, {
      headers: { 'Content-Type': 'multipart/form-data' }
    });
    return response.data;
  },

  async exportPreset(id: number): Promise<Blob> {
    const response = await axios.get(`${API_BASE}/api/presets/${id}/export`, {
      responseType: 'blob'
    });
    return response.data;
  }
};
