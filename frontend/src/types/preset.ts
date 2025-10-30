// TypeScript types for parameter presets

export type Runtime = 'sglang' | 'vllm';

export interface Preset {
  id: number;
  name: string;
  description?: string;
  category?: string;
  runtime?: Runtime;  // Optional: sglang, vllm, or undefined for universal
  is_system: boolean;
  parameters: ParameterMap;
  metadata?: PresetMetadata;
  created_at: string;
  updated_at?: string;
}

export interface ParameterMap {
  [key: string]: any[];
}

export interface PresetMetadata {
  author?: string;
  tags?: string[];
  recommended_for?: string[];
  [key: string]: any;
}

export interface MergeResult {
  parameters: ParameterMap;
  applied_presets: string[];
  conflicts?: Conflict[];
}

export interface Conflict {
  parameter: string;
  reason: string;
  overridden_by?: string;
  previous_values?: any[];
  new_values?: any[];
}

export type MergeStrategy = 'union' | 'intersection' | 'last_wins';

export interface PresetCreate {
  name: string;
  description?: string;
  category?: string;
  runtime?: Runtime;
  parameters: ParameterMap;
  metadata?: PresetMetadata;
}

export interface PresetUpdate {
  name?: string;
  description?: string;
  category?: string;
  runtime?: Runtime;
  parameters?: ParameterMap;
  metadata?: PresetMetadata;
}
