/**
 * Service for runtime parameter information
 */

import axios from 'axios';

const API_BASE = 'http://localhost:8000';

export interface ParameterListResponse {
  runtime: string;
  count: number;
  parameters: string[];
}

export interface CommonlyTunedResponse {
  runtime: string;
  parameters: string[];
}

export interface ParameterCompatibilityResponse {
  common: string[];
  sglang_only: string[];
  vllm_only: string[];
  stats: {
    common_count: number;
    sglang_only_count: number;
    vllm_only_count: number;
    sglang_total: number;
    vllm_total: number;
  };
}

export interface ParameterCounts {
  sglang_count: number;
  vllm_count: number;
  common_count: number;
}

export const runtimeParamsService = {
  /**
   * Get parameter counts for each runtime
   */
  async getCounts(): Promise<ParameterCounts> {
    const response = await axios.get(`${API_BASE}/api/runtime-params/`);
    return response.data;
  },

  /**
   * Get all parameters for a specific runtime
   */
  async getParameters(
    runtime: 'sglang' | 'vllm',
    commonlyTunedOnly: boolean = false
  ): Promise<ParameterListResponse> {
    const response = await axios.get(
      `${API_BASE}/api/runtime-params/${runtime}`,
      {
        params: { commonly_tuned_only: commonlyTunedOnly },
      }
    );
    return response.data;
  },

  /**
   * Get commonly tuned parameters for optimization
   */
  async getCommonlyTuned(runtime: 'sglang' | 'vllm'): Promise<CommonlyTunedResponse> {
    const response = await axios.get(
      `${API_BASE}/api/runtime-params/${runtime}/commonly-tuned`
    );
    return response.data;
  },

  /**
   * Get parameter compatibility information
   */
  async getCompatibility(): Promise<ParameterCompatibilityResponse> {
    const response = await axios.get(`${API_BASE}/api/runtime-params/compatibility`);
    return response.data;
  },

  /**
   * Validate if a parameter is valid for a given runtime
   */
  async validateParameter(runtime: 'sglang' | 'vllm', parameter: string): Promise<boolean> {
    try {
      const response = await axios.post(`${API_BASE}/api/runtime-params/validate`, {
        runtime,
        parameter,
      });
      return response.data.is_valid;
    } catch (error) {
      return false;
    }
  },
};
