import axios, { AxiosInstance } from 'axios';
import type { Task, TaskCreate, Experiment, HealthResponse, SystemInfoResponse } from '@/types/api';

class ApiClient {
  private client: AxiosInstance;

  constructor() {
    this.client = axios.create({
      baseURL: import.meta.env.VITE_API_URL || '/api',
      headers: {
        'Content-Type': 'application/json',
      },
    });
  }

  // Health & System
  async getHealth(): Promise<HealthResponse> {
    const { data } = await this.client.get('/health');
    return data;
  }

  async getSystemInfo(): Promise<SystemInfoResponse> {
    const { data } = await this.client.get('/system/info');
    return data;
  }

  // Tasks
  async getTasks(): Promise<Task[]> {
    const { data } = await this.client.get('/tasks/');
    return data;
  }

  async getTask(id: number): Promise<Task> {
    const { data } = await this.client.get(`/tasks/${id}`);
    return data;
  }

  async getTaskByName(name: string): Promise<Task> {
    const { data } = await this.client.get(`/tasks/name/${name}`);
    return data;
  }

  async createTask(task: TaskCreate): Promise<Task> {
    const { data } = await this.client.post('/tasks/', task);
    return data;
  }

  async startTask(id: number): Promise<{ status: string; message: string; job_id: string }> {
    const { data } = await this.client.post(`/tasks/${id}/start`);
    return data;
  }

  async cancelTask(id: number): Promise<{ status: string; message: string }> {
    const { data } = await this.client.post(`/tasks/${id}/cancel`);
    return data;
  }

  // Experiments
  async getExperiments(): Promise<Experiment[]> {
    const { data } = await this.client.get('/experiments/');
    return data;
  }

  async getExperiment(id: number): Promise<Experiment> {
    const { data } = await this.client.get(`/experiments/${id}`);
    return data;
  }

  async getExperimentsByTask(taskId: number): Promise<Experiment[]> {
    const { data } = await this.client.get(`/experiments/task/${taskId}`);
    return data;
  }
}

export const apiClient = new ApiClient();
export default apiClient;
