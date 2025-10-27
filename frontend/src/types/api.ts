// API Response Types based on backend schemas

export interface Task {
  id: number;
  task_name: string;
  description: string | null;
  status: TaskStatus;
  model_config: Record<string, any>;
  base_runtime: string;
  runtime_image_tag: string | null;
  parameters: Record<string, any>;
  optimization_config: Record<string, any>;
  benchmark_config: Record<string, any>;
  deployment_mode: string;
  total_experiments: number;
  successful_experiments: number;
  best_experiment_id: number | null;
  created_at: string;
  started_at: string | null;
  completed_at: string | null;
  elapsed_time: number | null;
}

export enum TaskStatus {
  PENDING = 'pending',
  RUNNING = 'running',
  COMPLETED = 'completed',
  FAILED = 'failed',
  CANCELLED = 'cancelled',
}

export interface Experiment {
  id: number;
  task_id: number;
  experiment_id: number;
  parameters: Record<string, any>;
  status: ExperimentStatus;
  error_message: string | null;
  metrics: Record<string, any> | null;
  objective_score: number | null;
  service_name: string | null;
  service_url: string | null;
  created_at: string;
  started_at: string | null;
  completed_at: string | null;
  elapsed_time: number | null;
}

export enum ExperimentStatus {
  PENDING = 'pending',
  DEPLOYING = 'deploying',
  BENCHMARKING = 'benchmarking',
  SUCCESS = 'success',
  FAILED = 'failed',
}

export interface HealthResponse {
  status: string;
  database?: string;
  redis?: string;
}

export interface SystemInfoResponse {
  app_name: string;
  version: string;
  deployment_mode: string;
  available_runtimes: string[];
}

// Request Types
export interface TaskCreate {
  task_name: string;
  description?: string;
  model_config: Record<string, any>;
  base_runtime: string;
  runtime_image_tag?: string;
  parameters: Record<string, any>;
  optimization_config: Record<string, any>;
  benchmark_config: Record<string, any>;
  deployment_mode?: string;
}
