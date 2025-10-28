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
	PENDING = "pending",
	RUNNING = "running",
	COMPLETED = "completed",
	FAILED = "failed",
	CANCELLED = "cancelled"
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
	PENDING = "pending",
	DEPLOYING = "deploying",
	BENCHMARKING = "benchmarking",
	SUCCESS = "success",
	FAILED = "failed"
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

// Docker Types
export interface ContainerInfo {
	id: string;
	short_id: string;
	name: string;
	image: string;
	status: string;
	state: string;
	created: string;
	started_at: string | null;
	finished_at: string | null;
	ports: Record<string, string>;
	labels: Record<string, string>;
	command: string | null;
}

export interface ContainerStats {
	cpu_percent: number;
	memory_usage: string;
	memory_limit: string;
	memory_percent: number;
	network_rx: string;
	network_tx: string;
	block_read: string;
	block_write: string;
}

export interface ContainerLogs {
	logs: string;
	lines: number;
}

export interface DockerInfo {
	version: string;
	api_version: string;
	containers: number;
	containers_running: number;
	containers_paused: number;
	containers_stopped: number;
	images: number;
	driver: string;
	memory_total: string;
	cpus: number;
	operating_system: string;
	architecture: string;
}
