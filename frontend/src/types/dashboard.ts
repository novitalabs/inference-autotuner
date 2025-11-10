// Dashboard API types

export interface GPUInfo {
	index: number;
	name: string;
	memory_total_mb: number;
	memory_used_mb: number;
	memory_free_mb: number;
	utilization_percent: number;
	temperature_c: number;
	memory_usage_percent: number;
}

export interface GPUStatus {
	available: boolean;
	gpus?: GPUInfo[];
	error?: string;
	timestamp: string;
}

export interface WorkerStatus {
	worker_running: boolean;
	worker_pid: number | null;
	worker_cpu_percent: number;
	worker_memory_mb: number;
	worker_uptime_seconds: number | null;
	redis_available: boolean;
	timestamp: string;
	error?: string;
}

export interface RunningTaskInfo {
	id: number;
	name: string;
	started_at: string | null;
	max_iterations: number;
	completed_experiments: number;
}

export interface DBStatistics {
	total_tasks: number;
	total_experiments: number;
	tasks_by_status: Record<string, number>;
	experiments_by_status: Record<string, number>;
	tasks_last_24h: number;
	experiments_last_24h: number;
	avg_experiment_duration_seconds: number | null;
	running_tasks: RunningTaskInfo[];
	timestamp: string;
}

export interface ExperimentTimelineItem {
	id: number;
	task_id: number;
	experiment_id: number;
	status: string;
	created_at: string | null;
	started_at: string | null;
	completed_at: string | null;
	elapsed_time: number | null;
	objective_score: number | null;
}
