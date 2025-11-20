// Dashboard API client

import axios from 'axios';
import type {
	GPUStatus,
	WorkerStatus,
	DBStatistics,
	ExperimentTimelineItem,
} from '../types/dashboard';

const API_BASE_URL = 'http://localhost:8000';

export interface ClusterGPUStatus {
	available: boolean;
	mode: 'cluster';
	nodes?: Array<{
		index: number;
		node_name: string;
		name: string;
		capacity: number;
		allocatable: number;
		has_metrics?: boolean;
		memory_total_mb?: number;
		memory_used_mb?: number;
		memory_free_mb?: number;
		memory_usage_percent?: number;
		utilization_percent?: number;
		temperature_c?: number;
	}>;
	total_gpus?: number;
	total_allocatable_gpus?: number;
	error?: string;
	timestamp: string;
}

export const dashboardApi = {
	async getGPUStatus(): Promise<GPUStatus> {
		const response = await axios.get<GPUStatus>(`${API_BASE_URL}/api/dashboard/gpu-status`);
		return response.data;
	},

	async getClusterGPUStatus(): Promise<ClusterGPUStatus> {
		const response = await axios.get<ClusterGPUStatus>(`${API_BASE_URL}/api/dashboard/cluster-gpu-status`);
		return response.data;
	},

	async getWorkerStatus(): Promise<WorkerStatus> {
		const response = await axios.get<WorkerStatus>(`${API_BASE_URL}/api/dashboard/worker-status`);
		return response.data;
	},

	async getDBStatistics(): Promise<DBStatistics> {
		const response = await axios.get<DBStatistics>(`${API_BASE_URL}/api/dashboard/db-statistics`);
		return response.data;
	},

	async getExperimentTimeline(hours: number = 24): Promise<ExperimentTimelineItem[]> {
		const response = await axios.get<ExperimentTimelineItem[]>(
			`${API_BASE_URL}/api/dashboard/experiment-timeline`,
			{ params: { hours } }
		);
		return response.data;
	},
};
