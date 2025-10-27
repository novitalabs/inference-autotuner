import axios, { AxiosInstance, AxiosError } from "axios";
import toast from "react-hot-toast";
import type { Task, TaskCreate, Experiment, HealthResponse, SystemInfoResponse } from "@/types/api";

class ApiClient {
	private client: AxiosInstance;

	constructor() {
		this.client = axios.create({
			baseURL: import.meta.env.VITE_API_URL || "/api",
			headers: {
				"Content-Type": "application/json"
			}
		});

		// Response interceptor to handle errors
		this.client.interceptors.response.use(
			(response) => response,
			(error: AxiosError) => {
				// Handle HTTP errors (4xx, 5xx)
				if (error.response) {
					const status = error.response.status;
					const data = error.response.data as any;

					// Extract error message
					const message =
						data?.detail || data?.message || error.message || "An error occurred";

					// Show error toast based on status code
					if (status >= 400 && status < 500) {
						// Client errors (4xx)
						toast.error(`${status}: ${message}`, {
							duration: 4000,
							position: "bottom-right"
						});
					} else if (status >= 500) {
						// Server errors (5xx)
						toast.error(`Server Error: ${message}`, {
							duration: 5000,
							position: "bottom-right"
						});
					}
				} else if (error.request) {
					// Network error (no response received)
					toast.error("Network error: Unable to reach the server", {
						duration: 4000,
						position: "bottom-right"
					});
				}

				return Promise.reject(error);
			}
		);
	}

	// Health & System
	async getHealth(): Promise<HealthResponse> {
		const { data } = await this.client.get("/health");
		return data;
	}

	async getSystemInfo(): Promise<SystemInfoResponse> {
		const { data } = await this.client.get("/system/info");
		return data;
	}

	// Tasks
	async getTasks(): Promise<Task[]> {
		const { data } = await this.client.get("/tasks/");
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
		const { data } = await this.client.post("/tasks/", task);
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
		const { data } = await this.client.get("/experiments/");
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

	// Task Logs
	async getTaskLogs(taskId: number): Promise<{ logs: string }> {
		const { data } = await this.client.get(`/tasks/${taskId}/logs`);
		return data;
	}

	async clearTaskLogs(taskId: number): Promise<void> {
		await this.client.delete(`/tasks/${taskId}/logs`);
	}
}

export const apiClient = new ApiClient();
export default apiClient;
