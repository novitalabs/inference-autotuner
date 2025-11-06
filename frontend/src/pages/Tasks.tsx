import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { apiClient } from "@/services/api";
import type { Task } from "@/types/api";
import LogViewer from "@/components/LogViewer";
import TaskResults from "@/components/TaskResults";
import ExperimentProgressBar from "@/components/ExperimentProgressBar";
import { navigateTo } from "@/components/Layout";
import { setEditingTaskId } from "@/utils/editTaskStore";
import { useEscapeKey } from "@/hooks/useEscapeKey";

export default function Tasks() {
	const queryClient = useQueryClient();
	const [selectedTaskId, setSelectedTaskId] = useState<number | null>(null);
	const [showCreateForm, setShowCreateForm] = useState(false);
	const [statusFilter, setStatusFilter] = useState<string>("all");
	const [logViewerTask, setLogViewerTask] = useState<Task | null>(null);
	const [resultsTask, setResultsTask] = useState<Task | null>(null);

	// Fetch full task details when a task is selected
	const { data: selectedTask } = useQuery({
		queryKey: ["task", selectedTaskId],
		queryFn: () => selectedTaskId ? apiClient.getTask(selectedTaskId) : null,
		enabled: selectedTaskId !== null
	});

	// Fetch tasks
	const {
		data: tasks = [],
		isLoading,
		error
	} = useQuery({
		queryKey: ["tasks"],
		queryFn: () => apiClient.getTasks(),
		refetchInterval: 5000 // Auto-refresh every 5 seconds
	});

	// Start task mutation
	const startTaskMutation = useMutation({
		mutationFn: (taskId: number) => apiClient.startTask(taskId),
		onSuccess: () => {
			queryClient.invalidateQueries({ queryKey: ["tasks"] });
		}
	});

	// Cancel task mutation
	const cancelTaskMutation = useMutation({
		mutationFn: (taskId: number) => apiClient.cancelTask(taskId),
		onSuccess: () => {
			queryClient.invalidateQueries({ queryKey: ["tasks"] });
		}
	});

	// Restart task mutation
	const restartTaskMutation = useMutation({
		mutationFn: (taskId: number) => apiClient.restartTask(taskId),
		onSuccess: () => {
			queryClient.invalidateQueries({ queryKey: ["tasks"] });
		}
	});

	// Note: Create task mutation will be implemented when form is added
	// const createTaskMutation = useMutation({
	// 	mutationFn: (task: TaskCreate) => apiClient.createTask(task),
	// 	onSuccess: () => {
	// 		queryClient.invalidateQueries({ queryKey: ["tasks"] });
	// 		setShowCreateForm(false);
	// 	}
	// });

	// Filter tasks by status
	const filteredTasks =
		statusFilter === "all" ? tasks : tasks.filter((task: Task) => task.status === statusFilter);

	const getStatusColor = (status: string) => {
		switch (status) {
			case "completed":
				return "bg-green-100 text-green-800";
			case "running":
				return "bg-blue-100 text-blue-800";
			case "failed":
				return "bg-red-100 text-red-800";
			case "cancelled":
				return "bg-gray-100 text-gray-800";
			case "pending":
				return "bg-yellow-100 text-yellow-800";
			default:
				return "bg-gray-100 text-gray-800";
		}
	};

	const formatDuration = (seconds: number | null) => {
		if (!seconds) return "N/A";
		const hours = Math.floor(seconds / 3600);
		const mins = Math.floor((seconds % 3600) / 60);
		const secs = Math.floor(seconds % 60);
		if (hours > 0) return `${hours}h ${mins}m ${secs}s`;
		if (mins > 0) return `${mins}m ${secs}s`;
		return `${secs}s`;
	};

	const canStartTask = (task: Task) => {
		return task.status === "pending";
	};

	const canCancelTask = (task: Task) => {
		return task.status === "running";
	};

	const canRestartTask = (task: Task) => {
		return task.status === "completed" || task.status === "failed" || task.status === "cancelled";
	};

	return (
		<div className="px-4 py-6 sm:px-0">
			<div className="sm:flex sm:items-center">
				<div className="sm:flex-auto">
					<h1 className="text-2xl font-bold text-gray-900">Tasks</h1>
					<p className="mt-2 text-sm text-gray-700">
						Manage and monitor autotuning tasks for LLM inference optimization.
					</p>
				</div>
				<div className="mt-4 sm:mt-0 sm:ml-16 sm:flex-none flex items-center gap-3">
					<button
						onClick={() => navigateTo('quick-create')}
						className="inline-flex items-center justify-center rounded-md bg-blue-600 px-4 py-2 text-sm font-semibold text-white shadow-sm hover:bg-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2"
					>
						<svg
							className="h-5 w-5 mr-2"
							fill="none"
							viewBox="0 0 24 24"
							stroke="currentColor"
						>
							<path
								strokeLinecap="round"
								strokeLinejoin="round"
								strokeWidth={2}
								d="M13 10V3L4 14h7v7l9-11h-7z"
							/>
						</svg>
						Quick Create
					</button>
					<button
						onClick={() => navigateTo('new-task')}
						className="inline-flex items-center justify-center rounded-md bg-white border border-gray-300 px-4 py-2 text-sm font-semibold text-gray-700 shadow-sm hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2"
					>
						<svg
							className="h-5 w-5 mr-2"
							fill="none"
							viewBox="0 0 24 24"
							stroke="currentColor"
						>
							<path
								strokeLinecap="round"
								strokeLinejoin="round"
								strokeWidth={2}
								d="M12 4v16m8-8H4"
							/>
						</svg>
						Advanced
					</button>
				</div>
			</div>

			{/* Filter Section */}
			<div className="mt-6 flex items-center gap-4">
				<label htmlFor="status-filter" className="text-sm font-medium text-gray-700">
					Filter by Status:
				</label>
				<select
					id="status-filter"
					className="rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500 sm:text-sm"
					value={statusFilter}
					onChange={(e) => setStatusFilter(e.target.value)}
				>
					<option value="all">All Tasks</option>
					<option value="pending">Pending</option>
					<option value="running">Running</option>
					<option value="completed">Completed</option>
					<option value="failed">Failed</option>
					<option value="cancelled">Cancelled</option>
				</select>
				<span className="text-sm text-gray-500">
					{filteredTasks.length} task{filteredTasks.length !== 1 ? "s" : ""}
				</span>
			</div>

			{/* Tasks Table */}
			<div className="mt-6 flow-root">
				{isLoading ? (
					<div className="text-center py-12">
						<div className="inline-block h-8 w-8 animate-spin rounded-full border-4 border-solid border-current border-r-transparent"></div>
						<p className="mt-2 text-sm text-gray-600">Loading tasks...</p>
					</div>
				) : error ? (
					<div className="rounded-md bg-red-50 p-4">
						<p className="text-sm text-red-800">
							Error loading tasks: {(error as Error).message}
						</p>
					</div>
				) : filteredTasks.length === 0 ? (
					<div className="text-center py-12 border-2 border-dashed border-gray-300 rounded-lg">
						<svg
							className="mx-auto h-12 w-12 text-gray-400"
							fill="none"
							viewBox="0 0 24 24"
							stroke="currentColor"
						>
							<path
								strokeLinecap="round"
								strokeLinejoin="round"
								strokeWidth={2}
								d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2"
							/>
						</svg>
						<h3 className="mt-2 text-sm font-medium text-gray-900">No tasks</h3>
						<p className="mt-1 text-sm text-gray-500">
							{statusFilter === "all"
								? "Get started by creating a new task."
								: `No ${statusFilter} tasks found.`}
						</p>
						{statusFilter === "all" && (
							<div className="mt-6">
								<button
									onClick={() => setShowCreateForm(true)}
									className="inline-flex items-center px-4 py-2 border border-transparent shadow-sm text-sm font-medium rounded-md text-white bg-blue-600 hover:bg-blue-700"
								>
									<svg
										className="h-5 w-5 mr-2"
										fill="none"
										viewBox="0 0 24 24"
										stroke="currentColor"
									>
										<path
											strokeLinecap="round"
											strokeLinejoin="round"
											strokeWidth={2}
											d="M12 4v16m8-8H4"
										/>
									</svg>
									Create Task
								</button>
							</div>
						)}
					</div>
				) : (
					<div className="overflow-x-auto">
						<table className="min-w-full divide-y divide-gray-300">
							<thead>
								<tr>
									<th className="py-3.5 pl-4 pr-3 text-left text-sm font-semibold text-gray-900">
										Task Name
									</th>
									<th className="px-3 py-3.5 text-left text-sm font-semibold text-gray-900">
										Status
									</th>
									<th className="px-3 py-3.5 text-left text-sm font-semibold text-gray-900">
										Runtime
									</th>
									<th className="px-3 py-3.5 text-left text-sm font-semibold text-gray-900">
										Experiments
									</th>
									<th className="px-3 py-3.5 text-left text-sm font-semibold text-gray-900">
										Duration
									</th>
									<th className="px-3 py-3.5 text-left text-sm font-semibold text-gray-900">
										Created
									</th>
									<th className="relative py-3.5 pl-3 pr-4">
										<span className="sr-only">Actions</span>
									</th>
								</tr>
							</thead>
							<tbody className="divide-y divide-gray-200 bg-white">
								{filteredTasks.map((task: Task) => (
									<tr key={task.id} className="hover:bg-gray-50">
										<td className="whitespace-nowrap py-4 pl-4 pr-3 text-sm">
											<div className="font-medium text-gray-900">
												{task.task_name}
											</div>
											{task.description && (
												<div className="text-gray-500 max-w-xs truncate">
													{task.description}
												</div>
											)}
										</td>
										<td className="whitespace-nowrap px-3 py-4 text-sm">
											<span
												className={`inline-flex rounded-full px-2 text-xs font-semibold leading-5 ${getStatusColor(task.status)}`}
											>
												{task.status}
											</span>
										</td>
										<td className="whitespace-nowrap px-3 py-4 text-sm text-gray-700">
											{task.base_runtime}
										</td>
										<td className="whitespace-nowrap px-3 py-4 text-sm text-gray-700">
											<ExperimentProgressBar
												taskId={task.id}
												totalExperiments={task.total_experiments}
												successfulExperiments={task.successful_experiments}
											/>
										</td>
										<td className="whitespace-nowrap px-3 py-4 text-sm text-gray-700">
											{formatDuration(task.elapsed_time)}
										</td>
										<td className="whitespace-nowrap px-3 py-4 text-sm text-gray-700">
											{new Date(task.created_at).toLocaleDateString()}
										</td>
										<td className="relative whitespace-nowrap py-4 pl-3 pr-4 text-right text-sm">
											<div className="flex items-center justify-end gap-1">
												{/* View Button */}
												<button
													onClick={() => setSelectedTaskId(task.id)}
													className="p-1.5 text-blue-600 hover:text-blue-900 hover:bg-blue-50 rounded transition-colors"
													title="View Details"
												>
													<svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
														<path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
														<path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
													</svg>
												</button>

												{/* Edit Button - For pending, cancelled, and failed tasks */}
											{(task.status === 'pending' || task.status === 'cancelled' || task.status === 'failed') && (
												<button
													onClick={() => {
															setEditingTaskId(task.id);
															navigateTo('new-task');
														}}
													className="p-1.5 text-gray-600 hover:text-gray-900 hover:bg-gray-50 rounded transition-colors"
													title="Edit Task"
												>
													<svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
														<path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z" />
													</svg>
												</button>
											)}

												{/* Logs Button */}
												<button
													onClick={() => setLogViewerTask(task)}
													className="p-1.5 text-purple-600 hover:text-purple-900 hover:bg-purple-50 rounded transition-colors"
													title="View Logs"
												>
													<svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
														<path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
													</svg>
												</button>

												{/* Results Button */}
												{task.status === 'completed' && task.successful_experiments > 0 && (
													<button
														onClick={() => setResultsTask(task)}
														className="p-1.5 text-emerald-600 hover:text-emerald-900 hover:bg-emerald-50 rounded transition-colors"
														title="View Results"
													>
														<svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
															<path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
														</svg>
													</button>
												)}

												{/* Start Button */}
												{canStartTask(task) && (
													<button
														onClick={() =>
															startTaskMutation.mutate(task.id)
														}
														disabled={startTaskMutation.isPending}
														className="p-1.5 text-green-600 hover:text-green-900 hover:bg-green-50 rounded transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
														title="Start Task"
													>
														<svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
															<path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z" />
															<path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
														</svg>
													</button>
												)}

												{/* Cancel Button */}
												{canCancelTask(task) && (
													<button
														onClick={() =>
															cancelTaskMutation.mutate(task.id)
														}
														disabled={cancelTaskMutation.isPending}
														className="p-1.5 text-red-600 hover:text-red-900 hover:bg-red-50 rounded transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
														title="Cancel Task"
													>
														<svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
															<path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 14l2-2m0 0l2-2m-2 2l-2-2m2 2l2 2m7-2a9 9 0 11-18 0 9 9 0 0118 0z" />
														</svg>
													</button>
												)}

												{/* Restart Button */}
												{canRestartTask(task) && (
													<button
														onClick={() => {
															// Only confirm for completed tasks, not for failed/cancelled
															if (task.status === "completed") {
																if (
																	confirm(
																		`Are you sure you want to restart task "${task.task_name}"? This will reset the task to PENDING status and clear all previous results.`
																	)
																) {
																	restartTaskMutation.mutate(task.id);
																}
															} else {
																// Directly restart failed/cancelled tasks
																restartTaskMutation.mutate(task.id);
															}
														}}
														disabled={restartTaskMutation.isPending}
														className="p-1.5 text-orange-600 hover:text-orange-900 hover:bg-orange-50 rounded transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
														title="Restart Task"
													>
														<svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
															<path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
														</svg>
													</button>
												)}
											</div>
										</td>
									</tr>
								))}
							</tbody>
						</table>
					</div>
				)}
			</div>

			{/* Task Detail Modal */}
			{selectedTask && (
				<TaskDetailModal task={selectedTask} onClose={() => setSelectedTaskId(null)} />
			)}

			{/* Log Viewer Modal */}
			{logViewerTask && (
				<LogViewer
					taskId={logViewerTask.id}
					taskName={logViewerTask.task_name}
					onClose={() => setLogViewerTask(null)}
				/>
			)}

			{/* Task Results Modal */}
			{resultsTask && (
				<TaskResults
					task={resultsTask}
					onClose={() => setResultsTask(null)}
				/>
			)}

			{/* Edit Task Modal */}
			{/*editTask && (
				<EditTaskModal
					task={editTask}
					onClose={() => setEditTaskId(null)}
					onUpdate={(updates) => updateTaskMutation.mutate({ taskId: editTask.id, updates })}
					isUpdating={updateTaskMutation.isPending}
				/>
			)*/}

						{/* Create Task Modal */}
			{showCreateForm && <CreateTaskModal onClose={() => setShowCreateForm(false)} />}
		</div>
	);
}

// Task Detail Modal Component
function TaskDetailModal({ task, onClose }: { task: Task; onClose: () => void }) {
	const formatDuration = (seconds: number | null) => {
		if (!seconds) return "N/A";
		const hours = Math.floor(seconds / 3600);
		const mins = Math.floor((seconds % 3600) / 60);
		const secs = Math.floor(seconds % 60);
		if (hours > 0) return `${hours}h ${mins}m ${secs}s`;
		if (mins > 0) return `${mins}m ${secs}s`;
		return `${secs}s`;
	};

	return (
		<div className="fixed inset-0 bg-gray-500 bg-opacity-75 flex items-center justify-center p-4 z-50">
			<div className="bg-white rounded-lg shadow-xl max-w-4xl w-full max-h-[90vh] overflow-y-auto">
				<div className="px-6 py-4 border-b border-gray-200">
					<div className="flex items-center justify-between">
						<h2 className="text-xl font-bold text-gray-900">{task.task_name}</h2>
						<button onClick={onClose} className="text-gray-400 hover:text-gray-500">
							<svg
								className="h-6 w-6"
								fill="none"
								viewBox="0 0 24 24"
								stroke="currentColor"
							>
								<path
									strokeLinecap="round"
									strokeLinejoin="round"
									strokeWidth={2}
									d="M6 18L18 6M6 6l12 12"
								/>
							</svg>
						</button>
					</div>
				</div>

				<div className="px-6 py-4 space-y-6">
					{/* Basic Info */}
					<div>
						<h3 className="text-sm font-medium text-gray-900 mb-3">
							Basic Information
						</h3>
						<div className="grid grid-cols-2 gap-4">
							<div>
								<span className="text-sm text-gray-500">Status:</span>
								<span className="ml-2 text-sm text-gray-900">{task.status}</span>
							</div>
							<div>
								<span className="text-sm text-gray-500">Runtime:</span>
								<span className="ml-2 text-sm text-gray-900">
									{task.base_runtime}
								</span>
							</div>
							<div>
								<span className="text-sm text-gray-500">Deployment Mode:</span>
								<span className="ml-2 text-sm text-gray-900">
									{task.deployment_mode}
								</span>
							</div>
							<div>
								<span className="text-sm text-gray-500">Duration:</span>
								<span className="ml-2 text-sm text-gray-900">
									{formatDuration(task.elapsed_time)}
								</span>
							</div>
							<div>
								<span className="text-sm text-gray-500">Total Experiments:</span>
								<span className="ml-2 text-sm text-gray-900">
									{task.total_experiments}
								</span>
							</div>
							<div>
								<span className="text-sm text-gray-500">Successful:</span>
								<span className="ml-2 text-sm text-gray-900">
									{task.successful_experiments}
								</span>
							</div>
						</div>
						{task.description && (
							<div className="mt-4">
								<span className="text-sm text-gray-500">Description:</span>
								<p className="mt-1 text-sm text-gray-900">{task.description}</p>
							</div>
						)}
					</div>

					{/* Model Config */}
					<div>
						<h3 className="text-sm font-medium text-gray-900 mb-3">
							Model Configuration
						</h3>
						<div className="bg-gray-50 rounded-lg p-4">
							<pre className="text-sm text-gray-900 overflow-x-auto">
								{JSON.stringify(task.model, null, 2)}
							</pre>
						</div>
					</div>

					{/* Parameters */}
					<div>
						<h3 className="text-sm font-medium text-gray-900 mb-3">
							Tuning Parameters
						</h3>
						<div className="bg-gray-50 rounded-lg p-4">
							<pre className="text-sm text-gray-900 overflow-x-auto">
								{JSON.stringify(task.parameters, null, 2)}
							</pre>
						</div>
					</div>

					{/* Optimization Config */}
					<div>
						<h3 className="text-sm font-medium text-gray-900 mb-3">
							Optimization Configuration
						</h3>
						<div className="bg-gray-50 rounded-lg p-4">
							<pre className="text-sm text-gray-900 overflow-x-auto">
								{JSON.stringify(task.optimization, null, 2)}
							</pre>
						</div>
					</div>

					{/* Benchmark Config */}
					<div>
						<h3 className="text-sm font-medium text-gray-900 mb-3">
							Benchmark Configuration
						</h3>
						<div className="bg-gray-50 rounded-lg p-4">
							<pre className="text-sm text-gray-900 overflow-x-auto">
								{JSON.stringify(task.benchmark, null, 2)}
							</pre>
						</div>
					</div>

					{/* Timeline */}
					<div>
						<h3 className="text-sm font-medium text-gray-900 mb-3">Timeline</h3>
						<div className="space-y-2 text-sm">
							<div>
								<span className="text-gray-500">Created:</span>
								<span className="ml-2 text-gray-900">
									{new Date(task.created_at).toLocaleString()}
								</span>
							</div>
							{task.started_at && (
								<div>
									<span className="text-gray-500">Started:</span>
									<span className="ml-2 text-gray-900">
										{new Date(task.started_at).toLocaleString()}
									</span>
								</div>
							)}
							{task.completed_at && (
								<div>
									<span className="text-gray-500">Completed:</span>
									<span className="ml-2 text-gray-900">
										{new Date(task.completed_at).toLocaleString()}
									</span>
								</div>
							)}
						</div>
					</div>
				</div>

				<div className="px-6 py-4 border-t border-gray-200 bg-gray-50">
					<button
						onClick={onClose}
						className="w-full sm:w-auto px-4 py-2 bg-gray-200 text-gray-700 rounded-md hover:bg-gray-300"
					>
						Close
					</button>
				</div>
			</div>
		</div>
	);
}

// Create Task Modal Component
function CreateTaskModal({ onClose }: { onClose: () => void }) {
	// Handle Escape key to close modal
	useEscapeKey(onClose);

	return (
		<div className="fixed inset-0 bg-gray-500 bg-opacity-75 flex items-center justify-center p-4 z-50">
			<div className="bg-white rounded-lg shadow-xl max-w-2xl w-full max-h-[90vh] overflow-y-auto">
				<div className="px-6 py-4 border-b border-gray-200">
					<div className="flex items-center justify-between">
						<h2 className="text-xl font-bold text-gray-900">Create New Task</h2>
						<button onClick={onClose} className="text-gray-400 hover:text-gray-500">
							<svg
								className="h-6 w-6"
								fill="none"
								viewBox="0 0 24 24"
								stroke="currentColor"
							>
								<path
									strokeLinecap="round"
									strokeLinejoin="round"
									strokeWidth={2}
									d="M6 18L18 6M6 6l12 12"
								/>
							</svg>
						</button>
					</div>
				</div>

				<div className="px-6 py-4">
					<p className="text-sm text-gray-600">
						Task creation form will be implemented here. For now, please use the API or
						backend interface to create tasks.
					</p>
				</div>

				<div className="px-6 py-4 border-t border-gray-200 bg-gray-50">
					<button
						onClick={onClose}
						className="w-full sm:w-auto px-4 py-2 bg-gray-200 text-gray-700 rounded-md hover:bg-gray-300"
					>
						Close
					</button>
				</div>
			</div>
		</div>
	);
}
