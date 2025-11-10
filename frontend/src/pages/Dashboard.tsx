import { useQuery } from '@tanstack/react-query';
import { dashboardApi } from '../services/dashboardApi';
import {
	CpuChipIcon,
	ServerIcon,
	CircleStackIcon,
	ClockIcon,
} from '@heroicons/react/24/outline';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

function formatBytes(mb: number): string {
	if (mb >= 1024) {
		return `${(mb / 1024).toFixed(1)} GB`;
	}
	return `${mb.toFixed(0)} MB`;
}

function formatUptime(seconds: number | null): string {
	if (!seconds) return 'N/A';
	const hours = Math.floor(seconds / 3600);
	const minutes = Math.floor((seconds % 3600) / 60);
	return `${hours}h ${minutes}m`;
}

export default function Dashboard() {
	// Fetch dashboard data with auto-refresh
	const { data: gpuStatus, isLoading: gpuLoading } = useQuery({
		queryKey: ['gpuStatus'],
		queryFn: dashboardApi.getGPUStatus,
		refetchInterval: 5000, // Refresh every 5 seconds
	});

	const { data: workerStatus, isLoading: workerLoading } = useQuery({
		queryKey: ['workerStatus'],
		queryFn: dashboardApi.getWorkerStatus,
		refetchInterval: 5000,
	});

	const { data: dbStats, isLoading: dbStatsLoading } = useQuery({
		queryKey: ['dbStatistics'],
		queryFn: dashboardApi.getDBStatistics,
		refetchInterval: 10000, // Refresh every 10 seconds
	});

	const { data: timeline, isLoading: timelineLoading } = useQuery({
		queryKey: ['experimentTimeline', 24],
		queryFn: () => dashboardApi.getExperimentTimeline(24),
		refetchInterval: 30000, // Refresh every 30 seconds
	});

	// GPU Status Card
	const renderGPUCard = () => (
		<div className="bg-white shadow rounded-lg p-6">
			<div className="flex items-center justify-between mb-4">
				<h3 className="text-lg font-medium text-gray-900 flex items-center">
					<CpuChipIcon className="h-6 w-6 mr-2 text-blue-500" />
					GPU Status
				</h3>
				{gpuStatus && (
					<span className={`px-2 py-1 text-xs rounded-full ${gpuStatus.available ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'}`}>
						{gpuStatus.available ? 'Available' : 'Unavailable'}
					</span>
				)}
			</div>

			{gpuLoading && <p className="text-gray-500">Loading...</p>}
			{gpuStatus?.error && <p className="text-red-500">Error: {gpuStatus.error}</p>}

			{gpuStatus?.gpus && (
				<div className="space-y-4">
					{gpuStatus.gpus.map((gpu) => (
						<div key={gpu.index} className="border rounded p-3">
							<div className="flex justify-between items-start mb-2">
								<div>
									<p className="font-medium">GPU {gpu.index}: {gpu.name}</p>
									<p className="text-sm text-gray-500">{gpu.temperature_c}°C</p>
								</div>
								<span className={`px-2 py-1 text-xs rounded ${gpu.utilization_percent > 80 ? 'bg-red-100 text-red-800' : gpu.utilization_percent > 50 ? 'bg-yellow-100 text-yellow-800' : 'bg-green-100 text-green-800'}`}>
									{gpu.utilization_percent}% Util
								</span>
							</div>

							{/* Memory Usage Bar */}
							<div className="mt-2">
								<div className="flex justify-between text-sm text-gray-600 mb-1">
									<span>Memory</span>
									<span>{formatBytes(gpu.memory_used_mb)} / {formatBytes(gpu.memory_total_mb)}</span>
								</div>
								<div className="w-full bg-gray-200 rounded-full h-2.5">
									<div
										className={`h-2.5 rounded-full ${gpu.memory_usage_percent > 90 ? 'bg-red-600' : gpu.memory_usage_percent > 70 ? 'bg-yellow-500' : 'bg-blue-600'}`}
										style={{ width: `${gpu.memory_usage_percent}%` }}
									></div>
								</div>
								<p className="text-xs text-gray-500 mt-1">{gpu.memory_usage_percent.toFixed(1)}% used</p>
							</div>
						</div>
					))}
				</div>
			)}
		</div>
	);

	// Worker Status Card
	const renderWorkerCard = () => (
		<div className="bg-white shadow rounded-lg p-6">
			<div className="flex items-center justify-between mb-4">
				<h3 className="text-lg font-medium text-gray-900 flex items-center">
					<ServerIcon className="h-6 w-6 mr-2 text-purple-500" />
					ARQ Worker
				</h3>
				{workerStatus && (
					<span className={`px-2 py-1 text-xs rounded-full ${workerStatus.worker_running ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'}`}>
						{workerStatus.worker_running ? 'Running' : 'Stopped'}
					</span>
				)}
			</div>

			{workerLoading && <p className="text-gray-500">Loading...</p>}
			{workerStatus?.error && <p className="text-red-500">Error: {workerStatus.error}</p>}

			{workerStatus && (
				<div className="space-y-3">
					<div className="flex justify-between">
						<span className="text-gray-600">Process ID:</span>
						<span className="font-medium">{workerStatus.worker_pid || 'N/A'}</span>
					</div>
					<div className="flex justify-between">
						<span className="text-gray-600">CPU Usage:</span>
						<span className="font-medium">{workerStatus.worker_cpu_percent.toFixed(1)}%</span>
					</div>
					<div className="flex justify-between">
						<span className="text-gray-600">Memory:</span>
						<span className="font-medium">{workerStatus.worker_memory_mb.toFixed(1)} MB</span>
					</div>
					<div className="flex justify-between">
						<span className="text-gray-600">Uptime:</span>
						<span className="font-medium">{formatUptime(workerStatus.worker_uptime_seconds)}</span>
					</div>
					<div className="flex justify-between">
						<span className="text-gray-600">Redis:</span>
						<span className={`font-medium ${workerStatus.redis_available ? 'text-green-600' : 'text-red-600'}`}>
							{workerStatus.redis_available ? 'Connected' : 'Disconnected'}
						</span>
					</div>
				</div>
			)}
		</div>
	);

	// Database Statistics Card
	const renderDBStatsCard = () => (
		<div className="bg-white shadow rounded-lg p-6">
			<div className="flex items-center mb-4">
				<CircleStackIcon className="h-6 w-6 mr-2 text-green-500" />
				<h3 className="text-lg font-medium text-gray-900">Database Statistics</h3>
			</div>

			{dbStatsLoading && <p className="text-gray-500">Loading...</p>}

			{dbStats && (
				<div className="space-y-4">
					{/* Total counts */}
					<div className="grid grid-cols-2 gap-4">
						<div className="text-center p-3 bg-blue-50 rounded">
							<p className="text-2xl font-bold text-blue-600">{dbStats.total_tasks}</p>
							<p className="text-sm text-gray-600">Total Tasks</p>
						</div>
						<div className="text-center p-3 bg-purple-50 rounded">
							<p className="text-2xl font-bold text-purple-600">{dbStats.total_experiments}</p>
							<p className="text-sm text-gray-600">Total Experiments</p>
						</div>
					</div>

					{/* 24h activity */}
					<div className="grid grid-cols-2 gap-4">
						<div className="text-center p-3 bg-green-50 rounded">
							<p className="text-xl font-bold text-green-600">{dbStats.tasks_last_24h}</p>
							<p className="text-xs text-gray-600">Tasks (24h)</p>
						</div>
						<div className="text-center p-3 bg-yellow-50 rounded">
							<p className="text-xl font-bold text-yellow-600">{dbStats.experiments_last_24h}</p>
							<p className="text-xs text-gray-600">Experiments (24h)</p>
						</div>
					</div>

					{/* Average duration */}
					{dbStats.avg_experiment_duration_seconds && (
						<div className="text-center p-3 bg-gray-50 rounded">
							<p className="text-xl font-bold text-gray-700">
								{Math.round(dbStats.avg_experiment_duration_seconds / 60)} min
							</p>
							<p className="text-xs text-gray-600">Avg Experiment Duration</p>
						</div>
					)}

					{/* Status breakdown */}
					<div>
						<p className="text-sm font-medium text-gray-700 mb-2">Task Status:</p>
						<div className="space-y-1">
							{Object.entries(dbStats.tasks_by_status).map(([status, count]) => (
								<div key={status} className="flex justify-between text-sm">
									<span className="text-gray-600 capitalize">{status}:</span>
									<span className="font-medium">{count}</span>
								</div>
							))}
						</div>
					</div>
				</div>
			)}
		</div>
	);

	// Running Tasks Card
	const renderRunningTasksCard = () => (
		<div className="bg-white shadow rounded-lg p-6">
			<div className="flex items-center mb-4">
				<ClockIcon className="h-6 w-6 mr-2 text-orange-500" />
				<h3 className="text-lg font-medium text-gray-900">Running Tasks</h3>
			</div>

			{dbStatsLoading && <p className="text-gray-500">Loading...</p>}

			{dbStats?.running_tasks && dbStats.running_tasks.length === 0 && (
				<p className="text-gray-500 text-sm">No tasks currently running</p>
			)}

			{dbStats?.running_tasks && dbStats.running_tasks.length > 0 && (
				<div className="space-y-3">
					{dbStats.running_tasks.map((task) => (
						<div key={task.id} className="border rounded p-3">
							<div className="flex justify-between items-start mb-2">
								<div>
									<p className="font-medium">Task {task.id}: {task.name}</p>
									{task.started_at && (
										<p className="text-xs text-gray-500">
											Started: {new Date(task.started_at).toLocaleString()}
										</p>
									)}
								</div>
							</div>
							<div className="mt-2">
								<div className="flex justify-between text-sm text-gray-600 mb-1">
									<span>Progress</span>
									<span>{task.completed_experiments} / {task.max_iterations}</span>
								</div>
								<div className="w-full bg-gray-200 rounded-full h-2">
									<div
										className="bg-blue-600 h-2 rounded-full"
										style={{
											width: `${(task.completed_experiments / task.max_iterations) * 100}%`,
										}}
									></div>
								</div>
							</div>
						</div>
					))}
				</div>
			)}
		</div>
	);

	// Experiment Timeline Chart
	const renderTimelineChart = () => {
		if (timelineLoading || !timeline) {
			return (
				<div className="bg-white shadow rounded-lg p-6">
					<h3 className="text-lg font-medium text-gray-900 mb-4">Experiment Timeline (24h)</h3>
					<p className="text-gray-500">Loading...</p>
				</div>
			);
		}

		// Prepare data for chart - group by hour
		const hourlyData = timeline.reduce((acc, exp) => {
			if (!exp.created_at) return acc;
			const date = new Date(exp.created_at);
			const hour = `${date.getHours()}:00`;

			if (!acc[hour]) {
				acc[hour] = { hour, success: 0, failed: 0, total: 0 };
			}

			acc[hour].total++;
			if (exp.status === 'success') {
				acc[hour].success++;
			} else if (exp.status === 'failed') {
				acc[hour].failed++;
			}

			return acc;
		}, {} as Record<string, { hour: string; success: number; failed: number; total: number }>);

		const chartData = Object.values(hourlyData).sort((a, b) =>
			parseInt(a.hour) - parseInt(b.hour)
		);

		return (
			<div className="bg-white shadow rounded-lg p-6">
				<h3 className="text-lg font-medium text-gray-900 mb-4">Experiment Timeline (24h)</h3>
				<ResponsiveContainer width="100%" height={300}>
					<BarChart data={chartData}>
						<CartesianGrid strokeDasharray="3 3" />
						<XAxis dataKey="hour" />
						<YAxis />
						<Tooltip />
						<Legend />
						<Bar dataKey="success" fill="#10b981" name="Success" stackId="a" />
						<Bar dataKey="failed" fill="#ef4444" name="Failed" stackId="a" />
					</BarChart>
				</ResponsiveContainer>

				{/* Statistics summary */}
				<div className="grid grid-cols-3 gap-4 mt-4">
					<div className="text-center p-3 bg-green-50 rounded">
						<p className="text-xl font-bold text-green-600">
							{timeline.filter(e => e.status === 'success').length}
						</p>
						<p className="text-xs text-gray-600">Successful</p>
					</div>
					<div className="text-center p-3 bg-red-50 rounded">
						<p className="text-xl font-bold text-red-600">
							{timeline.filter(e => e.status === 'failed').length}
						</p>
						<p className="text-xs text-gray-600">Failed</p>
					</div>
					<div className="text-center p-3 bg-gray-50 rounded">
						<p className="text-xl font-bold text-gray-700">{timeline.length}</p>
						<p className="text-xs text-gray-600">Total</p>
					</div>
				</div>
			</div>
		);
	};

	return (
		<div className="px-4 py-6 sm:px-6 lg:px-8">
			<div className="mb-6">
				<h1 className="text-2xl font-bold text-gray-900">Dashboard</h1>
				<p className="text-gray-600">System overview and real-time status</p>
			</div>

			{/* Grid layout */}
			<div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-6">
				{renderGPUCard()}
				{renderWorkerCard()}
				{renderDBStatsCard()}
			</div>

			<div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-6">
				<div className="lg:col-span-1">
					{renderRunningTasksCard()}
				</div>
				<div className="lg:col-span-2">
					{renderTimelineChart()}
				</div>
			</div>
		</div>
	);
}
