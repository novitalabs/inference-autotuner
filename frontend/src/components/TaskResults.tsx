import { useQuery } from '@tanstack/react-query';
import { apiClient } from '@/services/api';
import type { Task } from '@/types/api';
import toast from 'react-hot-toast';
import { useEscapeKey } from '@/hooks/useEscapeKey';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  Cell,
} from 'recharts';

interface TaskResultsProps {
  task: Task;
  onClose: () => void;
}

export default function TaskResults({ task, onClose }: TaskResultsProps) {
  // Handle Escape key to close modal
  useEscapeKey(onClose);
  // Fetch experiments for this task
  const {
    data: experiments = [],
    isLoading,
  } = useQuery({
    queryKey: ['experiments', task.id],
    queryFn: () => apiClient.getExperimentsByTask(task.id),
  });

  if (isLoading) {
    return (
      <div className="fixed inset-0 bg-gray-600 bg-opacity-50 flex items-center justify-center z-50">
        <div className="bg-white rounded-lg p-6">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto"></div>
          <p className="mt-4 text-gray-600">Loading results...</p>
        </div>
      </div>
    );
  }

  const successfulExperiments = experiments.filter((exp) => exp.status === 'success');
  const bestExperiment = experiments.find((exp) => exp.id === task.best_experiment_id);

  // Helper function to format metric value for display
  const formatMetricValue = (value: any): string => {
    if (value === null || value === undefined) return 'N/A';
    if (typeof value === 'number') return value.toFixed(2);
    if (typeof value === 'boolean') return value ? 'true' : 'false';
    if (typeof value === 'object') return JSON.stringify(value);
    return String(value);
  };

  // Filter metrics to only include primitive values for charts
  const getPrimitiveMetrics = (metrics: any): Record<string, number> => {
    if (!metrics) return {};
    const result: Record<string, number> = {};
    for (const [key, value] of Object.entries(metrics)) {
      if (typeof value === 'number') {
        result[key] = value;
      }
    }
    return result;
  };

  // Prepare data for charts (only primitive numeric values)
  const chartData = successfulExperiments.map((exp) => ({
    name: `Exp ${exp.experiment_id}`,
    experiment_id: exp.experiment_id,
    objective_score: exp.objective_score || 0,
    ...getPrimitiveMetrics(exp.metrics),
  }));

  // Get all numeric metric keys from successful experiments
  const metricKeys = successfulExperiments.length > 0 && successfulExperiments[0].metrics
    ? Object.keys(successfulExperiments[0].metrics).filter(key =>
        successfulExperiments[0].metrics && typeof successfulExperiments[0].metrics[key] === 'number'
      )
    : [];

  const formatDuration = (seconds: number | null) => {
    if (!seconds) return 'N/A';
    const hours = Math.floor(seconds / 3600);
    const mins = Math.floor((seconds % 3600) / 60);
    const secs = Math.floor(seconds % 60);
    if (hours > 0) return `${hours}h ${mins}m`;
    if (mins > 0) return `${mins}m ${secs}s`;
    return `${secs}s`;
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'success':
        return 'bg-green-100 text-green-800';
      case 'failed':
        return 'bg-red-100 text-red-800';
      case 'deploying':
        return 'bg-blue-100 text-blue-800';
      case 'benchmarking':
        return 'bg-yellow-100 text-yellow-800';
      default:
        return 'bg-gray-100 text-gray-800';
    }
  };

  const COLORS = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899'];

  return (
    <div className="fixed inset-0 bg-gray-600 bg-opacity-50 overflow-y-auto z-50">
      <div className="min-h-screen px-4 py-8">
        <div className="max-w-7xl mx-auto bg-white rounded-lg shadow-xl">
          {/* Header */}
          <div className="border-b border-gray-200 px-6 py-4 flex justify-between items-center">
            <div>
              <h2 className="text-2xl font-bold text-gray-900">{task.task_name} - Results</h2>
              <p className="text-sm text-gray-500 mt-1">
                {task.successful_experiments} / {task.total_experiments} successful experiments
              </p>
            </div>
            <button
              onClick={onClose}
              className="text-gray-400 hover:text-gray-600 transition-colors"
            >
              <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>

          <div className="p-6 space-y-6">
            {/* Best Configuration Card */}
            {bestExperiment && (
              <div className="bg-gradient-to-r from-green-50 to-emerald-50 border-2 border-green-200 rounded-lg p-6">
                <div className="flex items-center mb-4">
                  <svg className="w-6 h-6 text-green-600 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                  <h3 className="text-xl font-bold text-green-900">Best Configuration</h3>
                  <span className="ml-auto text-sm text-green-700">Experiment #{bestExperiment.experiment_id}</span>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  {/* Parameters Section - Enhanced */}
                  <div className="md:col-span-2">
                    <div className="flex items-center justify-between mb-2">
                      <h4 className="text-sm font-medium text-green-700">Optimal Parameters</h4>
                      <button
                        onClick={() => {
                          const paramsText = Object.entries(bestExperiment.parameters)
                            .map(([key, value]) => `${key}=${value}`)
                            .join(' ');
                          navigator.clipboard.writeText(paramsText);
                          toast.success('Parameters copied to clipboard!');
                        }}
                        className="text-xs px-2 py-1 bg-green-600 text-white rounded hover:bg-green-700 transition-colors"
                        title="Copy parameters to clipboard"
                      >
                        Copy
                      </button>
                    </div>
                    <div className="bg-white rounded-md p-3 border border-green-200">
                      <div className="space-y-2">
                        {Object.entries(bestExperiment.parameters).map(([key, value]) => (
                          <div key={key} className="flex items-center justify-between">
                            <span className="text-sm font-medium text-gray-700">{key}:</span>
                            <span className="text-sm font-mono font-bold text-green-800 bg-green-50 px-2 py-1 rounded">
                              {String(value)}
                            </span>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>

                  {/* Score Section */}
                  <div>
                    <h4 className="text-sm font-medium text-green-700 mb-2">Objective Score</h4>
                    <div className="bg-white rounded-md p-3 border border-green-200 text-center">
                      <div className="text-4xl font-bold text-green-900">
                        {bestExperiment.objective_score?.toFixed(4) || 'N/A'}
                      </div>
                      <p className="text-xs text-green-600 mt-2">
                        {task.optimization?.objective === 'minimize_latency' && 'Lower is better'}
                        {task.optimization?.objective === 'maximize_throughput' && 'Higher is better'}
                        {task.optimization?.objective === 'balanced' && 'Balanced score'}
                      </p>
                    </div>
                  </div>
                </div>

                {/* Metrics Section - Full Width Below */}
                {bestExperiment.metrics && Object.keys(bestExperiment.metrics).length > 0 && (
                  <div className="mt-4 pt-4 border-t border-green-200">
                    <h4 className="text-sm font-medium text-green-700 mb-2">Performance Metrics</h4>
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                      {Object.entries(bestExperiment.metrics)
                        .filter(([_, value]) => typeof value === 'number')
                        .map(([key, value]) => (
                          <div key={key} className="bg-white rounded-md p-2 border border-green-100">
                            <div className="text-xs text-gray-500 mb-1">{key}</div>
                            <div className="text-lg font-semibold text-gray-900">
                              {formatMetricValue(value)}
                            </div>
                          </div>
                        ))}
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* Charts */}
            {successfulExperiments.length > 0 && (
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* Objective Score Bar Chart */}
                <div className="bg-white border border-gray-200 rounded-lg p-4">
                  <h3 className="text-lg font-semibold text-gray-900 mb-4">
                    Objective Scores by Experiment
                    <span className="ml-3 text-sm font-normal text-gray-500">
                      <span className="inline-block w-3 h-3 bg-green-500 rounded mr-1"></span>
                      Best
                    </span>
                  </h3>
                  <ResponsiveContainer width="100%" height={300}>
                    <BarChart data={chartData}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="name" />
                      <YAxis />
                      <Tooltip
                        content={({ active, payload }) => {
                          if (active && payload && payload.length) {
                            const isBest = payload[0].payload.experiment_id === bestExperiment?.experiment_id;
                            return (
                              <div className="bg-white border border-gray-200 rounded shadow-lg p-2">
                                <p className="text-sm font-semibold text-gray-900">{payload[0].payload.name}</p>
                                <p className="text-sm text-gray-600">
                                  Score: <span className="font-mono">{(payload[0].value as number).toFixed(4)}</span>
                                </p>
                                {isBest && (
                                  <p className="text-xs text-green-600 font-semibold mt-1">⭐ Best Experiment</p>
                                )}
                              </div>
                            );
                          }
                          return null;
                        }}
                      />
                      <Bar dataKey="objective_score" name="Objective Score" label={(props: any) => {
                        const { x, y, width, index } = props;
                        if (x === undefined || y === undefined || width === undefined || index === undefined) return null;
                        const isBest = chartData[index]?.experiment_id === bestExperiment?.experiment_id;
                        if (isBest) {
                          return (
                            <text x={Number(x) + Number(width) / 2} y={Number(y) - 5} fill="#10b981" textAnchor="middle" fontSize={16}>
                              ⭐
                            </text>
                          );
                        }
                        return null;
                      }}>
                        {chartData.map((entry, index) => (
                          <Cell
                            key={`cell-${index}`}
                            fill={entry.experiment_id === bestExperiment?.experiment_id ? '#10b981' : '#3b82f6'}
                          />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>

                {/* Metrics Comparison */}
                {metricKeys.length > 0 && (
                  <div className="bg-white border border-gray-200 rounded-lg p-4">
                    <h3 className="text-lg font-semibold text-gray-900 mb-4">Performance Metrics</h3>
                    <ResponsiveContainer width="100%" height={300}>
                      <BarChart data={chartData}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="name" />
                        <YAxis />
                        <Tooltip />
                        <Legend />
                        {metricKeys.slice(0, 3).map((key, idx) => (
                          <Bar key={key} dataKey={key} fill={COLORS[idx % COLORS.length]} />
                        ))}
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                )}
              </div>
            )}

            {/* Experiments Table */}
            <div className="bg-white border border-gray-200 rounded-lg overflow-hidden">
              <div className="px-4 py-3 bg-gray-50 border-b border-gray-200">
                <h3 className="text-lg font-semibold text-gray-900">All Experiments</h3>
              </div>
              <div className="overflow-x-auto">
                <table className="min-w-full divide-y divide-gray-200">
                  <thead className="bg-gray-50">
                    <tr>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">ID</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Status</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Parameters</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Objective Score</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Duration</th>
                    </tr>
                  </thead>
                  <tbody className="bg-white divide-y divide-gray-200">
                    {experiments.map((exp) => (
                      <tr
                        key={exp.id}
                        className={exp.id === task.best_experiment_id ? 'bg-green-50' : ''}
                      >
                        <td className="px-4 py-3 whitespace-nowrap">
                          <div className="flex items-center">
                            <span className="text-sm font-medium text-gray-900">#{exp.experiment_id}</span>
                            {exp.id === task.best_experiment_id && (
                              <span className="ml-2 text-xs bg-green-100 text-green-800 px-2 py-0.5 rounded-full">
                                Best
                              </span>
                            )}
                          </div>
                        </td>
                        <td className="px-4 py-3 whitespace-nowrap">
                          <span className={`inline-flex px-2 py-1 text-xs font-semibold rounded-full ${getStatusColor(exp.status)}`}>
                            {exp.status}
                          </span>
                        </td>
                        <td className="px-4 py-3">
                          <div className="text-sm space-y-0.5">
                            {Object.entries(exp.parameters).map(([key, value]) => (
                              <div key={key} className="flex gap-2">
                                <span className="text-gray-500">{key}:</span>
                                <span className="font-mono text-gray-900">{String(value)}</span>
                              </div>
                            ))}
                          </div>
                        </td>
                        <td className="px-4 py-3 whitespace-nowrap">
                          <span className="text-sm font-mono text-gray-900">
                            {exp.objective_score !== null ? exp.objective_score.toFixed(4) : 'N/A'}
                          </span>
                        </td>
                        <td className="px-4 py-3 whitespace-nowrap text-sm text-gray-500">
                          {formatDuration(exp.elapsed_time)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            {/* Task Summary */}
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
              <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
                <div className="text-sm text-blue-600 font-medium">Total Experiments</div>
                <div className="text-2xl font-bold text-blue-900 mt-1">{task.total_experiments}</div>
              </div>
              <div className="bg-green-50 border border-green-200 rounded-lg p-4">
                <div className="text-sm text-green-600 font-medium">Successful</div>
                <div className="text-2xl font-bold text-green-900 mt-1">{task.successful_experiments}</div>
              </div>
              <div className="bg-red-50 border border-red-200 rounded-lg p-4">
                <div className="text-sm text-red-600 font-medium">Failed</div>
                <div className="text-2xl font-bold text-red-900 mt-1">
                  {task.total_experiments - task.successful_experiments}
                </div>
              </div>
              <div className="bg-purple-50 border border-purple-200 rounded-lg p-4">
                <div className="text-sm text-purple-600 font-medium">Total Duration</div>
                <div className="text-2xl font-bold text-purple-900 mt-1">{formatDuration(task.elapsed_time)}</div>
              </div>
            </div>

            {/* Close Button */}
            <div className="flex justify-end pt-4 border-t border-gray-200">
              <button
                onClick={onClose}
                className="px-6 py-2 bg-gray-600 text-white rounded-md hover:bg-gray-700 transition-colors"
              >
                Close
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
